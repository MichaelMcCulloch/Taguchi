//! Model-driven design selection (spec §"Design selection with a model").
//!
//! `select` enumerates candidate base arrays in ascending row count, searches
//! column assignments inside each candidate, replicates the rows, and keeps
//! the first design where every requested term is estimable. Estimability
//! comes from [`crate::model::estimability`] — the same function the analyzer
//! uses, so both sides read one answer. The test
//! `estimability_is_the_shared_function` holds that call in place.

use anyhow::Result;
use std::collections::BTreeSet;
use std::fmt;

use crate::factor::Factor;
use crate::model::{
    Cell, Estimability, Model, ReplicateRole, Term, Unit, estimability, hierarchy_warnings,
    model_matrix,
};
use crate::oa::lookup::{self, CatalogRef};
use crate::oa::{ArrayInfo, Selected, build_array, bush, gf::Gf};

/// Column assignments tried per catalog or Bush entry (spec: 5000 per entry).
pub const ASSIGNMENT_CAP: usize = 5000;

/// Construct-time limits from the CLI.
#[derive(Debug, Clone, Default)]
pub struct SelectOpts {
    /// Refuse designs above this many runs, counted after replication.
    pub max_runs: Option<usize>,
    pub min_residual_df: usize,
    /// Write the smallest candidate even when a requested term is not estimable.
    pub allow_aliased: bool,
}

/// No candidate design estimates the requested model within the limits.
/// The construct CLI arm maps this to exit code 3. The payload is the
/// construct-time report of the smallest candidate plus the reason.
#[derive(Debug)]
pub struct Refusal(pub String);

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Refusal {}

/// The chosen design: array, replicated cell list, and what it can estimate.
#[derive(Debug)]
pub struct Selection {
    pub selected: Selected,
    /// Array label including the chosen column assignment, e.g.
    /// `L8 (Sloane oa.8.7.2.2.txt, cols 1,2,4,7)`.
    pub label: String,
    /// Base rows × every combination of replicate levels, block-major.
    pub cells: Vec<Cell>,
    pub estimability: Estimability,
    /// The union of every requested formula's terms; what was checked.
    pub model: Model,
    /// The construct-time report, ready to print to stderr.
    pub report: String,
    /// True when `--allow-aliased` kept a design with a non-estimable term.
    pub aliased: bool,
}

/// The union of every formula's terms. With no formula this is the default
/// model of all main effects.
pub fn union_model(models: &[Model], factors: &[Factor]) -> Model {
    if models.is_empty() {
        return Model::main_effects(factors);
    }
    let mut terms: Vec<Term> = models
        .iter()
        .flat_map(|m| m.terms.iter().cloned())
        .collect();
    terms.sort_by(|a, b| a.0.len().cmp(&b.0.len()).then(a.0.cmp(&b.0)));
    terms.dedup();
    let formula = format!(
        ". ~ {}",
        terms
            .iter()
            .map(|t| t.label(factors))
            .collect::<Vec<_>>()
            .join(" + ")
    );
    Model {
        formula,
        response: None,
        terms,
    }
}

/// Degrees of freedom a term contributes under deviation coding.
fn term_df(term: &Term, factors: &[Factor]) -> usize {
    term.0
        .iter()
        .map(|&f| factors[f].level_count() - 1)
        .product()
}

/// Every combination of replicate levels, in level order, last role fastest.
/// With no roles this is one empty combination.
fn replicate_combinations(roles: &[ReplicateRole]) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new()];
    for role in roles {
        out = out
            .into_iter()
            .flat_map(|prefix| {
                (0..role.levels.len()).map(move |level| {
                    let mut c = prefix.clone();
                    c.push(level);
                    c
                })
            })
            .collect();
    }
    out
}

/// Base rows × replicate combinations, block-major: one contiguous block per
/// replicate combination, so execution order can shuffle inside a block.
/// OA cells are folded to resolved level indices here, once.
pub fn build_cells(selected: &Selected, factors: &[Factor], roles: &[ReplicateRole]) -> Vec<Cell> {
    let base: Vec<Vec<usize>> = selected
        .array
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(j, &cell)| cell as usize % factors[j].level_count())
                .collect()
        })
        .collect();
    let mut cells = Vec::with_capacity(base.len() * replicate_combinations(roles).len());
    for combination in replicate_combinations(roles) {
        for row in &base {
            cells.push(Cell {
                factor_levels: row.clone(),
                replicate_levels: combination.clone(),
            });
        }
    }
    cells
}

/// A candidate base array. Catalog and Bush candidates carry a column pool
/// that the assignment search draws from.
enum Candidate {
    /// `build_array`'s result: one assignment.
    Fixed(Selected),
    /// Materialized only when this candidate survives the limits.
    FullFactorial {
        rows: usize,
    },
    Catalog(CatalogRef),
    Bush {
        q: usize,
        k: usize,
        cols: usize,
    },
}

impl Candidate {
    fn rows(&self) -> usize {
        match self {
            Candidate::Fixed(s) => s.array.len(),
            Candidate::FullFactorial { rows } => *rows,
            Candidate::Catalog(e) => e.n_runs,
            Candidate::Bush { q, k, .. } => q.pow(*k as u32),
        }
    }
}

/// A candidate with its column pool materialised, ready for the assignment
/// search.
enum Pool {
    Fixed(Selected),
    Catalog(CatalogRef),
    Bush {
        array: Vec<Vec<u8>>,
        q: usize,
        k: usize,
    },
}

impl Pool {
    /// Level count of every column the assignment search may use.
    fn column_levels(&self) -> Vec<usize> {
        match self {
            Pool::Fixed(s) => s.column_levels.clone(),
            Pool::Catalog(e) => e.columns.clone(),
            Pool::Bush { array, q, .. } => vec![*q; array.first().map_or(0, Vec::len)],
        }
    }

    fn take(&self, assignment: &[usize]) -> Selected {
        match self {
            Pool::Fixed(s) => s.clone(),
            Pool::Catalog(e) => lookup::select_columns(e, assignment),
            Pool::Bush { array, q, k } => {
                let cells: Vec<Vec<u8>> = array
                    .iter()
                    .map(|row| assignment.iter().map(|&c| row[c]).collect())
                    .collect();
                let cols: Vec<String> = assignment.iter().map(|&c| (c + 1).to_string()).collect();
                let method = format!("Bush GF({}) k={}", q, k);
                let label = format!("L{} ({}, cols {})", cells.len(), method, cols.join(","));
                Selected {
                    info: ArrayInfo {
                        rows: cells.len(),
                        cols: assignment.len(),
                        label,
                        method,
                    },
                    array: cells,
                    column_levels: vec![*q; assignment.len()],
                }
            }
        }
    }
}

/// Ordered injections of factors into columns of a matching level count, in
/// ascending column index, at most `cap` of them. A fixed candidate already
/// carries exactly the factor columns, so it yields one identity assignment.
fn assignments(pool: &Pool, factors: &[Factor], cap: usize) -> Vec<Vec<usize>> {
    if let Pool::Fixed(_) = pool {
        return vec![(0..factors.len()).collect()];
    }
    let columns = pool.column_levels();
    // A Bush column carries q levels and folds down to any factor with ≤ q
    // levels; a catalog column must match the factor's level count exactly.
    let folds = matches!(pool, Pool::Bush { .. });
    let fits = |column: usize, factor: usize| {
        let (have, need) = (columns[column], factors[factor].level_count());
        if folds { have >= need } else { have == need }
    };
    let mut out = Vec::new();
    let mut current = Vec::with_capacity(factors.len());
    let mut used = vec![false; columns.len()];
    // Iterative DFS: `next[depth]` is the column index to try at that depth.
    let mut next = vec![0usize; factors.len() + 1];
    let mut depth = 0usize;
    loop {
        if depth == factors.len() {
            out.push(current.clone());
            if out.len() >= cap {
                return out;
            }
            depth -= 1;
            let column = current.pop().expect("depth > 0 implies a chosen column");
            used[column] = false;
            next[depth] = column + 1;
            continue;
        }
        let chosen =
            (next[depth]..columns.len()).find(|&column| !used[column] && fits(column, depth));
        match chosen {
            Some(column) => {
                used[column] = true;
                current.push(column);
                depth += 1;
                next[depth] = 0;
            }
            None => {
                if depth == 0 {
                    return out;
                }
                depth -= 1;
                let column = current.pop().expect("depth > 0 implies a chosen column");
                used[column] = false;
                next[depth] = column + 1;
            }
        }
    }
}

/// The full factorial, rows in lexicographic level order (last factor fastest).
fn full_factorial(factors: &[Factor]) -> Selected {
    let levels: Vec<usize> = factors.iter().map(Factor::level_count).collect();
    let rows: usize = levels.iter().product();
    let mut array = Vec::with_capacity(rows);
    for index in 0..rows {
        let mut row = vec![0u8; levels.len()];
        let mut rest = index;
        for (j, &l) in levels.iter().enumerate().rev() {
            row[j] = (rest % l) as u8;
            rest /= l;
        }
        array.push(row);
    }
    let method = "Full factorial".to_string();
    Selected {
        info: ArrayInfo {
            rows,
            cols: levels.len(),
            label: format!("L{} (full factorial)", rows),
            method: method.clone(),
        },
        array,
        column_levels: levels,
    }
}

/// Bush candidates at k_min+1 and k_min+2, where k_min is the depth
/// `oa::bush::try_construct` would pick. Homogeneous level counts only.
fn bush_candidates(factors: &[Factor]) -> Vec<Candidate> {
    let levels: Vec<usize> = factors.iter().map(Factor::level_count).collect();
    if levels.iter().any(|&l| l != levels[0]) {
        return Vec::new();
    }
    let q = bush::next_prime_power_at_least(levels[0]);
    let capacity = |k: usize| -> Option<usize> {
        let rows = (q as u64).checked_pow(k as u32)?;
        usize::try_from((rows - 1) / (q as u64 - 1)).ok()
    };
    let mut k_min = 2usize;
    while k_min <= 10 && capacity(k_min).is_none_or(|cap| cap < factors.len()) {
        k_min += 1;
    }
    if k_min > 10 {
        return Vec::new();
    }
    [k_min + 1, k_min + 2]
        .into_iter()
        .filter_map(|k| {
            let cols = capacity(k)?;
            (q as u64).checked_pow(k as u32)?;
            Some(Candidate::Bush { q, k, cols })
        })
        .collect()
}

/// Select a base array that estimates every requested term.
///
/// `unit` takes no part in the search: the selection matrix carries blocks and
/// the union model only (spec §"Design selection with a model", step 2). It is
/// accepted here so callers pass the whole design description in one place.
pub fn select(
    factors: &[Factor],
    models: &[Model],
    roles: &[ReplicateRole],
    unit: Option<&Unit>,
    opts: &SelectOpts,
) -> Result<Selection> {
    let _ = unit;
    let model = union_model(models, factors);
    let replicates: usize = roles.iter().map(|r| r.levels.len()).product();
    let base = build_array(factors)?;

    // With no formula the design is whatever v1 built, so long as the user's
    // own limits allow it: no term was requested, so none has to be estimable.
    // When `--max-runs` or `--min-residual-df` rules that array out, the search
    // below runs on the default main-effects model instead of writing a design
    // the user said they did not want.
    if models.is_empty() {
        let cells = build_cells(&base, factors, roles);
        let mm = model_matrix(factors, roles, &model, &cells, None);
        let est = estimability(&mm, &model, factors);
        let runs = cells.len();
        let within_limits =
            opts.max_runs.is_none_or(|max| runs <= max) && est.residual_df >= opts.min_residual_df;
        if within_limits {
            let report = render(factors, models, &model, &base.info.label, roles, &est, runs);
            return Ok(Selection {
                label: base.info.label.clone(),
                selected: base,
                cells,
                estimability: est,
                model,
                report,
                aliased: false,
            });
        }
    }

    // Candidates: (a) today's array, (b) catalog entries, (c) deeper Bush
    // arrays, (d) the full factorial. Nothing larger than the full factorial
    // can ever be preferable to it, so (b) and (c) stop there.
    let levels: Vec<usize> = factors.iter().map(Factor::level_count).collect();
    // Keep the factorial symbolic until it survives the run budget and is
    // actually visited. The factor space may exceed the machine word size.
    let full_rows = levels.iter().try_fold(1usize, |n, &l| n.checked_mul(l));
    let row_ceiling = full_rows.unwrap_or(usize::MAX);
    let mut candidates = vec![Candidate::Fixed(base)];
    candidates.extend(
        lookup::entries_supplying(&levels)
            .into_iter()
            .filter(|e| e.n_runs <= row_ceiling)
            .map(Candidate::Catalog),
    );
    candidates.extend(
        bush_candidates(factors)
            .into_iter()
            .filter(|c| c.rows() <= row_ceiling),
    );
    if let Some(rows) = full_rows {
        candidates.push(Candidate::FullFactorial { rows });
    }
    // Stable: equal row counts keep the a, b, c, d enumeration order.
    candidates.sort_by_key(Candidate::rows);

    // rank == 1 + Σ df(T) is necessary for every term to be estimable, and
    // residual df = rows − rank, so a candidate with fewer rows than
    // `floor + min_residual_df` cannot win. Skipping it costs no correctness.
    let floor = 1 + model
        .terms
        .iter()
        .map(|t| term_df(t, factors))
        .sum::<usize>();

    let mut smallest: Option<(Selected, Vec<Cell>, Estimability)> = None;
    for candidate in &candidates {
        let Some(runs) = candidate.rows().checked_mul(replicates) else {
            continue;
        };
        let over_max = opts.max_runs.is_some_and(|max| runs > max);
        if smallest.is_some() && (over_max || runs < floor + opts.min_residual_df) {
            continue;
        }
        let pool = match candidate {
            Candidate::Fixed(s) => Pool::Fixed(s.clone()),
            Candidate::FullFactorial { .. } => Pool::Fixed(full_factorial(factors)),
            Candidate::Catalog(e) => Pool::Catalog(e.clone()),
            Candidate::Bush { q, k, cols } => {
                let mut array = bush::bush_at(&Gf::new(*q)?, *k);
                for row in &mut array {
                    row.truncate(*cols);
                }
                Pool::Bush {
                    array,
                    q: *q,
                    k: *k,
                }
            }
        };
        for assignment in assignments(&pool, factors, ASSIGNMENT_CAP) {
            let selected = pool.take(&assignment);
            let cells = build_cells(&selected, factors, roles);
            let mm = model_matrix(factors, roles, &model, &cells, None);
            let est = estimability(&mm, &model, factors);
            let winner = !over_max
                && est.terms.iter().all(|t| t.estimable)
                && est.residual_df >= opts.min_residual_df;
            if winner {
                let report = render(
                    factors,
                    models,
                    &model,
                    &selected.info.label,
                    roles,
                    &est,
                    cells.len(),
                );
                return Ok(Selection {
                    label: selected.info.label.clone(),
                    selected,
                    cells,
                    estimability: est,
                    model,
                    report,
                    aliased: false,
                });
            }
            if smallest.is_none() {
                smallest = Some((selected, cells, est));
            }
            // The smallest candidate only needs its first assignment on
            // record; the limits exclude this one, so the rest add nothing.
            if over_max {
                break;
            }
        }
    }

    let (selected, cells, est) = smallest.expect("the candidate list is never empty");
    let runs = cells.len();
    let mut report = render(
        factors,
        models,
        &model,
        &selected.info.label,
        roles,
        &est,
        runs,
    );
    let unmet: Vec<String> = est
        .terms
        .iter()
        .filter(|t| !t.estimable)
        .map(|t| t.term.label(factors))
        .collect();
    // `--max-runs` is the user's hard budget: no flag waives it.
    let over_max = opts.max_runs.filter(|&max| runs > max);
    let reason = if let Some(max) = over_max {
        format!("the smallest candidate is {runs} runs, above --max-runs {max}")
    } else if unmet.is_empty() {
        format!(
            "no candidate design reaches residual df {} within the limits",
            opts.min_residual_df
        )
    } else {
        format!(
            "no candidate design within the limits estimates: {}",
            unmet.join(", ")
        )
    };
    if opts.allow_aliased && over_max.is_none() {
        report.push_str(&format!("warning: {reason}; --allow-aliased kept it\n"));
        return Ok(Selection {
            label: selected.info.label.clone(),
            selected,
            cells,
            estimability: est,
            model,
            report,
            aliased: true,
        });
    }
    let advice = if over_max.is_some() {
        "Raise --max-runs, or ask for a model the budget can carry."
    } else {
        "Raise --max-runs, add --replicate, relax --min-residual-df, or pass \
         --allow-aliased to write it anyway."
    };
    Err(Refusal(format!("{report}refused: {reason}. {advice}")).into())
}

/// The construct-time report of spec §"Design selection with a model", step 4.
fn render(
    factors: &[Factor],
    models: &[Model],
    model: &Model,
    label: &str,
    roles: &[ReplicateRole],
    est: &Estimability,
    runs: usize,
) -> String {
    let mut out = String::new();
    if models.is_empty() {
        out.push_str(&format!(
            "Model:    {} (default: main effects)\n",
            model.formula
        ));
    } else {
        for m in models {
            out.push_str(&format!("Model:    {}\n", m.formula));
        }
    }
    let mut design = label.to_string();
    for role in roles {
        design.push_str(&format!(
            " × {} replicates({})",
            role.levels.len(),
            role.name
        ));
    }
    out.push_str(&format!("Design:   {design} = {runs} runs\n"));
    out.push_str(&format!(
        "Params:   {}   rank: {}   residual df: {}\n",
        est.n_params, est.rank, est.residual_df
    ));
    out.push_str("  term            df  estimable  aliased with\n");
    for term in &est.terms {
        let row = format!(
            "  {:<16}{:>2}  {:<9}  {}",
            term.term.label(factors),
            term.df,
            if term.estimable { "yes" } else { "no" },
            term.aliased_with.join(", ")
        );
        out.push_str(row.trim_end());
        out.push('\n');
    }
    if !est.blocks_estimable && !roles.is_empty() {
        out.push_str("warning: replicate block effects are not estimable\n");
    }
    let mut warned = BTreeSet::new();
    let listed: Vec<&Model> = if models.is_empty() {
        vec![model]
    } else {
        models.iter().collect()
    };
    for m in listed {
        for warning in hierarchy_warnings(m, factors) {
            if warned.insert(warning.clone()) {
                out.push_str(&format!("warning: {warning}\n"));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factor::Factor;

    const NAMES: [&str; 4] = ["a", "b", "c", "d"];

    fn factors(specs: &[&str]) -> Vec<Factor> {
        specs
            .iter()
            .enumerate()
            .map(|(i, spec)| Factor::parse(NAMES[i], spec).unwrap())
            .collect()
    }

    fn seed_role(levels: i64) -> ReplicateRole {
        ReplicateRole {
            name: "seed".into(),
            levels: (1..=levels).map(crate::factor::FactorValue::Int).collect(),
        }
    }

    fn run(
        specs: &[&str],
        formula: &str,
        roles: &[ReplicateRole],
        opts: &SelectOpts,
    ) -> Result<Selection> {
        let fs = factors(specs);
        let model = Model::parse(formula, &fs)?;
        select(&fs, &[model], roles, None, opts)
    }

    #[test]
    fn large_factor_space_does_not_materialize_the_factorial() {
        // 3^50 overflows usize; its main effects still fit in a small OA.
        let fs: Vec<Factor> = (0..50)
            .map(|i| Factor::parse(&format!("f{i}"), "d[1-3]").unwrap())
            .collect();
        let formula = fs
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>()
            .join(" + ");
        let model = Model::parse(&formula, &fs).unwrap();
        let s = select(
            &fs,
            &[model],
            &[],
            None,
            &SelectOpts {
                max_runs: Some(243),
                ..SelectOpts::default()
            },
        )
        .unwrap();
        assert!(s.cells.len() <= 243);
        assert_eq!(s.estimability.rank, 101);
    }

    /// Outcome 3: the run counts the spec pins.
    #[test]
    fn pinned_run_counts() {
        // (a) three 2-level factors, every interaction: 8 parameters, so the
        // 4-row candidates cannot carry the model; L8 can.
        let two = ["d[1,2]", "d[1,2]", "d[1,2]", "d[1,2]"];
        let s = run(&two[..3], "a*b*c", &[], &SelectOpts::default()).unwrap();
        assert_eq!(s.cells.len(), 8);
        assert!(s.estimability.terms.iter().all(|t| t.estimable));

        // (b) four 2-level factors with one interaction: an L8 assignment
        // keeps a:b clear of c and d (a:b aliases c:d, which is not in the
        // model), so 6 parameters fit in 8 rows.
        let s = run(&two, "a + b + c + d + a:b", &[], &SelectOpts::default()).unwrap();
        assert_eq!(s.cells.len(), 8);
        assert!(s.estimability.terms.iter().all(|t| t.estimable));

        // (c) three 3-level factors with a 3×3 interaction: 11 parameters
        // (intercept + 3×2 main + 4 for a:b). L9 is too small, so the search
        // lands on the smallest catalog entry above it that keeps a:b clear —
        // an 18-row array, well under the 27-run full factorial.
        let three = ["d[1-3]", "d[1-3]", "d[1-3]"];
        let s = run(&three, "a*b + c", &[], &SelectOpts::default()).unwrap();
        assert_eq!(s.cells.len(), 18);
        assert_eq!(s.estimability.n_params, 11);
        assert!(s.estimability.terms.iter().all(|t| t.estimable));

        // (d) two 2-level factors with a:b need 4 parameters; the 4-run full
        // factorial is the largest candidate, so residual df 1 is out of
        // reach without replicates.
        let strict = SelectOpts {
            min_residual_df: 1,
            ..SelectOpts::default()
        };
        let err = run(&two[..2], "a*b", &[], &strict).unwrap_err();
        assert!(
            err.downcast_ref::<Refusal>().is_some(),
            "refusal maps to exit 3: {err:#}"
        );
        assert!(format!("{err:#}").contains("residual df 1"), "{err:#}");
        let s = run(&two[..2], "a*b", &[seed_role(2)], &strict).unwrap();
        assert_eq!(s.cells.len(), 8);
        assert!(s.estimability.residual_df >= 1);
    }

    /// Outcome 1: the reported estimability is `model::estimability` on the
    /// replicated cells — construct and analyze cannot drift apart.
    #[test]
    fn estimability_is_the_shared_function() {
        let fs = factors(&["d[1,2]", "d[1-3]"]);
        let model = Model::parse("a*b", &fs).unwrap();
        let roles = [seed_role(2)];
        let s = select(
            &fs,
            std::slice::from_ref(&model),
            &roles,
            None,
            &SelectOpts::default(),
        )
        .unwrap();
        let union = union_model(&[model], &fs);
        let mm = model_matrix(&fs, &roles, &union, &s.cells, None);
        let direct = estimability(&mm, &union, &fs);
        assert_eq!(s.estimability.n_params, direct.n_params);
        assert_eq!(s.estimability.rank, direct.rank);
        assert_eq!(s.estimability.residual_df, direct.residual_df);
        assert_eq!(s.cells.len(), s.selected.array.len() * 2);
        // Cells are block-major: one contiguous block per replicate level.
        assert!(
            s.cells[..s.selected.array.len()]
                .iter()
                .all(|c| c.replicate_levels == [0])
        );
        assert!(
            s.cells[s.selected.array.len()..]
                .iter()
                .all(|c| c.replicate_levels == [1])
        );
    }

    /// `--max-runs` refuses; `--allow-aliased` writes the smallest candidate
    /// and says so.
    #[test]
    fn limits_and_allow_aliased() {
        let two = ["d[1,2]", "d[1,2]", "d[1,2]"];
        let capped = SelectOpts {
            max_runs: Some(4),
            ..SelectOpts::default()
        };
        let err = run(&two, "a*b*c", &[], &capped).unwrap_err();
        let refusal = err.downcast_ref::<Refusal>().expect("refusal");
        assert!(
            refusal.0.contains("term"),
            "the report prints: {}",
            refusal.0
        );
        assert!(refusal.0.contains("refused:"), "{}", refusal.0);

        let s = run(
            &two,
            "a*b*c",
            &[],
            &SelectOpts {
                max_runs: Some(4),
                allow_aliased: true,
                ..SelectOpts::default()
            },
        )
        .unwrap();
        assert!(s.aliased);
        assert_eq!(s.cells.len(), 4);
        assert!(s.estimability.terms.iter().any(|t| !t.estimable));
        assert!(s.report.contains("--allow-aliased kept it"), "{}", s.report);
    }

    /// The user's limits bind with no formula too: `--min-residual-df` and
    /// `--max-runs` apply to the default main-effects model, and the search
    /// takes over when the v1 array misses them.
    #[test]
    fn limits_bind_without_a_formula() {
        let fs = factors(&["d[1,2]", "d[1,2]"]);
        // The v1 default is 4 rows with 3 parameters, so residual df is 1.
        let plain = select(&fs, &[], &[], None, &SelectOpts::default()).unwrap();
        assert_eq!(plain.cells.len(), 4);
        assert_eq!(plain.estimability.residual_df, 1);

        // Two residual df is out of reach up to the 4-run full factorial.
        let strict = SelectOpts {
            min_residual_df: 2,
            ..SelectOpts::default()
        };
        let err = select(&fs, &[], &[], None, &strict).unwrap_err();
        let refusal = err.downcast_ref::<Refusal>().expect("refusal");
        assert!(refusal.0.contains("residual df 2"), "{}", refusal.0);

        // Replicates supply the rows, so the same request now succeeds.
        let s = select(&fs, &[], &[seed_role(2)], None, &strict).unwrap();
        assert_eq!(s.cells.len(), 8);
        assert!(s.estimability.residual_df >= 2);

        // --max-runs still refuses when no candidate fits under it.
        let err = select(
            &fs,
            &[],
            &[],
            None,
            &SelectOpts {
                max_runs: Some(2),
                ..SelectOpts::default()
            },
        )
        .unwrap_err();
        assert!(
            err.downcast_ref::<Refusal>()
                .is_some_and(|r| r.0.contains("--max-runs 2")),
            "{err:#}"
        );
    }

    /// `--allow-aliased` waives estimability only. The run budget is absolute.
    #[test]
    fn allow_aliased_does_not_waive_max_runs() {
        let opts = SelectOpts {
            max_runs: Some(2),
            allow_aliased: true,
            ..SelectOpts::default()
        };
        let err = run(&["d[1,2]", "d[1,2]"], "a*b", &[], &opts).unwrap_err();
        let refusal = err.downcast_ref::<Refusal>().expect("refusal");
        assert!(refusal.0.contains("--max-runs 2"), "{}", refusal.0);
        assert!(refusal.0.contains("refused:"), "{}", refusal.0);
        // The same request without the cap is written, aliased.
        let s = run(
            &["d[1,2]", "d[1,2]"],
            "a*b",
            &[],
            &SelectOpts {
                min_residual_df: 1,
                allow_aliased: true,
                ..SelectOpts::default()
            },
        )
        .unwrap();
        assert!(s.aliased);
        assert_eq!(s.cells.len(), 4);
    }

    /// With no formula the search does not run: the design is whatever
    /// `build_array` gives, and the report still prints the default model.
    #[test]
    fn no_model_keeps_the_v1_array() {
        let fs = factors(&["d[1,2]", "d[1-3]", "d[1-4]"]);
        let s = select(&fs, &[], &[], None, &SelectOpts::default()).unwrap();
        let expected = build_array(&fs).unwrap();
        assert_eq!(s.selected.array, expected.array);
        assert_eq!(s.model.terms.len(), 3);
        assert!(s.report.contains("default: main effects"), "{}", s.report);
        assert!(!s.aliased);
    }

    /// Hierarchy warnings print in the construct-time report.
    #[test]
    fn report_carries_hierarchy_warnings() {
        let s = run(
            &["d[1,2]", "d[1,2]"],
            "a + a:b",
            &[],
            &SelectOpts::default(),
        )
        .unwrap();
        assert!(
            s.report.contains("warning: a:b requested without b"),
            "{}",
            s.report
        );
    }

    /// The assignment search enumerates ordered injections into columns of a
    /// matching level count, in ascending column index, capped.
    #[test]
    fn assignment_search_is_ordered_and_capped() {
        let entry = lookup::entries_supplying(&[2, 2])
            .into_iter()
            .find(|e| e.columns.len() >= 4)
            .expect("a catalog entry with four 2-level columns");
        let pool = Pool::Catalog(entry);
        let fs = factors(&["d[1,2]", "d[1,2]"]);
        let all = assignments(&pool, &fs, ASSIGNMENT_CAP);
        assert_eq!(all[0], vec![0, 1]);
        assert_eq!(all[1], vec![0, 2]);
        assert!(all.iter().all(|a| a[0] != a[1]), "columns are distinct");
        assert_eq!(assignments(&pool, &fs, 3).len(), 3, "the cap holds");

        // A mixed entry only offers columns whose level count matches.
        let mixed = factors(&["d[1,2]", "d[1-3]"]);
        let entry = lookup::entries_supplying(&[2, 3])
            .into_iter()
            .next()
            .expect("a mixed catalog entry");
        let levels = entry.columns.clone();
        for assignment in assignments(&Pool::Catalog(entry), &mixed, 50) {
            assert_eq!(levels[assignment[0]], 2);
            assert_eq!(levels[assignment[1]], 3);
        }
    }

    /// The full factorial is the last resort and is in lexicographic order.
    #[test]
    fn full_factorial_is_lexicographic() {
        let fs = factors(&["d[1,2]", "d[1-3]"]);
        let full = full_factorial(&fs);
        assert_eq!(full.array.len(), 6);
        assert_eq!(
            full.array,
            vec![
                vec![0, 0],
                vec![0, 1],
                vec![0, 2],
                vec![1, 0],
                vec![1, 1],
                vec![1, 2]
            ]
        );
        assert_eq!(full.info.method, "Full factorial");
    }
}
