//! Augment an existing design with extra runs that resolve an alias or sharpen
//! a chosen interaction, without discarding completed results.
//!
//! The greedy loop follows SPEC §"Augmentation (W4)": each step scores every
//! candidate row first by the rank gain it gives the target terms' columns and
//! then by its D-gain. Every estimability decision goes through
//! `model::estimability`, the same function construct and analyze call.
use anyhow::{Context, Result, anyhow, bail};
use nalgebra::{Cholesky, DMatrix};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::construct::{Augmentation, Design, Run, sidecar_path};
use crate::factor::Factor;
use crate::model::{
    Cell, ColumnSource, Estimability, Model, ModelMatrix, ReplicateRole, Unit, estimability,
    model_matrix,
};
use crate::rng::Rng;

/// SPEC: the candidate set is capped; above the cap we sample uniformly.
pub const CANDIDATE_CAP: usize = 200_000;
/// SPEC: D-gain uses log det(X'X + xx' + εI) − log det(X'X + εI) with ε = 1e-8.
const RIDGE: f64 = 1e-8;

#[derive(Debug)]
pub struct AugmentArgs {
    pub csv_path: PathBuf,
    pub design_path: Option<PathBuf>,
    pub targets: Vec<String>,
    pub runs: Option<usize>,
    pub seed: Option<u64>,
    pub include_pending: bool,
    pub output: PathBuf,
}

/// What the augmentation did. `run_augment` turns a false `targets_estimable`
/// into exit code 3; tests read the rest.
#[derive(Debug)]
pub struct Outcome {
    pub added: usize,
    pub targets: Vec<String>,
    pub targets_estimable: bool,
    pub before: Estimability,
    pub after: Estimability,
    pub log_det_before: f64,
    pub log_det_after: f64,
    pub run_ids: Vec<String>,
    pub renumbered: bool,
}

pub fn run_augment(args: AugmentArgs) -> Result<()> {
    let outcome = augment(args)?;
    if !outcome.targets_estimable {
        eprintln!(
            "✗ still not estimable after augmentation: {}",
            outcome.targets.join(", ")
        );
        std::process::exit(3);
    }
    Ok(())
}

// ---------------------------------------------------------------- input ----

/// Column positions in the input CSV. v1 files carry `experiment` instead of
/// `run_id`/`order`; both shapes are accepted.
struct Table {
    headers: csv::StringRecord,
    rows: Vec<csv::StringRecord>,
    factor_col: Vec<usize>,
    role_col: Vec<usize>,
    result_col: Vec<usize>,
    run_id_col: Option<usize>,
    order_col: Option<usize>,
    unit_col: Option<usize>,
    experiment_col: Option<usize>,
}

fn read_table(path: &Path, design: &Design) -> Result<Table> {
    let mut rdr =
        csv::Reader::from_path(path).with_context(|| format!("opening {}", path.display()))?;
    let headers = rdr.headers()?.clone();
    let rows: Vec<csv::StringRecord> = rdr.records().collect::<std::result::Result<_, _>>()?;

    let find = |name: &str| headers.iter().position(|h| h == name);
    let factor_col: Vec<usize> = design
        .factors
        .iter()
        .map(|f| find(&f.name).ok_or_else(|| anyhow!("factor column '{}' not in CSV", f.name)))
        .collect::<Result<_>>()?;
    let role_col: Vec<usize> = design
        .replicates
        .iter()
        .map(|r| find(&r.name).ok_or_else(|| anyhow!("replicate column '{}' not in CSV", r.name)))
        .collect::<Result<_>>()?;
    let result_col: Vec<usize> = design
        .results
        .iter()
        .map(|r| find(r).ok_or_else(|| anyhow!("result column '{}' not in CSV", r)))
        .collect::<Result<_>>()?;
    let unit_col = design.unit.as_ref().and_then(|u| find(&u.name));

    // A factor or result may legitimately be named "run_id"; do not steal it.
    let taken: Vec<usize> = factor_col
        .iter()
        .chain(&role_col)
        .chain(&result_col)
        .chain(unit_col.iter())
        .copied()
        .collect();
    let free = |name: &str| find(name).filter(|c| !taken.contains(c));
    let run_id_col = free("run_id");
    let order_col = free("order");
    let experiment_col = run_id_col.is_none().then(|| free("experiment")).flatten();

    Ok(Table {
        headers,
        rows,
        factor_col,
        role_col,
        result_col,
        run_id_col,
        order_col,
        unit_col,
        experiment_col,
    })
}

/// Resolved level index of a CSV cell. Every statistic downstream uses these
/// indices, never the raw text.
fn level_index(values: &[crate::factor::FactorValue], cell: &str) -> Option<usize> {
    use crate::factor::FactorValue;
    values.iter().position(|v| {
        if v.to_string() == cell {
            return true;
        }
        match v {
            FactorValue::Text(s) => s == cell,
            FactorValue::Int(n) => cell
                .parse::<f64>()
                .map(|c| (c - *n as f64).abs() < 1e-9)
                .unwrap_or(false),
            FactorValue::Float(x) => cell
                .parse::<f64>()
                .map(|c| (c - x).abs() < 1e-4 * x.abs().max(1.0))
                .unwrap_or(false),
        }
    })
}

fn row_cell(table: &Table, design: &Design, index: usize) -> Result<Cell> {
    let row = &table.rows[index];
    let read = |col: usize, name: &str, values: &[crate::factor::FactorValue]| {
        let text = row.get(col).unwrap_or("").trim();
        level_index(values, text).ok_or_else(|| {
            anyhow!(
                "row {}: '{}' value '{}' is not one of the design levels",
                index + 2,
                name,
                text
            )
        })
    };
    Ok(Cell {
        factor_levels: design
            .factors
            .iter()
            .zip(&table.factor_col)
            .map(|(f, &c)| read(c, &f.name, &f.values))
            .collect::<Result<_>>()?,
        replicate_levels: design
            .replicates
            .iter()
            .zip(&table.role_col)
            .map(|(r, &c)| read(c, &r.name, &r.levels))
            .collect::<Result<_>>()?,
    })
}

/// SPEC: a row is completed when every result cell is non-empty.
fn is_complete(table: &Table, index: usize) -> bool {
    let row = &table.rows[index];
    table
        .result_col
        .iter()
        .all(|&c| !row.get(c).unwrap_or("").trim().is_empty())
}

// ---------------------------------------------------------------- model ----

/// The union of every formula's terms; main effects when no formula exists.
/// Canonicalisation is `Model::parse`'s, so the order matches construct's.
fn union_model(design: &Design, extra: &[String]) -> Result<Model> {
    let mut labels: Vec<String> = Vec::new();
    for model in &design.models {
        for term in &model.terms {
            let label = term.label(&design.factors);
            if !labels.contains(&label) {
                labels.push(label);
            }
        }
    }
    if labels.is_empty() {
        for term in Model::main_effects(&design.factors).terms {
            labels.push(term.label(&design.factors));
        }
    }
    for label in extra {
        if !labels.contains(label) {
            labels.push(label.clone());
        }
    }
    if labels.is_empty() {
        bail!("the design has no factors, so there is nothing to augment");
    }
    Model::parse(&labels.join(" + "), &design.factors)
        .context("building the union of the design's model formulas")
}

/// Canonical label of a single `--for` term, so that `b:a` and `a:b` agree.
fn target_label(src: &str, factors: &[Factor]) -> Result<String> {
    let parsed =
        Model::parse(src, factors).with_context(|| format!("--for {}: invalid term", src))?;
    match parsed.terms.as_slice() {
        [term] => Ok(term.label(factors)),
        _ => bail!("--for {}: expected one term, such as 'a' or 'a:b'", src),
    }
}

// ------------------------------------------------------------ candidates ----

struct Candidates {
    cells: Vec<Cell>,
    /// Set when the full factorial exceeded the cap: (kept, total).
    sampled: Option<(usize, u128)>,
}

fn candidate_cells(factors: &[Factor], roles: &[ReplicateRole], rng: &mut Rng) -> Candidates {
    let dims: Vec<usize> = factors
        .iter()
        .map(Factor::level_count)
        .chain(roles.iter().map(|r| r.levels.len()))
        .collect();
    let n_factors = factors.len();
    let total: u128 = dims.iter().map(|&d| d as u128).product();

    if total <= CANDIDATE_CAP as u128 {
        let mut rows: Vec<Vec<usize>> = vec![Vec::new()];
        for &d in &dims {
            rows = rows
                .into_iter()
                .flat_map(|prefix| {
                    (0..d).map(move |j| {
                        let mut next = prefix.clone();
                        next.push(j);
                        next
                    })
                })
                .collect();
        }
        return Candidates {
            cells: rows.iter().map(|r| split_cell(r, n_factors)).collect(),
            sampled: None,
        };
    }

    // Above the cap: CANDIDATE_CAP uniform draws, kept in draw order, deduped.
    let mut seen = std::collections::HashSet::new();
    let mut cells = Vec::new();
    for _ in 0..CANDIDATE_CAP {
        let draw: Vec<usize> = dims.iter().map(|&d| rng.below(d as u64) as usize).collect();
        if seen.insert(draw.clone()) {
            cells.push(split_cell(&draw, n_factors));
        }
    }
    let kept = cells.len();
    Candidates {
        cells,
        sampled: Some((kept, total)),
    }
}

fn split_cell(indices: &[usize], n_factors: usize) -> Cell {
    Cell {
        factor_levels: indices[..n_factors].to_vec(),
        replicate_levels: indices[n_factors..].to_vec(),
    }
}

// --------------------------------------------------------- linear algebra ----

/// The right singular vectors of a matrix, kept so that many row-space tests
/// share one decomposition. The tolerance rule is `model::is_estimable`'s; the
/// unit test `fast_row_space_matches_is_estimable` holds the two in agreement.
struct RowSpace {
    vt: DMatrix<f64>,
    rank: usize,
    tolerance: f64,
    p: usize,
}

fn row_space(x: &[Vec<f64>], p: usize) -> RowSpace {
    if p == 0 || x.is_empty() {
        return RowSpace {
            vt: DMatrix::identity(p, p),
            rank: 0,
            tolerance: 0.0,
            p,
        };
    }
    // Pad wide matrices with zero rows so V is the complete p × p basis.
    let a = DMatrix::from_fn(
        x.len().max(p),
        p,
        |i, j| if i < x.len() { x[i][j] } else { 0.0 },
    );
    let svd = a.svd(false, true);
    let singular = svd.singular_values.as_slice().to_vec();
    let tolerance = 1e-10 * singular.iter().copied().fold(0.0, f64::max) * x.len().max(p) as f64;
    RowSpace {
        rank: singular.iter().filter(|&&s| s > tolerance).count(),
        vt: svd.v_t.expect("requested V transpose"),
        tolerance,
        p,
    }
}

impl RowSpace {
    /// True when `v`, read through `cols`, lies in the row space.
    fn contains_at(&self, v: &[f64], cols: &[usize]) -> bool {
        let null_norm = (self.rank..self.p)
            .map(|k| {
                cols.iter()
                    .enumerate()
                    .map(|(j, &c)| self.vt[(k, j)] * v[c])
                    .sum::<f64>()
                    .powi(2)
            })
            .sum::<f64>()
            .sqrt();
        null_norm <= self.tolerance
    }
}

/// (X'X + εI)⁻¹ and its log determinant.
struct Ridge {
    inverse: DMatrix<f64>,
    log_det: f64,
}

fn ridge(x: &[Vec<f64>], p: usize) -> Result<Ridge> {
    let m = DMatrix::from_fn(x.len(), p, |i, j| x[i][j]);
    let mut a = m.transpose() * &m;
    for i in 0..p {
        a[(i, i)] += RIDGE;
    }
    let chol = Cholesky::new(a).ok_or_else(|| {
        anyhow!(
            "the information matrix X'X + {}I is not positive definite",
            RIDGE
        )
    })?;
    let log_det = 2.0 * chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();
    Ok(Ridge {
        inverse: chol.inverse(),
        log_det,
    })
}

fn quadratic_form(inverse: &DMatrix<f64>, x: &[f64]) -> f64 {
    x.iter()
        .enumerate()
        .filter(|(_, v)| **v != 0.0)
        .map(|(i, v)| {
            v * x
                .iter()
                .enumerate()
                .map(|(j, w)| inverse[(i, j)] * w)
                .sum::<f64>()
        })
        .sum()
}

// ---------------------------------------------------------------- output ----

fn parse_id(prefix: char, s: &str) -> Option<(usize, usize)> {
    let digits = s.strip_prefix(prefix)?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some((digits.parse().ok()?, digits.len()))
}

struct IdPlan {
    new_ids: Vec<String>,
    new_orders: Vec<usize>,
    /// Set when the width grew: the re-padded ids of the original CSV rows.
    rewritten: Option<Vec<String>>,
    width: usize,
}

fn plan_ids(table: &Table, design: &Design, n_add: usize) -> Result<IdPlan> {
    let from_csv = table.run_id_col.is_some();
    let source: Vec<String> = match table.run_id_col {
        Some(c) => table
            .rows
            .iter()
            .map(|r| r.get(c).unwrap_or("").trim().to_string())
            .collect(),
        None => design.runs.iter().map(|r| r.run_id.clone()).collect(),
    };
    let mut numbers = Vec::new();
    let mut width = 0usize;
    for id in &source {
        match parse_id('r', id) {
            Some((n, w)) => {
                numbers.push(n);
                width = width.max(w);
            }
            // A CSV run_id we cannot read would make the continued sequence a
            // guess; refuse instead of inventing one.
            None if from_csv => bail!(
                "run_id '{}' is not of the form r<digits>, so the sequence cannot be continued",
                id
            ),
            None => {}
        }
    }
    let width = width.max(4);
    let next = numbers
        .iter()
        .copied()
        .max()
        .map_or(table.rows.len(), |m| m.max(table.rows.len()))
        + 1;
    let last = next + n_add.saturating_sub(1);
    let final_width = width.max(last.to_string().len());
    // Re-pad whenever an existing id does not already carry the final width,
    // so that every id in the file reads the same.
    let inconsistent = source
        .iter()
        .any(|id| parse_id('r', id).is_some_and(|(_, w)| w != final_width));
    let rewritten = (inconsistent && from_csv).then(|| {
        source
            .iter()
            .map(|id| match parse_id('r', id) {
                Some((n, _)) => format!("r{:0final_width$}", n),
                None => id.clone(),
            })
            .collect()
    });

    let max_order = match table.order_col {
        Some(c) => table
            .rows
            .iter()
            .filter_map(|r| r.get(c).and_then(|s| s.trim().parse::<usize>().ok()))
            .max(),
        None => design.runs.iter().map(|r| r.order).max(),
    }
    .unwrap_or(0)
    .max(table.rows.len());

    Ok(IdPlan {
        new_ids: (0..n_add)
            .map(|i| format!("r{:0final_width$}", next + i))
            .collect(),
        new_orders: (0..n_add).map(|i| max_order + 1 + i).collect(),
        rewritten,
        width: final_width,
    })
}

/// The unit-identity key of a cell: resolved level indices of the unit's key
/// columns, in the unit's declared order.
fn unit_key(cell: &Cell, unit: &Unit, factors: &[Factor], roles: &[ReplicateRole]) -> Vec<usize> {
    unit.key
        .iter()
        .map(|name| {
            if let Some(f) = factors.iter().position(|f| &f.name == name) {
                cell.factor_levels[f]
            } else if let Some(r) = roles.iter().position(|r| &r.name == name) {
                cell.replicate_levels[r]
            } else {
                usize::MAX
            }
        })
        .collect()
}

struct Units {
    ids: HashMap<Vec<usize>, String>,
    next: usize,
    width: usize,
}

impl Units {
    fn get_or_make(&mut self, key: Vec<usize>) -> String {
        if let Some(id) = self.ids.get(&key) {
            return id.clone();
        }
        let id = format!("u{:0width$}", self.next, width = self.width);
        self.next += 1;
        self.ids.insert(key, id.clone());
        id
    }
}

// ----------------------------------------------------------------- report ----

fn print_estimability(title: &str, est: &Estimability, factors: &[Factor], has_blocks: bool) {
    println!("{}", title);
    println!(
        "Params:   {}   rank: {}   residual df: {}   rows: {}",
        est.n_params, est.rank, est.residual_df, est.n_rows
    );
    if has_blocks {
        println!(
            "Blocks:   {}",
            if est.blocks_estimable {
                "estimable"
            } else {
                "not estimable"
            }
        );
    }
    println!(
        "  {:<16}{:>3}  {:<9}  aliased with",
        "term", "df", "estimable"
    );
    for term in &est.terms {
        println!(
            "  {:<16}{:>3}  {:<9}  {}",
            term.term.label(factors),
            term.df,
            if term.estimable { "yes" } else { "no" },
            term.aliased_with.join(", ")
        );
    }
}

// -------------------------------------------------------------- the work ----

pub fn augment(args: AugmentArgs) -> Result<Outcome> {
    if args.runs == Some(0) {
        bail!("--runs must be at least 1");
    }
    let design_path = args
        .design_path
        .clone()
        .unwrap_or_else(|| sidecar_path(&args.csv_path));
    let mut design: Design = serde_json::from_reader(
        File::open(&design_path)
            .with_context(|| format!("opening design file {}", design_path.display()))?,
    )
    .with_context(|| format!("reading design file {}", design_path.display()))?;

    let table = read_table(&args.csv_path, &design)?;
    let seed = args.seed.or(design.randomization_seed).unwrap_or_else(|| {
        // Any value is a valid seed; the one used is recorded in the sidecar.
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0x9e37_79b9_7f4a_7c15, |d| d.as_nanos() as u64)
    });
    let mut rng = Rng::new(seed);

    // Step 1: completed rows form the base; pending rows join only on request.
    let mut base_cells = Vec::new();
    let mut n_pending = 0usize;
    for i in 0..table.rows.len() {
        let complete = is_complete(&table, i);
        if !complete {
            n_pending += 1;
        }
        if complete || args.include_pending {
            base_cells.push(row_cell(&table, &design, i)?);
        }
    }

    let target_labels: Vec<String> = args
        .targets
        .iter()
        .map(|src| target_label(src, &design.factors))
        .collect::<Result<_>>()?;
    let model = union_model(&design, &target_labels)?;
    let added_terms: Vec<&String> = target_labels
        .iter()
        .filter(|label| {
            !design
                .models
                .iter()
                .any(|m| m.terms.iter().any(|t| &t.label(&design.factors) == *label))
        })
        .collect();

    println!("Augmenting: {}", args.csv_path.display());
    println!("Model:      {}", model.formula);
    if !added_terms.is_empty() && !design.models.is_empty() {
        println!(
            "Note:       {} requested by --for but absent from the design's formulas;",
            added_terms
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!("            add it to the design file before you analyze the result.");
    }
    println!(
        "Base rows:  {} of {} ({} pending, {})",
        base_cells.len(),
        table.rows.len(),
        n_pending,
        if args.include_pending {
            "included"
        } else {
            "excluded"
        }
    );

    let base_mm = model_matrix(
        &design.factors,
        &design.replicates,
        &model,
        &base_cells,
        None,
    );
    let p = base_mm.columns.len();
    let before = estimability(&base_mm, &model, &design.factors);
    let has_blocks = !design.replicates.is_empty();
    print_estimability(
        "\nEstimability before augmentation",
        &before,
        &design.factors,
        has_blocks,
    );

    // Step 3's targets: the requested terms, else every non-estimable term.
    let target_terms: Vec<usize> = if target_labels.is_empty() {
        before
            .terms
            .iter()
            .enumerate()
            .filter(|(_, t)| !t.estimable)
            .map(|(i, _)| i)
            .collect()
    } else {
        target_labels
            .iter()
            .map(|label| {
                model
                    .terms
                    .iter()
                    .position(|t| &t.label(&design.factors) == label)
                    .ok_or_else(|| anyhow!("term '{}' is not in the model", label))
            })
            .collect::<Result<_>>()?
    };
    let targets: Vec<String> = target_terms
        .iter()
        .map(|&t| model.terms[t].label(&design.factors))
        .collect();
    if target_terms.is_empty() && args.runs.is_none() {
        bail!(
            "every term of '{}' is already estimable; pass --runs N to add runs for precision, \
             or --for TERM to name a target",
            model.formula
        );
    }
    println!(
        "\nTargets:    {}",
        if targets.is_empty() {
            "none (pure D-optimal expansion)".to_string()
        } else {
            targets.join(", ")
        }
    );
    if !targets.is_empty()
        && target_terms.iter().all(|&t| before.terms[t].estimable)
        && args.runs.is_none()
    {
        println!("            already estimable; nothing to add (pass --runs N for precision)");
    }

    // Step 2: the candidate set.
    let candidates = candidate_cells(&design.factors, &design.replicates, &mut rng);
    if let Some((kept, total)) = candidates.sampled {
        println!(
            "Candidates: {} sampled uniformly with seed {} from {} combinations (cap {})",
            kept, seed, total, CANDIDATE_CAP
        );
    } else {
        println!("Candidates: {} combinations", candidates.cells.len());
    }
    if candidates.cells.is_empty() {
        bail!("the design has no candidate combinations to add");
    }
    let candidate_mm = model_matrix(
        &design.factors,
        &design.replicates,
        &model,
        &candidates.cells,
        None,
    );
    assert_eq!(
        candidate_mm.columns.len(),
        p,
        "candidate and base matrices must share their columns"
    );

    // Step 3: the greedy loop.
    let target_cols: Vec<usize> = base_mm
        .columns
        .iter()
        .enumerate()
        .filter(|(_, c)| matches!(c.source, ColumnSource::Term { term } if target_terms.contains(&term)))
        .map(|(j, _)| j)
        .collect();
    let all_cols: Vec<usize> = (0..p).collect();
    let reduced_cols: Vec<usize> = all_cols
        .iter()
        .copied()
        .filter(|j| !target_cols.contains(j))
        .collect();

    let mut current = base_mm.data.clone();
    let mut chosen: Vec<usize> = Vec::new();
    let hard_cap = args.runs.unwrap_or(p);
    let log_det_before = ridge(&current, p)?.log_det;
    loop {
        let now = ModelMatrix {
            columns: base_mm.columns.clone(),
            data: current.clone(),
        };
        let est = estimability(&now, &model, &design.factors);
        let targets_done = target_terms.iter().all(|&t| est.terms[t].estimable);
        let enough = match args.runs {
            Some(n) => chosen.len() >= n,
            None => targets_done,
        };
        if (targets_done && enough) || chosen.len() >= hard_cap {
            break;
        }
        let full = row_space(&current, p);
        let reduced: Vec<Vec<f64>> = current
            .iter()
            .map(|row| reduced_cols.iter().map(|&j| row[j]).collect())
            .collect();
        let partial = row_space(&reduced, reduced_cols.len());
        let information = ridge(&current, p)?;
        let mut best: Option<(i32, f64, usize)> = None;
        for (i, row) in candidate_mm.data.iter().enumerate() {
            let rank_gain = if target_cols.is_empty() {
                0
            } else {
                i32::from(!full.contains_at(row, &all_cols))
                    - i32::from(!partial.contains_at(row, &reduced_cols))
            };
            let d_gain = (1.0 + quadratic_form(&information.inverse, row)).ln();
            // Strictly better wins, so ties keep enumeration order.
            if best.is_none_or(|(g, d, _)| rank_gain > g || (rank_gain == g && d_gain > d)) {
                best = Some((rank_gain, d_gain, i));
            }
        }
        let (_, _, pick) = best.expect("candidate set is not empty");
        current.push(candidate_mm.data[pick].clone());
        chosen.push(pick);
    }

    let final_mm = ModelMatrix {
        columns: base_mm.columns.clone(),
        data: current.clone(),
    };
    let after = estimability(&final_mm, &model, &design.factors);
    let log_det_after = ridge(&current, p)?.log_det;
    let targets_estimable = target_terms.iter().all(|&t| after.terms[t].estimable);
    print_estimability(
        "\nEstimability after augmentation",
        &after,
        &design.factors,
        has_blocks,
    );
    println!(
        "\nAdded {} run(s). log det(X'X + εI): {:.6} → {:.6}",
        chosen.len(),
        log_det_before,
        log_det_after
    );

    // Step 4: write the CSV and the sidecar.
    let plan = plan_ids(&table, &design, chosen.len())?;
    if plan.rewritten.is_some() {
        println!(
            "Note:       run ids widened to {} digits; every id was re-padded.",
            plan.width
        );
    }
    let new_cells: Vec<&Cell> = chosen.iter().map(|&i| &candidates.cells[i]).collect();
    let mut units = collect_units(&table, &design)?;
    let new_units: Vec<Option<String>> = new_cells
        .iter()
        .map(|cell| {
            design
                .unit
                .as_ref()
                .map(|u| units.get_or_make(unit_key(cell, u, &design.factors, &design.replicates)))
        })
        .collect();

    let next_experiment = table.experiment_col.map(|c| {
        table
            .rows
            .iter()
            .filter_map(|r| r.get(c).and_then(|s| s.trim().parse::<usize>().ok()))
            .max()
            .unwrap_or(0)
            .max(table.rows.len())
            + 1
    });

    let mut wtr = csv::Writer::from_path(&args.output)
        .with_context(|| format!("creating {}", args.output.display()))?;
    wtr.write_record(&table.headers)?;
    for (i, row) in table.rows.iter().enumerate() {
        match (&plan.rewritten, table.run_id_col) {
            (Some(ids), Some(col)) => {
                let cells: Vec<&str> = row
                    .iter()
                    .enumerate()
                    .map(|(j, cell)| if j == col { ids[i].as_str() } else { cell })
                    .collect();
                wtr.write_record(&cells)?;
            }
            _ => wtr.write_record(row)?,
        }
    }
    for (k, cell) in new_cells.iter().enumerate() {
        let record: Vec<String> = (0..table.headers.len())
            .map(|col| {
                new_cell_text(
                    &table,
                    &design,
                    cell,
                    col,
                    k,
                    &plan,
                    &new_units,
                    next_experiment,
                )
            })
            .collect();
        wtr.write_record(&record)?;
    }
    wtr.flush()?;
    println!("✓ wrote {}", args.output.display());

    let new_runs: Vec<Run> = new_cells
        .iter()
        .enumerate()
        .map(|(k, cell)| Run {
            run_id: plan.new_ids[k].clone(),
            order: plan.new_orders[k],
            design_row: usize::MAX,
            factors: design
                .factors
                .iter()
                .zip(&cell.factor_levels)
                .map(|(f, &l)| (f.name.clone(), f.values[l].clone()))
                .collect(),
            replicates: design
                .replicates
                .iter()
                .zip(&cell.replicate_levels)
                .map(|(r, &l)| (r.name.clone(), r.levels[l].clone()))
                .collect(),
            unit: new_units[k].clone(),
        })
        .collect();
    if plan.rewritten.is_some() {
        let width = plan.width;
        for run in &mut design.runs {
            if let Some((n, _)) = parse_id('r', &run.run_id) {
                run.run_id = format!("r{:0width$}", n);
            }
        }
    }
    design.runs.extend(new_runs);
    design.augmentations.push(Augmentation {
        seed,
        targets: targets.clone(),
        run_ids: plan.new_ids.clone(),
    });
    let out_sidecar = sidecar_path(&args.output);
    let mut f = File::create(&out_sidecar)
        .with_context(|| format!("creating {}", out_sidecar.display()))?;
    serde_json::to_writer_pretty(&mut f, &design)?;
    writeln!(f)?;
    println!("✓ wrote {}", out_sidecar.display());

    Ok(Outcome {
        added: chosen.len(),
        targets,
        targets_estimable,
        before,
        after,
        log_det_before,
        log_det_after,
        run_ids: plan.new_ids,
        renumbered: plan.rewritten.is_some(),
    })
}

/// Unit ids already in use, keyed by the resolved level indices of the key.
fn collect_units(table: &Table, design: &Design) -> Result<Units> {
    let mut units = Units {
        ids: HashMap::new(),
        next: 1,
        width: 4,
    };
    let Some(unit) = design.unit.as_ref() else {
        return Ok(units);
    };
    let by_run_id: HashMap<&str, &Run> =
        design.runs.iter().map(|r| (r.run_id.as_str(), r)).collect();
    let mut pending = Vec::new();
    for i in 0..table.rows.len() {
        let row = &table.rows[i];
        let key = unit_key(
            &row_cell(table, design, i)?,
            unit,
            &design.factors,
            &design.replicates,
        );
        let existing = table
            .unit_col
            .map(|c| row.get(c).unwrap_or("").trim().to_string())
            .filter(|s| !s.is_empty())
            .or_else(|| {
                let run = match table.run_id_col {
                    Some(c) => by_run_id.get(row.get(c).unwrap_or("").trim()).copied(),
                    None => design.runs.get(i),
                };
                run.and_then(|r| r.unit.clone())
            });
        match existing {
            Some(id) => {
                if let Some((n, w)) = parse_id('u', &id) {
                    units.next = units.next.max(n + 1);
                    units.width = units.width.max(w);
                }
                units.ids.entry(key).or_insert(id);
            }
            None => pending.push(key),
        }
    }
    // Only after every explicit id is known, so generated ids cannot collide.
    for key in pending {
        units.get_or_make(key);
    }
    Ok(units)
}

#[allow(clippy::too_many_arguments)]
fn new_cell_text(
    table: &Table,
    design: &Design,
    cell: &Cell,
    col: usize,
    k: usize,
    plan: &IdPlan,
    new_units: &[Option<String>],
    next_experiment: Option<usize>,
) -> String {
    if Some(col) == table.run_id_col {
        return plan.new_ids[k].clone();
    }
    if Some(col) == table.order_col {
        return plan.new_orders[k].to_string();
    }
    if Some(col) == table.experiment_col {
        return next_experiment.map_or(String::new(), |n| (n + k).to_string());
    }
    if Some(col) == table.unit_col {
        return new_units[k].clone().unwrap_or_default();
    }
    if let Some(f) = table.factor_col.iter().position(|&c| c == col) {
        return design.factors[f].values[cell.factor_levels[f]].to_string();
    }
    if let Some(r) = table.role_col.iter().position(|&c| c == col) {
        return design.replicates[r].levels[cell.replicate_levels[r]].to_string();
    }
    // Result cells stay empty, and so does any column we do not own.
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::is_estimable;
    use crate::oa::ArrayInfo;

    /// A self-describing temp directory that removes itself, pass or fail.
    struct Scratch(PathBuf);
    impl Scratch {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "taguchi-augment-test-{}-{}",
                name,
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Scratch(dir)
        }
        fn at(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn two_level(names: &[&str]) -> Vec<Factor> {
        names
            .iter()
            .map(|n| Factor::parse(n, "d[1,2]").unwrap())
            .collect()
    }

    fn fixture_design(names: &[&str], formulas: &[&str]) -> Design {
        let factors = two_level(names);
        Design {
            version: 2,
            models: formulas
                .iter()
                .map(|f| Model::parse(f, &factors).unwrap())
                .collect(),
            replicates: Vec::new(),
            unit: None,
            array: ArrayInfo {
                rows: 4,
                cols: factors.len(),
                label: "L4".into(),
                method: "fixture".into(),
            },
            randomization_seed: Some(7),
            estimability: None,
            runs: Vec::new(),
            augmentations: Vec::new(),
            results: vec!["y".to_string()],
            factors,
        }
    }

    fn put(path: &Path, text: &str) {
        std::fs::write(path, text).unwrap();
    }

    fn put_design(csv: &Path, design: &Design) {
        put(
            &sidecar_path(csv),
            &serde_json::to_string_pretty(design).unwrap(),
        );
    }

    fn read_json(path: &Path) -> serde_json::Value {
        serde_json::from_reader(File::open(path).unwrap()).unwrap()
    }

    fn args(csv: &Path, out: &Path) -> AugmentArgs {
        AugmentArgs {
            csv_path: csv.to_path_buf(),
            design_path: None,
            targets: Vec::new(),
            runs: None,
            seed: Some(1),
            include_pending: false,
            output: out.to_path_buf(),
        }
    }

    /// (a) L4 half fraction: a:b is aliased with c. `--for a:b` must repair it
    /// with a small number of extra rows.
    #[test]
    fn half_fraction_alias_is_resolved() {
        let dir = Scratch::new("alias");
        let csv = dir.at("exp.csv");
        let out = dir.at("aug.csv");
        put(
            &csv,
            "run_id,order,a,b,c,y\n\
             r0001,1,1,1,1,1.0\n\
             r0002,2,1,2,2,2.0\n\
             r0003,3,2,1,2,3.5\n\
             r0004,4,2,2,1,4.0\n",
        );
        let design = fixture_design(&["a", "b", "c"], &["y ~ a + b + c + a:b"]);
        put_design(&csv, &design);

        let outcome = augment(AugmentArgs {
            targets: vec!["a:b".to_string()],
            ..args(&csv, &out)
        })
        .unwrap();

        let interaction = outcome
            .before
            .terms
            .iter()
            .find(|t| t.term.label(&design.factors) == "a:b")
            .unwrap();
        assert!(!interaction.estimable, "a:b must start out aliased");
        assert!(interaction.aliased_with.contains(&"c".to_string()));
        assert!(
            outcome.added >= 1 && outcome.added <= 4,
            "expected at most 4 added rows, got {}",
            outcome.added
        );
        assert!(outcome.targets_estimable, "a:b must be estimable after");
        assert!(
            outcome
                .after
                .terms
                .iter()
                .find(|t| t.term.label(&design.factors) == "a:b")
                .unwrap()
                .estimable
        );
        assert!(!outcome.renumbered);

        let text = std::fs::read_to_string(&out).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "run_id,order,a,b,c,y");
        assert_eq!(lines[1], "r0001,1,1,1,1,1.0");
        assert_eq!(lines[4], "r0004,4,2,2,1,4.0");
        assert_eq!(lines.len(), 5 + outcome.added);
        assert!(lines[5].starts_with("r0005,5,"), "got {}", lines[5]);
        assert!(lines[5].ends_with(','), "result cell must stay empty");

        let side = read_json(&sidecar_path(&out));
        assert_eq!(side["augmentations"][0]["targets"][0], "a:b");
        assert_eq!(side["augmentations"][0]["seed"], 1);
        assert_eq!(
            side["augmentations"][0]["run_ids"]
                .as_array()
                .unwrap()
                .len(),
            outcome.added
        );
        assert_eq!(side["runs"].as_array().unwrap().len(), outcome.added);
        assert_eq!(side["runs"][0]["run_id"], "r0005");
        assert_eq!(side["runs"][0]["design_row"], usize::MAX);
    }

    /// (b) Every term already estimable: `--runs 4` adds 4 rows by D-gain and
    /// det(X'X) grows. Without `--runs` the command is an error.
    #[test]
    fn pure_d_gain_needs_runs_and_improves_the_determinant() {
        let dir = Scratch::new("dgain");
        let csv = dir.at("exp.csv");
        let out = dir.at("aug.csv");
        put(
            &csv,
            "run_id,order,a,b,y\n\
             r0001,1,1,1,1.0\n\
             r0002,2,1,2,2.0\n\
             r0003,3,2,1,3.0\n\
             r0004,4,2,2,4.0\n",
        );
        put_design(&csv, &fixture_design(&["a", "b"], &["y ~ a + b + a:b"]));

        let refused = augment(args(&csv, &out)).unwrap_err().to_string();
        assert!(refused.contains("--runs"), "got: {}", refused);

        let outcome = augment(AugmentArgs {
            runs: Some(4),
            ..args(&csv, &out)
        })
        .unwrap();
        assert!(outcome.before.terms.iter().all(|t| t.estimable));
        assert_eq!(outcome.added, 4);
        assert!(outcome.targets_estimable);
        assert!(
            outcome.log_det_after > outcome.log_det_before,
            "det(X'X) must grow: {} -> {}",
            outcome.log_det_before,
            outcome.log_det_after
        );
        assert_eq!(outcome.run_ids, ["r0005", "r0006", "r0007", "r0008"]);
        assert_eq!(std::fs::read_to_string(&out).unwrap().lines().count(), 9);
    }

    /// (c) A version-1 sidecar (no `runs`, no `version`) with the old
    /// `experiment` column loads and augments.
    #[test]
    fn version_one_sidecar_and_experiment_column() {
        let dir = Scratch::new("v1");
        let csv = dir.at("exp.csv");
        let out = dir.at("aug.csv");
        put(
            &csv,
            "experiment,a,b,c,y\n\
             1,1,1,1,1.0\n\
             2,1,2,2,2.0\n\
             3,2,1,2,3.0\n\
             4,2,2,1,4.0\n",
        );
        let factors = two_level(&["a", "b", "c"]);
        put(
            &sidecar_path(&csv),
            &serde_json::to_string_pretty(&serde_json::json!({
                "factors": serde_json::to_value(&factors).unwrap(),
                "results": ["y"],
                "array": {"rows": 4, "cols": 3, "label": "L4", "method": "fixture"},
            }))
            .unwrap(),
        );

        let outcome = augment(AugmentArgs {
            runs: Some(2),
            seed: Some(3),
            ..args(&csv, &out)
        })
        .unwrap();
        assert_eq!(outcome.added, 2);
        assert!(outcome.targets_estimable);
        // No [model] section: the union model is the main effects.
        assert_eq!(outcome.before.terms.len(), 3);

        let text = std::fs::read_to_string(&out).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "experiment,a,b,c,y");
        assert_eq!(lines[1], "1,1,1,1,1.0");
        assert_eq!(lines.len(), 7);
        assert!(lines[5].starts_with("5,"), "got {}", lines[5]);
        assert!(lines[6].starts_with("6,"), "got {}", lines[6]);

        let side = read_json(&sidecar_path(&out));
        assert_eq!(side["version"], 2);
        assert_eq!(side["runs"].as_array().unwrap().len(), 2);
        assert_eq!(side["runs"][0]["run_id"], "r0005");
        assert_eq!(side["runs"][0]["order"], 5);
    }

    /// (d) The same seed produces the same added rows.
    #[test]
    fn same_seed_gives_the_same_rows() {
        let dir = Scratch::new("seed");
        let csv = dir.at("exp.csv");
        put(
            &csv,
            "run_id,order,a,b,c,y\n\
             r0001,1,1,1,1,1.0\n\
             r0002,2,1,2,2,2.0\n\
             r0003,3,2,1,2,3.5\n\
             r0004,4,2,2,1,4.0\n",
        );
        put_design(
            &csv,
            &fixture_design(&["a", "b", "c"], &["y ~ a + b + c + a:b"]),
        );
        let run = |name: &str| {
            let out = dir.at(name);
            let outcome = augment(AugmentArgs {
                targets: vec!["a:b".to_string()],
                runs: Some(3),
                seed: Some(11),
                ..args(&csv, &out)
            })
            .unwrap();
            (outcome.run_ids, std::fs::read_to_string(&out).unwrap())
        };
        let (ids_a, csv_a) = run("one.csv");
        let (ids_b, csv_b) = run("two.csv");
        assert_eq!(ids_a, ids_b);
        assert_eq!(csv_a, csv_b);
        assert_eq!(csv_a.lines().count(), 8);
    }

    /// Pending rows stay out of the base unless `--include-pending`.
    #[test]
    fn pending_rows_are_excluded_by_default() {
        let dir = Scratch::new("pending");
        let csv = dir.at("exp.csv");
        put(
            &csv,
            "run_id,order,a,b,y\n\
             r0001,1,1,1,1.0\n\
             r0002,2,1,2,2.0\n\
             r0003,3,2,1,\n\
             r0004,4,2,2,\n",
        );
        put_design(&csv, &fixture_design(&["a", "b"], &["y ~ a + b + a:b"]));
        let excluded = augment(AugmentArgs {
            runs: Some(1),
            ..args(&csv, &dir.at("x.csv"))
        })
        .unwrap();
        let included = augment(AugmentArgs {
            runs: Some(1),
            include_pending: true,
            ..args(&csv, &dir.at("y.csv"))
        })
        .unwrap();
        assert_eq!(excluded.before.n_rows, 2);
        assert_eq!(included.before.n_rows, 4);
        // Both keep all four original rows in the output.
        assert_eq!(
            std::fs::read_to_string(dir.at("x.csv"))
                .unwrap()
                .lines()
                .count(),
            6
        );
    }

    /// The greedy row-space test and `model::is_estimable` must agree; the
    /// greedy loop only amortises the decomposition.
    #[test]
    fn fast_row_space_matches_is_estimable() {
        let full = vec![
            vec![1.0, 1.0, 1.0, 1.0],
            vec![1.0, 1.0, -1.0, -1.0],
            vec![1.0, -1.0, 1.0, -1.0],
            vec![1.0, -1.0, -1.0, 1.0],
        ];
        // Rank deficient: the third column repeats the second.
        let deficient = vec![
            vec![1.0, 1.0, 1.0],
            vec![1.0, -1.0, -1.0],
            vec![1.0, 1.0, 1.0],
        ];
        for matrix in [&full, &deficient] {
            let p = matrix[0].len();
            let cols: Vec<usize> = (0..p).collect();
            let space = row_space(matrix, p);
            let mut vectors: Vec<Vec<f64>> = vec![vec![]];
            for _ in 0..p {
                vectors = vectors
                    .into_iter()
                    .flat_map(|prefix| {
                        [-1.0f64, 0.0, 1.0].into_iter().map(move |v| {
                            let mut next = prefix.clone();
                            next.push(v);
                            next
                        })
                    })
                    .collect();
            }
            for v in vectors {
                assert_eq!(
                    space.contains_at(&v, &cols),
                    is_estimable(matrix, &v),
                    "disagreement on {:?} for {:?}",
                    v,
                    matrix
                );
            }
        }
    }

    /// A `--for` term the design never declared is added to the working model
    /// rather than refused, so an alias can be resolved after the fact.
    #[test]
    fn for_term_outside_the_declared_model_is_added() {
        let dir = Scratch::new("extend");
        let csv = dir.at("exp.csv");
        put(
            &csv,
            "run_id,order,a,b,c,y\n\
             r0001,1,1,1,1,1.0\n\
             r0002,2,1,2,2,2.0\n\
             r0003,3,2,1,2,3.0\n\
             r0004,4,2,2,1,4.0\n",
        );
        put_design(&csv, &fixture_design(&["a", "b", "c"], &["y ~ a + b + c"]));
        let outcome = augment(AugmentArgs {
            targets: vec!["b:a".to_string()],
            ..args(&csv, &dir.at("aug.csv"))
        })
        .unwrap();
        // 'b:a' canonicalises to 'a:b'.
        assert_eq!(outcome.targets, ["a:b"]);
        assert_eq!(outcome.before.terms.len(), 4);
        assert!(outcome.targets_estimable);
        assert!(outcome.added >= 1);
    }

    /// A run_id we cannot continue is an error, not a silent renumbering.
    #[test]
    fn unreadable_run_id_is_refused() {
        let dir = Scratch::new("badid");
        let csv = dir.at("exp.csv");
        put(
            &csv,
            "run_id,order,a,b,y\n\
             x1,1,1,1,1.0\n\
             x2,2,1,2,2.0\n\
             x3,3,2,1,3.0\n\
             x4,4,2,2,4.0\n",
        );
        put_design(&csv, &fixture_design(&["a", "b"], &["y ~ a + b + a:b"]));
        let err = augment(AugmentArgs {
            runs: Some(1),
            ..args(&csv, &dir.at("aug.csv"))
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains("r<digits>"), "got: {}", err);
    }

    /// Acceptance 3: a new row whose unit key matches an existing unit takes
    /// that unit's id; a new key gets a fresh one.
    #[test]
    fn unit_ids_are_reused_when_the_key_matches() {
        let dir = Scratch::new("units");
        let csv = dir.at("exp.csv");
        let out = dir.at("aug.csv");
        put(
            &csv,
            "run_id,order,a,m,seed,checkpoint,y\n\
             r0001,1,1,1,1,u0001,1.0\n\
             r0002,2,1,2,1,u0001,1.5\n\
             r0003,3,2,1,1,u0002,2.0\n\
             r0004,4,2,2,1,u0002,2.5\n",
        );
        let mut design = fixture_design(&["a", "m"], &["y ~ a + m + a:m"]);
        // One replicate level, so every candidate lands in an existing unit.
        design.replicates = vec![ReplicateRole {
            name: "seed".into(),
            levels: vec![crate::factor::FactorValue::Int(1)],
        }];
        design.unit = Some(Unit {
            name: "checkpoint".into(),
            key: vec!["a".into(), "seed".into()],
        });
        put_design(&csv, &design);

        let outcome = augment(AugmentArgs {
            runs: Some(2),
            ..args(&csv, &out)
        })
        .unwrap();
        assert_eq!(outcome.added, 2);
        let text = std::fs::read_to_string(&out).unwrap();
        for line in text.lines().skip(5) {
            let cells: Vec<&str> = line.split(',').collect();
            let expected = if cells[2] == "1" { "u0001" } else { "u0002" };
            assert_eq!(cells[5], expected, "unit id must be reused in {}", line);
        }
        let side = read_json(&sidecar_path(&out));
        assert!(side["runs"][0]["unit"] == "u0001" || side["runs"][0]["unit"] == "u0002");
        assert_eq!(side["runs"][0]["replicates"]["seed"], 1);
    }

    /// Acceptance 3: when the sequence outgrows the padding, every id widens.
    #[test]
    fn run_ids_widen_and_every_id_is_rewritten() {
        let dir = Scratch::new("widen");
        let csv = dir.at("exp.csv");
        let out = dir.at("aug.csv");
        put(
            &csv,
            "run_id,order,a,b,y\n\
             r9996,1,1,1,1.0\n\
             r9997,2,1,2,2.0\n\
             r9998,3,2,1,3.0\n\
             r9999,4,2,2,4.0\n",
        );
        put_design(&csv, &fixture_design(&["a", "b"], &["y ~ a + b + a:b"]));
        let outcome = augment(AugmentArgs {
            runs: Some(2),
            ..args(&csv, &out)
        })
        .unwrap();
        assert!(outcome.renumbered);
        assert_eq!(outcome.run_ids, ["r10000", "r10001"]);
        let lines: Vec<String> = std::fs::read_to_string(&out)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect();
        assert!(lines[1].starts_with("r09996,"), "got {}", lines[1]);
        assert!(lines[4].starts_with("r09999,"), "got {}", lines[4]);
        assert!(lines[5].starts_with("r10000,"), "got {}", lines[5]);
    }
}
