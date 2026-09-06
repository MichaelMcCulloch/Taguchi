//! Shared formula, deviation coding, and rank-aware statistical operations.
use anyhow::{Result, bail};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::factor::{Factor, FactorValue};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Term(pub Vec<usize>);

impl Term {
    pub fn label(&self, factors: &[Factor]) -> String {
        self.0
            .iter()
            .map(|&i| factors[i].name.as_str())
            .collect::<Vec<_>>()
            .join(":")
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Model {
    pub formula: String,
    pub response: Option<String>,
    pub terms: Vec<Term>,
}

impl Model {
    pub fn parse(src: &str, factors: &[Factor]) -> Result<Model> {
        for unsupported in ["^", "%in%"] {
            if src.contains(unsupported) {
                bail!("unsupported formula syntax '{}'", unsupported);
            }
        }
        let (response, expr) = match src.split_once('~') {
            Some((response, expr)) => {
                let response = response.trim();
                if response != "." && !valid_name(response) {
                    bail!("invalid response name '{}'", response);
                }
                ((response != ".").then(|| response.to_owned()), expr)
            }
            None => (None, src),
        };
        let mut parser = Parser {
            tokens: tokenize(expr)?,
            pos: 0,
            factors,
        };
        let terms = parser.expr()?;
        if parser.pos != parser.tokens.len() {
            bail!(
                "unexpected token '{}' in formula",
                parser.tokens[parser.pos]
            );
        }
        Ok(Self::canonical(response, terms, factors))
    }

    fn canonical(response: Option<String>, mut terms: Vec<Term>, factors: &[Factor]) -> Model {
        terms.sort_by(|a, b| a.0.len().cmp(&b.0.len()).then(a.0.cmp(&b.0)));
        terms.dedup();
        let formula = format!(
            "{} ~ {}",
            response.as_deref().unwrap_or("."),
            terms
                .iter()
                .map(|t| t.label(factors))
                .collect::<Vec<_>>()
                .join(" + ")
        );
        Model {
            formula,
            response,
            terms,
        }
    }

    pub fn main_effects(factors: &[Factor]) -> Model {
        Self::canonical(
            None,
            (0..factors.len()).map(|i| Term(vec![i])).collect(),
            factors,
        )
    }

    pub fn for_result<'a>(models: &'a [Model], result: &str) -> Option<&'a Model> {
        models
            .iter()
            .find(|m| m.response.as_deref() == Some(result))
            .or_else(|| models.iter().find(|m| m.response.is_none()))
    }
}

fn valid_name(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
}

pub(crate) fn formula_names(expr: &str) -> Result<Vec<String>> {
    if expr.contains("%in%") {
        bail!("unsupported formula syntax '%in%'");
    }
    Ok(tokenize(expr.split_once('~').map_or(expr, |(_, rhs)| rhs))?
        .into_iter()
        .filter(|s| valid_name(s))
        .collect())
}

fn tokenize(src: &str) -> Result<Vec<String>> {
    let mut tokens = Vec::new();
    let mut name = String::new();
    for ch in src.chars() {
        if ch.is_alphanumeric() || ch == '_' || ch == '-' {
            name.push(ch);
        } else {
            if !name.is_empty() {
                tokens.push(std::mem::take(&mut name));
            }
            if "+*:()".contains(ch) {
                tokens.push(ch.to_string());
            } else if !ch.is_whitespace() {
                bail!("unsupported formula syntax '{}'", ch);
            }
        }
    }
    if !name.is_empty() {
        tokens.push(name);
    }
    if tokens.iter().any(|token| token == "-1") {
        bail!("unsupported formula syntax '-1'");
    }
    Ok(tokens)
}

struct Parser<'a> {
    tokens: Vec<String>,
    pos: usize,
    factors: &'a [Factor],
}

impl Parser<'_> {
    fn eat(&mut self, token: &str) -> bool {
        if self.tokens.get(self.pos).is_some_and(|s| s == token) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn expr(&mut self) -> Result<Vec<Term>> {
        let mut terms = self.prod()?;
        while self.eat("+") {
            terms.extend(self.prod()?);
        }
        Ok(terms)
    }
    fn prod(&mut self) -> Result<Vec<Term>> {
        let mut terms = self.inter()?;
        while self.eat("*") {
            let right = self.inter()?;
            let products = cross_terms(&terms, &right)?;
            terms.extend(right);
            terms.extend(products);
        }
        Ok(terms)
    }
    fn inter(&mut self) -> Result<Vec<Term>> {
        let mut terms = self.atom()?;
        while self.eat(":") {
            terms = cross_terms(&terms, &self.atom()?)?;
        }
        Ok(terms)
    }
    fn atom(&mut self) -> Result<Vec<Term>> {
        if self.eat("(") {
            let terms = self.expr()?;
            if !self.eat(")") {
                bail!("expected ')' in formula");
            }
            return Ok(terms);
        }
        let Some(name) = self.tokens.get(self.pos) else {
            bail!("expected factor name in formula");
        };
        if !valid_name(name) {
            bail!("expected factor name, got '{}'", name);
        }
        let Some(index) = self.factors.iter().position(|f| f.name == *name) else {
            if name.contains("-1") {
                bail!("unsupported formula syntax '-1'");
            }
            bail!("unknown factor name '{}' in formula", name);
        };
        self.pos += 1;
        Ok(vec![Term(vec![index])])
    }
}

fn cross_terms(left: &[Term], right: &[Term]) -> Result<Vec<Term>> {
    let mut result = Vec::new();
    for a in left {
        for b in right {
            if a.0.iter().any(|i| b.0.contains(i)) {
                bail!("repeated factor in interaction (a:a is not supported)");
            }
            let mut indices = a.0.clone();
            indices.extend(&b.0);
            indices.sort_unstable();
            result.push(Term(indices));
        }
    }
    Ok(result)
}

pub fn hierarchy_warnings(model: &Model, factors: &[Factor]) -> Vec<String> {
    let mut warnings = Vec::new();
    for term in &model.terms {
        if term.0.len() < 2 {
            continue;
        }
        // Check each immediate lower-order term; this includes main effects for pairs.
        for omitted in (0..term.0.len()).rev() {
            let mut lower = term.clone();
            lower.0.remove(omitted);
            if !model.terms.contains(&lower) {
                warnings.push(format!(
                    "{} requested without {}",
                    term.label(factors),
                    lower.label(factors)
                ));
            }
        }
    }
    warnings
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ReplicateRole {
    pub name: String,
    pub levels: Vec<FactorValue>,
}
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Unit {
    pub name: String,
    pub key: Vec<String>,
}
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Cell {
    pub factor_levels: Vec<usize>,
    pub replicate_levels: Vec<usize>,
}
#[derive(Clone, Debug)]
pub struct ModelMatrix {
    pub columns: Vec<ColumnInfo>,
    pub data: Vec<Vec<f64>>,
}
#[derive(Clone, Debug)]
pub struct ColumnInfo {
    pub source: ColumnSource,
    pub label: String,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ColumnSource {
    Intercept,
    Block { role: usize },
    Term { term: usize },
    Unit,
}

fn code(level: usize, column: usize, count: usize) -> f64 {
    assert!(level < count, "cell must contain resolved level indices");
    if level == count - 1 {
        -1.0
    } else if level == column {
        1.0
    } else {
        0.0
    }
}

pub fn model_matrix(
    factors: &[Factor],
    roles: &[ReplicateRole],
    model: &Model,
    cells: &[Cell],
    include_unit: Option<&Unit>,
) -> ModelMatrix {
    let mut mm = ModelMatrix {
        columns: vec![ColumnInfo {
            source: ColumnSource::Intercept,
            label: "(Intercept)".into(),
        }],
        data: vec![vec![1.0]; cells.len()],
    };
    for (r, role) in roles.iter().enumerate() {
        for j in 0..role.levels.len().saturating_sub(1) {
            mm.columns.push(ColumnInfo {
                source: ColumnSource::Block { role: r },
                label: format!("{}[{}]", role.name, role.levels[j]),
            });
            for (row, cell) in mm.data.iter_mut().zip(cells) {
                row.push(code(cell.replicate_levels[r], j, role.levels.len()));
            }
        }
    }
    for (t, term) in model.terms.iter().enumerate() {
        let mut combinations: Vec<Vec<(usize, usize)>> = vec![vec![]];
        for &f in &term.0 {
            combinations = combinations
                .into_iter()
                .flat_map(|prefix| {
                    (0..factors[f].level_count() - 1).map(move |j| {
                        let mut c = prefix.clone();
                        c.push((f, j));
                        c
                    })
                })
                .collect();
        }
        for combination in combinations {
            let label = combination
                .iter()
                .map(|&(f, j)| format!("{}[{}]", factors[f].name, factors[f].values[j]))
                .collect::<Vec<_>>()
                .join(":");
            mm.columns.push(ColumnInfo {
                source: ColumnSource::Term { term: t },
                label,
            });
            for (row, cell) in mm.data.iter_mut().zip(cells) {
                row.push(
                    combination
                        .iter()
                        .map(|&(f, j)| code(cell.factor_levels[f], j, factors[f].level_count()))
                        .product(),
                );
            }
        }
    }
    if let Some(unit) = include_unit {
        let mut identities = std::collections::HashMap::new();
        let mut unit_indices = Vec::new();
        for cell in cells {
            let key: Vec<usize> = unit
                .key
                .iter()
                .map(|name| {
                    if let Some(f) = factors.iter().position(|f| &f.name == name) {
                        cell.factor_levels[f]
                    } else {
                        cell.replicate_levels[roles
                            .iter()
                            .position(|r| &r.name == name)
                            .expect("validated unit key")]
                    }
                })
                .collect();
            let next = identities.len();
            unit_indices.push(*identities.entry(key).or_insert(next));
        }
        for j in 0..identities.len().saturating_sub(1) {
            mm.columns.push(ColumnInfo {
                source: ColumnSource::Unit,
                label: format!("unit[u{:03}]", j + 1),
            });
            for (row, &index) in mm.data.iter_mut().zip(&unit_indices) {
                row.push(code(index, j, identities.len()));
            }
        }
    }
    mm
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Estimability {
    pub n_rows: usize,
    pub n_params: usize,
    pub rank: usize,
    pub residual_df: usize,
    pub terms: Vec<TermEstimability>,
    pub blocks_estimable: bool,
}
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TermEstimability {
    pub term: Term,
    pub df: usize,
    pub estimable: bool,
    pub aliased_with: Vec<String>,
}
/// The sidecar uses the shared estimability result without recomputing statistics.
pub type EstimabilityReport = Estimability;

struct Decomposition {
    singular: Vec<f64>,
    u: DMatrix<f64>,
    vt: DMatrix<f64>,
    rank: usize,
    tolerance: f64,
}

fn decompose(x: &[Vec<f64>], p: usize) -> Decomposition {
    assert!(
        x.iter()
            .all(|r| r.len() == p && r.iter().all(|v| v.is_finite())),
        "expected a finite rectangular matrix"
    );
    if p == 0 || x.is_empty() {
        return Decomposition {
            singular: vec![0.0; p],
            u: DMatrix::zeros(x.len().max(p), p),
            vt: DMatrix::identity(p, p),
            rank: 0,
            tolerance: 0.0,
        };
    }
    // Padding wide matrices with zero rows supplies the complete right null space.
    let a = DMatrix::from_fn(
        x.len().max(p),
        p,
        |i, j| if i < x.len() { x[i][j] } else { 0.0 },
    );
    let svd = a.svd(true, true);
    let singular = svd.singular_values.as_slice().to_vec();
    let tol = 1e-10 * singular.iter().copied().fold(0.0, f64::max) * x.len().max(p) as f64;
    let rank = singular.iter().filter(|&&s| s > tol).count();
    Decomposition {
        singular,
        u: svd.u.expect("requested U"),
        vt: svd.v_t.expect("requested V transpose"),
        rank,
        tolerance: tol,
    }
}

pub fn estimability(mm: &ModelMatrix, model: &Model, factors: &[Factor]) -> Estimability {
    let p = mm.columns.len();
    let d = decompose(&mm.data, p);
    let independent = |indices: &[usize]| {
        let reduced: Vec<Vec<f64>> = mm
            .data
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter(|(j, _)| !indices.contains(j))
                    .map(|(_, &v)| v)
                    .collect()
            })
            .collect();
        d.rank
            .checked_sub(decompose(&reduced, p - indices.len()).rank)
            == Some(indices.len())
    };
    let null_basis = pivoted_null_basis(&mm.data, p, d.tolerance);
    let column_sources: Vec<_> = mm.columns.iter().map(alias_source).collect();
    let source_labels: Vec<_> = mm
        .columns
        .iter()
        .map(|column| alias_source_label(column, model, factors))
        .collect();
    let mut aliases = vec![Vec::new(); model.terms.len()];
    for null_vector in null_basis {
        let mut sources = Vec::new();
        for (column, &coefficient) in null_vector.iter().enumerate() {
            if coefficient.abs() > 1e-8 && !sources.contains(&column_sources[column]) {
                sources.push(column_sources[column]);
            }
        }
        for &source in &sources {
            let AliasSource::Term { term } = source else {
                continue;
            };
            for &other in &sources {
                if other == source {
                    continue;
                }
                let label = &source_labels[mm
                    .columns
                    .iter()
                    .position(|column| alias_source(column) == other)
                    .expect("every alias source has a column")];
                if !aliases[term].contains(label) {
                    aliases[term].push(label.clone());
                }
            }
        }
    }
    let mut terms = Vec::new();
    for (t, term) in model.terms.iter().enumerate() {
        let indices: Vec<_> = mm
            .columns
            .iter()
            .enumerate()
            .filter(|(_, c)| c.source == (ColumnSource::Term { term: t }))
            .map(|(j, _)| j)
            .collect();
        let estimable = independent(&indices);
        terms.push(TermEstimability {
            term: term.clone(),
            df: indices.len(),
            estimable,
            aliased_with: if !estimable {
                aliases[t].clone()
            } else {
                Vec::new()
            },
        });
    }
    let blocks: Vec<_> = mm
        .columns
        .iter()
        .enumerate()
        .filter(|(_, c)| matches!(c.source, ColumnSource::Block { .. }))
        .map(|(j, _)| j)
        .collect();
    Estimability {
        n_rows: mm.data.len(),
        n_params: p,
        rank: d.rank,
        residual_df: mm.data.len() - d.rank,
        terms,
        blocks_estimable: independent(&blocks),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AliasSource {
    Intercept,
    Block { role: usize },
    Term { term: usize },
    Unit,
}

fn alias_source(column: &ColumnInfo) -> AliasSource {
    match column.source {
        ColumnSource::Intercept => AliasSource::Intercept,
        ColumnSource::Block { role } => AliasSource::Block { role },
        ColumnSource::Term { term } => AliasSource::Term { term },
        ColumnSource::Unit => AliasSource::Unit,
    }
}

fn alias_source_label(column: &ColumnInfo, model: &Model, factors: &[Factor]) -> String {
    match column.source {
        ColumnSource::Intercept => "(Intercept)".into(),
        ColumnSource::Block { .. } => column
            .label
            .split_once('[')
            .map_or_else(|| column.label.clone(), |(role, _)| role.into()),
        ColumnSource::Term { term } => model.terms[term].label(factors),
        ColumnSource::Unit => "unit".into(),
    }
}

/// RREF supplies one null vector per non-pivot column, avoiding mixed SVD bases.
fn pivoted_null_basis(x: &[Vec<f64>], p: usize, tolerance: f64) -> Vec<Vec<f64>> {
    let mut rref = x.to_vec();
    let mut pivot_columns = Vec::new();
    let mut pivot_row = 0;
    for column in 0..p {
        let Some((row, magnitude)) = (pivot_row..rref.len())
            .map(|row| (row, rref[row][column].abs()))
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
        else {
            break;
        };
        if magnitude <= tolerance {
            continue;
        }
        rref.swap(pivot_row, row);
        let pivot = rref[pivot_row][column];
        for value in &mut rref[pivot_row][column..] {
            *value /= pivot;
        }
        let pivot_values = rref[pivot_row][column + 1..].to_vec();
        for (row, rref_row) in rref.iter_mut().enumerate() {
            if row == pivot_row {
                continue;
            }
            let scale = rref_row[column];
            rref_row[column] = 0.0;
            for (entry, pivot_value) in rref_row[column + 1..].iter_mut().zip(&pivot_values) {
                *entry -= scale * pivot_value;
            }
        }
        pivot_columns.push(column);
        pivot_row += 1;
        if pivot_row == rref.len() {
            break;
        }
    }
    let mut pivots = vec![false; p];
    for &column in &pivot_columns {
        pivots[column] = true;
    }
    (0..p)
        .filter(|&column| !pivots[column])
        .map(|free_column| {
            let mut vector = vec![0.0; p];
            vector[free_column] = 1.0;
            for (row, &pivot_column) in pivot_columns.iter().enumerate() {
                vector[pivot_column] = -rref[row][free_column];
            }
            vector
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct Fit {
    pub coef: Vec<f64>,
    pub rank: usize,
    pub rss: f64,
    pub residual_df: usize,
    pub fitted: Vec<f64>,
    pub residuals: Vec<f64>,
    pub estimable_coef: Vec<bool>,
    pub cov_unscaled: Option<Vec<Vec<f64>>>,
}

fn row_space_residual_norm(d: &Decomposition, v: &[f64], rank: usize) -> f64 {
    (rank..v.len())
        .map(|k| {
            v.iter()
                .enumerate()
                .map(|(j, v)| d.vt[(k, j)] * v)
                .sum::<f64>()
                .powi(2)
        })
        .sum::<f64>()
        .sqrt()
}

fn in_numerical_row_space(d: &Decomposition, v: &[f64]) -> bool {
    row_space_residual_norm(d, v, d.rank) <= 1e-8 * v_norm(v).max(1.0)
}

fn projector_rank(d: &Decomposition) -> usize {
    let tolerance = f64::EPSILON
        * d.singular.iter().copied().fold(0.0, f64::max)
        * d.u.nrows().max(d.vt.ncols()) as f64;
    d.singular.iter().filter(|&&s| s > tolerance).count()
}

fn v_norm(v: &[f64]) -> f64 {
    v.iter().map(|value| value * value).sum::<f64>().sqrt()
}

pub fn fit_least_squares(x: &[Vec<f64>], y: &[f64]) -> Fit {
    assert_eq!(x.len(), y.len(), "matrix and response lengths differ");
    assert!(y.iter().all(|v| v.is_finite()), "expected finite responses");
    let p = x.first().map_or(0, Vec::len);
    let d = decompose(x, p);
    let mut coef = vec![0.0; p];
    let mut covariance = vec![vec![0.0; p]; p];
    for k in 0..d.rank {
        let scale = y
            .iter()
            .enumerate()
            .map(|(i, y)| d.u[(i, k)] * y)
            .sum::<f64>()
            / d.singular[k];
        for (j, coef) in coef.iter_mut().enumerate() {
            *coef += d.vt[(k, j)] * scale;
            for (l, entry) in covariance[j].iter_mut().enumerate() {
                *entry += d.vt[(k, j)] * d.vt[(k, l)] / d.singular[k].powi(2);
            }
        }
    }
    let fitted: Vec<f64> = x
        .iter()
        .map(|row| row.iter().zip(&coef).map(|(x, b)| x * b).sum())
        .collect();
    let residuals: Vec<f64> = y
        .iter()
        .zip(&fitted)
        .map(|(y, fitted)| y - fitted)
        .collect();
    let estimable_coef = (0..p)
        .map(|j| {
            let mut e = vec![0.0; p];
            e[j] = 1.0;
            in_numerical_row_space(&d, &e)
        })
        .collect();
    Fit {
        coef,
        rank: d.rank,
        rss: residuals.iter().map(|r| r * r).sum(),
        residual_df: x.len() - d.rank,
        fitted,
        residuals,
        estimable_coef,
        cov_unscaled: Some(covariance),
    }
}

pub fn is_estimable(x: &[Vec<f64>], v: &[f64]) -> bool {
    if !v.iter().all(|value| value.is_finite())
        || x.iter()
            .any(|row| row.len() != v.len() || row.iter().any(|value| !value.is_finite()))
    {
        return false;
    }
    let d = decompose(x, v.len());
    row_space_residual_norm(&d, v, projector_rank(&d)) <= 1e-8 * v_norm(v).max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn factors(levels: &[usize]) -> Vec<Factor> {
        levels
            .iter()
            .enumerate()
            .map(|(i, l)| {
                Factor::parse(
                    &((b'a' + i as u8) as char).to_string(),
                    &format!("d[1-{}]", l),
                )
                .unwrap()
            })
            .collect()
    }
    fn grid(levels: &[usize]) -> Vec<Cell> {
        let mut rows = vec![vec![]];
        for &l in levels {
            rows = rows
                .into_iter()
                .flat_map(|r| {
                    (0..l).map(move |j| {
                        let mut r = r.clone();
                        r.push(j);
                        r
                    })
                })
                .collect();
        }
        rows.into_iter()
            .map(|factor_levels| Cell {
                factor_levels,
                replicate_levels: vec![],
            })
            .collect()
    }
    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} != {b}");
    }

    #[test]
    fn formula_grammar_and_canonicalization() {
        let fs = factors(&[2, 2, 2]);
        for (src, expected) in [
            (
                "response ~ b + a + c + b:a + a",
                "response ~ a + b + c + a:b",
            ),
            ("(a+b)*c", ". ~ a + b + c + a:c + b:c"),
            ("a*b*c", ". ~ a + b + c + a:b + a:c + b:c + a:b:c"),
            (". ~ b:a", ". ~ a:b"),
            ("a+b*c", ". ~ a + b + c + b:c"),
            ("a*b:c", ". ~ a + b:c + a:b:c"),
            ("(a+b):c", ". ~ a:c + b:c"),
            ("((a))", ". ~ a"),
        ] {
            let m = Model::parse(src, &fs).unwrap();
            assert_eq!(m.formula, expected);
            assert_eq!(Model::parse(&m.formula, &fs).unwrap(), m);
        }
        assert_eq!(Model::parse("a*b*c", &fs).unwrap().terms.len(), 7);
        let named = [Factor::parse("level-1", "d[1,2]").unwrap()];
        assert_eq!(
            Model::parse("response-1 ~ level-1", &named)
                .unwrap()
                .formula,
            "response-1 ~ level-1"
        );
        assert_eq!(Model::main_effects(&fs).formula, ". ~ a + b + c");
        assert_eq!(Term(vec![0, 2]).label(&fs), "a:c");
    }

    #[test]
    fn formula_errors() {
        let fs = factors(&[2, 2, 2]);
        for src in [
            "a:a", "a:(a+b)", "", "a+", "a b", "(a+b", "a)", "a**b", "~a", "a~b~c",
        ] {
            assert!(Model::parse(src, &fs).is_err(), "accepted {src}");
        }
        for src in ["a -1", "a-1", "a^2", "a %in% b"] {
            let error = Model::parse(src, &fs).unwrap_err().to_string();
            assert!(error.contains("unsupported"), "{error}");
        }
        assert!(
            Model::parse("a + absent", &fs)
                .unwrap_err()
                .to_string()
                .contains("absent")
        );
    }

    #[test]
    fn response_selection_and_hierarchy() {
        let fs = factors(&[2, 2]);
        let models = vec![
            Model::parse("a", &fs).unwrap(),
            Model::parse("y ~ a:b", &fs).unwrap(),
        ];
        assert_eq!(Model::for_result(&models, "y"), Some(&models[1]));
        assert_eq!(Model::for_result(&models, "z"), Some(&models[0]));
        assert!(Model::for_result(&models[1..], "z").is_none());
        assert_eq!(
            hierarchy_warnings(&models[1], &fs),
            ["a:b requested without a", "a:b requested without b"]
        );
        assert!(hierarchy_warnings(&Model::parse("a*b", &fs).unwrap(), &fs).is_empty());
    }

    #[test]
    fn deviation_coding_full_factorial() {
        let fs = factors(&[2, 3]);
        let m = Model::parse("a*b", &fs).unwrap();
        let mm = model_matrix(&fs, &[], &m, &grid(&[2, 3]), None);
        assert_eq!(
            mm.columns
                .iter()
                .map(|c| c.label.as_str())
                .collect::<Vec<_>>(),
            [
                "(Intercept)",
                "a[1]",
                "b[1]",
                "b[2]",
                "a[1]:b[1]",
                "a[1]:b[2]"
            ]
        );
        assert_eq!(
            mm.data,
            vec![
                vec![1., 1., 1., 0., 1., 0.],
                vec![1., 1., 0., 1., 0., 1.],
                vec![1., 1., -1., -1., -1., -1.],
                vec![1., -1., 1., 0., -1., 0.],
                vec![1., -1., 0., 1., 0., -1.],
                vec![1., -1., -1., -1., 1., 1.],
            ]
        );
        for j in 1..6 {
            close(mm.data.iter().map(|r| r[j]).sum(), 0.);
        }
        for j in 0..6 {
            for k in j + 1..6 {
                let dot = mm.data.iter().map(|r| r[j] * r[k]).sum();
                if mm.columns[j].source != mm.columns[k].source {
                    close(dot, 0.);
                } else {
                    close(dot, 2.);
                } // Deviation columns within a three-level term are correlated.
            }
        }
        assert_eq!(estimability(&mm, &m, &fs).rank, 6);
    }

    #[test]
    fn block_and_unit_order_and_pairing() {
        let fs = factors(&[2, 2]);
        let roles = vec![ReplicateRole {
            name: "seed".into(),
            levels: vec![FactorValue::Int(3), FactorValue::Int(9)],
        }];
        let unit = Unit {
            name: "checkpoint".into(),
            key: vec!["a".into(), "seed".into()],
        };
        let mut cells = grid(&[2, 2]);
        for c in &mut cells {
            c.replicate_levels = vec![0];
        }
        let second: Vec<_> = cells
            .iter()
            .map(|c| Cell {
                factor_levels: c.factor_levels.clone(),
                replicate_levels: vec![1],
            })
            .collect();
        cells.extend(second);
        let mm = model_matrix(&fs, &roles, &Model::main_effects(&fs), &cells, Some(&unit));
        assert_eq!(
            mm.columns
                .iter()
                .map(|c| c.label.as_str())
                .collect::<Vec<_>>(),
            [
                "(Intercept)",
                "seed[3]",
                "a[1]",
                "b[1]",
                "unit[u001]",
                "unit[u002]",
                "unit[u003]"
            ]
        );
        assert_eq!(&mm.data[0][4..], &[1., 0., 0.]);
        assert_eq!(&mm.data[1][4..], &mm.data[0][4..]);
        assert_eq!(&mm.data[2][4..], &[0., 1., 0.]);
        assert_eq!(&mm.data[4][4..], &[0., 0., 1.]);
        assert_eq!(&mm.data[6][4..], &[-1., -1., -1.]);
        close(mm.data.iter().map(|r| r[1]).sum(), 0.);
    }

    #[test]
    fn l4_alias_and_full_factorial_replication() {
        let fs = factors(&[2, 2, 2]);
        let m = Model::parse("a+b+c+a:b", &fs).unwrap();
        let cells: Vec<_> = [vec![0, 0, 0], vec![0, 1, 1], vec![1, 0, 1], vec![1, 1, 0]]
            .into_iter()
            .map(|factor_levels| Cell {
                factor_levels,
                replicate_levels: vec![],
            })
            .collect();
        let mm = model_matrix(&fs, &[], &m, &cells, None);
        let e = estimability(&mm, &m, &fs);
        assert_eq!((e.n_rows, e.n_params, e.rank, e.residual_df), (4, 5, 4, 0));
        assert!(e.terms[0].estimable && e.terms[1].estimable);
        assert!(!e.terms[3].estimable);
        assert_eq!(e.terms[3].df, 1);
        assert_eq!(e.terms[3].aliased_with, ["c"]);
        assert_eq!(e.terms[2].aliased_with, ["a:b"]);
        let m = Model::parse("a+b+c+a:b+a:c+b:c", &fs).unwrap();
        let aliases = estimability(&model_matrix(&fs, &[], &m, &cells, None), &m, &fs);
        assert_eq!(aliases.terms[0].aliased_with, ["b:c"]);
        assert_eq!(aliases.terms[1].aliased_with, ["a:c"]);
        assert_eq!(aliases.terms[2].aliased_with, ["a:b"]);
        assert_eq!(aliases.terms[3].aliased_with, ["c"]);
        assert_eq!(aliases.terms[4].aliased_with, ["b"]);
        assert_eq!(aliases.terms[5].aliased_with, ["a"]);
        let m = Model::parse("a*b*c", &fs).unwrap();
        assert_eq!(
            estimability(&model_matrix(&fs, &[], &m, &cells, None), &m, &fs).terms[6].aliased_with,
            ["(Intercept)"]
        );
        let mut cells = grid(&[2, 2, 2]);
        let full = estimability(&model_matrix(&fs, &[], &m, &cells, None), &m, &fs);
        assert_eq!((full.rank, full.residual_df), (8, 0));
        assert!(
            full.terms
                .iter()
                .all(|t| t.estimable && t.aliased_with.is_empty())
        );
        let roles = vec![ReplicateRole {
            name: "seed".into(),
            levels: vec![FactorValue::Int(1), FactorValue::Int(2)],
        }];
        for c in &mut cells {
            c.replicate_levels = vec![0];
        }
        cells.extend(cells.clone().into_iter().map(|mut c| {
            c.replicate_levels = vec![1];
            c
        }));
        let replicated = estimability(&model_matrix(&fs, &roles, &m, &cells, None), &m, &fs);
        assert_eq!((replicated.rank, replicated.residual_df), (9, 7));
        assert!(replicated.blocks_estimable);
        assert!(replicated.terms.iter().all(|t| t.estimable));
    }

    #[test]
    fn aliases_name_block_sources() {
        let fs = factors(&[2]);
        let model = Model::main_effects(&fs);
        let roles = vec![ReplicateRole {
            name: "seed".into(),
            levels: vec![FactorValue::Int(1), FactorValue::Int(2)],
        }];
        let cells = vec![
            Cell {
                factor_levels: vec![0],
                replicate_levels: vec![0],
            },
            Cell {
                factor_levels: vec![1],
                replicate_levels: vec![1],
            },
        ];
        let report = estimability(
            &model_matrix(&fs, &roles, &model, &cells, None),
            &model,
            &fs,
        );
        assert!(!report.terms[0].estimable);
        assert_eq!(report.terms[0].aliased_with, ["seed"]);

        let unit = Unit {
            name: "checkpoint".into(),
            key: vec!["a".into()],
        };
        let report = estimability(
            &model_matrix(&fs, &[], &model, &cells, Some(&unit)),
            &model,
            &fs,
        );
        assert!(!report.terms[0].estimable);
        assert_eq!(report.terms[0].aliased_with, ["unit"]);
    }

    #[test]
    fn least_squares_full_rank_matches_normal_equations() {
        let x = vec![
            vec![1., 0., 2.],
            vec![1., 1., 0.],
            vec![1., 2., 1.],
            vec![1., 4., 3.],
            vec![1., -1., 2.],
        ];
        let y = vec![1., 3., 2., 7., -1.];
        let fit = fit_least_squares(&x, &y);
        let a = DMatrix::from_fn(x.len(), 3, |i, j| x[i][j]);
        let inverse = (a.transpose() * &a).try_inverse().unwrap();
        let expected = &inverse * a.transpose() * nalgebra::DVector::from_vec(y.clone());
        assert_eq!((fit.rank, fit.residual_df), (3, 2));
        assert_eq!(fit.estimable_coef, [true, true, true]);
        for j in 0..3 {
            close(fit.coef[j], expected[j]);
            for k in 0..3 {
                close(fit.cov_unscaled.as_ref().unwrap()[j][k], inverse[(j, k)]);
            }
        }
        for (i, row) in x.iter().enumerate() {
            close(
                fit.fitted[i],
                row.iter().zip(&fit.coef).map(|(x, b)| x * b).sum(),
            );
            close(fit.residuals[i], y[i] - fit.fitted[i]);
        }
        close(fit.rss, fit.residuals.iter().map(|r| r * r).sum());
    }

    #[test]
    fn minimum_norm_covariance_and_row_space() {
        let x = vec![vec![1., 0., 0.], vec![1., 1., 1.], vec![1., 2., 2.]];
        let fit = fit_least_squares(&x, &[1., 3., 5.]);
        assert_eq!((fit.rank, fit.residual_df), (2, 1));
        for coef in &fit.coef {
            close(*coef, 1.);
        }
        close(fit.rss, 0.);
        assert_eq!(fit.estimable_coef, [true, false, false]);
        for row in &x {
            assert!(is_estimable(&x, row));
        }
        assert!(!is_estimable(&x, &[0., 1., -1.]));
        assert!(is_estimable(&x, &[0., 0., 0.]));
        let cov = fit.cov_unscaled.unwrap();
        let c = DMatrix::from_fn(3, 3, |i, j| cov[i][j]);
        let a = DMatrix::from_fn(3, 3, |i, j| x[i][j]);
        let gram = a.transpose() * a;
        let gc = &gram * &c;
        let cg = &c * &gram;
        assert!((&gc * &gram - &gram).norm() < 1e-9);
        assert!((&cg * &c - &c).norm() < 1e-9);
        assert!((&gc - gc.transpose()).norm() < 1e-9);
        assert!((&cg - cg.transpose()).norm() < 1e-9);
        let wide = vec![vec![1., 2., 3.]];
        let fit = fit_least_squares(&wide, &[14.]);
        assert_eq!(fit.rank, 1);
        for (a, b) in fit.coef.iter().zip([1., 2., 3.]) {
            close(*a, b);
        }
        assert_eq!(fit.estimable_coef, [false, false, false]);
        assert!(is_estimable(&wide, &wide[0]));
        assert!(!is_estimable(&wide, &[0., 0., 1.]));
    }

    #[test]
    fn rank_tolerance_empty_and_zero_data() {
        for scale in [1e-12, 1., 1e12] {
            let x = vec![vec![scale, 0.], vec![0., scale * 1e-11]];
            assert_eq!(fit_least_squares(&x, &[0., 0.]).rank, 1);
            let x = vec![vec![scale, 0.], vec![0., scale * 1e-8]];
            assert_eq!(fit_least_squares(&x, &[0., 0.]).rank, 2);
        }
        let near_null = vec![vec![1., 0.], vec![0., 1e-11]];
        assert!(is_estimable(&near_null, &near_null[1]));
        assert!(is_estimable(&near_null, &[0., 1.]));
        let scaled = vec![vec![1e9, 0.], vec![0., 0.1]];
        assert!(is_estimable(&scaled, &[0.9, 0.1]));
        assert!(!is_estimable(&scaled, &[0., 0., 0.]));
        let rank_one = vec![vec![1., 1.], vec![2., 2.]];
        assert!(!is_estimable(&rank_one, &[1., -1.]));
        let zero = vec![vec![0., 0.]; 3];
        let fit = fit_least_squares(&zero, &[1., 2., 3.]);
        assert_eq!((fit.rank, fit.residual_df), (0, 3));
        assert_eq!(fit.estimable_coef, [false, false]);
        close(fit.rss, 14.);
        assert!(is_estimable(&[], &[0., 0.]));
        assert!(!is_estimable(&[], &[1., 0.]));
        assert_eq!(fit_least_squares(&[], &[]).rank, 0);
        let fs = factors(&[2]);
        let m = Model::main_effects(&fs);
        let e = estimability(&model_matrix(&fs, &[], &m, &[], None), &m, &fs);
        assert_eq!((e.n_params, e.rank, e.residual_df), (2, 0, 0));
        assert!(!e.terms[0].estimable);
    }
}
