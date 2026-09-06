use anyhow::{anyhow, bail, Context, Result};
use statrs::distribution::{ContinuousCDF, FisherSnedecor, StudentsT};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs::File;
use std::path::PathBuf;

use crate::construct::{sidecar_path, Design};
use crate::factor::FactorValue;
use crate::model::{self, Cell, ColumnSource, Fit, Model, ModelMatrix};
use crate::report::*;
use crate::rng::Rng;

#[derive(Debug)]
pub struct AnalyzeArgs {
    pub csv_path: PathBuf,
    pub design_path: Option<PathBuf>,
    pub maximize: Vec<String>,
    pub minimize: Vec<String>,
    pub tolerate_noise: Option<f64>,
    pub alpha: f64,
    pub pool: Vec<String>,
    pub pool_auto: bool,
    pub bootstrap: Option<usize>,
    pub seed: Option<u64>,
    pub json: Option<PathBuf>,
}

pub fn run_analyze(args: AnalyzeArgs) -> Result<()> {
    let report = analyze(&args)?;
    for result in &report.results {
        print!("{}", render_result(result, args.alpha));
    }
    if let Some(path) = &args.json {
        serde_json::to_writer_pretty(File::create(path)?, &report)?;
    }
    if report.results.iter().all(|r| !has_evidence(r)) {
        std::process::exit(2);
    }
    Ok(())
}

pub fn analyze(args: &AnalyzeArgs) -> Result<Report> {
    if !args.alpha.is_finite() || !(0.0..1.0).contains(&args.alpha) || args.alpha == 0.0 {
        bail!("alpha must be between 0 and 1");
    }
    if args
        .tolerate_noise
        .is_some_and(|s| !s.is_finite() || s < 0.0)
    {
        bail!("tolerate-noise must be finite and nonnegative");
    }
    let path = args
        .design_path
        .clone()
        .unwrap_or_else(|| sidecar_path(&args.csv_path));
    let design: Design = serde_json::from_reader(
        File::open(&path).with_context(|| format!("opening {}", path.display()))?,
    )?;
    for name in args.maximize.iter().chain(&args.minimize) {
        if !design.results.contains(name) {
            bail!("unknown result '{name}'");
        }
    }
    let mut reader = csv::Reader::from_path(&args.csv_path)?;
    let headers = reader.headers()?.clone();
    let index = |name: &str| {
        headers
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| anyhow!("column '{name}' not in CSV"))
    };
    let factor_indices = design
        .factors
        .iter()
        .map(|f| index(&f.name))
        .collect::<Result<Vec<_>>>()?;
    let role_indices = design
        .replicates
        .iter()
        .map(|r| index(&r.name))
        .collect::<Result<Vec<_>>>()?;
    let result_indices = design
        .results
        .iter()
        .map(|r| index(r))
        .collect::<Result<Vec<_>>>()?;
    let id_index = headers.iter().position(|h| h == "run_id");
    let mut seen = BTreeSet::new();
    let mut data = vec![Vec::new(); design.results.len()];
    let mut missing = vec![Vec::new(); design.results.len()];
    for (i, record) in reader.records().enumerate() {
        let record = record?;
        let row = i + 2;
        let id = id_index.map_or_else(|| row.to_string(), |j| record[j].trim().to_owned());
        if id.is_empty() || !seen.insert(id.clone()) {
            bail!("row {row}: empty or duplicate run_id '{id}'");
        }
        let resolve = |values: &[FactorValue], j: usize, name: &str| {
            values
                .iter()
                .position(|v| value_matches(v, record[j].trim()))
                .ok_or_else(|| {
                    anyhow!(
                        "row {row}: column '{name}' value '{}' not in design levels",
                        &record[j]
                    )
                })
        };
        let cell = Cell {
            factor_levels: design
                .factors
                .iter()
                .zip(&factor_indices)
                .map(|(f, &j)| resolve(&f.values, j, &f.name))
                .collect::<Result<_>>()?,
            replicate_levels: design
                .replicates
                .iter()
                .zip(&role_indices)
                .map(|(r, &j)| resolve(&r.levels, j, &r.name))
                .collect::<Result<_>>()?,
        };
        for (ri, &j) in result_indices.iter().enumerate() {
            let raw = record[j].trim();
            if raw.is_empty() {
                missing[ri].push(id.clone());
                continue;
            }
            let y = raw
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite())
                .ok_or_else(|| {
                    anyhow!(
                        "row {row}: column '{}' must contain a finite number, got '{raw}'",
                        design.results[ri]
                    )
                })?;
            data[ri].push((cell.clone(), y));
        }
    }
    if id_index.is_some() {
        for run in &design.runs {
            if !seen.contains(&run.run_id) {
                for list in &mut missing {
                    list.push(run.run_id.clone());
                }
            }
        }
    }
    let results = design
        .results
        .iter()
        .enumerate()
        .map(|(i, name)| analyze_result(&design, name, &data[i], missing[i].clone(), args))
        .collect::<Result<_>>()?;
    Ok(Report {
        version: 2,
        design: path.display().to_string(),
        results,
    })
}

fn value_matches(v: &FactorValue, cell: &str) -> bool {
    if v.to_string() == cell {
        return true;
    }
    match v {
        FactorValue::Text(s) => s == cell,
        FactorValue::Int(n) => {
            cell.parse::<i64>().is_ok_and(|c| c == *n)
                || cell
                    .parse::<f64>()
                    .is_ok_and(|c| (c - *n as f64).abs() < 1e-9)
        }
        FactorValue::Float(x) => cell
            .parse::<f64>()
            .is_ok_and(|c| (c - x).abs() < 1e-4 * x.abs().max(1.0)),
    }
}

fn subset(mm: &ModelMatrix, keep: impl Fn(&ColumnSource) -> bool) -> Vec<Vec<f64>> {
    mm.data
        .iter()
        .map(|row| {
            row.iter()
                .zip(&mm.columns)
                .filter(|(_, c)| keep(&c.source))
                .map(|(&v, _)| v)
                .collect()
        })
        .collect()
}
fn ms(ss: f64, df: usize) -> Option<f64> {
    (df > 0).then(|| ss / df as f64)
}
fn dot(x: &[f64], b: &[f64]) -> f64 {
    x.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn se(fit: &Fit, row: &[f64], variance: Option<f64>) -> Option<f64> {
    let cov = fit.cov_unscaled.as_ref()?;
    Some(
        (variance?
            * row
                .iter()
                .enumerate()
                .map(|(i, x)| x * dot(&cov[i], row))
                .sum::<f64>())
        .max(0.0)
        .sqrt(),
    )
}
fn critical(df: usize, alpha: f64) -> Option<f64> {
    (df > 0).then(|| {
        StudentsT::new(0.0, 1.0, df as f64)
            .expect("positive df")
            .inverse_cdf(1.0 - alpha / 2.0)
    })
}
fn stratum_index(design: &Design, model: &Model, term: usize) -> usize {
    usize::from(design.unit.as_ref().is_some_and(|u| {
        !model.terms[term]
            .0
            .iter()
            .all(|&f| u.key.contains(&design.factors[f].name))
    }))
}
fn effect_range(term: &model::Term, data: &[(Cell, f64)]) -> f64 {
    let mut means: BTreeMap<Vec<usize>, (f64, usize)> = BTreeMap::new();
    for (cell, y) in data {
        let entry = means
            .entry(term.0.iter().map(|&f| cell.factor_levels[f]).collect())
            .or_default();
        entry.0 += y;
        entry.1 += 1;
    }
    let values: Vec<_> = means.values().map(|&(s, n)| s / n as f64).collect();
    if values.is_empty() {
        0.0
    } else {
        values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - values.iter().copied().fold(f64::INFINITY, f64::min)
    }
}

fn analyze_result(
    design: &Design,
    name: &str,
    data: &[(Cell, f64)],
    missing_runs: Vec<String>,
    args: &AnalyzeArgs,
) -> Result<ResultReport> {
    let model = Model::for_result(&design.models, name)
        .cloned()
        .unwrap_or_else(|| Model::main_effects(&design.factors));
    let cells: Vec<_> = data.iter().map(|(c, _)| c.clone()).collect();
    let y: Vec<_> = data.iter().map(|(_, y)| *y).collect();
    let mm = model::model_matrix(&design.factors, &design.replicates, &model, &cells, None);
    let est = model::estimability(&mm, &model, &design.factors);
    let mut warnings = model::hierarchy_warnings(&model, &design.factors);
    for term in &est.terms {
        if !term.estimable
            && design
                .estimability
                .as_ref()
                .is_some_and(|e| e.terms.iter().any(|t| t.term == term.term && t.estimable))
        {
            warnings.push(format!(
                "{}: LOST (missing runs)",
                term.term.label(&design.factors)
            ));
        }
    }
    let original_fit = model::fit_least_squares(&mm.data, &y);
    let mut strata = if let Some(unit) = &design.unit {
        let unit_mm = model::model_matrix(
            &design.factors,
            &design.replicates,
            &model,
            &cells,
            Some(unit),
        );
        let whole = |s: &ColumnSource| match s {
            ColumnSource::Term { term } => stratum_index(design, &model, *term) == 0,
            ColumnSource::Unit => false,
            _ => true,
        };
        let base = model::fit_least_squares(&subset(&unit_mm, whole), &y);
        let with_unit = model::fit_least_squares(
            &subset(&unit_mm, |s| *s == ColumnSource::Unit || whole(s)),
            &y,
        );
        let full_unit = model::fit_least_squares(&unit_mm.data, &y);
        let df = with_unit.rank.saturating_sub(base.rank);
        vec![
            Stratum {
                name: "between-unit".into(),
                df,
                ms: ms((base.rss - with_unit.rss).max(0.0), df),
            },
            Stratum {
                name: "within-unit".into(),
                df: full_unit.residual_df,
                ms: ms(full_unit.rss, full_unit.residual_df),
            },
        ]
    } else {
        vec![Stratum {
            name: "residual".into(),
            df: original_fit.residual_df,
            ms: ms(original_fit.rss, original_fit.residual_df),
        }]
    };
    let mut anova = Vec::new();
    for (t, term) in model.terms.iter().enumerate() {
        let keep = |s: &ColumnSource| match s {
            ColumnSource::Term { term: other } => {
                !term.0.iter().all(|f| model.terms[*other].0.contains(f))
            }
            _ => true,
        };
        let reduced = model::fit_least_squares(&subset(&mm, keep), &y);
        let added = model::fit_least_squares(
            &subset(&mm, |s| *s == ColumnSource::Term { term: t } || keep(s)),
            &y,
        );
        let df = added.rank.saturating_sub(reduced.rank);
        let ss = (reduced.rss - added.rss).max(0.0);
        anova.push(Anova {
            term: term.label(&design.factors),
            stratum: if design.unit.is_none() {
                "residual"
            } else if stratum_index(design, &model, t) == 0 {
                "whole-plot"
            } else {
                "sub-plot"
            }
            .into(),
            df,
            ss,
            ms: ms(ss, df),
            f: None,
            p: None,
            significant: false,
            pooled: false,
        });
    }
    for label in &args.pool {
        let Some(term) = anova.iter_mut().find(|a| &a.term == label) else {
            bail!("pool term '{label}' not in model for '{name}'");
        };
        term.pooled = true;
    }
    if args.pool_auto {
        for (s, stratum) in strata.iter().enumerate() {
            if stratum.df != 0 {
                continue;
            }
            let mut candidates: Vec<_> = (0..anova.len())
                .filter(|&t| stratum_index(design, &model, t) == s)
                .collect();
            candidates.sort_by(|&a, &b| anova[a].ss.total_cmp(&anova[b].ss));
            for &t in candidates.iter().take(candidates.len().div_ceil(2)) {
                anova[t].pooled = true;
            }
        }
    }
    for (t, a) in anova.iter().enumerate().filter(|(_, a)| a.pooled) {
        let s = &mut strata[stratum_index(design, &model, t)];
        let ss = s.ms.unwrap_or(0.0) * s.df as f64 + a.ss;
        s.df += a.df;
        s.ms = ms(ss, s.df);
    }
    for s in &strata {
        if s.df == 0 {
            warnings.push(format!("insufficient evidence under this model: residual df = 0 in {}; add replicates, or pass --pool <terms> / --pool-auto.",s.name));
        }
    }
    for (t, a) in anova.iter_mut().enumerate() {
        if a.pooled {
            continue;
        }
        let s = &strata[stratum_index(design, &model, t)];
        if let (Some(numerator), Some(denominator)) = (a.ms, s.ms) {
            if denominator > 0.0 {
                a.f = Some(numerator / denominator);
                a.p = Some(FisherSnedecor::new(a.df as f64, s.df as f64)?.sf(a.f.unwrap()));
            } else if numerator > 0.0 {
                a.f = Some(f64::INFINITY);
                a.p = Some(0.0);
            }
        }
        a.significant = est.terms[t].estimable
            && args.tolerate_noise.map_or_else(
                || a.p.is_some_and(|p| p < args.alpha),
                |sigma| effect_range(&model.terms[t], data) > sigma,
            );
    }
    let active = |s: &ColumnSource| !matches!(s,ColumnSource::Term{term} if anova[*term].pooled);
    let x = subset(&mm, active);
    let fit = model::fit_least_squares(&x, &y);
    let variance = ms(fit.rss, fit.residual_df);
    let tcrit = critical(fit.residual_df, args.alpha);
    let columns: Vec<_> = mm.columns.iter().filter(|c| active(&c.source)).collect();
    let mut coefficients = Vec::new();
    for (j, col) in columns.iter().enumerate() {
        let estimable = fit.estimable_coef.get(j).copied().unwrap_or(false);
        let stderr = if estimable {
            variance.map(|v| {
                (v * fit.cov_unscaled.as_ref().unwrap()[j][j])
                    .max(0.0)
                    .sqrt()
            })
        } else {
            None
        };
        let estimate = estimable.then(|| fit.coef[j]);
        let width = stderr.zip(tcrit).map(|(s, t)| s * t);
        coefficients.push(Coefficient {
            label: col.label.clone(),
            estimate,
            se: stderr,
            ci_low: estimate.zip(width).map(|(b, w)| b - w),
            ci_high: estimate.zip(width).map(|(b, w)| b + w),
            boot_low: None,
            boot_high: None,
            estimable,
        });
    }
    let mut observed: BTreeMap<Vec<usize>, (f64, usize)> = BTreeMap::new();
    for (c, y) in data {
        let e = observed.entry(c.factor_levels.clone()).or_default();
        e.0 += y;
        e.1 += 1;
    }
    let prediction_row = |levels: &[usize]| {
        let cell = Cell {
            factor_levels: levels.to_vec(),
            replicate_levels: vec![0; design.replicates.len()],
        };
        let mut m = model::model_matrix(&design.factors, &design.replicates, &model, &[cell], None);
        for (j, c) in m.columns.iter().enumerate() {
            if matches!(c.source, ColumnSource::Block { .. }) {
                m.data[0][j] = 0.0;
            }
        }
        subset(&m, active).remove(0)
    };
    let capped = design
        .factors
        .iter()
        .try_fold(1usize, |n, f| n.checked_mul(f.level_count()))
        .is_none_or(|n| n > 100_000);
    if capped {
        warnings.push("prediction grid exceeds 100000 cells; exporting measured cells and recommendation cells".into());
    }
    let mut predictions = Vec::new();
    let mut prediction_rows = Vec::new();
    let mut extrema: [Option<(Prediction, Vec<f64>)>; 2] = [None, None];
    let mut levels = vec![0; design.factors.len()];
    loop {
        let row = prediction_row(&levels);
        let estimable = !data.is_empty() && model::is_estimable(&x, &row);
        let mean = dot(&row, &fit.coef);
        let observed_mean = observed.get(&levels).map(|&(s, n)| s / n as f64);
        let p = Prediction {
            factors: design
                .factors
                .iter()
                .zip(&levels)
                .map(|(f, &l)| (f.name.clone(), f.values[l].to_string()))
                .collect(),
            mean,
            se: if estimable {
                se(&fit, &row, variance)
            } else {
                None
            },
            estimable,
            measured: observed_mean.is_some(),
            observed_mean,
            boot_low: None,
            boot_high: None,
        };
        if estimable {
            for (i, best) in extrema.iter_mut().enumerate() {
                if best
                    .as_ref()
                    .is_none_or(|(b, _)| if i == 0 { mean > b.mean } else { mean < b.mean })
                {
                    *best = Some((p.clone(), row.clone()));
                }
            }
        }
        if !capped || p.measured {
            predictions.push(p);
            prediction_rows.push(row);
        }
        if !advance(&mut levels, &design.factors) {
            break;
        }
    }
    if capped {
        for (p, row) in extrema.iter().flatten() {
            if !predictions.iter().any(|v| v.factors == p.factors) {
                predictions.push(p.clone());
                prediction_rows.push(row.clone());
            }
        }
    }
    bootstrap(
        &x,
        &y,
        &cells,
        &prediction_rows,
        &mut coefficients,
        &mut predictions,
        design,
        args,
        &mut warnings,
    );
    let significant = anova.iter().any(|a| a.significant);
    let mut recommendation = Recommendation::default();
    if significant {
        for (i, best) in extrema.into_iter().enumerate() {
            if (i == 0
                && args.minimize.iter().any(|n| n == name)
                && !args.maximize.iter().any(|n| n == name))
                || (i == 1
                    && args.maximize.iter().any(|n| n == name)
                    && !args.minimize.iter().any(|n| n == name))
            {
                continue;
            }
            if let Some((p, _)) = best {
                let pred = predictions.iter().find(|v| v.factors == p.factors).unwrap();
                let width = p.se.zip(tcrit).map(|(s, t)| s * t);
                let optimum = Optimum {
                    factors: p.factors,
                    mean: p.mean,
                    ci_low: pred.boot_low.or(width.map(|w| p.mean - w)),
                    ci_high: pred.boot_high.or(width.map(|w| p.mean + w)),
                    measured: p.measured,
                };
                if i == 0 {
                    recommendation.maximize = Some(optimum);
                } else {
                    recommendation.minimize = Some(optimum);
                }
            }
        }
    }
    if !design.replicates.is_empty() {
        let base = model::fit_least_squares(&subset(&mm, |s| *s == ColumnSource::Intercept), &y);
        let blocks = model::fit_least_squares(
            &subset(&mm, |s| !matches!(s, ColumnSource::Term { .. })),
            &y,
        );
        let df = blocks.rank.saturating_sub(base.rank);
        let ss = (base.rss - blocks.rss).max(0.0);
        let (f, p) = match (ms(ss, df), strata[0].ms) {
            (Some(numerator), Some(denominator)) if denominator > 0.0 => {
                let f = numerator / denominator;
                (
                    Some(f),
                    Some(FisherSnedecor::new(df as f64, strata[0].df as f64)?.sf(f)),
                )
            }
            (Some(numerator), Some(0.0)) if numerator > 0.0 => (Some(f64::INFINITY), Some(0.0)),
            _ => (None, None),
        };
        anova.push(Anova {
            term: "(blocks)".into(),
            stratum: if design.unit.is_some() {
                "whole-plot"
            } else {
                "residual"
            }
            .into(),
            df,
            ss,
            ms: ms(ss, df),
            f,
            p,
            significant: false,
            pooled: false,
        });
    }
    let mean = if y.is_empty() {
        0.0
    } else {
        y.iter().sum::<f64>() / y.len() as f64
    };
    let total = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
    Ok(ResultReport {
        name: name.into(),
        model: model.formula,
        n_complete: y.len(),
        missing_runs,
        estimability: EstimabilityReport {
            n_params: est.n_params,
            rank: est.rank,
            residual_df: est.residual_df,
            terms: est
                .terms
                .into_iter()
                .map(|t| TermReport {
                    term: t.term.label(&design.factors),
                    df: t.df,
                    estimable: t.estimable,
                    aliased_with: t.aliased_with,
                })
                .collect(),
        },
        fit: FitReport {
            sigma: variance.map(f64::sqrt),
            rss: fit.rss,
            r_squared: (total > 0.0).then(|| 1.0 - fit.rss / total),
        },
        anova,
        strata,
        coefficients,
        predictions,
        recommendation,
        warnings,
    })
}

fn advance(levels: &mut [usize], factors: &[crate::factor::Factor]) -> bool {
    for i in (0..levels.len()).rev() {
        levels[i] += 1;
        if levels[i] < factors[i].level_count() {
            return true;
        }
        levels[i] = 0;
    }
    false
}
fn percentile(values: &mut [f64], alpha: f64) -> (Option<f64>, Option<f64>) {
    if values.is_empty() {
        return (None, None);
    }
    values.sort_by(f64::total_cmp);
    let quantile = |q: f64| {
        let p = q * (values.len() - 1) as f64;
        let i = p.floor() as usize;
        values[i] + (values[p.ceil() as usize] - values[i]) * (p - i as f64)
    };
    (
        Some(quantile(alpha / 2.0)),
        Some(quantile(1.0 - alpha / 2.0)),
    )
}
#[allow(clippy::too_many_arguments)]
fn bootstrap(
    x: &[Vec<f64>],
    y: &[f64],
    cells: &[Cell],
    rows: &[Vec<f64>],
    coefficients: &mut [Coefficient],
    predictions: &mut [Prediction],
    design: &Design,
    args: &AnalyzeArgs,
    warnings: &mut Vec<String>,
) {
    let b = args.bootstrap.unwrap_or(if design.replicates.is_empty() {
        0
    } else {
        1000
    });
    if b == 0 {
        return;
    }
    if design.replicates.is_empty() {
        warnings.push("bootstrap requires a replicate role".into());
        return;
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, c) in cells.iter().enumerate() {
        groups.entry(c.replicate_levels[0]).or_default().push(i);
    }
    let groups: Vec<_> = groups.into_values().collect();
    if groups.len() < 5 {
        warnings.push(format!(
            "bootstrap has fewer than 5 groups ({}); intervals may be unreliable",
            groups.len()
        ));
    }
    if groups.is_empty() {
        return;
    }
    let mut rng = Rng::new(args.seed.or(design.randomization_seed).unwrap_or(0));
    let mut cs = vec![Vec::new(); coefficients.len()];
    let mut ps = vec![Vec::new(); rows.len()];
    for _ in 0..b {
        let draws: Vec<_> = (0..groups.len())
            .map(|_| rng.below(groups.len() as u64) as usize)
            .collect();
        let indices: Vec<_> = draws
            .iter()
            .flat_map(|&g| groups[g].iter().copied())
            .collect();
        let mut bx: Vec<_> = indices.iter().map(|&i| x[i].clone()).collect();
        // Center the sampled seed blocks at the mean of the sampled groups.
        for j in 1..design.replicates[0].levels.len() {
            let center =
                draws.iter().map(|&g| x[groups[g][0]][j]).sum::<f64>() / draws.len() as f64;
            for row in &mut bx {
                row[j] -= center;
            }
        }
        let by: Vec<_> = indices.iter().map(|&i| y[i]).collect();
        let fit = model::fit_least_squares(&bx, &by);
        for (j, samples) in cs.iter_mut().enumerate() {
            if fit.estimable_coef[j] {
                samples.push(fit.coef[j]);
            }
        }
        for (row, samples) in rows.iter().zip(&mut ps) {
            if model::is_estimable(&bx, row) {
                samples.push(dot(row, &fit.coef));
            }
        }
    }
    for (c, samples) in coefficients.iter_mut().zip(&mut cs) {
        if c.estimable {
            (c.boot_low, c.boot_high) = percentile(samples, args.alpha);
        }
    }
    for (p, samples) in predictions.iter_mut().zip(&mut ps) {
        if p.estimable {
            (p.boot_low, p.boot_high) = percentile(samples, args.alpha);
        }
    }
}
fn has_evidence(r: &ResultReport) -> bool {
    r.anova.iter().any(|a| a.significant)
}
fn display(v: Option<f64>) -> String {
    v.map_or_else(|| "n/a".into(), |v| format!("{v:.6}"))
}
fn render_result(r: &ResultReport, alpha: f64) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "\n=== Result: {} ===\nModel: {}\nComplete: {}\nMissing runs ({}): {}",
        r.name,
        r.model,
        r.n_complete,
        r.missing_runs.len(),
        r.missing_runs.join(", ")
    )
    .unwrap();
    writeln!(out,"Estimability under observed data\nParams: {} rank: {} residual df: {}\n  term  df  estimable  aliased with",r.estimability.n_params,r.estimability.rank,r.estimability.residual_df).unwrap();
    for t in &r.estimability.terms {
        writeln!(
            out,
            "  {}  {}  {}  {}",
            t.term,
            t.df,
            if t.estimable { "yes" } else { "no" },
            t.aliased_with.join(", ")
        )
        .unwrap();
    }
    writeln!(out,"RSS: {:.6} sigma: {} R squared: {}\nType II ANOVA\n  term  df  SS  MS  F  p  stratum  significance",r.fit.rss,display(r.fit.sigma),display(r.fit.r_squared)).unwrap();
    for a in &r.anova {
        writeln!(
            out,
            "  {}  {}  {:.6}  {}  {}  {}  {}  {}{}",
            a.term,
            a.df,
            a.ss,
            display(a.ms),
            display(a.f),
            display(a.p),
            a.stratum,
            if a.significant { "*" } else { "" },
            if a.pooled { " [pooled]" } else { "" }
        )
        .unwrap();
        let s = r.strata.iter().find(|s| {
            s.name == a.stratum
                || (a.stratum == "whole-plot" && s.name == "between-unit")
                || (a.stratum == "sub-plot" && s.name == "within-unit")
        });
        if !a.pooled && s.is_some_and(|s| s.df == 0) {
            writeln!(out, "    F: n/a (0 df in its error stratum)").unwrap();
        }
    }
    for s in &r.strata {
        writeln!(out, "{}: df={} MS={}", s.name, s.df, display(s.ms)).unwrap();
    }
    writeln!(
        out,
        "Coefficients: estimate SE CI low CI high bootstrap low bootstrap high"
    )
    .unwrap();
    for c in &r.coefficients {
        if c.estimable {
            writeln!(
                out,
                "{}: {} {} {} {} {} {}",
                c.label,
                display(c.estimate),
                display(c.se),
                display(c.ci_low),
                display(c.ci_high),
                display(c.boot_low),
                display(c.boot_high)
            )
            .unwrap();
        } else {
            writeln!(out, "{}: n/e", c.label).unwrap();
        }
    }
    for p in &r.predictions {
        writeln!(out,"Prediction {:?}: mean={:.6} SE={} estimable={} measured={} observed_mean={} bootstrap=[{}, {}]",p.factors,p.mean,display(p.se),p.estimable,p.measured,display(p.observed_mean),display(p.boot_low),display(p.boot_high)).unwrap();
    }
    for (label, p) in [
        ("MAXIMIZE", &r.recommendation.maximize),
        ("MINIMIZE", &r.recommendation.minimize),
    ] {
        if let Some(p) = p {
            writeln!(
                out,
                "{label}: {:?} predicted={:.6} interval=[{}, {}] measured={}",
                p.factors,
                p.mean,
                display(p.ci_low),
                display(p.ci_high),
                p.measured
            )
            .unwrap();
        }
    }
    for w in &r.warnings {
        writeln!(out, "{w}").unwrap();
    }
    if !has_evidence(r) {
        writeln!(out, "insufficient evidence under this model (α={alpha})").unwrap();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factor::Factor;
    use crate::model::{ReplicateRole, Unit};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn args() -> AnalyzeArgs {
        AnalyzeArgs {
            csv_path: "unused.csv".into(),
            design_path: None,
            maximize: vec![],
            minimize: vec![],
            tolerate_noise: None,
            alpha: 0.05,
            pool: vec![],
            pool_auto: false,
            bootstrap: Some(0),
            seed: Some(42),
            json: None,
        }
    }
    fn design(names: &[&str], formula: &str) -> Design {
        let factors: Vec<_> = names
            .iter()
            .map(|n| Factor::parse(n, "d[1,2]").unwrap())
            .collect();
        let models = vec![Model::parse(formula, &factors).unwrap()];
        serde_json::from_value(serde_json::json!({"factors":factors,"results":["y"],"models":models,"array":{"rows":4,"cols":names.len(),"label":"test","method":"test"}})).unwrap()
    }
    fn factorial(d: &Design, groups: usize, f: impl Fn(&[f64], usize) -> f64) -> Vec<(Cell, f64)> {
        let mut data = Vec::new();
        for g in 0..groups {
            let mut levels = vec![0; d.factors.len()];
            loop {
                let coded: Vec<_> = levels
                    .iter()
                    .map(|&l| if l == 0 { 1.0 } else { -1.0 })
                    .collect();
                data.push((
                    Cell {
                        factor_levels: levels.clone(),
                        replicate_levels: if d.replicates.is_empty() {
                            vec![]
                        } else {
                            vec![g]
                        },
                    },
                    f(&coded, g),
                ));
                if !advance(&mut levels, &d.factors) {
                    break;
                }
            }
        }
        data
    }
    fn fit(d: &Design, data: &[(Cell, f64)], a: &AnalyzeArgs) -> ResultReport {
        analyze_result(d, "y", data, vec![], a).unwrap()
    }
    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} != {b}");
    }

    #[test]
    fn golden_joint_interaction_and_missing_alias() {
        let mut d = design(&["a", "b"], "y ~ a*b");
        let mut data = factorial(&d, 1, |v, _| 10.0 + 2.0 * v[0] + 3.0 * v[1] + v[0] * v[1]);
        let mm = model::model_matrix(
            &d.factors,
            &[],
            &d.models[0],
            &data.iter().map(|(c, _)| c.clone()).collect::<Vec<_>>(),
            None,
        );
        d.estimability = Some(model::estimability(&mm, &d.models[0], &d.factors));
        let r = fit(&d, &data, &args());
        for (c, true_value) in r.coefficients.iter().zip([10.0, 2.0, 3.0, 1.0]) {
            close(c.estimate.unwrap(), true_value);
        }
        assert!(r.estimability.terms.iter().all(|t| t.estimable));
        close(r.fit.rss, 0.0);
        data.pop();
        let r = analyze_result(&d, "y", &data, vec!["r0004".into()], &args()).unwrap();
        assert!(!r.estimability.terms[2].estimable);
        let text = render_result(&r, 0.05);
        assert!(text.contains("Missing runs (1): r0004"));
        assert!(text.contains("a:b: LOST (missing runs)"));
        assert!(text.contains("n/e"));
        assert_eq!(r.predictions.iter().filter(|p| p.measured).count(), 3);
        assert!(!r.predictions[3].estimable);
    }
    #[test]
    fn split_plot_denominators_are_nested_fit_mean_squares() {
        let mut d = design(&["a", "b"], "a+b");
        d.replicates = vec![ReplicateRole {
            name: "seed".into(),
            levels: vec![FactorValue::Int(1), FactorValue::Int(2)],
        }];
        d.unit = Some(Unit {
            name: "checkpoint".into(),
            key: vec!["a".into(), "seed".into()],
        });
        let data = factorial(&d, 2, |v, g| {
            let s = if g == 0 { 1.0 } else { -1.0 };
            10.0 + 2.0 * v[0] + 3.0 * v[1] + 20.0 * v[0] * s + v[1] * s
        });
        let r = fit(&d, &data, &args());
        // Orthogonal contrasts: the whole-only residual contains b, a:seed,
        // and b:seed; unit columns remove a:seed, with SS=8*20^2, df=1.
        close(r.strata[0].ms.unwrap(), 3200.0);
        assert_eq!(r.strata[0].df, 1);
        // Full + unit leaves b:seed: SS=8, residual df=3.
        close(r.strata[1].ms.unwrap(), 8.0 / 3.0);
        assert_eq!(r.strata[1].df, 3);
        let a = &r.anova[0];
        let b = &r.anova[1];
        close(a.ms.unwrap() / a.f.unwrap(), 3200.0);
        close(b.ms.unwrap() / b.f.unwrap(), 8.0 / 3.0);
        assert_eq!(a.stratum, "whole-plot");
        assert_eq!(b.stratum, "sub-plot");
    }
    #[test]
    fn saturated_pooling_is_opt_in_and_refits() {
        let d = design(&["a", "b", "c"], "a+b+c");
        let data: Vec<_> = [vec![0, 0, 0], vec![0, 1, 1], vec![1, 0, 1], vec![1, 1, 0]]
            .into_iter()
            .map(|levels| {
                let v: Vec<_> = levels
                    .iter()
                    .map(|&l| if l == 0 { 1.0 } else { -1.0 })
                    .collect();
                (
                    Cell {
                        factor_levels: levels,
                        replicate_levels: vec![],
                    },
                    10.0 + 10.0 * v[0] + 0.1 * v[1] + 0.2 * v[2],
                )
            })
            .collect();
        let r = fit(&d, &data, &args());
        assert!(render_result(&r,0.05).contains("insufficient evidence under this model: residual df = 0 in residual; add replicates, or pass --pool <terms> / --pool-auto."));
        assert!(r.anova.iter().all(|a| !a.pooled && a.f.is_none()));
        let mut a = args();
        a.pool_auto = true;
        let r = fit(&d, &data, &a);
        assert_eq!(r.anova.iter().filter(|a| a.pooled).count(), 2);
        assert!(r.anova[0].f.is_some());
        assert!(has_evidence(&r));
        assert_eq!(r.coefficients.len(), 2);
        a.pool_auto = false;
        a.pool = vec!["c".into()];
        let r = fit(&d, &data, &a);
        assert!(r.anova[2].pooled);
        assert_eq!(r.strata[0].df, 1);
    }
    #[test]
    fn paired_bootstrap_contains_truth_and_repeats() {
        let mut d = design(&["a", "b"], "a*b");
        d.replicates = vec![ReplicateRole {
            name: "seed".into(),
            levels: (1..=6).map(FactorValue::Int).collect(),
        }];
        let data = factorial(&d, 6, |v, g| {
            let noise = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0][g];
            10.0 + 2.0 * v[0] + 3.0 * v[1] + v[0] * v[1] + noise + 0.1 * noise * v[0]
        });
        let mut a = args();
        a.bootstrap = Some(100);
        let first = fit(&d, &data, &a);
        let second = fit(&d, &data, &a);
        assert_eq!(
            serde_json::to_value(&first).unwrap(),
            serde_json::to_value(&second).unwrap()
        );
        let effect = first
            .coefficients
            .iter()
            .find(|c| c.label == "a[1]")
            .unwrap();
        assert!(effect.boot_low.unwrap() < 2.0 && effect.boot_high.unwrap() > 2.0);
        let p = &first.predictions[0];
        assert!(p.boot_low.unwrap() < 16.0 && p.boot_high.unwrap() > 16.0);
        assert!(first.warnings.is_empty());
    }
    #[test]
    fn type_two_omits_containing_terms_and_reports_blocks() {
        let mut d = design(&["a", "b"], "a*b");
        d.replicates = vec![ReplicateRole {
            name: "seed".into(),
            levels: vec![FactorValue::Int(1), FactorValue::Int(2)],
        }];
        let mut data = factorial(&d, 2, |v, g| {
            10.0 + 2.0 * v[0] + 3.0 * v[1] + v[0] * v[1] + g as f64
        });
        data.pop();
        let r = fit(&d, &data, &args());
        let cells: Vec<_> = data.iter().map(|(c, _)| c.clone()).collect();
        let y: Vec<_> = data.iter().map(|(_, y)| *y).collect();
        let reduced = Model::parse("b", &d.factors).unwrap();
        let added = Model::parse("a+b", &d.factors).unwrap();
        let rss = |m: &Model| {
            model::fit_least_squares(
                &model::model_matrix(&d.factors, &d.replicates, m, &cells, None).data,
                &y,
            )
            .rss
        };
        close(r.anova[0].ss, rss(&reduced) - rss(&added));
        assert_eq!(r.anova.last().unwrap().term, "(blocks)");
    }
    #[test]
    fn readme_three_factor_optimum_and_json_shape() {
        let d = design(
            &["learning_rate", "batch_size", "optimizer"],
            "learning_rate+batch_size+optimizer",
        );
        let data = factorial(&d, 2, |v, g| {
            20.0 + 4.0 * v[0] - 3.0 * v[1] + 2.0 * v[2] + if g == 0 { 0.1 } else { -0.1 }
        });
        let r = fit(&d, &data, &args());
        let optimum = r.recommendation.maximize.as_ref().unwrap();
        close(optimum.mean, 29.0);
        close(r.coefficients[0].se.unwrap(), (1.0_f64 / 1200.0).sqrt());
        assert!(r.coefficients[0].ci_low.unwrap() < 20.0);
        assert!(r.coefficients[0].ci_high.unwrap() > 20.0);
        assert_eq!(optimum.factors["batch_size"], "2");
        assert!(optimum.measured);
        assert!(render_result(&r, 0.05).contains("MAXIMIZE"));
        let report = Report {
            version: 2,
            design: "fixture.design.json".into(),
            results: vec![r],
        };
        let json = serde_json::to_string(&report).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        for key in ["version", "design", "results"] {
            assert!(value.get(key).is_some());
        }
        let result = &value["results"][0];
        for key in [
            "name",
            "model",
            "n_complete",
            "missing_runs",
            "estimability",
            "fit",
            "anova",
            "strata",
            "coefficients",
            "predictions",
            "recommendation",
            "warnings",
        ] {
            assert!(result.get(key).is_some(), "{key}");
        }
        for key in [
            "factors",
            "mean",
            "se",
            "estimable",
            "measured",
            "observed_mean",
            "boot_low",
            "boot_high",
        ] {
            assert!(result["predictions"][0].get(key).is_some(), "{key}");
        }
        let _: Report = serde_json::from_str(&json).unwrap();
    }
    #[test]
    fn readme_mixed_level_example() {
        let mut d = design(
            &["cpu_freq", "cache_mb", "prefetch"],
            "cpu_freq+cache_mb+prefetch",
        );
        d.factors = vec![
            Factor::parse("cpu_freq", "uniform[1.2,3.6,3]").unwrap(),
            Factor::parse("cache_mb", "d[2,4,8,16]").unwrap(),
            Factor::parse("prefetch", r#"d["off","soft","aggressive"]"#).unwrap(),
        ];
        let mut data = Vec::new();
        let mut levels = vec![0; 3];
        loop {
            for noise in [-0.1, 0.1] {
                let y = 10.0 + 2.0 * levels[0] as f64 + 3.0 * levels[1] as f64 - levels[2] as f64
                    + noise;
                data.push((
                    Cell {
                        factor_levels: levels.clone(),
                        replicate_levels: vec![],
                    },
                    y,
                ));
            }
            if !advance(&mut levels, &d.factors) {
                break;
            }
        }
        let r = fit(&d, &data, &args());
        let best = r.recommendation.maximize.as_ref().unwrap();
        close(best.mean, 23.0);
        assert_eq!(best.factors["cpu_freq"], "3.6");
        assert_eq!(best.factors["cache_mb"], "16");
        assert_eq!(best.factors["prefetch"], "off");
        assert!(render_result(&r, 0.05).contains("MAXIMIZE"));
    }

    #[test]
    fn oversized_grid_keeps_measured_and_exact_recommendation_cells() {
        let names: Vec<_> = (0..17).map(|i| format!("f{i}")).collect();
        let refs: Vec<_> = names.iter().map(String::as_str).collect();
        let d = design(&refs, "f0");
        let data: Vec<_> = [(0, 11.9), (0, 12.1), (1, 7.9), (1, 8.1)]
            .into_iter()
            .map(|(l, y)| {
                (
                    Cell {
                        factor_levels: vec![l; 17],
                        replicate_levels: vec![],
                    },
                    y,
                )
            })
            .collect();
        let r = fit(&d, &data, &args());
        assert_eq!(r.predictions.len(), 3);
        assert_eq!(r.predictions.iter().filter(|p| p.measured).count(), 2);
        let best = r.recommendation.minimize.as_ref().unwrap();
        close(best.mean, 8.0);
        assert!(!best.measured);
        assert!(r.predictions.iter().any(|p| p.factors == best.factors));
        assert!(r.warnings.iter().any(|w| w.contains("100000")));
    }

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let p = std::env::temp_dir().join(format!(
                "taguchi-w3-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&p).unwrap();
            Self(p)
        }
        fn args(&self, d: &Design, csv: &str) -> AnalyzeArgs {
            let mut a = args();
            a.csv_path = self.0.join("data.csv");
            std::fs::write(&a.csv_path, csv).unwrap();
            serde_json::to_writer(File::create(sidecar_path(&a.csv_path)).unwrap(), d).unwrap();
            a
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }
    #[test]
    fn csv_v1_v2_missing_and_invalid_cells() {
        let d = design(&["a", "b"], "a*b");
        let temp = Fixture::new();
        for (header, id, expected) in [("experiment", "1", "3"), ("run_id", "r0001", "r0002")] {
            let second = if header == "experiment" { "2" } else { "r0002" };
            let a = temp.args(
                &d,
                &format!("{header},b,y,a\n{id},1,16,1.0\n{second},2,,1\n"),
            );
            let r = analyze(&a).unwrap();
            assert_eq!(r.results[0].missing_runs, [expected]);
            assert_eq!(r.results[0].n_complete, 1);
        }
        for raw in ["bad", "NaN", "inf"] {
            let a = temp.args(&d, &format!("experiment,a,b,y\n1,1,1,{raw}\n"));
            let err = analyze(&a).unwrap_err().to_string();
            assert!(err.contains("row 2") && err.contains("column 'y'"), "{err}");
        }
        let a = temp.args(&d, "experiment,a,b,y\n1,1,1,\n");
        let r = analyze(&a).unwrap();
        assert_eq!(r.results[0].n_complete, 0);
        assert!(!has_evidence(&r.results[0]));
        assert!(r.results[0].coefficients.iter().all(|c| !c.estimable));
    }
    #[test]
    fn manifest_reports_deleted_rows_and_multiple_results_gate() {
        let mut d = design(&["a", "b"], "a+b");
        d.runs = vec![crate::construct::Run {
            run_id: "r0002".into(),
            order: 2,
            design_row: 1,
            factors: BTreeMap::new(),
            replicates: BTreeMap::new(),
            unit: None,
        }];
        let fixture = Fixture::new();
        let a = fixture.args(&d, "run_id,a,b,y\nr0001,1,1,2\n");
        let r = analyze(&a).unwrap();
        assert_eq!(r.results[0].missing_runs, ["r0002"]);
        d.results.push("flat".into());
        let mut a = fixture.args(
            &d,
            "experiment,a,b,y,flat\n1,1,1,12,10\n2,1,2,12,10\n3,2,1,8,10\n4,2,2,8,10\n",
        );
        a.tolerate_noise = Some(1.0);
        let r = analyze(&a).unwrap();
        assert!(has_evidence(&r.results[0]));
        assert!(!has_evidence(&r.results[1]));
        assert!(!r.results.iter().all(|r| !has_evidence(r)));
    }

    #[test]
    fn tolerate_noise_uses_resolved_term_cell_means() {
        let d = design(&["a", "b"], "a*b");
        let data = factorial(&d, 1, |v, _| 10.0 + 2.0 * v[0] * v[1]);
        let mut a = args();
        a.tolerate_noise = Some(3.0);
        let r = fit(&d, &data, &a);
        assert!(!r.anova[0].significant);
        assert!(!r.anova[1].significant);
        assert!(r.anova[2].significant);
        a.tolerate_noise = Some(4.0);
        assert!(!has_evidence(&fit(&d, &data, &a)));
    }
}
