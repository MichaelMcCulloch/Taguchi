use anyhow::{Context, Result, anyhow};
use statrs::distribution::{ContinuousCDF, FisherSnedecor};
use std::fs::File;
use std::path::PathBuf;

use crate::construct::{Design, sidecar_path};
use crate::factor::{Factor, FactorValue};

#[derive(Debug)]
pub struct AnalyzeArgs {
    pub csv_path: PathBuf,
    pub design_path: Option<PathBuf>,
    pub maximize: Vec<String>,
    pub minimize: Vec<String>,
    pub tolerate_noise: Option<f64>,
    pub alpha: f64,
}

#[derive(Debug)]
struct LevelStat {
    value: FactorValue,
    mean: f64,
    n: usize,
}

#[derive(Debug)]
struct FactorAnalysis {
    name: String,
    level_stats: Vec<LevelStat>,
    ss: f64,
    df: usize,
    ms: f64,
    f_stat: Option<f64>,
    p_value: Option<f64>,
    contribution_pct: f64,
    effect_range: f64,
    pooled: bool,
    best_level_for_max: usize,
    best_level_for_min: usize,
}

#[derive(Debug)]
struct ResultAnalysis {
    name: String,
    n_obs: usize,
    grand_mean: f64,
    total_ss: f64,
    error_ss: f64,
    error_df: usize,
    error_ms: f64,
    factors: Vec<FactorAnalysis>,
    pooled_names: Vec<String>,
}

#[derive(Copy, Clone)]
enum Direction {
    Max,
    Min,
    Both,
}

pub fn run_analyze(args: AnalyzeArgs) -> Result<()> {
    let design_path = args
        .design_path
        .clone()
        .unwrap_or_else(|| sidecar_path(&args.csv_path));
    let design: Design = serde_json::from_reader(
        File::open(&design_path)
            .with_context(|| format!("opening design file {}", design_path.display()))?,
    )?;

    let mut rdr = csv::Reader::from_path(&args.csv_path)
        .with_context(|| format!("opening {}", args.csv_path.display()))?;
    let headers = rdr.headers()?.clone();

    let factor_idx: Vec<usize> = design
        .factors
        .iter()
        .map(|f| {
            headers
                .iter()
                .position(|h| h == f.name)
                .ok_or_else(|| anyhow!("factor column '{}' not in CSV", f.name))
        })
        .collect::<Result<_>>()?;

    let result_idx: Vec<usize> = design
        .results
        .iter()
        .map(|r| {
            headers
                .iter()
                .position(|h| h == r)
                .ok_or_else(|| anyhow!("result column '{}' not in CSV", r))
        })
        .collect::<Result<_>>()?;

    let mut data_per_result: Vec<Vec<(Vec<usize>, f64)>> = vec![Vec::new(); design.results.len()];

    for (row_idx, row) in rdr.records().enumerate() {
        let row = row?;
        let level_indices: Vec<usize> = factor_idx
            .iter()
            .enumerate()
            .map(|(fi, &ci)| {
                let cell = row.get(ci).unwrap_or("").trim();
                level_index_for(&design.factors[fi], cell).ok_or_else(|| {
                    anyhow!(
                        "row {}: factor '{}' value '{}' not in design levels",
                        row_idx + 2,
                        design.factors[fi].name,
                        cell
                    )
                })
            })
            .collect::<Result<_>>()?;

        for (ri, &ci) in result_idx.iter().enumerate() {
            let cell = row.get(ci).unwrap_or("").trim();
            if cell.is_empty() {
                continue;
            }
            let y: f64 = cell.parse().with_context(|| {
                format!(
                    "row {}: result '{}' value '{}'",
                    row_idx + 2,
                    design.results[ri],
                    cell
                )
            })?;
            data_per_result[ri].push((level_indices.clone(), y));
        }
    }

    let mut any_recommendation = false;
    for (ri, result_name) in design.results.iter().enumerate() {
        let data = &data_per_result[ri];
        if data.is_empty() {
            eprintln!("⚠ '{}': no data, skipping", result_name);
            continue;
        }
        let analysis = analyze_result(result_name, data, &design.factors)?;
        let dir = if args.maximize.iter().any(|c| c == result_name) {
            Direction::Max
        } else if args.minimize.iter().any(|c| c == result_name) {
            Direction::Min
        } else {
            Direction::Both
        };
        let had_rec = print_result(&analysis, args.tolerate_noise, args.alpha, dir);
        any_recommendation |= had_rec;
    }

    if !any_recommendation {
        std::process::exit(2);
    }
    Ok(())
}

fn level_index_for(factor: &Factor, cell: &str) -> Option<usize> {
    for (i, v) in factor.values.iter().enumerate() {
        if value_matches(v, cell) {
            return Some(i);
        }
    }
    None
}

fn value_matches(v: &FactorValue, cell: &str) -> bool {
    if v.to_string() == cell {
        return true;
    }
    match v {
        FactorValue::Text(s) => s == cell,
        FactorValue::Int(n) => cell.parse::<i64>().map(|c| c == *n).unwrap_or(false)
            || cell
                .parse::<f64>()
                .map(|c| (c - *n as f64).abs() < 1e-9)
                .unwrap_or(false),
        FactorValue::Float(x) => match cell.parse::<f64>() {
            Ok(c) => {
                let scale = x.abs().max(1.0);
                (c - x).abs() < 1e-4 * scale
            }
            Err(_) => false,
        },
    }
}

fn analyze_result(
    name: &str,
    data: &[(Vec<usize>, f64)],
    factors: &[Factor],
) -> Result<ResultAnalysis> {
    let n = data.len();
    let grand_mean = data.iter().map(|(_, y)| y).sum::<f64>() / n as f64;
    let total_ss: f64 = data.iter().map(|(_, y)| (y - grand_mean).powi(2)).sum();

    let mut factor_analyses: Vec<FactorAnalysis> = Vec::with_capacity(factors.len());
    let mut sum_factor_ss = 0.0;
    let mut sum_factor_df = 0usize;

    for (fi, factor) in factors.iter().enumerate() {
        let l = factor.level_count();
        let mut sums = vec![0.0; l];
        let mut counts = vec![0usize; l];
        for (lvls, y) in data {
            sums[lvls[fi]] += y;
            counts[lvls[fi]] += 1;
        }
        let means: Vec<f64> = (0..l)
            .map(|i| {
                if counts[i] > 0 {
                    sums[i] / counts[i] as f64
                } else {
                    grand_mean
                }
            })
            .collect();
        let ss: f64 = (0..l)
            .map(|i| counts[i] as f64 * (means[i] - grand_mean).powi(2))
            .sum();
        let effective_l = counts.iter().filter(|&&c| c > 0).count();
        let df = effective_l.saturating_sub(1);

        let level_stats: Vec<LevelStat> = (0..l)
            .map(|i| LevelStat {
                value: factor.values[i].clone(),
                mean: means[i],
                n: counts[i],
            })
            .collect();

        let observed: Vec<usize> = (0..l).filter(|&i| counts[i] > 0).collect();
        let best_max = *observed
            .iter()
            .max_by(|&&a, &&b| means[a].partial_cmp(&means[b]).unwrap())
            .unwrap_or(&0);
        let best_min = *observed
            .iter()
            .min_by(|&&a, &&b| means[a].partial_cmp(&means[b]).unwrap())
            .unwrap_or(&0);
        let effect_range = if observed.is_empty() {
            0.0
        } else {
            let max_m = observed
                .iter()
                .map(|&i| means[i])
                .fold(f64::NEG_INFINITY, f64::max);
            let min_m = observed
                .iter()
                .map(|&i| means[i])
                .fold(f64::INFINITY, f64::min);
            max_m - min_m
        };

        factor_analyses.push(FactorAnalysis {
            name: factor.name.clone(),
            level_stats,
            ss,
            df,
            ms: if df > 0 { ss / df as f64 } else { 0.0 },
            f_stat: None,
            p_value: None,
            contribution_pct: if total_ss > 0.0 {
                100.0 * ss / total_ss
            } else {
                0.0
            },
            effect_range,
            pooled: false,
            best_level_for_max: best_max,
            best_level_for_min: best_min,
        });
        sum_factor_ss += ss;
        sum_factor_df += df;
    }

    let mut error_ss = (total_ss - sum_factor_ss).max(0.0);
    let mut error_df = (n - 1).saturating_sub(sum_factor_df);
    let mut pooled_names: Vec<String> = Vec::new();

    // Saturated design → pool smallest-SS factors into the error estimate.
    if error_df == 0 && !factor_analyses.is_empty() {
        let mut indices: Vec<usize> = (0..factor_analyses.len()).collect();
        indices.sort_by(|&a, &b| {
            factor_analyses[a]
                .ss
                .partial_cmp(&factor_analyses[b].ss)
                .unwrap()
        });
        let n_pool = (factor_analyses.len() + 1) / 2;
        for &i in indices.iter().take(n_pool) {
            error_ss += factor_analyses[i].ss;
            error_df += factor_analyses[i].df;
            factor_analyses[i].pooled = true;
            pooled_names.push(factor_analyses[i].name.clone());
        }
    }

    let error_ms = if error_df > 0 {
        error_ss / error_df as f64
    } else {
        0.0
    };

    for fa in factor_analyses.iter_mut() {
        if fa.pooled {
            continue;
        }
        if error_ms > 0.0 && fa.df > 0 && error_df > 0 {
            let f_stat = fa.ms / error_ms;
            fa.f_stat = Some(f_stat);
            if let Ok(dist) = FisherSnedecor::new(fa.df as f64, error_df as f64) {
                let p = 1.0 - dist.cdf(f_stat);
                fa.p_value = Some(p);
            }
        }
    }

    Ok(ResultAnalysis {
        name: name.to_string(),
        n_obs: n,
        grand_mean,
        total_ss,
        error_ss,
        error_df,
        error_ms,
        factors: factor_analyses,
        pooled_names,
    })
}

fn print_result(
    a: &ResultAnalysis,
    tolerate: Option<f64>,
    alpha: f64,
    dir: Direction,
) -> bool {
    println!();
    println!("=== Result: {} ===", a.name);
    println!("  observations: {}", a.n_obs);
    println!("  grand mean:   {:.6}", a.grand_mean);
    println!("  total SS:     {:.6}", a.total_ss);
    let noise_std = a.error_ms.sqrt();
    if a.error_df > 0 {
        println!(
            "  residual:     SS={:.6}  df={}  σ̂={:.6}",
            a.error_ss, a.error_df, noise_std
        );
    } else {
        println!(
            "  residual:     SS={:.6}  df=0  (cannot estimate noise)",
            a.error_ss
        );
    }
    if !a.pooled_names.is_empty() {
        println!("  pooled into error: {}", a.pooled_names.join(", "));
    }
    println!();
    println!("  Factor decomposition (sorted by contribution):");

    let mut sorted: Vec<&FactorAnalysis> = a.factors.iter().collect();
    sorted.sort_by(|x, y| y.ss.partial_cmp(&x.ss).unwrap());

    let mut significant: Vec<&FactorAnalysis> = Vec::new();
    for fa in &sorted {
        let is_sig = if fa.pooled {
            false
        } else {
            match tolerate {
                Some(s) => fa.effect_range > s,
                None => fa.p_value.map(|p| p < alpha).unwrap_or(false),
            }
        };
        let marker = if is_sig {
            "★"
        } else if fa.pooled {
            "·"
        } else {
            " "
        };
        let stats = match (fa.f_stat, fa.p_value) {
            (Some(f), Some(p)) => format!("F={:6.2}  p={:.4}", f, p),
            _ => "  (no F-test)         ".to_string(),
        };
        let tag = if fa.pooled { "  [pooled→err]" } else { "" };
        println!(
            "    {} {:<20} {:5.1}%  {}  range={:.4}{}",
            marker, fa.name, fa.contribution_pct, stats, fa.effect_range, tag
        );
        for ls in &fa.level_stats {
            if ls.n == 0 {
                continue;
            }
            println!("        {:>14} : μ={:>10.4}  n={}", ls.value, ls.mean, ls.n);
        }
        if is_sig {
            significant.push(*fa);
        }
    }
    let residual_pct = if a.total_ss > 0.0 {
        100.0 * a.error_ss / a.total_ss
    } else {
        0.0
    };
    println!("      (residual)           {:5.1}%", residual_pct);

    println!();
    if significant.is_empty() {
        match tolerate {
            Some(s) => println!(
                "  ⚠ no factor exceeds noise floor σ={}. treating result as noise.",
                s
            ),
            None => {
                println!(
                    "  ⚠ no factor reached significance at α={}. treating result as noise.",
                    alpha
                );
                println!("    pass --tolerate-noise <σ> to report effects whose level-range exceeds σ.");
            }
        }
        return false;
    }

    let (show_max, show_min) = match dir {
        Direction::Max => (true, false),
        Direction::Min => (false, true),
        Direction::Both => (true, true),
    };

    if show_max {
        print_recommendation(&significant, a.grand_mean, true);
    }
    if show_min {
        print_recommendation(&significant, a.grand_mean, false);
    }
    true
}

fn print_recommendation(sig: &[&FactorAnalysis], grand_mean: f64, maximize: bool) {
    let label = if maximize { "MAXIMIZE" } else { "MINIMIZE" };
    println!("  → To {}:", label);
    let mut predicted = grand_mean;
    for fa in sig {
        let idx = if maximize {
            fa.best_level_for_max
        } else {
            fa.best_level_for_min
        };
        let level = &fa.level_stats[idx];
        println!("      {:<20} = {}", fa.name, level.value);
        predicted += level.mean - grand_mean;
    }
    println!("      predicted response ≈ {:.4}", predicted);
    println!();
}
