use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use crate::design_select::{self, SelectOpts};
use crate::factor::{Factor, FactorValue};
pub use crate::model::EstimabilityReport;
use crate::model::{Cell, Model, ReplicateRole, Unit, formula_names};
use crate::oa::{ArrayInfo, verify_strength2};
use crate::rng::Rng;

#[derive(Debug, Serialize, Deserialize)]
pub struct Design {
    #[serde(default = "two")]
    pub version: u32,
    pub factors: Vec<Factor>,
    pub results: Vec<String>,
    #[serde(default)]
    pub models: Vec<Model>,
    #[serde(default)]
    pub replicates: Vec<ReplicateRole>,
    #[serde(default)]
    pub unit: Option<Unit>,
    pub array: ArrayInfo,
    #[serde(default)]
    pub randomization_seed: Option<u64>,
    #[serde(default)]
    pub estimability: Option<EstimabilityReport>,
    #[serde(default)]
    pub runs: Vec<Run>,
    #[serde(default)]
    pub augmentations: Vec<Augmentation>,
}

fn two() -> u32 {
    2
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    pub run_id: String,
    pub order: usize,
    pub design_row: usize,
    pub factors: BTreeMap<String, FactorValue>,
    pub replicates: BTreeMap<String, FactorValue>,
    pub unit: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Augmentation {
    pub seed: u64,
    pub targets: Vec<String>,
    pub run_ids: Vec<String>,
}

/// Construct options from the CLI. The interactive and the `--from` paths
/// take the same struct, so both get the same behaviour.
#[derive(Debug, Clone, Default)]
pub struct ConstructOpts {
    /// `--model FORMULA`, appended after the file's `[model]` lines.
    pub models: Vec<String>,
    /// `--replicate NAME=SPEC`, appended after the file's `[replicates]` lines.
    pub replicates: Vec<String>,
    /// `--unit NAME=k1,k2,…`; an error when the file already declares one.
    pub unit: Option<String>,
    /// `--seed`; without it the seed is nanoseconds since the epoch.
    pub seed: Option<u64>,
    pub max_runs: Option<usize>,
    pub min_residual_df: usize,
    pub allow_aliased: bool,
    /// `--no-shuffle`: execution order is design order, still recorded.
    pub no_shuffle: bool,
}

impl ConstructOpts {
    fn select_opts(&self) -> SelectOpts {
        SelectOpts {
            max_runs: self.max_runs,
            min_residual_df: self.min_residual_df,
            allow_aliased: self.allow_aliased,
        }
    }
}

pub fn run_construct(output: PathBuf, opts: &ConstructOpts) -> Result<()> {
    let stdin = std::io::stdin();
    let mut reader = stdin.lock();
    let mut prompt_line = String::new();

    eprintln!("=== taguchi construct ===");
    eprintln!("Enter factors one per line as 'name = spec'.");
    eprintln!(
        "Specs: d[1,2-4,9]  uniform[0,10,5]  normal[0,10,5,2]  logLow[1,1000,4]  logHigh[1,1000,4]"
    );
    eprintln!("Blank line to finish factors.");

    let mut factors: Vec<Factor> = Vec::new();
    loop {
        eprint!("factor {}> ", factors.len() + 1);
        std::io::stderr().flush().ok();
        prompt_line.clear();
        if reader.read_line(&mut prompt_line)? == 0 {
            break;
        }
        let line = prompt_line.trim();
        if line.is_empty() {
            if factors.is_empty() {
                eprintln!("need at least one factor");
                continue;
            }
            break;
        }
        match parse_factor_line(line) {
            Ok(f) => factors.push(f),
            Err(e) => eprintln!("error: {:#}", e),
        }
    }
    if factors.is_empty() {
        bail!("no factors entered");
    }

    eprintln!();
    eprintln!("Enter result column names, one per line (blank to finish).");
    let mut results: Vec<String> = Vec::new();
    loop {
        eprint!("result {}> ", results.len() + 1);
        std::io::stderr().flush().ok();
        prompt_line.clear();
        if reader.read_line(&mut prompt_line)? == 0 {
            break;
        }
        let line = prompt_line.trim();
        if line.is_empty() {
            if results.is_empty() {
                eprintln!("need at least one result column");
                continue;
            }
            break;
        }
        if !is_valid_name(line) {
            eprintln!("name must be non-empty identifier characters");
            continue;
        }
        results.push(line.to_string());
    }
    if results.is_empty() {
        bail!("no result columns entered");
    }

    let mut parsed = ParsedDesign {
        factors,
        results,
        models: Vec::new(),
        replicates: Vec::new(),
        unit: None,
    };
    apply_cli(&mut parsed, opts)?;
    write_design(&parsed, opts, &output)
}

pub fn run_construct_from_file(
    input: PathBuf,
    output: PathBuf,
    opts: &ConstructOpts,
) -> Result<()> {
    let file = File::open(&input).with_context(|| format!("opening {}", input.display()))?;
    let mut parsed = parse_design_file(BufReader::new(file), &input)?;
    apply_cli(&mut parsed, opts)?;
    write_design(&parsed, opts, &output)
}

/// Append the CLI's `--model`, `--replicate` and `--unit` values after the
/// file's own sections, with the same validation the file lines get.
fn apply_cli(parsed: &mut ParsedDesign, opts: &ConstructOpts) -> Result<()> {
    for src in &opts.replicates {
        let role = parse_replicate(src).with_context(|| format!("--replicate '{}'", src))?;
        if parsed.factors.iter().any(|f| f.name == role.name)
            || parsed.replicates.iter().any(|r| r.name == role.name)
        {
            bail!(
                "--replicate '{}': duplicate factor or replicate-role name '{}'",
                src,
                role.name
            );
        }
        parsed.replicates.push(role);
    }
    for src in &opts.models {
        let model = build_model(
            src,
            &parsed.factors,
            &parsed.results,
            &parsed.replicates,
            &parsed.models,
        )
        .with_context(|| format!("--model '{}'", src))?;
        parsed.models.push(model);
    }
    if let Some(src) = &opts.unit {
        if parsed.unit.is_some() {
            bail!("--unit '{}': a unit is already declared (at most one)", src);
        }
        parsed.unit = Some(
            parse_unit(src, &parsed.factors, &parsed.replicates)
                .with_context(|| format!("--unit '{}'", src))?,
        );
    }
    Ok(())
}

struct ParsedDesign {
    factors: Vec<Factor>,
    results: Vec<String>,
    models: Vec<Model>,
    replicates: Vec<ReplicateRole>,
    unit: Option<Unit>,
}

fn parse_design_file(reader: impl BufRead, input: &Path) -> Result<ParsedDesign> {
    let mut factors = Vec::new();
    let mut results = Vec::new();
    let mut formulas = Vec::new();
    let mut role_lines = Vec::new();
    let mut unit_line = None;
    let mut section = Section::Factors;
    for (lineno, line) in reader.lines().enumerate() {
        let lineno = lineno + 1;
        let line = line.with_context(|| format!("{}:{}", input.display(), lineno))?;
        let s = line.trim();
        if s.is_empty() || s.starts_with('#') {
            continue;
        }
        section = match s.to_ascii_lowercase().as_str() {
            "[factors]" => Section::Factors,
            "[results]" => Section::Results,
            "[model]" => Section::Model,
            "[replicates]" => Section::Replicates,
            "[units]" => Section::Units,
            _ => {
                let context = || format!("{}:{}", input.display(), lineno);
                match section {
                    Section::Factors => factors.push(parse_factor_line(s).with_context(context)?),
                    Section::Results => {
                        if !is_valid_name(s) {
                            bail!(
                                "{}:{}: invalid result name '{}'",
                                input.display(),
                                lineno,
                                s
                            );
                        }
                        results.push(s.to_string());
                    }
                    Section::Model => formulas.push((lineno, strip_comment(s).trim().to_string())),
                    Section::Replicates => {
                        role_lines.push((lineno, strip_comment(s).trim().to_string()))
                    }
                    Section::Units => {
                        if unit_line.is_some() {
                            bail!(
                                "{}:{}: [units] permits at most one line",
                                input.display(),
                                lineno
                            );
                        }
                        unit_line = Some((lineno, strip_comment(s).trim().to_string()));
                    }
                }
                continue;
            }
        };
    }
    if factors.is_empty() {
        bail!("no factors in {}", input.display());
    }
    if results.is_empty() {
        bail!(
            "no result columns in {} (use a [results] section)",
            input.display()
        );
    }
    let mut replicates = Vec::new();
    for (line, src) in role_lines {
        let role =
            parse_replicate(&src).with_context(|| format!("{}:{}", input.display(), line))?;
        if factors.iter().any(|f| f.name == role.name)
            || replicates
                .iter()
                .any(|r: &ReplicateRole| r.name == role.name)
        {
            bail!(
                "{}:{}: duplicate factor or replicate-role name '{}'",
                input.display(),
                line,
                role.name
            );
        }
        replicates.push(role);
    }
    let mut models = Vec::new();
    for (line, src) in formulas {
        let model = build_model(&src, &factors, &results, &replicates, &models)
            .with_context(|| format!("{}:{}", input.display(), line))?;
        models.push(model);
    }
    let unit = unit_line
        .map(|(line, src)| {
            parse_unit(&src, &factors, &replicates)
                .with_context(|| format!("{}:{}", input.display(), line))
        })
        .transpose()?;
    Ok(ParsedDesign {
        factors,
        results,
        models,
        replicates,
        unit,
    })
}

/// Parse and validate one formula against the factors, results, replicate
/// roles, and the formulas accepted so far. Used for `[model]` lines and for
/// `--model` alike, so both reject the same things.
fn build_model(
    src: &str,
    factors: &[Factor],
    results: &[String],
    replicates: &[ReplicateRole],
    models: &[Model],
) -> Result<Model> {
    for name in formula_names(src)? {
        if replicates.iter().any(|r| r.name == name) {
            bail!("replicate role '{}' cannot appear in a model formula", name);
        }
    }
    let model = Model::parse(src, factors)?;
    if let Some(response) = &model.response
        && !results.contains(response)
    {
        bail!("unknown result name '{}'", response);
    }
    if models.iter().any(|m| m.response == model.response) {
        bail!(
            "duplicate model for response '{}'",
            model.response.as_deref().unwrap_or(".")
        );
    }
    Ok(model)
}

// Keep quoted labels containing '#' intact in the new sections.
fn strip_comment(s: &str) -> &str {
    let mut quoted = false;
    for (i, c) in s.char_indices() {
        if c == '"' {
            quoted = !quoted;
        }
        if c == '#' && !quoted {
            return &s[..i];
        }
    }
    s
}

fn parse_replicate(src: &str) -> Result<ReplicateRole> {
    let (name, spec) = src
        .split_once('=')
        .ok_or_else(|| anyhow::anyhow!("expected 'name = spec'"))?;
    let (name, spec) = (name.trim(), spec.trim());
    if !is_valid_name(name) {
        bail!("invalid replicate-role name '{}'", name);
    }
    let levels = if spec.contains('[') {
        // Factor::parse requires two levels; a one-level discrete role is valid.
        let expanded = spec.split_once('[').and_then(|(kind, rest)| {
            if !matches!(kind.trim(), "d" | "discrete") {
                return None;
            }
            rest.strip_suffix(']')
                .filter(|args| !args.trim().is_empty())
                .map(|args| format!("d[{},{}]", args, args))
        });
        if let Some(expanded) = expanded {
            let mut values = Factor::parse(name, &expanded)?.values;
            values.truncate(values.len() / 2);
            values
        } else {
            Factor::parse(name, spec)?.values
        }
    } else {
        let count: i64 = spec
            .parse()
            .with_context(|| format!("invalid replicate count '{}'", spec))?;
        if count < 1 {
            bail!("replicate count must be positive");
        }
        (1..=count).map(FactorValue::Int).collect()
    };
    for (i, value) in levels.iter().enumerate() {
        if levels[..i].contains(value) {
            bail!("duplicate level '{}' for replicate role '{}'", value, name);
        }
    }
    Ok(ReplicateRole {
        name: name.to_owned(),
        levels,
    })
}

fn parse_unit(src: &str, factors: &[Factor], roles: &[ReplicateRole]) -> Result<Unit> {
    let (name, keys) = src
        .split_once('=')
        .ok_or_else(|| anyhow::anyhow!("expected 'name = key1, key2, …'"))?;
    let name = name.trim();
    if !is_valid_name(name) {
        bail!("invalid unit name '{}'", name);
    }
    let mut key = Vec::new();
    for name in keys.split(',').map(str::trim) {
        if !factors.iter().any(|f| f.name == name) && !roles.iter().any(|r| r.name == name) {
            bail!("unknown unit key '{}'", name);
        }
        if key.iter().any(|k| k == name) {
            bail!("duplicate unit key '{}'", name);
        }
        key.push(name.to_string());
    }
    Ok(Unit {
        name: name.to_string(),
        key,
    })
}

enum Section {
    Factors,
    Results,
    Model,
    Replicates,
    Units,
}

fn parse_factor_line(line: &str) -> Result<Factor> {
    let eq = line
        .find('=')
        .ok_or_else(|| anyhow::anyhow!("expected 'name = spec'"))?;
    let name = line[..eq].trim();
    let spec = line[eq + 1..].trim();
    if !is_valid_name(name) {
        bail!("invalid factor name '{}'", name);
    }
    Factor::parse(name, spec)
}

fn is_valid_name(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
}

/// Select the design, print the construct-time report, then write the v2 CSV
/// and its sidecar manifest. Nothing is written before the report prints.
fn write_design(parsed: &ParsedDesign, opts: &ConstructOpts, output: &Path) -> Result<()> {
    let ParsedDesign {
        factors,
        results,
        models,
        replicates,
        unit,
    } = parsed;
    let selection = design_select::select(
        factors,
        models,
        replicates,
        unit.as_ref(),
        &opts.select_opts(),
    )?;
    eprintln!();
    eprint!("{}", selection.report);

    let base_rows = selection.selected.array.len();
    let n_runs = selection.cells.len();
    eprintln!();
    eprintln!(
        "Building {}: {} runs × {} factor columns",
        selection.selected.info.label,
        n_runs,
        factors.len(),
    );
    eprintln!("Method: {}", selection.selected.info.method);

    // Sanity-check the chosen array before writing it out.
    if let Err(e) = verify_strength2(&selection.selected.array, &selection.selected.column_levels) {
        eprintln!("⚠ strength-2 verification failed: {:#}", e);
    }

    // Unit ids follow first appearance in design order, which is also the
    // order the model matrix enumerates unit identities in.
    let unit_ids = unit_ids(unit.as_ref(), factors, replicates, &selection.cells);

    // Execution order: Fisher–Yates inside each replicate block, blocks in
    // level order. The cell list is block-major, so the blocks are the
    // consecutive chunks of `base_rows` cells.
    let seed = opts.seed.unwrap_or_else(default_seed);
    let mut rng = Rng::new(seed);
    let mut execution: Vec<usize> = Vec::with_capacity(n_runs);
    for block in 0..n_runs / base_rows {
        let mut cells: Vec<usize> = (block * base_rows..(block + 1) * base_rows).collect();
        if !opts.no_shuffle {
            rng.shuffle(&mut cells);
        }
        execution.extend(cells);
    }

    let width = n_runs.to_string().len().max(4);
    let runs: Vec<Run> = execution
        .iter()
        .enumerate()
        .map(|(position, &cell)| {
            let c = &selection.cells[cell];
            Run {
                run_id: format!("r{:0width$}", cell + 1),
                order: position + 1,
                design_row: cell % base_rows,
                factors: factors
                    .iter()
                    .zip(&c.factor_levels)
                    .map(|(f, &level)| (f.name.clone(), f.values[level].clone()))
                    .collect(),
                replicates: replicates
                    .iter()
                    .zip(&c.replicate_levels)
                    .map(|(r, &level)| (r.name.clone(), r.levels[level].clone()))
                    .collect(),
                unit: unit_ids[cell].clone(),
            }
        })
        .collect();

    // CSV v2: run_id, order, factors…, replicate roles…, [unit], results…
    let mut wtr =
        csv::Writer::from_path(output).with_context(|| format!("creating {}", output.display()))?;
    let mut header = vec!["run_id".to_string(), "order".to_string()];
    header.extend(factors.iter().map(|f| f.name.clone()));
    header.extend(replicates.iter().map(|r| r.name.clone()));
    if let Some(unit) = unit {
        header.push(unit.name.clone());
    }
    header.extend(results.iter().cloned());
    wtr.write_record(&header)?;
    for run in &runs {
        let mut rec: Vec<String> = Vec::with_capacity(header.len());
        rec.push(run.run_id.clone());
        rec.push(run.order.to_string());
        for f in factors {
            rec.push(run.factors[&f.name].to_string());
        }
        for r in replicates {
            rec.push(run.replicates[&r.name].to_string());
        }
        if unit.is_some() {
            rec.push(run.unit.clone().unwrap_or_default());
        }
        for _ in results {
            rec.push(String::new());
        }
        wtr.write_record(&rec)?;
    }
    wtr.flush()?;
    eprintln!("✓ wrote {}", output.display());

    let design = Design {
        version: two(),
        models: models.clone(),
        replicates: replicates.clone(),
        unit: unit.clone(),
        randomization_seed: Some(seed),
        estimability: Some(selection.estimability),
        runs,
        augmentations: Vec::new(),
        factors: factors.clone(),
        results: results.clone(),
        array: selection.selected.info,
    };
    let design_path = sidecar_path(output);
    let mut f = File::create(&design_path)
        .with_context(|| format!("creating {}", design_path.display()))?;
    serde_json::to_writer_pretty(&mut f, &design)?;
    writeln!(f)?;
    eprintln!("✓ wrote {}", design_path.display());

    Ok(())
}

/// One unit id per cell, `u0001`… by first appearance of the key values.
/// Every entry is `None` when the design declares no unit.
fn unit_ids(
    unit: Option<&Unit>,
    factors: &[Factor],
    replicates: &[ReplicateRole],
    cells: &[Cell],
) -> Vec<Option<String>> {
    let Some(unit) = unit else {
        return vec![None; cells.len()];
    };
    let mut seen: BTreeMap<Vec<usize>, String> = BTreeMap::new();
    cells
        .iter()
        .map(|cell| {
            let key: Vec<usize> = unit
                .key
                .iter()
                .map(|name| match factors.iter().position(|f| &f.name == name) {
                    Some(f) => cell.factor_levels[f],
                    None => {
                        let r = replicates
                            .iter()
                            .position(|r| &r.name == name)
                            .expect("unit keys are validated at parse time");
                        cell.replicate_levels[r]
                    }
                })
                .collect();
            let next = format!("u{:04}", seen.len() + 1);
            Some(seen.entry(key).or_insert(next).clone())
        })
        .collect()
}

/// Nanoseconds since the epoch. The value is recorded in the manifest, so a
/// clock that predates the epoch costs reproducibility of nothing.
fn default_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64)
}

pub fn sidecar_path(csv_path: &Path) -> PathBuf {
    let stem = csv_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("design");
    let parent = csv_path.parent().unwrap_or_else(|| Path::new("."));
    parent.join(format!("{}.design.json", stem))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oa::{build_array, resolve_cell};
    use std::collections::BTreeSet;
    use std::io::Cursor;

    fn parse(src: &str) -> Result<ParsedDesign> {
        parse_design_file(Cursor::new(src), Path::new("fixture.design"))
    }
    const OLD: &str = "[factors]\na = d[1,2]\nb = d[1,2]\n[results]\ny\n";

    #[test]
    fn legacy_and_new_sections() {
        let old = parse(OLD).unwrap();
        assert!(old.models.is_empty() && old.replicates.is_empty() && old.unit.is_none());
        let new = parse(&format!("{OLD}[model]\ny ~ b*a # comment\n[replicates]\nseed = 2 # count\nfold = d[11,22,33]\n[units]\ncheckpoint = a, seed\n")).unwrap();
        assert_eq!(new.models[0].formula, "y ~ a + b + a:b");
        assert_eq!(
            new.replicates[0].levels,
            [FactorValue::Int(1), FactorValue::Int(2)]
        );
        assert_eq!(
            new.replicates[1].levels,
            [
                FactorValue::Int(11),
                FactorValue::Int(22),
                FactorValue::Int(33)
            ]
        );
        assert_eq!(new.unit.unwrap().key, ["a", "seed"]);
        // Formula and unit references resolve after all sections have been read.
        let forward = parse("[units]\nu = a, seed\n[model]\n. ~ a\n[replicates]\nseed = 1\n[factors]\na = d[1,2]\n[results]\ny\n").unwrap();
        assert_eq!(forward.replicates[0].levels, [FactorValue::Int(1)]);
        assert!(forward.models[0].response.is_none());
        for spec in ["d[11]", "d [11]", "discrete[11]", "1"] {
            assert_eq!(
                parse_replicate(&format!("seed = {spec}"))
                    .unwrap()
                    .levels
                    .len(),
                1
            );
        }
        assert_eq!(
            parse_replicate("seed = uniform[0,1,3]")
                .unwrap()
                .levels
                .len(),
            3
        );
        let labels = parse(&format!(
            "{OLD}[replicates]\nseed = d[\"a#b\",\"c\"] # comment\n"
        ))
        .unwrap();
        assert_eq!(
            labels.replicates[0].levels[0],
            FactorValue::Text("a#b".into())
        );
    }

    #[test]
    fn section_errors_name_file_and_line() {
        for (suffix, line, message) in [
            ("[model]\na + unknown\n", 7, "unknown"),
            (
                "[model]\na + seed\n[replicates]\nseed = 2\n",
                7,
                "replicate role 'seed'",
            ),
            ("[model]\nz ~ a\n", 7, "unknown result name 'z'"),
            ("[model]\na -1\n", 7, "unsupported"),
            ("[replicates]\nseed = 0\n", 7, "positive"),
            ("[replicates]\nseed = bad\n", 7, "invalid replicate count"),
            ("[replicates]\na = 2\n", 7, "duplicate"),
            ("[units]\nu = a, missing\n", 7, "missing"),
            ("[units]\nu = a\nv = b\n", 8, "at most one"),
            ("[units]\nu = a\n[units]\nv = b\n", 9, "at most one"),
            ("[units]\nu =\n", 7, "unknown unit key"),
            ("[replicates]\nseed = d[1,1]\n", 7, "duplicate level"),
        ] {
            let err = match parse(&format!("{OLD}{suffix}")) {
                Ok(_) => panic!("accepted {suffix}"),
                Err(e) => format!("{e:#}"),
            };
            assert!(err.contains(&format!("fixture.design:{line}")), "{err}");
            assert!(err.contains(message), "{err}");
        }
    }

    #[test]
    fn sidecar_v1_defaults_and_v2_round_trip() {
        let v1 = serde_json::json!({ "factors": [], "results": ["y"],
            "array": {"rows":4,"cols":2,"label":"L4","method":"test"} });
        let mut design: Design = serde_json::from_value(v1).unwrap();
        assert_eq!(design.version, 2);
        assert!(design.models.is_empty() && design.replicates.is_empty() && design.runs.is_empty());
        assert!(
            design.unit.is_none()
                && design.randomization_seed.is_none()
                && design.estimability.is_none()
        );
        assert!(design.augmentations.is_empty());
        design.factors = vec![Factor::parse("a", "d[1,2]").unwrap()];
        design.models = vec![Model::main_effects(&design.factors)];
        design.replicates = vec![parse_replicate("seed = 2").unwrap()];
        design.unit = Some(Unit {
            name: "checkpoint".into(),
            key: vec!["a".into(), "seed".into()],
        });
        design.randomization_seed = Some(42);
        let mm = crate::model::model_matrix(&design.factors, &[], &design.models[0], &[], None);
        design.estimability = Some(crate::model::estimability(
            &mm,
            &design.models[0],
            &design.factors,
        ));
        design.runs = vec![Run {
            run_id: "r0001".into(),
            order: 1,
            design_row: 0,
            factors: BTreeMap::from([("a".into(), FactorValue::Int(1))]),
            replicates: BTreeMap::from([("seed".into(), FactorValue::Int(1))]),
            unit: Some("u001".into()),
        }];
        design.augmentations = vec![Augmentation {
            seed: 42,
            targets: vec!["a".into()],
            run_ids: vec!["r0002".into()],
        }];
        let value = serde_json::to_value(design).unwrap();
        let loaded: Design = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(loaded).unwrap(), value);
    }

    /// Run construct on a design-file source in a fresh temp dir. Returns the
    /// output path, the sidecar, and the CSV text; the caller removes the dir.
    fn construct_to_temp(
        name: &str,
        src: &str,
        opts: &ConstructOpts,
    ) -> Result<(PathBuf, Design, String)> {
        let dir = std::env::temp_dir().join(format!("taguchi-w2-{}-{}", std::process::id(), name));
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("input.txt");
        let output = dir.join("exp.csv");
        std::fs::write(&input, src).unwrap();
        run_construct_from_file(input, output.clone(), opts)?;
        let design: Design =
            serde_json::from_reader(File::open(sidecar_path(&output)).unwrap()).unwrap();
        let csv = std::fs::read_to_string(&output).unwrap();
        Ok((output, design, csv))
    }

    fn cleanup(output: &Path) {
        std::fs::remove_dir_all(output.parent().unwrap()).unwrap();
    }

    fn source(specs: &[(&str, &str)], extra: &str) -> String {
        let factors: Vec<String> = specs.iter().map(|(n, s)| format!("{n} = {s}")).collect();
        format!("[factors]\n{}\n[results]\ny\n{extra}", factors.join("\n"))
    }

    /// Outcome 2: with no model, no replicates and no unit the design is
    /// exactly the array `build_array` produces — same rows, same order.
    #[test]
    fn no_model_reproduces_the_v1_array() {
        let cases: [&[(&str, &str)]; 3] = [
            &[
                ("a", "d[1,2]"),
                ("b", "d[1,2]"),
                ("c", "d[1,2]"),
                ("d", "d[1,2]"),
            ],
            &[
                ("a", "d[1-3]"),
                ("b", "d[1-3]"),
                ("c", "d[1-3]"),
                ("d", "d[1-3]"),
            ],
            &[("a", "d[1,2]"), ("b", "d[1-3]"), ("c", "d[1-4]")],
        ];
        for (i, specs) in cases.iter().enumerate() {
            let factors: Vec<Factor> = specs
                .iter()
                .map(|(n, s)| Factor::parse(n, s).unwrap())
                .collect();
            let expected = build_array(&factors).unwrap();
            let (output, design, _) = construct_to_temp(
                &format!("v1-{i}"),
                &source(specs, ""),
                &ConstructOpts::default(),
            )
            .unwrap();
            assert_eq!(design.array.rows, expected.rows());
            assert_eq!(design.array.label, expected.info.label);
            assert_eq!(design.runs.len(), expected.rows());
            let mut seen = vec![false; expected.rows()];
            for run in &design.runs {
                assert!(!std::mem::replace(&mut seen[run.design_row], true));
                assert_eq!(run.run_id, format!("r{:04}", run.design_row + 1));
                for (j, f) in factors.iter().enumerate() {
                    assert_eq!(
                        run.factors[&f.name],
                        resolve_cell(f, expected.array[run.design_row][j])
                    );
                }
            }
            cleanup(&output);
        }
    }

    /// Outcomes 6 and 8: replicate blocks, unit ids by first appearance, the
    /// v2 CSV layout, and CSV row k agreeing with `runs[k]`.
    #[test]
    fn v2_csv_and_manifest_agree() {
        let specs: &[(&str, &str)] = &[("a", "d[1,2]"), ("b", "d[1,2]"), ("c", "d[1,2]")];
        let src = source(
            specs,
            "[model]\na*b + c\n[replicates]\nseed = 2\n[units]\nchk = a, b, seed\n",
        );
        let (output, design, csv) =
            construct_to_temp("v2-csv", &src, &ConstructOpts::default()).unwrap();
        let base = design.array.rows;
        assert_eq!(design.runs.len(), base * 2);
        assert!(design.randomization_seed.is_some());
        assert!(design.estimability.is_some());

        let mut rdr = csv::Reader::from_reader(csv.as_bytes());
        let mut header = vec!["run_id", "order", "a", "b", "c", "seed", "chk", "y"];
        assert_eq!(
            rdr.headers().unwrap(),
            &csv::StringRecord::from(header.split_off(0))
        );
        let mut orders = Vec::new();
        for (k, record) in rdr.records().enumerate() {
            let record = record.unwrap();
            let run = &design.runs[k];
            assert_eq!(&record[0], run.run_id);
            assert_eq!(record[1].parse::<usize>().unwrap(), run.order);
            for (j, (name, _)) in specs.iter().enumerate() {
                assert_eq!(record[2 + j], run.factors[*name].to_string());
            }
            assert_eq!(record[5], run.replicates["seed"].to_string());
            assert_eq!(&record[6], run.unit.as_ref().unwrap());
            assert_eq!(&record[7], "");
            orders.push(run.order);
        }
        orders.sort_unstable();
        assert_eq!(orders, (1..=design.runs.len()).collect::<Vec<_>>());

        // One unit per (a, b, seed) combination, ids by first appearance.
        let units: BTreeSet<&String> = design.runs.iter().filter_map(|r| r.unit.as_ref()).collect();
        assert_eq!(units.len(), 8);
        assert!(units.contains(&"u0001".to_string()));
        assert!(units.contains(&format!("u{:04}", units.len())));
        // Rows that share a unit share its key values.
        for run in &design.runs {
            for other in &design.runs {
                if run.unit == other.unit {
                    assert_eq!(run.factors["a"], other.factors["a"]);
                    assert_eq!(run.factors["b"], other.factors["b"]);
                    assert_eq!(run.replicates["seed"], other.replicates["seed"]);
                }
            }
        }
        cleanup(&output);
    }

    /// Outcome 7: the seed decides execution order; `--no-shuffle` keeps
    /// design order; blocks stay whole.
    #[test]
    fn execution_order_follows_the_seed() {
        let specs: &[(&str, &str)] = &[
            ("a", "d[1,2]"),
            ("b", "d[1,2]"),
            ("c", "d[1,2]"),
            ("d", "d[1,2]"),
        ];
        let src = source(specs, "[replicates]\nseed = 2\n");
        let order_for = |name: &str, opts: &ConstructOpts| {
            let (output, design, _) = construct_to_temp(name, &src, opts).unwrap();
            let order: Vec<String> = design.runs.iter().map(|r| r.run_id.clone()).collect();
            let seed = design.randomization_seed.unwrap();
            let rows = design.array.rows;
            cleanup(&output);
            (order, seed, rows)
        };
        let seeded = |seed: u64| ConstructOpts {
            seed: Some(seed),
            ..ConstructOpts::default()
        };
        let (a, _, rows) = order_for("shuffle-a", &seeded(7));
        let (b, seed_b, _) = order_for("shuffle-b", &seeded(7));
        let (c, _, _) = order_for("shuffle-c", &seeded(8));
        assert_eq!(a.len(), 16, "a 16-row design");
        assert_eq!(a, b, "same seed → same order");
        assert_eq!(seed_b, 7, "the seed is recorded");
        assert_ne!(a, c, "different seed → different order");
        let mut sorted = a.clone();
        sorted.sort();
        assert_ne!(a, sorted, "the default order is shuffled");
        // Blocks stay whole: the first `rows` runs are the first replicate.
        let (plain, _, _) = order_for(
            "no-shuffle",
            &ConstructOpts {
                no_shuffle: true,
                seed: Some(7),
                ..ConstructOpts::default()
            },
        );
        assert_eq!(plain, sorted, "--no-shuffle → order = design order");
        for (block, chunk) in a.chunks(rows).enumerate() {
            for run_id in chunk {
                let cell: usize = run_id[1..].parse().unwrap();
                assert_eq!((cell - 1) / rows, block, "replicate blocks stay whole");
            }
        }
    }

    /// Outcome 9: `--model`, `--replicate` and `--unit` merge after the file's
    /// own sections and take the same validation.
    #[test]
    fn cli_options_merge_with_the_file() {
        let specs: &[(&str, &str)] = &[("a", "d[1,2]"), ("b", "d[1,2]")];
        let opts = ConstructOpts {
            models: vec!["y ~ a*b".into()],
            replicates: vec!["seed=2".into()],
            unit: Some("chk=a,seed".into()),
            seed: Some(1),
            ..ConstructOpts::default()
        };
        let (output, design, csv) =
            construct_to_temp("cli-merge", &source(specs, ""), &opts).unwrap();
        assert_eq!(design.models[0].formula, "y ~ a + b + a:b");
        assert_eq!(design.replicates[0].name, "seed");
        assert_eq!(design.unit.as_ref().unwrap().key, ["a", "seed"]);
        assert!(csv.starts_with("run_id,order,a,b,seed,chk,y\n"));
        cleanup(&output);

        // The file's lines come first, then the CLI's.
        let (output, design, _) = construct_to_temp(
            "cli-append",
            &source(specs, "[model]\ny ~ a\n[replicates]\nfold = 2\n"),
            &ConstructOpts {
                models: vec![". ~ b".into()],
                replicates: vec!["seed=2".into()],
                seed: Some(1),
                ..ConstructOpts::default()
            },
        )
        .unwrap();
        assert_eq!(design.models[0].formula, "y ~ a");
        assert_eq!(design.models[1].formula, ". ~ b");
        assert_eq!(design.replicates[0].name, "fold");
        assert_eq!(design.replicates[1].name, "seed");
        cleanup(&output);

        for (name, opts, message) in [
            (
                "cli-dup-model",
                ConstructOpts {
                    models: vec!["y ~ a".into(), "y ~ b".into()],
                    ..ConstructOpts::default()
                },
                "duplicate model",
            ),
            (
                "cli-role-in-model",
                ConstructOpts {
                    models: vec!["y ~ a + seed".into()],
                    replicates: vec!["seed=2".into()],
                    ..ConstructOpts::default()
                },
                "replicate role 'seed'",
            ),
            (
                "cli-dup-role",
                ConstructOpts {
                    replicates: vec!["a=2".into()],
                    ..ConstructOpts::default()
                },
                "duplicate factor or replicate-role name",
            ),
            (
                "cli-unknown-key",
                ConstructOpts {
                    unit: Some("chk=missing".into()),
                    ..ConstructOpts::default()
                },
                "unknown unit key",
            ),
        ] {
            let err = construct_to_temp(name, &source(specs, ""), &opts)
                .map(|_| ())
                .unwrap_err();
            assert!(format!("{err:#}").contains(message), "{err:#}");
            std::fs::remove_dir_all(std::env::temp_dir().join(format!(
                "taguchi-w2-{}-{}",
                std::process::id(),
                name
            )))
            .unwrap();
        }
        let err = construct_to_temp(
            "cli-two-units",
            &source(specs, "[units]\nu = a\n"),
            &ConstructOpts {
                unit: Some("chk=a".into()),
                ..ConstructOpts::default()
            },
        )
        .map(|_| ())
        .unwrap_err();
        assert!(format!("{err:#}").contains("at most one"), "{err:#}");
        std::fs::remove_dir_all(
            std::env::temp_dir().join(format!("taguchi-w2-{}-cli-two-units", std::process::id())),
        )
        .unwrap();
    }
}
