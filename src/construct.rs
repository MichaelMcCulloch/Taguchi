use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use crate::factor::{Factor, FactorValue};
pub use crate::model::EstimabilityReport;
use crate::model::{Model, ReplicateRole, Unit, formula_names, hierarchy_warnings};
use crate::oa::{ArrayInfo, build_array, resolve_cell, verify_strength2};

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

pub fn run_construct(output: PathBuf) -> Result<()> {
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

    write_design(&factors, &results, &[], &[], None, &output)
}

pub fn run_construct_from_file(input: PathBuf, output: PathBuf) -> Result<()> {
    let file = File::open(&input).with_context(|| format!("opening {}", input.display()))?;
    let parsed = parse_design_file(BufReader::new(file), &input)?;
    write_design(
        &parsed.factors,
        &parsed.results,
        &parsed.models,
        &parsed.replicates,
        parsed.unit.as_ref(),
        &output,
    )
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
        let model = (|| -> Result<Model> {
            for name in formula_names(&src)? {
                if replicates.iter().any(|r| r.name == name) {
                    bail!("replicate role '{}' cannot appear in a model formula", name);
                }
            }
            let model = Model::parse(&src, &factors)?;
            if let Some(response) = &model.response
                && !results.contains(response)
            {
                bail!("unknown result name '{}'", response);
            }
            if models.iter().any(|m: &Model| m.response == model.response) {
                bail!(
                    "duplicate model for response '{}'",
                    model.response.as_deref().unwrap_or(".")
                );
            }
            Ok(model)
        })()
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

fn write_design(
    factors: &[Factor],
    results: &[String],
    models: &[Model],
    replicates: &[ReplicateRole],
    unit: Option<&Unit>,
    output: &Path,
) -> Result<()> {
    let mut warned = BTreeSet::new();
    for model in models {
        for warning in hierarchy_warnings(model, factors) {
            if warned.insert(warning.clone()) {
                eprintln!("warning: {}", warning);
            }
        }
    }
    let selected = build_array(factors)?;
    let n_runs = selected.rows();
    eprintln!();
    eprintln!(
        "Building {}: {} runs × {} factor columns",
        selected.info.label,
        n_runs,
        factors.len(),
    );
    eprintln!("Method: {}", selected.info.method);

    // Sanity-check the chosen array before writing it out.
    if let Err(e) = verify_strength2(&selected.array, &selected.column_levels) {
        eprintln!("⚠ strength-2 verification failed: {:#}", e);
    }

    // CSV: experiment, factor1, factor2, ..., result1, result2, ...
    let mut wtr =
        csv::Writer::from_path(output).with_context(|| format!("creating {}", output.display()))?;
    let mut header = vec!["experiment".to_string()];
    for f in factors {
        header.push(f.name.clone());
    }
    for r in results {
        header.push(r.clone());
    }
    wtr.write_record(&header)?;

    for (i, row) in selected.array.iter().enumerate() {
        let mut rec: Vec<String> = Vec::with_capacity(header.len());
        rec.push((i + 1).to_string());
        for (j, &cell) in row.iter().enumerate() {
            rec.push(resolve_cell(&factors[j], cell).to_string());
        }
        for _ in results {
            rec.push(String::new());
        }
        wtr.write_record(&rec)?;
    }
    wtr.flush()?;
    eprintln!("✓ wrote {}", output.display());

    let width = n_runs.to_string().len().max(4);
    let runs = selected
        .array
        .iter()
        .enumerate()
        .map(|(i, row)| Run {
            run_id: format!("r{:0width$}", i + 1),
            order: i + 1,
            design_row: i,
            factors: factors
                .iter()
                .zip(row)
                .map(|(f, &cell)| (f.name.clone(), resolve_cell(f, cell)))
                .collect(),
            replicates: BTreeMap::new(),
            unit: None,
        })
        .collect();
    let design = Design {
        version: two(),
        models: models.to_vec(),
        replicates: replicates.to_vec(),
        unit: unit.cloned(),
        randomization_seed: None,
        estimability: None,
        runs,
        augmentations: Vec::new(),
        factors: factors.to_vec(),
        results: results.to_vec(),
        array: selected.info,
    };
    let design_path = sidecar_path(output);
    let mut f = File::create(&design_path)
        .with_context(|| format!("creating {}", design_path.display()))?;
    serde_json::to_writer_pretty(&mut f, &design)?;
    writeln!(f)?;
    eprintln!("✓ wrote {}", design_path.display());

    Ok(())
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
            ("[model]\na %in% b\n", 7, "%in%"),
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

    #[test]
    fn file_entry_point_preserves_csv_and_writes_plain_runs() {
        let dir = std::env::temp_dir().join(format!("taguchi-phase1-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("input.txt");
        let output = dir.join("output.csv");
        std::fs::write(
            &input,
            format!("{OLD}[model]\na*b\n[replicates]\nseed = 3\n[units]\nu = a, seed\n"),
        )
        .unwrap();
        run_construct_from_file(input.clone(), output.clone()).unwrap();
        let design: Design =
            serde_json::from_reader(File::open(sidecar_path(&output)).unwrap()).unwrap();
        let selected = build_array(&design.factors).unwrap();
        assert_eq!(design.array.rows, selected.rows());
        assert_eq!(design.runs.len(), selected.rows());
        assert_eq!(design.models[0].formula, ". ~ a + b + a:b");
        assert_eq!(design.replicates[0].levels.len(), 3);
        assert_eq!(design.unit.as_ref().unwrap().key, ["a", "seed"]);
        let csv_new = std::fs::read_to_string(&output).unwrap();
        let mut csv = csv::Reader::from_reader(csv_new.as_bytes());
        assert_eq!(
            csv.headers().unwrap(),
            &csv::StringRecord::from(vec!["experiment", "a", "b", "y"])
        );
        for (i, record) in csv.records().enumerate() {
            let record = record.unwrap();
            let run = &design.runs[i];
            assert_eq!(run.run_id, format!("r{:04}", i + 1));
            assert_eq!((run.order, run.design_row), (i + 1, i));
            assert!(run.replicates.is_empty() && run.unit.is_none());
            for (j, factor) in design.factors.iter().enumerate() {
                assert_eq!(
                    run.factors[&factor.name],
                    resolve_cell(factor, selected.array[i][j])
                );
                assert_eq!(run.factors[&factor.name].to_string(), record[j + 1]);
            }
            assert_eq!(&record[3], "");
        }
        std::fs::write(&input, OLD).unwrap();
        run_construct_from_file(input, output.clone()).unwrap();
        assert_eq!(std::fs::read_to_string(output).unwrap(), csv_new);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
