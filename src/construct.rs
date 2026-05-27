use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use crate::factor::Factor;
use crate::oa::{ArrayInfo, build_array, resolve_cell, verify_strength2};

#[derive(Debug, Serialize, Deserialize)]
pub struct Design {
    pub factors: Vec<Factor>,
    pub results: Vec<String>,
    pub array: ArrayInfo,
}

pub fn run_construct(output: PathBuf) -> Result<()> {
    let stdin = std::io::stdin();
    let mut reader = stdin.lock();
    let mut prompt_line = String::new();

    eprintln!("=== taguchi construct ===");
    eprintln!("Enter factors one per line as 'name = spec'.");
    eprintln!("Specs: d[1,2-4,9]  uniform[0,10,5]  normal[0,10,5,2]  logLow[1,1000,4]  logHigh[1,1000,4]");
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

    write_design(&factors, &results, &output)
}

pub fn run_construct_from_file(input: PathBuf, output: PathBuf) -> Result<()> {
    let file = File::open(&input).with_context(|| format!("opening {}", input.display()))?;
    let reader = BufReader::new(file);
    let mut factors = Vec::new();
    let mut results = Vec::new();
    let mut section = Section::Factors;
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        let s = line.trim();
        if s.is_empty() || s.starts_with('#') {
            continue;
        }
        if s.eq_ignore_ascii_case("[factors]") {
            section = Section::Factors;
            continue;
        }
        if s.eq_ignore_ascii_case("[results]") {
            section = Section::Results;
            continue;
        }
        match section {
            Section::Factors => {
                let f = parse_factor_line(s)
                    .with_context(|| format!("{}:{}", input.display(), lineno + 1))?;
                factors.push(f);
            }
            Section::Results => {
                if !is_valid_name(s) {
                    bail!("{}:{}: invalid result name '{}'", input.display(), lineno + 1, s);
                }
                results.push(s.to_string());
            }
        }
    }
    if factors.is_empty() {
        bail!("no factors in {}", input.display());
    }
    if results.is_empty() {
        bail!("no result columns in {} (use a [results] section)", input.display());
    }
    write_design(&factors, &results, &output)
}

enum Section {
    Factors,
    Results,
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

fn write_design(factors: &[Factor], results: &[String], output: &Path) -> Result<()> {
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
    let mut wtr = csv::Writer::from_path(output)
        .with_context(|| format!("creating {}", output.display()))?;
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

    let design = Design {
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
    let stem = csv_path.file_stem().and_then(|s| s.to_str()).unwrap_or("design");
    let parent = csv_path.parent().unwrap_or_else(|| Path::new("."));
    parent.join(format!("{}.design.json", stem))
}
