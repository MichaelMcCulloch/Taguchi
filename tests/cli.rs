//! End-to-end tests that drive the built `taguchi` binary.
//!
//! Every test spawns `env!("CARGO_BIN_EXE_taguchi")` as a child process, so it
//! exercises the real CLI surface: argument parsing, the printed reports, the
//! files written to disk and the process exit code. Nothing here reaches into
//! the library, which is what the unit tests in `src/` already cover.
//!
//! The scenario in `a`–`d` is the customer request quoted in
//! `docs/SPEC-model-driven-design.md`: four factors, the formula
//! `response ~ denominator * projection + prior * evaluation_mode`, three
//! training seeds as a replicate role, and a `checkpoint` unit that pairs the
//! evaluation modes sharing one trained checkpoint.
//!
//! `--bootstrap 0` is passed to every `analyze` call. The default of 1000
//! resamples costs about 140 s per call in a debug build; the bootstrap itself
//! has its own unit tests in `analyze.rs`.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

const BIN: &str = env!("CARGO_BIN_EXE_taguchi");

/// The customer's design file, verbatim from the spec's grammar example.
const LANDER_DESIGN: &str = r#"[factors]
denominator     = d["a","b"]
projection      = d["p","q","r"]
prior           = d["flat","learned"]
evaluation_mode = d["greedy","beam","sample"]

[model]
response ~ denominator * projection + prior * evaluation_mode

[replicates]
seed = 3

[units]
checkpoint = denominator, projection, prior, seed

[results]
response
"#;

/// The four factor names, in declaration order.
const LANDER_FACTORS: [&str; 4] = ["denominator", "projection", "prior", "evaluation_mode"];

/// Every term the customer's formula expands to, in the order the reports
/// print them.
const LANDER_TERMS: [&str; 6] = [
    "denominator",
    "projection",
    "prior",
    "evaluation_mode",
    "denominator:projection",
    "prior:evaluation_mode",
];

// ------------------------------------------------------------- harness ----

/// A scratch directory that deletes itself. Named after the test that owns it
/// so a leftover directory says where it came from.
struct Scratch {
    path: PathBuf,
}

impl Scratch {
    fn new(tag: &str) -> Scratch {
        static N: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "taguchi-cli-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create scratch dir");
        Scratch { path }
    }

    fn join(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }

    /// Write `body` to `name` inside the scratch dir and return the path.
    fn write(&self, name: &str, body: &str) -> PathBuf {
        let p = self.join(name);
        fs::write(&p, body).expect("write scratch file");
        p
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn run(args: &[&str]) -> Output {
    Command::new(BIN)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("spawning {BIN}: {e}"))
}

fn code(out: &Output) -> i32 {
    out.status.code().expect("child exited via signal")
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// Panic with both streams attached; a bare `assert_eq!` on the exit code hides
/// the reason the binary refused.
fn expect_code(out: &Output, want: i32, what: &str) {
    assert_eq!(
        code(out),
        want,
        "{what}: wanted exit {want}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        stdout(out),
        stderr(out)
    );
}

fn read_json(path: &Path) -> serde_json::Value {
    let body = fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {path:?}: {e}"));
    serde_json::from_str(&body).unwrap_or_else(|e| panic!("parsing {path:?}: {e}"))
}

fn s(path: &Path) -> &str {
    path.to_str().expect("scratch path is utf-8")
}

// ------------------------------------------------------- the CSV, typed ----

/// A parsed v2 experiment CSV. The generated cells hold no commas or quotes, so
/// a split on ',' is exact for these files.
struct Table {
    header: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl Table {
    fn parse(text: &str) -> Table {
        let mut lines = text.lines();
        let header: Vec<String> = lines
            .next()
            .expect("csv has a header")
            .split(',')
            .map(str::to_owned)
            .collect();
        let rows = lines
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.split(',').map(str::to_owned).collect())
            .collect();
        Table { header, rows }
    }

    fn col(&self, name: &str) -> usize {
        self.header
            .iter()
            .position(|c| c == name)
            .unwrap_or_else(|| panic!("column {name} in {:?}", self.header))
    }

    fn get<'a>(&self, row: &'a [String], name: &str) -> &'a str {
        row[self.col(name)].as_str()
    }

    fn render(&self) -> String {
        let mut out = self.header.join(",");
        out.push('\n');
        for r in &self.rows {
            out.push_str(&r.join(","));
            out.push('\n');
        }
        out
    }
}

// --------------------------------------------------- the known response ----

// Sum-to-zero level scores. The response below is linear in these, so the
// true optimum is exactly recoverable and the ANOVA has a known answer.
fn den(v: &str) -> f64 {
    if v == "b" { 1.0 } else { -1.0 }
}
fn proj(v: &str) -> f64 {
    match v {
        "p" => -1.0,
        "q" => 0.0,
        _ => 1.0,
    }
}
fn pri(v: &str) -> f64 {
    if v == "learned" { 1.0 } else { -1.0 }
}
fn mode(v: &str) -> f64 {
    match v {
        "greedy" => -1.0,
        "beam" => 0.0,
        _ => 1.0,
    }
}

/// The truth the experiment is meant to discover. It carries both requested
/// interactions; `denominator:projection` is the largest single effect after
/// `denominator` itself.
fn truth(d: &str, p: &str, pr: &str, m: &str) -> f64 {
    50.0 + 4.0 * den(d)
        + 2.0 * proj(p)
        + 1.5 * pri(pr)
        + 2.0 * mode(m)
        + 3.0 * den(d) * proj(p)
        + 1.0 * pri(pr) * mode(m)
}

/// Per-training-seed offset: the block effect the replicate role must absorb.
fn seed_offset(seed: &str) -> f64 {
    match seed {
        "1" => -0.8,
        "2" => 0.0,
        _ => 0.8,
    }
}

/// Deterministic measurement noise, ±0.08 at most. Small against the effects,
/// large enough that RSS is non-zero so the F-tests have a denominator.
fn noise(i: usize) -> f64 {
    0.01 * (((i * 37) % 17) as f64 - 8.0)
}

/// The grid cell that maximises `truth`, found by search rather than asserted,
/// so the test cannot agree with itself about a wrong optimum.
fn true_optimum() -> BTreeMap<String, String> {
    let mut best: Option<(f64, [&str; 4])> = None;
    for d in ["a", "b"] {
        for p in ["p", "q", "r"] {
            for pr in ["flat", "learned"] {
                for m in ["greedy", "beam", "sample"] {
                    let v = truth(d, p, pr, m);
                    if best.is_none_or(|(bv, _)| v > bv) {
                        best = Some((v, [d, p, pr, m]));
                    }
                }
            }
        }
    }
    let (_, cell) = best.expect("the grid is not empty");
    [
        ("denominator", cell[0]),
        ("projection", cell[1]),
        ("prior", cell[2]),
        ("evaluation_mode", cell[3]),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v.to_owned()))
    .collect()
}

/// Fill the `response` column of a constructed customer CSV.
fn fill_lander(csv: &str) -> String {
    let mut t = Table::parse(csv);
    let ri = t.col("response");
    let (cd, cp, cr, cm, cs) = (
        t.col("denominator"),
        t.col("projection"),
        t.col("prior"),
        t.col("evaluation_mode"),
        t.col("seed"),
    );
    for (i, row) in t.rows.iter_mut().enumerate() {
        let v = truth(&row[cd], &row[cp], &row[cr], &row[cm]) + seed_offset(&row[cs]) + noise(i);
        row[ri] = format!("{v:.6}");
    }
    t.render()
}

/// Construct the customer design into `dir` and return (csv path, design path).
fn construct_lander(dir: &Scratch, seed: &str) -> (PathBuf, PathBuf) {
    let spec = dir.write("lander.txt", LANDER_DESIGN);
    let csv = dir.join("exp.csv");
    let out = run(&[
        "construct",
        "--from",
        s(&spec),
        "-o",
        s(&csv),
        "--seed",
        seed,
    ]);
    expect_code(&out, 0, "construct the customer design");
    (csv, dir.join("exp.design.json"))
}

// ============================================================== a =========

/// 1a. `construct` reports estimability on stderr and writes both artefacts.
#[test]
fn construct_reports_estimability_and_writes_the_manifest() {
    let dir = Scratch::new("construct");
    let spec = dir.write("lander.txt", LANDER_DESIGN);
    let csv = dir.join("exp.csv");
    let out = run(&[
        "construct",
        "--from",
        s(&spec),
        "-o",
        s(&csv),
        "--seed",
        "7",
    ]);
    expect_code(&out, 0, "construct");

    // The estimability report goes to stderr, before anything is written.
    let err = stderr(&out);
    assert!(
        err.contains("response ~ denominator + projection + prior + evaluation_mode + denominator:projection + prior:evaluation_mode"),
        "stderr should carry the canonical formula:\n{err}"
    );
    assert!(
        err.contains("term"),
        "stderr should carry the term table:\n{err}"
    );
    for term in LANDER_TERMS {
        let line = err
            .lines()
            .find(|l| l.trim_start().starts_with(&format!("{term} ")))
            .unwrap_or_else(|| panic!("no table row for {term} in:\n{err}"));
        assert!(
            line.contains("yes"),
            "term {term} is not estimable: {line}\n{err}"
        );
    }
    assert!(
        stdout(&out).is_empty(),
        "construct should keep stdout clean for piping"
    );

    // The CSV: header plus 36 base rows × 3 seeds.
    let design_path = dir.join("exp.design.json");
    let t = Table::parse(&fs::read_to_string(&csv).expect("read exp.csv"));
    assert_eq!(t.rows.len(), 108, "36 base rows × 3 replicates");
    for c in [
        "run_id",
        "order",
        "denominator",
        "projection",
        "prior",
        "evaluation_mode",
        "seed",
        "checkpoint",
        "response",
    ] {
        assert!(t.header.contains(&c.to_owned()), "CSV lacks column {c}");
    }

    // The manifest.
    let d = read_json(&design_path);
    assert_eq!(d["randomization_seed"], 7);
    assert_eq!(
        d["models"][0]["formula"],
        "response ~ denominator + projection + prior + evaluation_mode + denominator:projection + prior:evaluation_mode"
    );
    assert!(
        d["estimability"]["rank"].is_number(),
        "manifest carries the construct-time estimability report"
    );
    assert_eq!(d["estimability"]["residual_df"], 95);
    let runs = d["runs"].as_array().expect("runs is an array");
    assert_eq!(runs.len(), 108);
    for (i, r) in runs.iter().enumerate() {
        assert!(
            r["run_id"].as_str().is_some_and(|s| s.starts_with('r')),
            "run {i} has no run_id: {r}"
        );
        assert_eq!(r["order"], (i + 1) as u64, "runs are stored in order");
        assert!(
            r["unit"].as_str().is_some_and(|s| s.starts_with('u')),
            "run {i} has no unit id: {r}"
        );
        assert!(
            r["replicates"]["seed"].is_number(),
            "run {i} has no seed replicate level: {r}"
        );
    }
}

// ============================================================== b =========

/// 1b. A full set of results recovers the model, the strata and the optimum.
#[test]
fn analyze_recovers_the_customer_model() {
    let dir = Scratch::new("analyze");
    let (csv, _) = construct_lander(&dir, "7");
    let filled = fill_lander(&fs::read_to_string(&csv).expect("read exp.csv"));
    fs::write(&csv, &filled).expect("write filled csv");

    let json = dir.join("out.json");
    let out = run(&[
        "analyze",
        s(&csv),
        "--json",
        s(&json),
        "--maximize",
        "response",
        "--bootstrap",
        "0",
    ]);
    expect_code(&out, 0, "analyze the filled design");

    let r = &read_json(&json)["results"][0];
    assert_eq!(r["name"], "response");
    assert_eq!(r["n_complete"], 108);
    assert_eq!(
        r["missing_runs"]
            .as_array()
            .expect("missing_runs array")
            .len(),
        0,
        "no cell was left blank"
    );

    // The requested interaction is significant, and in the whole-plot stratum:
    // both of its factors are in the checkpoint key, so it varies between
    // units, not within one.
    let anova = r["anova"].as_array().expect("anova array");
    let dp = anova
        .iter()
        .find(|a| a["term"] == "denominator:projection")
        .expect("anova row for denominator:projection");
    assert_eq!(dp["significant"], true, "denominator:projection: {dp}");
    assert_eq!(dp["stratum"], "whole-plot", "denominator:projection: {dp}");
    // evaluation_mode varies within a checkpoint, so its terms sit below.
    let pe = anova
        .iter()
        .find(|a| a["term"] == "prior:evaluation_mode")
        .expect("anova row for prior:evaluation_mode");
    assert_eq!(pe["stratum"], "sub-plot", "prior:evaluation_mode: {pe}");

    // Two error strata, because the design declares a unit.
    let strata = r["strata"].as_array().expect("strata array");
    assert_eq!(strata.len(), 2, "strata: {strata:?}");
    let names: Vec<&str> = strata.iter().map(|s| s["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["between-unit", "within-unit"]);

    // Predictions cover the whole factor grid and flag what was measured. This
    // L36 happens to be the full factorial, so every grid cell is a design
    // cell; `measured == false` is checked in analyze_reports_missing_runs,
    // where three blanks leave one grid cell unobserved.
    let preds = r["predictions"].as_array().expect("predictions array");
    assert_eq!(preds.len(), 2 * 3 * 2 * 3, "one prediction per grid cell");
    let ft = Table::parse(&filled);
    let cols: Vec<usize> = LANDER_FACTORS.iter().map(|f| ft.col(f)).collect();
    let design_cells: std::collections::BTreeSet<Vec<String>> = ft
        .rows
        .iter()
        .map(|row| cols.iter().map(|&c| row[c].clone()).collect())
        .collect();
    assert_eq!(
        design_cells.len(),
        preds.len(),
        "this design is the full factorial, so every grid cell is measured"
    );
    for p in preds {
        assert_eq!(
            p["measured"], true,
            "every design cell is marked measured: {p}"
        );
        assert!(
            p["observed_mean"].is_number(),
            "measured cell has a mean: {p}"
        );
    }

    // The recommendation is the true optimum.
    let rec = &r["recommendation"]["maximize"];
    let got: BTreeMap<String, String> = rec["factors"]
        .as_object()
        .expect("recommendation factors")
        .iter()
        .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_owned()))
        .collect();
    assert_eq!(got, true_optimum(), "recommendation: {rec}");
}

// ============================================================== c =========

/// 1c. Blanks are reported, not fatal, and the estimability table still prints.
#[test]
fn analyze_reports_missing_runs() {
    let dir = Scratch::new("missing");
    let (csv, _) = construct_lander(&dir, "7");
    let filled = fill_lander(&fs::read_to_string(&csv).expect("read exp.csv"));

    // Blank all three replicates of one factor combination. That leaves three
    // missing runs and, as a consequence, one grid cell that was never
    // measured but is still predicted.
    let mut t = Table::parse(&filled);
    let ri = t.col("response");
    let (cd, cp, cr, cm) = (
        t.col("denominator"),
        t.col("projection"),
        t.col("prior"),
        t.col("evaluation_mode"),
    );
    let mut blanked: Vec<String> = Vec::new();
    for row in t.rows.iter_mut() {
        if (
            row[cd].as_str(),
            row[cp].as_str(),
            row[cr].as_str(),
            row[cm].as_str(),
        ) == ("a", "p", "flat", "greedy")
        {
            row[ri].clear();
            blanked.push(row[0].clone());
        }
    }
    assert_eq!(blanked.len(), 3, "one combination has three replicates");
    fs::write(&csv, t.render()).expect("write blanked csv");

    let json = dir.join("out.json");
    let out = run(&[
        "analyze",
        s(&csv),
        "--json",
        s(&json),
        "--maximize",
        "response",
        "--bootstrap",
        "0",
    ]);
    expect_code(&out, 0, "analyze with three blank cells");

    let text = stdout(&out);
    assert!(
        text.contains("Estimability under observed data"),
        "the estimability table is still printed:\n{text}"
    );
    for term in LANDER_TERMS {
        assert!(
            text.contains(term),
            "term {term} is missing from the report:\n{text}"
        );
    }

    let r = &read_json(&json)["results"][0];
    assert_eq!(r["n_complete"], 105);
    let mut missing: Vec<String> = r["missing_runs"]
        .as_array()
        .expect("missing_runs array")
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    missing.sort();
    blanked.sort();
    assert_eq!(missing, blanked, "missing_runs names the blanked run_ids");

    // The unobserved combination is still predicted, and flagged as such.
    let preds = r["predictions"].as_array().expect("predictions array");
    let unmeasured: Vec<&serde_json::Value> =
        preds.iter().filter(|p| p["measured"] == false).collect();
    assert_eq!(unmeasured.len(), 1, "exactly one grid cell went unmeasured");
    let f = &unmeasured[0]["factors"];
    assert_eq!(f["denominator"], "a");
    assert_eq!(f["projection"], "p");
    assert_eq!(f["prior"], "flat");
    assert_eq!(f["evaluation_mode"], "greedy");
    assert!(
        unmeasured[0]["mean"].is_number(),
        "an unmeasured cell still gets a predicted mean"
    );
    assert!(
        unmeasured[0]["observed_mean"].is_null(),
        "an unmeasured cell has no observed mean"
    );
}

// ============================================================== d =========

/// 1d. `augment` repairs the incomplete experiment without discarding results.
#[test]
fn augment_adds_runs_and_records_the_augmentation() {
    let dir = Scratch::new("augment");
    let (csv, _) = construct_lander(&dir, "7");
    let filled = fill_lander(&fs::read_to_string(&csv).expect("read exp.csv"));

    let mut t = Table::parse(&filled);
    let ri = t.col("response");
    let (cd, cp, cr, cm) = (
        t.col("denominator"),
        t.col("projection"),
        t.col("prior"),
        t.col("evaluation_mode"),
    );
    for row in t.rows.iter_mut() {
        if (
            row[cd].as_str(),
            row[cp].as_str(),
            row[cr].as_str(),
            row[cm].as_str(),
        ) == ("a", "p", "flat", "greedy")
        {
            row[ri].clear();
        }
    }
    let before = t.render();
    fs::write(&csv, &before).expect("write blanked csv");

    let aug = dir.join("aug.csv");
    let out = run(&[
        "augment",
        s(&csv),
        "--for",
        "denominator:projection",
        "--runs",
        "3",
        "-o",
        s(&aug),
    ]);
    expect_code(&out, 0, "augment for denominator:projection");

    let before_t = Table::parse(&before);
    let after_t = Table::parse(&fs::read_to_string(&aug).expect("read aug.csv"));
    assert_eq!(
        after_t.rows.len(),
        before_t.rows.len() + 3,
        "--runs 3 adds exactly three rows"
    );
    // The original rows survive byte for byte.
    assert_eq!(
        after_t.rows[..before_t.rows.len()],
        before_t.rows[..],
        "augment must not rewrite completed rows"
    );
    let old: std::collections::BTreeSet<&str> =
        before_t.rows.iter().map(|r| r[0].as_str()).collect();
    let added: Vec<&str> = after_t.rows[before_t.rows.len()..]
        .iter()
        .map(|r| r[0].as_str())
        .collect();
    assert_eq!(added.len(), 3);
    for id in &added {
        assert!(!old.contains(id), "run_id {id} is not new");
    }

    let d = read_json(&dir.join("aug.design.json"));
    let augs = d["augmentations"].as_array().expect("augmentations array");
    assert_eq!(augs.len(), 1, "one augmentation entry: {augs:?}");
    assert_eq!(augs[0]["targets"][0], "denominator:projection");
    let ids: Vec<&str> = augs[0]["run_ids"]
        .as_array()
        .expect("run_ids array")
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(ids, added, "the entry names the rows that were appended");
}

// ============================================================== e =========

/// 1e. A saturated design has no residual df: `analyze` refuses with exit 2
/// and the spec's wording, and `--pool-auto` is the explicit opt-in.
#[test]
fn saturated_design_needs_explicit_pooling() {
    let dir = Scratch::new("saturated");
    let spec = dir.write(
        "sat.txt",
        r#"[factors]
a = d["lo","hi"]
b = d["lo","hi"]
c = d["lo","hi"]

[model]
y ~ a + b + c

[results]
y
"#,
    );
    let csv = dir.join("sat.csv");
    let out = run(&[
        "construct",
        "--from",
        s(&spec),
        "-o",
        s(&csv),
        "--seed",
        "1",
        "--no-shuffle",
    ]);
    expect_code(&out, 0, "construct the saturated design");
    assert!(
        stderr(&out).contains("residual df: 0"),
        "construct warns that the design is saturated:\n{}",
        stderr(&out)
    );

    // y = 10 + 5a + 0.1b + 0.05c, coded ±1. Four runs, four parameters.
    let mut t = Table::parse(&fs::read_to_string(&csv).expect("read sat.csv"));
    let iy = t.col("y");
    let (ia, ib, ic) = (t.col("a"), t.col("b"), t.col("c"));
    let pm = |v: &str| if v == "hi" { 1.0 } else { -1.0 };
    for row in t.rows.iter_mut() {
        let y = 10.0 + 5.0 * pm(&row[ia]) + 0.1 * pm(&row[ib]) + 0.05 * pm(&row[ic]);
        row[iy] = format!("{y:.6}");
    }
    fs::write(&csv, t.render()).expect("write filled sat.csv");

    let refused = run(&["analyze", s(&csv), "--maximize", "y", "--bootstrap", "0"]);
    expect_code(&refused, 2, "a saturated design without pooling");
    let text = stdout(&refused);
    assert!(
        text.contains("insufficient evidence under this model"),
        "the spec's wording, not 'treating result as noise':\n{text}"
    );
    assert!(
        text.contains("--pool-auto"),
        "the refusal names the opt-in flag:\n{text}"
    );

    let pooled = run(&[
        "analyze",
        s(&csv),
        "--maximize",
        "y",
        "--pool-auto",
        "--bootstrap",
        "0",
    ]);
    expect_code(&pooled, 0, "the same design with --pool-auto");
    let text = stdout(&pooled);
    assert!(
        text.contains("[pooled]"),
        "the report marks the pooled terms:\n{text}"
    );
    assert!(
        text.contains("MAXIMIZE"),
        "pooling restores a recommendation:\n{text}"
    );
}

// ============================================================== f =========

/// 1f. A model the budget cannot carry is refused with exit 3.
#[test]
fn construct_refuses_a_model_that_exceeds_max_runs() {
    let dir = Scratch::new("maxruns");
    let spec = dir.write(
        "tiny.txt",
        r#"[factors]
a = d["lo","hi"]
b = d["lo","hi"]

[model]
y ~ a * b

[results]
y
"#,
    );
    let csv = dir.join("tiny.csv");
    let out = run(&[
        "construct",
        "--from",
        s(&spec),
        "-o",
        s(&csv),
        "--max-runs",
        "3",
        "--seed",
        "1",
    ]);
    expect_code(&out, 3, "a model that needs 4 runs under --max-runs 3");
    let err = stderr(&out);
    assert!(
        err.contains("--max-runs"),
        "the refusal names the flag to raise:\n{err}"
    );
    assert!(
        !csv.exists(),
        "a refused construct writes no CSV: {}",
        csv.display()
    );
    assert!(
        !dir.join("tiny.design.json").exists(),
        "a refused construct writes no manifest"
    );

    // Raising the budget to the design's real size succeeds.
    let ok = run(&[
        "construct",
        "--from",
        s(&spec),
        "-o",
        s(&csv),
        "--max-runs",
        "4",
        "--seed",
        "1",
    ]);
    expect_code(&ok, 0, "the same model under --max-runs 4");
    assert!(csv.exists(), "the CSV is written once the budget allows it");
}

// ============================================================== g =========

/// 1g. A hand-written version-1 sidecar and its `experiment`-keyed CSV still
/// analyze. Nothing but `factors`, `results` and `array` is present.
#[test]
fn analyze_accepts_a_version_1_sidecar() {
    let dir = Scratch::new("v1");
    dir.write(
        "v1.design.json",
        r#"{
  "factors": [
    {"name":"a","spec":"d[\"lo\",\"hi\"]","kind":{"type":"discrete"},"values":["lo","hi"]},
    {"name":"b","spec":"d[\"lo\",\"hi\"]","kind":{"type":"discrete"},"values":["lo","hi"]}
  ],
  "results": ["y"],
  "array": {"rows":4,"cols":2,"label":"L4 (hand-written v1)","method":"Sloane lookup"}
}
"#,
    );
    let csv = dir.write(
        "v1.csv",
        "experiment,a,b,y\n\
         1,lo,lo,4.91\n\
         2,lo,hi,5.09\n\
         3,hi,lo,14.89\n\
         4,hi,hi,15.11\n",
    );

    let json = dir.join("v1.json");
    let out = run(&[
        "analyze",
        s(&csv),
        "--json",
        s(&json),
        "--maximize",
        "y",
        "--bootstrap",
        "0",
    ]);
    expect_code(&out, 0, "analyze a version-1 sidecar");

    let r = &read_json(&json)["results"][0];
    // No [model] section: the default model is all main effects.
    assert_eq!(r["model"], ". ~ a + b");
    assert_eq!(r["n_complete"], 4);
    let a = r["anova"]
        .as_array()
        .expect("anova array")
        .iter()
        .find(|x| x["term"] == "a")
        .expect("anova row for a");
    assert_eq!(a["significant"], true, "the large effect is found: {a}");
    assert_eq!(r["recommendation"]["maximize"]["factors"]["a"], "hi");
}

// ============================================================== h =========

/// 1h. The execution order is reproducible from the seed, and `--no-shuffle`
/// leaves it as the design order.
#[test]
fn execution_order_is_reproducible() {
    let dir = Scratch::new("order");
    let spec = dir.write("lander.txt", LANDER_DESIGN);

    let mut bytes = Vec::new();
    for name in ["a.csv", "b.csv"] {
        let csv = dir.join(name);
        let out = run(&[
            "construct",
            "--from",
            s(&spec),
            "-o",
            s(&csv),
            "--seed",
            "42",
        ]);
        expect_code(&out, 0, "construct with --seed 42");
        bytes.push(fs::read(&csv).expect("read constructed csv"));
    }
    assert_eq!(bytes[0], bytes[1], "the same seed gives the same CSV bytes");

    let other = dir.join("c.csv");
    let out = run(&[
        "construct",
        "--from",
        s(&spec),
        "-o",
        s(&other),
        "--seed",
        "99",
    ]);
    expect_code(&out, 0, "construct with --seed 99");
    assert_ne!(
        bytes[0],
        fs::read(&other).expect("read c.csv"),
        "a different seed shuffles differently"
    );

    let plain = dir.join("d.csv");
    let out = run(&[
        "construct",
        "--from",
        s(&spec),
        "-o",
        s(&plain),
        "--seed",
        "42",
        "--no-shuffle",
    ]);
    expect_code(&out, 0, "construct with --no-shuffle");
    let t = Table::parse(&fs::read_to_string(&plain).expect("read d.csv"));
    for (i, row) in t.rows.iter().enumerate() {
        assert_eq!(
            t.get(row, "order"),
            (i + 1).to_string(),
            "row {i} of a --no-shuffle design"
        );
    }
    // The seed is still recorded, so the run stays reproducible either way.
    let d = read_json(&dir.join("d.design.json"));
    assert_eq!(d["randomization_seed"], 42);
}
