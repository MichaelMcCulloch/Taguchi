# SPEC — Model-driven design, estimability, replication, joint fit, manifest, augmentation

Status: accepted 2026-09-05. This document is the contract for the work items
below. Workers implement against it; validators check against it. When code
and this document disagree, the document wins unless a deviation record is
appended under "Deviations" at the bottom.

## Why

An external user (the Lander project) wrote this request. It is quoted
verbatim because it is the acceptance bar:

> 1. **Let the user specify the model before constructing the design.**
>    For example: `response ~ denominator * projection + prior * evaluation_mode`.
>    Construct a design that can estimate those requested main effects and
>    interactions. Fall back to a full factorial when that is the smallest
>    suitable design.
> 2. **Report what the design can distinguish before any experiments run.**
>    Check the model matrix rank, residual degrees of freedom, and aliases.
>    Identify exactly which requested effects cannot be separated. Reject an
>    unidentifiable requested model by default. This would prevent us spending
>    hours on runs that cannot answer our question.
> 3. **Support replication, blocks, and paired observations explicitly.**
>    Give training seed and checkpoint identity roles distinct from
>    experimental factors. In our case, several evaluation modes share one
>    trained checkpoint; those measurements are paired. Estimate uncertainty
>    across independent training runs while preserving that pairing. A
>    practical first implementation could support seed blocks and bootstrap
>    whole training-seed groups.
> 4. **Fit the specified model jointly, including interactions.**
>    The current analyzer subtracts marginal factor sums of squares from total
>    variation. That decomposition needs orthogonality; missing results can
>    break it. Use a rank-aware least-squares fit, and report missing runs and
>    the resulting estimability. Make pooling small effects into error
>    **explicitly opt-in**. Also replace "treating result as noise" with
>    "insufficient evidence under this model."
> 5. **Produce a machine-readable, reproducible experiment manifest.**
>    Include stable run IDs, factor assignments, replicate/block IDs,
>    randomized execution order, randomization seed, and the requested model.
>    Analysis should export effect estimates, uncertainty intervals,
>    diagnostics, and predictions as JSON. Clearly distinguish measured
>    combinations from predicted ones.
> 6. **Allow an existing design to be augmented.**
>    Given completed runs, propose additional combinations that resolve a
>    specified alias or improve precision for a chosen interaction. That would
>    let us repair an incomplete experiment without discarding useful results.

Items 1–4 are essential. 5 and 6 follow.

## Terms used in this document

- **factor** — an experimental variable with ≥ 2 levels (`[factors]`).
- **term** — a set of one or more distinct factor names. A one-factor term is
  a main effect. A k-factor term is a k-way interaction. Written `a:b`.
- **model** — an ordered list of terms plus an implicit intercept.
- **replicate role** — a blocking variable that is not a factor: e.g. the
  training seed. Each design row is run once per replicate level. Replicate
  roles enter the fit as fixed block effects. They never appear in the model
  formula.
- **unit** — the experimental unit identity, given by a key of factor and
  replicate-role names. Rows that share the key values share a unit
  (e.g. one trained checkpoint). Measurements within a unit are paired.
- **whole-plot term** — a term whose factors are all in the unit key.
- **sub-plot term** — any other term (at least one factor varies within a unit).
- **resolved level index** — the index into `Factor.values` that the CSV cell
  shows. Bush arrays with folding (q > level count) map OA cells with modulo;
  every statistic in this spec is computed on resolved indices, never on raw
  OA cells.

## Design file grammar (additive; old files still parse)

```
[factors]
denominator     = d["a","b"]
projection      = d["p","q","r"]
prior           = d["flat","learned"]
evaluation_mode = d["greedy","beam","sample"]

[model]
response ~ denominator * projection + prior * evaluation_mode

[replicates]
seed = 5                    # levels 1..=5
# or:  seed = d[11,22,33]   # explicit labels, same spec syntax as factors

[units]
checkpoint = denominator, projection, prior, seed

[results]
response
```

- `[model]` holds one formula per line. A formula is `[response ~] expr`.
  `response` is a result name or `.`; omitted or `.` means "all results".
  A result with no formula of its own uses the `.` formula. With no `[model]`
  section at all, the model is all main effects (today's behaviour).
- `[replicates]` lines: `name = N` (levels are the integers 1..=N) or
  `name = <factor spec>`. Zero or more roles. Multiple roles are crossed.
- `[units]` holds at most one line: `name = key1, key2, …`. Every key is a
  factor name or a replicate-role name. When absent, every run is its own
  unit.
- Names in `[model]` must be factor names. A replicate-role name in a formula
  is an error. A name that is neither is an error naming the line.

Formula grammar, precedence from tightest to loosest:

```
formula := [ response '~' ] expr
expr    := prod ( '+' prod )*
prod    := inter ( '*' inter )*
inter   := atom ( ':' atom )*
atom    := NAME | '(' expr ')'
```

Semantics: `a:b` is the single term {a,b}. `a*b` expands to `a + b + a:b`.
`(a+b)*c` expands to `a + b + c + a:c + b:c`. `a*b*c` expands to all seven
terms. `a:a` is an error. Duplicate terms merge. Terms are canonicalised:
factors within a term ordered by position in `[factors]`; terms ordered by
size, then lexicographically by factor positions. The intercept is always
present. `-1`, `^`, `%in%` are not supported and produce an error that says so.
Hierarchy is not enforced: `a:b` without `a` is allowed, with a warning
printed once at construct time and once at analyze time.

## CLI surface

```
taguchi construct [--from FILE] [-o exp.csv]
                  [--model FORMULA]...      # adds formulas; same as [model] lines
                  [--replicate NAME=SPEC]... # same as [replicates] lines
                  [--unit NAME=k1,k2,...]   # same as [units] line
                  [--seed U64]              # randomisation seed; default derives from time and is recorded
                  [--max-runs N]            # refuse designs above N runs (after replication)
                  [--min-residual-df N]     # default 0
                  [--allow-aliased]         # write the design even when some term is not estimable
                  [--no-shuffle]            # execution order = design order (still recorded)
taguchi analyze  exp.csv [--design PATH] [--maximize a,b] [--minimize c]
                  [--alpha 0.05] [--tolerate-noise SIGMA]
                  [--pool TERM,TERM,...]    # opt-in: pool the named terms into error
                  [--pool-auto]             # opt-in: pool smallest-SS half of terms when residual df = 0
                  [--bootstrap B]           # default 1000 when a replicate role exists, else 0
                  [--json PATH]             # write the analysis report as JSON
taguchi augment  exp.csv [--design PATH] [--for TERM]... [--runs N] [--seed U64]
                  [--include-pending] -o exp_aug.csv
```

Exit codes: 0 ok; 1 usage/IO error; 2 analyze found insufficient evidence
under the model for every result; 3 construct/augment refused because a
requested term is not estimable within `--max-runs` (or the aliased design
was not allowed).

## Shared types — module `src/model.rs` (Phase 1 owns this file)

Phase 2 workers depend on these exact signatures. Phase 1 may add items but
must not rename or remove these.

```rust
pub struct Term(pub Vec<usize>);          // factor indices, strictly increasing, non-empty
impl Term { pub fn label(&self, factors: &[Factor]) -> String }   // "a:b"

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Model {
    pub formula: String,                  // canonical text, e.g. "response ~ a + b + a:b"
    pub response: Option<String>,         // None = "."
    pub terms: Vec<Term>,                 // canonical order, deduped, no intercept
}
impl Model {
    pub fn parse(src: &str, factors: &[Factor]) -> Result<Model>;
    pub fn main_effects(factors: &[Factor]) -> Model;   // the default model
    pub fn for_result(models: &[Model], result: &str) -> Option<&Model>; // named, else "."
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ReplicateRole { pub name: String, pub levels: Vec<FactorValue> }

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Unit { pub name: String, pub key: Vec<String> }   // factor or role names

/// One observation's position in the design: resolved factor level indices
/// and replicate level indices, in declaration order.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Cell { pub factor_levels: Vec<usize>, pub replicate_levels: Vec<usize> }

/// Sum-to-zero (deviation) coded model matrix.
pub struct ModelMatrix {
    pub columns: Vec<ColumnInfo>,        // columns[0] is the intercept
    pub data: Vec<Vec<f64>>,             // n_rows × columns.len(), row-major
}
pub struct ColumnInfo { pub source: ColumnSource, pub label: String }
pub enum ColumnSource { Intercept, Block { role: usize }, Term { term: usize }, Unit }

pub fn model_matrix(
    factors: &[Factor], roles: &[ReplicateRole], model: &Model,
    cells: &[Cell], include_unit: Option<&Unit>,
) -> ModelMatrix;
```

Coding: for a factor (or role) with L levels, columns j = 0..L-2 take value
1 at level j, −1 at level L−1, 0 otherwise. Interaction columns are the
element-wise products of one column from each participating factor, all
combinations, in nested order. Block columns for each role come right after
the intercept, then term columns in model order. When `include_unit` is
`Some`, one sum-to-zero column set for the unit identity (units enumerated in
order of first appearance) is appended last; these columns are only used by
the split-plot strata in analysis.

Labels: `(Intercept)`, `seed[3]`, `a[x]`, `a[x]:b[y]`, `unit[u007]`.

```rust
pub struct Estimability {
    pub n_rows: usize,
    pub n_params: usize,
    pub rank: usize,
    pub residual_df: usize,               // n_rows - rank
    pub terms: Vec<TermEstimability>,
    pub blocks_estimable: bool,
}
pub struct TermEstimability {
    pub term: Term,
    pub df: usize,                        // number of columns for the term
    pub estimable: bool,                  // all its columns independent of every other column
    pub aliased_with: Vec<String>,        // labels of the *terms* (not columns) that alias it; empty when estimable
}
pub fn estimability(mm: &ModelMatrix, model: &Model, factors: &[Factor]) -> Estimability;
```

"Estimable" for a term T means: rank(X) − rank(X without T's columns) == df(T).
`aliased_with` is computed from a *pivoted* null-space basis of X (reduced
row echelon form or column-pivoted QR; never an arbitrary SVD basis, which
mixes independent aliases when the null space has more than one dimension):
for each of T's columns that is dependent, list every other source with a
non-zero coefficient (|c| > 1e-8) in its pivoted null vector. Sources are
term labels, or `(Intercept)`, a replicate-role name, or `unit`, so that a
non-estimable term never reports an empty list. Numerical rank uses a tolerance of `1e-10 × max singular
value × max(n_rows, n_cols)`.

```rust
/// Rank-revealing least squares. Column-pivoted Householder QR or SVD.
pub struct Fit {
    pub coef: Vec<f64>,                   // minimum-norm solution for rank-deficient X
    pub rank: usize,
    pub rss: f64,
    pub residual_df: usize,
    pub fitted: Vec<f64>,
    pub residuals: Vec<f64>,
    pub estimable_coef: Vec<bool>,        // per column: is e_j in the row space of X
    pub cov_unscaled: Option<Vec<Vec<f64>>>, // (X'X)^+ ; multiply by sigma² for coef covariance
}
pub fn fit_least_squares(x: &[Vec<f64>], y: &[f64]) -> Fit;
pub fn is_estimable(x: &[Vec<f64>], v: &[f64]) -> bool;   // v in row space of X: ||v - P v|| <= 1e-8 * max(||v||, 1), P = projector onto row space
```

Linear algebra: the crate may add `nalgebra` (latest 0.33.x) as a dependency.
No other new runtime dependencies. No `rand`: the randomiser is a splitmix64
+ Fisher–Yates in `src/rng.rs` (Phase 1 provides it, ~30 lines) so execution
orders are reproducible across platforms.

`Design` (in `construct.rs`) grows; all new fields have serde defaults so an
old `*.design.json` still loads:

```rust
pub struct Design {
    #[serde(default = "two")] pub version: u32,          // 2
    pub factors: Vec<Factor>,
    pub results: Vec<String>,
    #[serde(default)] pub models: Vec<Model>,             // empty = main effects for all results
    #[serde(default)] pub replicates: Vec<ReplicateRole>,
    #[serde(default)] pub unit: Option<Unit>,
    pub array: ArrayInfo,
    #[serde(default)] pub randomization_seed: Option<u64>,
    #[serde(default)] pub estimability: Option<EstimabilityReport>, // as computed at construct time, for the "." model
    #[serde(default)] pub runs: Vec<Run>,                 // empty for version-1 sidecars; analyze then derives from CSV
    #[serde(default)] pub augmentations: Vec<Augmentation>,
}
pub struct Run {
    pub run_id: String,        // "r0001"… stable, zero-padded to the width of the final count
    pub order: usize,          // 1-based execution order
    pub design_row: usize,     // 0-based index into the base array; usize::MAX for augmented rows
    pub factors: BTreeMap<String, FactorValue>,
    pub replicates: BTreeMap<String, FactorValue>,
    pub unit: Option<String>,  // "u007"
}
pub struct Augmentation { pub seed: u64, pub targets: Vec<String>, pub run_ids: Vec<String> }
```

Phase 1 moves `Design` into `src/model.rs`? No. `Design` stays in
`construct.rs`; Phase 1 adds the new fields there with defaults and leaves
`write_design` producing `runs` in plain order (no shuffle, no replicates).
Phase 2/W2 completes it.

## CSV layout (v2)

```
run_id, order, <factor columns…>, <replicate role columns…>, [<unit name>], <result columns…>
```

Rows are written in execution order. `order` is 1-based. The `experiment`
column of v1 is gone; analyze accepts either (it keys on the design's factor
and result names, and on `run_id` when present). Execution order is a
Fisher–Yates shuffle within each replicate block (block = one combination of
replicate levels), blocks in level order, seeded by `randomization_seed`.

## Design selection with a model (W2)

Given factors, the `.` model (or the union of all formulas' terms when several
formulas exist — the design must estimate every requested term for every
result), replicate roles, and the unit:

1. Enumerate candidate base arrays in ascending row count:
   a. `oa::build_array` result as today.
   b. Every Sloane catalog entry whose level multiset supplies the factors.
      For each entry, enumerate column assignments (which catalog column each
      factor takes, restricted to matching level counts) in a deterministic
      order; stop after 5000 assignments per entry.
   c. For homogeneous level counts: Bush arrays at k = k_min+1 and k_min+2,
      with the same assignment search.
   d. The full factorial (rows in lexicographic level order).
2. For each candidate (in ascending rows; assignments within a candidate in
   enumeration order): build the replicated cell list, build the model matrix
   (blocks + union model, no unit columns), compute `estimability`. The first
   candidate where every requested term is estimable and
   `residual_df ≥ --min-residual-df` wins.
3. If `--max-runs` is set, candidates above it are skipped. If nothing wins:
   print the estimability report of the smallest candidate, exit 3 — unless
   `--allow-aliased`, which writes that smallest candidate and prints a
   warning.
4. Print, before writing anything:

```
Model:    response ~ a + b + a:b + c
Design:   L8 (Sloane oa.8.7.2.2.txt, cols 1,2,4,7) × 3 replicates(seed) = 24 runs
Params:   10   rank: 10   residual df: 14
  term            df  estimable  aliased with
  a                1  yes
  b                1  yes
  c                1  yes
  a:b              1  yes
```

The v1 behaviour (no model given) must produce exactly the array that
`build_array` produces today, so that existing tests and the README examples
keep their run counts.

`oa::lookup` gains `pub fn entries_supplying(levels: &[usize]) -> Vec<CatalogRef>`
and `pub fn select_columns(entry: &CatalogRef, assignment: &[usize]) -> Selected`
as additive hunks. The existing `find` is unchanged.

## Analysis (W3)

Replaces the body of `analyze.rs`. For each result:

1. Load rows. A row is **complete** for a result when its cell parses as f64;
   otherwise it is **missing**. Report missing run_ids (or row numbers when
   there is no run_id) and the count.
2. Build cells from the CSV factor and replicate columns (resolved indices).
   Build the model matrix for this result's model with blocks. Compute
   `estimability` on the observed rows. Print the same term table as
   construct, headed "Estimability under observed data". A term that was
   estimable at construct time and is not now is flagged
   `LOST (missing runs)`.
3. Fit by `fit_least_squares`. σ̂² = RSS / residual_df.
4. ANOVA per term, Type II: SS(T) = RSS(model with all terms that neither
   equal nor contain T) − RSS(that model plus T). df(T) = rank gain of adding
   T's columns. Block SS reported the same way (blocks vs intercept-only).
5. Error strata. With no unit, one stratum: residual MS with residual_df.
   With a unit: fit once more with `include_unit = Some(unit)`. Between-unit
   MS = (RSS(blocks + whole-plot terms) − RSS(blocks + whole-plot terms +
   unit columns)) / (rank gain of the unit columns). Within-unit MS =
   RSS(full model + unit columns) / (n − rank(full + unit)). Whole-plot terms
   are tested against the between-unit MS; sub-plot terms against the
   within-unit MS. If a stratum has 0 df, every term in that stratum reports
   `F: n/a (0 df in its error stratum)`.
6. Pooling is opt-in. `--pool a:b,c` moves those terms' SS and df into the
   residual of their stratum and refits without them. `--pool-auto` reproduces
   the v1 rule (smallest-SS half of terms) only when residual df = 0. With
   neither flag and residual df = 0 for a stratum, the report says
   `insufficient evidence under this model: residual df = 0 in <stratum>;
   add replicates, or pass --pool <terms> / --pool-auto.`
7. Uncertainty. Coefficient SE = sqrt(σ̂² × cov_unscaled[j][j]) for estimable
   columns; t-intervals at 1−α. When `--bootstrap B > 0` and a replicate role
   exists: resample the first replicate role's groups with replacement B
   times (whole groups; units stay intact inside groups). In each resample
   every drawn group becomes its own block level (a group drawn twice is two
   distinct blocks), so the refit uses the same model with no re-centring or
   other adjustment. Report percentile intervals for each coefficient and for
   each predicted cell mean.
   RNG is `rng.rs` seeded by `--seed` (analyze gains `--seed U64`, default
   from the design's randomization_seed). Warn when fewer than 5 groups exist.
8. Predictions: over the full factorial grid of factor levels (block effects
   at their mean, i.e. zero under sum-to-zero coding). Cap the grid at
   100 000 cells; above that, predict only measured cells and the
   recommendation cell. For each cell: predicted mean, SE, `estimable`
   (row-space check), `measured` (appears in complete rows), and the observed
   mean when measured.
9. Significance gate. A term is significant when its p < α on its estimable
   part (df = rank gain; a term with 0 estimable df has no p and is not
   significant). Estimability of the whole term is reported but is not a
   condition for significance. Or, with `--tolerate-noise σ`, when its
   effect range across levels of the term's cells exceeds σ. If no term is significant for a result, print
   `insufficient evidence under this model (α=…)` and do not recommend. If
   that holds for every result, exit 2.
10. Recommendation: the estimable grid cell with the highest (or lowest)
    predicted mean, plus its interval and whether it was measured. v1's
    additive "grand mean + effects" prediction is gone.
11. `--json PATH` writes:

```json
{
  "version": 2,
  "design": "<path>",
  "results": [{
    "name": "response",
    "model": "response ~ ...",
    "n_complete": 170, "missing_runs": ["r0003", "r0111"],
    "estimability": { "n_params": 24, "rank": 24, "residual_df": 146,
                       "terms": [{"term":"a:b","df":2,"estimable":true,"aliased_with":[]}] },
    "fit": { "sigma": 0.93, "rss": 55.7, "r_squared": 0.98 },
    "anova": [{"term":"a","stratum":"whole-plot","df":1,"ss":..., "ms":..., "f":..., "p":..., "significant":true, "pooled":false}],
    "strata": [{"name":"between-unit","df":..,"ms":..},{"name":"within-unit","df":..,"ms":..}],
    "coefficients": [{"label":"a[x]","estimate":..,"se":..,"ci_low":..,"ci_high":..,"boot_low":..,"boot_high":..,"estimable":true}],
    "predictions": [{"factors":{"a":"x","b":"y"},"mean":..,"se":..,"estimable":true,"measured":true,"observed_mean":..,"boot_low":..,"boot_high":..}],
    "recommendation": {"maximize": {"factors":{...},"mean":..,"ci_low":..,"ci_high":..,"measured":false}, "minimize": {...}},
    "warnings": ["a:b requested without a"]
  }]
}
```

Text output mirrors the JSON. The phrase "treating result as noise" must not
appear anywhere in the crate after this work.

## Augmentation (W4)

`taguchi augment exp.csv --for a:b --runs 8 -o exp_aug.csv`

1. Load design + CSV. Completed rows = every result cell non-empty. Pending
   rows are excluded from the base unless `--include-pending`.
2. Candidates = full factorial of factor levels × replicate levels (cap
   200 000; above that, sample candidates uniformly with the seeded RNG and
   say so).
3. Greedy loop. Current X = model matrix of base rows (union model, blocks).
   Each step scores every candidate row x by (a) rank gain for the target
   terms' columns given current X, then (b) D-gain: log det(X'X + xx' + εI) −
   log det(X'X + εI), ε = 1e-8. Pick the best; ties by enumeration order.
   Stop when every target term is estimable and either `--runs` rows were
   added or `--runs` was not given. With no `--for`, targets = every
   non-estimable term of the union model; if all are estimable, run pure
   D-gain for `--runs` rows (error if `--runs` absent).
4. Print estimability before and after. Write the new CSV = original rows
   verbatim + new rows (run_ids continue the sequence, order continues, unit
   ids reuse existing units when the key matches). Write the updated design
   JSON next to the output with an `augmentations` entry. Exit 3 if the
   targets are still not estimable after the loop.

## Work split

| Phase | Item | Files owned | Depends on |
|---|---|---|---|
| 1 | Foundation: `model.rs` (formula, matrix, estimability, fit), `rng.rs`, `Design` fields with defaults, `[model]/[replicates]/[units]` parsing in `construct.rs` into the `Design`, unit tests | `src/model.rs`, `src/rng.rs`, `src/lib.rs`, `src/construct.rs` (parsing + Design only), `Cargo.toml` | — |
| 2/W2 | Design selection, manifest, shuffle, v2 CSV, construct CLI flags | `src/construct.rs`, `src/design_select.rs`, `src/oa/lookup.rs` (additive), `src/main.rs` (construct arm only) | Phase 1 |
| 2/W3 | Analysis rewrite, JSON export, analyze CLI flags | `src/analyze.rs`, `src/report.rs`, `src/main.rs` (analyze arm only) | Phase 1 |
| 2/W4 | Augment | `src/augment.rs`, `src/main.rs` (new subcommand arm only) | Phase 1 |
| 3 | Docs: README, CONTRIBUTING; integration tests in `tests/` driving the binary | `README.md`, `CONTRIBUTING.md`, `tests/` | Phase 2 |

`src/main.rs` and `src/lib.rs` are shared wiring files: minimal additive
hunks only, noted in the worker report.

## Gates

```
cargo build --all-targets
cargo test
cargo clippy --all-targets 2>&1 | rg -n "^\s+--> src/(model|rng|design_select|report|augment)\.rs"   # must print nothing; add the files you own
rustfmt --check <every file you own>
rg -n "treating result as noise" src/ ; test $? -eq 1     # after W3
```

The pre-existing code has clippy warnings and format drift in files nobody
owns here. Leave those alone; the gate is scoped to owned files.

## Deviations

(append here; one line each: date, who, what, why)

- 2026-09-05, Phase 1: Retain the specified deviation coding; test orthogonality between distinct terms, not within multi-column terms, because the item’s mutually-orthogonal 2×3-column requirement contradicts that coding (the two b columns have dot product 2).
- 2026-09-05, Fix-up 1: `aliased_with` may name `(Intercept)`, a replicate-role name, or `unit` as well as terms, because a non-estimable requested term must identify its non-term confounding source.
