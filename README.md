# taguchi

A command-line tool for **model-driven experiment design and analysis**. You
say which effects you want to measure; the tool picks the smallest orthogonal
array that can measure them, tells you what it can and cannot separate *before*
you run anything, and then fits the model you asked for.

Two steps, with your experiments in the middle:

1. **`taguchi construct`** — describe the factors, the model, the replication
   and the pairing. The tool selects a design, prints an estimability report,
   writes a CSV with one row per run and a sidecar JSON manifest.

2. Run the experiments yourself. Fill in the result columns of the CSV.

3. **`taguchi analyze`** — pass the filled CSV back. The tool fits the model,
   gives a Type II ANOVA with the right error term for each effect, exports
   estimates, intervals and predictions as JSON, and names the settings that
   maximize or minimize each result.

If a result is missing or an effect turns out to be confounded,
**`taguchi augment`** proposes the extra runs that repair the experiment,
without discarding what you already measured.

If you have never used an orthogonal array before: it is a way to test many
factors in few runs and still tell which factor caused which effect. A full
grid of 5 factors × 3 levels is 243 runs; the L27 array is 27 runs and still
measures every main effect.

## Install

```bash
cargo install --path .
```

Or run from the source tree:

```bash
cargo build --release
./target/release/taguchi --help
```

## A complete example

A team tunes a model-evaluation pipeline. Four knobs:

- `denominator`: two normalisation choices, `a` and `b`
- `projection`: three projections, `p`, `q` and `r`
- `prior`: `flat` or `learned`
- `evaluation_mode`: `greedy`, `beam` or `sample`

They believe `denominator` and `projection` interact, and that `prior` and
`evaluation_mode` interact. They train each checkpoint with 3 different
training seeds. One trained checkpoint is evaluated in all three evaluation
modes, so those three measurements are paired.

That is the whole design file:

```
# lander.txt
[factors]
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
```

### Construct

```bash
$ taguchi construct --from lander.txt -o exp.csv --seed 7

Model:    response ~ denominator + projection + prior + evaluation_mode + denominator:projection + prior:evaluation_mode
Design:   L36 (Sloane MA.36.2.2.3.2.6.2.txt, 4 of 6 cols used) × 3 replicates(seed) = 108 runs
Params:   13   rank: 13   residual df: 95
  term            df  estimable  aliased with
  denominator      1  yes
  projection       2  yes
  prior            1  yes
  evaluation_mode  2  yes
  denominator:projection 2  yes
  prior:evaluation_mode 2  yes

Building L36 (Sloane MA.36.2.2.3.2.6.2.txt, 4 of 6 cols used): 108 runs × 4 factor columns
Method: Sloane lookup (MA.36.2.2.3.2.6.2.txt)
✓ wrote exp.csv
✓ wrote exp.design.json
```

`a * b` expanded to `a + b + a:b`, so the six terms above are the model. Every
one is estimable, and 95 degrees of freedom are left over for error. The report
goes to stderr and prints **before** anything is written, so you can read it and
stop.

`exp.csv` holds 108 rows in the order you should run them:

```
run_id,order,denominator,projection,prior,evaluation_mode,seed,checkpoint,response
r0003,1,a,p,learned,greedy,1,u0003,
r0011,2,a,p,learned,sample,1,u0003,
r0028,3,a,r,learned,sample,1,u0012,
r0006,4,b,p,learned,beam,1,u0002,
```

- `run_id` is stable. It names the run everywhere else: in the manifest, in the
  missing-run list, in the augmentation record.
- `order` is the randomized execution order, shuffled inside each seed block.
- `checkpoint` is the unit id. Rows that share it share one trained checkpoint.
- `response` is yours to fill in.

`exp.design.json` records all of it — factors, model, replicate roles, the unit
key, the randomization seed, the estimability report and every run — so
`analyze` and `augment` need no other input.

### Analyze

```bash
$ taguchi analyze exp.csv --maximize response --json report.json

=== Result: response ===
Model: response ~ denominator + projection + prior + evaluation_mode + denominator:projection + prior:evaluation_mode
Complete: 108
Missing runs (0):
Estimability under observed data
Params: 13 rank: 13 residual df: 95
  term  df  estimable  aliased with
  denominator  1  yes
  projection  2  yes
  prior  1  yes
  evaluation_mode  2  yes
  denominator:projection  2  yes
  prior:evaluation_mode  2  yes
RSS: 0.237308 sigma: 0.049980 R squared: 0.999928
Type II ANOVA
  term  df  SS  MS  F  p  stratum  significance
  denominator  1  1726.800208  1726.800208  714021.271323  0.000000  whole-plot  *
  projection  2  290.686939  145.343469  60098.631174  0.000000  whole-plot  *
  prior  1  240.934408  240.934408  99624.896797  0.000000  whole-plot  *
  evaluation_mode  2  288.120050  144.060025  37250.585289  0.000000  sub-plot  *
  denominator:projection  2  649.502706  324.751353  134282.688093  0.000000  whole-plot  *
  prior:evaluation_mode  2  71.148272  35.574136  9198.647517  0.000000  sub-plot  *
  (blocks)  2  46.464800  23.232400  9606.454588  0.000000  whole-plot
between-unit: df=27 MS=0.002418
within-unit: df=68 MS=0.003867
Coefficients: estimate SE CI low CI high bootstrap low bootstrap high
(Intercept): 49.999722 0.004809 49.990175 50.009270 49.196389 50.803056
seed[1]: -0.803333 0.006801 -0.816836 -0.789831 -1.071110 1.071111
denominator[a]: -3.998611 0.004809 -4.008159 -3.989063 -4.002500 -3.996389
denominator[a]:projection[p]: 3.006389 0.006801 2.992886 3.019891 2.998333 3.015278
[... 9 more coefficients ...]
Prediction {"denominator": "a", "evaluation_mode": "greedy", "prior": "flat", "projection": "p"}: mean=44.504722 SE=0.015951 estimable=true measured=true observed_mean=44.543333 bootstrap=[43.703611, 45.338611]
[... 35 more predictions ...]
MAXIMIZE: {"denominator": "b", "evaluation_mode": "sample", "prior": "learned", "projection": "r"} predicted=63.492500 interval=[62.656944, 64.308611] measured=true
bootstrap has fewer than 5 groups (3); intervals may be unreliable
```

The interesting column is `stratum`. `denominator:projection` is tested against
the **whole-plot** error, because both of its factors are part of the checkpoint
key: that interaction only changes when you train a new checkpoint.
`evaluation_mode` is tested against the **sub-plot** error, because it varies
inside one checkpoint. Testing both against one pooled error would overstate the
evidence for the whole-plot effects.

The last line is a real warning, not decoration: three training seeds is a small
number of groups to bootstrap.

## The design file

Five sections. Only `[factors]` and `[results]` are required, and a file with
only those two is the version-1 format, which still works.

### `[factors]`

`name = spec` per line. See the factor spec table below.

### `[model]`

One formula per line: `[response ~] expr`. `response` is a result name, or `.`,
or omitted — both mean "every result". A result with no formula of its own uses
the `.` formula. **With no `[model]` section, the model is all main effects**,
which is what the version-1 tool always did.

| Written | Means |
|---|---|
| `a` | the main effect of `a` |
| `a:b` | the single two-way interaction term |
| `a*b` | `a + b + a:b` |
| `a*b*c` | all seven terms |
| `(a+b)*c` | `a + b + c + a:c + b:c` |

Precedence, tightest first: `:` then `*` then `+`. Duplicate terms merge.
`^` and `%in%` are refused by name (`unsupported formula syntax '^'`), and `-1`
is refused as `unexpected token '-' in formula`. `a:b` without `a` is allowed,
with `warning: a:b requested without a` at construct time and again at analyze
time. Names in a formula must be factor names; a replicate-role name in a
formula is an error.

### `[replicates]`

`name = N` gives levels `1..=N`; `name = <factor spec>` gives explicit labels.
Zero or more roles, crossed with each other. A replicate role is **not** a
factor: it never appears in a formula. It enters the fit as a fixed block
effect, and every design row is run once per replicate level. Use it for the
thing you repeat the experiment across but do not want to optimise — a training
seed, a batch of material, a day.

### `[units]`

At most one line: `name = key1, key2, …`, where each key is a factor or
replicate-role name. Rows whose key values all match are **one experimental
unit**, and their measurements are paired. In the example above, one trained
checkpoint is determined by `denominator, projection, prior, seed`, and the
three evaluation modes are three measurements of that same checkpoint.

Declaring a unit splits the error into two strata:

- **between-unit** — the variation from building a new unit. A term whose
  factors all sit in the unit key (a *whole-plot* term) is tested against this.
- **within-unit** — the variation between measurements of one unit. Every other
  term (a *sub-plot* term) is tested against this.

With no `[units]` line every run is its own unit and there is one error term.

### `[results]`

One result column name per line.

## The estimability report

This is the part that stops you wasting a week. Before any file is written,
`construct` builds the model matrix and asks, for each requested term, whether
the data will be able to separate it from everything else. Ask for too much in
too few runs and it refuses:

```
# screen.txt
[factors]
a = d["off","on"]
b = d["off","on"]
c = d["off","on"]

[model]
y ~ a + b + c + a:b

[results]
y
```

```bash
$ taguchi construct --from screen.txt -o screen.csv --max-runs 4
Model:    y ~ a + b + c + a:b
Design:   L4 (Sloane oa.4.3.2.2.txt, 3 of 3 cols used) = 4 runs
Params:   5   rank: 4   residual df: 0
  term            df  estimable  aliased with
  a                1  yes
  b                1  yes
  c                1  no         a:b
  a:b              1  no         c
refused: no candidate design within the limits estimates: c, a:b. Raise --max-runs, add --replicate, relax --min-residual-df, or pass --allow-aliased to write it anyway.
```

Exit code 3. In four runs, the `a:b` interaction and the main effect of `c` are
the same column: no arithmetic can tell them apart. The `aliased with` column
names the confounding partner, not a vague "some alias exists".

Raise the budget and the tool finds a design that works:

```bash
$ taguchi construct --from screen.txt -o screen.csv --seed 5

Model:    y ~ a + b + c + a:b
Design:   L8 (Sloane oa.8.4.2.3.txt, cols 1,2,3) = 8 runs
Params:   5   rank: 5   residual df: 3
  term            df  estimable  aliased with
  a                1  yes
  b                1  yes
  c                1  yes
  a:b              1  yes

Building L8 (Sloane oa.8.4.2.3.txt, cols 1,2,3): 8 runs × 3 factor columns
Method: Sloane lookup (oa.8.4.2.3.txt)
✓ wrote screen.csv
✓ wrote screen.design.json
```

`--allow-aliased` writes the aliased design anyway, with a warning, when you
know what you are giving up. `--min-residual-df N` refuses a design that leaves
fewer than `N` degrees of freedom for error.

`analyze` runs the same check again on the rows that actually have data, and
flags any term that was estimable at design time and is not any more.

## Insufficient evidence, and pooling

A saturated design — as many parameters as runs — has no degrees of freedom
left to estimate error. Three two-level factors in four runs is one:

```
run_id,order,a,b,c,y
r0001,1,off,off,off,4.70
r0002,2,off,on,on,5.30
r0003,3,on,off,on,14.90
r0004,4,on,on,off,15.10
```

Four runs, four parameters, nothing left over. There is no F-test to run:

```bash
$ taguchi analyze small.csv --maximize y
[...]
insufficient evidence under this model: residual df = 0 in residual; add replicates, or pass --pool <terms> / --pool-auto.
insufficient evidence under this model (α=0.05)
```

Exit code 2. The tool does not silently borrow the smallest effects as an error
estimate. That is a real statistical choice and it is yours to make:

```bash
$ taguchi analyze small.csv --maximize y --pool-auto
[...]
Type II ANOVA
  term  df  SS  MS  F  p  stratum  significance
  a  1  100.000000  100.000000  1000.000000  0.000999  residual  *
  b  1  0.160000  0.160000  n/a  n/a  residual   [pooled]
  c  1  0.040000  0.040000  n/a  n/a  residual   [pooled]
residual: df=2 MS=0.100000
```

`--pool-auto` pools the smallest-SS half of the terms, and only when residual
df is 0. `--pool a:b,c` pools exactly the terms you name. Pooled terms are
marked `[pooled]` and lose their own test. Adding replicates is the better
answer when you can afford it.

## Missing results

Blanks are data, not errors. Here the three `a`/`p`/`flat`/`greedy` rows of the
example above were never run, so their `response` cells are empty:

```bash
$ taguchi analyze exp.csv --maximize response
=== Result: response ===
Model: response ~ denominator + projection + prior + evaluation_mode + denominator:projection + prior:evaluation_mode
Complete: 105
Missing runs (3): r0001, r0037, r0073
Estimability under observed data
Params: 13 rank: 13 residual df: 92
```

The fit is rank-aware, so it does not need a balanced design. If losing those
rows costs you a term, the estimability table says so instead of returning a
wrong number.

## Augment

Given completed runs, `augment` proposes more:

```bash
$ taguchi augment exp.csv --for denominator:projection --runs 3 -o exp_aug.csv
Augmenting: exp.csv
Model:      . ~ denominator + projection + prior + evaluation_mode + denominator:projection + prior:evaluation_mode
Base rows:  105 of 108 (3 pending, excluded)

Estimability before augmentation
Params:   13   rank: 13   residual df: 92   rows: 105
Blocks:   estimable
  term             df  estimable  aliased with
  denominator       1  yes
  [... 5 more terms ...]

Targets:    denominator:projection
Candidates: 108 combinations

Estimability after augmentation
Params:   13   rank: 13   residual df: 95   rows: 108
Blocks:   estimable
  [... 6 terms, all estimable ...]

Added 3 run(s). log det(X'X + εI): 54.953660 → 55.374645
✓ wrote exp_aug.csv
✓ wrote exp_aug.design.json
```

The new rows continue the run-id and order sequences, and reuse an existing
unit id when the unit key matches:

```
r0109,109,a,p,flat,greedy,3,u0025,
r0110,110,a,p,flat,greedy,1,u0001,
r0111,111,a,p,flat,greedy,2,u0013,
```

Every original row is copied through unchanged, so completed results survive the
round trip; `tests/cli.rs` checks that. `exp_aug.design.json` records what was
added, under `augmentations`.

Each step of the search scores every candidate row first by the rank it adds to
the target terms, then by D-optimality (`log det(X'X)` gain). With no `--for`,
the targets are every term the current data cannot estimate; if they are all
estimable already, the search runs on precision alone and `--runs` is then
required. If the targets are still not estimable when the loop ends, augment
exits 3 and names them.

## The JSON export

`analyze --json report.json` writes the whole analysis. Abridged:

```json
{
  "version": 2,
  "design": "exp.design.json",
  "results": [{
    "name": "response",
    "model": "response ~ denominator + projection + ... + prior:evaluation_mode",
    "n_complete": 108,
    "missing_runs": [],
    "estimability": {
      "n_params": 13, "rank": 13, "residual_df": 95,
      "terms": [{"term": "denominator:projection", "df": 2,
                 "estimable": true, "lost": false, "aliased_with": []}]
    },
    "fit": {"sigma": 0.04997982048927725, "rss": 0.23730833333333592,
            "r_squared": 0.9999283898990725},
    "anova": [{"term": "denominator:projection", "stratum": "whole-plot",
               "df": 2, "ss": 649.5027055555555, "ms": 324.75135277777775,
               "f": 134282.68809289776, "p": 1.0731146353741813e-54,
               "significant": true, "pooled": false}],
    "strata": [{"name": "between-unit", "df": 27, "ms": 0.0024184156378602756},
               {"name": "within-unit",  "df": 68, "ms": 0.0038673224563214323}],
    "coefficients": [{"label": "evaluation_mode[beam]",
                      "estimate": 0.0008333333324945045,
                      "se": 0.006801392090812613,
                      "ci_low": -0.012669137141022434,
                      "ci_high": 0.014335803806011443,
                      "boot_low": -0.018055555555817948,
                      "boot_high": 0.023611111110719674,
                      "estimable": true}],
    "predictions": [{"factors": {"denominator": "b", "evaluation_mode": "sample",
                                 "prior": "learned", "projection": "r"},
                     "mean": 63.49250000000002, "se": 0.01595067832574323,
                     "estimable": true, "measured": true,
                     "observed_mean": 63.51333333333333,
                     "boot_low": 62.65694444444447,
                     "boot_high": 64.30861111111113}],
    "recommendation": {"maximize": {"factors": {"denominator": "b", "...": "..."},
                                    "mean": 63.49250000000002,
                                    "ci_low": 62.65694444444447,
                                    "ci_high": 64.30861111111113,
                                    "measured": true},
                       "minimize": null},
    "warnings": ["bootstrap has fewer than 5 groups (3); intervals may be unreliable"]
  }]
}
```

`predictions` covers the full grid of factor levels, not just the cells you ran.
Two flags separate the two kinds:

- `measured: true` — this combination appears in the data. `observed_mean` is
  what you actually saw; `mean` is what the model says.
- `measured: false` — the model predicts this combination but nobody ran it.
  `observed_mean` is `null`.

`estimable: false` marks a cell the model cannot predict at all. The grid is
capped at 100 000 cells; above that only measured cells and the recommendation
are predicted.

`boot_low` / `boot_high` are percentile intervals from the cluster bootstrap:
whole replicate groups are resampled, so the pairing inside a unit stays intact.
They are `null` when the bootstrap did not run.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success. |
| 1 | Usage or I/O error: a bad flag, an unreadable file, a name that is not a factor. |
| 2 | `analyze` found insufficient evidence under the model for **every** result. |
| 3 | `construct` or `augment` refused: a requested term is not estimable inside the limits. |

## Flags

### `taguchi construct`

| Flag | Effect |
|---|---|
| `-o`, `--output <PATH>` | Output CSV. A sidecar `<stem>.design.json` goes next to it. Default `experiments.csv`. |
| `-f`, `--from <PATH>` | Read the design from a file instead of prompting. |
| `--model <FORMULA>` | Add a formula. Repeatable; appended after the file's `[model]` lines. |
| `--replicate <NAME=SPEC>` | Add a replicate role, e.g. `seed=5`. Repeatable. |
| `--unit <NAME=k1,k2,…>` | Set the experimental unit key. |
| `--seed <U64>` | Randomization seed. Default: nanoseconds since the epoch, recorded in the manifest either way. |
| `--max-runs <N>` | Refuse a design above `N` runs, counted after replication. |
| `--min-residual-df <N>` | Require at least `N` residual degrees of freedom. Default 0. |
| `--allow-aliased` | Write the design even when a requested term is not estimable. |
| `--no-shuffle` | Execution order = design order. The seed is still recorded. |

### `taguchi analyze`

| Flag | Effect |
|---|---|
| `<CSV>` | The filled experiment CSV (positional). |
| `-d`, `--design <PATH>` | Override the sidecar path. Default `<csv-stem>.design.json`. |
| `--maximize <a,b>` | Result columns to optimize upward. |
| `--minimize <c>` | Result columns to optimize downward. |
| `--alpha <A>` | Significance level for the F-tests. Default 0.05. |
| `--tolerate-noise <SIGMA>` | Instead of the F-test, call a term significant when its effect range across levels exceeds `SIGMA`. |
| `--pool <TERM,…>` | Pool the named terms into their error stratum and refit. |
| `--pool-auto` | Pool the smallest-SS half of the terms, in strata with 0 residual df. |
| `--bootstrap <B>` | Whole-group bootstrap resamples. Default 1000 with a replicate role, 0 without. |
| `--seed <U64>` | Bootstrap RNG seed. Default: the design's randomization seed. |
| `--json <PATH>` | Write the version-2 report as JSON. |

With neither `--maximize` nor `--minimize`, the report shows both directions.

### `taguchi augment`

| Flag | Effect |
|---|---|
| `<CSV>` | The design so far (positional). Completed rows are the base. |
| `-d`, `--design <PATH>` | Override the sidecar path. |
| `--for <TERM>` | Term to make estimable, e.g. `--for a:b`. Repeatable. Default: every non-estimable term of the model. |
| `--runs <N>` | Number of runs to add. Required when every term is estimable already. |
| `--seed <U64>` | Seed for candidate sampling. Default: the design's randomization seed. |
| `--include-pending` | Count rows with empty results as part of the base. |
| `-o`, `--output <PATH>` | Output CSV. A sidecar `<stem>.design.json` goes next to it. |

## Reproducibility

`construct` records its randomization seed in the manifest, whether you passed
one or not. Given the same design file and the same `--seed`, you get the same
CSV, byte for byte — `tests/cli.rs` checks exactly that. The shuffle is a
Fisher–Yates over a splitmix64 stream in `src/rng.rs` rather than the system
RNG, so the order does not depend on the standard library's generator. The
bootstrap draws from the same stream, so its intervals are reproducible too.

## Factor spec syntax

Inside the `name = spec` line, `spec` is one of:

| Kind | Syntax | Levels generated |
|------|--------|------------------|
| Discrete list | `d[1,2-4,9]` | 1, 2, 3, 4, 9 (ranges expanded) |
| Discrete strings | `d["off","soft","aggressive"]` | the three given strings |
| Uniform | `uniform[low, high, n]` | n points evenly spaced |
| Normal | `normal[low, high, n, std]` | n quantile points of N(mid, std), clamped to [low, high] |
| Log-low | `logLow[low, high, n]` | geometric spacing — denser near `low` |
| Log-high | `logHigh[low, high, n]` | geometric spacing — denser near `high` |

Negative ranges use the dash twice: `d[-5--3,0]` → -5, -4, -3, 0.

## How big can the array be?

The construction is fast. Bush construction over GF(2^k) and prime GF
fields uses AVX2 SIMD (~300 M cells/s on a modern x86_64). Practical
upper limits on a typical workstation:

| Array | Time | Memory |
|------|------|--------|
| L256 — any field | ≈ 100 µs | < 1 MB |
| L729 (3 levels) | 340 µs | 0.5 MB |
| L4096 (2 levels) | 47 ms | 17 MB |
| L8192 (2 levels) | 206 ms | 67 MB |
| L16384 (2 levels) | 843 ms | 268 MB |
| L32768 (2 levels) | 3.9 s | 1.07 GB |

For any reasonable Taguchi experiment (≤ L256), construction is essentially
instantaneous. Design *selection* costs more than construction when a model is
given, because each candidate array needs a rank decomposition; the search is
capped at 5000 column assignments per catalog entry.

## What's happening under the hood

**Choosing the design.** With a `[model]`, `construct` enumerates candidate
arrays in ascending row count — the array `build_array` would have picked, then
every Sloane catalog entry whose level multiset covers the factors, then Bush
arrays one and two steps larger, then the full factorial. For each candidate it
tries column assignments in a deterministic order, builds the replicated model
matrix and computes estimability. The first candidate that estimates every
requested term and meets `--min-residual-df` wins. With no `[model]`, the
selection is exactly what the version-1 tool did, so old run counts do not move.

**Building the array.** `build_array` tries four construction methods in order:

1. **Sloane catalog lookup.** All 280 orthogonal arrays from
   neilsloane.com/oadir/ are embedded in the binary. If the level structure
   matches a catalog entry (or a subset of one), use it.

2. **Bush construction over GF(q).** A closed-form algebraic construction
   that handles homogeneous factors (all the same level count) for prime
   powers q ≤ 49. SIMD-accelerated.

3. **Kronecker product.** For mixed-level inputs (different level counts
   across factors), build a Bush block per level group and combine via
   row tensor product.

4. **Backtracking search.** Last resort for cases the upper tiers cannot
   serve. Capped at small N to keep search bounded.

**Fitting the model.** The model matrix is sum-to-zero coded: each factor's
columns are built so that its effects sum to zero across its levels, which
makes the intercept the grand mean of a balanced design. Fits go through a
rank-revealing least squares, which is what lets a design with missing rows
still produce a report. Sums of squares are Type II: each term's SS is the drop
in residual sum of squares when its columns are added to a model holding every
term that does not contain it. That needs no orthogonality, which the older
"subtract marginal sums from the total" approach did.

See `docs/SPEC-model-driven-design.md` for the contract this implements, and
`CONTRIBUTING.md` for the development guide.

## License

GNU Affero General Public License v3.0 — see `LICENSE` for the full text.
