# Contributing

Notes for hacking on this crate.

## The spec is the contract

`docs/SPEC-model-driven-design.md` is the contract for the model-driven design
work: the formula grammar, the design-file sections, the CLI surface, the exit
codes, the JSON shapes and the numeric tolerances. When the code and that
document disagree, the document wins. Change the document first, in its
`## Deviations` section, and say why.

## Source layout

```
src/
  main.rs           — clap entrypoint (thin shim)
  lib.rs            — re-exports the modules
  factor.rs         — Factor parsing (d[…], uniform[…], normal[…], logLow/High[…])
  model.rs          — formula parsing, sum-to-zero model matrix, estimability, least squares
  rng.rs            — splitmix64 + Fisher–Yates; every shuffle and resample goes through it
  design_select.rs  — candidate enumeration, the model-driven pick, the construct-time report
  construct.rs      — design-file parsing, the Design manifest, the v2 CSV writer
  analyze.rs        — rank-aware fit, Type II ANOVA, strata, bootstrap, predictions
  report.rs         — the serde types behind `analyze --json`
  augment.rs        — greedy rank-then-D-gain augmentation of an existing design
  oa/
    mod.rs          — Method enum, Selected struct, build_array dispatch, verify_strength2
    lookup.rs       — Tier 1: Sloane catalog (embedded via include_dir)
    gf.rs           — finite-field arithmetic for GF(prime^k)
    bush.rs         — Tier 2: Bush construction (scalar + SSSE3 + AVX2 paths)
    kronecker.rs    — Tier 3: row-tensor combine of per-level Bush blocks
    backtrack.rs    — Tier 4: DFS column extension with pair-balance pruning
tests/
  cli.rs            — end-to-end tests that spawn the built binary and read what it writes
benches/
  oa.rs             — criterion benchmarks for each tier
data/sloane_arrays/ — 280 .txt files from neilsloane.com/oadir/, baked in at compile time
```

## Statistical core

Five ideas carry the whole v2 analysis. Each one lives behind a single
function, and both `construct` and `analyze` call that same function rather
than each keeping its own copy of the rule. Keep it that way: an estimability
rule implemented twice is an estimability rule that disagrees with itself.

**Sum-to-zero coding.** `model::model_matrix` turns level indices into a
design matrix. A factor with `L` levels gets `L-1` columns: column `j` holds
1 at level `j`, −1 at the last level, and 0 elsewhere. An interaction column
is the element-wise product of one column from each participating factor, over
all combinations. The intercept comes first, then one block column set per
replicate role, then the model terms in order, then — only when the caller
asks — one column set for the unit identity. The coding makes every factor's
effects sum to zero across its levels, so the intercept is the grand mean of a
balanced design and a prediction with the block columns at zero is a prediction
"at the average seed". Labels are `(Intercept)`, `seed[3]`, `a[x]`, `a[x]:b[y]`, `unit[u001]`.

**Estimability by rank difference.** `model::estimability` asks, for each term
T: does dropping T's columns cost the matrix exactly `df(T)` of rank? If yes,
T is estimable; the data can separate it from everything else in the model. If
no, T is confounded and the report names what with. Those names come from
`model::pivoted_null_basis`, a reduced-row-echelon basis of the null space.
The pivoting matters: an arbitrary SVD basis mixes independent aliases
together when the null space has more than one dimension, so it would report
"a:b is aliased with c and d and e" where the truth is "a:b is aliased with
c". Numerical rank uses a tolerance of `1e-10 × σ_max × max(n_rows, n_cols)`.
`design_select::select` calls this before writing anything; `analyze` calls it
again on the rows that actually have data, and flags a term that was estimable
at design time and is not any more.

**Type II sums of squares by nested fits.** `analyze_result` does not subtract
marginal factor sums from the total. That shortcut needs an orthogonal design,
and a single missing result destroys orthogonality. Instead, for each term T
it fits twice: once with every term that neither equals nor contains T, and
once with T's columns added. `SS(T)` is the drop in RSS, and `df(T)` is the
rank gained. Both fits go through `model::fit_least_squares`, a rank-revealing
least squares that returns the minimum-norm solution when the matrix is
rank-deficient, so a confounded model still produces a report instead of an
error.

**Split-plot strata.** When the design file declares a `[units]` key, rows
that share the key values are one experimental unit — one trained checkpoint,
say — and measurements inside a unit are paired. A term whose factors all sit
in the unit key is a *whole-plot* term: it can only change when you build a
new unit. Any other term is a *sub-plot* term. The two kinds have different
error. `analyze_result` fits the model a second time with the unit columns
included: the between-unit mean square is the RSS drop those columns cause on
top of the whole-plot terms, and the within-unit mean square is what is left
after everything. Whole-plot terms are tested against the first, sub-plot
terms against the second. A stratum with zero degrees of freedom reports
`F: n/a` rather than inventing a denominator.

**Cluster bootstrap.** Coefficient t-intervals assume the model is right.
`analyze::bootstrap` gives a second opinion that does not. It resamples whole
replicate groups — every row of one training seed moves together, so the
pairing inside a unit stays intact — and refits `B` times, then reports
percentile intervals for every coefficient and every predicted cell. A group
drawn twice becomes two distinct block levels, so the refit needs no
re-centring. The default is 1000 resamples when a replicate role exists and
0 when there is none; below five groups the report warns that the intervals
are not trustworthy. The RNG is `rng::Rng` seeded from `--seed` or the
manifest's `randomization_seed`, so a bootstrap is reproducible.

## The dispatch contract

`oa::build_array(&[Factor]) -> Result<Selected>` is the public entry point.
Internally it calls `build_with_methods` which iterates through a list of
`Method` variants and returns the first one that succeeds:

```rust
pub enum Method { Lookup, Bush, Kronecker, Backtrack }

pub fn build_with_methods(factors: &[Factor], methods: &[Method]) -> Result<Selected>;
```

Each tier exposes a `try_construct(levels: &[usize]) -> Result<Option<Selected>>`
function. **Returning `Ok(None)` means "this tier declines, try the next."**
Returning `Ok(Some(_))` means "I produced an OA, use it." Returning `Err(_)`
is a real failure that aborts the whole dispatch.

Tier preconditions (so dispatch stays correct):

- **Lookup** never errors; returns `None` when no catalog entry matches.
- **Bush** returns `None` for mixed-level inputs (Kronecker is exact for those).
- **Kronecker** returns `None` for homogeneous inputs (Bush is smaller for those).
- **Backtrack** returns `None` past its size cap (MAX_N=16, MAX_K=8).

Tests in `oa/mod.rs` force individual tiers by passing a subset of `Method`
variants — that's the right way to verify a single tier works in isolation
without depending on what dispatch picks.

## Selected and verify_strength2

```rust
pub struct Selected {
    pub info: ArrayInfo,           // metadata for the report + design.json
    pub array: Vec<Vec<u8>>,       // n_rows × n_factors; cell ∈ 0..column_levels[col]
    pub column_levels: Vec<usize>, // per-factor level count (the "q" of that column)
}
```

**Every constructor must produce an array satisfying `verify_strength2`** —
every pair of columns shows every level-combination N/(q_i·q_j) times. The
function is in `oa/mod.rs`; call it from a test if you add a new constructor.

## The four tiers in detail

### Tier 1 — Sloane lookup

280 `.txt` files from neilsloane.com/oadir/ are embedded via the
`include_dir` crate at compile time. At first lookup we parse them into an
index sorted by N (rows). Match is by level-count multiset: the catalog
entry's column-level multiset must be a superset of the user's.

Filename parsing is in `parse_filename` (`oa.N.k.s.t.[label].txt` for
homogeneous, `MA.N.s1.k1.s2.k2.[…].txt` for mixed). Content parsing handles
both packed (s ≤ 10, no separators) and space-separated formats.

Adding more catalog files: drop them in `data/sloane_arrays/` and they'll be
picked up automatically. Run `cargo test oa::lookup` to verify parsing.

### Tier 2 — Bush over GF(q)

Implements OA(q^k, (q^k-1)/(q-1), q, 2) via the standard Bush construction:
rows indexed by k-tuples (a_0, …, a_{k-1}) ∈ GF(q)^k, columns indexed by
canonical representatives of 1-dimensional subspaces (leading non-zero
entry = 1). Cell value at row a, column c is Σ c_i · a_i in GF(q).

For prime q, GF(q) is just modular arithmetic. For prime power q = p^r, GF(q)
is implemented via polynomial representation with a hardcoded primitive
irreducible polynomial — see `factor_prime_power` in `gf.rs`. Supported
fields: GF(2/3/4/5/7/8/9/11/13/16/17/19/23/25/27/32/49).

To add a new GF (e.g., GF(64)):

1. Add a `(2, 6) => Some((2, 6, &[…irr coeffs…]))` case in `factor_prime_power`.
   The irreducible polynomial coefficients are constant-term-first; the
   leading 1 is implicit.
2. Add a `#[test] fn gf64()` calling `check_field(&Gf::new(64).unwrap())`.
3. Optionally add a corresponding `bush::tests::lXXX` case.

Bush has three implementations, picked at runtime based on CPU features:

| Path | Condition | Speed |
|------|-----------|-------|
| `bush_at_simd_gf2pow_avx2` | GF(2^k), q ≤ 16, AVX2 | ~300 M cells/s |
| `bush_at_simd_gf2pow` | GF(2^k), q ≤ 16, SSSE3 | ~150 M cells/s |
| `bush_at_simd_prime_avx2` | prime q ≤ 16, AVX2 | ~300 M cells/s |
| `bush_at_scalar` | everything else | ~40 M cells/s |

The SIMD paths process 32 (AVX2) or 16 (SSSE3) rows in parallel per inner
iteration. The mult table for the current direction coefficient is
broadcast into a vector register and a single `pshufb` gives 32 lookups.
The accumulator add is either `_mm256_xor_si256` (GF(2^k)) or
`_mm256_min_epu8(sum, sum - q)` (prime modular).

Output is computed column-major into a flat buffer, then transposed to the
row-major `Vec<Vec<u8>>` the caller expects. The transpose is small
relative to construction (linear, memory-bandwidth bound).

**Cases that still take the scalar path:**

- `GF(p^k)` extension fields with p ≠ 2: GF(9), GF(25), GF(27), GF(49).
  Their add is per-digit `mod p` on the polynomial encoding, not
  `(a+b) mod q`. SIMDing them needs per-digit pshufb decomposition —
  doable but more code.
- Primes > 16: GF(17), GF(19), GF(23). The mult table doesn't fit in a
  single 16-byte `pshufb` register. Would need split-table tricks.

These are the natural extension points if you want more SIMD coverage.

### Tier 3 — Kronecker

Groups factors by level count, builds a Bush OA for each group, then
combines via row tensor:

```
row_index (i_1, …, i_g) ∈ N_1 × … × N_g
            ↦ concat(OA_1[i_1], OA_2[i_2], …, OA_g[i_g])
```

Total rows = ∏ N_i. Strength-2 holds because:

- within a group: inherited from that group's Bush array
- across groups: every (x, y) pair occurs (N_1/q_x) · (N_2/q_y) = N/(q_x q_y)

Declines on homogeneous inputs (one level count) — Bush handles those
with smaller N.

### Tier 4 — Backtracking search

DFS column extension. For each new column to add:

1. Generate balanced candidates (each level appearing N/q times)
2. Reject candidates that break pair-balance with any already-placed column
3. Recurse on remaining factor slots; backtrack on failure

Bounded by `MAX_N=16, MAX_K=8` because generation cost is multinomial. The
search engine is exposed as `backtrack::search(n, levels)` so tests can
exercise it without going through dispatch.

In practice this tier rarely fires — the upper three cover essentially all
real inputs. It's there as a correctness safety net.

## Testing

```bash
cargo test                  # everything: 110 unit tests + 8 CLI tests
cargo test --test cli       # just the end-to-end tests
cargo test oa::             # just the OA modules
cargo test oa::bush::       # just Bush
```

Test conventions:

- Each tier has its own `tests` module with constructor-correctness checks.
- `oa/mod.rs` has dispatch tests that force each tier in isolation via
  `build_with_methods(&factors, &[Method::Xxx])` and assert the resulting
  array is strength-2. **Don't write tests that assume a specific tier
  fires from full dispatch** — the catalog is greedy and what fires depends
  on which files happen to be embedded.
- `gf.rs` tests verify the field axioms (identity, commutativity, inverse,
  distributivity) for every implemented prime power.
- `tests/cli.rs` spawns `env!("CARGO_BIN_EXE_taguchi")` and reads what the
  process writes: stdout, stderr, the CSV, the sidecar and the exit code.
  It touches no library function. Each test owns a scratch directory that
  deletes itself, so the tests run in parallel and leave nothing behind.
  Pass `--bootstrap 0` to every `analyze` call you add there: the default of
  1000 resamples costs about 140 s per call in a debug build, and the
  bootstrap has its own unit tests in `analyze.rs`.
- The scenario in `tests/cli.rs` is the customer request quoted in the spec,
  end to end: construct, fill, analyze, blank three cells, augment. Its
  response function is a known linear model, so the expected ANOVA verdict
  and the true optimum are facts, not fixtures.

## CI

`.gitlab-ci.yml` runs on every push to any branch — the gate is
`$CI_COMMIT_BRANCH`, not a merge request. Two stages: `check` runs
`rustfmt --check` and `test` runs `cargo build --all-targets` then
`cargo test`. There is no image build and no deploy stage; this is a CLI
crate.

Two things about that file are load-bearing:

- `RUSTFLAGS: ""` overrides `.cargo/config.toml`'s `-C target-cpu=native`.
  A shared runner is not the machine that wrote that config.
- The `fmt` job lists individual files rather than running `cargo fmt --check`.
  Most of the pre-v2 tree carries format drift that predates the pipeline.
  When you format a file, add it to that list; never remove one.

## Benchmarks

```bash
cargo bench                 # all groups
cargo bench --bench oa -- bush_raw     # one group
cargo bench --bench oa -- "q2_k1[2-5]" # regex filter
```

Criterion stores results under `target/criterion/`; subsequent runs report
percent changes vs. the last baseline. To establish a fresh baseline, delete
that directory.

Bench groups:

| Group | What it measures |
|-------|------------------|
| `bush_raw` | Bush construction at (q, k) — the dominant cost for large arrays |
| `lookup` | Catalog search (warm — index already built) |
| `kronecker` | Mixed-level combine at varying group sizes |
| `backtrack` | Search at small N (exponential) |
| `dispatch` | End-to-end `build_with_methods` on realistic inputs |

## Build configuration

`.cargo/config.toml` sets `-C target-cpu=native` so the compiler can emit
AVX2 / BMI / etc. when building locally. Drops binary portability — if you
need a portable build, override via `RUSTFLAGS=""`.

`Cargo.toml` enables LTO and `codegen-units=1` for both `release` and
`bench` profiles. The SIMD intrinsics need `target_feature` annotations
that survive LTO; don't change `codegen-units` without re-checking those.

## Things I'd want to do next

In rough order of impact:

1. **GF(p^k) SIMD for p ≠ 2.** Per-digit pshufb decomposition for GF(9),
   GF(25), GF(27), GF(49). Brings ternary extension fields up to ~300 M
   cells/s like the rest.
2. **Eliminate the transpose.** Change `Selected.array` to a flat
   `Vec<u8>` with explicit row/col indexing. Saves the column→row
   transpose at output and aligns better with how downstream code touches
   the data. Touches a lot of consumers.
3. **AVX-512 `vpermi2b`.** 64-wide pshufb-equivalent. ~2× over AVX2 on
   machines that support it.
4. **Bush capacity heuristic for mixed inputs.** Currently mixed → Kronecker
   gives correct-but-large designs (product of block sizes). For users
   willing to trade strict orthogonality for fewer runs, an explicit
   "compromise" mode could fall back to Bush with modulo folding.
5. **A real isomorphism check in backtrack.** The current search has no
   symmetry pruning, so its useful range is tiny. The literature has good
   algorithms (see Schoen's `oapackage`); porting one would push tier 4
   from "exists as a safety net" to "actually competitive at small sizes
   the catalog doesn't cover."

## Commit conventions

Subject under 72 chars, imperative ("Add X", not "Added X"). Body explains
the *why*; the diff already shows the *what*. Performance commits should
include before/after measurements in the body.
