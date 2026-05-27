# Contributing

Notes for hacking on this crate.

## Source layout

```
src/
  main.rs           — clap entrypoint (thin shim)
  lib.rs            — re-exports the modules
  factor.rs         — Factor parsing (d[…], uniform[…], normal[…], logLow/High[…])
  construct.rs      — interactive + file-driven design construction, CSV writer
  analyze.rs        — ANOVA decomposition, F-tests, noise-floor handling, recommendation
  oa/
    mod.rs          — Method enum, Selected struct, build_array dispatch, verify_strength2
    lookup.rs       — Tier 1: Sloane catalog (embedded via include_dir)
    gf.rs           — finite-field arithmetic for GF(prime^k)
    bush.rs         — Tier 2: Bush construction (scalar + SSSE3 + AVX2 paths)
    kronecker.rs    — Tier 3: row-tensor combine of per-level Bush blocks
    backtrack.rs    — Tier 4: DFS column extension with pair-balance pruning
benches/
  oa.rs             — criterion benchmarks for each tier
data/sloane_arrays/ — 280 .txt files from neilsloane.com/oadir/, baked in at compile time
```

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
cargo test                  # all unit tests (44 currently)
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
