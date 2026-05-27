# taguchi

A command-line tool for **Taguchi / orthogonal-array experiment design and analysis**.

Two-step workflow:

1. **`taguchi construct`** — describe the factors you want to vary and the
   results you want to measure. The tool picks an orthogonal array sized to
   fit, writes a CSV with one row per experiment, and saves a sidecar JSON
   with the full design metadata.

2. Run the experiments yourself, fill in the result columns in the CSV.

3. **`taguchi analyze`** — pass the filled-in CSV back. The tool runs an ANOVA
   decomposition, reports each factor's contribution and an F-test, and tells
   you the level settings that maximize or minimize each result.

If you've never used a Taguchi array before: it's a way to test many factors
with a small number of experiments while still being able to disentangle
which factor caused which effect. A full grid of 5 factors × 3 levels each
is 243 runs; the equivalent orthogonal array (L27) is 27 runs and still lets
you measure every factor's main effect.

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

Suppose you're tuning a database query path. You have three knobs:

- `cpu_freq`: three CPU frequencies between 1.2 and 3.6 GHz
- `cache_mb`: cache sizes of 2, 4, 8, or 16 MB
- `prefetch`: three modes: off, soft, aggressive

And you want to maximize `throughput` and minimize `latency`.

### Construct

```bash
$ taguchi construct -o exp.csv
factor 1> cpu_freq = uniform[1.2, 3.6, 3]
factor 2> cache_mb = d[2,4,8,16]
factor 3> prefetch = d["off","soft","aggressive"]
factor 4>
result 1> throughput
result 2> latency
result 3>

Building L72 (Sloane lookup, 3 of 21 cols used): 72 runs × 3 factor columns
Method: Sloane lookup (MA.72.4.1.6.5.3.8.2.7.txt)
✓ wrote exp.csv
✓ wrote exp.design.json
```

This writes 72 experiments to `exp.csv` (header + 72 rows), with the
factor cells pre-filled and the result cells blank. The sidecar
`exp.design.json` records the design so `analyze` can interpret the data
later.

You can skip the interactive prompt by putting the factors in a file:

```
# design.txt
[factors]
cpu_freq = uniform[1.2, 3.6, 3]
cache_mb = d[2,4,8,16]
prefetch = d["off","soft","aggressive"]

[results]
throughput
latency
```

```bash
$ taguchi construct --from design.txt -o exp.csv
```

### Run the experiments and fill in the CSV

This part is on you. Each row of `exp.csv` is one experimental configuration;
run that configuration, measure the results, and put them in the empty cells.

### Analyze

```bash
$ taguchi analyze exp.csv --maximize throughput --minimize latency

=== Result: throughput ===
  observations: 72
  grand mean:   54.243
  residual:     SS=55.75  df=64  σ̂=0.93

  Factor decomposition:
    ★ cpu_freq      94.1%  F=16023  p<0.0001  range=48.2
        1.2 : μ=30.17  n=24
        2.4 : μ=54.17  n=24
        3.6 : μ=78.40  n=24
    ★ prefetch       3.1%  F=  524  p<0.0001  range= 8.7
    ★ cache_mb       2.7%  F=  304  p<0.0001  range= 8.5

  → To MAXIMIZE throughput:
      cpu_freq    = 3.6
      prefetch    = aggressive
      cache_mb    = 16
      predicted response ≈ 88.3
```

If no factor reaches significance (α=0.05 by default), the tool refuses to
make a recommendation and exits with code 2:

```
⚠ no factor reached significance at α=0.05. treating result as noise.
  pass --tolerate-noise <σ> to report effects whose level-range exceeds σ.
```

Pass `--tolerate-noise 0.5` to lower the bar to "any factor whose level
means span more than 0.5 units."

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

## analyze flags

```
--maximize col1,col2     result columns to optimize by maximization
--minimize col3          result columns to optimize by minimization
--tolerate-noise <σ>     report factors whose level-range exceeds σ;
                         without this, F-test (α=0.05) is the gate and
                         inconclusive results exit with code 2
--alpha <a>              override F-test significance level (default 0.05)
--design <path>          override the design sidecar path
```

If `--maximize` and `--minimize` are both omitted, the report shows
recommendations for both directions and lets you pick.

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
instantaneous.

## What's happening under the hood

The tool tries four construction methods in order:

1. **Sloane catalog lookup.** All 280 orthogonal arrays from
   neilsloane.com/oadir/ are embedded in the binary. If the user's level
   structure matches a catalog entry (or a subset of one), use it.

2. **Bush construction over GF(q).** A closed-form algebraic construction
   that handles homogeneous factors (all the same level count) for prime
   powers q ≤ 49. SIMD-accelerated.

3. **Kronecker product.** For mixed-level inputs (different level counts
   across factors), build a Bush block per level group and combine via
   row tensor product.

4. **Backtracking search.** Last resort for cases the upper tiers can't
   serve. Capped at small N to keep search bounded.

See `CONTRIBUTING.md` for the development guide.

## License

Not specified yet — add one if you intend to redistribute.
