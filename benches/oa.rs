use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::time::Duration;
use taguchi::factor::Factor;
use taguchi::oa::{Method, build_with_methods};
use taguchi::oa::{backtrack, bush, gf::Gf, kronecker, lookup};

fn mkfac(name: &str, n: usize) -> Factor {
    Factor::parse(name, &format!("d[1-{}]", n)).unwrap()
}

/// Bush construction at increasing (q, k). Each emits N = q^k rows by
/// (q^k - 1)/(q - 1) columns.
fn bench_bush_raw(c: &mut Criterion) {
    let mut group = c.benchmark_group("bush_raw");
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(20);
    let cases: &[(usize, usize)] = &[
        (2, 4),   // L16, 15 cols
        (2, 6),   // L64, 63 cols
        (2, 8),   // L256, 255 cols
        (2, 10),  // L1024, 1023 cols
        (2, 12),  // L4096, 4095 cols
        (3, 4),   // L81, 40 cols
        (3, 6),   // L729, 364 cols
        (4, 3),   // L64, 21 cols  — GF(4) prime power
        (4, 4),   // L256, 85 cols — GF(4) prime power
        (5, 4),   // L625, 156 cols
        (7, 3),   // L343, 57 cols
    ];
    for &(q, k) in cases {
        let n = (q as u64).pow(k as u32) as usize;
        let cells = n * ((n - 1) / (q - 1));
        group.throughput(Throughput::Elements(cells as u64));
        let label = format!("q{}_k{}_n{}", q, k, n);
        let gf = Gf::new(q).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(label), &k, |b, &k| {
            b.iter(|| {
                let arr = bush::bush_at(&gf, black_box(k));
                black_box(arr);
            });
        });
    }
    group.finish();
}

/// Tier-1 lookup: first call is cold (parses 280 files), subsequent are warm
/// (just a hashmap-ish search). Bench both.
fn bench_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("lookup");
    group.sample_size(50);

    // Warm: index already initialized after the first call. We do a one-time
    // priming call before the iter loop.
    let _ = lookup::find(&[2, 2, 2]);
    let queries: &[(&str, Vec<usize>)] = &[
        ("homo_2_4", vec![2; 4]),
        ("homo_3_7", vec![3; 7]),
        ("homo_5_5", vec![5; 5]),
        ("mixed_2_3", vec![2, 2, 3, 3, 3]),
        ("mixed_6_3_2", vec![6, 3, 3, 3, 2, 2, 2, 2]),
        ("large_2_31", vec![2; 31]),
        ("large_3_40", vec![3; 40]),
    ];
    for (label, q) in queries {
        group.bench_with_input(BenchmarkId::from_parameter(format!("warm_{}", label)), q, |b, q| {
            b.iter(|| black_box(lookup::find(black_box(q))));
        });
    }
    group.finish();
}

/// Tier-3 Kronecker: scale with number of mixed groups + per-group factor count.
fn bench_kronecker(c: &mut Criterion) {
    let mut group = c.benchmark_group("kronecker");
    group.measurement_time(Duration::from_secs(6));
    group.sample_size(20);
    let cases: &[(&str, &[usize])] = &[
        ("2x2_3x2", &[2, 2, 3, 3]),                  // 4 × 9 = 36 rows
        ("2x3_3x3", &[2, 2, 2, 3, 3, 3]),            // 8 × 9 = 72 rows
        ("2x2_5x3", &[2, 2, 5, 5, 5]),               // 4 × 25 = 100 rows
        ("3x4_5x3", &[3, 3, 3, 3, 5, 5, 5]),         // 9 × 25 = 225 rows
        ("2x4_3x4_5x4", &[2, 2, 2, 2, 3, 3, 3, 3, 5, 5, 5, 5]), // 16×81×625
    ];
    for (label, levels) in cases {
        group.bench_with_input(BenchmarkId::from_parameter(*label), levels, |b, levels| {
            b.iter(|| black_box(kronecker::try_construct(black_box(levels))));
        });
    }
    group.finish();
}

/// Tier-4 backtrack: only small cases, exponential in nature.
fn bench_backtrack(c: &mut Criterion) {
    let mut group = c.benchmark_group("backtrack");
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(15);
    let cases: &[(&str, usize, &[usize])] = &[
        ("L4_3cols", 4, &[2, 2, 2]),
        ("L8_4cols", 8, &[2, 2, 2, 2]),
        ("L8_7cols", 8, &[2, 2, 2, 2, 2, 2, 2]),
        ("L12_5cols", 12, &[2, 2, 2, 2, 2]),
    ];
    for (label, n, levels) in cases {
        group.bench_with_input(BenchmarkId::from_parameter(*label), &(*n, *levels), |b, &(n, lv)| {
            b.iter(|| {
                let r = backtrack::search(black_box(n), black_box(lv));
                black_box(r);
            });
        });
    }
    group.finish();
}

/// End-to-end dispatch: realistic user input → Selected.
fn bench_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("dispatch");
    group.sample_size(30);

    let small: Vec<Factor> = (0..4).map(|i| mkfac(&format!("f{}", i), 3)).collect();
    let medium: Vec<Factor> = (0..15).map(|i| mkfac(&format!("f{}", i), 2)).collect();
    let large: Vec<Factor> = (0..30).map(|i| mkfac(&format!("f{}", i), 2)).collect();
    let mixed: Vec<Factor> = vec![mkfac("a", 2), mkfac("b", 2), mkfac("c", 3), mkfac("d", 3), mkfac("e", 5)];

    group.bench_function("small_homo_3lvl_4fac", |b| {
        b.iter(|| black_box(build_with_methods(&small, &[Method::Lookup, Method::Bush]).unwrap()));
    });
    group.bench_function("medium_homo_2lvl_15fac", |b| {
        b.iter(|| black_box(build_with_methods(&medium, &[Method::Lookup, Method::Bush]).unwrap()));
    });
    group.bench_function("large_homo_2lvl_30fac", |b| {
        b.iter(|| black_box(build_with_methods(&large, &[Method::Lookup, Method::Bush]).unwrap()));
    });
    group.bench_function("mixed_5fac", |b| {
        b.iter(|| black_box(build_with_methods(&mixed, &[Method::Lookup, Method::Kronecker]).unwrap()));
    });
    group.finish();
}

criterion_group!(benches, bench_bush_raw, bench_lookup, bench_kronecker, bench_backtrack, bench_dispatch);
criterion_main!(benches);
