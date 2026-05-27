//! Orthogonal-array construction with tiered fall-through.
//!
//! Methods tried, in order:
//!   1. Lookup     — known mixed-level catalog entries (L12, L18, L36).
//!   2. GF Bush    — Bush construction over GF(prime) or GF(prime power).
//!   3. Kronecker  — combine per-level-group Bush OAs by tensor product.
//!   4. Backtrack  — DFS column extension with pair-balance pruning.
//!
//! Every `Selected` carries the human-readable `method` so callers can show
//! which tier fired. All paths return an OA where `array[row][col]` is in
//! `0..column_levels[col]`; `column_levels` matches the per-factor level count.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::factor::{Factor, FactorValue};

pub mod backtrack;
pub mod bush;
pub mod gf;
pub mod kronecker;
pub mod lookup;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayInfo {
    pub rows: usize,
    pub cols: usize,
    pub label: String,
    pub method: String,
}

#[derive(Debug, Clone)]
pub struct Selected {
    pub info: ArrayInfo,
    pub array: Vec<Vec<usize>>,
    pub column_levels: Vec<usize>,
}

impl Selected {
    pub fn rows(&self) -> usize {
        self.array.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Lookup,
    Bush,
    Kronecker,
    Backtrack,
}

pub const ALL_METHODS: &[Method] = &[
    Method::Lookup,
    Method::Bush,
    Method::Kronecker,
    Method::Backtrack,
];

pub fn build_array(factors: &[Factor]) -> Result<Selected> {
    build_with_methods(factors, ALL_METHODS)
}

/// Try each method in order; first one that returns Some wins. Useful for
/// tests that need to verify a specific tier fires (by passing a subset).
pub fn build_with_methods(factors: &[Factor], methods: &[Method]) -> Result<Selected> {
    if factors.is_empty() {
        bail!("need at least one factor");
    }
    let levels: Vec<usize> = factors.iter().map(|f| f.level_count()).collect();
    if levels.iter().any(|&l| l < 2) {
        bail!("each factor needs ≥ 2 levels");
    }
    for m in methods {
        let res = match m {
            Method::Lookup => lookup::find(&levels),
            Method::Bush => bush::try_construct(&levels)?,
            Method::Kronecker => kronecker::try_construct(&levels)?,
            Method::Backtrack => backtrack::try_construct(&levels)?,
        };
        if let Some(sel) = res {
            return Ok(sel);
        }
    }
    bail!(
        "no method in {:?} produced an OA for level structure {:?}",
        methods,
        levels
    );
}

/// Map an OA cell value (in 0..column_levels[col]) to the corresponding
/// FactorValue. When column_levels[col] > factor.level_count() the overflow
/// folds via modulo (distributes overhang across all factor levels).
pub fn resolve_cell(factor: &Factor, oa_value: usize) -> FactorValue {
    let l = factor.values.len();
    factor.values[oa_value % l].clone()
}

/// Strength-2 check: every pair of columns shows every (a, b) combination
/// exactly N / (q_a * q_b) times.
pub fn verify_strength2(array: &[Vec<usize>], column_levels: &[usize]) -> Result<()> {
    let n = array.len();
    let cols = column_levels.len();
    for i in 0..cols {
        for j in (i + 1)..cols {
            let qi = column_levels[i];
            let qj = column_levels[j];
            let expected = n as f64 / (qi * qj) as f64;
            if (expected.fract()).abs() > 1e-9 {
                bail!(
                    "cols {},{}: N={} not divisible by qi*qj={}",
                    i,
                    j,
                    n,
                    qi * qj
                );
            }
            let expected = expected as usize;
            let mut counts = vec![0usize; qi * qj];
            for row in array {
                if row[i] >= qi {
                    bail!("col {} value {} ≥ qi={}", i, row[i], qi);
                }
                if row[j] >= qj {
                    bail!("col {} value {} ≥ qj={}", j, row[j], qj);
                }
                counts[row[i] * qj + row[j]] += 1;
            }
            for (k, c) in counts.iter().enumerate() {
                if *c != expected {
                    bail!(
                        "cols {},{}: combo ({},{}) appears {} times, expected {}",
                        i,
                        j,
                        k / qj,
                        k % qj,
                        c,
                        expected
                    );
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factor::Factor;

    fn fac(name: &str, n: usize) -> Factor {
        let spec = format!("d[1-{}]", n);
        Factor::parse(name, &spec).unwrap()
    }

    fn assert_valid(sel: &Selected) {
        verify_strength2(&sel.array, &sel.column_levels)
            .unwrap_or_else(|e| panic!("invalid OA: {}\nmethod: {}", e, sel.info.method));
    }

    // ===== Each tier is forced in isolation via build_with_methods. =====

    #[test]
    fn tier1_lookup_isolated() {
        // L12 Plackett-Burman: 11 factors at 2 levels each. Catalog has it.
        let fs: Vec<Factor> = (0..11).map(|i| fac(&format!("f{}", i), 2)).collect();
        let sel = build_with_methods(&fs, &[Method::Lookup]).unwrap();
        assert!(sel.info.method.contains("Sloane"), "got {}", sel.info.method);
        assert_eq!(sel.rows(), 12);
        assert_valid(&sel);
    }

    #[test]
    fn tier2_bush_isolated_prime() {
        // 4 factors at 3 levels each → L9 via Bush GF(3) when lookup is skipped.
        let fs: Vec<Factor> = (0..4).map(|i| fac(&format!("f{}", i), 3)).collect();
        let sel = build_with_methods(&fs, &[Method::Bush]).unwrap();
        assert!(sel.info.method.contains("Bush GF(3)"), "got {}", sel.info.method);
        assert_eq!(sel.rows(), 9);
        assert_valid(&sel);
    }

    #[test]
    fn tier2_bush_isolated_prime_power() {
        // 5 factors at 4 levels each → L16 via Bush GF(4=2^2).
        let fs: Vec<Factor> = (0..5).map(|i| fac(&format!("f{}", i), 4)).collect();
        let sel = build_with_methods(&fs, &[Method::Bush]).unwrap();
        assert!(sel.info.method.contains("GF(4=2^2)"), "got {}", sel.info.method);
        assert_eq!(sel.rows(), 16);
        assert_valid(&sel);
    }

    #[test]
    fn tier2_bush_declines_mixed_level() {
        // 2 different level counts → Bush should return None.
        let fs = vec![fac("a", 2), fac("b", 2), fac("c", 3), fac("d", 3)];
        let r = build_with_methods(&fs, &[Method::Bush]);
        assert!(r.is_err(), "Bush should decline mixed-level");
    }

    #[test]
    fn tier3_kronecker_isolated() {
        // Mixed 2 and 3 — Kronecker builds L4 ⊗ L9 = 36 rows.
        let fs = vec![fac("a", 2), fac("b", 2), fac("c", 3), fac("d", 3)];
        let sel = build_with_methods(&fs, &[Method::Kronecker]).unwrap();
        assert!(sel.info.method.contains("Kronecker"), "got {}", sel.info.method);
        assert_valid(&sel);
    }

    #[test]
    fn tier4_backtrack_isolated() {
        // Small homogeneous case backtrack can solve.
        let fs: Vec<Factor> = (0..3).map(|i| fac(&format!("f{}", i), 2)).collect();
        let sel = build_with_methods(&fs, &[Method::Backtrack]).unwrap();
        assert!(sel.info.method.contains("Backtrack"), "got {}", sel.info.method);
        assert_valid(&sel);
    }

    // ===== Full dispatch picks the smallest/best tier =====

    #[test]
    fn dispatch_prefers_lookup_when_available() {
        let fs: Vec<Factor> = (0..4).map(|i| fac(&format!("f{}", i), 3)).collect();
        let sel = build_array(&fs).unwrap();
        // Catalog has oa.9.4.3.2 → lookup wins over Bush.
        assert!(sel.info.method.contains("Sloane"), "got {}", sel.info.method);
        assert_valid(&sel);
    }

    #[test]
    fn dispatch_falls_through_to_bush() {
        // Force a homogeneous case lookup may or may not have, then skip lookup.
        let fs: Vec<Factor> = (0..6).map(|i| fac(&format!("f{}", i), 5)).collect();
        let sel = build_with_methods(
            &fs,
            &[Method::Bush, Method::Kronecker, Method::Backtrack],
        )
        .unwrap();
        assert!(sel.info.method.starts_with("Bush"), "got {}", sel.info.method);
        assert_valid(&sel);
    }

    #[test]
    fn dispatch_falls_through_to_kronecker() {
        // Mixed level, force past lookup and bush.
        let fs = vec![fac("a", 2), fac("b", 2), fac("c", 5), fac("d", 5), fac("e", 5)];
        let sel = build_with_methods(
            &fs,
            &[Method::Kronecker, Method::Backtrack],
        )
        .unwrap();
        assert!(sel.info.method.contains("Kronecker"), "got {}", sel.info.method);
        assert_valid(&sel);
    }

    #[test]
    fn dispatch_falls_through_to_backtrack() {
        let fs: Vec<Factor> = (0..3).map(|i| fac(&format!("f{}", i), 2)).collect();
        let sel = build_with_methods(&fs, &[Method::Backtrack]).unwrap();
        assert!(sel.info.method.contains("Backtrack"), "got {}", sel.info.method);
        assert_valid(&sel);
    }
}
