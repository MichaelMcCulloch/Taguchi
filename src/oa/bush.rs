//! Bush construction over GF(q): OA(q^k, (q^k-1)/(q-1), q, 2).
//!
//! Rows are k-tuples (a_0,…,a_{k-1}) ∈ GF(q)^k. Columns are canonical
//! representatives of 1-dimensional subspaces (vectors whose leading non-zero
//! entry is 1). Cell (a, c) = Σ_i c_i · a_i, with arithmetic in GF(q).
//!
//! For prime q this reduces to mod-q. For prime power q=p^k we use the field
//! tables in [`super::gf`].

use anyhow::Result;

use super::gf::{Gf, is_prime_power};
use super::{ArrayInfo, Selected};

/// Try to build an OA covering the requested level structure with a single
/// homogeneous Bush array. Declines on mixed-level inputs (Kronecker handles
/// those without folding loss) — fires only when all factors share the same
/// level count.
pub fn try_construct(levels: &[usize]) -> Result<Option<Selected>> {
    let max_level = *levels.iter().max().unwrap();
    let min_level = *levels.iter().min().unwrap();
    if max_level != min_level {
        return Ok(None);
    }
    let n_factors = levels.len();

    // q = smallest prime power ≥ level count. When q > level, we fold via
    // modulo when writing the CSV; the method label flags this.
    let q = next_prime_power_at_least(max_level);
    let folded = q != max_level;
    let n_factors_needed = n_factors;

    // Choose k: smallest k ≥ 2 such that capacity ≥ n_factors_needed.
    let mut k = 2usize;
    let cap = loop {
        let cap = ((q as u64).pow(k as u32) - 1) / (q as u64 - 1);
        if cap as usize >= n_factors_needed {
            break cap as usize;
        }
        k += 1;
        if k > 10 {
            return Ok(None);
        }
    };

    let gf = Gf::new(q)?;
    let array = bush_at(&gf, k);
    // Truncate to needed columns.
    let array: Vec<Vec<u8>> = array
        .into_iter()
        .map(|row| row.into_iter().take(n_factors).collect())
        .collect();
    let column_levels = vec![q; n_factors];

    let method = if folded {
        format!("Bush GF({}) k={} with {}→{} folding", q, k, q, max_level)
    } else if gf.k == 1 {
        format!("Bush GF({}) k={}", q, k)
    } else {
        format!("Bush GF({}={}^{}) k={}", q, gf.p, gf.k, k)
    };
    let label = format!("L{} ({}, cap {} factors)", array.len(), method, cap);

    Ok(Some(Selected {
        info: ArrayInfo {
            rows: array.len(),
            cols: column_levels.len(),
            label,
            method,
        },
        array,
        column_levels,
    }))
}

/// Bush construction at a given GF and depth k. Returns full
/// q^k × (q^k-1)/(q-1) array with cell values as `u8`.
pub fn bush_at(gf: &Gf, k: usize) -> Vec<Vec<u8>> {
    let q = gf.q;
    let cols = enumerate_directions(q, k);
    let n_rows = (q as u64).pow(k as u32) as usize;
    let mut oa = Vec::with_capacity(n_rows);
    for row_idx in 0..n_rows {
        let a = idx_to_tuple(row_idx, q, k);
        let mut row = Vec::with_capacity(cols.len());
        for c in &cols {
            let mut acc = 0u8;
            for i in 0..k {
                acc = gf.add(acc, gf.mul(c[i], a[i]));
            }
            row.push(acc);
        }
        oa.push(row);
    }
    oa
}

fn enumerate_directions(q: usize, k: usize) -> Vec<Vec<u8>> {
    let mut cols = Vec::new();
    for lead in 0..k {
        let n_trailing = k - lead - 1;
        let n_combos = (q as u64).pow(n_trailing as u32) as usize;
        for combo in 0..n_combos {
            let mut c = vec![0u8; k];
            c[lead] = 1;
            let mut t = combo;
            for pos in (lead + 1)..k {
                c[pos] = (t % q) as u8;
                t /= q;
            }
            cols.push(c);
        }
    }
    cols
}

fn idx_to_tuple(mut idx: usize, q: usize, k: usize) -> Vec<u8> {
    let mut out = vec![0u8; k];
    for i in 0..k {
        out[i] = (idx % q) as u8;
        idx /= q;
    }
    out
}

pub fn next_prime_power_at_least(n: usize) -> usize {
    let mut m = n.max(2);
    while !is_prime_power(m) {
        m += 1;
    }
    m
}

#[cfg(test)]
mod tests {
    use super::super::verify_strength2;
    use super::*;

    fn check(q: usize, k: usize, expected_cols: usize) {
        let gf = Gf::new(q).unwrap();
        let arr = bush_at(&gf, k);
        let n = (q as u64).pow(k as u32) as usize;
        assert_eq!(arr.len(), n);
        assert_eq!(arr[0].len(), expected_cols);
        let levels = vec![q; expected_cols];
        verify_strength2(&arr, &levels).unwrap();
    }

    #[test]
    fn l4() {
        check(2, 2, 3);
    }
    #[test]
    fn l8() {
        check(2, 3, 7);
    }
    #[test]
    fn l9() {
        check(3, 2, 4);
    }
    #[test]
    fn l16_2() {
        check(2, 4, 15);
    }
    #[test]
    fn l16_4() {
        check(4, 2, 5);
    } // GF(4) prime power!
    #[test]
    fn l25() {
        check(5, 2, 6);
    }
    #[test]
    fn l27() {
        check(3, 3, 13);
    }
    #[test]
    fn l64_4() {
        check(4, 3, 21);
    } // GF(4) k=3 gives L64 with 21 cols
    #[test]
    fn l81_3() {
        check(3, 4, 40);
    }
    #[test]
    fn l81_9() {
        check(9, 2, 10);
    } // GF(9) prime power gives L81 with 10 cols
}
