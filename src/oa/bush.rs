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
///
/// Dispatches to an x86_64 SSSE3 path for GF(2^k) with q ≤ 16 (uses
/// `pshufb` for 16-way GF mult per cycle); falls back to scalar for
/// everything else.
pub fn bush_at(gf: &Gf, k: usize) -> Vec<Vec<u8>> {
    #[cfg(target_arch = "x86_64")]
    {
        if gf.q <= 16 {
            let has_avx2 = std::is_x86_feature_detected!("avx2");
            let has_ssse3 = std::is_x86_feature_detected!("ssse3");
            // GF(2^k): add is XOR → fastest path.
            if gf.p == 2 {
                if has_avx2 {
                    let cols = enumerate_directions(gf.q, k);
                    return unsafe { bush_at_simd_gf2pow_avx2(gf, k, &cols) };
                }
                if has_ssse3 {
                    let cols = enumerate_directions(gf.q, k);
                    return unsafe { bush_at_simd_gf2pow(gf, k, &cols) };
                }
            }
            // Prime q with p ≠ 2 (q ∈ {3, 5, 7, 11, 13}): modular SIMD add.
            // gf.k == 1 means q is a true prime (not an extension field).
            if gf.k == 1 && has_avx2 {
                let cols = enumerate_directions(gf.q, k);
                return unsafe { bush_at_simd_prime_avx2(gf, k, &cols) };
            }
        }
    }
    bush_at_scalar(gf, k)
}

fn bush_at_scalar(gf: &Gf, k: usize) -> Vec<Vec<u8>> {
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

/// SSSE3 SIMD path for GF(2^k) with q ≤ 16. Processes 16 rows in parallel
/// per (column, digit) iteration using `_mm_shuffle_epi8` for the GF
/// multiplication table lookup and `_mm_xor_si128` for the GF add.
///
/// Computes column-major into a flat buffer (so each 16-row tile stores as
/// one `_mm_storeu_si128`), then transposes to the row-major `Vec<Vec<u8>>`
/// the caller expects. The transpose is memory-bandwidth bound and small
/// compared to the construction itself.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn bush_at_simd_gf2pow(gf: &Gf, k: usize, directions: &[Vec<u8>]) -> Vec<Vec<u8>> {
    use std::arch::x86_64::*;

    let q = gf.q;
    let n_rows = (q as u64).pow(k as u32) as usize;
    let n_cols = directions.len();

    // mul_rows[a] = 16 bytes where index i gives gf.mul(a, i). Padded with
    // zero for i ≥ q (those slots are never addressed since a-values < q).
    let mut mul_rows: Vec<[u8; 16]> = Vec::with_capacity(q);
    for a in 0..q {
        let mut row = [0u8; 16];
        for c in 0..q {
            row[c] = gf.mul(a as u8, c as u8);
        }
        mul_rows.push(row);
    }

    // Column-major flat buffer: cell(r, c) at offset c*n_rows + r.
    let mut flat = vec![0u8; n_rows * n_cols];

    let mut a_buf: Vec<[u8; 16]> = vec![[0u8; 16]; k];

    unsafe {
        for r_block in (0..n_rows).step_by(16) {
            let batch = (n_rows - r_block).min(16);
            for r in 0..batch {
                let mut idx = r_block + r;
                for i in 0..k {
                    a_buf[i][r] = (idx % q) as u8;
                    idx /= q;
                }
            }
            for r in batch..16 {
                for i in 0..k {
                    a_buf[i][r] = 0;
                }
            }

            let a_vecs: Vec<__m128i> = a_buf
                .iter()
                .map(|arr| _mm_loadu_si128(arr.as_ptr() as *const __m128i))
                .collect();

            for (col_idx, c) in directions.iter().enumerate() {
                let mut acc = _mm_setzero_si128();
                for i in 0..k {
                    let c_i = c[i] as usize;
                    let mul_row =
                        _mm_loadu_si128(mul_rows[c_i].as_ptr() as *const __m128i);
                    let mul_result = _mm_shuffle_epi8(mul_row, a_vecs[i]);
                    acc = _mm_xor_si128(acc, mul_result);
                }
                let out_ptr = flat.as_mut_ptr().add(col_idx * n_rows + r_block);
                if batch == 16 {
                    _mm_storeu_si128(out_ptr as *mut __m128i, acc);
                } else {
                    let mut arr = [0u8; 16];
                    _mm_storeu_si128(arr.as_mut_ptr() as *mut __m128i, acc);
                    for r in 0..batch {
                        *out_ptr.add(r) = arr[r];
                    }
                }
            }
        }
    }

    // Transpose column-major flat → row-major Vec<Vec<u8>>.
    let mut rows = vec![vec![0u8; n_cols]; n_rows];
    for c in 0..n_cols {
        let base = c * n_rows;
        for r in 0..n_rows {
            rows[r][c] = flat[base + r];
        }
    }
    rows
}

/// AVX2 variant — same algorithm, 32-row batches via `_mm256_shuffle_epi8`.
/// `_mm256_shuffle_epi8` is lane-wise (two parallel 16-byte shuffles within
/// each 128-bit half), so we broadcast the 16-byte mul row to both lanes
/// with `_mm256_broadcastsi128_si256` and a single shuffle does 32 lookups.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bush_at_simd_gf2pow_avx2(gf: &Gf, k: usize, directions: &[Vec<u8>]) -> Vec<Vec<u8>> {
    use std::arch::x86_64::*;

    let q = gf.q;
    let n_rows = (q as u64).pow(k as u32) as usize;
    let n_cols = directions.len();

    let mut mul_rows: Vec<[u8; 16]> = Vec::with_capacity(q);
    for a in 0..q {
        let mut row = [0u8; 16];
        for c in 0..q {
            row[c] = gf.mul(a as u8, c as u8);
        }
        mul_rows.push(row);
    }

    let mut flat = vec![0u8; n_rows * n_cols];
    let mut a_buf: Vec<[u8; 32]> = vec![[0u8; 32]; k];

    unsafe {
        for r_block in (0..n_rows).step_by(32) {
            let batch = (n_rows - r_block).min(32);
            for r in 0..batch {
                let mut idx = r_block + r;
                for i in 0..k {
                    a_buf[i][r] = (idx % q) as u8;
                    idx /= q;
                }
            }
            for r in batch..32 {
                for i in 0..k {
                    a_buf[i][r] = 0;
                }
            }

            let a_vecs: Vec<__m256i> = a_buf
                .iter()
                .map(|arr| _mm256_loadu_si256(arr.as_ptr() as *const __m256i))
                .collect();

            for (col_idx, c) in directions.iter().enumerate() {
                let mut acc = _mm256_setzero_si256();
                for i in 0..k {
                    let c_i = c[i] as usize;
                    let mul_row_128 =
                        _mm_loadu_si128(mul_rows[c_i].as_ptr() as *const __m128i);
                    let mul_row_256 = _mm256_broadcastsi128_si256(mul_row_128);
                    let mul_result = _mm256_shuffle_epi8(mul_row_256, a_vecs[i]);
                    acc = _mm256_xor_si256(acc, mul_result);
                }
                let out_ptr = flat.as_mut_ptr().add(col_idx * n_rows + r_block);
                if batch == 32 {
                    _mm256_storeu_si256(out_ptr as *mut __m256i, acc);
                } else {
                    let mut arr = [0u8; 32];
                    _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, acc);
                    for r in 0..batch {
                        *out_ptr.add(r) = arr[r];
                    }
                }
            }
        }
    }

    let mut rows = vec![vec![0u8; n_cols]; n_rows];
    for c in 0..n_cols {
        let base = c * n_rows;
        for r in 0..n_rows {
            rows[r][c] = flat[base + r];
        }
    }
    rows
}

/// AVX2 variant for prime q ≤ 16 with p ≠ 2 (q ∈ {3, 5, 7, 11, 13}).
///
/// Same 32-row tile structure as the GF(2^k) version, but the GF add is
/// modular `(a + b) mod q` instead of XOR. We use the branch-free modular
/// trick: `min(sum, sum - q)` where both are unsigned-byte values. When
/// `sum < q` the subtraction wraps to a large value (`sum + 256 - q`) and
/// `_mm256_min_epu8` picks `sum`; otherwise it picks `sum - q`. Correct as
/// long as `0 ≤ a, b < q`, so `sum < 2q ≤ 26 < 256`.
///
/// Does NOT apply to extension fields GF(p^k) with p ≠ 2 (GF(9), GF(25),
/// GF(27), GF(49)) — their addition is per-digit mod p, not (a+b) mod q.
/// Those still take the scalar fallback.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bush_at_simd_prime_avx2(gf: &Gf, k: usize, directions: &[Vec<u8>]) -> Vec<Vec<u8>> {
    use std::arch::x86_64::*;

    let q = gf.q;
    let n_rows = (q as u64).pow(k as u32) as usize;
    let n_cols = directions.len();

    let mut mul_rows: Vec<[u8; 16]> = Vec::with_capacity(q);
    for a in 0..q {
        let mut row = [0u8; 16];
        for c in 0..q {
            row[c] = gf.mul(a as u8, c as u8);
        }
        mul_rows.push(row);
    }

    let q_vec = _mm256_set1_epi8(q as i8);
    let mut flat = vec![0u8; n_rows * n_cols];
    let mut a_buf: Vec<[u8; 32]> = vec![[0u8; 32]; k];

    unsafe {
        for r_block in (0..n_rows).step_by(32) {
            let batch = (n_rows - r_block).min(32);
            for r in 0..batch {
                let mut idx = r_block + r;
                for i in 0..k {
                    a_buf[i][r] = (idx % q) as u8;
                    idx /= q;
                }
            }
            for r in batch..32 {
                for i in 0..k {
                    a_buf[i][r] = 0;
                }
            }

            let a_vecs: Vec<__m256i> = a_buf
                .iter()
                .map(|arr| _mm256_loadu_si256(arr.as_ptr() as *const __m256i))
                .collect();

            for (col_idx, c) in directions.iter().enumerate() {
                let mut acc = _mm256_setzero_si256();
                for i in 0..k {
                    let c_i = c[i] as usize;
                    let mul_row_128 =
                        _mm_loadu_si128(mul_rows[c_i].as_ptr() as *const __m128i);
                    let mul_row_256 = _mm256_broadcastsi128_si256(mul_row_128);
                    let mul_result = _mm256_shuffle_epi8(mul_row_256, a_vecs[i]);
                    let sum = _mm256_add_epi8(acc, mul_result);
                    let sum_minus_q = _mm256_sub_epi8(sum, q_vec);
                    acc = _mm256_min_epu8(sum, sum_minus_q);
                }
                let out_ptr = flat.as_mut_ptr().add(col_idx * n_rows + r_block);
                if batch == 32 {
                    _mm256_storeu_si256(out_ptr as *mut __m256i, acc);
                } else {
                    let mut arr = [0u8; 32];
                    _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, acc);
                    for r in 0..batch {
                        *out_ptr.add(r) = arr[r];
                    }
                }
            }
        }
    }

    let mut rows = vec![vec![0u8; n_cols]; n_rows];
    for c in 0..n_cols {
        let base = c * n_rows;
        for r in 0..n_rows {
            rows[r][c] = flat[base + r];
        }
    }
    rows
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
