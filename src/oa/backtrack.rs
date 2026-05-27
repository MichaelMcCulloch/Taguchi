//! Tier 4: backtracking search.
//!
//! Last resort when catalog lookup, Bush, and Kronecker all decline. Builds
//! the OA column by column, generating balanced candidates and checking
//! pair-balance against every previously-placed column. Capped at small N
//! and small k — practical use of this tier is rare because the upper tiers
//! cover almost every reasonable input.

use anyhow::Result;

use super::{ArrayInfo, Selected};

const MAX_N: usize = 16;
const MAX_K: usize = 8;

pub fn try_construct(levels: &[usize]) -> Result<Option<Selected>> {
    if levels.len() > MAX_K {
        return Ok(None);
    }
    // Pick smallest N: must be divisible by every q and by every q_i·q_j.
    let mut n = lcm_pairs(levels);
    while n <= MAX_N {
        if let Some(array) = search(n, levels) {
            let column_levels = levels.to_vec();
            let method = format!("Backtracking search at N={}", n);
            let label = format!("L{} (backtrack, {} cols)", n, levels.len());
            return Ok(Some(Selected {
                info: ArrayInfo {
                    rows: n,
                    cols: column_levels.len(),
                    label,
                    method,
                },
                array,
                column_levels,
            }));
        }
        n += step(levels);
    }
    Ok(None)
}

/// Public so the test in `super::tests` can exercise the search engine.
pub fn search(n: usize, levels: &[usize]) -> Option<Vec<Vec<usize>>> {
    if levels.is_empty() {
        return Some(vec![vec![]; n]);
    }
    for &q in levels {
        if n % q != 0 {
            return None;
        }
    }
    // Pair divisibility: n / (q_i * q_j) must be integer for every pair.
    for i in 0..levels.len() {
        for j in (i + 1)..levels.len() {
            if n % (levels[i] * levels[j]) != 0 {
                return None;
            }
        }
    }
    let mut cols: Vec<Vec<usize>> = Vec::new();
    if extend(&mut cols, n, levels) {
        Some(transpose(&cols, n))
    } else {
        None
    }
}

fn extend(cols: &mut Vec<Vec<usize>>, n: usize, remaining: &[usize]) -> bool {
    if remaining.is_empty() {
        return true;
    }
    let q = remaining[0];
    let candidates = if cols.is_empty() {
        vec![canonical_first(n, q)]
    } else {
        gen_balanced(n, q)
    };
    for cand in candidates {
        if check_pairs(&cand, cols, q) {
            cols.push(cand);
            if extend(cols, n, &remaining[1..]) {
                return true;
            }
            cols.pop();
        }
    }
    false
}

fn canonical_first(n: usize, q: usize) -> Vec<usize> {
    let per = n / q;
    let mut out = Vec::with_capacity(n);
    for level in 0..q {
        for _ in 0..per {
            out.push(level);
        }
    }
    out
}

/// All distinct n-length sequences using each of 0..q exactly n/q times.
/// Generated lexicographically. Bounded by multinomial(n; n/q, …, n/q).
fn gen_balanced(n: usize, q: usize) -> Vec<Vec<usize>> {
    let per = n / q;
    let mut counts = vec![per; q];
    let mut current = vec![0usize; n];
    let mut out = Vec::new();
    fill(0, &mut counts, &mut current, &mut out, n);
    out
}

fn fill(
    pos: usize,
    counts: &mut [usize],
    current: &mut [usize],
    out: &mut Vec<Vec<usize>>,
    n: usize,
) {
    if pos == n {
        out.push(current.to_vec());
        return;
    }
    for level in 0..counts.len() {
        if counts[level] == 0 {
            continue;
        }
        counts[level] -= 1;
        current[pos] = level;
        fill(pos + 1, counts, current, out, n);
        counts[level] += 1;
    }
}

fn check_pairs(cand: &[usize], placed: &[Vec<usize>], q: usize) -> bool {
    for col in placed {
        let qj = *col.iter().max().unwrap_or(&0) + 1;
        let n = cand.len();
        let expected = n / (q * qj);
        if n != expected * q * qj {
            return false;
        }
        let mut counts = vec![0usize; q * qj];
        for i in 0..n {
            counts[cand[i] * qj + col[i]] += 1;
        }
        if counts.iter().any(|&c| c != expected) {
            return false;
        }
    }
    true
}

fn transpose(cols: &[Vec<usize>], n: usize) -> Vec<Vec<usize>> {
    let k = cols.len();
    let mut rows = vec![vec![0usize; k]; n];
    for (ci, col) in cols.iter().enumerate() {
        for (ri, &v) in col.iter().enumerate() {
            rows[ri][ci] = v;
        }
    }
    rows
}

fn lcm_pairs(levels: &[usize]) -> usize {
    let mut acc = 1usize;
    for i in 0..levels.len() {
        for j in i..levels.len() {
            acc = lcm(acc, levels[i] * levels[j]);
        }
    }
    acc
}

fn step(levels: &[usize]) -> usize {
    let mut s = 1usize;
    for i in 0..levels.len() {
        for j in i..levels.len() {
            s = lcm(s, levels[i] * levels[j]);
        }
    }
    s
}

fn lcm(a: usize, b: usize) -> usize {
    if a == 0 || b == 0 {
        return 0;
    }
    a / gcd(a, b) * b
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

#[cfg(test)]
mod tests {
    use super::super::verify_strength2;
    use super::*;

    #[test]
    fn search_l4() {
        let arr = search(4, &[2, 2, 2]).expect("L4 must exist");
        verify_strength2(&arr, &[2, 2, 2]).unwrap();
    }

    #[test]
    fn search_l8_partial() {
        // Just 4 cols of L8.
        let arr = search(8, &[2, 2, 2, 2]).expect("must find");
        verify_strength2(&arr, &[2, 2, 2, 2]).unwrap();
    }

    #[test]
    fn lcm_calc() {
        assert_eq!(lcm(6, 4), 12);
        assert_eq!(gcd(12, 8), 4);
    }
}
