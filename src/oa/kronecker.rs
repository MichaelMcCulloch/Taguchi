//! Tier 3: Kronecker (row-tensor) product across per-level Bush blocks.
//!
//! If factors split into groups by level count {q_i: k_i factors}, build a
//! Bush OA_i with q_i levels and k_i columns, then combine via row tensor:
//!   new_row_index = (i_1, i_2, …) ↦ concat(OA_1[i_1], OA_2[i_2], …)
//!
//! Total rows = ∏ N_i. Strength-2 holds across all column pairs because:
//!   - within a block: inherited from that block
//!   - across blocks: every (x, y) pair counted as (rows_with_col_a=x in OA_1)
//!     × (rows_with_col_b=y in OA_2) = (N_1/q_a)·(N_2/q_b) = N/(q_a q_b) ✓
//!
//! Only fires when ≥ 2 distinct level counts exist (otherwise Bush handled it).

use anyhow::Result;
use std::collections::BTreeMap;

use super::bush;
use super::gf::Gf;
use super::{ArrayInfo, Selected};

pub fn try_construct(levels: &[usize]) -> Result<Option<Selected>> {
    // Group factors by level count, preserving original positions.
    let mut by_level: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, &l) in levels.iter().enumerate() {
        by_level.entry(l).or_default().push(i);
    }
    if by_level.len() < 2 {
        return Ok(None);
    }

    // Build a Bush block for each level group.
    let mut blocks: Vec<(usize, Vec<Vec<usize>>, Vec<usize>)> = Vec::new(); // (q, block_oa, factor_positions)
    for (&q, positions) in by_level.iter() {
        let k_need = positions.len();
        let q_eff = bush::next_prime_power_at_least(q);
        let gf = Gf::new(q_eff)?;
        // Pick smallest k such that capacity ≥ k_need.
        let mut k = 2usize;
        loop {
            let cap = ((q_eff as u64).pow(k as u32) - 1) / (q_eff as u64 - 1);
            if cap as usize >= k_need {
                break;
            }
            k += 1;
            if k > 10 {
                return Ok(None);
            }
        }
        let full = bush::bush_at(&gf, k);
        let block: Vec<Vec<usize>> = full
            .into_iter()
            .map(|row| row.into_iter().take(k_need).collect())
            .collect();
        blocks.push((q_eff, block, positions.clone()));
    }

    // Row tensor: total rows = ∏ block_rows.
    let block_sizes: Vec<usize> = blocks.iter().map(|(_, b, _)| b.len()).collect();
    let total_rows: usize = block_sizes.iter().product();
    let total_cols = levels.len();
    let mut array = vec![vec![0usize; total_cols]; total_rows];

    for row_idx in 0..total_rows {
        // Decompose row_idx into block indices.
        let mut rem = row_idx;
        let mut block_idx = vec![0usize; blocks.len()];
        for (bi, &sz) in block_sizes.iter().enumerate() {
            block_idx[bi] = rem % sz;
            rem /= sz;
        }
        // Fill cells.
        for (bi, (_q, block, positions)) in blocks.iter().enumerate() {
            let src_row = &block[block_idx[bi]];
            for (col_in_block, &fac_pos) in positions.iter().enumerate() {
                array[row_idx][fac_pos] = src_row[col_in_block];
            }
        }
    }

    let mut column_levels = vec![0usize; total_cols];
    for (q, _, positions) in &blocks {
        for &p in positions {
            column_levels[p] = *q;
        }
    }

    let method_parts: Vec<String> = blocks
        .iter()
        .map(|(q, b, p)| format!("{}^{}@{}", q, p.len(), b.len()))
        .collect();
    let method = format!("Kronecker [{}]", method_parts.join(" ⊗ "));
    let label = format!("L{} ({})", total_rows, method);

    Ok(Some(Selected {
        info: ArrayInfo {
            rows: total_rows,
            cols: total_cols,
            label,
            method,
        },
        array,
        column_levels,
    }))
}

#[cfg(test)]
mod tests {
    use super::super::verify_strength2;
    use super::*;

    #[test]
    fn mixed_2_and_3() {
        // 2 factors at 2 levels + 2 factors at 3 levels.
        // Block 1: L4(2^3), takes 2 cols. Block 2: L9(3^4), takes 2 cols.
        // Product: 4·9 = 36 rows × 4 cols.
        let sel = try_construct(&[2, 2, 3, 3]).unwrap().unwrap();
        assert_eq!(sel.array.len(), 36);
        verify_strength2(&sel.array, &sel.column_levels).unwrap();
    }

    #[test]
    fn declines_when_homogeneous() {
        let sel = try_construct(&[3, 3, 3, 3]).unwrap();
        assert!(sel.is_none(), "should defer to Bush");
    }
}
