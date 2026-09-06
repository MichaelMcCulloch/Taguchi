//! Tier 1: lookup against the embedded Sloane orthogonal-array catalog.
//!
//! All 280 .txt files from neilsloane.com/oadir/ are baked in at compile
//! time. We parse each filename to recover its level structure and each
//! file body into the OA matrix. At lookup time, the user's level multiset
//! is matched against the catalog: the smallest array that can supply the
//! requested per-level column counts wins.

use include_dir::{Dir, include_dir};
use std::collections::BTreeMap;
use std::sync::OnceLock;

use super::{ArrayInfo, Selected};

static EMBED: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/data/sloane_arrays");

#[derive(Debug)]
struct CatalogEntry {
    name: String,
    n_runs: usize,
    /// Level count per column (length = total k).
    columns: Vec<usize>,
    array: Vec<Vec<u8>>,
}

fn index() -> &'static [CatalogEntry] {
    static IDX: OnceLock<Vec<CatalogEntry>> = OnceLock::new();
    IDX.get_or_init(build_index)
}

fn build_index() -> Vec<CatalogEntry> {
    let mut entries = Vec::new();
    for file in EMBED.files() {
        let name = file.path().file_name().and_then(|s| s.to_str()).unwrap_or("");
        let body = match file.contents_utf8() {
            Some(s) => s,
            None => continue,
        };
        if let Some(meta) = parse_filename(name) {
            if let Some(array) = parse_body(body, meta.n_runs, &meta.columns) {
                entries.push(CatalogEntry {
                    name: name.to_string(),
                    n_runs: meta.n_runs,
                    columns: meta.columns,
                    array,
                });
            }
        }
    }
    // Sort by N (rows) ascending, then by total cols ascending, then by name
    // for determinism.
    entries.sort_by(|a, b| {
        a.n_runs
            .cmp(&b.n_runs)
            .then(a.columns.len().cmp(&b.columns.len()))
            .then(a.name.cmp(&b.name))
    });
    entries
}

struct FilenameMeta {
    n_runs: usize,
    columns: Vec<usize>,
}

fn parse_filename(name: &str) -> Option<FilenameMeta> {
    let stem = name.strip_suffix(".txt")?;
    let mut tokens = stem.split('.');
    let prefix = tokens.next()?;
    // Collect leading numeric tokens.
    let nums: Vec<usize> = tokens
        .map_while(|t| t.parse::<usize>().ok())
        .collect();

    match prefix {
        "oa" => {
            // oa.N.k.s.t : N rows, k cols, all at s levels, strength t
            if nums.len() < 4 {
                return None;
            }
            let n = nums[0];
            let k = nums[1];
            let s = nums[2];
            // strength = nums[3] — we accept any but mostly want ≥ 2.
            Some(FilenameMeta {
                n_runs: n,
                columns: vec![s; k],
            })
        }
        "MA" => {
            // MA.N.s1.k1.s2.k2.[…]
            if nums.len() < 3 || (nums.len() - 1) % 2 != 0 {
                return None;
            }
            let n = nums[0];
            let mut columns = Vec::new();
            let mut i = 1;
            while i + 1 < nums.len() {
                let s = nums[i];
                let k = nums[i + 1];
                for _ in 0..k {
                    columns.push(s);
                }
                i += 2;
            }
            Some(FilenameMeta { n_runs: n, columns })
        }
        _ => None, // skip ds.*, had.*, etc.
    }
}

fn parse_body(body: &str, n_runs: usize, columns: &[usize]) -> Option<Vec<Vec<u8>>> {
    let max_level = *columns.iter().max().unwrap_or(&2);
    let packed = max_level <= 10;
    let mut rows: Vec<Vec<u8>> = Vec::with_capacity(n_runs);
    for line in body.lines() {
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        let row: Vec<u8> = if packed {
            let chars: Vec<char> = s.chars().filter(|c| !c.is_whitespace()).collect();
            if chars.len() != columns.len() {
                let parts: Vec<&str> = s.split_whitespace().collect();
                if parts.len() != columns.len() {
                    continue;
                }
                let parsed: Option<Vec<u8>> =
                    parts.iter().map(|p| p.parse::<u8>().ok()).collect();
                match parsed {
                    Some(v) => v,
                    None => continue,
                }
            } else if chars.iter().all(|c| c.is_ascii_digit()) {
                chars.iter().map(|c| (*c as u8) - b'0').collect()
            } else {
                continue;
            }
        } else {
            let parts: Vec<&str> = s.split_whitespace().collect();
            if parts.len() != columns.len() {
                continue;
            }
            let parsed: Option<Vec<u8>> =
                parts.iter().map(|p| p.parse::<u8>().ok()).collect();
            match parsed {
                Some(v) => v,
                None => continue,
            }
        };
        // Validate against column level bounds.
        if row.iter().zip(columns).any(|(v, q)| (*v as usize) >= *q) {
            return None;
        }
        rows.push(row);
        if rows.len() == n_runs {
            break;
        }
    }
    if rows.len() == n_runs {
        Some(rows)
    } else {
        None
    }
}

/// A catalog entry that can supply a requested level structure. Additive:
/// `find` does not use it.
#[derive(Debug, Clone)]
pub struct CatalogRef {
    pub name: String,
    pub n_runs: usize,
    /// Level count of every column in the entry, in catalog column order.
    pub columns: Vec<usize>,
    /// Position in the static catalog index.
    entry: usize,
}

/// Every catalog entry whose column level multiset supplies `levels`, in
/// ascending run count (the catalog index order).
pub fn entries_supplying(levels: &[usize]) -> Vec<CatalogRef> {
    if levels.is_empty() {
        return Vec::new();
    }
    let mut need: BTreeMap<usize, usize> = BTreeMap::new();
    for &l in levels {
        *need.entry(l).or_insert(0) += 1;
    }
    index()
        .iter()
        .enumerate()
        .filter(|(_, entry)| {
            let mut have: BTreeMap<usize, usize> = BTreeMap::new();
            for &l in &entry.columns {
                *have.entry(l).or_insert(0) += 1;
            }
            need.iter()
                .all(|(l, c)| have.get(l).copied().unwrap_or(0) >= *c)
        })
        .map(|(i, entry)| CatalogRef {
            name: entry.name.clone(),
            n_runs: entry.n_runs,
            columns: entry.columns.clone(),
            entry: i,
        })
        .collect()
}

/// Extract the catalog columns named by `assignment` (one column index per
/// factor, in factor order).
///
/// # Panics
/// When `assignment` names a column outside the entry, or names one twice.
pub fn select_columns(entry: &CatalogRef, assignment: &[usize]) -> Selected {
    let source = &index()[entry.entry];
    assert!(
        assignment.iter().all(|&c| c < source.columns.len()),
        "column assignment out of range for {}",
        source.name
    );
    assert!(
        assignment
            .iter()
            .enumerate()
            .all(|(i, c)| !assignment[..i].contains(c)),
        "column assignment repeats a column of {}",
        source.name
    );
    let array: Vec<Vec<u8>> = source
        .array
        .iter()
        .map(|row| assignment.iter().map(|&c| row[c]).collect())
        .collect();
    let column_levels: Vec<usize> = assignment.iter().map(|&c| source.columns[c]).collect();
    let cols: Vec<String> = assignment.iter().map(|&c| (c + 1).to_string()).collect();
    Selected {
        info: ArrayInfo {
            rows: array.len(),
            cols: column_levels.len(),
            label: format!(
                "L{} (Sloane {}, cols {})",
                source.n_runs,
                source.name,
                cols.join(",")
            ),
            method: format!("Sloane lookup ({})", source.name),
        },
        array,
        column_levels,
    }
}

pub fn find(user_levels: &[usize]) -> Option<Selected> {
    if user_levels.is_empty() {
        return None;
    }
    let mut need: BTreeMap<usize, usize> = BTreeMap::new();
    for &l in user_levels {
        *need.entry(l).or_insert(0) += 1;
    }

    for entry in index() {
        let mut have: BTreeMap<usize, usize> = BTreeMap::new();
        for &l in &entry.columns {
            *have.entry(l).or_insert(0) += 1;
        }
        let supplies = need
            .iter()
            .all(|(l, c)| have.get(l).copied().unwrap_or(0) >= *c);
        if !supplies {
            continue;
        }
        // Build assignment: for each user factor, pick a not-yet-used column
        // from the entry that has matching level count.
        let mut available_by_level: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, &l) in entry.columns.iter().enumerate() {
            available_by_level.entry(l).or_default().push(i);
        }
        let mut chosen = Vec::with_capacity(user_levels.len());
        for &l in user_levels {
            let pool = available_by_level.get_mut(&l)?;
            chosen.push(pool.remove(0));
        }
        // Extract those columns from the entry's array.
        let array: Vec<Vec<u8>> = entry
            .array
            .iter()
            .map(|row| chosen.iter().map(|&c| row[c]).collect())
            .collect();
        let column_levels: Vec<usize> = chosen.iter().map(|&c| entry.columns[c]).collect();
        let method = format!("Sloane lookup ({})", entry.name);
        let label = format!(
            "L{} (Sloane {}, {} of {} cols used)",
            entry.n_runs,
            entry.name,
            chosen.len(),
            entry.columns.len()
        );
        return Some(Selected {
            info: ArrayInfo {
                rows: array.len(),
                cols: column_levels.len(),
                label,
                method,
            },
            array,
            column_levels,
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::super::verify_strength2;
    use super::*;

    #[test]
    fn parses_oa_filename() {
        let m = parse_filename("oa.16.15.2.2.0.txt").unwrap();
        assert_eq!(m.n_runs, 16);
        assert_eq!(m.columns, vec![2; 15]);
    }

    #[test]
    fn parses_ma_filename() {
        let m = parse_filename("MA.36.6.1.3.5.2.4.txt").unwrap();
        assert_eq!(m.n_runs, 36);
        // 1 col @ 6, 5 cols @ 3, 4 cols @ 2.
        assert_eq!(m.columns.iter().filter(|&&l| l == 6).count(), 1);
        assert_eq!(m.columns.iter().filter(|&&l| l == 3).count(), 5);
        assert_eq!(m.columns.iter().filter(|&&l| l == 2).count(), 4);
    }

    #[test]
    fn index_has_entries() {
        let idx = index();
        assert!(idx.len() > 50, "got {} entries", idx.len());
    }

    #[test]
    fn l12_plackett_burman() {
        // 11 factors at 2 levels — L12 is the only thing that fits at minimum N.
        let levels = vec![2; 11];
        let sel = find(&levels).expect("L12 must be in catalog");
        assert_eq!(sel.array.len(), 12);
        verify_strength2(&sel.array, &sel.column_levels).unwrap();
    }

    #[test]
    fn entries_supplying_and_select_columns() {
        // Additive surface: every entry that can carry four 2-level factors,
        // in ascending run count, and the columns extracted from one of them.
        let entries = entries_supplying(&[2, 2, 2, 2]);
        assert!(entries.len() > 5, "got {} entries", entries.len());
        assert!(entries.windows(2).all(|w| w[0].n_runs <= w[1].n_runs));
        assert!(entries.iter().all(|e| {
            e.columns.iter().filter(|&&l| l == 2).count() >= 4 && e.columns.len() >= 4
        }));
        assert!(entries_supplying(&[]).is_empty());
        assert!(entries_supplying(&[97]).is_empty(), "no 97-level columns");

        let entry = entries.iter().find(|e| e.n_runs == 8).expect("an L8 entry");
        let selected = select_columns(entry, &[3, 1, 0, 2]);
        assert_eq!(selected.array.len(), 8);
        assert_eq!(selected.column_levels, vec![2; 4]);
        assert!(
            selected.info.label.contains("cols 4,2,1,3"),
            "{}",
            selected.info.label
        );
        verify_strength2(&selected.array, &selected.column_levels).unwrap();
        // The columns are the entry's own, in the order asked for.
        let straight = select_columns(entry, &[0, 1, 2, 3]);
        for (row, picked) in straight.array.iter().zip(&selected.array) {
            assert_eq!(vec![row[3], row[1], row[0], row[2]], *picked);
        }
    }

    #[test]
    #[should_panic(expected = "column assignment out of range")]
    fn select_columns_rejects_a_bad_column() {
        let entry = entries_supplying(&[2, 2]).into_iter().next().unwrap();
        let out_of_range = entry.columns.len();
        select_columns(&entry, &[0, out_of_range]);
    }

    #[test]
    fn mixed_l18_from_homogeneous() {
        // 7 factors at 3 levels — L18 oa.18.7.3.2 fits exactly.
        let levels = vec![3; 7];
        let sel = find(&levels).unwrap();
        assert_eq!(sel.array.len(), 18);
        verify_strength2(&sel.array, &sel.column_levels).unwrap();
    }
}
