//! Tiny, std-only dataset loaders used by the examples.
//!
//! Three formats, matching the prep scripts in `scripts/`:
//! * [`read_numeric_csv`] — `f1,f2,...,fn,label` (label integer, last column).
//! * [`read_binary_csv`]  — same layout but features are `0`/`1`.
//! * [`read_sparse_binary`] — `label idx1 idx2 ...` per line (set-bit indices),
//!   compact for high-dimensional bag-of-words data.

use std::fs;
use std::io;

/// Convenience constructor for an `InvalidData` I/O error with the given message.
fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// Read a CSV of numeric features with an integer label in the last column.
/// A non-numeric first line is treated as a header and skipped.
pub fn read_numeric_csv(path: &str) -> io::Result<(Vec<Vec<f64>>, Vec<usize>)> {
    let text = fs::read_to_string(path)?;
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 2 {
            continue;
        }
        let parsed: Result<Vec<f64>, _> = cols.iter().map(|c| c.trim().parse::<f64>()).collect();
        let vals = match parsed {
            Ok(v) => v,
            Err(_) if i == 0 => continue, // header
            Err(_) => return Err(invalid(format!("non-numeric data on line {}", i + 1))),
        };
        let (feat, label) = vals.split_at(vals.len() - 1);
        xs.push(feat.to_vec());
        ys.push(label[0].round() as usize);
    }
    Ok((xs, ys))
}

/// Read a CSV of `0`/`1` features with an integer label in the last column.
/// Parses values as integers directly, avoiding f64 conversion overhead.
pub fn read_binary_csv(path: &str) -> io::Result<(Vec<Vec<u8>>, Vec<usize>)> {
    let text = fs::read_to_string(path)?;
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 2 {
            continue;
        }
        // Skip a non-numeric header row.
        if i == 0 && cols[0].trim().parse::<i64>().is_err() {
            continue;
        }
        let n = cols.len() - 1;
        let row: Result<Vec<u8>, _> = cols[..n]
            .iter()
            .map(|c| c.trim().parse::<i64>().map(|v| (v != 0) as u8))
            .collect();
        let row = row.map_err(|_| invalid(format!("non-integer value on line {}", i + 1)))?;
        let label: usize = cols[n]
            .trim()
            .parse()
            .map_err(|_| invalid(format!("bad label on line {}", i + 1)))?;
        xs.push(row);
        ys.push(label);
    }
    Ok((xs, ys))
}

/// Read a sparse binary file: each line is `label idx1 idx2 ...`, where the
/// indices are the positions of `1` bits in a length-`n_features` vector.
pub fn read_sparse_binary(path: &str, n_features: usize) -> io::Result<(Vec<Vec<u8>>, Vec<usize>)> {
    let text = fs::read_to_string(path)?;
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let mut it = line.split_whitespace();
        let label = match it.next() {
            Some(t) => t
                .parse::<usize>()
                .map_err(|_| invalid(format!("bad label on line {}", i + 1)))?,
            None => continue,
        };
        let mut row = vec![0u8; n_features];
        for tok in it {
            let idx: usize = tok
                .parse()
                .map_err(|_| invalid(format!("bad index on line {}", i + 1)))?;
            if idx < n_features {
                row[idx] = 1;
            }
        }
        xs.push(row);
        ys.push(label);
    }
    Ok((xs, ys))
}
