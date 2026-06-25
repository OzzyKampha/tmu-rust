/// Quantile booleanizer: converts numeric features to binary using per-feature quantile thresholds.
///
/// For each feature, `n_thresholds` evenly-spaced quantile cut points are computed from the
/// training set. `transform_row` maps each value to 1 if it exceeds the threshold, else 0.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Booleanizer {
    thresholds: Vec<Vec<f64>>,
}

impl Booleanizer {
    /// Fit on `xs` (rows of length `n_features`), producing `n_thresholds` quantile
    /// thresholds per feature.
    pub fn fit(xs: &[&[f64]], n_features: usize, n_thresholds: usize) -> Self {
        let n = xs.len();
        let mut thresholds = Vec::with_capacity(n_features);
        for f in 0..n_features {
            let mut vals: Vec<f64> = xs.iter().map(|row| row[f]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mut thr = Vec::with_capacity(n_thresholds);
            for t in 0..n_thresholds {
                let idx = ((n - 1) * (t + 1)) / (n_thresholds + 1);
                thr.push(vals[idx]);
            }
            thresholds.push(thr);
        }
        Booleanizer { thresholds }
    }

    /// Total number of binary output features.
    pub fn n_output_features(&self) -> usize {
        self.thresholds.iter().map(|t| t.len()).sum()
    }

    /// Encode one row of numeric features into `out` (must have length `n_output_features()`).
    pub fn transform_row(&self, row: &[f64], out: &mut [u8]) {
        let mut bit = 0;
        for (f, thrs) in self.thresholds.iter().enumerate() {
            for &t in thrs {
                out[bit] = (row[f] > t) as u8;
                bit += 1;
            }
        }
    }

    /// Returns `(feature_index, threshold)` for the given output bit index.
    pub fn bit_origin(&self, bit: usize) -> (usize, f64) {
        let mut remaining = bit;
        for (f, thrs) in self.thresholds.iter().enumerate() {
            if remaining < thrs.len() {
                return (f, thrs[remaining]);
            }
            remaining -= thrs.len();
        }
        panic!(
            "bit index {bit} out of range (total bits: {})",
            self.n_output_features()
        );
    }
}
