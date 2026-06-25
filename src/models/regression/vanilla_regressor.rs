//! Vanilla Tsetlin Machine regressor.
//!
//! Mirrors TMU's `vanilla_regressor.py` / `TMRegressor`.

#[cfg(feature = "parallel")]
use crate::clause_bank::dense::PARALLEL_MIN;
use crate::clause_bank::dense::{
    bmask_word, clause_fire, digits_of, expand_bits_to_bytes, fire_predict, rebuild_include,
    type_i_update_bytes, type_ii_update_bytes, words_for, GOLDEN, MASK_BITS, WORD_BITS,
};
use crate::encoder::{EncodedBatch, EncodedSample};
use crate::rng::Rng;

/// A weighted Tsetlin Machine for continuous-output regression.
///
/// All clauses start with weight +1 and contribute `weight[j] * fires(j, x)` to
/// the vote sum.  The prediction is `clamp(sum, 0, threshold)` returned as `f64`.
/// Weights are in `[0, threshold]` — they floor at 0 (never go negative).
///
/// Training targets must be in `[0.0, threshold as f64]`.  The feedback rule
/// matches TMU's `TMRegressor` (vanilla_regressor.py):
/// - When `pred < y` (push up): Type I to all active clauses, increment weight
///   if the clause fires; `update_p = ((pred − y) / T)²`.
/// - When `pred > y` (push down): Type II to fired clauses only, decrement weight;
///   same `update_p`.
///
/// Clause polarity is learned through weight dynamics, not hardcoded by index.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TMRegressor {
    n_features: usize,
    n_literals: usize,
    words: usize,
    n_clauses: usize,
    threshold: i32,
    s: f64,
    boost_true_positive: bool,
    max_included_literals: usize,
    clause_drop_p: f64,
    literal_drop_p: f64,
    literal_rng: Rng,
    dig_lit_active: Vec<u8>,

    /// u8 TA counters.  Clause `j` occupies `ta[j * n_literals .. (j+1) * n_literals]`.
    ta: Vec<u8>,
    /// Include bitset.  Clause `j` occupies `include[j * words .. (j+1) * words]`.
    include: Vec<u64>,
    half: u8,
    max_state: u8,

    /// Per-clause integer weights (>= 1).
    weights: Vec<i32>,
    /// Per-clause RNG for lock-free parallel training.
    rngs: Vec<Rng>,
    valid: Vec<u64>,
    dig_inv: Vec<u8>,
    dig_keep: Vec<u8>,

    rng: Rng,
}

#[cfg(feature = "serde")]
impl crate::serial::SaveLoad for TMRegressor {
    const TAG: u8 = crate::serial::TAG_REGRESSOR;
}

/// Per-clause feedback kernel — matches TMU's vanilla_regressor.py.
///
/// `push_up=true`  (pred < target): Type I to every active clause; increment weight if fired.
/// `push_up=false` (pred > target): Type II only to fired clauses; decrement weight.
///
/// `update_p = ((pred - target) / T)²` — squared relative error, same for all clauses.
/// Weights may go negative (their sign is learned, not hardcoded by clause index).
#[allow(clippy::too_many_arguments)]
#[inline]
fn apply_one_clause(
    j: usize,
    ta: &mut [u8],
    inc: &mut [u64],
    w: &mut i32,
    rng: &mut Rng,
    push_up: bool,
    update_p: f64,
    drop_mask: &[bool],
    lit: &[u64],
    val: &[u64],
    lit_active: &[u64],
    words: usize,
    lit_b: &[u8],
    inv_b: &[u8],
    keep_b: &[u8],
    active_b: &[u8],
    n_literals: usize,
    boost: bool,
    wmax: i32,
    max_inc: usize,
    half: u8,
    max_state: u8,
) {
    if !drop_mask.is_empty() && drop_mask[j] {
        return;
    }
    if rng.next_f64() > update_p {
        return;
    }
    let fired = clause_fire(inc, lit, val, words, lit_active);
    if push_up {
        // Type I (Ia if fired and under literal limit, Ib otherwise).
        let under_limit = max_inc == usize::MAX || {
            let n: u32 = (0..words).map(|k| (inc[k] & val[k]).count_ones()).sum();
            (n as usize) < max_inc
        };
        let fired_under = fired && under_limit;
        if fired_under {
            *w = (*w + 1).min(wmax);
        }
        type_i_update_bytes(
            ta,
            n_literals,
            fired_under,
            boost,
            lit_b,
            inv_b,
            keep_b,
            active_b,
            max_state,
        );
    } else {
        // Type II — only applies when the clause fires.
        if !fired {
            return;
        }
        *w = (*w - 1).max(0); // TMU uses neg_weights=False: floor is 0, not negative
        type_ii_update_bytes(ta, n_literals, lit_b, active_b, half, max_state);
    }
    rebuild_include(ta, inc, val, words, n_literals, half);
}

impl TMRegressor {
    /// Create a regressor with default settings: 8 state bits, boost enabled, seed 42.
    pub fn new(n_features: usize, n_clauses: usize, threshold: i32, s: f64) -> Self {
        Self::with_config(n_features, n_clauses, threshold, s, 8, true, 42)
    }

    /// Create a regressor with full configuration.
    ///
    /// * `n_clauses` — total clause count (>= 2).
    /// * `threshold` — output clamped to `[0, threshold]`; training targets should be in this range.
    /// * `s` — specificity (> 1.0).
    /// * `state_bits` — TA counter precision (2–8 bits).
    /// * `boost_true_positive` — if `true`, Type Ia feedback always includes present literals.
    /// * `seed` — master RNG seed.
    pub fn with_config(
        n_features: usize,
        n_clauses: usize,
        threshold: i32,
        s: f64,
        state_bits: u8,
        boost_true_positive: bool,
        seed: u64,
    ) -> Self {
        assert!(n_features >= 1);
        assert!(n_clauses >= 2, "n_clauses must be >= 2");
        assert!(threshold >= 1);
        assert!(s > 1.0);
        assert!((2..=8).contains(&state_bits), "state_bits must be in 2..=8");

        let state_bits = state_bits as usize;
        let n_literals = 2 * n_features;
        let words = words_for(n_literals);
        let rng = Rng::new(seed);

        let mut valid = vec![0u64; words];
        for l in 0..n_literals {
            valid[l / WORD_BITS] |= 1u64 << (l % WORD_BITS);
        }

        let half = 1u8 << (state_bits - 1);
        let max_state = ((1u16 << state_bits) - 1) as u8;

        let mut ta = vec![0u8; n_clauses * n_literals];
        let mut include = vec![0u64; n_clauses * words];
        for cj in 0..n_clauses {
            let tb = cj * n_literals;
            for l in 0..n_literals {
                ta[tb + l] = half - 1; // all start in exclude state (matches TMU)
            }
            rebuild_include(
                &ta[tb..tb + n_literals],
                &mut include[cj * words..(cj + 1) * words],
                &valid,
                words,
                n_literals,
                half,
            );
        }

        let rngs = (0..n_clauses)
            .map(|i| Rng::new(seed ^ (i as u64).wrapping_add(1).wrapping_mul(GOLDEN)))
            .collect();

        let literal_rng = Rng::new(seed ^ 0x4C49_5445_5241_4C21u64);

        TMRegressor {
            n_features,
            n_literals,
            words,
            n_clauses,
            threshold,
            s,
            boost_true_positive,
            max_included_literals: usize::MAX,
            clause_drop_p: 0.0,
            literal_drop_p: 0.0,
            literal_rng,
            dig_lit_active: digits_of(1.0, MASK_BITS),
            ta,
            include,
            half,
            max_state,
            weights: vec![1i32; n_clauses],
            rngs,
            valid,
            dig_inv: digits_of(1.0 / s, MASK_BITS),
            dig_keep: digits_of((s - 1.0) / s, MASK_BITS),
            rng,
        }
    }

    /// Limit how many literals each clause may include (Type Ia guard).
    /// Mirrors TMU's `max_included_literals`.
    pub fn max_included_literals(mut self, max: usize) -> Self {
        self.max_included_literals = max;
        self
    }

    /// Per-clause dropout probability during training (default: 0.0).
    pub fn clause_drop_p(mut self, p: f64) -> Self {
        assert!((0.0..1.0).contains(&p), "clause_drop_p must be in [0, 1)");
        self.clause_drop_p = p;
        self
    }

    /// Per-literal dropout probability during training (default: 0.0).
    pub fn literal_drop_p(mut self, p: f64) -> Self {
        assert!((0.0..1.0).contains(&p), "literal_drop_p must be in [0, 1)");
        self.literal_drop_p = p;
        self.dig_lit_active = digits_of(1.0 - p, MASK_BITS);
        self
    }

    // ---- accessors -----------------------------------------------------------

    /// Number of input features.
    pub fn n_features(&self) -> usize {
        self.n_features
    }
    /// Total number of clauses.
    pub fn n_clauses(&self) -> usize {
        self.n_clauses
    }
    /// Output range: predictions are in `[0, threshold]`.
    pub fn threshold(&self) -> i32 {
        self.threshold
    }
    /// Weight of clause `j`.
    pub fn clause_weight(&self, clause: usize) -> i32 {
        self.weights[clause]
    }

    // ---- inference -----------------------------------------------------------

    #[inline]
    fn predict_lit(&self, lit: &[u64]) -> f64 {
        let words = self.words;
        let inc = self.include.as_slice();
        let val = self.valid.as_slice();
        let mut sum = 0i32;
        for j in 0..self.n_clauses {
            if fire_predict(&inc[j * words..(j + 1) * words], lit, val, words) {
                sum += self.weights[j]; // weights may be negative (learned polarity)
            }
        }
        sum.clamp(0, self.threshold) as f64
    }

    /// Predict the continuous output for an encoded sample.
    ///
    /// The return value is in `[0.0, threshold as f64]`.
    pub fn predict(&self, sample: &EncodedSample) -> f64 {
        self.predict_lit(&sample.0)
    }

    /// Predict outputs for all samples in a batch.
    pub fn predict_batch(&self, batch: &EncodedBatch) -> Vec<f64> {
        let packed = batch.data.as_slice();
        let n = batch.n;
        let w = self.words;
        (0..n)
            .map(|i| self.predict_lit(&packed[i * w..(i + 1) * w]))
            .collect()
    }

    // ---- training ------------------------------------------------------------

    fn fit_one_lit(&mut self, lit: &[u64], y: f64) {
        debug_assert_eq!(lit.len(), self.words);

        let v = self.predict_lit(lit);
        let t = self.threshold as f64;

        // Squared relative error (matches TMU vanilla_regressor.py: update_p = (error/T)^2).
        // Larger error → higher probability of feedback; no update when prediction is exact.
        let err = (v - y) / t;
        let update_p = (err * err).min(1.0);
        if update_p < f64::EPSILON {
            return;
        }

        // push_up=true  → pred < target: Type I to all clauses, increment weights of fired clauses
        // push_up=false → pred > target: Type II to fired clauses only, decrement their weights
        let push_up = y > v;

        let n_literals = self.n_literals;
        let words = self.words;
        let n_clauses = self.n_clauses;

        let lit_b = expand_bits_to_bytes(lit, n_literals);

        let lit_active: Vec<u64> = if self.literal_drop_p > 0.0 {
            let rng = &mut self.literal_rng;
            let dig = &self.dig_lit_active;
            (0..words).map(|_| bmask_word(rng, dig)).collect()
        } else {
            vec![!0u64; words]
        };
        let active_b = expand_bits_to_bytes(&lit_active, n_literals);

        let boost = self.boost_true_positive;
        let wmax = self.threshold;
        let max_inc = self.max_included_literals;
        let half = self.half;
        let max_state = self.max_state;
        let drop_p = self.clause_drop_p;

        let Self {
            ta,
            include,
            weights,
            rngs,
            valid,
            dig_inv,
            dig_keep,
            rng,
            ..
        } = self;

        let inv_mask: Vec<u64> = (0..words).map(|_| bmask_word(rng, dig_inv)).collect();
        let keep_mask: Vec<u64> = (0..words).map(|_| bmask_word(rng, dig_keep)).collect();
        let inv_b = expand_bits_to_bytes(&inv_mask, n_literals);
        let keep_b = expand_bits_to_bytes(&keep_mask, n_literals);

        let drop_mask: Vec<bool> = if drop_p > 0.0 {
            (0..n_clauses).map(|_| rng.next_f64() < drop_p).collect()
        } else {
            vec![]
        };

        let val = valid.as_slice();

        #[cfg(feature = "parallel")]
        if n_clauses >= PARALLEL_MIN {
            use rayon::prelude::*;
            ta.par_chunks_mut(n_literals)
                .zip(include.par_chunks_mut(words))
                .zip(weights.par_iter_mut())
                .zip(rngs.par_iter_mut())
                .enumerate()
                .for_each(|(j, (((ta_j, inc_j), w), rng_j))| {
                    apply_one_clause(
                        j,
                        ta_j,
                        inc_j,
                        w,
                        rng_j,
                        push_up,
                        update_p,
                        &drop_mask,
                        lit,
                        val,
                        &lit_active,
                        words,
                        &lit_b,
                        &inv_b,
                        &keep_b,
                        &active_b,
                        n_literals,
                        boost,
                        wmax,
                        max_inc,
                        half,
                        max_state,
                    );
                });
            return;
        }

        for j in 0..n_clauses {
            apply_one_clause(
                j,
                &mut ta[j * n_literals..(j + 1) * n_literals],
                &mut include[j * words..(j + 1) * words],
                &mut weights[j],
                &mut rngs[j],
                push_up,
                update_p,
                &drop_mask,
                lit,
                val,
                &lit_active,
                words,
                &lit_b,
                &inv_b,
                &keep_b,
                &active_b,
                n_literals,
                boost,
                wmax,
                max_inc,
                half,
                max_state,
            );
        }
    }

    /// Train on a single encoded sample with real-valued target `y ∈ [0, threshold]`.
    pub fn fit_one(&mut self, sample: &EncodedSample, y: f64) {
        self.fit_one_lit(&sample.0, y);
    }

    /// Run one training epoch over an encoded batch with continuous targets.
    ///
    /// Targets in `ys` should be in `[0.0, threshold as f64]`.
    /// Samples are presented in a random order each epoch (Fisher-Yates shuffle).
    pub fn fit_epoch(&mut self, batch: &EncodedBatch, ys: &[f64]) {
        debug_assert_eq!(batch.words, self.words);
        let n = batch.n;
        assert_eq!(n, ys.len());
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let k = self.rng.below(i + 1);
            order.swap(i, k);
        }
        let w = self.words;
        let data = batch.data.as_slice();
        for &i in &order {
            self.fit_one_lit(&data[i * w..(i + 1) * w], ys[i]);
        }
    }

    // ---- metrics -------------------------------------------------------------

    /// Mean absolute error over an encoded batch.
    pub fn mae(&self, batch: &EncodedBatch, ys: &[f64]) -> f64 {
        let n = batch.n;
        assert_eq!(n, ys.len());
        let packed = batch.data.as_slice();
        let w = self.words;
        let sum: f64 = (0..n)
            .map(|i| (self.predict_lit(&packed[i * w..(i + 1) * w]) - ys[i]).abs())
            .sum();
        sum / n as f64
    }

    /// Root mean squared error over an encoded batch.
    pub fn rmse(&self, batch: &EncodedBatch, ys: &[f64]) -> f64 {
        let n = batch.n;
        assert_eq!(n, ys.len());
        let packed = batch.data.as_slice();
        let w = self.words;
        let sum: f64 = (0..n)
            .map(|i| {
                let e = self.predict_lit(&packed[i * w..(i + 1) * w]) - ys[i];
                e * e
            })
            .sum();
        (sum / n as f64).sqrt()
    }

    // ---- interpretability ----------------------------------------------------

    /// Return the included literals for clause `j` as `(feature_index, is_negated)` pairs.
    pub fn clause_rule(&self, clause: usize) -> Vec<(usize, bool)> {
        let inc = &self.include[clause * self.words..(clause + 1) * self.words];
        let mut rule = Vec::new();
        for l in 0..self.n_literals {
            if (inc[l / WORD_BITS] >> (l % WORD_BITS)) & 1 != 0 {
                if l < self.n_features {
                    rule.push((l, false));
                } else {
                    rule.push((l - self.n_features, true));
                }
            }
        }
        rule
    }

    /// Return `true` if clause `j` is a positive clause (even index).
    pub fn clause_is_positive(&self, clause: usize) -> bool {
        clause & 1 == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::Encoder;
    use crate::rng::Rng;

    fn make_count_dataset(
        n: usize,
        n_features: usize,
        threshold: i32,
        seed: u64,
    ) -> (Vec<Vec<u8>>, Vec<f64>) {
        let mut rng = Rng::new(seed);
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        let scale = threshold as f64 / n_features as f64;
        for _ in 0..n {
            let f: Vec<u8> = (0..n_features)
                .map(|_| (rng.next_u64() & 1) as u8)
                .collect();
            let count = f.iter().map(|&b| b as usize).sum::<usize>();
            ys.push(count as f64 * scale);
            xs.push(f);
        }
        (xs, ys)
    }

    fn as_slices(xs: &[Vec<u8>]) -> Vec<&[u8]> {
        xs.iter().map(|v| v.as_slice()).collect()
    }

    #[test]
    fn regressor_constructs() {
        let tm = TMRegressor::new(12, 20, 100, 3.9);
        assert_eq!(tm.n_features(), 12);
        assert_eq!(tm.n_clauses(), 20);
        assert_eq!(tm.threshold(), 100);
    }

    #[test]
    fn regressor_predict_in_range() {
        let tm = TMRegressor::new(12, 20, 100, 3.9);
        let enc = Encoder::for_binary(12);
        let sample = enc.encode_one(&[0u8, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1]);
        let v = tm.predict(&sample);
        assert!((0.0..=100.0).contains(&v));
    }

    #[test]
    fn regressor_trains_without_panic() {
        let (xs, ys) = make_count_dataset(500, 10, 100, 1);
        let enc = Encoder::for_binary(10);
        let rows: Vec<&[u8]> = xs.iter().map(|v| v.as_slice()).collect();
        let batch = enc.encode_batch(&rows);
        let mut tm = TMRegressor::new(10, 40, 100, 3.0);
        for _ in 0..5 {
            tm.fit_epoch(&batch, &ys);
        }
        let mae = tm.mae(&batch, &ys);
        assert!(
            mae < 50.0,
            "MAE {mae} should improve beyond random baseline"
        );
    }

    #[test]
    fn regressor_learns_count_function() {
        let (xtr, ytr) = make_count_dataset(2000, 10, 100, 1);
        let (xte, yte) = make_count_dataset(500, 10, 100, 2);
        let enc = Encoder::for_binary(10);
        let btr = enc.encode_batch(&as_slices(&xtr));
        let bte = enc.encode_batch(&as_slices(&xte));
        let mut tm = TMRegressor::new(10, 80, 100, 3.0);
        for _ in 0..30 {
            tm.fit_epoch(&btr, &ytr);
        }
        let mae = tm.mae(&bte, &yte);
        // Loose bound: the count function is learnable; random baseline ≈ T/3 ≈ 33
        assert!(
            mae < 25.0,
            "MAE {mae:.2} should be well below random baseline"
        );
    }

    #[test]
    fn regressor_clause_rule_returns_vec() {
        let tm = TMRegressor::new(8, 10, 50, 3.0);
        let rule = tm.clause_rule(0);
        assert!(rule.iter().all(|&(f, _)| f < 8));
    }

    #[test]
    fn regressor_clause_polarity() {
        let tm = TMRegressor::new(4, 6, 10, 2.0);
        assert!(tm.clause_is_positive(0));
        assert!(!tm.clause_is_positive(1));
        assert!(tm.clause_is_positive(2));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn regressor_save_load_roundtrip() {
        use crate::serial::SaveLoad;
        let (xs, ys) = make_count_dataset(500, 8, 50, 3);
        let enc = Encoder::for_binary(8);
        let rows: Vec<&[u8]> = xs.iter().map(|v| v.as_slice()).collect();
        let batch = enc.encode_batch(&rows);
        let mut tm = TMRegressor::new(8, 20, 50, 3.0);
        for _ in 0..5 {
            tm.fit_epoch(&batch, &ys);
        }
        let tmp = std::env::temp_dir().join("test_regressor.tmrs");
        tm.save(&tmp).unwrap();
        let loaded = TMRegressor::load(&tmp).unwrap();
        let _ = std::fs::remove_file(&tmp);
        let preds_orig = tm.predict_batch(&batch);
        let preds_loaded = loaded.predict_batch(&batch);
        assert_eq!(preds_orig, preds_loaded);
    }
}
