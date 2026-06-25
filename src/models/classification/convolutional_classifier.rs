//! Convolutional Tsetlin Machine classifier.
//!
//! Mirrors TMU's convolutional TM variant.
//!
//! Each clause operates on a fixed-size receptive field (*patch*) of
//! `kernel_size` consecutive input features rather than the full input.
//! During prediction the clause is applied to every patch position and
//! weighted votes are summed across positions.  During training a single
//! random patch is chosen per clause per sample for feedback, implementing
//! weight tying across spatial positions.
//!
//! Only 1-D receptive fields (sequences / flat feature vectors) are supported
//! in this port.  2-D (image) inputs can be handled by pre-flattening rows.

#[cfg(feature = "parallel")]
use crate::clause_bank::dense::PARALLEL_MIN;
use crate::clause_bank::dense::{
    bmask_word, clause_fire, digits_of, expand_bits_to_bytes, fire_predict, pack, rebuild_include,
    type_i_update_bytes, type_ii_update_bytes, words_for, GOLDEN, MASK_BITS, WORD_BITS,
};
use crate::rng::Rng;

/// Convolutional Tsetlin Machine for multiclass classification over structured inputs.
///
/// ## Layout
///
/// Given `n_input_features` input features, `kernel_size`, and `stride`:
/// - `n_patches = (n_input_features − kernel_size) / stride + 1` patch positions
/// - Each clause has `n_literals = 2 * kernel_size` literals (one positive + one negated per patch feature)
/// - Clause weight tying: the same clause fires at every position; votes are summed
///
/// ## Training
///
/// For each sample the feedback probability for class `c` is computed from the
/// clamped vote sum across all patches.  Then for each clause a *single random
/// patch* is drawn to apply Type I / II TA updates (weight-tied learning).
///
/// ## Example
///
/// ```rust,no_run
/// use tmu_rs::ConvolutionalTsetlinMachine;
///
/// // 16 features, kernel=4, stride=2  →  7 patch positions
/// let mut ctm = ConvolutionalTsetlinMachine::new(2, 16, 4, 2, 20, 100, 3.9);
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConvolutionalTsetlinMachine {
    n_classes: usize,
    n_input_features: usize,
    kernel_size: usize,
    stride: usize,
    n_patches: usize,
    n_literals: usize,
    words: usize,
    clauses_per_class: usize,
    threshold: i32,
    s: f64,
    boost_true_positive: bool,
    max_included_literals: usize,
    clause_drop_p: f64,
    literal_drop_p: f64,
    literal_rng: Rng,
    dig_lit_active: Vec<u8>,

    /// TA counters: clause `cj = c*CPC + j` occupies `ta[cj*n_literals..(cj+1)*n_literals]`.
    ta: Vec<u8>,
    /// Include bitset: clause `cj` occupies `include[cj*words..(cj+1)*words]`.
    include: Vec<u64>,
    half: u8,
    max_state: u8,

    weights: Vec<i32>,
    rngs: Vec<Rng>,
    class_rngs: Vec<Rng>,
    valid: Vec<u64>,
    dig_inv: Vec<u8>,
    dig_keep: Vec<u8>,

    rng: Rng,
}

#[cfg(feature = "serde")]
impl crate::serial::SaveLoad for ConvolutionalTsetlinMachine {
    const TAG: u8 = crate::serial::TAG_CONVOLUTIONAL;
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn apply_one_clause(
    j: usize,
    ta: &mut [u8],
    inc: &mut [u64],
    w: &mut i32,
    rng: &mut Rng,
    target: u8,
    p: f64,
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
    if rng.next_f64() > p {
        return;
    }
    let positive = j & 1 == 0;
    if (target == 1) == positive {
        let fired = clause_fire(inc, lit, val, words, lit_active);
        let under_limit = max_inc == usize::MAX || {
            let n: u32 = (0..words).map(|k| (inc[k] & val[k]).count_ones()).sum();
            (n as usize) < max_inc
        };
        let fired_under = fired && under_limit;
        if fired_under {
            *w = (*w + 1).min(wmax);
        }
        type_i_update_bytes(
            ta, n_literals, fired_under, boost, lit_b, inv_b, keep_b, active_b, max_state,
        );
    } else {
        if !clause_fire(inc, lit, val, words, lit_active) {
            return;
        }
        *w = (*w - 1).max(1);
        type_ii_update_bytes(ta, n_literals, lit_b, active_b, half, max_state);
    }
    rebuild_include(ta, inc, val, words, n_literals, half);
}

impl ConvolutionalTsetlinMachine {
    /// Create a convolutional TM with default settings: 8 state bits, boost enabled, seed 42.
    ///
    /// * `n_input_features` — total number of input features.
    /// * `kernel_size` — width of each receptive field patch; must be ≤ `n_input_features`.
    /// * `stride` — step between patches (1 = fully overlapping).
    /// * `clauses_per_class` — must be even and ≥ 2.
    pub fn new(
        n_classes: usize,
        n_input_features: usize,
        kernel_size: usize,
        stride: usize,
        clauses_per_class: usize,
        threshold: i32,
        s: f64,
    ) -> Self {
        Self::with_config(
            n_classes,
            n_input_features,
            kernel_size,
            stride,
            clauses_per_class,
            threshold,
            s,
            8,
            true,
            42,
        )
    }

    /// Create a convolutional TM with full configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn with_config(
        n_classes: usize,
        n_input_features: usize,
        kernel_size: usize,
        stride: usize,
        clauses_per_class: usize,
        threshold: i32,
        s: f64,
        state_bits: u8,
        boost_true_positive: bool,
        seed: u64,
    ) -> Self {
        assert!(n_classes >= 2);
        assert!(kernel_size >= 1 && kernel_size <= n_input_features);
        assert!(stride >= 1);
        assert!(clauses_per_class >= 2 && clauses_per_class.is_multiple_of(2));
        assert!(threshold >= 1);
        assert!(s > 1.0);
        assert!((2..=8).contains(&state_bits));

        let n_patches = (n_input_features - kernel_size) / stride + 1;
        assert!(n_patches >= 1, "kernel_size and stride must yield at least 1 patch");

        let state_bits = state_bits as usize;
        let n_literals = 2 * kernel_size;
        let words = words_for(n_literals);
        let n_clauses = n_classes * clauses_per_class;
        let mut rng = Rng::new(seed);

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
                ta[tb + l] = if rng.next_u64() & 1 == 0 { half - 1 } else { half };
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
        let class_rngs = (0..n_classes)
            .map(|c| Rng::new(seed ^ (c as u64 + n_clauses as u64 + 1).wrapping_mul(GOLDEN)))
            .collect();
        let literal_rng = Rng::new(seed ^ 0x4C49_5445_5241_4C21u64);

        ConvolutionalTsetlinMachine {
            n_classes,
            n_input_features,
            kernel_size,
            stride,
            n_patches,
            n_literals,
            words,
            clauses_per_class,
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
            class_rngs,
            valid,
            dig_inv: digits_of(1.0 / s, MASK_BITS),
            dig_keep: digits_of((s - 1.0) / s, MASK_BITS),
            rng,
        }
    }

    /// Limit how many literals each clause may include (Type Ia guard).
    pub fn max_included_literals(mut self, max: usize) -> Self {
        self.max_included_literals = max;
        self
    }

    /// Per-clause dropout probability during training.
    pub fn clause_drop_p(mut self, p: f64) -> Self {
        assert!((0.0..1.0).contains(&p));
        self.clause_drop_p = p;
        self
    }

    /// Per-literal dropout probability during training.
    pub fn literal_drop_p(mut self, p: f64) -> Self {
        assert!((0.0..1.0).contains(&p));
        self.literal_drop_p = p;
        self.dig_lit_active = digits_of(1.0 - p, MASK_BITS);
        self
    }

    // ---- accessors -----------------------------------------------------------

    pub fn n_classes(&self) -> usize { self.n_classes }
    pub fn n_input_features(&self) -> usize { self.n_input_features }
    pub fn kernel_size(&self) -> usize { self.kernel_size }
    pub fn stride(&self) -> usize { self.stride }
    pub fn n_patches(&self) -> usize { self.n_patches }
    pub fn clauses_per_class(&self) -> usize { self.clauses_per_class }

    // ---- patch packing -------------------------------------------------------

    /// Pack patch `p_idx` from a raw feature vector into `out`.
    fn pack_patch(&self, x: &[u8], p_idx: usize, out: &mut [u64]) {
        let start = p_idx * self.stride;
        pack(&x[start..start + self.kernel_size], self.kernel_size, out);
    }

    /// Pack all patches from `x` into a pre-allocated flat buffer
    /// (row-major: patch 0 at `[0..words]`, patch 1 at `[words..2*words]`, …).
    fn pack_all_patches(&self, x: &[u8], buf: &mut [u64]) {
        let w = self.words;
        for p in 0..self.n_patches {
            self.pack_patch(x, p, &mut buf[p * w..(p + 1) * w]);
        }
    }

    // ---- inference -----------------------------------------------------------

    /// Compute clamped weighted clause sums for each class from pre-packed patches.
    ///
    /// `patches_buf` is a flat row-major array of packed patches
    /// (`n_patches * words` elements).
    fn class_scores_from_patches(&self, patches_buf: &[u64], out: &mut [i32]) {
        let cps = self.clauses_per_class;
        let words = self.words;
        let inc = self.include.as_slice();
        let val = self.valid.as_slice();

        for (c, score) in out.iter_mut().enumerate() {
            let cw = &self.weights[c * cps..(c + 1) * cps];
            let mut sum = 0i32;
            for (j, &w) in cw.iter().enumerate() {
                let cj = c * cps + j;
                let clause_inc = &inc[cj * words..(cj + 1) * words];
                // Sum over all patch positions.
                for p in 0..self.n_patches {
                    let patch = &patches_buf[p * words..(p + 1) * words];
                    if fire_predict(clause_inc, patch, val, words) {
                        if j & 1 == 0 { sum += w; } else { sum -= w; }
                    }
                }
            }
            *score = sum.clamp(-self.threshold, self.threshold);
        }
    }

    /// Predict the class for a raw (unpacked) input vector.
    ///
    /// `x` must have exactly `n_input_features` 0/1 elements.
    pub fn predict(&self, x: &[u8]) -> usize {
        assert_eq!(x.len(), self.n_input_features);
        let mut patches_buf = vec![0u64; self.n_patches * self.words];
        self.pack_all_patches(x, &mut patches_buf);
        let mut scores = vec![0i32; self.n_classes];
        self.class_scores_from_patches(&patches_buf, &mut scores);
        scores.iter().enumerate().max_by_key(|&(_, &v)| v).map(|(i, _)| i).unwrap()
    }

    /// Fill `out` with the clamped weighted clause sums for each class.
    pub fn scores(&self, x: &[u8], out: &mut [i32]) {
        assert_eq!(x.len(), self.n_input_features);
        assert_eq!(out.len(), self.n_classes);
        let mut patches_buf = vec![0u64; self.n_patches * self.words];
        self.pack_all_patches(x, &mut patches_buf);
        self.class_scores_from_patches(&patches_buf, out);
    }

    /// Predict classes for a batch of raw input vectors.
    pub fn predict_batch(&self, xs: &[&[u8]]) -> Vec<usize> {
        xs.iter().map(|x| self.predict(x)).collect()
    }

    // ---- training ------------------------------------------------------------

    /// Compute the training-mode clause sum for class `c` over all patches.
    ///
    /// Uses `clause_fire` (respects literal dropout mask `lit_active`) and sums
    /// across all `n_patches` patch positions.
    fn class_sum_train(&self, c: usize, patches_buf: &[u64], lit_active: &[u64]) -> i32 {
        let cps = self.clauses_per_class;
        let words = self.words;
        let inc = self.include.as_slice();
        let val = self.valid.as_slice();
        let mut sum = 0i32;
        for j in 0..cps {
            let cj = c * cps + j;
            let clause_inc = &inc[cj * words..(cj + 1) * words];
            for p in 0..self.n_patches {
                let patch = &patches_buf[p * words..(p + 1) * words];
                if clause_fire(clause_inc, patch, val, words, lit_active) {
                    let w = self.weights[c * cps + j];
                    if j & 1 == 0 { sum += w; } else { sum -= w; }
                }
            }
        }
        sum.clamp(-self.threshold, self.threshold)
    }

    /// Apply feedback to all clauses of class `c`.
    ///
    /// Each clause picks a random patch for its TA update, implementing weight
    /// tying: the same clause weights govern all patch positions.
    fn update_class(
        &mut self,
        c: usize,
        target: u8,
        sum: i32,
        patches_buf: &[u64],
        lit_active: &[u64],
    ) {
        let cps = self.clauses_per_class;
        let words = self.words;
        let n_literals = self.n_literals;
        let boost = self.boost_true_positive;
        let wmax = self.threshold;
        let max_inc = self.max_included_literals;
        let drop_p = self.clause_drop_p;
        let half = self.half;
        let max_state = self.max_state;
        let n_patches = self.n_patches;
        let cw = self.class_weights_dummy();

        let t = wmax as f64;
        let v = sum as f64;
        let p = if target == 1 {
            ((t - v) / (2.0 * t) * cw).min(1.0)
        } else {
            ((t + v) / (2.0 * t) * cw).min(1.0)
        };

        let Self {
            ta,
            include,
            weights,
            rngs,
            class_rngs,
            valid,
            dig_inv,
            dig_keep,
            rng: _,
            ..
        } = self;

        let crng = &mut class_rngs[c];

        let drop_mask: Vec<bool> = if drop_p > 0.0 {
            (0..cps).map(|_| crng.next_f64() < drop_p).collect()
        } else {
            vec![]
        };

        let inv_mask: Vec<u64> = (0..words).map(|_| bmask_word(crng, dig_inv)).collect();
        let keep_mask: Vec<u64> = (0..words).map(|_| bmask_word(crng, dig_keep)).collect();
        let inv_b = expand_bits_to_bytes(&inv_mask, n_literals);
        let keep_b = expand_bits_to_bytes(&keep_mask, n_literals);
        let active_b = expand_bits_to_bytes(lit_active, n_literals);

        let val = valid.as_slice();
        let class_ta = &mut ta[c * cps * n_literals..(c + 1) * cps * n_literals];
        let class_inc = &mut include[c * cps * words..(c + 1) * cps * words];
        let class_w = &mut weights[c * cps..(c + 1) * cps];
        let class_rng = &mut rngs[c * cps..(c + 1) * cps];

        #[cfg(feature = "parallel")]
        if cps >= PARALLEL_MIN {
            use rayon::prelude::*;
            class_ta
                .par_chunks_mut(n_literals)
                .zip(class_inc.par_chunks_mut(words))
                .zip(class_w.par_iter_mut())
                .zip(class_rng.par_iter_mut())
                .enumerate()
                .for_each(|(j, (((ta_c, inc_c), w), rng_c))| {
                    // Each clause picks a random patch (per-clause RNG).
                    let p_idx = rng_c.below(n_patches);
                    let patch = &patches_buf[p_idx * words..(p_idx + 1) * words];
                    let lit_b = expand_bits_to_bytes(patch, n_literals);
                    apply_one_clause(
                        j, ta_c, inc_c, w, rng_c, target, p, &drop_mask, patch, val, lit_active,
                        words, &lit_b, &inv_b, &keep_b, &active_b, n_literals, boost, wmax,
                        max_inc, half, max_state,
                    );
                });
            return;
        }

        for j in 0..cps {
            // Each clause picks a random patch for feedback.
            let p_idx = class_rng[j].below(n_patches);
            let patch = &patches_buf[p_idx * words..(p_idx + 1) * words];
            let lit_b = expand_bits_to_bytes(patch, n_literals);
            apply_one_clause(
                j,
                &mut class_ta[j * n_literals..(j + 1) * n_literals],
                &mut class_inc[j * words..(j + 1) * words],
                &mut class_w[j],
                &mut class_rng[j],
                target,
                p,
                &drop_mask,
                patch,
                val,
                lit_active,
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

    fn class_weights_dummy(&self) -> f64 {
        1.0
    }

    /// Train on a single raw (unpacked) input vector with class label `y`.
    pub fn fit_one(&mut self, x: &[u8], y: usize) {
        assert_eq!(x.len(), self.n_input_features);
        assert!(y < self.n_classes);

        // Pack all patches once.
        let mut patches_buf = vec![0u64; self.n_patches * self.words];
        self.pack_all_patches(x, &mut patches_buf);

        // Sample a random negative class.
        let mut neg = self.rng.below(self.n_classes);
        while neg == y {
            neg = self.rng.below(self.n_classes);
        }

        // Generate per-sample literal-active mask (shared across patches and classes).
        let lit_active: Vec<u64> = if self.literal_drop_p > 0.0 {
            let rng = &mut self.literal_rng;
            let dig = &self.dig_lit_active;
            let w = self.words;
            (0..w).map(|_| bmask_word(rng, dig)).collect()
        } else {
            vec![!0u64; self.words]
        };

        let sum_y = self.class_sum_train(y, &patches_buf, &lit_active);
        let sum_neg = self.class_sum_train(neg, &patches_buf, &lit_active);

        self.update_class(y, 1, sum_y, &patches_buf, &lit_active);
        self.update_class(neg, 0, sum_neg, &patches_buf, &lit_active);
    }

    /// Run one training epoch over a dataset of raw inputs.
    ///
    /// `xs` — raw feature vectors (each must have `n_input_features` 0/1 elements).
    /// `ys` — class labels.
    pub fn fit_epoch(&mut self, xs: &[&[u8]], ys: &[usize]) {
        let n = xs.len();
        assert_eq!(n, ys.len());
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let k = self.rng.below(i + 1);
            order.swap(i, k);
        }
        for &i in &order {
            self.fit_one(xs[i], ys[i]);
        }
    }

    // ---- metrics -------------------------------------------------------------

    /// Fraction of correctly predicted samples.
    pub fn accuracy(&self, xs: &[&[u8]], ys: &[usize]) -> f64 {
        assert_eq!(xs.len(), ys.len());
        let correct = xs.iter().zip(ys).filter(|(&x, &y)| self.predict(x) == y).count();
        correct as f64 / xs.len() as f64
    }

    // ---- interpretability ----------------------------------------------------

    /// Return the included literals for clause `clause` of `class`
    /// as `(patch_feature_index, is_negated)` pairs.
    ///
    /// Indices are relative to the patch (`0..kernel_size`).
    pub fn clause_rule(&self, class: usize, clause: usize) -> Vec<(usize, bool)> {
        let cj = class * self.clauses_per_class + clause;
        let inc = &self.include[cj * self.words..(cj + 1) * self.words];
        let mut rule = Vec::new();
        for l in 0..self.n_literals {
            if (inc[l / WORD_BITS] >> (l % WORD_BITS)) & 1 != 0 {
                if l < self.kernel_size {
                    rule.push((l, false));
                } else {
                    rule.push((l - self.kernel_size, true));
                }
            }
        }
        rule
    }

    /// Return `true` if `clause` is a positive clause (even index within class).
    pub fn clause_is_positive(&self, clause: usize) -> bool {
        clause & 1 == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_xor_sequence(n: usize, n_features: usize, seed: u64) -> (Vec<Vec<u8>>, Vec<usize>) {
        let mut rng = Rng::new(seed);
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for _ in 0..n {
            let f: Vec<u8> = (0..n_features).map(|_| (rng.next_u64() & 1) as u8).collect();
            let y = (f[0] ^ f[1]) as usize;
            ys.push(y);
            xs.push(f);
        }
        (xs, ys)
    }

    fn as_slices(xs: &[Vec<u8>]) -> Vec<&[u8]> {
        xs.iter().map(|v| v.as_slice()).collect()
    }

    #[test]
    fn convolutional_constructs() {
        let ctm = ConvolutionalTsetlinMachine::new(2, 16, 4, 2, 20, 100, 3.9);
        assert_eq!(ctm.n_classes(), 2);
        assert_eq!(ctm.kernel_size(), 4);
        assert_eq!(ctm.n_patches(), 7); // (16 - 4) / 2 + 1 = 7
    }

    #[test]
    fn convolutional_stride_1_n_patches() {
        let ctm = ConvolutionalTsetlinMachine::new(2, 10, 3, 1, 4, 10, 2.0);
        assert_eq!(ctm.n_patches(), 8); // (10 - 3) / 1 + 1 = 8
    }

    #[test]
    fn convolutional_full_kernel_one_patch() {
        let ctm = ConvolutionalTsetlinMachine::new(2, 8, 8, 1, 4, 10, 2.0);
        assert_eq!(ctm.n_patches(), 1);
    }

    #[test]
    fn convolutional_predict_returns_valid_class() {
        let ctm = ConvolutionalTsetlinMachine::new(3, 12, 4, 2, 8, 20, 3.0);
        let x = vec![0u8, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1];
        let pred = ctm.predict(&x);
        assert!(pred < 3);
    }

    #[test]
    fn convolutional_trains_without_panic() {
        let (xs, ys) = make_xor_sequence(200, 12, 1);
        let slices = as_slices(&xs);
        let mut ctm = ConvolutionalTsetlinMachine::new(2, 12, 4, 2, 10, 20, 3.0);
        for _ in 0..3 {
            ctm.fit_epoch(&slices, &ys);
        }
        let acc = ctm.accuracy(&slices, &ys);
        assert!((0.0..=1.0).contains(&acc));
    }

    #[test]
    fn convolutional_learns_xor_in_first_features() {
        // 4 features, kernel=2, stride=1 → 3 patches ([0,1], [1,2], [2,3]).
        // XOR of features 0,1 lives in patch 0; patches 1,2 add noise (2 features
        // irrelevant).  60 clauses and 40 epochs are sufficient to learn through
        // the signal dilution from the two irrelevant patches.
        let (xtr, ytr) = make_xor_sequence(3000, 4, 1);
        let (xte, yte) = make_xor_sequence(800, 4, 2);
        let tr = as_slices(&xtr);
        let te = as_slices(&xte);
        let mut ctm = ConvolutionalTsetlinMachine::with_config(2, 4, 2, 1, 60, 50, 3.5, 8, true, 42);
        for _ in 0..40 {
            ctm.fit_epoch(&tr, &ytr);
        }
        let acc = ctm.accuracy(&te, &yte);
        assert!(acc > 0.65, "accuracy {acc:.3} should exceed 65% for learnable XOR in 3-patch setup");
    }

    #[test]
    fn convolutional_clause_rule_feature_indices_in_range() {
        let ctm = ConvolutionalTsetlinMachine::new(2, 12, 4, 2, 8, 20, 3.0);
        for rule_feat in ctm.clause_rule(0, 0).iter().map(|&(f, _)| f) {
            assert!(rule_feat < ctm.kernel_size(), "feature index out of patch range");
        }
    }

    #[test]
    fn convolutional_scores_argmax_matches_predict() {
        let (xs, ys) = make_xor_sequence(200, 8, 3);
        let slices = as_slices(&xs);
        let mut ctm = ConvolutionalTsetlinMachine::new(2, 8, 2, 1, 10, 20, 3.0);
        for _ in 0..5 {
            ctm.fit_epoch(&slices, &ys);
        }
        let mut out = vec![0i32; 2];
        for x in &xs[..20] {
            ctm.scores(x, &mut out);
            let argmax = out.iter().enumerate().max_by_key(|&(_, &v)| v).map(|(i, _)| i).unwrap();
            assert_eq!(argmax, ctm.predict(x));
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn convolutional_save_load_roundtrip() {
        use crate::serial::SaveLoad;
        let (xs, ys) = make_xor_sequence(300, 8, 5);
        let slices = as_slices(&xs);
        let mut ctm = ConvolutionalTsetlinMachine::new(2, 8, 2, 1, 10, 20, 3.0);
        for _ in 0..5 {
            ctm.fit_epoch(&slices, &ys);
        }
        let tmp = std::env::temp_dir().join("test_ctm.tmrs");
        ctm.save(&tmp).unwrap();
        let loaded = ConvolutionalTsetlinMachine::load(&tmp).unwrap();
        let _ = std::fs::remove_file(&tmp);
        for x in &xs[..50] {
            assert_eq!(ctm.predict(x), loaded.predict(x));
        }
    }
}
