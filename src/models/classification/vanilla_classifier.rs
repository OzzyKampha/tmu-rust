//! Vanilla weighted multiclass Tsetlin Machine classifier.
//!
//! Mirrors TMU's `vanilla_classifier.py` / `TMClassifier`.

use crate::clause_bank::dense::{
    bmask_word, clause_fire, digits_of, expand_bits_to_bytes, fire_predict, rebuild_include,
    type_i_update_bytes, type_ii_update_bytes, words_for, GOLDEN, MASK_BITS, WORD_BITS,
};
#[cfg(feature = "parallel")]
use crate::clause_bank::dense::PARALLEL_MIN;
use crate::encoder::{EncodedBatch, EncodedSample};
use crate::rng::Rng;

/// A weighted multiclass Tsetlin Machine with u8 per-TA counters (matches TMU's 8-bit states).
///
/// Each TA counter is a `u8` in `[0, max_state]`; the include bitset is maintained
/// as a separate `Vec<u64>` for O(words) fire checks.  Optional clause-level parallelism
/// via `--features parallel`.
#[derive(Clone, Debug)]
pub struct TsetlinMachine {
    n_classes: usize,
    n_features: usize,
    n_literals: usize,
    words: usize,
    clauses_per_class: usize,
    threshold: i32,
    s: f64,
    boost_true_positive: bool,
    max_included_literals: usize,
    clause_drop_p: f64,
    /// Per-literal dropout probability during training (mirrors TMU's `literal_drop_p`).
    literal_drop_p: f64,
    /// Dedicated RNG for literal-active mask generation (independent of clause/class RNGs).
    literal_rng: Rng,
    /// Precomputed binary digits for Bernoulli(1 - literal_drop_p) mask generation.
    dig_lit_active: Vec<u8>,

    /// u8 TA counters (matches TMU's 8-bit states).  Clause `cj = c*CPC + j` occupies
    /// `ta[cj * n_literals .. (cj+1) * n_literals]`.
    ta: Vec<u8>,
    /// Include bitset.  Clause `cj` occupies `include[cj * words .. (cj+1) * words]`.
    /// Rebuilt after every clause update; kept in sync with `ta`.
    include: Vec<u64>,
    /// TA threshold for inclusion: `ta[l] >= half` → literal l is included.
    half: u8,
    /// Maximum TA counter value: `(1 << state_bits) - 1`.
    max_state: u8,

    /// Per-clause integer weights (>= 1), indexed `c*CPC + j`.
    weights: Vec<i32>,
    /// Per-clause RNG (enables lock-free parallel training).
    rngs: Vec<Rng>,
    /// Per-class RNG for drop/inv/keep mask generation.
    class_rngs: Vec<Rng>,
    /// Per-word mask of real literal bits.
    valid: Vec<u64>,
    dig_inv: Vec<u8>,
    dig_keep: Vec<u8>,

    literals: Vec<u64>,
    rng: Rng, // for shuffling and negative-class selection only
}

/// Per-clause feedback kernel shared by the sequential and parallel training paths.
///
/// `j` is the clause index within the class (0..clauses_per_class); it determines
/// clause polarity (even = positive) and is used to index `drop_mask`.
///
/// `lit_b`, `inv_b`, `keep_b`, `active_b` are byte-expanded versions of the packed
/// bit arrays, precomputed once per sample/class-update to enable SIMD auto-vectorisation
/// of the inner TA update loops (avoids per-literal bit extraction).
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
    // Packed bit arrays (for O(words) fire check):
    lit: &[u64],
    val: &[u64],
    lit_active: &[u64],
    words: usize,
    // Byte-expanded arrays (for SIMD TA update):
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
        // Type I: fire check (O(words)), then SIMD TA update (O(n_literals)).
        let fired = clause_fire(inc, lit, val, words, lit_active);
        let under_limit = max_inc == usize::MAX || {
            let n: u32 = (0..words).map(|k| (inc[k] & val[k]).count_ones()).sum();
            (n as usize) < max_inc
        };
        let fired_under = fired && under_limit;
        if fired_under {
            *w = (*w + 1).min(wmax);
        }
        type_i_update_bytes(ta, n_literals, fired_under, boost, lit_b, inv_b, keep_b, active_b, max_state);
    } else {
        // Type II: fire check, then SIMD TA update.
        if !clause_fire(inc, lit, val, words, lit_active) {
            return;
        }
        *w = (*w - 1).max(1);
        type_ii_update_bytes(ta, n_literals, lit_b, active_b, half, max_state);
    }
    rebuild_include(ta, inc, val, words, n_literals, half);
}

impl TsetlinMachine {
    /// Create a TsetlinMachine with default settings: 8 state bits, boost enabled, seed 42.
    pub fn new(
        n_classes: usize,
        n_features: usize,
        clauses_per_class: usize,
        threshold: i32,
        s: f64,
    ) -> Self {
        Self::with_config(n_classes, n_features, clauses_per_class, threshold, s, 8, true, 42)
    }

    /// Create a TsetlinMachine with full configuration.
    ///
    /// * `state_bits` — TA counter precision in bits (2–8); higher values slow convergence but
    ///   allow absorbing states to provide stronger regularisation.  Counters are stored as
    ///   `u8`, so the maximum is 8 bits (matching TMU's default 8-bit states).
    /// * `boost_true_positive` — if `true`, Type Ia feedback always includes present literals
    ///   (skips the stochastic keep mask), matching TMU's `boost_true_positive_feedback`.
    /// * `seed` — master RNG seed; fully deterministic for a given seed.
    #[allow(clippy::too_many_arguments)]
    pub fn with_config(
        n_classes: usize,
        n_features: usize,
        clauses_per_class: usize,
        threshold: i32,
        s: f64,
        state_bits: u8,
        boost_true_positive: bool,
        seed: u64,
    ) -> Self {
        assert!(n_classes >= 2);
        assert!(n_features >= 1);
        assert!(clauses_per_class >= 2);
        assert!(threshold >= 1);
        assert!(s > 1.0);
        assert!((2..=8).contains(&state_bits), "state_bits must be in 2..=8");

        let state_bits = state_bits as usize;
        let n_literals = 2 * n_features;
        let words = words_for(n_literals);
        let n_clauses = n_classes * clauses_per_class;
        let mut rng = Rng::new(seed);

        let mut valid = vec![0u64; words];
        for l in 0..n_literals {
            valid[l / WORD_BITS] |= 1u64 << (l % WORD_BITS);
        }

        let half = 1u8 << (state_bits - 1);
        // Use a u16 intermediate so state_bits == 8 (max_state 255) doesn't overflow `1u8 << 8`.
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
                &valid, words, n_literals, half,
            );
        }

        let rngs = (0..n_clauses)
            .map(|i| Rng::new(seed ^ (i as u64).wrapping_add(1).wrapping_mul(GOLDEN)))
            .collect();

        let class_rngs = (0..n_classes)
            .map(|c| Rng::new(seed ^ (c as u64 + n_clauses as u64 + 1).wrapping_mul(GOLDEN)))
            .collect();

        // Dedicated seed for literal-active RNG so it doesn't disturb the other streams.
        let literal_rng = Rng::new(seed ^ 0x4C49_5445_5241_4C21u64);

        TsetlinMachine {
            n_classes,
            n_features,
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
            literals: vec![0u64; words],
            rng,
        }
    }

    /// Limit how many literals each clause may include (Type Ia guard).
    /// Mirrors TMU's `max_included_literals` (default: no limit).
    pub fn max_included_literals(mut self, max: usize) -> Self {
        self.max_included_literals = max;
        self
    }

    /// Per-clause dropout probability during training (default: 0.0 = no drop).
    /// Mirrors TMU's `clause_drop_p`. Typical value for large models: 0.75.
    pub fn clause_drop_p(mut self, p: f64) -> Self {
        assert!((0.0..1.0).contains(&p), "clause_drop_p must be in [0, 1)");
        self.clause_drop_p = p;
        self
    }

    /// Per-literal dropout probability during training (default: 0.0 = no drop).
    ///
    /// Each literal is independently suppressed with this probability on every
    /// training sample — both its feedback and its contribution to the firing
    /// check are masked out.  Mirrors TMU's `literal_drop_p`.
    pub fn literal_drop_p(mut self, p: f64) -> Self {
        assert!((0.0..1.0).contains(&p), "literal_drop_p must be in [0, 1)");
        self.literal_drop_p = p;
        self.dig_lit_active = digits_of(1.0 - p, MASK_BITS);
        self
    }

    // ---- dimensions / accessors ------------------------------------------

    /// Return the number of output classes.
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }
    /// Return the number of input features.
    pub fn n_features(&self) -> usize {
        self.n_features
    }
    /// Return the number of clauses allocated per class.
    pub fn clauses_per_class(&self) -> usize {
        self.clauses_per_class
    }
    /// Return the number of 64-bit words used to represent one packed sample.
    pub fn words_per_sample(&self) -> usize {
        self.words
    }
    /// Return the specificity parameter `s` used for Type I feedback probability.
    pub fn s(&self) -> f64 {
        self.s
    }
    /// Return the integer weight of clause `clause` for class `class`.
    pub fn clause_weight(&self, class: usize, clause: usize) -> i32 {
        self.weights[class * self.clauses_per_class + clause]
    }

    // ---- inference -------------------------------------------------------

    /// Internal: predict from a raw literal slice without allocation.
    #[inline]
    fn predict_lit(&self, lit: &[u64]) -> usize {
        debug_assert_eq!(lit.len(), self.words);
        let cps = self.clauses_per_class;
        let words = self.words;
        let include = self.include.as_slice();
        let valid = self.valid.as_slice();
        let mut best = 0usize;
        let mut best_score = i32::MIN;
        for c in 0..self.n_classes {
            let cw = &self.weights[c * cps..(c + 1) * cps];
            let mut sum = 0i32;
            for (j, &w) in cw.iter().enumerate() {
                let cj = c * cps + j;
                if fire_predict(&include[cj * words..(cj + 1) * words], lit, valid, words) {
                    if j & 1 == 0 { sum += w; } else { sum -= w; }
                }
            }
            let v = sum.clamp(-self.threshold, self.threshold);
            if v > best_score {
                best_score = v;
                best = c;
            }
        }
        best
    }

    /// Predict the class for an encoded sample.
    #[inline]
    pub fn predict(&self, sample: &EncodedSample) -> usize {
        self.predict_lit(&sample.0)
    }

    /// Fill `out` with the clamped weighted clause sums for each class for an encoded sample.
    pub fn scores(&self, sample: &EncodedSample, out: &mut [i32]) {
        let lit = &sample.0;
        debug_assert_eq!(out.len(), self.n_classes);
        let cps = self.clauses_per_class;
        let words = self.words;
        let include = self.include.as_slice();
        let valid = self.valid.as_slice();
        for (c, out_c) in out.iter_mut().enumerate() {
            let cw = &self.weights[c * cps..(c + 1) * cps];
            let mut sum = 0i32;
            for (j, &w) in cw.iter().enumerate() {
                let cj = c * cps + j;
                if fire_predict(&include[cj * words..(cj + 1) * words], lit, valid, words) {
                    if j & 1 == 0 { sum += w; } else { sum -= w; }
                }
            }
            *out_c = sum.clamp(-self.threshold, self.threshold);
        }
    }

    /// Predict classes for all samples in an encoded batch, returning one class index per sample.
    pub fn predict_batch(&self, batch: &EncodedBatch) -> Vec<usize> {
        debug_assert_eq!(batch.data.len(), batch.n * self.words);
        let packed = batch.data.as_slice();
        let n = batch.n;
        let w = self.words;
        #[cfg(feature = "parallel")]
        if n >= PARALLEL_MIN && self.clauses_per_class >= PARALLEL_MIN {
            use rayon::prelude::*;
            return (0..n)
                .into_par_iter()
                .map(|i| self.predict_lit(&packed[i * w..(i + 1) * w]))
                .collect();
        }
        (0..n).map(|i| self.predict_lit(&packed[i * w..(i + 1) * w])).collect()
    }

    // ---- training helpers ------------------------------------------------

    /// Compute the clamped weighted clause sum for class `c` using the current `self.literals` buffer.
    /// `lit_active` is the per-sample literal dropout mask for this training step.
    fn class_sum_train(&self, c: usize, lit_active: &[u64]) -> i32 {
        let cps = self.clauses_per_class;
        let words = self.words;
        let include = self.include.as_slice();
        let valid = self.valid.as_slice();
        let lit = self.literals.as_slice();
        let mut sum: i32 = 0;
        for j in 0..cps {
            let cj = c * cps + j;
            if clause_fire(&include[cj * words..(cj + 1) * words], lit, valid, words, lit_active) {
                let w = self.weights[c * cps + j];
                if j & 1 == 0 { sum += w; } else { sum -= w; }
            }
        }
        sum.clamp(-self.threshold, self.threshold)
    }

    /// Apply Type I / II feedback to all clauses of class `c`.
    ///
    /// `target` is 1 for the true class and 0 for the sampled negative class.
    /// `sum` is the pre-computed clamped clause sum from `class_sum_train`.
    /// `lit_b` / `active_b` are byte-expanded per-sample arrays (precomputed in `fit_one_lit`).
    /// When `--features parallel` is active and `clauses_per_class >= PARALLEL_MIN`,
    /// the per-clause loop runs in parallel via rayon.
    fn update_class(
        &mut self,
        c: usize,
        target: u8,
        sum: i32,
        lit_active: &[u64],
        lit_b: &[u8],
        active_b: &[u8],
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

        let Self { ta, include, weights, rngs, class_rngs, literals, valid, dig_inv, dig_keep, .. } =
            self;
        let lit = literals.as_slice();
        let val = valid.as_slice();
        let crng = &mut class_rngs[c];

        let t = wmax as f64;
        let v = sum as f64;
        let p = if target == 1 {
            (t - v) / (2.0 * t)
        } else {
            (t + v) / (2.0 * t)
        };

        let drop_mask: Vec<bool> = if drop_p > 0.0 {
            (0..cps).map(|_| crng.next_f64() < drop_p).collect()
        } else {
            vec![]
        };

        // Generate class-specific Bernoulli masks and byte-expand once for all clauses.
        let inv_mask: Vec<u64> = (0..words).map(|_| bmask_word(crng, dig_inv)).collect();
        let keep_mask: Vec<u64> = (0..words).map(|_| bmask_word(crng, dig_keep)).collect();
        let inv_b = expand_bits_to_bytes(&inv_mask, n_literals);
        let keep_b = expand_bits_to_bytes(&keep_mask, n_literals);

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
                .for_each(|(j, (((ta_c, inc_c), w), rng))| {
                    apply_one_clause(
                        j, ta_c, inc_c, w, rng, target, p, &drop_mask,
                        lit, val, lit_active, words,
                        lit_b, &inv_b, &keep_b, active_b,
                        n_literals, boost, wmax, max_inc, half, max_state,
                    );
                });
            return;
        }

        for j in 0..cps {
            apply_one_clause(
                j,
                &mut class_ta[j * n_literals..(j + 1) * n_literals],
                &mut class_inc[j * words..(j + 1) * words],
                &mut class_w[j],
                &mut class_rng[j],
                target, p, &drop_mask,
                lit, val, lit_active, words,
                lit_b, &inv_b, &keep_b, active_b,
                n_literals, boost, wmax, max_inc, half, max_state,
            );
        }
    }

    /// Internal: train from a raw literal slice without allocation.
    fn fit_one_lit(&mut self, lit: &[u64], y: usize) {
        debug_assert_eq!(lit.len(), self.words);
        debug_assert!(y < self.n_classes);
        self.literals.copy_from_slice(lit);

        let mut neg = self.rng.below(self.n_classes);
        while neg == y {
            neg = self.rng.below(self.n_classes);
        }

        let n_literals = self.n_literals;
        let words = self.words;

        // Byte-expand the sample literals once; reused for all clause updates this step.
        let lit_b = expand_bits_to_bytes(lit, n_literals);

        // Generate per-sample literal-active mask once; shared by both class updates.
        let lit_active: Vec<u64> = if self.literal_drop_p > 0.0 {
            let rng = &mut self.literal_rng;
            let dig = &self.dig_lit_active;
            (0..words).map(|_| bmask_word(rng, dig)).collect()
        } else {
            vec![!0u64; words]
        };

        // Byte-expand active mask (valid & lit_active). Since valid_b is all-1s for
        // l < n_literals, this is equivalent to expand_bits_to_bytes(&lit_active, …).
        let active_b = expand_bits_to_bytes(&lit_active, n_literals);

        let sum_y   = self.class_sum_train(y,   &lit_active);
        let sum_neg = self.class_sum_train(neg, &lit_active);
        self.update_class(y,   1, sum_y,   &lit_active, &lit_b, &active_b);
        self.update_class(neg, 0, sum_neg, &lit_active, &lit_b, &active_b);
    }

    /// Train on a single encoded sample with true label `y`.
    pub fn fit_one(&mut self, sample: &EncodedSample, y: usize) {
        self.fit_one_lit(&sample.0, y);
    }

    /// Run one training epoch over an encoded batch, shuffling the order each epoch.
    pub fn fit_epoch(&mut self, batch: &EncodedBatch, ys: &[usize]) {
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

    // ---- dataset helpers -------------------------------------------------

    /// Compute the fraction of correctly predicted samples in an encoded batch.
    pub fn accuracy(&self, batch: &EncodedBatch, ys: &[usize]) -> f64 {
        debug_assert_eq!(batch.data.len(), batch.n * self.words);
        assert_eq!(batch.n, ys.len());
        let packed = batch.data.as_slice();
        let n = batch.n;
        let w = self.words;
        #[cfg(feature = "parallel")]
        if n >= PARALLEL_MIN {
            use rayon::prelude::*;
            let correct: usize = (0..n)
                .into_par_iter()
                .filter(|&i| self.predict_lit(&packed[i * w..(i + 1) * w]) == ys[i])
                .count();
            return correct as f64 / n as f64;
        }
        let correct = (0..n)
            .filter(|&i| self.predict_lit(&packed[i * w..(i + 1) * w]) == ys[i])
            .count();
        correct as f64 / n as f64
    }

    // ---- absorbing state introspection ------------------------------------

    /// Fraction of (clause, literal) pairs whose TA is at the **absorbing include**
    /// state (counter == max_state).
    /// Grows toward 1.0 as training converges; used to measure absorbing progress.
    pub fn absorbed_include_fraction(&self) -> f64 {
        let n_clauses = self.n_classes * self.clauses_per_class;
        let mut total = 0u64;
        let mut at_max = 0u64;
        for cj in 0..n_clauses {
            let base = cj * self.n_literals;
            for l in 0..self.n_literals {
                let k = l / WORD_BITS;
                let bit = 1u64 << (l % WORD_BITS);
                if self.valid[k] & bit != 0 {
                    total += 1;
                    if self.ta[base + l] == self.max_state {
                        at_max += 1;
                    }
                }
            }
        }
        if total == 0 { 0.0 } else { at_max as f64 / total as f64 }
    }

    /// Fraction of (clause, literal) pairs whose TA is at the **absorbing exclude**
    /// state (counter == 0).
    pub fn absorbed_exclude_fraction(&self) -> f64 {
        let n_clauses = self.n_classes * self.clauses_per_class;
        let mut total = 0u64;
        let mut at_min = 0u64;
        for cj in 0..n_clauses {
            let base = cj * self.n_literals;
            for l in 0..self.n_literals {
                let k = l / WORD_BITS;
                let bit = 1u64 << (l % WORD_BITS);
                if self.valid[k] & bit != 0 {
                    total += 1;
                    if self.ta[base + l] == 0 {
                        at_min += 1;
                    }
                }
            }
        }
        if total == 0 { 0.0 } else { at_min as f64 / total as f64 }
    }

    // ---- interpretability ------------------------------------------------

    /// Return the included literals for clause `clause` of `class` as `(feature_index, is_negated)` pairs.
    pub fn clause_rule(&self, class: usize, clause: usize) -> Vec<(usize, bool)> {
        let mut rule = Vec::new();
        let cj = class * self.clauses_per_class + clause;
        let inc = &self.include[cj * self.words..(cj + 1) * self.words];
        for l in 0..self.n_literals {
            let included = (inc[l / WORD_BITS] >> (l % WORD_BITS)) & 1 != 0;
            if included {
                if l < self.n_features {
                    rule.push((l, false));
                } else {
                    rule.push((l - self.n_features, true));
                }
            }
        }
        rule
    }

    /// Return `true` if `clause` is a positive clause (even index → votes for the class).
    pub fn clause_is_positive(&self, clause: usize) -> bool {
        clause & 1 == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clause_bank::dense::{clause_type_i_bytes, clause_type_ii_bytes, fire_predict};
    use crate::encoder::Encoder;

    // ---- helpers -------------------------------------------------------------

    fn enc(n_features: usize) -> Encoder {
        Encoder::for_binary(n_features)
    }

    /// Generate `n` XOR samples (12 random bits, label = bit0 XOR bit1) with optional label noise.
    fn make_xor(n: usize, noise: f64, seed: u64) -> (Vec<Vec<u8>>, Vec<usize>) {
        let mut rng = Rng::new(seed);
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for _ in 0..n {
            let f: Vec<u8> = (0..12).map(|_| (rng.next_u64() & 1) as u8).collect();
            let mut y = (f[0] ^ f[1]) as usize;
            if rng.next_f64() < noise {
                y = 1 - y;
            }
            xs.push(f);
            ys.push(y);
        }
        (xs, ys)
    }

    /// Convert a slice of owned vectors to a slice of byte slices for API calls.
    fn as_slices(xs: &[Vec<u8>]) -> Vec<&[u8]> {
        xs.iter().map(|v| v.as_slice()).collect()
    }

    // ---- existing tests ------------------------------------------------------

    #[test]
    fn weighted_learns_xor_with_few_clauses() {
        let (xtr, ytr) = make_xor(5000, 0.25, 1);
        let (xte, yte) = make_xor(2000, 0.0, 2);
        let e = enc(12);
        let btr = e.encode_batch(&as_slices(&xtr));
        let bte = e.encode_batch(&as_slices(&xte));

        let mut tm = TsetlinMachine::with_config(2, 12, 8, 15, 3.9, 8, true, 7);
        for _ in 0..15 {
            tm.fit_epoch(&btr, &ytr);
        }
        let acc = tm.accuracy(&bte, &yte);
        assert!(acc > 0.95, "expected >0.95, got {acc}");
        for c in 0..2 {
            for j in 0..tm.clauses_per_class() {
                assert!((1..=15).contains(&tm.clause_weight(c, j)));
            }
        }
    }

    #[test]
    fn inc_dec_saturate() {
        let tm = TsetlinMachine::with_config(2, 1, 2, 5, 3.0, 8, true, 1);
        let max_state = tm.max_state;

        // Saturating increment: adding 1 to max_state stays at max_state.
        assert_eq!(max_state.saturating_add(1).min(max_state), max_state);
        // Saturating decrement: subtracting 1 from 0 stays at 0.
        assert_eq!(0u8.saturating_sub(1), 0);

        // Check the full range for a representative value.
        let mut v = 0u8;
        for _ in 0..1000 { v = v.saturating_add(1).min(max_state); }
        assert_eq!(v, max_state);
        for _ in 0..1000 { v = v.saturating_sub(1); }
        assert_eq!(v, 0);
    }

    // ---- encode / predict API consistency ------------------------------------

    #[test]
    fn encode_predict_roundtrip_agrees() {
        let (xtr, ytr) = make_xor(500, 0.0, 10);
        let e = enc(12);
        let mut tm = TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 42);
        tm.fit_epoch(&e.encode_batch(&as_slices(&xtr)), &ytr);

        let (xte, _) = make_xor(200, 0.0, 20);
        for x in &xte {
            let s = e.encode_one(x);
            let by_predict = tm.predict(&s);
            let by_lit = tm.predict_lit(&s.0);
            assert_eq!(by_predict, by_lit, "predict and predict_lit disagree for {x:?}");
        }
    }

    #[test]
    fn accuracy_matches_manual_loop() {
        let (xtr, ytr) = make_xor(500, 0.0, 11);
        let e = enc(12);
        let mut tm = TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 42);
        tm.fit_epoch(&e.encode_batch(&as_slices(&xtr)), &ytr);

        let (xte, yte) = make_xor(300, 0.0, 21);
        let batch = e.encode_batch(&as_slices(&xte));
        let n = xte.len();
        let w = tm.words_per_sample();

        let api_acc = tm.accuracy(&batch, &yte);
        let manual_correct = (0..n)
            .filter(|&i| tm.predict_lit(&batch.data[i * w..(i + 1) * w]) == yte[i])
            .count();
        let manual_acc = manual_correct as f64 / n as f64;

        assert!((api_acc - manual_acc).abs() < 1e-12);
    }

    #[test]
    fn predict_batch_matches_single() {
        let (xtr, ytr) = make_xor(300, 0.0, 12);
        let e = enc(12);
        let mut tm = TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 42);
        tm.fit_epoch(&e.encode_batch(&as_slices(&xtr)), &ytr);

        let (xte, _) = make_xor(100, 0.0, 22);
        let batch = e.encode_batch(&as_slices(&xte));
        let n = xte.len();
        let w = tm.words_per_sample();

        let from_batch = tm.predict_batch(&batch);
        let from_single: Vec<usize> = (0..n)
            .map(|i| tm.predict_lit(&batch.data[i * w..(i + 1) * w]))
            .collect();

        assert_eq!(from_batch, from_single);
    }

    // ---- scores -------------------------------------------------------

    #[test]
    fn scores_correct_class_wins_after_training() {
        let (xtr, ytr) = make_xor(2000, 0.0, 13);
        let e = enc(12);
        let mut tm = TsetlinMachine::with_config(2, 12, 10, 15, 3.9, 8, true, 42);
        for _ in 0..10 {
            tm.fit_epoch(&e.encode_batch(&as_slices(&xtr)), &ytr);
        }

        let (xte, yte) = make_xor(100, 0.0, 23);
        let mut correct = 0usize;
        for (x, &y) in xte.iter().zip(&yte) {
            let sample = e.encode_one(x);
            let mut s = vec![0i32; 2];
            tm.scores(&sample, &mut s);
            let pred = if s[0] >= s[1] { 0 } else { 1 };
            if pred == y {
                correct += 1;
            }
        }
        let acc = correct as f64 / 100.0;
        assert!(acc > 0.90, "scores acc {acc} too low");
    }

    // ---- clause polarity -----------------------------------------------------

    #[test]
    fn clause_is_positive_matches_index_parity() {
        let tm = TsetlinMachine::with_config(3, 4, 6, 10, 3.0, 8, true, 1);
        for j in 0..tm.clauses_per_class() {
            assert_eq!(tm.clause_is_positive(j), j & 1 == 0);
        }
    }

    // ---- determinism ---------------------------------------------------------

    #[test]
    fn same_seed_same_result() {
        let (xtr, ytr) = make_xor(500, 0.0, 14);
        let e = enc(12);
        let btr = e.encode_batch(&as_slices(&xtr));

        let mut tm1 = TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 99);
        let mut tm2 = TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 99);
        for _ in 0..5 {
            tm1.fit_epoch(&btr, &ytr);
            tm2.fit_epoch(&btr, &ytr);
        }

        let (xte, yte) = make_xor(200, 0.0, 24);
        let bte = e.encode_batch(&as_slices(&xte));
        assert_eq!(
            tm1.accuracy(&bte, &yte),
            tm2.accuracy(&bte, &yte),
            "same seed must produce identical results"
        );
    }

    // ---- clause_drop_p -------------------------------------------------------

    #[test]
    fn clause_drop_p_one_leaves_state_unchanged() {
        let (xtr, ytr) = make_xor(200, 0.0, 15);
        let btr = enc(12).encode_batch(&as_slices(&xtr));
        let mut tm = TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 42)
            .clause_drop_p(0.9999);
        let ta_before = tm.ta.clone();
        tm.fit_epoch(&btr, &ytr);
        let ta_changed = tm.ta.iter().zip(&ta_before).filter(|(a, b)| a != b).count();
        let total = tm.ta.len();
        assert!(
            ta_changed < total / 100,
            "drop_p≈1 should leave >99% of state unchanged, but {ta_changed}/{total} changed"
        );
    }

    #[test]
    fn clause_drop_p_zero_trains_normally() {
        let (xtr, ytr) = make_xor(2000, 0.0, 16);
        let (xte, yte) = make_xor(500, 0.0, 26);
        let e = enc(12);
        let btr = e.encode_batch(&as_slices(&xtr));
        let bte = e.encode_batch(&as_slices(&xte));
        let mut tm = TsetlinMachine::with_config(2, 12, 10, 15, 3.9, 8, true, 42)
            .clause_drop_p(0.0);
        for _ in 0..10 {
            tm.fit_epoch(&btr, &ytr);
        }
        let acc = tm.accuracy(&bte, &yte);
        assert!(acc > 0.90, "drop_p=0 should still converge, got {acc}");
    }

    // ---- max_included_literals -----------------------------------------------

    #[test]
    fn clause_type_i_stops_including_at_limit() {
        let n_literals = 8usize;
        let words = 1usize;
        let half = 128u8;
        let max_state = 255u8;
        let max_included = 2usize;

        // 2 literals already included (bits 0 and 1 at half=included threshold).
        let mut ta = vec![0u8; n_literals];
        ta[0] = half;
        ta[1] = half;
        let valid = vec![0b1111_1111u64];
        let mut inc = vec![0b11u64]; // bits 0 and 1 set
        let lit = vec![0b1111_1111u64]; // all present
        let inv_mask = vec![!0u64];
        let keep_mask = vec![!0u64];
        let all_active = vec![!0u64];
        let mut weight = 5i32;

        clause_type_i_bytes(
            &mut ta, &mut inc, &mut weight, &lit, &valid,
            words, n_literals, false, &inv_mask, &keep_mask, 10, max_included, &all_active,
            half, max_state,
        );

        let n_after = (inc[0] & valid[0]).count_ones() as usize;
        assert!(
            n_after <= max_included,
            "Type Ia added literals beyond limit: {n_after} > {max_included}"
        );
    }

    #[test]
    fn max_included_literals_reduces_clause_size() {
        let (xtr, ytr) = make_xor(2000, 0.0, 17);
        let btr = enc(12).encode_batch(&as_slices(&xtr));

        let mut tm_tight = TsetlinMachine::with_config(2, 12, 10, 15, 3.9, 8, true, 42)
            .max_included_literals(2);
        let mut tm_free = TsetlinMachine::with_config(2, 12, 10, 15, 3.9, 8, true, 42);
        for _ in 0..10 {
            tm_tight.fit_epoch(&btr, &ytr);
            tm_free.fit_epoch(&btr, &ytr);
        }

        let avg = |tm: &TsetlinMachine| {
            let total: usize = (0..2)
                .flat_map(|c| (0..tm.clauses_per_class()).map(move |j| (c, j)))
                .map(|(c, j)| tm.clause_rule(c, j).len())
                .sum();
            total as f64 / (2 * tm.clauses_per_class()) as f64
        };

        let tight_avg = avg(&tm_tight);
        let free_avg = avg(&tm_free);
        assert!(
            tight_avg < free_avg,
            "max_included_literals=2 should produce smaller clauses: {tight_avg:.2} vs {free_avg:.2}"
        );
    }

    // ---- multiclass ----------------------------------------------------------

    #[test]
    fn multiclass_4_class_learns() {
        let mut rng = Rng::new(50);
        let n = 3000usize;
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for _ in 0..n {
            let f: Vec<u8> = (0..8).map(|_| (rng.next_u64() & 1) as u8).collect();
            let y = ((f[0] ^ f[1]) as usize) * 2 + (f[2] ^ f[3]) as usize;
            xs.push(f);
            ys.push(y);
        }
        let mut rng2 = Rng::new(51);
        let nte = 500usize;
        let mut xte = Vec::with_capacity(nte);
        let mut yte = Vec::with_capacity(nte);
        for _ in 0..nte {
            let f: Vec<u8> = (0..8).map(|_| (rng2.next_u64() & 1) as u8).collect();
            let y = ((f[0] ^ f[1]) as usize) * 2 + (f[2] ^ f[3]) as usize;
            xte.push(f);
            yte.push(y);
        }

        let e = enc(8);
        let btr = e.encode_batch(&as_slices(&xs));
        let bte = e.encode_batch(&as_slices(&xte));
        let mut tm = TsetlinMachine::with_config(4, 8, 20, 30, 3.9, 8, true, 42);
        for _ in 0..20 {
            tm.fit_epoch(&btr, &ys);
        }
        let acc = tm.accuracy(&bte, &yte);
        assert!(acc > 0.85, "4-class XOR should reach >0.85, got {acc}");
    }

    // ---- weight bounds -------------------------------------------------------

    #[test]
    fn weights_stay_in_1_to_threshold() {
        let threshold = 20i32;
        let (xtr, ytr) = make_xor(1000, 0.1, 18);
        let btr = enc(12).encode_batch(&as_slices(&xtr));
        let mut tm = TsetlinMachine::with_config(2, 12, 12, threshold, 3.9, 8, true, 42);
        for _ in 0..10 {
            tm.fit_epoch(&btr, &ytr);
        }
        for c in 0..2 {
            for j in 0..tm.clauses_per_class() {
                let w = tm.clause_weight(c, j);
                assert!(
                    (1..=threshold).contains(&w),
                    "weight {w} out of [1, {threshold}] for clause ({c},{j})"
                );
            }
        }
    }

    // ---- words_per_sample / pack dimensions ----------------------------------

    #[test]
    fn words_per_sample_correct() {
        for &nf in &[1usize, 32, 63, 64, 65, 100, 128, 784] {
            let tm = TsetlinMachine::with_config(2, nf, 2, 5, 2.0, 8, true, 1);
            let expected = (2 * nf + 63) / 64;
            assert_eq!(
                tm.words_per_sample(),
                expected,
                "n_features={nf}: expected {expected} words, got {}",
                tm.words_per_sample()
            );
        }
    }

    #[test]
    fn pack_sets_positive_and_negated_bits() {
        let nf = 4usize;
        let x: Vec<u8> = vec![1, 0, 1, 0];
        let words = (2 * nf + 63) / 64;
        let mut lit = vec![0u64; words];
        crate::clause_bank::dense::pack(&x, nf, &mut lit);
        let bits = lit[0];
        assert_eq!((bits >> 0) & 1, 1, "x[0]=1: positive bit should be set");
        assert_eq!((bits >> 1) & 1, 0, "x[1]=0: positive bit should be clear");
        assert_eq!((bits >> 2) & 1, 1, "x[2]=1: positive bit should be set");
        assert_eq!((bits >> 3) & 1, 0, "x[3]=0: positive bit should be clear");
        assert_eq!((bits >> (nf + 0)) & 1, 0, "x[0]=1: negated bit should be clear");
        assert_eq!((bits >> (nf + 1)) & 1, 1, "x[1]=0: negated bit should be set");
        assert_eq!((bits >> (nf + 2)) & 1, 0, "x[2]=1: negated bit should be clear");
        assert_eq!((bits >> (nf + 3)) & 1, 1, "x[3]=0: negated bit should be set");
    }

    // ---- fire_predict semantics ----------------------------------------------

    #[test]
    fn fire_predict_empty_clause_returns_false() {
        let words = 2usize;
        let inc = vec![0u64; words];
        let lit = vec![0u64; words];
        let valid = vec![!0u64; words];
        assert!(!fire_predict(&inc, &lit, &valid, words));
    }

    #[test]
    fn fire_predict_satisfied_clause_returns_true() {
        let words = 1usize;
        let inc = vec![1u64]; // bit 0 included
        let lit = vec![1u64]; // bit 0 present
        let valid = vec![1u64];
        assert!(fire_predict(&inc, &lit, &valid, words));
    }

    #[test]
    fn fire_predict_violated_clause_returns_false() {
        let words = 1usize;
        let inc = vec![1u64]; // bit 0 included
        let lit = vec![0u64]; // bit 0 absent → violation
        let valid = vec![1u64];
        assert!(!fire_predict(&inc, &lit, &valid, words));
    }

    // ---- literal_drop_p -------------------------------------------------------

    #[test]
    fn literal_drop_p_one_leaves_state_unchanged() {
        let (xtr, ytr) = make_xor(200, 0.0, 30);
        let btr = enc(12).encode_batch(&as_slices(&xtr));
        let mut tm = TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 42)
            .literal_drop_p(0.9999);
        let ta_before = tm.ta.clone();
        tm.fit_epoch(&btr, &ytr);
        let ta_changed = tm.ta.iter().zip(&ta_before).filter(|(a, b)| a != b).count();
        let total = tm.ta.len();
        assert!(
            ta_changed < total / 100,
            "literal_drop_p≈1 should leave >99% of state unchanged, but {ta_changed}/{total} changed"
        );
    }

    #[test]
    fn literal_drop_p_zero_trains_normally() {
        let (xtr, ytr) = make_xor(2000, 0.0, 31);
        let (xte, yte) = make_xor(500, 0.0, 41);
        let e = enc(12);
        let btr = e.encode_batch(&as_slices(&xtr));
        let bte = e.encode_batch(&as_slices(&xte));
        let mut tm = TsetlinMachine::with_config(2, 12, 10, 15, 3.9, 8, true, 42)
            .literal_drop_p(0.0);
        for _ in 0..10 {
            tm.fit_epoch(&btr, &ytr);
        }
        let acc = tm.accuracy(&bte, &yte);
        assert!(acc > 0.90, "literal_drop_p=0 should still converge, got {acc}");
    }

    // ---- absorbing states -------------------------------------------------------

    #[test]
    fn absorbing_include_at_max_resists_ib() {
        // A literal at the maximum TA state must survive 1 000 rounds of Type Ib
        // "exclude absent" feedback completely unchanged.
        let n_literals = 1usize;
        let words = 1usize;
        let half = 8u8;     // sb=4 → half=8
        let max_state = 15u8; // (1<<4)-1
        // Literal 0 at max state (included); absent from x → violation → Ib path.
        let mut ta = vec![max_state];
        let mut inc = vec![1u64]; // bit 0 included
        let lit = vec![0u64];     // absent
        let valid = vec![1u64];
        let inv_mask = vec![!0u64];
        let keep_mask = vec![!0u64];
        let all_active = vec![!0u64];
        let mut weight = 1i32;

        for _ in 0..1_000 {
            clause_type_i_bytes(
                &mut ta, &mut inc, &mut weight, &lit, &valid,
                words, n_literals, false, &inv_mask, &keep_mask, 100, usize::MAX, &all_active,
                half, max_state,
            );
        }

        assert_eq!(
            ta[0], max_state,
            "absorbing max-state literal must resist all Ib decrement pressure"
        );
    }

    #[test]
    fn non_max_include_is_decremented_by_ib() {
        // A literal one step below max (still included: 14 >= half=8) must be
        // decremented by a single Ib round.
        let n_literals = 1usize;
        let words = 1usize;
        let half = 8u8;
        let max_state = 15u8;
        let below_max = max_state - 1; // 14, still included
        let mut ta = vec![below_max];
        let mut inc = vec![1u64]; // included
        let before = ta[0];

        let lit = vec![0u64]; // absent → violation on included literal → Ib path
        let valid = vec![1u64];
        let inv_mask = vec![!0u64];
        let keep_mask = vec![!0u64];
        let all_active = vec![!0u64];
        let mut weight = 1i32;

        clause_type_i_bytes(
            &mut ta, &mut inc, &mut weight, &lit, &valid,
            words, n_literals, false, &inv_mask, &keep_mask, 100, usize::MAX, &all_active,
            half, max_state,
        );

        assert_ne!(
            ta[0], before,
            "non-max literal must be decremented by Ib feedback"
        );
    }

    #[test]
    fn absorbing_exclude_at_min_resists_type_ii() {
        // A literal at state 0 (absorbing exclude) must survive 1 000 rounds of
        // Type II "include absent excluded" feedback unchanged.
        let n_literals = 1usize;
        let words = 1usize;
        let half = 8u8;
        let max_state = 15u8;
        // All-zero: every literal at min state; empty clause fires (no violations).
        let mut ta = vec![0u8];
        let mut inc = vec![0u64]; // excluded
        let lit = vec![0u64];     // literal 0 absent from x
        let valid = vec![1u64];
        let all_active = vec![!0u64];
        let mut weight = 5i32;

        for _ in 0..1_000 {
            clause_type_ii_bytes(
                &mut ta, &mut inc, &mut weight, &lit, &valid,
                words, n_literals, &all_active, half, max_state,
            );
        }

        assert_eq!(
            ta[0], 0,
            "absorbing min-state literal must resist all Type II increment pressure"
        );
    }

    #[test]
    fn absorbing_stabilizes_clause_at_literal_limit() {
        // Two included literals, max_included_literals = 1 (over the limit → always Ib).
        //   literal 0: max state (absorbing) — should survive
        //   literal 1: half (just included, not absorbing) — should be expelled
        // After enough rounds the clause should settle to exactly literal 0.
        let n_literals = 2usize;
        let words = 1usize;
        let half = 8u8;
        let max_state = 15u8;
        let mut ta = vec![max_state, half]; // literal 0 absorbing, literal 1 just included
        let mut inc = vec![0b11u64];         // both included
        let lit = vec![0u64];               // both absent → violations on both → Ib path
        let valid = vec![0b11u64];
        let inv_mask = vec![!0u64];
        let keep_mask = vec![!0u64];
        let all_active = vec![!0u64];
        let mut weight = 1i32;

        for _ in 0..500 {
            clause_type_i_bytes(
                &mut ta, &mut inc, &mut weight, &lit, &valid,
                words, n_literals, false, &inv_mask, &keep_mask, 5, 1, &all_active,
                half, max_state,
            );
        }

        assert_eq!((inc[0] >> 0) & 1, 1, "absorbing literal 0 must stay included");
        assert_eq!((inc[0] >> 1) & 1, 0, "non-absorbing literal 1 must be expelled");
    }
}
