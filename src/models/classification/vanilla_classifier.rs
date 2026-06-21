//! Vanilla weighted multiclass Tsetlin Machine classifier.
//!
//! Mirrors TMU's `vanilla_classifier.py` / `TMClassifier`.

use crate::clause_bank::dense::{
    bmask_word, clause_type_i, clause_type_ii, digits_of, fire_predict, words_for, GOLDEN,
    MASK_BITS, WORD_BITS,
};
#[cfg(feature = "parallel")]
use crate::clause_bank::dense::PARALLEL_MIN;
use crate::encoder::{EncodedBatch, EncodedSample};
use crate::rng::Rng;

/// A bit-packed weighted multiclass Tsetlin Machine.
///
/// Bit-plane TA state, weighted clauses, and optional clause-level parallelism
/// (`--features parallel`).
#[derive(Clone, Debug)]
pub struct TsetlinMachine {
    n_classes: usize,
    n_features: usize,
    n_literals: usize,
    words: usize,
    clauses_per_class: usize,
    threshold: i32,
    s: f64,
    state_bits: usize,
    boost_true_positive: bool,
    max_included_literals: usize,
    clause_drop_p: f64,
    /// Per-literal dropout probability during training (mirrors TMU's `literal_drop_p`).
    literal_drop_p: f64,
    /// Dedicated RNG for literal-active mask generation (independent of clause/class RNGs).
    literal_rng: Rng,
    /// Precomputed binary digits for Bernoulli(1 - literal_drop_p) mask generation.
    dig_lit_active: Vec<u8>,

    /// Bit-plane TA counters. Clause `cj = c*CPC + j` occupies
    /// `state[cj*state_bits*words .. (cj+1)*state_bits*words]`; within that
    /// chunk plane `b` word `w` is at `b*words + w`. Top plane = include bitset.
    state: Vec<u64>,
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
#[allow(clippy::too_many_arguments)]
#[inline]
fn apply_one_clause(
    j: usize,
    chunk: &mut [u64],
    w: &mut i32,
    rng: &mut Rng,
    target: u8,
    p: f64,
    drop_mask: &[bool],
    lit: &[u64],
    val: &[u64],
    inv_mask: &[u64],
    keep_mask: &[u64],
    lit_active: &[u64],
    words: usize,
    sb: usize,
    boost: bool,
    wmax: i32,
    max_inc: usize,
) {
    if !drop_mask.is_empty() && drop_mask[j] {
        return;
    }
    if rng.next_f64() > p {
        return;
    }
    let positive = j & 1 == 0;
    if (target == 1) == positive {
        clause_type_i(chunk, w, lit, val, words, sb, boost, inv_mask, keep_mask, wmax, max_inc, lit_active);
    } else {
        clause_type_ii(chunk, w, lit, val, words, sb, lit_active);
    }
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
    /// * `state_bits` — TA counter precision in bits (2–16); higher values slow convergence but
    ///   allow absorbing states to provide stronger regularisation.
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
        assert!((2..=16).contains(&state_bits), "state_bits must be in 2..=16");

        let state_bits = state_bits as usize;
        let n_literals = 2 * n_features;
        let words = words_for(n_literals);
        let n_clauses = n_classes * clauses_per_class;
        let mut rng = Rng::new(seed);

        let mut valid = vec![0u64; words];
        for l in 0..n_literals {
            valid[l / WORD_BITS] |= 1u64 << (l % WORD_BITS);
        }

        let half: u64 = 1u64 << (state_bits - 1);
        let mut state = vec![0u64; n_clauses * state_bits * words];
        for cj in 0..n_clauses {
            let cb = cj * state_bits * words;
            for l in 0..n_literals {
                let v = if rng.next_u64() & 1 == 0 { half - 1 } else { half };
                let w = l / WORD_BITS;
                let bit = 1u64 << (l % WORD_BITS);
                for b in 0..state_bits {
                    if (v >> b) & 1 == 1 {
                        state[cb + b * words + w] |= bit;
                    }
                }
            }
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
            state_bits,
            boost_true_positive,
            max_included_literals: usize::MAX,
            clause_drop_p: 0.0,
            literal_drop_p: 0.0,
            literal_rng,
            dig_lit_active: digits_of(1.0, MASK_BITS),
            state,
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

    // ---- internal indexing -----------------------------------------------

    /// Return the flat index into `state` for the first word of clause `j` in class `c`.
    #[inline(always)]
    fn clause_base(&self, c: usize, j: usize) -> usize {
        (c * self.clauses_per_class + j) * self.state_bits * self.words
    }

    /// Return the flat index into `state` for the top (include) bit-plane of clause `j` in class `c`.
    #[inline(always)]
    fn top_base(&self, c: usize, j: usize) -> usize {
        self.clause_base(c, j) + (self.state_bits - 1) * self.words
    }

    // ---- inference -------------------------------------------------------

    /// Internal: predict from a raw literal slice without allocation.
    #[inline]
    fn predict_lit(&self, lit: &[u64]) -> usize {
        debug_assert_eq!(lit.len(), self.words);
        let cps = self.clauses_per_class;
        let bw = self.state_bits * self.words;
        let tb_off = (self.state_bits - 1) * self.words;
        let words = self.words;
        let state = self.state.as_slice();
        let valid = self.valid.as_slice();
        let mut best = 0usize;
        let mut best_score = i32::MIN;
        for c in 0..self.n_classes {
            let cb = c * cps * bw + tb_off;
            let cw = &self.weights[c * cps..(c + 1) * cps];
            let mut sum = 0i32;
            for (j, &w) in cw.iter().enumerate() {
                if fire_predict(state, cb + j * bw, lit, valid, words) {
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
        let bw = self.state_bits * self.words;
        let tb_off = (self.state_bits - 1) * self.words;
        let words = self.words;
        let state = self.state.as_slice();
        let valid = self.valid.as_slice();
        for (c, out_c) in out.iter_mut().enumerate() {
            let cb = c * cps * bw + tb_off;
            let cw = &self.weights[c * cps..(c + 1) * cps];
            let mut sum = 0i32;
            for (j, &w) in cw.iter().enumerate() {
                if fire_predict(state, cb + j * bw, lit, valid, words) {
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
        let sb = self.state_bits;
        let bw = sb * words;
        let tb_offset = (sb - 1) * words;
        let base = c * cps * bw;
        let mut sum: i32 = 0;
        for j in 0..cps {
            let jbase = base + j * bw + tb_offset;
            let mut fired = true;
            for (k, &la) in lit_active.iter().enumerate() {
                if self.state[jbase + k] & self.valid[k] & la & !self.literals[k] != 0 {
                    fired = false;
                    break;
                }
            }
            if fired {
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
    /// When `--features parallel` is active and `clauses_per_class >= PARALLEL_MIN`,
    /// the per-clause loop runs in parallel via rayon.
    fn update_class(&mut self, c: usize, target: u8, sum: i32, lit_active: &[u64]) {
        let cps = self.clauses_per_class;
        let words = self.words;
        let sb = self.state_bits;
        let bw = sb * words;
        let boost = self.boost_true_positive;
        let wmax = self.threshold;
        let max_inc = self.max_included_literals;
        let drop_p = self.clause_drop_p;

        let Self { state, weights, rngs, class_rngs, literals, valid, dig_inv, dig_keep, .. } =
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
        let inv_mask: Vec<u64> =
            (0..words).map(|_| bmask_word(crng, dig_inv)).collect();
        let keep_mask: Vec<u64> =
            (0..words).map(|_| bmask_word(crng, dig_keep)).collect();

        let class_state = &mut state[c * cps * bw..(c + 1) * cps * bw];
        let class_w = &mut weights[c * cps..(c + 1) * cps];
        let class_rng = &mut rngs[c * cps..(c + 1) * cps];

        #[cfg(feature = "parallel")]
        if cps >= PARALLEL_MIN {
            use rayon::prelude::*;
            class_state
                .par_chunks_mut(bw)
                .zip(class_w.par_iter_mut())
                .zip(class_rng.par_iter_mut())
                .enumerate()
                .for_each(|(j, ((chunk, w), rng))| {
                    apply_one_clause(
                        j, chunk, w, rng, target, p, &drop_mask,
                        lit, val, &inv_mask, &keep_mask, lit_active,
                        words, sb, boost, wmax, max_inc,
                    );
                });
            return;
        }

        for j in 0..cps {
            apply_one_clause(
                j,
                &mut class_state[j * bw..(j + 1) * bw],
                &mut class_w[j],
                &mut class_rng[j],
                target, p, &drop_mask,
                lit, val, &inv_mask, &keep_mask, lit_active,
                words, sb, boost, wmax, max_inc,
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

        // Generate per-sample literal-active mask once; shared by both class updates.
        let words = self.words;
        let lit_active: Vec<u64> = if self.literal_drop_p > 0.0 {
            let rng = &mut self.literal_rng;
            let dig = &self.dig_lit_active;
            (0..words).map(|_| bmask_word(rng, dig)).collect()
        } else {
            vec![!0u64; words]
        };

        let sum_y   = self.class_sum_train(y,   &lit_active);
        let sum_neg = self.class_sum_train(neg, &lit_active);
        self.update_class(y,   1, sum_y,   &lit_active);
        self.update_class(neg, 0, sum_neg, &lit_active);
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
    /// state (all `state_bits` planes set = max counter value).
    /// Grows toward 1.0 as training converges; used to measure absorbing progress.
    pub fn absorbed_include_fraction(&self) -> f64 {
        let n_clauses = self.n_classes * self.clauses_per_class;
        let bw = self.state_bits * self.words;
        let mut total = 0u64;
        let mut at_max = 0u64;
        for cj in 0..n_clauses {
            let cb = cj * bw;
            for k in 0..self.words {
                let mask = (0..self.state_bits)
                    .fold(!0u64, |acc, b| acc & self.state[cb + b * self.words + k]);
                let valid = self.valid[k];
                at_max += (mask & valid).count_ones() as u64;
                total  += valid.count_ones() as u64;
            }
        }
        if total == 0 { 0.0 } else { at_max as f64 / total as f64 }
    }

    /// Fraction of (clause, literal) pairs whose TA is at the **absorbing exclude**
    /// state (all `state_bits` planes clear = counter 0).
    pub fn absorbed_exclude_fraction(&self) -> f64 {
        let n_clauses = self.n_classes * self.clauses_per_class;
        let bw = self.state_bits * self.words;
        let mut total = 0u64;
        let mut at_min = 0u64;
        for cj in 0..n_clauses {
            let cb = cj * bw;
            for k in 0..self.words {
                let not_min = (0..self.state_bits)
                    .fold(0u64, |acc, b| acc | self.state[cb + b * self.words + k]);
                let valid = self.valid[k];
                at_min += (!not_min & valid).count_ones() as u64;
                total  += valid.count_ones() as u64;
            }
        }
        if total == 0 { 0.0 } else { at_min as f64 / total as f64 }
    }

    // ---- interpretability ------------------------------------------------

    /// Return the included literals for clause `clause` of `class` as `(feature_index, is_negated)` pairs.
    pub fn clause_rule(&self, class: usize, clause: usize) -> Vec<(usize, bool)> {
        let mut rule = Vec::new();
        let tb = self.top_base(class, clause);
        for l in 0..self.n_literals {
            let included = (self.state[tb + l / WORD_BITS] >> (l % WORD_BITS)) & 1 != 0;
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
    use crate::clause_bank::dense::{clause_dec, clause_inc, clause_type_i, clause_type_ii, fire_predict};
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
        let mut tm = TsetlinMachine::with_config(2, 1, 2, 5, 3.0, 8, true, 1);
        let cb = tm.clause_base(0, 0);
        let valid0 = tm.valid[0];
        let words = tm.words;
        let sb = tm.state_bits;
        let chunk = &mut tm.state[cb..cb + sb * words];
        for _ in 0..1000 {
            clause_inc(chunk, words, sb, 0, valid0);
        }
        for b in 0..sb {
            assert_eq!(chunk[b * words] & valid0, valid0);
        }
        for _ in 0..1000 {
            clause_dec(chunk, words, sb, 0, valid0);
        }
        for b in 0..sb {
            assert_eq!(chunk[b * words] & valid0, 0);
        }
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
        let state_before = tm.state.clone();
        tm.fit_epoch(&btr, &ytr);
        let state_changed = tm.state.iter().zip(&state_before).filter(|(a, b)| a != b).count();
        let total = tm.state.len();
        assert!(
            state_changed < total / 100,
            "drop_p≈1 should leave >99% of state unchanged, but {state_changed}/{total} changed"
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
        let words = 1usize;
        let sb = 8usize;
        let bw = sb * words;
        let mut chunk = vec![0u64; bw];
        let top = (sb - 1) * words;

        let max_included = 2usize;
        chunk[top] = 0b11; // 2 bits already included

        let lit = vec![0b1111_1111u64];
        let valid = vec![0b1111_1111u64];
        let inv_mask = vec![!0u64];
        let keep_mask = vec![!0u64];
        let all_active = vec![!0u64; words];
        let mut weight = 5i32;

        clause_type_i(
            &mut chunk, &mut weight, &lit, &valid,
            words, sb, false, &inv_mask, &keep_mask, 10, max_included, &all_active,
        );

        let n_after = (chunk[top] & valid[0]).count_ones() as usize;
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
        let state = vec![0u64; words];
        let lit = vec![0u64; words];
        let valid = vec![!0u64; words];
        assert!(!fire_predict(&state, 0, &lit, &valid, words));
    }

    #[test]
    fn fire_predict_satisfied_clause_returns_true() {
        let words = 1usize;
        let state = vec![1u64];
        let lit = vec![1u64];
        let valid = vec![1u64];
        assert!(fire_predict(&state, 0, &lit, &valid, words));
    }

    #[test]
    fn fire_predict_violated_clause_returns_false() {
        let words = 1usize;
        let state = vec![1u64];
        let lit = vec![0u64];
        let valid = vec![1u64];
        assert!(!fire_predict(&state, 0, &lit, &valid, words));
    }

    // ---- literal_drop_p -------------------------------------------------------

    #[test]
    fn literal_drop_p_one_leaves_state_unchanged() {
        let (xtr, ytr) = make_xor(200, 0.0, 30);
        let btr = enc(12).encode_batch(&as_slices(&xtr));
        let mut tm = TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 42)
            .literal_drop_p(0.9999);
        let state_before = tm.state.clone();
        tm.fit_epoch(&btr, &ytr);
        let state_changed = tm.state.iter().zip(&state_before).filter(|(a, b)| a != b).count();
        let total = tm.state.len();
        assert!(
            state_changed < total / 100,
            "literal_drop_p≈1 should leave >99% of state unchanged, but {state_changed}/{total} changed"
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

    /// Build a one-word chunk with `sb` state planes; `states[l]` is the integer TA state for literal bit `l`.
    fn make_chunk(sb: usize, states: &[(usize, u64)]) -> Vec<u64> {
        let words = 1usize;
        let mut chunk = vec![0u64; sb * words];
        for &(l, v) in states {
            for b in 0..sb {
                if (v >> b) & 1 == 1 {
                    chunk[b] |= 1u64 << l;
                }
            }
        }
        chunk
    }

    /// Read back the integer TA state for literal bit `l` from a one-word chunk.
    fn read_state(chunk: &[u64], sb: usize, l: usize) -> u64 {
        (0..sb).fold(0u64, |acc, b| acc | (((chunk[b] >> l) & 1) << b))
    }

    #[test]
    fn absorbing_include_at_max_resists_ib() {
        // A literal at the maximum TA state (all sb planes set) must survive
        // 1 000 rounds of Type Ib "exclude absent" feedback completely unchanged.
        let words = 1usize;
        let sb = 4usize;
        let max_state = (1u64 << sb) - 1; // 0b1111
        // Literal 0 at max state; literal 0 absent from x → violation → Ib path.
        let mut chunk = make_chunk(sb, &[(0, max_state)]);

        let lit = vec![0u64];
        let valid = vec![1u64];
        let inv_mask = vec![!0u64];
        let keep_mask = vec![!0u64];
        let all_active = vec![!0u64; words];
        let mut weight = 1i32;

        for _ in 0..1_000 {
            clause_type_i(
                &mut chunk, &mut weight, &lit, &valid,
                words, sb, false, &inv_mask, &keep_mask, 100, usize::MAX, &all_active,
            );
        }

        assert_eq!(
            read_state(&chunk, sb, 0), max_state,
            "absorbing max-state literal must resist all Ib decrement pressure"
        );
    }

    #[test]
    fn non_max_include_is_decremented_by_ib() {
        // A literal one step below max (plane 0 missing) is NOT at the absorbing
        // state and must be decremented by a single Ib round.
        let words = 1usize;
        let sb = 4usize;
        let below_max = ((1u64 << sb) - 1) & !1u64; // 0b1110 — top plane still set → included
        let mut chunk = make_chunk(sb, &[(0, below_max)]);
        let before = read_state(&chunk, sb, 0);

        let lit = vec![0u64]; // absent → violation on included literal → Ib path
        let valid = vec![1u64];
        let inv_mask = vec![!0u64];
        let keep_mask = vec![!0u64];
        let all_active = vec![!0u64; words];
        let mut weight = 1i32;

        clause_type_i(
            &mut chunk, &mut weight, &lit, &valid,
            words, sb, false, &inv_mask, &keep_mask, 100, usize::MAX, &all_active,
        );

        assert_ne!(
            read_state(&chunk, sb, 0), before,
            "non-max literal must be decremented by Ib feedback"
        );
    }

    #[test]
    fn absorbing_exclude_at_min_resists_type_ii() {
        // A literal at the minimum TA state (all sb planes clear) must survive
        // 1 000 rounds of Type II "include absent excluded" feedback unchanged.
        let words = 1usize;
        let sb = 4usize;
        // All-zero chunk: every literal at min state; empty clause fires (no violations).
        let mut chunk = vec![0u64; sb * words];

        let lit = vec![0u64]; // literal 0 absent from x
        let valid = vec![1u64];
        let all_active = vec![!0u64; words];
        let mut weight = 5i32;

        for _ in 0..1_000 {
            clause_type_ii(
                &mut chunk, &mut weight, &lit, &valid,
                words, sb, &all_active,
            );
        }

        assert_eq!(
            read_state(&chunk, sb, 0), 0,
            "absorbing min-state literal must resist all Type II increment pressure"
        );
    }

    #[test]
    fn absorbing_stabilizes_clause_at_literal_limit() {
        // Two included literals, max_included_literals = 1 (over the limit → always Ib).
        //   literal 0: max state (absorbing) — should survive
        //   literal 1: only top-plane set (just in include action, not absorbing) — should be expelled
        // After enough rounds the clause should settle to exactly literal 0.
        let words = 1usize;
        let sb = 4usize;
        let max_state = (1u64 << sb) - 1;
        let top_only = 1u64 << (sb - 1); // only top plane = 0b1000 for sb=4
        let mut chunk = make_chunk(sb, &[(0, max_state), (1, top_only)]);

        let lit = vec![0u64]; // both absent → violations on both → Ib path
        let valid = vec![0b11u64];
        let inv_mask = vec![!0u64];
        let keep_mask = vec![!0u64];
        let all_active = vec![!0u64; words];
        let mut weight = 1i32;

        for _ in 0..500 {
            clause_type_i(
                &mut chunk, &mut weight, &lit, &valid,
                words, sb, false, &inv_mask, &keep_mask, 5, 1, &all_active,
            );
        }

        let top = sb - 1;
        assert_eq!((chunk[top] >> 0) & 1, 1, "absorbing literal 0 must stay included");
        assert_eq!((chunk[top] >> 1) & 1, 0, "non-absorbing literal 1 must be expelled");
    }
}
