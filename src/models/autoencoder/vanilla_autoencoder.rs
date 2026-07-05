//! Vanilla weighted Tsetlin Machine autoencoder.
//!
//! Mirrors TMU's `vanilla_autoencoder.py` / `TMAutoEncoder`.

#[cfg(feature = "parallel")]
use crate::clause_bank::dense::{use_parallel, PARALLEL_MIN};
use crate::clause_bank::dense::{
    bmask_word, clause_fire, digits_of, expand_bits_to_bytes, fire_predict, rebuild_include,
    type_i_update_bytes, type_ii_update_bytes, type_iii_update, words_for, GOLDEN, MASK_BITS,
    WORD_BITS,
};
use crate::encoder::{EncodedBatch, EncodedSample};
use crate::rng::Rng;

/// Unsupervised Tsetlin Machine that learns to reconstruct its binary input.
///
/// Each output bit position `o` has `clauses_per_output` dedicated clauses (half positive,
/// half negative).  Training derives the target for bit `o` directly from the input sample
/// — no class labels are required.  Inference reconstructs a binary vector from an encoded
/// sample by thresholding the weighted clause vote for each output bit.
///
/// Structurally identical to [`TsetlinMachine`] except the `n_classes` dimension is replaced
/// by `n_features` (input = output) and the training loop iterates over all output positions
/// rather than updating one true class and one sampled negative class.
///
/// [`TsetlinMachine`]: crate::TsetlinMachine
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TMAutoEncoder {
    n_features: usize,
    n_literals: usize,
    words: usize,
    clauses_per_output: usize,
    threshold: i32,
    s: f64,
    boost_true_positive: bool,
    max_included_literals: usize,
    clause_drop_p: f64,
    literal_drop_p: f64,
    literal_rng: Rng,
    dig_lit_active: Vec<u8>,

    /// u8 TA counters. Output `o`, clause `j` occupies
    /// `ta[(o*CPO + j) * n_literals .. (o*CPO + j + 1) * n_literals]`.
    ta: Vec<u8>,
    /// Include bitset. Same indexing as `ta` but in 64-bit words.
    include: Vec<u64>,
    half: u8,
    max_state: u8,

    weights: Vec<i32>,
    rngs: Vec<Rng>,
    output_rngs: Vec<Rng>,
    valid: Vec<u64>,
    dig_inv: Vec<u8>,
    dig_keep: Vec<u8>,

    ind: Vec<u8>,
    cat: Vec<u64>,
    d: f64,
    type_iii: bool,

    literals: Vec<u64>,
    rng: Rng,
}

#[cfg(feature = "serde")]
impl crate::serial::SaveLoad for TMAutoEncoder {
    const TAG: u8 = crate::serial::TAG_AUTOENCODER;
}

/// Per-clause feedback kernel (identical to the one in vanilla_classifier.rs).
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
        if !clause_fire(inc, lit, val, words, lit_active) {
            return;
        }
        *w = (*w - 1).max(1);
        type_ii_update_bytes(ta, n_literals, lit_b, active_b, half, max_state);
    }
    rebuild_include(ta, inc, val, words, n_literals, half);
}

impl TMAutoEncoder {
    /// Create a TMAutoEncoder with default settings: 8 state bits, boost enabled, seed 42.
    pub fn new(n_features: usize, clauses_per_output: usize, threshold: i32, s: f64) -> Self {
        Self::with_config(n_features, clauses_per_output, threshold, s, 8, true, 42)
    }

    /// Create a TMAutoEncoder with full configuration.
    ///
    /// * `state_bits` — TA counter precision in bits (2–8).
    /// * `boost_true_positive` — if `true`, Type Ia feedback always includes present literals.
    /// * `seed` — master RNG seed; fully deterministic for a given seed.
    pub fn with_config(
        n_features: usize,
        clauses_per_output: usize,
        threshold: i32,
        s: f64,
        state_bits: u8,
        boost_true_positive: bool,
        seed: u64,
    ) -> Self {
        assert!(n_features >= 1);
        assert!(clauses_per_output >= 2);
        assert!(threshold >= 1);
        assert!(s > 1.0);
        assert!((2..=8).contains(&state_bits), "state_bits must be in 2..=8");

        let state_bits = state_bits as usize;
        let n_literals = 2 * n_features;
        let words = words_for(n_literals);
        let n_clauses = n_features * clauses_per_output;
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
                ta[tb + l] = if rng.next_u64() & 1 == 0 {
                    half - 1
                } else {
                    half
                };
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

        let output_rngs = (0..n_features)
            .map(|o| Rng::new(seed ^ (o as u64 + n_clauses as u64 + 1).wrapping_mul(GOLDEN)))
            .collect();

        let literal_rng = Rng::new(seed ^ 0x4C49_5445_5241_4C21u64);

        TMAutoEncoder {
            n_features,
            n_literals,
            words,
            clauses_per_output,
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
            output_rngs,
            valid,
            dig_inv: digits_of(1.0 / s, MASK_BITS),
            dig_keep: digits_of((s - 1.0) / s, MASK_BITS),
            ind: vec![half; n_clauses * n_literals],
            cat: vec![0u64; n_clauses * words],
            d: 200.0,
            type_iii: false,
            literals: vec![0u64; words],
            rng,
        }
    }

    /// Limit how many literals each clause may include (Type Ia guard).
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

    /// Enable Type III feedback with indicator strength `d` (must be > 1.0).
    pub fn type_iii_feedback(mut self, d: f64) -> Self {
        assert!(d > 1.0, "d must be > 1.0");
        self.d = d;
        self.type_iii = true;
        self
    }

    // ---- accessors -----------------------------------------------------------

    /// Number of input (and output) features.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Number of clauses per output bit position.
    pub fn clauses_per_output(&self) -> usize {
        self.clauses_per_output
    }

    /// Number of 64-bit words per packed sample.
    pub fn words_per_sample(&self) -> usize {
        self.words
    }

    /// The specificity parameter `s`.
    pub fn s(&self) -> f64 {
        self.s
    }

    /// Integer weight of clause `clause` for output `output`.
    pub fn clause_weight(&self, output: usize, clause: usize) -> i32 {
        self.weights[output * self.clauses_per_output + clause]
    }

    // ---- inference -----------------------------------------------------------

    #[inline]
    fn reconstruct_lit(&self, lit: &[u64]) -> Vec<u8> {
        debug_assert_eq!(lit.len(), self.words);
        let cpo = self.clauses_per_output;
        let words = self.words;
        let n_features = self.n_features;
        let include = self.include.as_slice();
        let valid = self.valid.as_slice();

        let mut out = vec![0u8; n_features];
        for (o, out_o) in out.iter_mut().enumerate() {
            let cw = &self.weights[o * cpo..(o + 1) * cpo];
            let mut sum = 0i32;
            for (j, &w) in cw.iter().enumerate() {
                let cj = o * cpo + j;
                if fire_predict(&include[cj * words..(cj + 1) * words], lit, valid, words) {
                    if j & 1 == 0 {
                        sum += w;
                    } else {
                        sum -= w;
                    }
                }
            }
            *out_o = (sum > 0) as u8;
        }
        out
    }

    /// Reconstruct a binary vector from an encoded sample.
    ///
    /// Each output bit is 1 when the weighted clause vote for that position is positive.
    pub fn reconstruct(&self, sample: &EncodedSample) -> Vec<u8> {
        self.reconstruct_lit(&sample.0)
    }

    /// Reconstruct all samples in an encoded batch.
    pub fn reconstruct_batch(&self, batch: &EncodedBatch) -> Vec<Vec<u8>> {
        debug_assert_eq!(batch.data.len(), batch.n * self.words);
        let n = batch.n;
        let w = self.words;
        let packed = batch.data.as_slice();
        #[cfg(feature = "parallel")]
        if n >= PARALLEL_MIN && use_parallel(self.clauses_per_output, w) {
            use rayon::prelude::*;
            return (0..n)
                .into_par_iter()
                .map(|i| self.reconstruct_lit(&packed[i * w..(i + 1) * w]))
                .collect();
        }
        (0..n)
            .map(|i| self.reconstruct_lit(&packed[i * w..(i + 1) * w]))
            .collect()
    }

    /// Fill `out` with the clamped weighted clause sums for each output bit.
    pub fn reconstruct_scores(&self, sample: &EncodedSample, out: &mut [i32]) {
        let lit = &sample.0;
        debug_assert_eq!(out.len(), self.n_features);
        let cpo = self.clauses_per_output;
        let words = self.words;
        let include = self.include.as_slice();
        let valid = self.valid.as_slice();
        for (o, out_o) in out.iter_mut().enumerate() {
            let cw = &self.weights[o * cpo..(o + 1) * cpo];
            let mut sum = 0i32;
            for (j, &w) in cw.iter().enumerate() {
                let cj = o * cpo + j;
                if fire_predict(&include[cj * words..(cj + 1) * words], lit, valid, words) {
                    if j & 1 == 0 {
                        sum += w;
                    } else {
                        sum -= w;
                    }
                }
            }
            *out_o = sum.clamp(-self.threshold, self.threshold);
        }
    }

    /// Return the indices of all clauses (local to `output`) that fire for `sample`.
    pub fn fired_clauses(&self, sample: &EncodedSample, output: usize) -> Vec<usize> {
        let lit = &sample.0;
        let cpo = self.clauses_per_output;
        let words = self.words;
        let include = self.include.as_slice();
        let valid = self.valid.as_slice();
        (0..cpo)
            .filter(|&j| fire_predict(&include[(output * cpo + j) * words..(output * cpo + j + 1) * words], lit, valid, words))
            .collect()
    }

    // ---- training helpers ----------------------------------------------------

    fn output_sum_train(&self, o: usize, lit_active: &[u64]) -> i32 {
        let cpo = self.clauses_per_output;
        let words = self.words;
        let include = self.include.as_slice();
        let valid = self.valid.as_slice();
        let lit = self.literals.as_slice();
        let mut sum = 0i32;
        for j in 0..cpo {
            let cj = o * cpo + j;
            if clause_fire(
                &include[cj * words..(cj + 1) * words],
                lit,
                valid,
                words,
                lit_active,
            ) {
                let w = self.weights[o * cpo + j];
                if j & 1 == 0 {
                    sum += w;
                } else {
                    sum -= w;
                }
            }
        }
        sum.clamp(-self.threshold, self.threshold)
    }

    fn update_output(
        &mut self,
        o: usize,
        target: u8,
        sum: i32,
        lit_active: &[u64],
        lit_b: &[u8],
        active_b: &[u8],
    ) {
        let cpo = self.clauses_per_output;
        let words = self.words;
        let n_literals = self.n_literals;
        let boost = self.boost_true_positive;
        let wmax = self.threshold;
        let max_inc = self.max_included_literals;
        let drop_p = self.clause_drop_p;
        let half = self.half;
        let max_state = self.max_state;
        let type_iii_en = self.type_iii;
        let d_val = self.d;
        let target_bool = target == 1;

        let Self {
            ta,
            include,
            weights,
            rngs,
            output_rngs,
            literals,
            valid,
            dig_inv,
            dig_keep,
            ind,
            cat,
            ..
        } = self;
        let lit = literals.as_slice();
        let val = valid.as_slice();
        let orng = &mut output_rngs[o];

        let t = wmax as f64;
        let v = sum as f64;
        let p = if target == 1 {
            ((t - v) / (2.0 * t)).min(1.0)
        } else {
            ((t + v) / (2.0 * t)).min(1.0)
        };

        let drop_mask: Vec<bool> = if drop_p > 0.0 {
            (0..cpo).map(|_| orng.next_f64() < drop_p).collect()
        } else {
            vec![]
        };

        let inv_mask: Vec<u64> = (0..words).map(|_| bmask_word(orng, dig_inv)).collect();
        let keep_mask: Vec<u64> = (0..words).map(|_| bmask_word(orng, dig_keep)).collect();
        let inv_b = expand_bits_to_bytes(&inv_mask, n_literals);
        let keep_b = expand_bits_to_bytes(&keep_mask, n_literals);

        let out_ta = &mut ta[o * cpo * n_literals..(o + 1) * cpo * n_literals];
        let out_inc = &mut include[o * cpo * words..(o + 1) * cpo * words];
        let out_w = &mut weights[o * cpo..(o + 1) * cpo];
        let out_rng = &mut rngs[o * cpo..(o + 1) * cpo];
        let out_ind = &mut ind[o * cpo * n_literals..(o + 1) * cpo * n_literals];
        let out_cat = &mut cat[o * cpo * words..(o + 1) * cpo * words];

        #[cfg(feature = "parallel")]
        if cpo >= PARALLEL_MIN {
            use rayon::prelude::*;
            if type_iii_en {
                out_ta
                    .par_chunks_mut(n_literals)
                    .zip(out_inc.par_chunks_mut(words))
                    .zip(out_w.par_iter_mut())
                    .zip(out_rng.par_iter_mut())
                    .zip(out_ind.par_chunks_mut(n_literals))
                    .zip(out_cat.par_chunks_mut(words))
                    .enumerate()
                    .for_each(|(j, (((((ta_o, inc_o), w), rng), ind_o), cat_o))| {
                        apply_one_clause(
                            j, ta_o, inc_o, w, rng, target, p, &drop_mask, lit, val, lit_active,
                            words, lit_b, &inv_b, &keep_b, active_b, n_literals, boost, wmax,
                            max_inc, half, max_state,
                        );
                        if drop_mask.is_empty() || !drop_mask[j] {
                            if type_iii_update(
                                ta_o, ind_o, cat_o, inc_o, lit, val, lit_active, active_b, words, n_literals,
                                d_val, p, target_bool, rng, half, max_state,
                            ) {
                                rebuild_include(ta_o, inc_o, val, words, n_literals, half);
                            }
                        }
                    });
            } else {
                out_ta
                    .par_chunks_mut(n_literals)
                    .zip(out_inc.par_chunks_mut(words))
                    .zip(out_w.par_iter_mut())
                    .zip(out_rng.par_iter_mut())
                    .enumerate()
                    .for_each(|(j, (((ta_o, inc_o), w), rng))| {
                        apply_one_clause(
                            j, ta_o, inc_o, w, rng, target, p, &drop_mask, lit, val, lit_active,
                            words, lit_b, &inv_b, &keep_b, active_b, n_literals, boost, wmax,
                            max_inc, half, max_state,
                        );
                    });
            }
            return;
        }

        for j in 0..cpo {
            apply_one_clause(
                j,
                &mut out_ta[j * n_literals..(j + 1) * n_literals],
                &mut out_inc[j * words..(j + 1) * words],
                &mut out_w[j],
                &mut out_rng[j],
                target,
                p,
                &drop_mask,
                lit,
                val,
                lit_active,
                words,
                lit_b,
                &inv_b,
                &keep_b,
                active_b,
                n_literals,
                boost,
                wmax,
                max_inc,
                half,
                max_state,
            );
            if type_iii_en && (drop_mask.is_empty() || !drop_mask[j]) {
                if type_iii_update(
                    &mut out_ta[j * n_literals..(j + 1) * n_literals],
                    &mut out_ind[j * n_literals..(j + 1) * n_literals],
                    &mut out_cat[j * words..(j + 1) * words],
                    &out_inc[j * words..(j + 1) * words],
                    lit,
                    val,
                    lit_active,
                    active_b,
                    words,
                    n_literals,
                    d_val,
                    p,
                    target_bool,
                    &mut out_rng[j],
                    half,
                    max_state,
                ) {
                    rebuild_include(
                        &out_ta[j * n_literals..(j + 1) * n_literals],
                        &mut out_inc[j * words..(j + 1) * words],
                        val,
                        words,
                        n_literals,
                        half,
                    );
                }
            }
        }
    }

    fn fit_one_lit(&mut self, lit: &[u64]) {
        debug_assert_eq!(lit.len(), self.words);
        self.literals.copy_from_slice(lit);

        let n_literals = self.n_literals;
        let words = self.words;
        let n_features = self.n_features;

        let lit_b = expand_bits_to_bytes(lit, n_literals);

        let mut lit_active: Vec<u64> = if self.literal_drop_p > 0.0 {
            let rng = &mut self.literal_rng;
            let dig = &self.dig_lit_active;
            (0..words).map(|_| bmask_word(rng, dig)).collect()
        } else {
            vec![!0u64; words]
        };

        let mut active_b = expand_bits_to_bytes(&lit_active, n_literals);

        for o in 0..n_features {
            let target = ((lit[o / WORD_BITS] >> (o % WORD_BITS)) & 1) as u8;

            // Mask out literal o and its negation (o + n_features) from the active mask
            // before training output o. This prevents trivial memorization where the
            // clause uses feature o directly to predict output o (proper autoencoder).
            let neg_o = o + n_features;
            let word_o = o / WORD_BITS;
            let word_neg_o = neg_o / WORD_BITS;
            let save_word_o = lit_active[word_o];
            let save_word_neg_o = lit_active[word_neg_o];
            let save_ab_o = active_b[o];
            let save_ab_neg_o = active_b[neg_o];
            lit_active[word_o] &= !(1u64 << (o % WORD_BITS));
            lit_active[word_neg_o] &= !(1u64 << (neg_o % WORD_BITS));
            active_b[o] = 0;
            active_b[neg_o] = 0;

            let sum = self.output_sum_train(o, &lit_active);
            self.update_output(o, target, sum, &lit_active, &lit_b, &active_b);

            // Restore the active mask for the next output's iteration.
            lit_active[word_o] = save_word_o;
            lit_active[word_neg_o] = save_word_neg_o;
            active_b[o] = save_ab_o;
            active_b[neg_o] = save_ab_neg_o;
        }
    }

    /// Train on a single encoded sample (unsupervised — no label required).
    pub fn fit_one(&mut self, sample: &EncodedSample) {
        self.fit_one_lit(&sample.0);
    }

    /// Run one training epoch over an encoded batch, shuffling the order each epoch.
    pub fn fit_epoch(&mut self, batch: &EncodedBatch) {
        debug_assert_eq!(batch.words, self.words);
        let n = batch.n;
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let k = self.rng.below(i + 1);
            order.swap(i, k);
        }
        let w = self.words;
        let data = batch.data.as_slice();
        for &i in &order {
            self.fit_one_lit(&data[i * w..(i + 1) * w]);
        }
    }

    // ---- evaluation ----------------------------------------------------------

    /// Fraction of (sample, bit) pairs correctly reconstructed across the batch.
    ///
    /// A value of 1.0 means perfect reconstruction; 0.5 is the random baseline
    /// for balanced binary inputs.
    pub fn reconstruction_accuracy(&self, batch: &EncodedBatch) -> f64 {
        debug_assert_eq!(batch.data.len(), batch.n * self.words);
        let n = batch.n;
        let w = self.words;
        let nf = self.n_features;
        let packed = batch.data.as_slice();

        #[cfg(feature = "parallel")]
        if n >= PARALLEL_MIN && use_parallel(self.clauses_per_output, w) {
            use rayon::prelude::*;
            let correct: usize = (0..n)
                .into_par_iter()
                .map(|i| {
                    let lit = &packed[i * w..(i + 1) * w];
                    let recon = self.reconstruct_lit(lit);
                    (0..nf)
                        .filter(|&o| {
                            let actual = ((lit[o / WORD_BITS] >> (o % WORD_BITS)) & 1) as u8;
                            recon[o] == actual
                        })
                        .count()
                })
                .sum();
            return correct as f64 / (n * nf) as f64;
        }

        let mut correct = 0usize;
        for i in 0..n {
            let lit = &packed[i * w..(i + 1) * w];
            let recon = self.reconstruct_lit(lit);
            for o in 0..nf {
                let actual = ((lit[o / WORD_BITS] >> (o % WORD_BITS)) & 1) as u8;
                correct += (recon[o] == actual) as usize;
            }
        }
        correct as f64 / (n * nf) as f64
    }

    // ---- absorbing state introspection ---------------------------------------

    /// Fraction of (clause, literal) pairs at the absorbing include state (counter == max_state).
    pub fn absorbed_include_fraction(&self) -> f64 {
        let n_clauses = self.n_features * self.clauses_per_output;
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
        if total == 0 {
            0.0
        } else {
            at_max as f64 / total as f64
        }
    }

    /// Fraction of (clause, literal) pairs at the absorbing exclude state (counter == 0).
    pub fn absorbed_exclude_fraction(&self) -> f64 {
        let n_clauses = self.n_features * self.clauses_per_output;
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
        if total == 0 {
            0.0
        } else {
            at_min as f64 / total as f64
        }
    }

    // ---- interpretability ----------------------------------------------------

    /// Included literals for clause `clause` of output `output` as `(feature_index, is_negated)`.
    pub fn clause_rule(&self, output: usize, clause: usize) -> Vec<(usize, bool)> {
        let mut rule = Vec::new();
        let cj = output * self.clauses_per_output + clause;
        let inc = &self.include[cj * self.words..(cj + 1) * self.words];
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

    /// Returns `true` if `clause` is a positive clause (even index — votes for output bit = 1).
    pub fn clause_is_positive(&self, clause: usize) -> bool {
        clause & 1 == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::Encoder;

    fn enc(n_features: usize) -> Encoder {
        Encoder::for_binary(n_features)
    }

    fn make_bits(n: usize, seed: u64) -> Vec<Vec<u8>> {
        let mut rng = Rng::new(seed);
        (0..n)
            .map(|_| (0..12).map(|_| (rng.next_u64() & 1) as u8).collect())
            .collect()
    }

    fn as_slices(xs: &[Vec<u8>]) -> Vec<&[u8]> {
        xs.iter().map(|v| v.as_slice()).collect()
    }

    // ---- functional ----------------------------------------------------------

    #[test]
    fn autoencoder_reconstructs_above_chance() {
        // Use structured data: bits 6-11 are copies of bits 0-5.
        // The autoencoder can learn bit 6+i from bit i (and vice versa).
        // Random i.i.d. data can't exceed ~0.5 (nothing to learn per feature).
        let mut rng = Rng::new(1);
        let xs: Vec<Vec<u8>> = (0..2000)
            .map(|_| {
                let half: Vec<u8> = (0..6).map(|_| (rng.next_u64() & 1) as u8).collect();
                let mut f = half.clone();
                f.extend_from_slice(&half);
                f
            })
            .collect();
        let e = enc(12);
        let batch = e.encode_batch(&as_slices(&xs));

        let mut ae = TMAutoEncoder::with_config(12, 20, 15, 3.9, 8, true, 42);
        for _ in 0..20 {
            ae.fit_epoch(&batch);
        }
        let acc = ae.reconstruction_accuracy(&batch);
        assert!(
            acc > 0.8,
            "expected reconstruction accuracy > 0.80, got {acc:.4}"
        );
    }

    #[test]
    fn autoencoder_learns_structured_input() {
        // Data: bits 6-11 mirror bits 0-5. Every output bit can be predicted
        // from its mirror counterpart — ensures all bits are learnable after
        // the feature-o masking fix (XOR-only structure limits learnable bits).
        let mut rng = Rng::new(7);
        let n = 3000usize;
        let xs: Vec<Vec<u8>> = (0..n)
            .map(|_| {
                let half: Vec<u8> = (0..6).map(|_| (rng.next_u64() & 1) as u8).collect();
                let mut f = half.clone();
                f.extend_from_slice(&half);
                f
            })
            .collect();
        let e = enc(12);
        let batch = e.encode_batch(&as_slices(&xs));

        let mut ae = TMAutoEncoder::with_config(12, 20, 15, 3.9, 8, true, 42);
        for _ in 0..25 {
            ae.fit_epoch(&batch);
        }
        let acc = ae.reconstruction_accuracy(&batch);
        assert!(
            acc > 0.80,
            "expected structured-input accuracy > 0.80, got {acc:.4}"
        );
    }

    // ---- behavioral / contract -----------------------------------------------

    #[test]
    fn same_seed_same_result() {
        let xs = make_bits(500, 10);
        let e = enc(12);
        let batch = e.encode_batch(&as_slices(&xs));

        let mut ae1 = TMAutoEncoder::with_config(12, 8, 10, 3.0, 8, true, 99);
        let mut ae2 = TMAutoEncoder::with_config(12, 8, 10, 3.0, 8, true, 99);
        for _ in 0..5 {
            ae1.fit_epoch(&batch);
            ae2.fit_epoch(&batch);
        }

        let xs_te = make_bits(100, 20);
        let bte = e.encode_batch(&as_slices(&xs_te));
        assert_eq!(
            ae1.reconstruction_accuracy(&bte),
            ae2.reconstruction_accuracy(&bte),
            "same seed must produce identical results"
        );
    }

    #[test]
    fn weights_stay_in_1_to_threshold() {
        let threshold = 20i32;
        let xs = make_bits(1000, 11);
        let batch = enc(12).encode_batch(&as_slices(&xs));
        let mut ae = TMAutoEncoder::with_config(12, 12, threshold, 3.9, 8, true, 42);
        for _ in 0..10 {
            ae.fit_epoch(&batch);
        }
        for o in 0..12 {
            for j in 0..ae.clauses_per_output() {
                let w = ae.clause_weight(o, j);
                assert!(
                    (1..=threshold).contains(&w),
                    "weight {w} out of [1, {threshold}] for output ({o},{j})"
                );
            }
        }
    }

    #[test]
    fn reconstruct_batch_matches_single() {
        let xs = make_bits(300, 12);
        let e = enc(12);
        let mut ae = TMAutoEncoder::with_config(12, 8, 10, 3.0, 8, true, 42);
        ae.fit_epoch(&e.encode_batch(&as_slices(&xs)));

        let xs_te = make_bits(100, 22);
        let batch = e.encode_batch(&as_slices(&xs_te));
        let n = xs_te.len();
        let w = ae.words_per_sample();

        let from_batch = ae.reconstruct_batch(&batch);
        let from_single: Vec<Vec<u8>> = (0..n)
            .map(|i| ae.reconstruct_lit(&batch.data[i * w..(i + 1) * w]))
            .collect();

        assert_eq!(from_batch, from_single);
    }

    #[test]
    fn reconstruction_accuracy_matches_manual() {
        let xs = make_bits(500, 13);
        let e = enc(12);
        let mut ae = TMAutoEncoder::with_config(12, 8, 10, 3.0, 8, true, 42);
        ae.fit_epoch(&e.encode_batch(&as_slices(&xs)));

        let xs_te = make_bits(200, 23);
        let batch = e.encode_batch(&as_slices(&xs_te));
        let n = xs_te.len();
        let w = ae.words_per_sample();
        let nf = ae.n_features();

        let api_acc = ae.reconstruction_accuracy(&batch);
        let mut manual_correct = 0usize;
        for i in 0..n {
            let lit = &batch.data[i * w..(i + 1) * w];
            let recon = ae.reconstruct_lit(lit);
            for o in 0..nf {
                let actual = ((lit[o / WORD_BITS] >> (o % WORD_BITS)) & 1) as u8;
                manual_correct += (recon[o] == actual) as usize;
            }
        }
        let manual_acc = manual_correct as f64 / (n * nf) as f64;
        assert!((api_acc - manual_acc).abs() < 1e-12);
    }

    #[test]
    fn reconstruct_scores_threshold_matches_reconstruct() {
        let xs = make_bits(500, 14);
        let e = enc(12);
        let mut ae = TMAutoEncoder::with_config(12, 8, 10, 3.0, 8, true, 42);
        ae.fit_epoch(&e.encode_batch(&as_slices(&xs)));

        let xs_te = make_bits(50, 24);
        for x in &xs_te {
            let sample = e.encode_one(x);
            let recon = ae.reconstruct(&sample);
            let mut scores = vec![0i32; 12];
            ae.reconstruct_scores(&sample, &mut scores);
            for o in 0..12 {
                let expected = (scores[o] > 0) as u8;
                assert_eq!(
                    recon[o], expected,
                    "output {o}: reconstruct={} but scores[{o}]={} implies {}",
                    recon[o], scores[o], expected
                );
            }
        }
    }

    // ---- dropout -------------------------------------------------------------

    #[test]
    fn clause_drop_p_one_leaves_state_unchanged() {
        let xs = make_bits(200, 15);
        let batch = enc(12).encode_batch(&as_slices(&xs));
        let mut ae = TMAutoEncoder::with_config(12, 8, 10, 3.0, 8, true, 42).clause_drop_p(0.9999);
        let ta_before = ae.ta.clone();
        ae.fit_epoch(&batch);
        let ta_changed = ae.ta.iter().zip(&ta_before).filter(|(a, b)| a != b).count();
        let total = ae.ta.len();
        assert!(
            ta_changed < total / 100,
            "drop_p≈1 should leave >99% of state unchanged, but {ta_changed}/{total} changed"
        );
    }

    #[test]
    fn literal_drop_p_one_leaves_state_unchanged() {
        let xs = make_bits(200, 30);
        let batch = enc(12).encode_batch(&as_slices(&xs));
        let mut ae = TMAutoEncoder::with_config(12, 8, 10, 3.0, 8, true, 42).literal_drop_p(0.9999);
        let ta_before = ae.ta.clone();
        ae.fit_epoch(&batch);
        let ta_changed = ae.ta.iter().zip(&ta_before).filter(|(a, b)| a != b).count();
        let total = ae.ta.len();
        assert!(
            ta_changed < total / 100,
            "literal_drop_p≈1 should leave >99% of state unchanged, but {ta_changed}/{total} changed"
        );
    }

    // ---- max_included_literals -----------------------------------------------

    #[test]
    fn max_included_literals_reduces_clause_size() {
        let xs = make_bits(2000, 17);
        let batch = enc(12).encode_batch(&as_slices(&xs));

        let mut ae_tight =
            TMAutoEncoder::with_config(12, 10, 15, 3.9, 8, true, 42).max_included_literals(2);
        let mut ae_free = TMAutoEncoder::with_config(12, 10, 15, 3.9, 8, true, 42);
        for _ in 0..10 {
            ae_tight.fit_epoch(&batch);
            ae_free.fit_epoch(&batch);
        }

        let avg = |ae: &TMAutoEncoder| {
            let total: usize = (0..ae.n_features())
                .flat_map(|o| (0..ae.clauses_per_output()).map(move |j| (o, j)))
                .map(|(o, j)| ae.clause_rule(o, j).len())
                .sum();
            total as f64 / (ae.n_features() * ae.clauses_per_output()) as f64
        };
        let tight_avg = avg(&ae_tight);
        let free_avg = avg(&ae_free);
        assert!(
            tight_avg < free_avg,
            "max_included_literals=2 should produce smaller clauses: {tight_avg:.2} vs {free_avg:.2}"
        );
    }

    // ---- clause polarity / dimensions ----------------------------------------

    #[test]
    fn clause_is_positive_matches_index_parity() {
        let ae = TMAutoEncoder::with_config(4, 6, 10, 3.0, 8, true, 1);
        for j in 0..ae.clauses_per_output() {
            assert_eq!(ae.clause_is_positive(j), j & 1 == 0);
        }
    }

    #[test]
    fn words_per_sample_correct() {
        for &nf in &[1usize, 32, 63, 64, 65, 100, 128, 784] {
            let ae = TMAutoEncoder::with_config(nf, 2, 5, 2.0, 8, true, 1);
            let expected = (2 * nf).div_ceil(64);
            assert_eq!(
                ae.words_per_sample(),
                expected,
                "n_features={nf}: expected {expected} words"
            );
        }
    }

    // ---- absorbing states ----------------------------------------------------

    #[test]
    fn absorbed_fractions_start_at_zero() {
        let ae = TMAutoEncoder::with_config(12, 10, 15, 3.9, 8, true, 42);
        assert_eq!(ae.absorbed_include_fraction(), 0.0);
        assert_eq!(ae.absorbed_exclude_fraction(), 0.0);
    }

    #[test]
    fn absorbed_fractions_increase_with_training() {
        let xs = make_bits(3000, 62);
        let batch = enc(12).encode_batch(&as_slices(&xs));
        let mut ae = TMAutoEncoder::with_config(12, 10, 15, 3.9, 4, true, 42);
        let before = ae.absorbed_include_fraction() + ae.absorbed_exclude_fraction();
        for _ in 0..50 {
            ae.fit_epoch(&batch);
        }
        let after = ae.absorbed_include_fraction() + ae.absorbed_exclude_fraction();
        assert!(
            after > before,
            "absorbing states must grow during training: before={before:.4}, after={after:.4}"
        );
    }

    // ---- state bits boundary -------------------------------------------------

    #[test]
    fn state_bits_2_trains_without_panic() {
        let xs = make_bits(500, 60);
        let xs_te = make_bits(200, 61);
        let e = enc(12);
        let btr = e.encode_batch(&as_slices(&xs));
        let bte = e.encode_batch(&as_slices(&xs_te));
        let mut ae = TMAutoEncoder::with_config(12, 8, 10, 3.0, 2, true, 42);
        for _ in 0..5 {
            ae.fit_epoch(&btr);
        }
        let acc = ae.reconstruction_accuracy(&bte);
        assert!(
            (0.0..=1.0).contains(&acc),
            "state_bits=2 accuracy out of range: {acc}"
        );
    }

    #[test]
    fn state_bits_8_max_state_is_255() {
        let ae = TMAutoEncoder::with_config(4, 2, 5, 2.0, 8, true, 1);
        assert_eq!(ae.max_state, 255u8, "state_bits=8 → max_state must be 255");
        assert_eq!(ae.half, 128u8, "state_bits=8 → half must be 128");
    }

    // ---- save / load (requires the `serde` feature) --------------------------

    #[cfg(feature = "serde")]
    use crate::serial::{self, SaveLoad};

    /// Structured data where bits 6..12 mirror bits 0..6 (learnable correlations).
    #[cfg(feature = "serde")]
    fn make_mirror(n: usize, seed: u64) -> Vec<Vec<u8>> {
        let mut rng = Rng::new(seed);
        (0..n)
            .map(|_| {
                let half: Vec<u8> = (0..6).map(|_| (rng.next_u64() & 1) as u8).collect();
                let mut f = half.clone();
                f.extend_from_slice(&half);
                f
            })
            .collect()
    }

    #[cfg(feature = "serde")]
    #[test]
    fn save_load_roundtrip_reconstructs_identically() {
        let xs = make_mirror(2000, 1);
        let e = enc(12);
        let btr = e.encode_batch(&as_slices(&xs));

        let mut ae = TMAutoEncoder::with_config(12, 8, 15, 3.9, 8, true, 7);
        for _ in 0..15 {
            ae.fit_epoch(&btr);
        }

        let mut buf = Vec::new();
        ae.write_to(&mut buf).unwrap();
        let loaded = TMAutoEncoder::read_from(&mut buf.as_slice()).unwrap();

        let xte = make_mirror(500, 2);
        let bte = e.encode_batch(&as_slices(&xte));
        assert_eq!(ae.reconstruct_batch(&bte), loaded.reconstruct_batch(&bte));
        assert_eq!(
            ae.reconstruction_accuracy(&bte),
            loaded.reconstruction_accuracy(&bte)
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn save_load_resumes_training_without_reinit() {
        let xs = make_mirror(2000, 3);
        let e = enc(12);
        let btr = e.encode_batch(&as_slices(&xs));

        let mut ae = TMAutoEncoder::with_config(12, 8, 15, 3.9, 8, true, 7);
        for _ in 0..10 {
            ae.fit_epoch(&btr);
        }

        let mut buf = Vec::new();
        ae.write_to(&mut buf).unwrap();
        let mut loaded = TMAutoEncoder::read_from(&mut buf.as_slice()).unwrap();

        for _ in 0..10 {
            ae.fit_epoch(&btr);
            loaded.fit_epoch(&btr);
        }

        assert_eq!(ae.ta, loaded.ta, "TA counters diverged after resume");
        assert_eq!(ae.weights, loaded.weights, "weights diverged after resume");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn save_load_via_file_path() {
        let xs = make_mirror(500, 5);
        let e = enc(12);
        let btr = e.encode_batch(&as_slices(&xs));
        let mut ae = TMAutoEncoder::with_config(12, 4, 10, 3.0, 8, true, 42);
        ae.fit_epoch(&btr);

        let mut path = std::env::temp_dir();
        path.push("tmu_rs_autoencoder_roundtrip.tmrs");
        ae.save(&path).unwrap();
        let loaded = TMAutoEncoder::load(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let bte = e.encode_batch(&as_slices(&make_mirror(200, 6)));
        assert_eq!(ae.reconstruct_batch(&bte), loaded.reconstruct_batch(&bte));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn load_rejects_corrupt_data() {
        assert!(TMAutoEncoder::read_from(&mut [].as_slice()).is_err());
        assert!(TMAutoEncoder::read_from(&mut b"XXXXjunkjunk".as_slice()).is_err());
        let mut buf = Vec::new();
        serial::write_header(&mut buf, serial::TAG_VANILLA).unwrap();
        assert!(TMAutoEncoder::read_from(&mut buf.as_slice()).is_err());
    }

    #[test]
    fn type_iii_constructs_without_panic() {
        let _ae = TMAutoEncoder::with_config(12, 20, 15, 3.9, 8, true, 42)
            .type_iii_feedback(200.0);
    }

    #[test]
    fn type_iii_d_must_be_greater_than_one() {
        let result = std::panic::catch_unwind(|| {
            TMAutoEncoder::with_config(12, 20, 15, 3.9, 8, true, 42).type_iii_feedback(0.5)
        });
        assert!(result.is_err(), "expected panic for d <= 1.0");
    }

    #[test]
    fn type_iii_trains_without_panic() {
        let xs = make_bits(300, 7);
        let e = enc(12);
        let batch = e.encode_batch(&as_slices(&xs));
        let mut ae = TMAutoEncoder::with_config(12, 20, 15, 3.9, 8, true, 42)
            .type_iii_feedback(200.0);
        for _ in 0..5 {
            ae.fit_epoch(&batch);
        }
    }
}
