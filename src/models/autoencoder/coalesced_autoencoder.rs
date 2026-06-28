//! Coalesced Tsetlin Machine autoencoder.
//!
//! A **single shared clause bank** of `n_clauses` clauses is scored by every output
//! bit position via signed per-output weights, mirroring the coalesced architecture
//! from [`crate::CoalescedTsetlinMachine`].

#[cfg(feature = "parallel")]
use crate::clause_bank::dense::PARALLEL_MIN;
use crate::clause_bank::dense::{
    bmask_word, clause_fire, digits_of, expand_bits_to_bytes, fire_predict, rebuild_include,
    type_i_update_bytes, type_ii_update_bytes, type_iii_update, words_for, GOLDEN, MASK_BITS,
    WORD_BITS,
};
use crate::encoder::{EncodedBatch, EncodedSample};
use crate::rng::Rng;

/// Coalesced Tsetlin Machine autoencoder with a single shared clause bank.
///
/// Unlike [`crate::TMAutoEncoder`], which gives each output bit its own dedicated clause
/// pool, this model keeps one shared bank of `n_clauses` clauses voted on by every output
/// position via a signed per-output weight matrix.  Polarity is the *sign* of the weight
/// rather than clause-index parity; weights are initialised to random ±1 and may grow
/// without bound.
///
/// Feature masking is applied during training: when updating output bit `o`, literals `o`
/// and `o + n_features` are excluded from the active set to prevent trivial self-prediction.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TMCoalescedAutoEncoder {
    n_features: usize,
    n_literals: usize,
    words: usize,
    /// Number of clauses in the single shared clause bank.
    n_clauses: usize,
    threshold: i32,
    s: f64,
    boost_true_positive: bool,
    max_included_literals: usize,
    clause_drop_p: f64,
    literal_drop_p: f64,
    literal_rng: Rng,
    dig_lit_active: Vec<u8>,

    /// u8 TA counters. Clause `j` occupies `ta[j * n_literals .. (j+1) * n_literals]`.
    ta: Vec<u8>,
    /// Include bitset. Clause `j` occupies `include[j * words .. (j+1) * words]`.
    include: Vec<u64>,
    half: u8,
    max_state: u8,

    /// Signed per-output weights indexed `output * n_clauses + clause`.
    /// Initialised to random ±1; unbounded growth during training.
    weights: Vec<i32>,
    /// Per-clause RNGs (n_clauses total).
    rngs: Vec<Rng>,
    /// Per-output RNGs for inv/keep Bernoulli masks and clause dropout.
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
impl crate::serial::SaveLoad for TMCoalescedAutoEncoder {
    const TAG: u8 = crate::serial::TAG_COALESCED_AUTOENCODER;
}

/// Per-clause feedback + weight-update kernel — identical to the one in
/// `coalesced_classifier.rs`.  Weight sign determines polarity; two independent
/// Bernoulli(`p`) draws gate the literal feedback and the weight step separately.
#[allow(clippy::too_many_arguments)]
#[inline]
fn apply_one_clause_coalesced(
    ta: &mut [u8],
    inc: &mut [u64],
    w: &mut i32,
    rng: &mut Rng,
    target: u8,
    p: f64,
    out_j: bool,
    active_j: bool,
    val: &[u64],
    words: usize,
    lit_b: &[u8],
    inv_b: &[u8],
    keep_b: &[u8],
    active_b: &[u8],
    n_literals: usize,
    boost: bool,
    max_inc: usize,
    half: u8,
    max_state: u8,
) {
    if !active_j {
        return;
    }
    let positive = *w >= 0;
    let give_type_i = if target == 1 { positive } else { !positive };

    if rng.next_f64() <= p {
        if give_type_i {
            let under_limit = max_inc == usize::MAX || {
                let n: u32 = (0..words).map(|k| (inc[k] & val[k]).count_ones()).sum();
                (n as usize) < max_inc
            };
            let fired_under = out_j && under_limit;
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
            rebuild_include(ta, inc, val, words, n_literals, half);
        } else if out_j {
            type_ii_update_bytes(ta, n_literals, lit_b, active_b, half, max_state);
            rebuild_include(ta, inc, val, words, n_literals, half);
        }
    }

    if out_j && rng.next_f64() <= p {
        *w = if target == 1 {
            w.saturating_add(1)
        } else {
            w.saturating_sub(1)
        };
    }
}

impl TMCoalescedAutoEncoder {
    /// Create with default settings: 8 state bits, boost enabled, seed 42.
    pub fn new(n_features: usize, n_clauses: usize, threshold: i32, s: f64) -> Self {
        Self::with_config(n_features, n_clauses, threshold, s, 8, true, 42)
    }

    /// Create with full configuration.
    ///
    /// * `n_clauses`   — size of the single shared clause bank.
    /// * `state_bits`  — TA counter precision in bits (2–8).
    /// * `boost_true_positive` — Type Ia always includes present literals when `true`.
    /// * `seed`        — master RNG seed; fully deterministic for a given seed.
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
        assert!(n_clauses >= 1);
        assert!(threshold >= 1);
        assert!(s > 1.0);
        assert!((2..=8).contains(&state_bits), "state_bits must be in 2..=8");

        let state_bits = state_bits as usize;
        let n_literals = 2 * n_features;
        let words = words_for(n_literals);
        let mut rng = Rng::new(seed);

        let mut valid = vec![0u64; words];
        for l in 0..n_literals {
            valid[l / WORD_BITS] |= 1u64 << (l % WORD_BITS);
        }

        let half = 1u8 << (state_bits - 1);
        let max_state = ((1u16 << state_bits) - 1) as u8;

        let mut ta = vec![0u8; n_clauses * n_literals];
        let mut include = vec![0u64; n_clauses * words];
        for j in 0..n_clauses {
            let tb = j * n_literals;
            for l in 0..n_literals {
                ta[tb + l] = if rng.next_u64() & 1 == 0 {
                    half - 1
                } else {
                    half
                };
            }
            rebuild_include(
                &ta[tb..tb + n_literals],
                &mut include[j * words..(j + 1) * words],
                &valid,
                words,
                n_literals,
                half,
            );
        }

        // Separate RNG stream for weight init so it does not perturb the TA sequence.
        let mut weight_rng = Rng::new(seed ^ 0x5747_4854_5F49_4E49u64);
        let weights = (0..n_features * n_clauses)
            .map(|_| {
                if weight_rng.next_u64() & 1 == 0 {
                    1i32
                } else {
                    -1i32
                }
            })
            .collect();

        let rngs = (0..n_clauses)
            .map(|i| Rng::new(seed ^ (i as u64).wrapping_add(1).wrapping_mul(GOLDEN)))
            .collect();

        let output_rngs = (0..n_features)
            .map(|o| Rng::new(seed ^ (o as u64 + n_clauses as u64 + 1).wrapping_mul(GOLDEN)))
            .collect();

        let literal_rng = Rng::new(seed ^ 0x4C49_5445_5241_4C21u64);

        TMCoalescedAutoEncoder {
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
            weights,
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

    // ---- builders -----------------------------------------------------------

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

    // ---- accessors ----------------------------------------------------------

    pub fn n_features(&self) -> usize {
        self.n_features
    }

    pub fn n_clauses(&self) -> usize {
        self.n_clauses
    }

    pub fn words_per_sample(&self) -> usize {
        self.words
    }

    pub fn s(&self) -> f64 {
        self.s
    }

    /// Signed weight of shared clause `clause` for output bit `output`.
    pub fn clause_weight(&self, output: usize, clause: usize) -> i32 {
        self.weights[output * self.n_clauses + clause]
    }

    // ---- inference ----------------------------------------------------------

    fn reconstruct_lit(&self, lit: &[u64]) -> Vec<u8> {
        debug_assert_eq!(lit.len(), self.words);
        let nc = self.n_clauses;
        let words = self.words;
        let n_features = self.n_features;
        let include = self.include.as_slice();
        let valid = self.valid.as_slice();

        let mut out = vec![0u8; n_features];
        for (o, out_o) in out.iter_mut().enumerate() {
            let ws = &self.weights[o * nc..(o + 1) * nc];
            let mut sum = 0i32;
            for (j, &w) in ws.iter().enumerate() {
                if fire_predict(&include[j * words..(j + 1) * words], lit, valid, words) {
                    sum += w;
                }
            }
            *out_o = (sum > 0) as u8;
        }
        out
    }

    /// Reconstruct a binary vector from an encoded sample.
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
        if n >= PARALLEL_MIN {
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
        let nc = self.n_clauses;
        let words = self.words;
        let include = self.include.as_slice();
        let valid = self.valid.as_slice();
        for (o, out_o) in out.iter_mut().enumerate() {
            let ws = &self.weights[o * nc..(o + 1) * nc];
            let mut sum = 0i32;
            for (j, &w) in ws.iter().enumerate() {
                if fire_predict(&include[j * words..(j + 1) * words], lit, valid, words) {
                    sum += w;
                }
            }
            *out_o = sum.clamp(-self.threshold, self.threshold);
        }
    }

    /// Return the indices of all shared clauses that fire for `sample`.
    /// The `_output` parameter is accepted for API uniformity but ignored — clauses are shared.
    pub fn fired_clauses(&self, sample: &EncodedSample, _output: usize) -> Vec<usize> {
        let lit = &sample.0;
        let words = self.words;
        let include = self.include.as_slice();
        let valid = self.valid.as_slice();
        (0..self.n_clauses)
            .filter(|&j| fire_predict(&include[j * words..(j + 1) * words], lit, valid, words))
            .collect()
    }

    // ---- training helpers ---------------------------------------------------

    /// Update all shared clauses for output `o`.
    ///
    /// `clause_outputs[j]` must already be computed (under the output-`o` feature mask)
    /// before this call.  Clause dropout and Bernoulli inv/keep masks are drawn from
    /// `output_rngs[o]`.
    fn update_output_coalesced(
        &mut self,
        o: usize,
        target: u8,
        sum: i32,
        clause_outputs: &[bool],
        lit_b: &[u8],
        active_b: &[u8],
        lit: &[u64],
        lit_active: &[u64],
    ) {
        let nc = self.n_clauses;
        let words = self.words;
        let n_literals = self.n_literals;
        let boost = self.boost_true_positive;
        let max_inc = self.max_included_literals;
        let half = self.half;
        let max_state = self.max_state;
        let drop_p = self.clause_drop_p;
        let type_iii_en = self.type_iii;
        let d_val = self.d;
        let target_bool = target == 1;

        let t = self.threshold as f64;
        let v = sum as f64;
        let p = if target == 1 {
            ((t - v) / (2.0 * t)).min(1.0)
        } else {
            ((t + v) / (2.0 * t)).min(1.0)
        };

        let Self {
            ta,
            include,
            weights,
            rngs,
            output_rngs,
            valid,
            dig_inv,
            dig_keep,
            ind,
            cat,
            ..
        } = self;
        let val = valid.as_slice();
        let orng = &mut output_rngs[o];

        let clause_active: Vec<bool> = if drop_p > 0.0 {
            (0..nc).map(|_| orng.next_f64() >= drop_p).collect()
        } else {
            vec![true; nc]
        };

        let inv_mask: Vec<u64> = (0..words).map(|_| bmask_word(orng, dig_inv)).collect();
        let keep_mask: Vec<u64> = (0..words).map(|_| bmask_word(orng, dig_keep)).collect();
        let inv_b = expand_bits_to_bytes(&inv_mask, n_literals);
        let keep_b = expand_bits_to_bytes(&keep_mask, n_literals);

        let out_w = &mut weights[o * nc..(o + 1) * nc];

        #[cfg(feature = "parallel")]
        if nc >= PARALLEL_MIN {
            use rayon::prelude::*;
            if type_iii_en {
                ta.par_chunks_mut(n_literals)
                    .zip(include.par_chunks_mut(words))
                    .zip(out_w.par_iter_mut())
                    .zip(rngs.par_iter_mut())
                    .zip(ind.par_chunks_mut(n_literals))
                    .zip(cat.par_chunks_mut(words))
                    .enumerate()
                    .for_each(|(j, (((((ta_j, inc_j), w), rng), ind_j), cat_j))| {
                        apply_one_clause_coalesced(
                            ta_j, inc_j, w, rng, target, p, clause_outputs[j], clause_active[j],
                            val, words, lit_b, &inv_b, &keep_b, active_b, n_literals, boost,
                            max_inc, half, max_state,
                        );
                        if clause_active[j] {
                            if type_iii_update(
                                ta_j, ind_j, cat_j, inc_j, lit, val, lit_active, active_b, words, n_literals,
                                d_val, p, target_bool, rng, half, max_state,
                            ) {
                                rebuild_include(ta_j, inc_j, val, words, n_literals, half);
                            }
                        }
                    });
            } else {
                ta.par_chunks_mut(n_literals)
                    .zip(include.par_chunks_mut(words))
                    .zip(out_w.par_iter_mut())
                    .zip(rngs.par_iter_mut())
                    .enumerate()
                    .for_each(|(j, (((ta_j, inc_j), w), rng))| {
                        apply_one_clause_coalesced(
                            ta_j, inc_j, w, rng, target, p, clause_outputs[j], clause_active[j],
                            val, words, lit_b, &inv_b, &keep_b, active_b, n_literals, boost,
                            max_inc, half, max_state,
                        );
                    });
            }
            return;
        }

        for j in 0..nc {
            apply_one_clause_coalesced(
                &mut ta[j * n_literals..(j + 1) * n_literals],
                &mut include[j * words..(j + 1) * words],
                &mut out_w[j],
                &mut rngs[j],
                target,
                p,
                clause_outputs[j],
                clause_active[j],
                val,
                words,
                lit_b,
                &inv_b,
                &keep_b,
                active_b,
                n_literals,
                boost,
                max_inc,
                half,
                max_state,
            );
            if type_iii_en && clause_active[j] {
                if type_iii_update(
                    &mut ta[j * n_literals..(j + 1) * n_literals],
                    &mut ind[j * n_literals..(j + 1) * n_literals],
                    &mut cat[j * words..(j + 1) * words],
                    &include[j * words..(j + 1) * words],
                    lit,
                    val,
                    lit_active,
                    active_b,
                    words,
                    n_literals,
                    d_val,
                    p,
                    target_bool,
                    &mut rngs[j],
                    half,
                    max_state,
                ) {
                    rebuild_include(
                        &ta[j * n_literals..(j + 1) * n_literals],
                        &mut include[j * words..(j + 1) * words],
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

        let nc = self.n_clauses;
        let n_features = self.n_features;
        let n_literals = self.n_literals;
        let words = self.words;

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

            // Mask literal o and its negation so clauses cannot trivially memorise bit o.
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

            // Compute clause fires under this output's feature mask.
            let clause_outputs: Vec<bool> = {
                let include = self.include.as_slice();
                let valid = self.valid.as_slice();
                let litv = self.literals.as_slice();
                (0..nc)
                    .map(|j| {
                        clause_fire(
                            &include[j * words..(j + 1) * words],
                            litv,
                            valid,
                            words,
                            &lit_active,
                        )
                    })
                    .collect()
            };

            // Weighted sum (no clause dropout — matches vanilla AE convention).
            let sum = {
                let ws = &self.weights[o * nc..(o + 1) * nc];
                let mut s = 0i32;
                for (j, &fire) in clause_outputs.iter().enumerate() {
                    if fire {
                        s += ws[j];
                    }
                }
                s.clamp(-self.threshold, self.threshold)
            };

            self.update_output_coalesced(o, target, sum, &clause_outputs, &lit_b, &active_b, lit, &lit_active);

            // Restore feature mask for the next output's iteration.
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

    // ---- evaluation ---------------------------------------------------------

    /// Fraction of (sample, bit) pairs correctly reconstructed across the batch.
    pub fn reconstruction_accuracy(&self, batch: &EncodedBatch) -> f64 {
        debug_assert_eq!(batch.data.len(), batch.n * self.words);
        let n = batch.n;
        let w = self.words;
        let nf = self.n_features;
        let packed = batch.data.as_slice();

        #[cfg(feature = "parallel")]
        if n >= PARALLEL_MIN {
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

    // ---- absorbing state introspection --------------------------------------

    pub fn absorbed_include_fraction(&self) -> f64 {
        self.absorbed_fraction(self.max_state)
    }

    pub fn absorbed_exclude_fraction(&self) -> f64 {
        self.absorbed_fraction(0)
    }

    fn absorbed_fraction(&self, target_state: u8) -> f64 {
        let mut total = 0u64;
        let mut at = 0u64;
        for j in 0..self.n_clauses {
            let base = j * self.n_literals;
            for l in 0..self.n_literals {
                let k = l / WORD_BITS;
                let bit = 1u64 << (l % WORD_BITS);
                if self.valid[k] & bit != 0 {
                    total += 1;
                    if self.ta[base + l] == target_state {
                        at += 1;
                    }
                }
            }
        }
        if total == 0 {
            0.0
        } else {
            at as f64 / total as f64
        }
    }

    // ---- interpretability ---------------------------------------------------

    /// Included literals for shared clause `clause` as `(feature_index, is_negated)`.
    pub fn clause_rule(&self, clause: usize) -> Vec<(usize, bool)> {
        let mut rule = Vec::new();
        let inc = &self.include[clause * self.words..(clause + 1) * self.words];
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::Encoder;

    fn enc(n_features: usize) -> Encoder {
        Encoder::for_binary(n_features)
    }

    fn make_mirrored(n: usize, half_n: usize, seed: u64) -> Vec<Vec<u8>> {
        let mut rng = Rng::new(seed);
        (0..n)
            .map(|_| {
                let half: Vec<u8> = (0..half_n).map(|_| (rng.next_u64() & 1) as u8).collect();
                let mut f = half.clone();
                f.extend_from_slice(&half);
                f
            })
            .collect()
    }

    fn as_slices(xs: &[Vec<u8>]) -> Vec<&[u8]> {
        xs.iter().map(|v| v.as_slice()).collect()
    }

    #[test]
    fn coalesced_ae_reconstructs_above_chance() {
        // Structured mirrored-half data: bits 6-11 = bits 0-5.
        // Each output can be predicted from its mirror; convergence verifies
        // feature masking and signed-weight training work correctly.
        let xs = make_mirrored(2000, 6, 1);
        let e = enc(12);
        let batch = e.encode_batch(&as_slices(&xs));

        let mut ae = TMCoalescedAutoEncoder::with_config(12, 20, 15, 3.9, 8, true, 42);
        for _ in 0..20 {
            ae.fit_epoch(&batch);
        }
        let acc = ae.reconstruction_accuracy(&batch);
        assert!(
            acc > 0.80,
            "expected reconstruction accuracy > 0.80, got {acc:.4}"
        );
    }

    #[test]
    fn coalesced_ae_weights_become_signed() {
        let xs = make_mirrored(2000, 6, 2);
        let e = enc(12);
        let batch = e.encode_batch(&as_slices(&xs));

        let mut ae = TMCoalescedAutoEncoder::with_config(12, 20, 15, 3.9, 8, true, 42);
        for _ in 0..10 {
            ae.fit_epoch(&batch);
        }

        let mut any_neg = false;
        let mut any_pos = false;
        for o in 0..12 {
            for j in 0..ae.n_clauses() {
                let w = ae.clause_weight(o, j);
                any_neg |= w < 0;
                any_pos |= w >= 0;
            }
        }
        assert!(any_neg, "some coalesced weights should become negative");
        assert!(any_pos, "some coalesced weights should remain non-negative");
    }

    #[test]
    fn coalesced_ae_matches_vanilla_on_reconstruction_shape() {
        let xs = make_mirrored(500, 6, 3);
        let e = enc(12);
        let batch = e.encode_batch(&as_slices(&xs));

        let mut ae = TMCoalescedAutoEncoder::with_config(12, 10, 10, 3.0, 8, true, 42);
        ae.fit_epoch(&batch);

        let recon_batch = ae.reconstruct_batch(&batch);
        assert_eq!(recon_batch.len(), 500);
        for r in &recon_batch {
            assert_eq!(r.len(), 12, "reconstruction must have n_features bits");
            for &b in r {
                assert!(b == 0 || b == 1, "reconstruction bits must be 0 or 1");
            }
        }
    }

    // ---- save / load (requires the `serde` feature) --------------------------

    #[cfg(feature = "serde")]
    use crate::serial::{self, SaveLoad};

    #[cfg(feature = "serde")]
    #[test]
    fn save_load_roundtrip_reconstructs_identically() {
        let xs = make_mirrored(2000, 6, 1);
        let e = enc(12);
        let batch = e.encode_batch(&as_slices(&xs));

        let mut ae = TMCoalescedAutoEncoder::with_config(12, 20, 15, 3.9, 8, true, 42);
        for _ in 0..15 {
            ae.fit_epoch(&batch);
        }

        let mut buf = Vec::new();
        ae.write_to(&mut buf).unwrap();
        let loaded = TMCoalescedAutoEncoder::read_from(&mut buf.as_slice()).unwrap();

        let test = e.encode_batch(&as_slices(&make_mirrored(500, 6, 2)));
        assert_eq!(ae.reconstruct_batch(&test), loaded.reconstruct_batch(&test));
        assert_eq!(
            ae.reconstruction_accuracy(&test),
            loaded.reconstruction_accuracy(&test)
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn save_load_resumes_training_without_reinit() {
        let xs = make_mirrored(2000, 6, 3);
        let e = enc(12);
        let batch = e.encode_batch(&as_slices(&xs));

        let mut ae = TMCoalescedAutoEncoder::with_config(12, 20, 15, 3.9, 8, true, 42);
        for _ in 0..10 {
            ae.fit_epoch(&batch);
        }

        let mut buf = Vec::new();
        ae.write_to(&mut buf).unwrap();
        let mut loaded = TMCoalescedAutoEncoder::read_from(&mut buf.as_slice()).unwrap();

        for _ in 0..10 {
            ae.fit_epoch(&batch);
            loaded.fit_epoch(&batch);
        }

        assert_eq!(ae.ta, loaded.ta, "TA counters diverged after resume");
        assert_eq!(ae.weights, loaded.weights, "weights diverged after resume");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn save_load_via_file_path() {
        let xs = make_mirrored(500, 6, 5);
        let e = enc(12);
        let batch = e.encode_batch(&as_slices(&xs));
        let mut ae = TMCoalescedAutoEncoder::with_config(12, 10, 10, 3.0, 8, true, 42);
        ae.fit_epoch(&batch);

        let mut path = std::env::temp_dir();
        path.push("tmu_rs_coalesced_ae_roundtrip.tmrs");
        ae.save(&path).unwrap();
        let loaded = TMCoalescedAutoEncoder::load(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let test = e.encode_batch(&as_slices(&make_mirrored(200, 6, 6)));
        assert_eq!(ae.reconstruct_batch(&test), loaded.reconstruct_batch(&test));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn load_rejects_corrupt_data() {
        assert!(TMCoalescedAutoEncoder::read_from(&mut [].as_slice()).is_err());
        assert!(TMCoalescedAutoEncoder::read_from(&mut b"XXXXjunkjunk".as_slice()).is_err());
        // Right magic/version but the wrong artifact type.
        let mut buf = Vec::new();
        serial::write_header(&mut buf, serial::TAG_VANILLA).unwrap();
        assert!(TMCoalescedAutoEncoder::read_from(&mut buf.as_slice()).is_err());
    }

    #[test]
    fn type_iii_constructs_without_panic() {
        let _ae = TMCoalescedAutoEncoder::with_config(12, 10, 10, 3.0, 8, true, 42)
            .type_iii_feedback(200.0);
    }

    #[test]
    fn type_iii_d_must_be_greater_than_one() {
        let result = std::panic::catch_unwind(|| {
            TMCoalescedAutoEncoder::with_config(12, 10, 10, 3.0, 8, true, 42)
                .type_iii_feedback(0.5)
        });
        assert!(result.is_err(), "expected panic for d <= 1.0");
    }

    #[test]
    fn type_iii_trains_without_panic() {
        let xs = make_mirrored(300, 6, 7);
        let e = enc(12);
        let batch = e.encode_batch(&as_slices(&xs));
        let mut ae = TMCoalescedAutoEncoder::with_config(12, 10, 10, 3.0, 8, true, 42)
            .type_iii_feedback(200.0);
        for _ in 0..5 {
            ae.fit_epoch(&batch);
        }
    }
}
