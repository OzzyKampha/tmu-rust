//! Coalesced weighted multiclass Tsetlin Machine classifier.
//!
//! Mirrors TMU's `coalesced_classifier.py` / `TMCoalescedClassifier`.
//!
//! ## How it differs from [`crate::TsetlinMachine`]
//!
//! The vanilla classifier gives every class its **own** pool of `clauses_per_class`
//! clauses and encodes polarity by clause-index parity (even = +, odd = −) with a
//! single positive integer weight per clause.
//!
//! The **coalesced** machine instead keeps a **single shared pool of `n_clauses`
//! clauses** and a **signed per-class weight matrix** `weights[class][clause]`
//! (`n_classes × n_clauses`).  Polarity is the *sign* of the weight, weights may be
//! negative, and every class scores against every shared clause.  Clauses are thus
//! "coalesced": one learned conjunction can serve several classes at once, so far
//! fewer total clauses are needed than `n_classes × clauses_per_class`.
//!
//! All bit-level primitives are reused verbatim from [`crate::clause_bank::dense`]
//! (the clause/literal bit layout is identical to the vanilla machine).

#[cfg(feature = "parallel")]
use crate::clause_bank::dense::PARALLEL_MIN;
use crate::clause_bank::dense::{
    bmask_word, clause_fire, digits_of, expand_bits_to_bytes, fire_predict, rebuild_include,
    type_i_update_bytes, type_ii_update_bytes, type_iii_update, words_for, GOLDEN, MASK_BITS,
    WORD_BITS,
};
use crate::encoder::{EncodedBatch, EncodedSample};
use crate::rng::Rng;

/// A coalesced weighted multiclass Tsetlin Machine with u8 per-TA counters.
///
/// One shared clause bank of `n_clauses` clauses is voted on by every class via a
/// signed weight matrix.  Optional clause-level parallelism via `--features parallel`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CoalescedTsetlinMachine {
    n_classes: usize,
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
    /// `true` to select the negative class proportionally to its update probability
    /// (focused negative sampling); `false` for uniform random selection.
    focused_negative_sampling: bool,

    /// Dedicated RNG for the per-sample literal-active mask.
    literal_rng: Rng,
    /// Precomputed digits for Bernoulli(1 - literal_drop_p) mask generation.
    dig_lit_active: Vec<u8>,

    /// u8 TA counters for the shared bank.  Clause `j` occupies
    /// `ta[j * n_literals .. (j+1) * n_literals]`.
    ta: Vec<u8>,
    /// Include bitset for the shared bank.  Clause `j` occupies
    /// `include[j * words .. (j+1) * words]`.  Kept in sync with `ta`.
    include: Vec<u64>,
    /// Inclusion threshold: `ta[l] >= half` → literal l is included.
    half: u8,
    /// Maximum TA counter value: `(1 << state_bits) - 1`.
    max_state: u8,

    /// Signed per-class clause weights, indexed `class * n_clauses + clause`.
    /// Initialised to random `±1`; may grow positive or negative without bound
    /// (matches TMU, which does not clamp coalesced weights to the threshold).
    weights: Vec<i32>,
    /// Per-clause RNG (one per shared clause); reused across both class passes so
    /// training stays deterministic and the per-clause loop is lock-free under rayon.
    rngs: Vec<Rng>,
    /// Per-class RNG for the inv/keep Bernoulli feedback masks.
    class_rngs: Vec<Rng>,
    /// Per-word mask of real literal bits.
    valid: Vec<u64>,
    dig_inv: Vec<u8>,
    dig_keep: Vec<u8>,

    ind: Vec<u8>,
    cat: Vec<u64>,
    d: f64,
    type_iii: bool,

    literals: Vec<u64>,
    /// Global RNG for shuffling, clause dropout, and negative-class selection.
    rng: Rng,
}

#[cfg(feature = "serde")]
impl crate::serial::SaveLoad for CoalescedTsetlinMachine {
    const TAG: u8 = crate::serial::TAG_COALESCED;
}

/// Per-clause feedback + weight-update kernel shared by the sequential and parallel
/// training paths.  Operates on one shared clause `j` from the perspective of one
/// class `c` (whose signed weight for this clause is `w`).
///
/// `target` is 1 for the true class, 0 for the negative class.  Feedback type is
/// chosen by weight sign and reversed for the negative class:
/// * true class:  Type I if `*w >= 0`, else Type II;
/// * neg class:   Type I if `*w < 0`,  else Type II.
///
/// Two independent Bernoulli(`p`) draws are taken (mirroring TMU's separate
/// `type_*_feedback` and `increment`/`decrement` passes): one gates the literal
/// feedback, the other gates the weight step.
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
    // Packed arrays:
    val: &[u64],
    words: usize,
    // Byte-expanded arrays:
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

    // Literal (TA) feedback.
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
            // Type II is only meaningful when the clause fired.
            type_ii_update_bytes(ta, n_literals, lit_b, active_b, half, max_state);
            rebuild_include(ta, inc, val, words, n_literals, half);
        }
    }

    // Weight step: increment (true class) / decrement (negative class) every fired,
    // active clause regardless of sign — matches TMU's positive_weights / negative_weights.
    if out_j && rng.next_f64() <= p {
        *w = if target == 1 {
            w.saturating_add(1)
        } else {
            w.saturating_sub(1)
        };
    }
}

impl CoalescedTsetlinMachine {
    /// Create a CoalescedTsetlinMachine with default settings: 8 state bits, boost
    /// enabled, uniform negative sampling, seed 42.
    pub fn new(
        n_classes: usize,
        n_features: usize,
        n_clauses: usize,
        threshold: i32,
        s: f64,
    ) -> Self {
        Self::with_config(n_classes, n_features, n_clauses, threshold, s, 8, true, 42)
    }

    /// Create a CoalescedTsetlinMachine with full configuration.
    ///
    /// * `n_clauses` — size of the single shared clause bank (shared across all classes).
    /// * `state_bits` — TA counter precision in bits (2–8).
    /// * `boost_true_positive` — Type Ia always includes present literals when `true`.
    /// * `seed` — master RNG seed; fully deterministic for a given seed.
    #[allow(clippy::too_many_arguments)]
    pub fn with_config(
        n_classes: usize,
        n_features: usize,
        n_clauses: usize,
        threshold: i32,
        s: f64,
        state_bits: u8,
        boost_true_positive: bool,
        seed: u64,
    ) -> Self {
        assert!(n_classes >= 2);
        assert!(n_features >= 1);
        assert!(n_clauses >= 2);
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

        // Signed ±1 weight init from a dedicated stream so it doesn't perturb the
        // TA-init / feedback RNG sequences.
        let mut weight_rng = Rng::new(seed ^ 0x5747_4854_5F49_4E49u64); // "WGHT_INI"
        let weights = (0..n_classes * n_clauses)
            .map(|_| {
                if weight_rng.next_u64() & 1 == 0 {
                    1
                } else {
                    -1
                }
            })
            .collect();

        let rngs = (0..n_clauses)
            .map(|i| Rng::new(seed ^ (i as u64).wrapping_add(1).wrapping_mul(GOLDEN)))
            .collect();

        let class_rngs = (0..n_classes)
            .map(|c| Rng::new(seed ^ (c as u64 + n_clauses as u64 + 1).wrapping_mul(GOLDEN)))
            .collect();

        let literal_rng = Rng::new(seed ^ 0x4C49_5445_5241_4C21u64);

        CoalescedTsetlinMachine {
            n_classes,
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
            focused_negative_sampling: false,
            literal_rng,
            dig_lit_active: digits_of(1.0, MASK_BITS),
            ta,
            include,
            half,
            max_state,
            weights,
            rngs,
            class_rngs,
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

    // ---- builders --------------------------------------------------------

    /// Limit how many literals each clause may include (Type Ia guard).
    /// Mirrors TMU's `max_included_literals` (default: no limit).
    pub fn max_included_literals(mut self, max: usize) -> Self {
        self.max_included_literals = max;
        self
    }

    /// Per-clause dropout probability during training (default: 0.0 = no drop).
    pub fn clause_drop_p(mut self, p: f64) -> Self {
        assert!((0.0..1.0).contains(&p), "clause_drop_p must be in [0, 1)");
        self.clause_drop_p = p;
        self
    }

    /// Per-literal dropout probability during training (default: 0.0 = no drop).
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

    /// Enable focused negative sampling (default: `false` = uniform random).
    ///
    /// When enabled the negative class for each sample is drawn with probability
    /// proportional to its current update probability `clamp((T + class_sum)/2T, 0, 1)`,
    /// focusing learning on the most confusable wrong classes.  Mirrors TMU's
    /// `focused_negative_sampling`.
    pub fn focused_negative_sampling(mut self, enabled: bool) -> Self {
        self.focused_negative_sampling = enabled;
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
    /// Return the number of clauses in the shared clause bank.
    pub fn n_clauses(&self) -> usize {
        self.n_clauses
    }
    /// Return the number of 64-bit words used to represent one packed sample.
    pub fn words_per_sample(&self) -> usize {
        self.words
    }
    /// Return the specificity parameter `s`.
    pub fn s(&self) -> f64 {
        self.s
    }
    /// Return the signed weight of shared clause `clause` for class `class`.
    pub fn clause_weight(&self, class: usize, clause: usize) -> i32 {
        self.weights[class * self.n_clauses + clause]
    }
    /// Return `true` if shared clause `clause` votes *for* `class` (weight `>= 0`).
    pub fn clause_is_positive(&self, class: usize, clause: usize) -> bool {
        self.weights[class * self.n_clauses + clause] >= 0
    }

    // ---- inference -------------------------------------------------------

    /// Compute the firing state of every shared clause for `lit` (inference semantics).
    #[inline]
    fn clause_outputs_predict(&self, lit: &[u64]) -> Vec<bool> {
        let words = self.words;
        let include = self.include.as_slice();
        let valid = self.valid.as_slice();
        (0..self.n_clauses)
            .map(|j| fire_predict(&include[j * words..(j + 1) * words], lit, valid, words))
            .collect()
    }

    /// Internal: predict from a raw literal slice without allocation beyond clause outputs.
    #[inline]
    fn predict_lit(&self, lit: &[u64]) -> usize {
        debug_assert_eq!(lit.len(), self.words);
        let outputs = self.clause_outputs_predict(lit);
        let nc = self.n_clauses;
        let mut best = 0usize;
        let mut best_score = i32::MIN;
        for c in 0..self.n_classes {
            let cw = &self.weights[c * nc..(c + 1) * nc];
            let mut sum = 0i32;
            for (j, &fired) in outputs.iter().enumerate() {
                if fired {
                    sum += cw[j];
                }
            }
            if sum > best_score {
                best_score = sum;
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

    /// Fill `out` with the clamped weighted clause sums for each class.
    pub fn scores(&self, sample: &EncodedSample, out: &mut [i32]) {
        debug_assert_eq!(out.len(), self.n_classes);
        let outputs = self.clause_outputs_predict(&sample.0);
        let nc = self.n_clauses;
        for (c, out_c) in out.iter_mut().enumerate() {
            let cw = &self.weights[c * nc..(c + 1) * nc];
            let mut sum = 0i32;
            for (j, &fired) in outputs.iter().enumerate() {
                if fired {
                    sum += cw[j];
                }
            }
            *out_c = sum.clamp(-self.threshold, self.threshold);
        }
    }

    /// Predict classes for all samples in an encoded batch.
    pub fn predict_batch(&self, batch: &EncodedBatch) -> Vec<usize> {
        debug_assert_eq!(batch.data.len(), batch.n * self.words);
        let packed = batch.data.as_slice();
        let n = batch.n;
        let w = self.words;
        #[cfg(feature = "parallel")]
        if n >= PARALLEL_MIN && self.n_clauses >= PARALLEL_MIN {
            use rayon::prelude::*;
            return (0..n)
                .into_par_iter()
                .map(|i| self.predict_lit(&packed[i * w..(i + 1) * w]))
                .collect();
        }
        (0..n)
            .map(|i| self.predict_lit(&packed[i * w..(i + 1) * w]))
            .collect()
    }

    // ---- training helpers ------------------------------------------------

    /// Clamped weighted clause sum for class `c` from cached shared-clause outputs.
    fn class_sum(&self, c: usize, clause_outputs: &[bool], clause_active: &[bool]) -> i32 {
        let nc = self.n_clauses;
        let cw = &self.weights[c * nc..(c + 1) * nc];
        let mut sum = 0i32;
        for j in 0..nc {
            if clause_active[j] && clause_outputs[j] {
                sum += cw[j];
            }
        }
        sum.clamp(-self.threshold, self.threshold)
    }

    /// Select the negative class (`!= y`) for a training sample.
    fn choose_negative(
        &mut self,
        y: usize,
        clause_outputs: &[bool],
        clause_active: &[bool],
    ) -> usize {
        if self.focused_negative_sampling {
            let t = self.threshold as f64;
            let mut probs = vec![0f64; self.n_classes];
            let mut total = 0f64;
            for (c, prob) in probs.iter_mut().enumerate() {
                if c == y {
                    continue;
                }
                let s = self.class_sum(c, clause_outputs, clause_active) as f64;
                *prob = ((t + s) / (2.0 * t)).clamp(0.0, 1.0);
                total += *prob;
            }
            if total > 0.0 {
                let mut r = self.rng.next_f64() * total;
                for (c, &prob) in probs.iter().enumerate() {
                    if c == y {
                        continue;
                    }
                    r -= prob;
                    if r <= 0.0 {
                        return c;
                    }
                }
            }
        }
        let mut neg = self.rng.below(self.n_classes);
        while neg == y {
            neg = self.rng.below(self.n_classes);
        }
        neg
    }

    /// Apply one class's feedback pass over the shared clause bank.
    ///
    /// `target` is 1 for the true class, 0 for the negative class.  `clause_outputs`
    /// and `clause_active` are the per-sample caches computed once in `fit_one_lit`.
    fn update_class(
        &mut self,
        c: usize,
        target: u8,
        clause_outputs: &[bool],
        clause_active: &[bool],
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
        let type_iii_en = self.type_iii;
        let d_val = self.d;
        let target_bool = target != 0;

        let sum = self.class_sum(c, clause_outputs, clause_active);
        let t = self.threshold as f64;
        let v = sum as f64;
        let p = if target == 1 {
            (t - v) / (2.0 * t)
        } else {
            (t + v) / (2.0 * t)
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
            ind,
            cat,
            ..
        } = self;
        let val = valid.as_slice();
        let crng = &mut class_rngs[c];

        // Bernoulli(1/s) and ((s-1)/s) masks, generated once per pass, byte-expanded.
        let inv_mask: Vec<u64> = (0..words).map(|_| bmask_word(crng, dig_inv)).collect();
        let keep_mask: Vec<u64> = (0..words).map(|_| bmask_word(crng, dig_keep)).collect();
        let inv_b = expand_bits_to_bytes(&inv_mask, n_literals);
        let keep_b = expand_bits_to_bytes(&keep_mask, n_literals);

        let class_w = &mut weights[c * nc..(c + 1) * nc];

        #[cfg(feature = "parallel")]
        if nc >= PARALLEL_MIN {
            use rayon::prelude::*;
            if type_iii_en {
                ta.par_chunks_mut(n_literals)
                    .zip(include.par_chunks_mut(words))
                    .zip(class_w.par_iter_mut())
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
                    .zip(class_w.par_iter_mut())
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
                &mut class_w[j],
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

    /// Internal: train from a raw literal slice without allocation.
    fn fit_one_lit(&mut self, lit: &[u64], y: usize) {
        debug_assert_eq!(lit.len(), self.words);
        debug_assert!(y < self.n_classes);
        self.literals.copy_from_slice(lit);

        let nc = self.n_clauses;
        let words = self.words;
        let n_literals = self.n_literals;

        // Byte-expand the sample literals once; reused for all clause updates this step.
        let lit_b = expand_bits_to_bytes(lit, n_literals);

        // Per-sample literal-active mask (literal dropout), shared by both passes.
        let lit_active: Vec<u64> = if self.literal_drop_p > 0.0 {
            let rng = &mut self.literal_rng;
            let dig = &self.dig_lit_active;
            (0..words).map(|_| bmask_word(rng, dig)).collect()
        } else {
            vec![!0u64; words]
        };
        let active_b = expand_bits_to_bytes(&lit_active, n_literals);

        // Per-sample clause-active mask (clause dropout), shared by both passes.
        let clause_active: Vec<bool> = if self.clause_drop_p > 0.0 {
            let p = self.clause_drop_p;
            (0..nc).map(|_| self.rng.next_f64() >= p).collect()
        } else {
            vec![true; nc]
        };

        // Shared-clause firing computed ONCE (training semantics: empty clauses fire),
        // then reused for both class passes — matches TMU, whose feedback uses the
        // pre-feedback clause outputs even though the first pass mutates the TA states.
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

        let neg = self.choose_negative(y, &clause_outputs, &clause_active);

        self.update_class(y, 1, &clause_outputs, &clause_active, &lit_b, &active_b, lit, &lit_active);
        self.update_class(neg, 0, &clause_outputs, &clause_active, &lit_b, &active_b, lit, &lit_active);
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

    // ---- evaluation ------------------------------------------------------

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

    // ---- absorbing state introspection -----------------------------------

    /// Fraction of (clause, literal) pairs whose TA is at the absorbing include
    /// state (counter == max_state).
    pub fn absorbed_include_fraction(&self) -> f64 {
        self.absorbed_fraction(self.max_state)
    }

    /// Fraction of (clause, literal) pairs whose TA is at the absorbing exclude
    /// state (counter == 0).
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

    // ---- interpretability ------------------------------------------------

    /// Return the included literals for shared clause `clause` as
    /// `(feature_index, is_negated)` pairs.
    pub fn clause_rule(&self, clause: usize) -> Vec<(usize, bool)> {
        let mut rule = Vec::new();
        let inc = &self.include[clause * self.words..(clause + 1) * self.words];
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::Encoder;

    fn enc(n_features: usize) -> Encoder {
        Encoder::for_binary(n_features)
    }

    /// Generate `n` XOR samples (12 random bits, label = bit0 XOR bit1) with optional noise.
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

    /// Generate a 4-class problem: y = 2*(b0^b1) + (b2^b3) over 8 bits.
    fn make_4class(n: usize, seed: u64) -> (Vec<Vec<u8>>, Vec<usize>) {
        let mut rng = Rng::new(seed);
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for _ in 0..n {
            let f: Vec<u8> = (0..8).map(|_| (rng.next_u64() & 1) as u8).collect();
            let y = ((f[0] ^ f[1]) as usize) * 2 + (f[2] ^ f[3]) as usize;
            xs.push(f);
            ys.push(y);
        }
        (xs, ys)
    }

    fn as_slices(xs: &[Vec<u8>]) -> Vec<&[u8]> {
        xs.iter().map(|v| v.as_slice()).collect()
    }

    #[test]
    fn coalesced_learns_xor() {
        let (xtr, ytr) = make_xor(5000, 0.25, 1);
        let (xte, yte) = make_xor(2000, 0.0, 2);
        let e = enc(12);
        let btr = e.encode_batch(&as_slices(&xtr));
        let bte = e.encode_batch(&as_slices(&xte));

        // A small shared bank generalises best on noisy XOR; too many low-T clauses
        // overfit the label noise.
        let mut tm = CoalescedTsetlinMachine::with_config(2, 12, 16, 15, 3.9, 8, true, 7);
        for _ in 0..40 {
            tm.fit_epoch(&btr, &ytr);
        }
        let acc = tm.accuracy(&bte, &yte);
        assert!(acc > 0.95, "expected >0.95, got {acc}");
    }

    #[test]
    fn coalesced_multiclass_4_class_learns() {
        let (xs, ys) = make_4class(3000, 50);
        let (xte, yte) = make_4class(500, 51);
        let e = enc(8);
        let btr = e.encode_batch(&as_slices(&xs));
        let bte = e.encode_batch(&as_slices(&xte));
        // Far fewer clauses than vanilla's n_classes * clauses_per_class would use.
        let mut tm = CoalescedTsetlinMachine::with_config(4, 8, 40, 30, 3.9, 8, true, 42);
        for _ in 0..40 {
            tm.fit_epoch(&btr, &ys);
        }
        let acc = tm.accuracy(&bte, &yte);
        assert!(
            acc > 0.85,
            "4-class coalesced should reach >0.85, got {acc}"
        );
    }

    #[test]
    fn same_seed_same_result() {
        let (xtr, ytr) = make_xor(500, 0.0, 14);
        let e = enc(12);
        let btr = e.encode_batch(&as_slices(&xtr));

        let mut tm1 = CoalescedTsetlinMachine::with_config(2, 12, 16, 10, 3.0, 8, true, 99);
        let mut tm2 = CoalescedTsetlinMachine::with_config(2, 12, 16, 10, 3.0, 8, true, 99);
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
        for c in 0..2 {
            for j in 0..tm1.n_clauses() {
                assert_eq!(tm1.clause_weight(c, j), tm2.clause_weight(c, j));
            }
        }
    }

    #[test]
    fn predict_batch_matches_single() {
        let (xtr, ytr) = make_xor(300, 0.0, 12);
        let e = enc(12);
        let mut tm = CoalescedTsetlinMachine::with_config(2, 12, 16, 10, 3.0, 8, true, 42);
        for _ in 0..5 {
            tm.fit_epoch(&e.encode_batch(&as_slices(&xtr)), &ytr);
        }

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

    #[test]
    fn scores_argmax_matches_predict() {
        let (xtr, ytr) = make_4class(2000, 30);
        let e = enc(8);
        let mut tm = CoalescedTsetlinMachine::with_config(4, 8, 40, 30, 3.9, 8, true, 42);
        for _ in 0..20 {
            tm.fit_epoch(&e.encode_batch(&as_slices(&xtr)), &ytr);
        }
        let (xte, _) = make_4class(200, 31);
        for x in &xte {
            let sample = e.encode_one(x);
            let pred = tm.predict(&sample);
            let mut s = vec![0i32; 4];
            tm.scores(&sample, &mut s);
            // Clamped scores must put the predicted class at (one of) the maxima.
            let max = *s.iter().max().unwrap();
            assert_eq!(
                s[pred], max,
                "predict must agree with a maximal clamped score"
            );
        }
    }

    #[test]
    fn weights_can_go_negative() {
        // Coalesced weights are signed and unbounded; after training both signs
        // must appear across the (class, clause) matrix.
        let (xtr, ytr) = make_4class(3000, 17);
        let btr = enc(8).encode_batch(&as_slices(&xtr));
        let mut tm = CoalescedTsetlinMachine::with_config(4, 8, 40, 30, 3.9, 8, true, 42);
        for _ in 0..30 {
            tm.fit_epoch(&btr, &ytr);
        }
        let mut any_neg = false;
        let mut any_pos = false;
        for c in 0..tm.n_classes() {
            for j in 0..tm.n_clauses() {
                let w = tm.clause_weight(c, j);
                any_neg |= w < 0;
                any_pos |= w >= 0;
            }
        }
        assert!(any_neg, "some coalesced weights should become negative");
        assert!(any_pos, "some coalesced weights should remain non-negative");
    }

    #[test]
    fn clause_drop_p_one_leaves_state_unchanged() {
        let (xtr, ytr) = make_xor(200, 0.0, 15);
        let btr = enc(12).encode_batch(&as_slices(&xtr));
        let mut tm = CoalescedTsetlinMachine::with_config(2, 12, 16, 10, 3.0, 8, true, 42)
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
    fn focused_vs_uniform_both_converge() {
        let (xtr, ytr) = make_4class(3000, 18);
        let (xte, yte) = make_4class(500, 28);
        let e = enc(8);
        let btr = e.encode_batch(&as_slices(&xtr));
        let bte = e.encode_batch(&as_slices(&xte));

        let mut tm_uniform = CoalescedTsetlinMachine::with_config(4, 8, 40, 30, 3.9, 8, true, 42);
        let mut tm_focused = CoalescedTsetlinMachine::with_config(4, 8, 40, 30, 3.9, 8, true, 42)
            .focused_negative_sampling(true);
        for _ in 0..40 {
            tm_uniform.fit_epoch(&btr, &ytr);
            tm_focused.fit_epoch(&btr, &ytr);
        }
        assert!(
            tm_uniform.accuracy(&bte, &yte) > 0.85,
            "uniform should converge"
        );
        assert!(
            tm_focused.accuracy(&bte, &yte) > 0.85,
            "focused should converge"
        );
    }

    #[test]
    fn clause_rule_consistent_with_include_bitset() {
        let (xtr, ytr) = make_xor(500, 0.0, 65);
        let e = enc(12);
        let btr = e.encode_batch(&as_slices(&xtr));
        let mut tm = CoalescedTsetlinMachine::with_config(2, 12, 16, 10, 3.0, 8, true, 42);
        tm.fit_epoch(&btr, &ytr);

        let n_literals = 2 * tm.n_features();
        let words = tm.words_per_sample();
        for clause in 0..tm.n_clauses() {
            let rule = tm.clause_rule(clause);
            let inc = &tm.include[clause * words..(clause + 1) * words];
            let bitset_count: usize = (0..n_literals)
                .filter(|&l| (inc[l / 64] >> (l % 64)) & 1 != 0)
                .count();
            assert_eq!(rule.len(), bitset_count);
        }
    }

    #[test]
    fn large_shared_bank_learns() {
        // n_clauses >= PARALLEL_MIN (128) so the rayon feedback branch is exercised
        // under `--features parallel` (and the scalar loop otherwise). Both must learn.
        let (xtr, ytr) = make_4class(3000, 33);
        let (xte, yte) = make_4class(500, 34);
        let e = enc(8);
        let btr = e.encode_batch(&as_slices(&xtr));
        let bte = e.encode_batch(&as_slices(&xte));
        let mut tm = CoalescedTsetlinMachine::with_config(4, 8, 160, 40, 3.9, 8, true, 42);
        for _ in 0..20 {
            tm.fit_epoch(&btr, &ytr);
        }
        let acc = tm.accuracy(&bte, &yte);
        assert!(
            acc > 0.85,
            "large shared bank should reach >0.85, got {acc}"
        );
    }

    #[test]
    fn weights_init_pm_one() {
        let tm = CoalescedTsetlinMachine::with_config(3, 4, 10, 10, 3.0, 8, true, 1);
        for c in 0..3 {
            for j in 0..10 {
                let w = tm.clause_weight(c, j);
                assert!(w == 1 || w == -1, "initial weight must be ±1, got {w}");
            }
        }
    }

    // ---- save / load (requires the `serde` feature) --------------------------

    #[cfg(feature = "serde")]
    use crate::serial::{self, SaveLoad};

    #[cfg(feature = "serde")]
    #[test]
    fn save_load_roundtrip_predicts_identically() {
        let (xtr, ytr) = make_4class(2000, 1);
        let (xte, yte) = make_4class(1000, 2);
        let e = enc(8);
        let btr = e.encode_batch(&as_slices(&xtr));
        let bte = e.encode_batch(&as_slices(&xte));

        let mut tm = CoalescedTsetlinMachine::with_config(4, 8, 24, 20, 3.9, 8, true, 7)
            .focused_negative_sampling(true);
        for _ in 0..20 {
            tm.fit_epoch(&btr, &ytr);
        }

        let mut buf = Vec::new();
        tm.write_to(&mut buf).unwrap();
        let loaded = CoalescedTsetlinMachine::read_from(&mut buf.as_slice()).unwrap();

        assert_eq!(tm.predict_batch(&bte), loaded.predict_batch(&bte));
        assert_eq!(tm.accuracy(&bte, &yte), loaded.accuracy(&bte, &yte));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn save_load_resumes_training_without_reinit() {
        let (xtr, ytr) = make_4class(2000, 3);
        let (xte, yte) = make_4class(1000, 4);
        let e = enc(8);
        let btr = e.encode_batch(&as_slices(&xtr));
        let bte = e.encode_batch(&as_slices(&xte));

        let mut tm = CoalescedTsetlinMachine::with_config(4, 8, 24, 20, 3.9, 8, true, 7);
        for _ in 0..15 {
            tm.fit_epoch(&btr, &ytr);
        }

        let mut buf = Vec::new();
        tm.write_to(&mut buf).unwrap();
        let mut loaded = CoalescedTsetlinMachine::read_from(&mut buf.as_slice()).unwrap();

        for _ in 0..15 {
            tm.fit_epoch(&btr, &ytr);
            loaded.fit_epoch(&btr, &ytr);
        }

        assert_eq!(tm.ta, loaded.ta, "TA counters diverged after resume");
        assert_eq!(tm.weights, loaded.weights, "weights diverged after resume");
        assert_eq!(tm.accuracy(&bte, &yte), loaded.accuracy(&bte, &yte));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn save_load_via_file_path() {
        let (xtr, ytr) = make_xor(500, 0.0, 5);
        let e = enc(12);
        let btr = e.encode_batch(&as_slices(&xtr));
        let mut tm = CoalescedTsetlinMachine::with_config(2, 12, 16, 15, 3.9, 8, true, 42);
        tm.fit_epoch(&btr, &ytr);

        let mut path = std::env::temp_dir();
        path.push("tmu_rs_coalesced_roundtrip.tmrs");
        tm.save(&path).unwrap();
        let loaded = CoalescedTsetlinMachine::load(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let (xte, _) = make_xor(200, 0.0, 6);
        let bte = e.encode_batch(&as_slices(&xte));
        assert_eq!(tm.predict_batch(&bte), loaded.predict_batch(&bte));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn load_rejects_corrupt_data() {
        assert!(CoalescedTsetlinMachine::read_from(&mut [].as_slice()).is_err());
        assert!(CoalescedTsetlinMachine::read_from(&mut b"XXXXjunkjunk".as_slice()).is_err());
        // Right magic, wrong type tag (vanilla tag on a coalesced read).
        let mut buf = Vec::new();
        serial::write_header(&mut buf, serial::TAG_VANILLA).unwrap();
        assert!(CoalescedTsetlinMachine::read_from(&mut buf.as_slice()).is_err());
    }

    #[test]
    fn type_iii_constructs_without_panic() {
        let _tm = CoalescedTsetlinMachine::with_config(2, 12, 16, 15, 3.9, 8, true, 42)
            .type_iii_feedback(200.0);
    }

    #[test]
    fn type_iii_d_must_be_greater_than_one() {
        let result = std::panic::catch_unwind(|| {
            CoalescedTsetlinMachine::with_config(2, 12, 16, 15, 3.9, 8, true, 42)
                .type_iii_feedback(0.5)
        });
        assert!(result.is_err(), "expected panic for d <= 1.0");
    }

    #[test]
    fn type_iii_trains_without_panic() {
        let (xtr, ytr) = make_xor(300, 0.0, 7);
        let e = enc(12);
        let btr = e.encode_batch(&as_slices(&xtr));
        let mut tm = CoalescedTsetlinMachine::with_config(2, 12, 16, 15, 3.9, 8, true, 42)
            .type_iii_feedback(200.0);
        for _ in 0..5 {
            tm.fit_epoch(&btr, &ytr);
        }
    }
}
