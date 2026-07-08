//! Vanilla weighted multiclass Tsetlin Machine classifier.
//!
//! Mirrors TMU's `vanilla_classifier.py` / `TMClassifier`.

#[cfg(feature = "parallel")]
use crate::clause_bank::dense::{DENSE_TRAIN_PARALLEL_MIN, PARALLEL_MIN, use_parallel};
use crate::clause_bank::dense::{
    GOLDEN, MASK_BITS, WORD_BITS, bmask_word, clause_fire, digits_of, expand_bits_to_bytes,
    fire_predict, grow_dense_state, rebuild_include, type_i_update_bytes, type_ii_update_bytes,
    type_iii_update, words_for,
};
use crate::encoder::{EncodedBatch, EncodedSample};
use crate::rng::Rng;

/// Per-replica RNG seeding for the data-parallel training path, shared by the
/// CPU (`fit_epoch_data_parallel`) and the GPU backend so both use exactly the
/// same seeding scheme. Given a replica seed `sd`, these derive the replica's
/// four independent RNG streams (`rng`, `literal_rng`, per-clause `rngs`,
/// per-class `class_rngs`).
#[cfg(any(feature = "parallel", feature = "gpu"))]
pub(crate) mod dp_seed {
    use super::{GOLDEN, Rng};

    /// XOR constant separating the literal-dropout stream from the sample stream.
    const LITERAL_XOR: u64 = 0x4C49_5445_5241_4C21;

    #[inline]
    pub(crate) fn replica_rng(sd: u64) -> Rng {
        Rng::new(sd)
    }
    #[inline]
    pub(crate) fn literal_rng(sd: u64) -> Rng {
        Rng::new(sd ^ LITERAL_XOR)
    }
    #[inline]
    pub(crate) fn clause_rng(sd: u64, i: usize) -> Rng {
        Rng::new(sd ^ (i as u64).wrapping_add(1).wrapping_mul(GOLDEN))
    }
    #[inline]
    pub(crate) fn class_rng(sd: u64, c: usize, n_clauses: usize) -> Rng {
        Rng::new(sd ^ (c as u64 + n_clauses as u64 + 1).wrapping_mul(GOLDEN))
    }
}

/// A weighted multiclass Tsetlin Machine with u8 per-TA counters (matches TMU's 8-bit states).
///
/// Each TA counter is a `u8` in `[0, max_state]`; the include bitset is maintained
/// as a separate `Vec<u64>` for O(words) fire checks.  Optional clause-level parallelism
/// via `--features parallel`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TsetlinMachine {
    // NOTE: data fields are `pub(crate)` so the optional `gpu` backend (src/gpu/)
    // can mirror them to device buffers and write them back after GPU training.
    // The public API is unchanged; only in-crate code can see these.
    pub(crate) n_classes: usize,
    pub(crate) n_features: usize,
    pub(crate) n_literals: usize,
    pub(crate) words: usize,
    pub(crate) clauses_per_class: usize,
    pub(crate) threshold: i32,
    pub(crate) s: f64,
    pub(crate) boost_true_positive: bool,
    pub(crate) max_included_literals: usize,
    pub(crate) clause_drop_p: f64,
    /// Per-literal dropout probability during training (mirrors TMU's `literal_drop_p`).
    pub(crate) literal_drop_p: f64,
    /// Dedicated RNG for literal-active mask generation (independent of clause/class RNGs).
    pub(crate) literal_rng: Rng,
    /// Precomputed binary digits for Bernoulli(1 - literal_drop_p) mask generation.
    pub(crate) dig_lit_active: Vec<u8>,

    /// u8 TA counters (matches TMU's 8-bit states).  Clause `cj = c*CPC + j` occupies
    /// `ta[cj * n_literals .. (cj+1) * n_literals]`.
    pub(crate) ta: Vec<u8>,
    /// Include bitset.  Clause `cj` occupies `include[cj * words .. (cj+1) * words]`.
    /// Rebuilt after every clause update; kept in sync with `ta`.
    pub(crate) include: Vec<u64>,
    /// TA threshold for inclusion: `ta[l] >= half` → literal l is included.
    pub(crate) half: u8,
    /// Maximum TA counter value: `(1 << state_bits) - 1`.
    pub(crate) max_state: u8,

    /// Per-clause integer weights (>= 1), indexed `c*CPC + j`.
    pub(crate) weights: Vec<i32>,
    /// Per-clause RNG (enables lock-free parallel training).
    pub(crate) rngs: Vec<Rng>,
    /// Per-class RNG for drop/inv/keep mask generation.
    pub(crate) class_rngs: Vec<Rng>,
    /// Per-word mask of real literal bits.
    pub(crate) valid: Vec<u64>,
    pub(crate) dig_inv: Vec<u8>,
    pub(crate) dig_keep: Vec<u8>,

    pub(crate) literals: Vec<u64>,
    pub(crate) rng: Rng, // for shuffling and negative-class selection only

    /// Per-class feedback scaling factors for imbalanced datasets.
    /// `class_weights[c]` multiplies the feedback probability for class `c`.
    /// Defaults to `1.0` for all classes (no reweighting).
    pub(crate) class_weights: Vec<f64>,

    /// Indicator TA states for Type III feedback (same layout as `ta`).
    /// Zero-initialised; only meaningful when `type_iii` is `true`.
    pub(crate) ind: Vec<u8>,
    /// `clause_and_target` bitsets for Type III feedback (same layout as `include`).
    pub(crate) cat: Vec<u64>,
    /// Type III strength: indicator is incremented with probability `1 − 1/d`.
    pub(crate) d: f64,
    /// Whether Type III feedback is active during training.
    pub(crate) type_iii: bool,

    /// When set, `update_class` never takes the nested Rayon path. Used to keep
    /// per-shard replica training single-threaded inside `fit_epoch_data_parallel`
    /// (which is already parallel across shards). Not serialised — a fresh/loaded
    /// model always trains normally. Only read on `--features parallel`.
    #[cfg_attr(feature = "serde", serde(skip))]
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    train_scalar: bool,

    /// Opt-in: let `fit_epoch` use the faster (approximate) data-parallel path when
    /// the workload is large enough. Off by default (exact training). Not serialised.
    /// Also read by the GPU backend to select its data-parallel replica path.
    #[cfg_attr(feature = "serde", serde(skip))]
    #[cfg_attr(not(any(feature = "parallel", feature = "gpu")), allow(dead_code))]
    pub(crate) data_parallel: bool,
    /// Whether the "use data_parallel" hint has already been printed (once per model).
    #[cfg_attr(feature = "serde", serde(skip))]
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    hint_shown: bool,
}

#[cfg(feature = "serde")]
impl crate::serial::SaveLoad for TsetlinMachine {
    const TAG: u8 = crate::serial::TAG_VANILLA;
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
        Self::with_config(
            n_classes,
            n_features,
            clauses_per_class,
            threshold,
            s,
            8,
            true,
            42,
        )
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
            class_weights: vec![1.0f64; n_classes],
            ind: vec![half; n_clauses * n_literals],
            cat: vec![0u64; n_clauses * words],
            d: 200.0,
            type_iii: false,
            train_scalar: false,
            data_parallel: false,
            hint_shown: false,
        }
    }

    /// Opt into **faster, approximate** training. When enabled (and built with
    /// `--features parallel`), [`fit_epoch`](Self::fit_epoch) transparently uses a
    /// data-parallel path for large enough models — samples are sharded across
    /// threads, each thread trains a replica, and the replicas are merged by
    /// averaging. Typically ~2–3× faster on multiple cores.
    ///
    /// Trade-off: results are **no longer bit-identical** to exact training and
    /// depend on the thread count (accuracy tracks exact within noise, occasionally
    /// ±a sample — not a strict guarantee). Leave it off (the default) for exact,
    /// reproducible training. No effect without `--features parallel` or on models
    /// too small to benefit. Unsupported with Type III feedback.
    pub fn data_parallel(mut self, on: bool) -> Self {
        self.data_parallel = on;
        self
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

    /// Per-class weights to compensate for label imbalance (default: all 1.0).
    ///
    /// `weights[c]` multiplies the feedback probability for every clause belonging
    /// to class `c`.  Values > 1.0 cause clauses for that class to receive feedback
    /// more often; values < 1.0 reduce feedback frequency.
    ///
    /// A standard formula for automatic computation:
    /// ```text
    /// weight[c] = n_samples / (n_classes * count[c])
    /// ```
    /// where `count[c]` is the number of training samples labelled `c`.
    /// Probability is clamped to `[0, 1]`, so very large weights saturate at p = 1.
    pub fn class_weights(mut self, weights: Vec<f64>) -> Self {
        assert_eq!(
            weights.len(),
            self.n_classes,
            "class_weights length must equal n_classes"
        );
        assert!(
            weights.iter().all(|&w| w > 0.0),
            "all class weights must be positive"
        );
        self.class_weights = weights;
        self
    }

    /// Enable Type III feedback with strength parameter `d` (default: disabled).
    ///
    /// Type III feedback maintains a per-literal *indicator state* alongside the primary TA state.
    /// Each literal accrues indicator credit when it is causally relevant to the target class;
    /// literals whose indicator state stays below the threshold are gradually excluded from
    /// clauses, producing smaller and more interpretable rules.
    ///
    /// Mirrors Python TMU's `type_iii_feedback=True` with the `d` parameter.
    /// `d` must be `> 1.0`; typical range is 100–500.
    pub fn type_iii_feedback(mut self, d: f64) -> Self {
        assert!(d > 1.0, "d must be > 1.0");
        self.d = d;
        self.type_iii = true;
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

    /// Return the per-class feedback scaling weight for `class`.
    pub fn class_weight(&self, class: usize) -> f64 {
        self.class_weights[class]
    }

    // ---- growing -----------------------------------------------------------

    /// Grow the input space to `new_n_features`, preserving all learned automata.
    ///
    /// New features start fully excluded (TA state `half - 1`, one increment from
    /// inclusion), so predictions on inputs whose new features are all 0 are
    /// bit-identical to before the grow, while the new literals remain immediately
    /// learnable. Clause weights, RNG streams, and hyperparameters are untouched.
    ///
    /// Typical use with a grown encoder:
    /// `if enc.extend_categorical(&new_samples) > 0 { tm.grow_features(enc.n_features()); }`
    ///
    /// **Re-encode after growing**: previously produced [`EncodedSample`] /
    /// [`EncodedBatch`] values use the old word stride and are geometry-incompatible
    /// with the grown machine (this is only caught by `debug_assert!` in release
    /// builds).
    ///
    /// # Panics
    /// Panics if `new_n_features < self.n_features()` (shrinking is not supported).
    /// A call with the current feature count is a no-op.
    pub fn grow_features(&mut self, new_n_features: usize) {
        assert!(
            new_n_features >= self.n_features,
            "grow_features cannot shrink: {} -> {new_n_features}",
            self.n_features
        );
        if new_n_features == self.n_features {
            return;
        }
        let n_clauses = self.n_classes * self.clauses_per_class;
        let (n_literals, words) = grow_dense_state(
            n_clauses,
            self.n_features,
            new_n_features,
            self.half,
            &mut self.ta,
            &mut self.include,
            &mut self.ind,
            &mut self.cat,
            &mut self.valid,
        );
        self.n_features = new_n_features;
        self.n_literals = n_literals;
        self.words = words;
        self.literals = vec![0u64; words];
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
                    if j & 1 == 0 {
                        sum += w;
                    } else {
                        sum -= w;
                    }
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
                    if j & 1 == 0 {
                        sum += w;
                    } else {
                        sum -= w;
                    }
                }
            }
            *out_c = sum.clamp(-self.threshold, self.threshold);
        }
    }

    /// Return the indices of all clauses (local to `class`) that fire for `sample`.
    pub fn fired_clauses(&self, sample: &EncodedSample, class: usize) -> Vec<usize> {
        let lit = &sample.0;
        let cps = self.clauses_per_class;
        let words = self.words;
        let include = self.include.as_slice();
        let valid = self.valid.as_slice();
        (0..cps)
            .filter(|&j| {
                fire_predict(
                    &include[(class * cps + j) * words..(class * cps + j + 1) * words],
                    lit,
                    valid,
                    words,
                )
            })
            .collect()
    }

    /// Predict classes for all samples in an encoded batch, returning one class index per sample.
    pub fn predict_batch(&self, batch: &EncodedBatch) -> Vec<usize> {
        debug_assert_eq!(batch.data.len(), batch.n * self.words);
        let packed = batch.data.as_slice();
        let n = batch.n;
        let w = self.words;
        #[cfg(feature = "parallel")]
        if n >= PARALLEL_MIN && use_parallel(self.clauses_per_class, w) {
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
            if clause_fire(
                &include[cj * words..(cj + 1) * words],
                lit,
                valid,
                words,
                lit_active,
            ) {
                let w = self.weights[c * cps + j];
                if j & 1 == 0 {
                    sum += w;
                } else {
                    sum -= w;
                }
            }
        }
        sum.clamp(-self.threshold, self.threshold)
    }

    /// Apply Type I / II feedback (and optionally Type III) to all clauses of class `c`.
    ///
    /// `target` is 1 for the true class and 0 for the sampled negative class.
    /// `sum` is the pre-computed clamped clause sum from `class_sum_train`.
    /// `lit_b` / `active_b` are byte-expanded per-sample arrays (precomputed in `fit_one_lit`).
    /// When `--features parallel` is active and `clauses_per_class >=
    /// DENSE_TRAIN_PARALLEL_MIN`, the per-clause loop runs in parallel via rayon.
    /// The threshold is high because exact clause-parallel training is
    /// memory-bandwidth bound and only wins for very large models; for a real
    /// multicore speedup at any size use `.data_parallel(true)` (data-parallel).
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
        let cw = self.class_weights[c];
        // Capture Type III config before the struct destructure.
        let type_iii_en = self.type_iii;
        let d_val = self.d;
        let target_bool = target != 0;
        #[cfg(feature = "parallel")]
        let force_scalar = self.train_scalar;

        let Self {
            ta,
            include,
            weights,
            rngs,
            class_rngs,
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
        let crng = &mut class_rngs[c];

        let t = wmax as f64;
        let v = sum as f64;
        let p = if target == 1 {
            ((t - v) / (2.0 * t) * cw).min(1.0)
        } else {
            ((t + v) / (2.0 * t) * cw).min(1.0)
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
        let class_ind = &mut ind[c * cps * n_literals..(c + 1) * cps * n_literals];
        let class_cat = &mut cat[c * cps * words..(c + 1) * cps * words];

        #[cfg(feature = "parallel")]
        if !force_scalar && cps >= DENSE_TRAIN_PARALLEL_MIN {
            use rayon::prelude::*;
            if type_iii_en {
                class_ta
                    .par_chunks_mut(n_literals)
                    .zip(class_inc.par_chunks_mut(words))
                    .zip(class_w.par_iter_mut())
                    .zip(class_rng.par_iter_mut())
                    .zip(class_ind.par_chunks_mut(n_literals))
                    .zip(class_cat.par_chunks_mut(words))
                    .enumerate()
                    .for_each(|(j, (((((ta_c, inc_c), w), rng), ind_c), cat_c))| {
                        apply_one_clause(
                            j, ta_c, inc_c, w, rng, target, p, &drop_mask, lit, val, lit_active,
                            words, lit_b, &inv_b, &keep_b, active_b, n_literals, boost, wmax,
                            max_inc, half, max_state,
                        );
                        if drop_mask.is_empty() || !drop_mask[j] {
                            if type_iii_update(
                                ta_c,
                                ind_c,
                                cat_c,
                                inc_c,
                                lit,
                                val,
                                lit_active,
                                active_b,
                                words,
                                n_literals,
                                d_val,
                                p,
                                target_bool,
                                rng,
                                half,
                                max_state,
                            ) {
                                rebuild_include(ta_c, inc_c, val, words, n_literals, half);
                            }
                        }
                    });
            } else {
                class_ta
                    .par_chunks_mut(n_literals)
                    .zip(class_inc.par_chunks_mut(words))
                    .zip(class_w.par_iter_mut())
                    .zip(class_rng.par_iter_mut())
                    .enumerate()
                    .for_each(|(j, (((ta_c, inc_c), w), rng))| {
                        apply_one_clause(
                            j, ta_c, inc_c, w, rng, target, p, &drop_mask, lit, val, lit_active,
                            words, lit_b, &inv_b, &keep_b, active_b, n_literals, boost, wmax,
                            max_inc, half, max_state,
                        );
                    });
            }
            return;
        }

        for j in 0..cps {
            apply_one_clause(
                j,
                &mut class_ta[j * n_literals..(j + 1) * n_literals],
                &mut class_inc[j * words..(j + 1) * words],
                &mut class_w[j],
                &mut class_rng[j],
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
                    &mut class_ta[j * n_literals..(j + 1) * n_literals],
                    &mut class_ind[j * n_literals..(j + 1) * n_literals],
                    &mut class_cat[j * words..(j + 1) * words],
                    &class_inc[j * words..(j + 1) * words],
                    lit,
                    val,
                    lit_active,
                    active_b,
                    words,
                    n_literals,
                    d_val,
                    p,
                    target_bool,
                    &mut class_rng[j],
                    half,
                    max_state,
                ) {
                    rebuild_include(
                        &class_ta[j * n_literals..(j + 1) * n_literals],
                        &mut class_inc[j * words..(j + 1) * words],
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

        let sum_y = self.class_sum_train(y, &lit_active);
        let sum_neg = self.class_sum_train(neg, &lit_active);
        self.update_class(y, 1, sum_y, &lit_active, &lit_b, &active_b);
        self.update_class(neg, 0, sum_neg, &lit_active, &lit_b, &active_b);
    }

    /// Train on a single encoded sample with true label `y`.
    pub fn fit_one(&mut self, sample: &EncodedSample, y: usize) {
        self.fit_one_lit(&sample.0, y);
    }

    /// Run one training epoch over an encoded batch, shuffling the order each epoch.
    ///
    /// **Exact by default** — bit-identical to sequential training and deterministic.
    /// If [`data_parallel(true)`](Self::data_parallel) was set *and* the build has
    /// `--features parallel` *and* the model/batch are large enough to benefit, this
    /// transparently switches to the faster **data-parallel** path (approximate; see
    /// [`data_parallel`](Self::data_parallel)). You always call `fit_epoch` either way.
    ///
    /// When training a large model on the exact path without the flag, a one-time
    /// hint is printed to stderr suggesting `data_parallel(true)`.
    pub fn fit_epoch(&mut self, batch: &EncodedBatch, ys: &[usize]) {
        debug_assert_eq!(batch.words, self.words);
        let n = batch.n;
        assert_eq!(n, ys.len());

        // Opt-in fast path: data-parallel, but only when it actually helps.
        #[cfg(feature = "parallel")]
        if self.data_parallel
            && rayon::current_num_threads() > 1
            && n >= PARALLEL_MIN
            && use_parallel(self.n_classes * self.clauses_per_class, self.words)
        {
            self.fit_epoch_data_parallel(batch, ys);
            return;
        }
        #[cfg(feature = "parallel")]
        self.maybe_hint_data_parallel(n);

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

    /// Reproduce one epoch's host-RNG-driven decisions for the GPU backend,
    /// advancing `self.rng` and `self.literal_rng` **exactly** as [`fit_epoch`]
    /// would — so a GPU-trained model's serialized RNG state matches CPU training
    /// bit-for-bit. Returns the shuffled sample order, the per-step negative
    /// class, and (if literal dropout is on) the per-step literal-active masks.
    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_epoch_plan(&mut self, n: usize, ys: &[usize]) -> crate::gpu::GpuEpochPlan {
        // Fisher–Yates shuffle with `self.rng` (identical to fit_epoch).
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let k = self.rng.below(i + 1);
            order.swap(i, k);
        }
        let words = self.words;
        let dropout = self.literal_drop_p > 0.0;
        let mut negs = Vec::with_capacity(n);
        let mut lit_active = if dropout {
            Vec::with_capacity(n * words)
        } else {
            Vec::new()
        };
        for &i in &order {
            let y = ys[i];
            // Negative-class rejection sampling (identical to fit_one_lit).
            let mut neg = self.rng.below(self.n_classes);
            while neg == y {
                neg = self.rng.below(self.n_classes);
            }
            negs.push(neg);
            if dropout {
                let rng = &mut self.literal_rng;
                let dig = &self.dig_lit_active;
                for _ in 0..words {
                    lit_active.push(bmask_word(rng, dig));
                }
            }
        }
        crate::gpu::GpuEpochPlan {
            order,
            negs,
            lit_active,
        }
    }

    /// Plan a **data-parallel** epoch for the GPU: shuffle with `self.rng`, draw
    /// `r` replica seeds (advancing `self.rng` exactly as CPU `fit_epoch_data_parallel`
    /// does), and, per replica, precompute the per-shard-step negative class and
    /// literal-dropout masks from that replica's own RNG streams. Mirrors the CPU
    /// replica semantics so the GPU fast path is the same algorithm.
    #[cfg(feature = "gpu")]
    pub(crate) fn dp_epoch_plan(&mut self, n: usize, ys: &[usize], r: usize) -> crate::gpu::DpPlan {
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let k = self.rng.below(i + 1);
            order.swap(i, k);
        }
        let seeds: Vec<u64> = (0..r).map(|_| self.rng.next_u64()).collect();
        let shard_len = n.div_ceil(r);
        let words = self.words;
        let dropout = self.literal_drop_p > 0.0;
        let mut negs = vec![0usize; r * shard_len];
        let mut lit_active = if dropout {
            vec![0u64; r * shard_len * words]
        } else {
            Vec::new()
        };
        for ri in 0..r {
            let mut rrng = dp_seed::replica_rng(seeds[ri]);
            let mut lrng = dp_seed::literal_rng(seeds[ri]);
            let start = ri * shard_len;
            let end = ((ri + 1) * shard_len).min(n);
            for s in 0..shard_len {
                let gk = start + s;
                if gk >= end {
                    break; // this replica's shard is exhausted
                }
                let y = ys[order[gk]];
                let mut neg = rrng.below(self.n_classes);
                while neg == y {
                    neg = rrng.below(self.n_classes);
                }
                negs[ri * shard_len + s] = neg;
                if dropout {
                    let dig = &self.dig_lit_active;
                    let base = (ri * shard_len + s) * words;
                    for w in 0..words {
                        lit_active[base + w] = bmask_word(&mut lrng, dig);
                    }
                }
            }
        }
        crate::gpu::DpPlan {
            order,
            seeds,
            shard_len,
            negs,
            lit_active,
        }
    }

    /// One-time stderr hint that a large model would train faster with
    /// `data_parallel(true)`. Shown only when ALL hold: the flag is off, the model
    /// is genuinely large, there is more than one core (so it would actually help),
    /// and it hasn't been shown yet for this model.
    #[cfg(feature = "parallel")]
    fn maybe_hint_data_parallel(&mut self, n: usize) {
        // Suppressed when the flag is already set, already shown, or single-core.
        if self.hint_shown || self.data_parallel || rayon::current_num_threads() <= 1 {
            return;
        }
        // Only nag when the workload is genuinely large enough that data-parallel
        // clearly pays (enough samples to amortise the per-epoch clone/merge, and a
        // model big enough — ~256+ clauses at moderate width — to scale on cores).
        let n_clauses = self.n_classes * self.clauses_per_class;
        if n >= 512 && n_clauses.saturating_mul(self.words) >= 8192 {
            eprintln!(
                "tmu-rs: training a large model ({n_clauses} clauses × {} features) on the \
                 exact path — call .data_parallel(true) for ~2-3× faster (approximate, \
                 data-parallel) training.",
                self.n_features
            );
            self.hint_shown = true;
        }
    }

    /// Data-parallel epoch: shard the samples across Rayon threads, train a private
    /// replica per shard, merge by averaging per-TA counters and per-clause weights
    /// (then rebuild the include bitsets). Approximate and thread-count dependent;
    /// invoked only from [`fit_epoch`] when [`data_parallel`](Self::data_parallel) is
    /// set and the workload is large enough to benefit. Panics under Type III.
    #[cfg(feature = "parallel")]
    fn fit_epoch_data_parallel(&mut self, batch: &EncodedBatch, ys: &[usize]) {
        {
            use rayon::prelude::*;
            assert!(
                !self.type_iii,
                "data_parallel does not support Type III feedback; unset it or use exact fit_epoch"
            );
            debug_assert_eq!(batch.words, self.words);
            let n = batch.n;
            assert_eq!(n, ys.len());
            if n == 0 {
                return;
            }

            // Deterministic shuffle — same RNG stream position as fit_epoch.
            let mut order: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let k = self.rng.below(i + 1);
                order.swap(i, k);
            }

            let w = self.words;
            let data = batch.data.as_slice();
            let n_shards = rayon::current_num_threads().clamp(1, n);
            if n_shards == 1 {
                for &i in &order {
                    self.fit_one_lit(&data[i * w..(i + 1) * w], ys[i]);
                }
                return;
            }

            // Distinct per-replica RNG seeds, drawn deterministically from the master.
            let seeds: Vec<u64> = (0..n_shards).map(|_| self.rng.next_u64()).collect();
            let n_clauses = self.n_classes * self.clauses_per_class;
            let n_cls = self.n_classes;

            // Replicas: clone state, force scalar (no nested Rayon), reseed streams.
            let mut replicas: Vec<TsetlinMachine> = seeds
                .iter()
                .map(|&sd| {
                    let mut r = self.clone();
                    r.train_scalar = true;
                    r.rng = dp_seed::replica_rng(sd);
                    r.literal_rng = dp_seed::literal_rng(sd);
                    r.rngs = (0..n_clauses).map(|i| dp_seed::clause_rng(sd, i)).collect();
                    r.class_rngs = (0..n_cls)
                        .map(|c| dp_seed::class_rng(sd, c, n_clauses))
                        .collect();
                    r
                })
                .collect();

            // Each replica trains sequentially on a contiguous shard of the order.
            let shard_len = n.div_ceil(n_shards);
            replicas
                .par_iter_mut()
                .enumerate()
                .for_each(|(s, replica)| {
                    let start = s * shard_len;
                    if start >= n {
                        return;
                    }
                    let end = (start + shard_len).min(n);
                    for &i in &order[start..end] {
                        replica.fit_one_lit(&data[i * w..(i + 1) * w], ys[i]);
                    }
                });

            // Merge: average TA counters and weights across replicas, then rebuild
            // the include bitsets so they stay consistent with the merged TA.
            let kf = replicas.len() as u32;
            self.ta.par_iter_mut().enumerate().for_each(|(i, t)| {
                let sum: u32 = replicas.iter().map(|r| r.ta[i] as u32).sum();
                *t = ((sum + kf / 2) / kf) as u8;
            });
            let wmax = self.threshold;
            self.weights.par_iter_mut().enumerate().for_each(|(i, wv)| {
                let sum: i64 = replicas.iter().map(|r| r.weights[i] as i64).sum();
                let avg = ((sum + kf as i64 / 2) / kf as i64) as i32;
                *wv = avg.clamp(1, wmax);
            });

            let words = self.words;
            let n_literals = self.n_literals;
            let half = self.half;
            let valid = self.valid.as_slice();
            self.include
                .par_chunks_mut(words)
                .zip(self.ta.par_chunks(n_literals))
                .for_each(|(inc_c, ta_c)| {
                    rebuild_include(ta_c, inc_c, valid, words, n_literals, half);
                });
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
        if n >= PARALLEL_MIN && use_parallel(self.clauses_per_class, w) {
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
        if total == 0 {
            0.0
        } else {
            at_max as f64 / total as f64
        }
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
        if total == 0 {
            0.0
        } else {
            at_min as f64 / total as f64
        }
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
        for _ in 0..1000 {
            v = v.saturating_add(1).min(max_state);
        }
        assert_eq!(v, max_state);
        for _ in 0..1000 {
            v = v.saturating_sub(1);
        }
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
            assert_eq!(
                by_predict, by_lit,
                "predict and predict_lit disagree for {x:?}"
            );
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
        let mut tm =
            TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 42).clause_drop_p(0.9999);
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
        let mut tm =
            TsetlinMachine::with_config(2, 12, 10, 15, 3.9, 8, true, 42).clause_drop_p(0.0);
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
            &mut ta,
            &mut inc,
            &mut weight,
            &lit,
            &valid,
            words,
            n_literals,
            false,
            &inv_mask,
            &keep_mask,
            10,
            max_included,
            &all_active,
            half,
            max_state,
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

        let mut tm_tight =
            TsetlinMachine::with_config(2, 12, 10, 15, 3.9, 8, true, 42).max_included_literals(2);
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
            let expected = (2 * nf).div_ceil(64);
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
        let words = (2 * nf).div_ceil(64);
        let mut lit = vec![0u64; words];
        crate::clause_bank::dense::pack(&x, nf, &mut lit);
        let bits = lit[0];
        assert_eq!(bits & 1, 1, "x[0]=1: positive bit should be set");
        assert_eq!((bits >> 1) & 1, 0, "x[1]=0: positive bit should be clear");
        assert_eq!((bits >> 2) & 1, 1, "x[2]=1: positive bit should be set");
        assert_eq!((bits >> 3) & 1, 0, "x[3]=0: positive bit should be clear");
        assert_eq!((bits >> nf) & 1, 0, "x[0]=1: negated bit should be clear");
        assert_eq!(
            (bits >> (nf + 1)) & 1,
            1,
            "x[1]=0: negated bit should be set"
        );
        assert_eq!(
            (bits >> (nf + 2)) & 1,
            0,
            "x[2]=1: negated bit should be clear"
        );
        assert_eq!(
            (bits >> (nf + 3)) & 1,
            1,
            "x[3]=0: negated bit should be set"
        );
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
        let mut tm =
            TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 42).literal_drop_p(0.9999);
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
        let mut tm =
            TsetlinMachine::with_config(2, 12, 10, 15, 3.9, 8, true, 42).literal_drop_p(0.0);
        for _ in 0..10 {
            tm.fit_epoch(&btr, &ytr);
        }
        let acc = tm.accuracy(&bte, &yte);
        assert!(
            acc > 0.90,
            "literal_drop_p=0 should still converge, got {acc}"
        );
    }

    // ---- absorbing states -------------------------------------------------------

    #[test]
    fn absorbing_include_at_max_resists_ib() {
        // A literal at the maximum TA state must survive 1 000 rounds of Type Ib
        // "exclude absent" feedback completely unchanged.
        let n_literals = 1usize;
        let words = 1usize;
        let half = 8u8; // sb=4 → half=8
        let max_state = 15u8; // (1<<4)-1
        // Literal 0 at max state (included); absent from x → violation → Ib path.
        let mut ta = vec![max_state];
        let mut inc = vec![1u64]; // bit 0 included
        let lit = vec![0u64]; // absent
        let valid = vec![1u64];
        let inv_mask = vec![!0u64];
        let keep_mask = vec![!0u64];
        let all_active = vec![!0u64];
        let mut weight = 1i32;

        for _ in 0..1_000 {
            clause_type_i_bytes(
                &mut ta,
                &mut inc,
                &mut weight,
                &lit,
                &valid,
                words,
                n_literals,
                false,
                &inv_mask,
                &keep_mask,
                100,
                usize::MAX,
                &all_active,
                half,
                max_state,
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
            &mut ta,
            &mut inc,
            &mut weight,
            &lit,
            &valid,
            words,
            n_literals,
            false,
            &inv_mask,
            &keep_mask,
            100,
            usize::MAX,
            &all_active,
            half,
            max_state,
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
        let lit = vec![0u64]; // literal 0 absent from x
        let valid = vec![1u64];
        let all_active = vec![!0u64];
        let mut weight = 5i32;

        for _ in 0..1_000 {
            clause_type_ii_bytes(
                &mut ta,
                &mut inc,
                &mut weight,
                &lit,
                &valid,
                words,
                n_literals,
                &all_active,
                half,
                max_state,
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
        let mut inc = vec![0b11u64]; // both included
        let lit = vec![0u64]; // both absent → violations on both → Ib path
        let valid = vec![0b11u64];
        let inv_mask = vec![!0u64];
        let keep_mask = vec![!0u64];
        let all_active = vec![!0u64];
        let mut weight = 1i32;

        for _ in 0..500 {
            clause_type_i_bytes(
                &mut ta,
                &mut inc,
                &mut weight,
                &lit,
                &valid,
                words,
                n_literals,
                false,
                &inv_mask,
                &keep_mask,
                5,
                1,
                &all_active,
                half,
                max_state,
            );
        }

        assert_eq!(inc[0] & 1, 1, "absorbing literal 0 must stay included");
        assert_eq!(
            (inc[0] >> 1) & 1,
            0,
            "non-absorbing literal 1 must be expelled"
        );
    }

    // ---- state_bits boundary configs ----------------------------------------

    #[test]
    fn state_bits_2_tm_trains_without_panic() {
        // state_bits=2 is the minimum allowed; max_state=3, half=2.
        // This test verifies no panics, no overflows, and a valid accuracy value.
        let (xtr, ytr) = make_xor(500, 0.0, 60);
        let (xte, yte) = make_xor(200, 0.0, 61);
        let e = enc(12);
        let btr = e.encode_batch(&as_slices(&xtr));
        let bte = e.encode_batch(&as_slices(&xte));
        let mut tm = TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 2, true, 42);
        for _ in 0..5 {
            tm.fit_epoch(&btr, &ytr);
        }
        let acc = tm.accuracy(&bte, &yte);
        assert!(
            (0.0..=1.0).contains(&acc),
            "state_bits=2 accuracy out of range: {acc}"
        );
    }

    #[test]
    fn state_bits_8_max_state_is_255() {
        // state_bits=8 is the maximum; max_state must be 255 (computed via u16
        // intermediate to avoid `1u8 << 8` overflow), half must be 128.
        let tm = TsetlinMachine::with_config(2, 4, 2, 5, 2.0, 8, true, 1);
        assert_eq!(tm.max_state, 255u8, "state_bits=8 → max_state must be 255");
        assert_eq!(tm.half, 128u8, "state_bits=8 → half must be 128");
    }

    // ---- absorbed state fractions -------------------------------------------

    #[test]
    fn absorbed_fractions_start_at_zero() {
        // Fresh TM: every counter is initialised to half-1 or half, never 0 or
        // max_state, so both absorbing fractions must be exactly 0.
        let tm = TsetlinMachine::with_config(2, 12, 10, 15, 3.9, 8, true, 42);
        assert_eq!(
            tm.absorbed_include_fraction(),
            0.0,
            "fresh TM must have no include-absorbed states"
        );
        assert_eq!(
            tm.absorbed_exclude_fraction(),
            0.0,
            "fresh TM must have no exclude-absorbed states"
        );
    }

    #[test]
    fn absorbed_fractions_increase_with_training() {
        // After sufficient training the combined absorbing fraction should be
        // strictly larger than at initialisation (which is 0.0).
        // Use state_bits=4 (max_state=15, half=8) so absorbing states are reachable
        // within a modest number of epochs — state_bits=8 would require 127 increments.
        let (xtr, ytr) = make_xor(3000, 0.0, 62);
        let e = enc(12);
        let btr = e.encode_batch(&as_slices(&xtr));
        let mut tm = TsetlinMachine::with_config(2, 12, 10, 15, 3.9, 4, true, 42);
        let before = tm.absorbed_include_fraction() + tm.absorbed_exclude_fraction();
        for _ in 0..50 {
            tm.fit_epoch(&btr, &ytr);
        }
        let after = tm.absorbed_include_fraction() + tm.absorbed_exclude_fraction();
        assert!(
            after > before,
            "absorbing states must grow during training: before={before:.4}, after={after:.4}"
        );
    }

    // ---- class_weights -------------------------------------------------------

    #[test]
    fn class_weights_unit_same_as_default() {
        // class_weights([1.0, 1.0]) must produce results identical to no weighting.
        let (xtr, ytr) = make_xor(500, 0.0, 70);
        let (xte, yte) = make_xor(200, 0.0, 71);
        let e = enc(12);
        let btr = e.encode_batch(&as_slices(&xtr));
        let bte = e.encode_batch(&as_slices(&xte));

        let mut tm_default = TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 42);
        let mut tm_unit = TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 42)
            .class_weights(vec![1.0, 1.0]);
        for _ in 0..5 {
            tm_default.fit_epoch(&btr, &ytr);
            tm_unit.fit_epoch(&btr, &ytr);
        }
        assert_eq!(
            tm_default.accuracy(&bte, &yte),
            tm_unit.accuracy(&bte, &yte),
            "unit class weights must produce the same result as no weighting"
        );
    }

    #[test]
    fn class_weights_accessor_roundtrip() {
        let tm = TsetlinMachine::with_config(3, 4, 4, 10, 3.0, 8, true, 1)
            .class_weights(vec![1.0, 2.0, 0.5]);
        assert!((tm.class_weight(0) - 1.0).abs() < 1e-12);
        assert!((tm.class_weight(1) - 2.0).abs() < 1e-12);
        assert!((tm.class_weight(2) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn class_weights_high_weight_increases_minority_recall() {
        // Build an imbalanced 2-class dataset: class 0 has 9x more samples.
        // With matching inverse-frequency weights the minority class (1) recall
        // should improve relative to uniform-weight training.
        let mut rng = Rng::new(80);
        let n = 2000usize;
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for _ in 0..n {
            let f: Vec<u8> = (0..8).map(|_| (rng.next_u64() & 1) as u8).collect();
            // Class 1 only when both f[0] and f[1] are 1 (~25% natural freq).
            // Then downsample class 1 to ~10% by rejection.
            let natural_y = (f[0] & f[1]) as usize;
            let y = if natural_y == 1 && rng.next_f64() > 0.4 {
                0
            } else {
                natural_y
            };
            xs.push(f);
            ys.push(y);
        }
        let class1_count = ys.iter().filter(|&&y| y == 1).count();
        let class0_count = n - class1_count;

        let e = enc(8);
        let btr = e.encode_batch(&as_slices(&xs));

        // Inverse-frequency weights: weight[c] = n / (n_classes * count[c]).
        let w0 = n as f64 / (2.0 * class0_count as f64);
        let w1 = n as f64 / (2.0 * class1_count as f64);

        let mut tm_balanced =
            TsetlinMachine::with_config(2, 8, 20, 20, 3.0, 8, true, 42).class_weights(vec![w0, w1]);
        let mut tm_uniform = TsetlinMachine::with_config(2, 8, 20, 20, 3.0, 8, true, 42);
        for _ in 0..15 {
            tm_balanced.fit_epoch(&btr, &ys);
            tm_uniform.fit_epoch(&btr, &ys);
        }

        // Measure per-class recall on training set.
        let recall_c1 = |tm: &TsetlinMachine| -> f64 {
            let w = tm.words_per_sample();
            let data = btr.data.as_slice();
            let (correct, total) =
                (0..n)
                    .filter(|&i| ys[i] == 1)
                    .fold((0usize, 0usize), |(c, t), i| {
                        let pred = tm.predict_lit(&data[i * w..(i + 1) * w]);
                        (c + (pred == 1) as usize, t + 1)
                    });
            if total == 0 {
                0.0
            } else {
                correct as f64 / total as f64
            }
        };

        let recall_balanced = recall_c1(&tm_balanced);
        let recall_uniform = recall_c1(&tm_uniform);
        assert!(
            recall_balanced >= recall_uniform,
            "inverse-frequency weights should improve minority-class recall: \
             balanced={recall_balanced:.3} uniform={recall_uniform:.3}"
        );
    }

    // ---- class_weights validation --------------------------------------------

    #[test]
    #[should_panic(expected = "class_weights length must equal n_classes")]
    fn class_weights_wrong_length_panics() {
        TsetlinMachine::with_config(2, 4, 4, 10, 3.0, 8, true, 1)
            .class_weights(vec![1.0, 1.0, 1.0]); // 3 weights for 2 classes
    }

    #[test]
    #[should_panic(expected = "all class weights must be positive")]
    fn class_weights_zero_panics() {
        TsetlinMachine::with_config(2, 4, 4, 10, 3.0, 8, true, 1).class_weights(vec![1.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "all class weights must be positive")]
    fn class_weights_negative_panics() {
        TsetlinMachine::with_config(2, 4, 4, 10, 3.0, 8, true, 1).class_weights(vec![-1.0, 1.0]);
    }

    // ---- class_weights saturation / safety -----------------------------------

    #[test]
    fn class_weights_very_high_clamped_safe() {
        // Weights >> 1 saturate p at 1.0 (clamp); must not panic and clause
        // weights must remain in [1, threshold].
        let (xtr, ytr) = make_xor(300, 0.0, 90);
        let btr = enc(12).encode_batch(&as_slices(&xtr));
        let threshold = 10i32;
        let mut tm = TsetlinMachine::with_config(2, 12, 8, threshold, 3.0, 8, true, 42)
            .class_weights(vec![1000.0, 1000.0]);
        for _ in 0..5 {
            tm.fit_epoch(&btr, &ytr);
        }
        for c in 0..2 {
            for j in 0..tm.clauses_per_class() {
                let w = tm.clause_weight(c, j);
                assert!(
                    (1..=threshold).contains(&w),
                    "clause ({c},{j}) weight {w} out of [1, {threshold}] with very high class weight"
                );
            }
        }
    }

    // ---- class_weights suppression -------------------------------------------

    #[test]
    fn class_weights_near_zero_suppresses_clause_updates() {
        // A weight of 1e-9 multiplies p by ~0; over 1000 training steps the
        // probability of any single clause receiving Type Ia feedback is ~1e-6.
        // All clause weights for the suppressed class must remain at the initial
        // value of 1; class 1 (weight 1.0) must evolve normally.
        let (xtr, ytr) = make_xor(200, 0.0, 91);
        let btr = enc(12).encode_batch(&as_slices(&xtr));
        let cps = 12usize;
        let mut tm = TsetlinMachine::with_config(2, 12, cps, 10, 3.0, 8, true, 42)
            .class_weights(vec![1e-9, 1.0]);
        for _ in 0..5 {
            tm.fit_epoch(&btr, &ytr);
        }
        let all_frozen = (0..cps).all(|j| tm.clause_weight(0, j) == 1);
        assert!(
            all_frozen,
            "near-zero class weight should keep all clause weights at initial value 1"
        );
        let any_evolved = (0..cps).any(|j| tm.clause_weight(1, j) > 1);
        assert!(
            any_evolved,
            "class 1 with weight 1.0 should evolve normally"
        );
    }

    // ---- class_weights determinism -------------------------------------------

    #[test]
    fn class_weights_same_seed_same_result() {
        // Same seed + same asymmetric weights must produce bit-identical outcomes.
        let (xtr, ytr) = make_xor(500, 0.0, 92);
        let (xte, yte) = make_xor(200, 0.0, 93);
        let e = enc(12);
        let btr = e.encode_batch(&as_slices(&xtr));
        let bte = e.encode_batch(&as_slices(&xte));

        let weights = vec![2.0, 0.5];
        let mut tm1 = TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 99)
            .class_weights(weights.clone());
        let mut tm2 =
            TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 99).class_weights(weights);
        for _ in 0..5 {
            tm1.fit_epoch(&btr, &ytr);
            tm2.fit_epoch(&btr, &ytr);
        }
        assert_eq!(
            tm1.accuracy(&bte, &yte),
            tm2.accuracy(&bte, &yte),
            "class_weights must not break determinism for the same seed"
        );
        for c in 0..2 {
            for j in 0..tm1.clauses_per_class() {
                assert_eq!(
                    tm1.clause_weight(c, j),
                    tm2.clause_weight(c, j),
                    "clause ({c},{j}) weight differs between identical runs"
                );
            }
        }
    }

    // ---- class_weights multiclass independence --------------------------------

    #[test]
    fn class_weights_multiclass_independent_per_class() {
        // 4-class problem: freeze classes 0 and 2 (weight ≈ 0), train 1 and 3
        // normally (weight 1.0). Frozen classes must have all clause weights at 1;
        // active classes must have at least one clause weight above 1.
        let mut rng = Rng::new(94);
        let n = 1000usize;
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for _ in 0..n {
            let f: Vec<u8> = (0..8).map(|_| (rng.next_u64() & 1) as u8).collect();
            let y = ((f[0] ^ f[1]) as usize) * 2 + (f[2] ^ f[3]) as usize;
            xs.push(f);
            ys.push(y);
        }
        let btr = enc(8).encode_batch(&as_slices(&xs));
        let cps = 10usize;

        let mut tm = TsetlinMachine::with_config(4, 8, cps, 15, 3.0, 8, true, 42)
            .class_weights(vec![1e-9, 1.0, 1e-9, 1.0]);
        for _ in 0..10 {
            tm.fit_epoch(&btr, &ys);
        }

        for &frozen in &[0usize, 2usize] {
            let all_at_one = (0..cps).all(|j| tm.clause_weight(frozen, j) == 1);
            assert!(
                all_at_one,
                "frozen class {frozen} should have all clause weights at initial 1"
            );
        }
        for &active in &[1usize, 3usize] {
            let any_evolved = (0..cps).any(|j| tm.clause_weight(active, j) > 1);
            assert!(
                any_evolved,
                "active class {active} should have some clause weights > 1 after training"
            );
        }
    }

    // ---- scores are clamped to ±threshold -----------------------------------

    #[test]
    fn scores_are_clamped_to_threshold() {
        let threshold = 10i32;
        let (xtr, ytr) = make_xor(500, 0.0, 63);
        let e = enc(12);
        let btr = e.encode_batch(&as_slices(&xtr));
        let mut tm = TsetlinMachine::with_config(2, 12, 8, threshold, 3.0, 8, true, 42);
        for _ in 0..5 {
            tm.fit_epoch(&btr, &ytr);
        }
        let (xte, _) = make_xor(50, 0.0, 64);
        for x in &xte {
            let sample = e.encode_one(x);
            let mut s = vec![0i32; 2];
            tm.scores(&sample, &mut s);
            for (c, &sc) in s.iter().enumerate() {
                assert!(
                    sc >= -threshold && sc <= threshold,
                    "class {c} score {sc} is outside [-{threshold}, {threshold}]"
                );
            }
        }
    }

    // ---- clause_rule consistent with include bitset -------------------------

    #[test]
    fn clause_rule_consistent_with_include_bitset() {
        // clause_rule() must report exactly the same set of included literals
        // as a direct popcount of the include bitset.
        let (xtr, ytr) = make_xor(500, 0.0, 65);
        let e = enc(12);
        let btr = e.encode_batch(&as_slices(&xtr));
        let mut tm = TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 42);
        tm.fit_epoch(&btr, &ytr);

        let n_literals = 2 * tm.n_features();
        let words = tm.words_per_sample();
        for class in 0..tm.n_classes() {
            for clause in 0..tm.clauses_per_class() {
                let rule = tm.clause_rule(class, clause);
                let cj = class * tm.clauses_per_class() + clause;
                let inc = &tm.include[cj * words..(cj + 1) * words];
                let bitset_count: usize = (0..n_literals)
                    .filter(|&l| (inc[l / 64] >> (l % 64)) & 1 != 0)
                    .count();
                assert_eq!(
                    rule.len(),
                    bitset_count,
                    "clause ({class},{clause}): rule has {} literals but bitset has {}",
                    rule.len(),
                    bitset_count
                );
            }
        }
    }

    // ---- save / load (requires the `serde` feature) --------------------------

    #[cfg(feature = "serde")]
    use crate::serial::{self, SaveLoad};

    #[cfg(feature = "serde")]
    #[test]
    fn save_load_roundtrip_predicts_identically() {
        let (xtr, ytr) = make_xor(2000, 0.1, 1);
        let (xte, yte) = make_xor(1000, 0.0, 2);
        let e = enc(12);
        let btr = e.encode_batch(&as_slices(&xtr));
        let bte = e.encode_batch(&as_slices(&xte));

        let mut tm = TsetlinMachine::with_config(2, 12, 8, 15, 3.9, 8, true, 7)
            .clause_drop_p(0.1)
            .literal_drop_p(0.1)
            .class_weights(vec![1.0, 2.0]);
        for _ in 0..10 {
            tm.fit_epoch(&btr, &ytr);
        }

        // Round-trip through an in-memory buffer.
        let mut buf = Vec::new();
        tm.write_to(&mut buf).unwrap();
        let loaded = TsetlinMachine::read_from(&mut buf.as_slice()).unwrap();

        // Predictions must be bit-identical.
        assert_eq!(tm.predict_batch(&bte), loaded.predict_batch(&bte));
        assert_eq!(tm.accuracy(&bte, &yte), loaded.accuracy(&bte, &yte));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn save_load_resumes_training_without_reinit() {
        let (xtr, ytr) = make_xor(2000, 0.1, 3);
        let (xte, yte) = make_xor(1000, 0.0, 4);
        let e = enc(12);
        let btr = e.encode_batch(&as_slices(&xtr));
        let bte = e.encode_batch(&as_slices(&xte));

        let mut tm = TsetlinMachine::with_config(2, 12, 8, 15, 3.9, 8, true, 7);
        for _ in 0..8 {
            tm.fit_epoch(&btr, &ytr);
        }

        // Save, reload, and keep training both copies in lockstep.
        let mut buf = Vec::new();
        tm.write_to(&mut buf).unwrap();
        let mut loaded = TsetlinMachine::read_from(&mut buf.as_slice()).unwrap();

        for _ in 0..8 {
            tm.fit_epoch(&btr, &ytr);
            loaded.fit_epoch(&btr, &ytr);
        }

        // Resumed training must follow the exact same RNG streams.
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
        let mut tm = TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 42);
        tm.fit_epoch(&btr, &ytr);

        let mut path = std::env::temp_dir();
        path.push("tmu_rs_vanilla_roundtrip.tmrs");
        tm.save(&path).unwrap();
        let loaded = TsetlinMachine::load(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let (xte, _) = make_xor(200, 0.0, 6);
        let bte = e.encode_batch(&as_slices(&xte));
        assert_eq!(tm.predict_batch(&bte), loaded.predict_batch(&bte));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn load_rejects_corrupt_data() {
        // Empty / truncated input.
        assert!(TsetlinMachine::read_from(&mut [].as_slice()).is_err());
        // Bad magic.
        assert!(TsetlinMachine::read_from(&mut b"XXXXjunkjunk".as_slice()).is_err());
        // Right magic, wrong type tag (coalesced tag on a vanilla read).
        let mut buf = Vec::new();
        serial::write_header(&mut buf, serial::TAG_COALESCED).unwrap();
        assert!(TsetlinMachine::read_from(&mut buf.as_slice()).is_err());
    }

    // --- Type III feedback integration tests ---

    #[test]
    fn type_iii_constructs_without_panic() {
        let _tm =
            TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 42).type_iii_feedback(200.0);
    }

    #[test]
    fn type_iii_d_must_be_greater_than_one() {
        let result = std::panic::catch_unwind(|| {
            TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 42).type_iii_feedback(0.5)
        });
        assert!(result.is_err(), "expected panic for d <= 1.0");
    }

    #[test]
    fn type_iii_trains_without_panic() {
        let (xtr, ytr) = make_xor(200, 0.0, 7);
        let e = enc(12);
        let btr = e.encode_batch(&as_slices(&xtr));
        let mut tm =
            TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 42).type_iii_feedback(200.0);
        for _ in 0..5 {
            tm.fit_epoch(&btr, &ytr);
        }
    }

    #[test]
    fn type_iii_learns_xor() {
        let (xtr, ytr) = make_xor(1000, 0.0, 9);
        let (xte, yte) = make_xor(500, 0.0, 10);
        let e = enc(12);
        let btr = e.encode_batch(&as_slices(&xtr));
        let bte = e.encode_batch(&as_slices(&xte));
        let mut tm =
            TsetlinMachine::with_config(2, 12, 20, 10, 3.0, 8, true, 42).type_iii_feedback(200.0);
        for _ in 0..20 {
            tm.fit_epoch(&btr, &ytr);
        }
        let acc = tm.accuracy(&bte, &yte);
        assert!(acc > 0.70, "Type III XOR accuracy too low: {acc:.3}");
    }

    #[test]
    fn type_iii_ind_and_cat_same_size_as_ta_and_include() {
        let tm =
            TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 42).type_iii_feedback(100.0);
        assert_eq!(tm.ind.len(), tm.ta.len());
        assert_eq!(tm.cat.len(), tm.include.len());
    }

    // ---- work-aware parallel gating -------------------------------------------

    #[test]
    fn use_parallel_wide_few_clauses_inference() {
        // 32 clauses/class is below the count threshold (128), but 256 features
        // → words=8, so cps*words=256 crosses the work floor and the work-aware
        // INFERENCE branch engages under `--features parallel`. Results are
        // path-independent, so predict_batch must equal the per-sample path and
        // the model must still learn — under a parallel build this exercises the
        // branch the old count-only gate would have skipped.
        let nf = 256;
        let mut rng = Rng::new(5);
        let mut gen_batch = |n: usize| -> (Vec<Vec<u8>>, Vec<usize>) {
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            for _ in 0..n {
                let f: Vec<u8> = (0..nf).map(|_| (rng.next_u64() & 1) as u8).collect();
                ys.push((f[0] ^ f[1]) as usize);
                xs.push(f);
            }
            (xs, ys)
        };
        let (xtr, ytr) = gen_batch(2000);
        let (xte, yte) = gen_batch(500);
        let e = enc(nf);
        let btr = e.encode_batch(&as_slices(&xtr));
        let bte = e.encode_batch(&as_slices(&xte));
        let mut tm = TsetlinMachine::with_config(2, nf, 32, 15, 3.9, 8, true, 7);
        for _ in 0..20 {
            tm.fit_epoch(&btr, &ytr);
        }
        let w = tm.words_per_sample();
        let batch = tm.predict_batch(&bte);
        for (i, &p) in batch.iter().enumerate() {
            assert_eq!(p, tm.predict_lit(&bte.data[i * w..(i + 1) * w]));
        }
        assert!(
            tm.accuracy(&bte, &yte) > 0.9,
            "wide/few-clause model failed to learn"
        );
    }

    #[test]
    fn data_parallel_learns_xor() {
        // With data_parallel(true), fit_epoch uses the approximate data-parallel
        // path when large enough; it must still learn noisy XOR to high accuracy.
        let (xtr, ytr) = make_xor(6000, 0.1, 1);
        let (xte, yte) = make_xor(2000, 0.0, 2);
        let e = enc(12);
        let btr = e.encode_batch(&as_slices(&xtr));
        let bte = e.encode_batch(&as_slices(&xte));
        let mut tm =
            TsetlinMachine::with_config(2, 12, 40, 15, 3.9, 8, true, 7).data_parallel(true);
        for _ in 0..30 {
            tm.fit_epoch(&btr, &ytr);
        }
        let acc = tm.accuracy(&bte, &yte);
        assert!(acc > 0.95, "data_parallel XOR accuracy too low: {acc}");
    }

    // ---- growing the feature space --------------------------------------------

    /// Zero-pad every sample from its current length up to `n`.
    fn pad_to(xs: &[Vec<u8>], n: usize) -> Vec<Vec<u8>> {
        xs.iter()
            .map(|x| {
                let mut p = x.clone();
                p.resize(n, 0);
                p
            })
            .collect()
    }

    #[test]
    fn grow_preserves_predictions() {
        let (xtr, ytr) = make_xor(3000, 0.1, 5);
        let (xte, _) = make_xor(500, 0.0, 6);
        let e12 = enc(12);
        let btr = e12.encode_batch(&as_slices(&xtr));

        let mut tm = TsetlinMachine::with_config(2, 12, 8, 15, 3.9, 8, true, 7);
        for _ in 0..20 {
            tm.fit_epoch(&btr, &ytr);
        }

        let bte12 = e12.encode_batch(&as_slices(&xte));
        let preds_before = tm.predict_batch(&bte12);
        let scores_before: Vec<[i32; 2]> = xte
            .iter()
            .map(|x| {
                let mut out = [0i32; 2];
                tm.scores(&e12.encode_one(x), &mut out);
                out
            })
            .collect();

        tm.grow_features(20);
        assert_eq!(tm.n_features(), 20);
        assert_eq!(tm.words_per_sample(), words_for(40));

        // Same rows, zero-padded to the new width: predictions and per-class
        // scores must be bit-identical (new literals are all excluded).
        let e20 = enc(20);
        let xte20 = pad_to(&xte, 20);
        let bte20 = e20.encode_batch(&as_slices(&xte20));
        assert_eq!(tm.predict_batch(&bte20), preds_before);
        for (x, before) in xte20.iter().zip(&scores_before) {
            let mut out = [0i32; 2];
            tm.scores(&e20.encode_one(x), &mut out);
            assert_eq!(&out, before, "scores changed after grow for {x:?}");
        }
    }

    #[test]
    fn grow_noop_when_equal() {
        let (xtr, ytr) = make_xor(500, 0.0, 8);
        let e = enc(12);
        let btr = e.encode_batch(&as_slices(&xtr));
        let mut tm = TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 42);
        tm.fit_epoch(&btr, &ytr);

        let (ta, include) = (tm.ta.clone(), tm.include.clone());
        tm.grow_features(12);
        assert_eq!(tm.n_features(), 12);
        assert_eq!(tm.ta, ta);
        assert_eq!(tm.include, include);
    }

    #[test]
    #[should_panic(expected = "cannot shrink")]
    fn grow_panics_on_shrink() {
        let mut tm = TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 42);
        tm.grow_features(11);
    }

    #[test]
    fn grow_preserves_clause_rules() {
        let (xtr, ytr) = make_xor(2000, 0.1, 9);
        let e = enc(12);
        let btr = e.encode_batch(&as_slices(&xtr));
        let mut tm = TsetlinMachine::with_config(2, 12, 8, 15, 3.9, 8, true, 7);
        for _ in 0..10 {
            tm.fit_epoch(&btr, &ytr);
        }

        let rules_before: Vec<Vec<(usize, bool)>> = (0..2)
            .flat_map(|c| (0..tm.clauses_per_class()).map(move |j| (c, j)))
            .map(|(c, j)| tm.clause_rule(c, j))
            .collect();

        tm.grow_features(30);

        // Feature indices and negation flags are stable across the grow, so the
        // interpretability mapping must be unchanged.
        let rules_after: Vec<Vec<(usize, bool)>> = (0..2)
            .flat_map(|c| (0..tm.clauses_per_class()).map(move |j| (c, j)))
            .map(|(c, j)| tm.clause_rule(c, j))
            .collect();
        assert_eq!(rules_before, rules_after);
    }

    #[test]
    fn grow_then_learns_new_feature() {
        // Pre-train on XOR over the first 12 features.
        let (xtr, ytr) = make_xor(2000, 0.1, 10);
        let e12 = enc(12);
        let mut tm = TsetlinMachine::with_config(2, 12, 8, 15, 3.9, 8, true, 7);
        let btr = e12.encode_batch(&as_slices(&xtr));
        for _ in 0..10 {
            tm.fit_epoch(&btr, &ytr);
        }

        tm.grow_features(16);

        // New task determined solely by an appended feature: label = bit 14.
        let mut rng = Rng::new(99);
        let make = |rng: &mut Rng, n: usize| -> (Vec<Vec<u8>>, Vec<usize>) {
            let mut xs = Vec::with_capacity(n);
            let mut ys = Vec::with_capacity(n);
            for _ in 0..n {
                let f: Vec<u8> = (0..16).map(|_| (rng.next_u64() & 1) as u8).collect();
                ys.push(f[14] as usize);
                xs.push(f);
            }
            (xs, ys)
        };
        let (xtr2, ytr2) = make(&mut rng, 3000);
        let (xte2, yte2) = make(&mut rng, 1000);

        let e16 = enc(16);
        let btr2 = e16.encode_batch(&as_slices(&xtr2));
        let bte2 = e16.encode_batch(&as_slices(&xte2));
        for _ in 0..15 {
            tm.fit_epoch(&btr2, &ytr2);
        }
        let acc = tm.accuracy(&bte2, &yte2);
        assert!(acc >= 0.95, "grown TM failed to learn new feature: {acc}");
    }

    #[test]
    fn grow_with_type_iii_continues_training() {
        let (xtr, ytr) = make_xor(1000, 0.1, 11);
        let e12 = enc(12);
        let btr = e12.encode_batch(&as_slices(&xtr));
        let mut tm =
            TsetlinMachine::with_config(2, 12, 8, 15, 3.9, 8, true, 7).type_iii_feedback(200.0);
        for _ in 0..5 {
            tm.fit_epoch(&btr, &ytr);
        }

        tm.grow_features(18);
        assert_eq!(tm.ind.len(), tm.ta.len());
        assert_eq!(tm.cat.len(), tm.include.len());

        let e18 = enc(18);
        let xtr18 = pad_to(&xtr, 18);
        let btr18 = e18.encode_batch(&as_slices(&xtr18));
        for _ in 0..5 {
            tm.fit_epoch(&btr18, &ytr);
        }

        // clause_rule must stay consistent with the include bitset after growing.
        for class in 0..2 {
            for clause in 0..tm.clauses_per_class() {
                let rule = tm.clause_rule(class, clause);
                let cj = class * tm.clauses_per_class() + clause;
                let inc = &tm.include[cj * tm.words..(cj + 1) * tm.words];
                let bitset_count: u32 = inc.iter().map(|w| w.count_ones()).sum();
                assert_eq!(rule.len(), bitset_count as usize);
            }
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn grown_model_serde_roundtrip() {
        let (xtr, ytr) = make_xor(2000, 0.1, 12);
        let (xte, _) = make_xor(500, 0.0, 13);
        let e12 = enc(12);
        let btr = e12.encode_batch(&as_slices(&xtr));
        let mut tm = TsetlinMachine::with_config(2, 12, 8, 15, 3.9, 8, true, 7);
        for _ in 0..8 {
            tm.fit_epoch(&btr, &ytr);
        }

        tm.grow_features(20);
        let e20 = enc(20);
        let xtr20 = pad_to(&xtr, 20);
        let btr20 = e20.encode_batch(&as_slices(&xtr20));
        for _ in 0..4 {
            tm.fit_epoch(&btr20, &ytr);
        }

        let mut buf = Vec::new();
        tm.write_to(&mut buf).unwrap();
        let loaded = TsetlinMachine::read_from(&mut buf.as_slice()).unwrap();

        let bte20 = e20.encode_batch(&as_slices(&pad_to(&xte, 20)));
        assert_eq!(tm.predict_batch(&bte20), loaded.predict_batch(&bte20));
    }
}
