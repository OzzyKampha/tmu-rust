//! Sparse weighted multiclass Tsetlin Machine classifier.
//!
//! Mirrors TMU's sparse clause-bank classifier: the same weighted-clause vanilla
//! algorithm as [`TsetlinMachine`](crate::TsetlinMachine), but backed by the
//! [`SparseClauseBank`] — per-clause index lists with **absorbing actions** that
//! permanently drop literals from candidacy as training converges, when
//! enabled with [`TMSparseClassifier::absorbing`] (off by default, as TMU).  See the
//! [`sparse`](crate::clause_bank::sparse) module docs for the data layout.
//!
//! ## Differences from the dense [`TsetlinMachine`]
//!
//! * **Rayon-parallel, but no AVX2.**  With `--features parallel`, training
//!   parallelises over clauses (each clause owns disjoint state, so it's lock-free
//!   and bit-identical to the scalar result) and inference parallelises over
//!   samples — both gated by `PARALLEL_MIN` like the dense model. As with dense,
//!   parallel training only pays off at large clause counts (≈1000+); at small
//!   counts the per-clause work is too light to amortise Rayon's dispatch overhead.
//!   AVX2 is deliberately **not** used: the hot path is a variable-length
//!   excluded-list scan dominated by per-index bit gathers, scalar Bernoulli RNG
//!   draws (kept scalar to preserve determinism), and mid-loop `swap_remove` — none
//!   of which vectorise. Upstream `cair/tmu`'s sparse C bank is likewise scalar.
//! * **No Type III feedback.**  Type III maintains a second per-literal indicator
//!   array, which conflicts with the sparse bank removing literals; it is omitted.
//! * **`absorbed_exclude_fraction` means "removed".**  In the dense model it
//!   counts literals at counter `0`; here those literals have been *removed* from
//!   the pool, so this returns the fraction of (clause, literal) slots that have
//!   been absorbed out.  The numbers are therefore not directly comparable to the
//!   dense model's, though both trend upward as training converges.
//!
//! All other public API mirrors [`TsetlinMachine`] one-to-one.

#[cfg(feature = "parallel")]
use crate::clause_bank::dense::PARALLEL_MIN;
#[cfg(feature = "parallel")]
use crate::clause_bank::dense::use_parallel;
use crate::clause_bank::dense::{GOLDEN, WORD_BITS, words_for};
use crate::clause_bank::sparse::SparseClauseBank;
use crate::encoder::{EncodedBatch, EncodedSample};
use crate::rng::Rng;

/// A weighted multiclass Tsetlin Machine backed by a sparse clause bank.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TMSparseClassifier {
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
    pub(crate) literal_drop_p: f64,

    /// Sparse clause storage; `n_classes * clauses_per_class` clauses total.
    pub(crate) bank: SparseClauseBank,
    /// Per-clause integer weights (>= 1), indexed `c * clauses_per_class + j`.
    pub(crate) weights: Vec<i32>,

    /// Per-clause feedback RNG (also gates the per-clause feedback probability).
    pub(crate) rngs: Vec<Rng>,
    /// Per-class RNG for the clause-dropout mask.
    pub(crate) class_rngs: Vec<Rng>,
    /// Dedicated RNG for the per-sample literal-active (dropout) mask.
    pub(crate) literal_rng: Rng,
    /// Master RNG: epoch shuffle + negative-class sampling only.
    pub(crate) rng: Rng,

    /// Per-class feedback scaling factors for imbalanced datasets (default 1.0).
    pub(crate) class_weights: Vec<f64>,
    /// TMU's `weighted_clauses` (default true here). Training setting.
    #[cfg_attr(
        feature = "serde",
        serde(skip, default = "crate::clause_bank::dense::serde_true")
    )]
    pub(crate) weighted_clauses: bool,
    /// TMU's `confidence_driven_updating`. Training setting.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) confidence_driven_updating: bool,
    /// TMU's `type_i_ii_ratio` factors. Training settings.
    #[cfg_attr(
        feature = "serde",
        serde(skip, default = "crate::clause_bank::dense::serde_one")
    )]
    pub(crate) type_i_p: f64,
    #[cfg_attr(
        feature = "serde",
        serde(skip, default = "crate::clause_bank::dense::serde_one")
    )]
    pub(crate) type_ii_p: f64,
}

#[cfg(feature = "serde")]
impl crate::serial::SaveLoad for TMSparseClassifier {
    const TAG: u8 = crate::serial::TAG_SPARSE;
}

impl TMSparseClassifier {
    /// Create a classifier with default settings: 8 state bits, boost enabled, seed 42.
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

    /// Create a classifier with full configuration (mirrors [`TsetlinMachine::with_config`]).
    ///
    /// [`TsetlinMachine::with_config`]: crate::TsetlinMachine::with_config
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

        let bank = SparseClauseBank::new(n_clauses, n_literals, state_bits, seed);

        let rngs = (0..n_clauses)
            .map(|i| Rng::new(seed ^ (i as u64).wrapping_add(1).wrapping_mul(GOLDEN)))
            .collect();
        let class_rngs = (0..n_classes)
            .map(|c| Rng::new(seed ^ (c as u64 + n_clauses as u64 + 1).wrapping_mul(GOLDEN)))
            .collect();
        let literal_rng = Rng::new(seed ^ 0x4C49_5445_5241_4C21u64);

        TMSparseClassifier {
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
            bank,
            weights: vec![1i32; n_clauses],
            rngs,
            class_rngs,
            literal_rng,
            rng: Rng::new(seed),
            class_weights: vec![1.0f64; n_classes],
            weighted_clauses: true,
            confidence_driven_updating: false,
            type_i_p: 1.0,
            type_ii_p: 1.0,
        }
    }

    // ---- builders --------------------------------------------------------

    /// Limit how many literals each clause may include (Type Ia guard).
    /// Sparse models especially benefit from this — it bounds clause size directly.
    pub fn max_included_literals(mut self, max: usize) -> Self {
        self.max_included_literals = max;
        self
    }

    /// Absorbing exclude threshold, as TMU's `absorbing` (default -1 = off).
    ///
    /// With `k >= 0`, an excluded literal whose state falls to `k` or below is
    /// absorbed: removed from the clause's candidates for good, so training and
    /// firing stop visiting it ("sparse computation with absorbing actions").
    /// Larger `k` absorbs sooner. The include side is always absorbing: a literal
    /// at the maximum state is never decremented.
    pub fn absorbing(mut self, k: i32) -> Self {
        assert!(k >= -1, "absorbing must be >= -1 (-1 = off)");
        self.bank.absorbing = k;
        self
    }

    /// TMU's `weighted_clauses` (default **true** here; TMU's default is
    /// false). With `false` every clause keeps a vote of 1.
    pub fn weighted_clauses(mut self, on: bool) -> Self {
        self.weighted_clauses = on;
        self
    }

    /// TMU's `confidence_driven_updating` (default false): feedback probability
    /// `(T - |class sum|) / T` for the true and the negative class alike.
    pub fn confidence_driven_updating(mut self, on: bool) -> Self {
        self.confidence_driven_updating = on;
        self
    }

    /// TMU's `type_i_ii_ratio` (default 1.0): above 1 makes Type II feedback
    /// rarer by `1/ratio`, below 1 makes Type I rarer by `ratio`.
    pub fn type_i_ii_ratio(mut self, ratio: f64) -> Self {
        assert!(ratio > 0.0, "type_i_ii_ratio must be > 0");
        (self.type_i_p, self.type_ii_p) = crate::clause_bank::dense::type_i_ii_factors(ratio);
        self
    }

    /// TMU's `feature_negation` (default true). With `false` no clause may use a
    /// negated literal. Re-initialises the clauses: set it before training.
    pub fn feature_negation(mut self, on: bool) -> Self {
        self.bank.feature_negation = on;
        self.bank.reinit();
        self
    }

    /// TMU's `literal_sampling` (default 1.0): each clause starts with a random
    /// `fraction` of the literals as candidates; the rest wait unallocated (see
    /// [`literal_insertion_state`](Self::literal_insertion_state)). Fewer
    /// candidates per clause is what keeps a sparse machine fast on a large
    /// vocabulary. Re-initialises the clauses: set it before training.
    pub fn literal_sampling(mut self, fraction: f64) -> Self {
        assert!(
            fraction > 0.0 && fraction <= 1.0,
            "literal_sampling must be in (0, 1]"
        );
        self.bank.literal_sampling = fraction;
        self.bank.reinit();
        self
    }

    /// TMU's `literal_insertion_state` (default -1 = off): when a candidate is
    /// absorbed, an unallocated literal takes its place at this state.
    pub fn literal_insertion_state(mut self, state: i32) -> Self {
        assert!(
            state >= -1,
            "literal_insertion_state must be >= -1 (-1 = off)"
        );
        assert!(
            state < i32::from(self.bank.half),
            "literal_insertion_state must be below the inclusion state {}",
            self.bank.half
        );
        self.bank.insertion_state = state;
        self
    }

    /// TMU's `feedback_rate_excluded_literals` (default 1): candidate literals
    /// get feedback with probability `1/rate`, moving `rate` states at a time.
    pub fn feedback_rate_excluded_literals(mut self, rate: u8) -> Self {
        assert!(rate >= 1, "feedback_rate_excluded_literals must be >= 1");
        self.bank.feedback_rate = rate;
        self
    }

    /// Per-clause dropout probability during training (default 0.0 = no drop).
    pub fn clause_drop_p(mut self, p: f64) -> Self {
        assert!((0.0..1.0).contains(&p), "clause_drop_p must be in [0, 1)");
        self.clause_drop_p = p;
        self
    }

    /// Per-literal dropout probability during training (default 0.0 = no drop).
    pub fn literal_drop_p(mut self, p: f64) -> Self {
        assert!((0.0..1.0).contains(&p), "literal_drop_p must be in [0, 1)");
        self.literal_drop_p = p;
        self
    }

    /// Per-class feedback scaling weights to compensate for label imbalance
    /// (default: all 1.0).  Mirrors [`TsetlinMachine::class_weights`].
    ///
    /// [`TsetlinMachine::class_weights`]: crate::TsetlinMachine::class_weights
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

    // ---- accessors -------------------------------------------------------

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
    /// Return the specificity parameter `s`.
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
    /// New features enter as just-excluded candidates (never included), so
    /// predictions on inputs whose new features are all 0 are identical to before
    /// the grow, while the new literals remain immediately learnable. Clause
    /// weights, RNG streams, and hyperparameters are untouched.
    ///
    /// Unlike the dense model, the sparse bank grows in place: existing negated
    /// literal indices are shifted and the new literals are appended to each
    /// clause's excluded pool — no full-array reallocation.
    ///
    /// **Re-encode after growing**: previously produced [`EncodedSample`] /
    /// [`EncodedBatch`] values use the old word stride and are geometry-incompatible
    /// with the grown machine.
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
        self.bank.grow(2 * new_n_features);
        self.n_features = new_n_features;
        self.n_literals = 2 * new_n_features;
        self.words = words_for(self.n_literals);
    }

    // ---- inference -------------------------------------------------------

    /// Internal: predict from a raw literal slice.
    #[inline]
    fn predict_lit(&self, lit: &[u64]) -> usize {
        let cps = self.clauses_per_class;
        let mut best = 0usize;
        let mut best_score = i32::MIN;
        for c in 0..self.n_classes {
            let mut sum = 0i32;
            for j in 0..cps {
                let cj = c * cps + j;
                if self.bank.fire_predict(cj, lit) {
                    let w = self.weights[cj];
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

    /// Seed clause `j` of `class` with prior knowledge before (or between)
    /// training: include `literals` (literal indices: feature `f` is `f`, its
    /// negation `n_features + f`) at the maximum state and set the clause's
    /// weight. Even `j` votes for the class, odd `j` against. Training then
    /// refines the clause like any other: feedback can add literals to it,
    /// wear its literals down, or change its weight.
    pub fn inject_clause(&mut self, class: usize, j: usize, literals: &[usize], weight: i32) {
        assert!(class < self.n_classes, "class {class} out of range");
        assert!(j < self.clauses_per_class, "clause {j} out of range");
        assert!(weight >= 1, "weight must be >= 1");
        let cj = class * self.clauses_per_class + j;
        for &l in literals {
            assert!(l < self.n_literals, "literal {l} out of range");
            self.bank.include_literal(cj, l as u32, self.bank.max_state);
        }
        self.weights[cj] = weight;
    }

    /// Seed clause `j` of `class` with a fixed rule: it holds exactly
    /// `literals` (at the maximum state) and never takes a candidate, so it
    /// costs memory only for its own literals, however large the vocabulary.
    /// It learns only where it fires: its weight rises on the class's samples
    /// it fires on and falls on the others, its literals' states move there,
    /// but a sample it does not fire on leaves it as it is (no Type Ib), so a
    /// rule the training data never triggers keeps what it was given. It
    /// cannot add a literal. Even `j` votes for the class, odd `j` against.
    pub fn inject_fixed_clause(&mut self, class: usize, j: usize, literals: &[usize], weight: i32) {
        self.inject_clause_with_candidates(class, j, literals, weight, 0.0);
    }

    /// [`inject_fixed_clause`](Self::inject_fixed_clause), with a random
    /// `candidates` share (0 to 1) of the other literals as candidates the
    /// clause can learn to include, as an ordinary clause does: e.g. a negated
    /// literal that excludes the clause's false positives. Its literals stay a
    /// closed set (no unallocated pool, nothing new when the vocabulary grows),
    /// so its memory is that share of the literals.
    pub fn inject_clause_with_candidates(
        &mut self,
        class: usize,
        j: usize,
        literals: &[usize],
        weight: i32,
        candidates: f64,
    ) {
        assert!(class < self.n_classes, "class {class} out of range");
        assert!(j < self.clauses_per_class, "clause {j} out of range");
        assert!(weight >= 1, "weight must be >= 1");
        let lits: Vec<u32> = literals
            .iter()
            .map(|&l| {
                assert!(l < self.n_literals, "literal {l} out of range");
                l as u32
            })
            .collect();
        let cj = class * self.clauses_per_class + j;
        assert!(
            (0.0..=1.0).contains(&candidates),
            "candidates must be in [0, 1]"
        );
        self.bank
            .fix_clause(cj, &lits, self.bank.max_state, candidates);
        self.weights[cj] = weight;
    }

    /// Disable clause `j` of `class`: it never fires (in training either, where
    /// an empty clause otherwise does), holds nothing and takes no candidates.
    /// For slots kept unused, e.g. the opposite-polarity partners of fixed
    /// clauses.
    pub fn disable_clause(&mut self, class: usize, j: usize) {
        assert!(class < self.n_classes, "class {class} out of range");
        assert!(j < self.clauses_per_class, "clause {j} out of range");
        self.bank.disable_clause(class * self.clauses_per_class + j);
    }

    /// Predict the class for an encoded sample.
    #[inline]
    pub fn predict(&self, sample: &EncodedSample) -> usize {
        self.predict_lit(&sample.0)
    }

    /// Fill `out` with the clamped weighted clause sums for each class.
    pub fn scores(&self, sample: &EncodedSample, out: &mut [i32]) {
        debug_assert_eq!(out.len(), self.n_classes);
        let lit = &sample.0;
        let cps = self.clauses_per_class;
        for (c, out_c) in out.iter_mut().enumerate() {
            let mut sum = 0i32;
            for j in 0..cps {
                let cj = c * cps + j;
                if self.bank.fire_predict(cj, lit) {
                    let w = self.weights[cj];
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
        (0..cps)
            .filter(|&j| self.bank.fire_predict(class * cps + j, lit))
            .collect()
    }

    /// Predict classes for all samples in an encoded batch.
    pub fn predict_batch(&self, batch: &EncodedBatch) -> Vec<usize> {
        let packed = batch.data.as_slice();
        let n = batch.len();
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

    /// Compute the fraction of correctly predicted samples in an encoded batch.
    pub fn accuracy(&self, batch: &EncodedBatch, ys: &[usize]) -> f64 {
        assert_eq!(batch.len(), ys.len());
        let packed = batch.data.as_slice();
        let n = batch.len();
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

    // ---- training --------------------------------------------------------

    /// Clamped weighted clause sum for class `c` under the current dropout mask.
    fn class_sum_train(&self, c: usize, lit: &[u64], lit_active: &[u64]) -> i32 {
        let cps = self.clauses_per_class;
        let mut sum = 0i32;
        for j in 0..cps {
            let cj = c * cps + j;
            if self.bank.fire_train(cj, lit, lit_active) {
                let w = self.weights[cj];
                if j & 1 == 0 {
                    sum += w;
                } else {
                    sum -= w;
                }
            }
        }
        sum.clamp(-self.threshold, self.threshold)
    }

    /// Apply Type I / II feedback to all clauses of class `c`.
    fn update_class(&mut self, c: usize, target: u8, sum: i32, lit: &[u64], lit_active: &[u64]) {
        let cps = self.clauses_per_class;
        #[cfg(feature = "parallel")]
        let words = self.words;
        let boost = self.boost_true_positive;
        let wmax = self.threshold;
        let opts = self.bank.opts();
        let rate = self.bank.feedback_rate;
        let weighted = self.weighted_clauses;
        let (type_i_p, type_ii_p) = (self.type_i_p, self.type_ii_p);
        let max_inc = self.max_included_literals;
        let drop_p = self.clause_drop_p;
        let cw = self.class_weights[c];
        let inv_p = 1.0 / self.s;
        let keep_p = (self.s - 1.0) / self.s;

        let t = wmax as f64;
        let v = sum as f64;
        let p = if self.confidence_driven_updating {
            ((t - v.abs()) / t * cw).min(1.0)
        } else if target == 1 {
            ((t - v) / (2.0 * t) * cw).min(1.0)
        } else {
            ((t + v) / (2.0 * t) * cw).min(1.0)
        };

        let Self {
            bank,
            weights,
            rngs,
            class_rngs,
            ..
        } = self;

        let drop_mask: Vec<bool> = if drop_p > 0.0 {
            let crng = &mut class_rngs[c];
            (0..cps).map(|_| crng.next_f64() < drop_p).collect()
        } else {
            Vec::new()
        };

        let (half, max_state) = bank.dims();

        // Per-clause feedback kernel, shared by the scalar and Rayon paths. `j` is
        // the clause index within the class (determines polarity). Each clause owns
        // disjoint state (`clause`, `w`, `rng`), so this is lock-free and produces
        // identical results regardless of evaluation order.
        let apply = |j: usize,
                     clause: &mut crate::clause_bank::sparse::SparseClause,
                     w: &mut i32,
                     rng: &mut Rng| {
            if !drop_mask.is_empty() && drop_mask[j] {
                return;
            }
            let positive = j & 1 == 0;
            let type_i = (target == 1) == positive;
            if rng.next_f64() > p * if type_i { type_i_p } else { type_ii_p } {
                return;
            }
            if type_i {
                // Type I.
                let fired = clause.fire_train(lit, lit_active);
                // A fixed clause learns only where it fires: Type Ib, which
                // wears down the literals of a clause that does not fire,
                // would erase a seeded rule whose events the data lacks.
                if !fired && clause.is_fixed() {
                    return;
                }
                let fired_under = fired && clause.n_included() < max_inc;
                if fired_under && weighted {
                    *w = (*w + 1).min(wmax);
                }
                clause.type_i(
                    lit,
                    lit_active,
                    fired_under,
                    boost,
                    rng,
                    inv_p,
                    keep_p,
                    max_inc,
                    half,
                    max_state,
                    opts,
                );
            } else {
                // Type II.
                if !clause.fire_train(lit, lit_active) {
                    return;
                }
                if weighted {
                    *w = (*w - 1).max(1);
                }
                clause.type_ii(lit, lit_active, half, max_state, rng, rate);
            }
        };

        let clause_slice = &mut bank.clauses_mut()[c * cps..(c + 1) * cps];
        let w_slice = &mut weights[c * cps..(c + 1) * cps];
        let rng_slice = &mut rngs[c * cps..(c + 1) * cps];

        // Sparse training benefits from the work term (its per-clause loop is
        // scalar, so wide clauses amortise Rayon well — measured in
        // parallel_scaling). `words` upper-bounds a clause's variable index-list
        // work; after absorbing convergence the real work is smaller, so this may
        // over-parallelise late training (mild overhead only), never under.
        #[cfg(feature = "parallel")]
        if use_parallel(cps, words) {
            use rayon::prelude::*;
            clause_slice
                .par_iter_mut()
                .zip(w_slice.par_iter_mut())
                .zip(rng_slice.par_iter_mut())
                .enumerate()
                .for_each(|(j, ((clause, w), rng))| apply(j, clause, w, rng));
            return;
        }

        for j in 0..cps {
            apply(j, &mut clause_slice[j], &mut w_slice[j], &mut rng_slice[j]);
        }
    }

    /// Internal: train from a raw literal slice.
    fn fit_one_lit(&mut self, lit: &[u64], y: usize) {
        debug_assert_eq!(lit.len(), self.words);
        debug_assert!(y < self.n_classes);

        let mut neg = self.rng.below(self.n_classes);
        while neg == y {
            neg = self.rng.below(self.n_classes);
        }

        // Per-sample literal-active mask (shared by both class updates).
        let lit_active: Vec<u64> = if self.literal_drop_p > 0.0 {
            let keep = 1.0 - self.literal_drop_p;
            let rng = &mut self.literal_rng;
            let n_literals = self.n_literals;
            let mut mask = vec![0u64; self.words];
            for l in 0..n_literals {
                if rng.next_f64() < keep {
                    mask[l / WORD_BITS] |= 1u64 << (l % WORD_BITS);
                }
            }
            mask
        } else {
            vec![!0u64; self.words]
        };

        let sum_y = self.class_sum_train(y, lit, &lit_active);
        let sum_neg = self.class_sum_train(neg, lit, &lit_active);
        self.update_class(y, 1, sum_y, lit, &lit_active);
        self.update_class(neg, 0, sum_neg, lit, &lit_active);
    }

    /// Train on a single encoded sample with true label `y`.
    pub fn fit_one(&mut self, sample: &EncodedSample, y: usize) {
        let lit = sample.0.clone();
        self.fit_one_lit(&lit, y);
    }

    /// Run one training epoch over an encoded batch, shuffling the order each epoch.
    pub fn fit_epoch(&mut self, batch: &EncodedBatch, ys: &[usize]) {
        let n = batch.len();
        assert_eq!(n, ys.len());
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let k = self.rng.below(i + 1);
            order.swap(i, k);
        }
        let w = self.words;
        let data = batch.data.clone();
        for &i in &order {
            self.fit_one_lit(&data[i * w..(i + 1) * w], ys[i]);
        }
    }

    // ---- absorbing-state introspection -----------------------------------

    /// Fraction of (clause, literal) slots whose TA is at the absorbing **include**
    /// state.  Grows toward the converged clause size as training proceeds.
    pub fn absorbed_include_fraction(&self) -> f64 {
        let total = self.bank.total_literals();
        if total == 0 {
            return 0.0;
        }
        let (at_max, _) = self.bank.count_absorbing_include();
        at_max as f64 / total as f64
    }

    /// Fraction of (clause, literal) slots that have been **absorbed out** (removed
    /// from the candidate pool at the exclude floor).
    ///
    /// Note: unlike [`TsetlinMachine::absorbed_exclude_fraction`] (which counts
    /// literals at counter `0`), the sparse bank *removes* such literals — so this
    /// reports the removed fraction.  Both trend upward as training converges but
    /// are not numerically comparable.
    ///
    /// [`TsetlinMachine::absorbed_exclude_fraction`]: crate::TsetlinMachine::absorbed_exclude_fraction
    pub fn absorbed_exclude_fraction(&self) -> f64 {
        let total = self.bank.total_literals();
        if total == 0 {
            return 0.0;
        }
        self.bank.count_absorbed_exclude() as f64 / total as f64
    }

    /// How many automata sit at each state: `counts[s]` is the number of
    /// literals at state `s` (`0..=max_state`) in the clauses' included and
    /// candidate lists. Literals absorbed out of the lists
    /// ([`absorbed_literals`](Self::absorbed_literals)) and literals not yet
    /// inserted (with `literal_insertion_state`) have no automaton and are not
    /// counted.
    pub fn ta_state_histogram(&self) -> Vec<u64> {
        let mut counts = vec![0u64; usize::from(self.bank.max_state) + 1];
        for c in &self.bank.clauses {
            for &s in c.included_state.iter().chain(&c.excluded_state) {
                counts[usize::from(s)] += 1;
            }
        }
        counts
    }

    /// Number of (clause, literal) pairs absorbed out of the candidate lists
    /// (see [`absorbing`](Self::absorbing)): removed for good.
    pub fn absorbed_literals(&self) -> u64 {
        self.bank.count_absorbed_exclude()
    }

    // ---- interpretability ------------------------------------------------

    /// Return the included literals for `clause` of `class` as
    /// `(feature_index, is_negated)` pairs (same decode as [`TsetlinMachine`]).
    pub fn clause_rule(&self, class: usize, clause: usize) -> Vec<(usize, bool)> {
        let cj = class * self.clauses_per_class + clause;
        let mut rule: Vec<(usize, bool)> = self
            .bank
            .included_literals(cj)
            .iter()
            .map(|&l| {
                let l = l as usize;
                if l < self.n_features {
                    (l, false)
                } else {
                    (l - self.n_features, true)
                }
            })
            .collect();
        // The sparse pool order is unstable (swap-remove); sort for a stable view.
        rule.sort_unstable();
        rule
    }

    /// Whether clause `clause` is a positive-polarity clause (even index).
    pub fn clause_is_positive(&self, clause: usize) -> bool {
        clause & 1 == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::Encoder;

    const N_FEATURES: usize = 12;

    fn make_xor(n: usize, noise: f64, seed: u64) -> (Vec<Vec<u8>>, Vec<usize>) {
        let mut rng = Rng::new(seed);
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for _ in 0..n {
            let f: Vec<u8> = (0..N_FEATURES)
                .map(|_| (rng.next_u64() & 1) as u8)
                .collect();
            let mut y = (f[0] ^ f[1]) as usize;
            if rng.next_f64() <= noise {
                y = 1 - y;
            }
            xs.push(f);
            ys.push(y);
        }
        (xs, ys)
    }

    fn encode(xs: &[Vec<u8>], enc: &Encoder) -> EncodedBatch {
        let rows: Vec<&[u8]> = xs.iter().map(|v| v.as_slice()).collect();
        enc.encode_batch(&rows)
    }

    #[test]
    fn an_injected_clause_votes_on_its_literals_and_keeps_training() {
        let mut tm = TMSparseClassifier::new(2, N_FEATURES, 10, 50, 3.0)
            .absorbing(10)
            .literal_sampling(0.5);
        // Clause 0 of class 1 (votes for it): feature 2 present, feature 5 absent.
        tm.inject_clause(1, 0, &[2, N_FEATURES + 5], 7);
        tm.inject_clause(1, 0, &[2], 7); // idempotent
        let c = &tm.bank.clauses[tm.clauses_per_class];
        let mut seen: Vec<u32> = c
            .included
            .iter()
            .chain(&c.excluded)
            .chain(&c.unallocated)
            .copied()
            .collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(
            seen.len() + c.absorbed_count() as usize,
            2 * N_FEATURES,
            "every literal once"
        );
        assert_eq!(c.included.len(), 2);
        assert!(c.included_state.iter().all(|&s| s == tm.bank.max_state));
        let sample = |f2: u8, f5: u8| {
            let mut x = vec![0u8; N_FEATURES];
            x[2] = f2;
            x[5] = f5;
            EncodedSample::from_bits(&x, N_FEATURES)
        };
        let mut out = [0i32; 2];
        tm.scores(&sample(1, 0), &mut out);
        assert_eq!(out, [0, 7], "only the injected clause has literals yet");
        tm.scores(&sample(1, 1), &mut out);
        assert_eq!(out, [0, 0], "feature 5 present: the clause stays silent");
        tm.scores(&sample(0, 0), &mut out);
        assert_eq!(out, [0, 0]);
        // Training continues from the injected state.
        let (xs, ys) = make_xor(400, 0.0, 9);
        let enc = Encoder::for_binary(N_FEATURES);
        tm.fit_epoch(&encode(&xs, &enc), &ys);
    }

    #[test]
    fn a_fixed_clause_holds_only_its_literals_through_training_growth_and_saving() {
        let mut tm = TMSparseClassifier::new(2, N_FEATURES, 10, 50, 3.0)
            .absorbing(10)
            .literal_insertion_state(100);
        tm.inject_fixed_clause(1, 0, &[2, N_FEATURES + 5], 7);
        tm.disable_clause(1, 1);
        let fixed = tm.clauses_per_class;
        let check = |tm: &TMSparseClassifier, n_literals: usize| {
            let c = &tm.bank.clauses[fixed];
            assert!(c.is_fixed() && c.unallocated.is_empty());
            assert!(
                c.included.len() + c.excluded.len() <= 2,
                "only its own literals"
            );
            assert_eq!(
                c.included.len() + c.excluded.len() + c.absorbed_count() as usize,
                n_literals
            );
            let d = &tm.bank.clauses[fixed + 1];
            assert!(d.is_disabled());
            assert!(d.included.is_empty() && d.excluded.is_empty() && d.unallocated.is_empty());
        };
        check(&tm, 2 * N_FEATURES);
        let sample = |f2: u8, f5: u8| {
            let mut x = vec![0u8; N_FEATURES];
            x[2] = f2;
            x[5] = f5;
            EncodedSample::from_bits(&x, N_FEATURES)
        };
        let mut out = [0i32; 2];
        tm.scores(&sample(1, 0), &mut out);
        assert_eq!(out, [0, 7]);
        tm.scores(&sample(1, 1), &mut out);
        assert_eq!(out, [0, 0]);
        // A disabled clause does not fire in training, where empty ones do.
        let lit = sample(0, 0).0;
        let all = vec![u64::MAX; lit.len()];
        assert!(!tm.bank.fire_train(fixed + 1, &lit, &all));
        assert!(
            tm.bank.fire_train(fixed + 3, &lit, &all),
            "an empty clause does"
        );

        let (xs, ys) = make_xor(400, 0.0, 9);
        let enc = Encoder::for_binary(N_FEATURES);
        for _ in 0..5 {
            tm.fit_epoch(&encode(&xs, &enc), &ys);
        }
        check(&tm, 2 * N_FEATURES);

        tm.grow_features(N_FEATURES + 4);
        check(&tm, 2 * (N_FEATURES + 4));
        assert!(
            !tm.bank.clauses[fixed + 2].unallocated.is_empty()
                || !tm.bank.clauses[fixed + 2].excluded.is_empty(),
            "an ordinary clause gets the new literals"
        );

        // Re-initialising the bank (literal_sampling) leaves them alone.
        let tm = tm.literal_sampling(0.5);
        check(&tm, 2 * (N_FEATURES + 4));

        #[cfg(feature = "serde")]
        {
            use crate::serial::SaveLoad;
            let dir = std::env::temp_dir().join(format!("tmu-fixed-{}", std::process::id()));
            tm.save(&dir).unwrap();
            let back = TMSparseClassifier::load(&dir).unwrap();
            std::fs::remove_file(&dir).ok();
            check(&back, 2 * (N_FEATURES + 4));
        }
    }

    #[test]
    fn a_rule_clause_with_candidates_can_learn_a_literal() {
        // Class 1 is "feature 0 present and feature 1 absent"; the injected
        // clause knows only feature 0 and has every other literal as a
        // candidate, so training can add NOT feature 1.
        let mut tm = TMSparseClassifier::new(2, N_FEATURES, 4, 10, 3.0);
        tm.inject_clause_with_candidates(1, 0, &[0], 5, 1.0);
        for j in 1..4 {
            tm.disable_clause(1, j);
            tm.disable_clause(0, j);
        }
        tm.disable_clause(0, 0);
        let fixed = tm.clauses_per_class;
        {
            let c = &tm.bank.clauses[fixed];
            assert!(c.is_fixed() && c.unallocated.is_empty());
            assert_eq!(c.excluded.len(), 2 * N_FEATURES - 1);
        }
        let mut rng = crate::rng::Rng::new(5);
        let (mut xs, mut ys) = (Vec::new(), Vec::new());
        for _ in 0..600 {
            let mut x = vec![0u8; N_FEATURES];
            for v in x.iter_mut() {
                *v = (rng.next_u64() & 1) as u8;
            }
            ys.push(usize::from(x[0] == 1 && x[1] == 0));
            xs.push(x);
        }
        let enc = Encoder::for_binary(N_FEATURES);
        let data = encode(&xs, &enc);
        for _ in 0..30 {
            tm.fit_epoch(&data, &ys);
        }
        let c = &tm.bank.clauses[fixed];
        assert!(
            c.included.contains(&(N_FEATURES as u32 + 1)),
            "learned NOT feature 1: {:?}",
            c.included
        );
        assert!(c.unallocated.is_empty());
        let without = TMSparseClassifier::new(2, N_FEATURES, 4, 10, 3.0);
        let mut without = without;
        without.inject_fixed_clause(1, 0, &[0], 5);
        assert!(
            without.bank.clauses[fixed].excluded.is_empty(),
            "none by default"
        );
    }

    #[test]
    fn a_fixed_clause_the_data_never_triggers_keeps_its_literals_and_weight() {
        // Feature 7 never occurs in the data; a clause on it must survive.
        let mut tm = TMSparseClassifier::new(2, N_FEATURES, 6, 10, 2.0);
        tm.inject_fixed_clause(1, 0, &[7], 5);
        tm.disable_clause(1, 1);
        let mut rng = crate::rng::Rng::new(3);
        let (mut xs, mut ys) = (Vec::new(), Vec::new());
        for _ in 0..500 {
            let mut x = vec![0u8; N_FEATURES];
            for v in x.iter_mut().take(7) {
                *v = (rng.next_u64() & 1) as u8;
            }
            ys.push(usize::from(x[0] == 1));
            xs.push(x);
        }
        let enc = Encoder::for_binary(N_FEATURES);
        let data = encode(&xs, &enc);
        for _ in 0..20 {
            tm.fit_epoch(&data, &ys);
        }
        let cj = tm.clauses_per_class;
        assert_eq!(tm.bank.clauses[cj].included, vec![7]);
        assert_eq!(tm.bank.clauses[cj].included_state, vec![tm.bank.max_state]);
        assert_eq!(tm.weights[cj], 5);
    }

    #[test]
    fn ta_state_histogram_and_absorbed_literals_cover_every_slot() {
        let (xtr, ytr) = make_xor(1000, 0.0, 62);
        let tr = encode(&xtr, &Encoder::for_binary(N_FEATURES));
        let mut tm =
            TMSparseClassifier::with_config(2, N_FEATURES, 10, 15, 3.9, 4, true, 42).absorbing(2);
        for _ in 0..20 {
            tm.fit_epoch(&tr, &ytr);
        }
        let h = tm.ta_state_histogram();
        assert_eq!(h.len(), 16, "4 state bits: states 0..=15");
        let slots = (2 * 10 * 2 * N_FEATURES) as u64;
        let absorbed = tm.absorbed_literals();
        assert!(
            absorbed > 0,
            "absorbing at 2 must absorb something in 20 epochs"
        );
        assert_eq!(h.iter().sum::<u64>() + absorbed, slots);
        assert_eq!(
            absorbed as f64 / slots as f64,
            tm.absorbed_exclude_fraction()
        );
    }

    fn train_sparse(seed: u64, epochs: usize) -> (TMSparseClassifier, EncodedBatch, Vec<usize>) {
        let (xtr, ytr) = make_xor(4000, 0.1, seed);
        let (xte, yte) = make_xor(2000, 0.0, seed ^ 0xABCD);
        let enc = Encoder::for_binary(N_FEATURES);
        let tr = encode(&xtr, &enc);
        let te = encode(&xte, &enc);
        let mut tm = TMSparseClassifier::with_config(2, N_FEATURES, 10, 15, 3.9, 8, true, seed)
            .max_included_literals(8);
        for _ in 0..epochs {
            tm.fit_epoch(&tr, &ytr);
        }
        (tm, te, yte)
    }

    #[test]
    fn sparse_learns_xor() {
        let (tm, te, yte) = train_sparse(7, 30);
        let acc = tm.accuracy(&te, &yte);
        assert!(acc > 0.95, "sparse XOR accuracy too low: {acc}");
    }

    /// Exercises the Rayon clause-parallel path: with `clauses_per_class >=
    /// PARALLEL_MIN` (128), a `--features parallel` build takes the `par_iter_mut`
    /// branch in `update_class` and the parallel `predict_batch`/`accuracy`. The
    /// scalar build runs the same logic sequentially; both must learn XOR. Because
    /// each clause consumes only its own RNG, the result is identical across builds.
    #[test]
    fn parallel_path_learns_xor() {
        let (xtr, ytr) = make_xor(4000, 0.1, 23);
        let (xte, yte) = make_xor(2000, 0.0, 77);
        let enc = Encoder::for_binary(N_FEATURES);
        let tr = encode(&xtr, &enc);
        let te = encode(&xte, &enc);
        // 130 clauses/class > PARALLEL_MIN, so the parallel branch runs under
        // `--features parallel`.
        let mut tm = TMSparseClassifier::with_config(2, N_FEATURES, 130, 20, 3.9, 8, true, 23)
            .max_included_literals(8);
        for _ in 0..15 {
            tm.fit_epoch(&tr, &ytr);
        }
        let acc = tm.accuracy(&te, &yte);
        assert!(acc > 0.95, "parallel-path XOR accuracy too low: {acc}");
        // predict_batch (parallel branch) must agree with per-sample predict.
        let batch = tm.predict_batch(&te);
        let w = tm.words;
        for (i, &p) in batch.iter().enumerate() {
            assert_eq!(p, tm.predict_lit(&te.data[i * w..(i + 1) * w]));
        }
    }

    #[test]
    fn same_seed_same_result() {
        let (tm_a, te, yte) = train_sparse(11, 10);
        let (tm_b, _, _) = train_sparse(11, 10);
        assert_eq!(tm_a.accuracy(&te, &yte), tm_b.accuracy(&te, &yte));
        assert_eq!(tm_a.weights, tm_b.weights);
    }

    #[test]
    fn predict_batch_matches_single() {
        let (tm, te, _) = train_sparse(5, 10);
        let batch = tm.predict_batch(&te);
        let n = te.len();
        let w = tm.words;
        let data = te.data.as_slice();
        for i in 0..n {
            let single = tm.predict_lit(&data[i * w..(i + 1) * w]);
            assert_eq!(batch[i], single);
        }
    }

    #[test]
    fn accuracy_matches_manual_loop() {
        let (tm, te, yte) = train_sparse(5, 10);
        let preds = tm.predict_batch(&te);
        let manual =
            preds.iter().zip(&yte).filter(|(p, y)| p == y).count() as f64 / yte.len() as f64;
        assert_eq!(manual, tm.accuracy(&te, &yte));
    }

    #[test]
    fn weights_stay_in_1_to_threshold() {
        let (tm, _, _) = train_sparse(9, 15);
        assert!(tm.weights.iter().all(|&w| (1..=tm.threshold).contains(&w)));
    }

    #[test]
    fn max_included_literals_reduces_clause_size() {
        let (xtr, ytr) = make_xor(3000, 0.1, 21);
        let enc = Encoder::for_binary(N_FEATURES);
        let tr = encode(&xtr, &enc);

        let mut unbounded =
            TMSparseClassifier::with_config(2, N_FEATURES, 10, 15, 3.9, 8, true, 21);
        let mut bounded = TMSparseClassifier::with_config(2, N_FEATURES, 10, 15, 3.9, 8, true, 21)
            .max_included_literals(2);
        for _ in 0..20 {
            unbounded.fit_epoch(&tr, &ytr);
            bounded.fit_epoch(&tr, &ytr);
        }
        let avg = |tm: &TMSparseClassifier| -> f64 {
            let cps = tm.clauses_per_class();
            let mut total = 0usize;
            for c in 0..tm.n_classes() {
                for j in 0..cps {
                    total += tm.clause_rule(c, j).len();
                }
            }
            total as f64 / (tm.n_classes() * cps) as f64
        };
        // `max_included_literals` is a soft Type Ia growth guard (Type II can still
        // promote a literal past it, exactly as in the dense model), so the bound is
        // not a hard cap — the meaningful property is that it yields smaller clauses.
        let bounded_avg = avg(&bounded);
        let unbounded_avg = avg(&unbounded);
        assert!(
            bounded_avg < unbounded_avg,
            "bounded avg {bounded_avg} should be smaller than unbounded {unbounded_avg}"
        );
    }

    #[test]
    fn clause_is_positive_matches_index_parity() {
        let tm = TMSparseClassifier::new(2, N_FEATURES, 10, 15, 3.9);
        for j in 0..tm.clauses_per_class() {
            assert_eq!(tm.clause_is_positive(j), j % 2 == 0);
        }
    }

    #[test]
    fn sparse_matches_dense_accuracy() {
        use crate::TsetlinMachine;
        let (xtr, ytr) = make_xor(4000, 0.1, 33);
        let (xte, yte) = make_xor(2000, 0.0, 99);
        let enc = Encoder::for_binary(N_FEATURES);
        let tr = encode(&xtr, &enc);
        let te = encode(&xte, &enc);

        let mut dense = TsetlinMachine::with_config(2, N_FEATURES, 10, 15, 3.9, 8, true, 33);
        let mut sparse = TMSparseClassifier::with_config(2, N_FEATURES, 10, 15, 3.9, 8, true, 33);
        for _ in 0..30 {
            dense.fit_epoch(&tr, &ytr);
            sparse.fit_epoch(&tr, &ytr);
        }
        let dense_acc = dense.accuracy(&te, &yte);
        let sparse_acc = sparse.accuracy(&te, &yte);
        assert!(
            sparse_acc >= dense_acc - 0.05,
            "sparse {sparse_acc} lags dense {dense_acc} by more than 0.05"
        );
    }

    #[test]
    fn use_parallel_wide_few_clauses() {
        // 32 clauses/class (< the 128 count threshold) but 256 features → words=8,
        // so cps*words=256 engages the work-aware branch for BOTH sparse training
        // and inference. Results are path-independent; predict_batch must match the
        // per-sample path and the model must learn. Under `--features parallel`
        // this covers the branch the old count-only gate skipped for wide,
        // few-clause sparse models.
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
        let e = Encoder::for_binary(nf);
        let btr = encode(&xtr, &e);
        let bte = encode(&xte, &e);
        let mut tm = TMSparseClassifier::with_config(2, nf, 32, 15, 3.9, 8, true, 7)
            .max_included_literals(8);
        for _ in 0..30 {
            tm.fit_epoch(&btr, &ytr);
        }
        let w = tm.words;
        let batch = tm.predict_batch(&bte);
        for (i, &p) in batch.iter().enumerate() {
            assert_eq!(p, tm.predict_lit(&bte.data[i * w..(i + 1) * w]));
        }
        assert!(
            tm.accuracy(&bte, &yte) > 0.9,
            "wide/few-clause sparse model failed to learn"
        );
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
        let (xtr, ytr) = make_xor(4000, 0.1, 5);
        let (xte, _) = make_xor(1000, 0.0, 6);
        let e12 = Encoder::for_binary(N_FEATURES);
        let mut tm = TMSparseClassifier::with_config(2, N_FEATURES, 10, 15, 3.9, 8, true, 7)
            .max_included_literals(8);
        for _ in 0..30 {
            tm.fit_epoch(&encode(&xtr, &e12), &ytr);
        }

        let bte12 = encode(&xte, &e12);
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

        // Same rows, zero-padded to 20 features: predictions and per-class scores
        // must be identical (new literals are all just-excluded, never included).
        let e20 = Encoder::for_binary(20);
        let xte20 = pad_to(&xte, 20);
        let bte20 = encode(&xte20, &e20);
        assert_eq!(tm.predict_batch(&bte20), preds_before);
        for (x, before) in xte20.iter().zip(&scores_before) {
            let mut out = [0i32; 2];
            tm.scores(&e20.encode_one(x), &mut out);
            assert_eq!(&out, before, "scores changed after grow for {x:?}");
        }
    }

    #[test]
    fn grow_noop_when_equal() {
        let (tm0, _, _) = train_sparse(8, 10);
        let mut tm = tm0.clone();
        let rules_before: Vec<Vec<(usize, bool)>> = (0..2)
            .flat_map(|c| (0..tm.clauses_per_class()).map(move |j| (c, j)))
            .map(|(c, j)| tm.clause_rule(c, j))
            .collect();
        tm.grow_features(N_FEATURES);
        assert_eq!(tm.n_features(), N_FEATURES);
        let rules_after: Vec<Vec<(usize, bool)>> = (0..2)
            .flat_map(|c| (0..tm.clauses_per_class()).map(move |j| (c, j)))
            .map(|(c, j)| tm.clause_rule(c, j))
            .collect();
        assert_eq!(rules_before, rules_after);
    }

    #[test]
    #[should_panic(expected = "cannot shrink")]
    fn grow_panics_on_shrink() {
        let mut tm = TMSparseClassifier::new(2, N_FEATURES, 10, 15, 3.9);
        tm.grow_features(N_FEATURES - 1);
    }

    #[test]
    fn grow_preserves_clause_rules() {
        let (tm0, _, _) = train_sparse(9, 20);
        let mut tm = tm0.clone();
        let rules_before: Vec<Vec<(usize, bool)>> = (0..2)
            .flat_map(|c| (0..tm.clauses_per_class()).map(move |j| (c, j)))
            .map(|(c, j)| tm.clause_rule(c, j))
            .collect();

        tm.grow_features(40);

        // Feature indices and negation flags are stable across the grow.
        let rules_after: Vec<Vec<(usize, bool)>> = (0..2)
            .flat_map(|c| (0..tm.clauses_per_class()).map(move |j| (c, j)))
            .map(|(c, j)| tm.clause_rule(c, j))
            .collect();
        assert_eq!(rules_before, rules_after);
    }

    #[test]
    fn grow_then_learns_new_feature() {
        // Pre-train on XOR over the first 12 features.
        let (xtr, ytr) = make_xor(3000, 0.1, 10);
        let e12 = Encoder::for_binary(N_FEATURES);
        let mut tm = TMSparseClassifier::with_config(2, N_FEATURES, 20, 15, 3.9, 8, true, 7)
            .max_included_literals(8);
        for _ in 0..20 {
            tm.fit_epoch(&encode(&xtr, &e12), &ytr);
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
        let (xtr2, ytr2) = make(&mut rng, 4000);
        let (xte2, yte2) = make(&mut rng, 1000);

        let e16 = Encoder::for_binary(16);
        for _ in 0..30 {
            tm.fit_epoch(&encode(&xtr2, &e16), &ytr2);
        }
        let acc = tm.accuracy(&encode(&xte2, &e16), &yte2);
        assert!(
            acc >= 0.95,
            "grown sparse TM failed to learn new feature: {acc}"
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn grown_model_serde_roundtrip() {
        use crate::serial::SaveLoad;
        let (xtr, ytr) = make_xor(3000, 0.1, 12);
        let (xte, _) = make_xor(500, 0.0, 13);
        let e12 = Encoder::for_binary(N_FEATURES);
        let mut tm = TMSparseClassifier::with_config(2, N_FEATURES, 10, 15, 3.9, 8, true, 7)
            .max_included_literals(8);
        for _ in 0..15 {
            tm.fit_epoch(&encode(&xtr, &e12), &ytr);
        }

        tm.grow_features(20);
        let e20 = Encoder::for_binary(20);
        let xtr20 = pad_to(&xtr, 20);
        for _ in 0..5 {
            tm.fit_epoch(&encode(&xtr20, &e20), &ytr);
        }

        let mut buf = Vec::new();
        tm.write_to(&mut buf).unwrap();
        let loaded = TMSparseClassifier::read_from(&mut buf.as_slice()).unwrap();

        let bte20 = encode(&pad_to(&xte, 20), &e20);
        assert_eq!(tm.predict_batch(&bte20), loaded.predict_batch(&bte20));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_round_trip() {
        use crate::serial::SaveLoad;
        let (tm, te, _) = train_sparse(3, 8);
        let mut buf = Vec::new();
        tm.write_to(&mut buf).unwrap();
        let loaded = TMSparseClassifier::read_from(&mut buf.as_slice()).unwrap();
        assert_eq!(tm.predict_batch(&te), loaded.predict_batch(&te));

        // Wrong-tag load must fail cleanly.
        let mut bad = buf.clone();
        bad[8] = 0xFE; // corrupt the type tag byte (after 4 magic + 4 version).
        assert!(TMSparseClassifier::read_from(&mut bad.as_slice()).is_err());
    }

    #[test]
    fn clause_inspect_reports_signed_votes() {
        use crate::ClauseInspect;
        let (xs, ys) = make_xor(1000, 0.1, 12);
        let e = Encoder::for_binary(N_FEATURES);
        let b = encode(&xs, &e);
        let mut tm = TMSparseClassifier::with_config(2, N_FEATURES, 10, 15, 3.9, 8, true, 6);
        for _ in 0..10 {
            tm.fit_epoch(&b, &ys);
        }
        let x = e.encode_one(&xs[0]);
        for c in 0..2 {
            // The fired clauses' signed weights add up to the class score.
            let sum: i32 = ClauseInspect::fired_clauses(&tm, &x, c)
                .iter()
                .map(|&j| ClauseInspect::clause_weight(&tm, c, j))
                .sum();
            let mut out = [0i32; 2];
            tm.scores(&x, &mut out);
            assert_eq!(sum.clamp(-15, 15), out[c]);
        }
        assert!(
            ClauseInspect::clause_weight(&tm, 0, 1) < 0,
            "odd clauses vote against"
        );
    }
}
