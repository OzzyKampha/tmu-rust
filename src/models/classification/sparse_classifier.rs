//! Sparse weighted multiclass Tsetlin Machine classifier.
//!
//! Mirrors TMU's sparse clause-bank classifier: the same weighted-clause vanilla
//! algorithm as [`TsetlinMachine`](crate::TsetlinMachine), but backed by the
//! [`SparseClauseBank`] — per-clause index lists with **absorbing actions** that
//! permanently drop literals from candidacy as training converges.  See the
//! [`sparse`](crate::clause_bank::sparse) module docs for the data layout.
//!
//! ## Differences from the dense [`TsetlinMachine`]
//!
//! * **Scalar, single-threaded.**  The variable-length per-clause lists don't map
//!   onto the dense bank's flat-array SIMD / rayon `par_chunks_mut` paths, so this
//!   model trains sequentially even with `--features parallel`.
//! * **No Type III feedback.**  Type III maintains a second per-literal indicator
//!   array, which conflicts with the sparse bank removing literals; it is omitted.
//! * **`absorbed_exclude_fraction` means "removed".**  In the dense model it
//!   counts literals at counter `0`; here those literals have been *removed* from
//!   the pool, so this returns the fraction of (clause, literal) slots that have
//!   been absorbed out.  The numbers are therefore not directly comparable to the
//!   dense model's, though both trend upward as training converges.
//!
//! All other public API mirrors [`TsetlinMachine`] one-to-one.

use crate::clause_bank::dense::{words_for, GOLDEN, WORD_BITS};
use crate::clause_bank::sparse::SparseClauseBank;
use crate::encoder::{EncodedBatch, EncodedSample};
use crate::rng::Rng;

/// A weighted multiclass Tsetlin Machine backed by a sparse clause bank.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TMSparseClassifier {
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
    literal_drop_p: f64,

    /// Sparse clause storage; `n_classes * clauses_per_class` clauses total.
    bank: SparseClauseBank,
    /// Per-clause integer weights (>= 1), indexed `c * clauses_per_class + j`.
    weights: Vec<i32>,

    /// Per-clause feedback RNG (also gates the per-clause feedback probability).
    rngs: Vec<Rng>,
    /// Per-class RNG for the clause-dropout mask.
    class_rngs: Vec<Rng>,
    /// Dedicated RNG for the per-sample literal-active (dropout) mask.
    literal_rng: Rng,
    /// Master RNG: epoch shuffle + negative-class sampling only.
    rng: Rng,

    /// Per-class feedback scaling factors for imbalanced datasets (default 1.0).
    class_weights: Vec<f64>,
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
        }
    }

    // ---- builders --------------------------------------------------------

    /// Limit how many literals each clause may include (Type Ia guard).
    /// Sparse models especially benefit from this — it bounds clause size directly.
    pub fn max_included_literals(mut self, max: usize) -> Self {
        self.max_included_literals = max;
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
        let boost = self.boost_true_positive;
        let wmax = self.threshold;
        let max_inc = self.max_included_literals;
        let drop_p = self.clause_drop_p;
        let cw = self.class_weights[c];
        let inv_p = 1.0 / self.s;
        let keep_p = (self.s - 1.0) / self.s;

        let t = wmax as f64;
        let v = sum as f64;
        let p = if target == 1 {
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

        for j in 0..cps {
            if !drop_mask.is_empty() && drop_mask[j] {
                continue;
            }
            let cj = c * cps + j;
            let rng = &mut rngs[cj];
            if rng.next_f64() > p {
                continue;
            }
            let positive = j & 1 == 0;
            if (target == 1) == positive {
                // Type I.
                let fired = bank.fire_train(cj, lit, lit_active);
                let under_limit = bank.n_included(cj) < max_inc;
                let fired_under = fired && under_limit;
                if fired_under {
                    weights[cj] = (weights[cj] + 1).min(wmax);
                }
                bank.type_i(
                    cj,
                    lit,
                    lit_active,
                    fired_under,
                    boost,
                    rng,
                    inv_p,
                    keep_p,
                    max_inc,
                );
            } else {
                // Type II.
                if !bank.fire_train(cj, lit, lit_active) {
                    continue;
                }
                weights[cj] = (weights[cj] - 1).max(1);
                bank.type_ii(cj, lit, lit_active);
            }
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
}
