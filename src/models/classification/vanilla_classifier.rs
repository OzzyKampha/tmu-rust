//! Vanilla weighted multiclass Tsetlin Machine classifier.
//!
//! Mirrors TMU's `vanilla_classifier.py` / `TMClassifier`.

use crate::clause_bank::dense::{
    bmask_word, clause_type_i, clause_type_ii, digits_of, fire_predict, words_for, GOLDEN,
    MASK_BITS, WORD_BITS,
};
#[cfg(feature = "parallel")]
use crate::clause_bank::dense::PARALLEL_MIN;
use crate::rng::Rng;

/// A bit-packed weighted multiclass Tsetlin Machine.
///
/// Bit-plane TA state, weighted clauses, and optional inter-class parallelism
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
    /// One per class keeps class updates independent for rayon::join parallelism.
    class_rngs: Vec<Rng>,
    /// Per-word mask of real literal bits.
    valid: Vec<u64>,
    dig_inv: Vec<u8>,
    dig_keep: Vec<u8>,

    literals: Vec<u64>,
    rng: Rng, // for shuffling and negative-class selection only
}

impl TsetlinMachine {
    pub fn new(
        n_classes: usize,
        n_features: usize,
        clauses_per_class: usize,
        threshold: i32,
        s: f64,
    ) -> Self {
        Self::with_config(n_classes, n_features, clauses_per_class, threshold, s, 8, true, 42)
    }

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

    pub fn n_classes(&self) -> usize {
        self.n_classes
    }
    pub fn n_features(&self) -> usize {
        self.n_features
    }
    pub fn clauses_per_class(&self) -> usize {
        self.clauses_per_class
    }
    pub fn words_per_sample(&self) -> usize {
        self.words
    }
    pub fn s(&self) -> f64 {
        self.s
    }
    pub fn clause_weight(&self, class: usize, clause: usize) -> i32 {
        self.weights[class * self.clauses_per_class + clause]
    }

    // ---- internal indexing -----------------------------------------------

    #[inline(always)]
    fn clause_base(&self, c: usize, j: usize) -> usize {
        (c * self.clauses_per_class + j) * self.state_bits * self.words
    }

    #[inline(always)]
    fn top_base(&self, c: usize, j: usize) -> usize {
        self.clause_base(c, j) + (self.state_bits - 1) * self.words
    }

    // ---- packing (thin wrappers over clause_bank::pack) ------------------

    /// Pack a raw feature vector into the bit-interleaved literal representation.
    #[inline]
    pub fn pack(x: &[u8], n_features: usize, out: &mut [u64]) {
        crate::clause_bank::dense::pack(x, n_features, out);
    }


    #[inline]
    pub fn pack_sample(&self, x: &[u8], out: &mut [u64]) {
        crate::clause_bank::dense::pack(x, self.n_features, out);
    }

    // ---- inference -------------------------------------------------------

    #[inline]
    pub fn predict_packed(&self, lit: &[u64]) -> usize {
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

    pub fn scores_packed(&self, lit: &[u64], out: &mut [i32]) {
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

    pub fn predict(&self, x: &[u8]) -> usize {
        let mut lit = vec![0u64; self.words];
        self.pack_sample(x, &mut lit);
        self.predict_packed(&lit)
    }

    pub fn predict_batch_packed(&self, packed: &[u64], n: usize) -> Vec<usize> {
        debug_assert_eq!(packed.len(), n * self.words);
        let w = self.words;
        #[cfg(feature = "parallel")]
        if n >= PARALLEL_MIN && self.clauses_per_class >= PARALLEL_MIN {
            use rayon::prelude::*;
            return (0..n)
                .into_par_iter()
                .map(|i| self.predict_packed(&packed[i * w..(i + 1) * w]))
                .collect();
        }
        (0..n)
            .map(|i| self.predict_packed(&packed[i * w..(i + 1) * w]))
            .collect()
    }

    // ---- training helpers ------------------------------------------------

    #[allow(dead_code)]
    #[inline(always)]
    fn fire_train(&self, c: usize, j: usize) -> bool {
        let cb = self.clause_base(c, j);
        let bw = self.state_bits * self.words;
        let all_active: Vec<u64> = vec![!0u64; self.words];
        crate::clause_bank::dense::clause_fire(
            &self.state[cb..cb + bw],
            &self.literals,
            &self.valid,
            self.words,
            self.state_bits,
            &all_active,
        )
    }

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

    #[cfg_attr(feature = "parallel", allow(dead_code))]
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

        for j in 0..cps {
            if !drop_mask.is_empty() && drop_mask[j] {
                continue;
            }
            if class_rng[j].next_f64() > p {
                continue;
            }
            let chunk = &mut class_state[j * bw..(j + 1) * bw];
            let w = &mut class_w[j];
            let positive = j & 1 == 0;
            if (target == 1) == positive {
                clause_type_i(
                    chunk, w, lit, val, words, sb, boost, &inv_mask, &keep_mask, wmax, max_inc,
                    lit_active,
                );
            } else {
                clause_type_ii(chunk, w, lit, val, words, sb, lit_active);
            }
        }
    }

    // Standalone class-update kernel — takes explicit slices so it can be called
    // from a rayon::join closure without needing &mut self.
    #[cfg(feature = "parallel")]
    #[allow(clippy::too_many_arguments)]
    fn update_class_par(
        sum: i32,
        target: u8,
        class_state: &mut [u64],
        class_weights: &mut [i32],
        clause_rngs: &mut [Rng],
        class_rng: &mut Rng,
        lit: &[u64],
        valid: &[u64],
        dig_inv: &[u8],
        dig_keep: &[u8],
        lit_active: &[u64],
        cps: usize,
        words: usize,
        sb: usize,
        boost: bool,
        wmax: i32,
        max_inc: usize,
        drop_p: f64,
    ) {
        let bw = sb * words;
        let t = wmax as f64;
        let v = sum as f64;
        let p = if target == 1 {
            (t - v) / (2.0 * t)
        } else {
            (t + v) / (2.0 * t)
        };

        let drop_mask: Vec<bool> = if drop_p > 0.0 {
            (0..cps).map(|_| class_rng.next_f64() < drop_p).collect()
        } else {
            vec![]
        };
        let inv_mask: Vec<u64> =
            (0..words).map(|_| bmask_word(class_rng, dig_inv)).collect();
        let keep_mask: Vec<u64> =
            (0..words).map(|_| bmask_word(class_rng, dig_keep)).collect();

        for j in 0..cps {
            if !drop_mask.is_empty() && drop_mask[j] {
                continue;
            }
            if clause_rngs[j].next_f64() > p {
                continue;
            }
            let chunk = &mut class_state[j * bw..(j + 1) * bw];
            let w = &mut class_weights[j];
            let positive = j & 1 == 0;
            if (target == 1) == positive {
                clause_type_i(
                    chunk, w, lit, valid, words, sb, boost, &inv_mask, &keep_mask, wmax, max_inc,
                    lit_active,
                );
            } else {
                clause_type_ii(chunk, w, lit, valid, words, sb, lit_active);
            }
        }
    }

    pub fn fit_one_packed(&mut self, lit: &[u64], y: usize) {
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

        #[cfg(feature = "parallel")]
        {
            // Compute both class sums concurrently (read-only).
            let (sum_y, sum_neg) = rayon::join(
                || self.class_sum_train(y, &lit_active),
                || self.class_sum_train(neg, &lit_active),
            );

            // Extract disjoint per-class slices for y and neg, then run both
            // class updates in parallel — each class owns exclusive state/weight/rng slices.
            let cps = self.clauses_per_class;
            let bw = self.state_bits * self.words;
            let sb = self.state_bits;
            let boost = self.boost_true_positive;
            let wmax = self.threshold;
            let max_inc = self.max_included_literals;
            let drop_p = self.clause_drop_p;

            let Self {
                state, weights, rngs, class_rngs, valid, dig_inv, dig_keep, literals, ..
            } = self;
            let lit_sl = literals.as_slice();
            let val = valid.as_slice();

            // Split two disjoint mutable chunks from a slice by class index.
            macro_rules! split2 {
                ($v:expr, $a:expr, $b:expr, $s:expr) => {{
                    if $a < $b {
                        let (l, r) = $v.split_at_mut($b * $s);
                        (&mut l[$a * $s..($a + 1) * $s], &mut r[0..$s])
                    } else {
                        let (l, r) = $v.split_at_mut($a * $s);
                        (&mut r[0..$s], &mut l[$b * $s..($b + 1) * $s])
                    }
                }};
            }
            macro_rules! split2_one {
                ($v:expr, $a:expr, $b:expr) => {{
                    if $a < $b {
                        let (l, r) = $v.split_at_mut($b);
                        (&mut l[$a], &mut r[0])
                    } else {
                        let (l, r) = $v.split_at_mut($a);
                        (&mut r[0], &mut l[$b])
                    }
                }};
            }

            let (cs_y, cs_neg) = split2!(state, y, neg, cps * bw);
            let (cw_y, cw_neg) = split2!(weights, y, neg, cps);
            let (cr_y, cr_neg) = split2!(rngs, y, neg, cps);
            let (crng_y, crng_neg) = split2_one!(class_rngs, y, neg);

            rayon::join(
                || {
                    Self::update_class_par(
                        sum_y, 1, cs_y, cw_y, cr_y, crng_y, lit_sl, val, dig_inv, dig_keep,
                        &lit_active, cps, words, sb, boost, wmax, max_inc, drop_p,
                    )
                },
                || {
                    Self::update_class_par(
                        sum_neg, 0, cs_neg, cw_neg, cr_neg, crng_neg, lit_sl, val, dig_inv,
                        dig_keep, &lit_active, cps, words, sb, boost, wmax, max_inc, drop_p,
                    )
                },
            );
        }

        #[cfg(not(feature = "parallel"))]
        {
            let sum_y = self.class_sum_train(y, &lit_active);
            let sum_neg = self.class_sum_train(neg, &lit_active);
            self.update_class(y, 1, sum_y, &lit_active);
            self.update_class(neg, 0, sum_neg, &lit_active);
        }
    }

    pub fn fit_one(&mut self, x: &[u8], y: usize) {
        debug_assert!(y < self.n_classes);
        let nf = self.n_features;
        let mut lit = vec![0u64; self.words];
        crate::clause_bank::dense::pack(x, nf, &mut lit);
        self.fit_one_packed(&lit, y);
    }

    pub fn fit_epoch_packed(&mut self, packed: &[u64], n: usize, ys: &[usize]) {
        debug_assert_eq!(packed.len(), n * self.words);
        assert_eq!(n, ys.len());
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let k = self.rng.below(i + 1);
            order.swap(i, k);
        }
        let w = self.words;
        for &i in &order {
            self.fit_one_packed(&packed[i * w..(i + 1) * w], ys[i]);
        }
    }

    pub fn fit_epoch(&mut self, xs: &[&[u8]], ys: &[usize]) {
        assert_eq!(xs.len(), ys.len());
        let n = xs.len();
        let w = self.words;
        let nf = self.n_features;
        let mut packed = vec![0u64; n * w];
        for (i, x) in xs.iter().enumerate() {
            crate::clause_bank::dense::pack(x, nf, &mut packed[i * w..(i + 1) * w]);
        }
        self.fit_epoch_packed(&packed, n, ys);
    }

    // ---- dataset helpers -------------------------------------------------

    pub fn pack_dataset(&self, xs: &[&[u8]]) -> Vec<u64> {
        let n = xs.len();
        let w = self.words;
        let nf = self.n_features;
        let mut packed = vec![0u64; n * w];
        for (i, x) in xs.iter().enumerate() {
            crate::clause_bank::dense::pack(x, nf, &mut packed[i * w..(i + 1) * w]);
        }
        packed
    }

    pub fn accuracy_packed(&self, packed: &[u64], n: usize, ys: &[usize]) -> f64 {
        debug_assert_eq!(packed.len(), n * self.words);
        assert_eq!(n, ys.len());
        let w = self.words;
        #[cfg(feature = "parallel")]
        if n >= PARALLEL_MIN {
            use rayon::prelude::*;
            let correct: usize = (0..n)
                .into_par_iter()
                .filter(|&i| self.predict_packed(&packed[i * w..(i + 1) * w]) == ys[i])
                .count();
            return correct as f64 / n as f64;
        }
        let correct = (0..n)
            .filter(|&i| self.predict_packed(&packed[i * w..(i + 1) * w]) == ys[i])
            .count();
        correct as f64 / n as f64
    }

    pub fn accuracy(&self, xs: &[&[u8]], ys: &[usize]) -> f64 {
        let packed = self.pack_dataset(xs);
        self.accuracy_packed(&packed, xs.len(), ys)
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

    pub fn clause_is_positive(&self, clause: usize) -> bool {
        clause & 1 == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clause_bank::dense::{clause_dec, clause_inc, clause_type_i, clause_type_ii, fire_predict};

    // ---- helpers -------------------------------------------------------------

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

    fn as_slices(xs: &[Vec<u8>]) -> Vec<&[u8]> {
        xs.iter().map(|v| v.as_slice()).collect()
    }

    // ---- existing tests ------------------------------------------------------

    #[test]
    fn weighted_learns_xor_with_few_clauses() {
        let (xtr, ytr) = make_xor(5000, 0.25, 1);
        let (xte, yte) = make_xor(2000, 0.0, 2);
        let xtr_r = as_slices(&xtr);
        let xte_r = as_slices(&xte);

        let mut tm = TsetlinMachine::with_config(2, 12, 8, 15, 3.9, 8, true, 7);
        for _ in 0..15 {
            tm.fit_epoch(&xtr_r, &ytr);
        }
        let acc = tm.accuracy(&xte_r, &yte);
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

    // ---- pack / predict API consistency --------------------------------------

    #[test]
    fn pack_roundtrip_predict_agrees() {
        let (xtr, ytr) = make_xor(500, 0.0, 10);
        let mut tm = TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 42);
        tm.fit_epoch(&as_slices(&xtr), &ytr);

        let (xte, _) = make_xor(200, 0.0, 20);
        for x in &xte {
            let unpacked = tm.predict(x);
            let mut lit = vec![0u64; tm.words_per_sample()];
            TsetlinMachine::pack(x, tm.n_features(), &mut lit);
            let packed = tm.predict_packed(&lit);
            assert_eq!(unpacked, packed, "predict and predict_packed disagree for {x:?}");
        }
    }

    #[test]
    fn accuracy_packed_matches_manual_loop() {
        let (xtr, ytr) = make_xor(500, 0.0, 11);
        let mut tm = TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 42);
        tm.fit_epoch(&as_slices(&xtr), &ytr);

        let (xte, yte) = make_xor(300, 0.0, 21);
        let xte_r = as_slices(&xte);
        let packed = tm.pack_dataset(&xte_r);
        let n = xte.len();
        let w = tm.words_per_sample();

        let api_acc = tm.accuracy_packed(&packed, n, &yte);
        let manual_correct = (0..n)
            .filter(|&i| tm.predict_packed(&packed[i * w..(i + 1) * w]) == yte[i])
            .count();
        let manual_acc = manual_correct as f64 / n as f64;

        assert!((api_acc - manual_acc).abs() < 1e-12);
    }

    #[test]
    fn predict_batch_packed_matches_single() {
        let (xtr, ytr) = make_xor(300, 0.0, 12);
        let mut tm = TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 42);
        tm.fit_epoch(&as_slices(&xtr), &ytr);

        let (xte, _) = make_xor(100, 0.0, 22);
        let xte_r = as_slices(&xte);
        let packed = tm.pack_dataset(&xte_r);
        let n = xte.len();
        let w = tm.words_per_sample();

        let batch = tm.predict_batch_packed(&packed, n);
        let single: Vec<usize> = (0..n)
            .map(|i| tm.predict_packed(&packed[i * w..(i + 1) * w]))
            .collect();

        assert_eq!(batch, single);
    }

    // ---- scores_packed -------------------------------------------------------

    #[test]
    fn scores_packed_correct_class_wins_after_training() {
        let (xtr, ytr) = make_xor(2000, 0.0, 13);
        let mut tm = TsetlinMachine::with_config(2, 12, 10, 15, 3.9, 8, true, 42);
        for _ in 0..10 {
            tm.fit_epoch(&as_slices(&xtr), &ytr);
        }

        let (xte, yte) = make_xor(100, 0.0, 23);
        let mut correct = 0usize;
        for (x, &y) in xte.iter().zip(&yte) {
            let mut lit = vec![0u64; tm.words_per_sample()];
            TsetlinMachine::pack(x, tm.n_features(), &mut lit);
            let mut scores = vec![0i32; 2];
            tm.scores_packed(&lit, &mut scores);
            let pred = if scores[0] >= scores[1] { 0 } else { 1 };
            if pred == y {
                correct += 1;
            }
        }
        let acc = correct as f64 / 100.0;
        assert!(acc > 0.90, "scores_packed acc {acc} too low");
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
        let xtr_r = as_slices(&xtr);

        let mut tm1 = TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 99);
        let mut tm2 = TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 99);
        for _ in 0..5 {
            tm1.fit_epoch(&xtr_r, &ytr);
            tm2.fit_epoch(&xtr_r, &ytr);
        }

        let (xte, yte) = make_xor(200, 0.0, 24);
        let xte_r = as_slices(&xte);
        assert_eq!(
            tm1.accuracy(&xte_r, &yte),
            tm2.accuracy(&xte_r, &yte),
            "same seed must produce identical results"
        );
    }

    // ---- clause_drop_p -------------------------------------------------------

    #[test]
    fn clause_drop_p_one_leaves_state_unchanged() {
        let (xtr, ytr) = make_xor(200, 0.0, 15);
        let xtr_r = as_slices(&xtr);
        let mut tm = TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 42)
            .clause_drop_p(0.9999);
        let state_before = tm.state.clone();
        tm.fit_epoch(&xtr_r, &ytr);
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
        let mut tm = TsetlinMachine::with_config(2, 12, 10, 15, 3.9, 8, true, 42)
            .clause_drop_p(0.0);
        for _ in 0..10 {
            tm.fit_epoch(&as_slices(&xtr), &ytr);
        }
        let acc = tm.accuracy(&as_slices(&xte), &yte);
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
        let xtr_r = as_slices(&xtr);

        let mut tm_tight = TsetlinMachine::with_config(2, 12, 10, 15, 3.9, 8, true, 42)
            .max_included_literals(2);
        let mut tm_free = TsetlinMachine::with_config(2, 12, 10, 15, 3.9, 8, true, 42);
        for _ in 0..10 {
            tm_tight.fit_epoch(&xtr_r, &ytr);
            tm_free.fit_epoch(&xtr_r, &ytr);
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

        let mut tm = TsetlinMachine::with_config(4, 8, 20, 30, 3.9, 8, true, 42);
        for _ in 0..20 {
            tm.fit_epoch(&as_slices(&xs), &ys);
        }
        let acc = tm.accuracy(&as_slices(&xte), &yte);
        assert!(acc > 0.85, "4-class XOR should reach >0.85, got {acc}");
    }

    // ---- weight bounds -------------------------------------------------------

    #[test]
    fn weights_stay_in_1_to_threshold() {
        let threshold = 20i32;
        let (xtr, ytr) = make_xor(1000, 0.1, 18);
        let mut tm = TsetlinMachine::with_config(2, 12, 12, threshold, 3.9, 8, true, 42);
        for _ in 0..10 {
            tm.fit_epoch(&as_slices(&xtr), &ytr);
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

    // ---- fit_epoch vs fit_epoch_packed equivalence ---------------------------

    #[test]
    fn fit_epoch_and_fit_epoch_packed_identical() {
        let (xtr, ytr) = make_xor(300, 0.0, 19);
        let xtr_r = as_slices(&xtr);

        let mut tm_a = TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 77);
        let mut tm_b = TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 77);

        let packed = tm_b.pack_dataset(&xtr_r);
        for _ in 0..5 {
            tm_a.fit_epoch(&xtr_r, &ytr);
            tm_b.fit_epoch_packed(&packed, xtr.len(), &ytr);
        }

        assert_eq!(tm_a.state, tm_b.state);
        assert_eq!(tm_a.weights, tm_b.weights);
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
        TsetlinMachine::pack(&x, nf, &mut lit);
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
        let xtr_r = as_slices(&xtr);
        let mut tm = TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 42)
            .literal_drop_p(0.9999);
        let state_before = tm.state.clone();
        tm.fit_epoch(&xtr_r, &ytr);
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
        let mut tm = TsetlinMachine::with_config(2, 12, 10, 15, 3.9, 8, true, 42)
            .literal_drop_p(0.0);
        for _ in 0..10 {
            tm.fit_epoch(&as_slices(&xtr), &ytr);
        }
        let acc = tm.accuracy(&as_slices(&xte), &yte);
        assert!(acc > 0.90, "literal_drop_p=0 should still converge, got {acc}");
    }

    // ---- absorbing states -------------------------------------------------------

    // Helper: build a one-word chunk with `sb` state planes.
    // `states[l]` is the integer TA state for literal bit `l`.
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

    // Read back TA state for literal bit `l` from a one-word chunk.
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
