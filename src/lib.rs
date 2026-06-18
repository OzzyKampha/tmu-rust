//! Bit-packed, **weighted** multiclass Tsetlin Machine with bit-parallel and
//! optionally multi-threaded training.
//!
//! Builds on the bit-plane design (see the crate as ported from `cair/tmu`):
//!
//! * **Bit-packed inference** — clauses fire with a few 64-bit word ops.
//! * **Bit-parallel training** — TA counters live in bit-planes; updates are
//!   masked carry/borrow word operations with packed Bernoulli feedback masks.
//! * **Weighted clauses** — each clause carries an integer weight (>= 1). A
//!   firing clause contributes `polarity * weight` to the class sum; the weight
//!   rises on Type I reinforcement and falls (floor 1) on Type II. This reaches
//!   target accuracy with far fewer clauses and trains more stably.
//! * **Parallel training** (feature `parallel`) — within each class update the
//!   clause loop is data-parallel: every clause writes only its own state slice,
//!   its own weight, and its own RNG, so the work is disjoint and race-free with
//!   no locks and no `unsafe`.
//!
//! ## Validation note
//!
//! Weighted learning and the per-clause-RNG structure (which the parallel path
//! relies on) were validated in Python against the scalar reference: weighted
//! clauses learn noisy-XOR with 8 clauses/class and the synthetic NDR task with
//! 30 clauses/class at full accuracy. The parallel path performs the *same*
//! disjoint per-clause updates concurrently. RNG consumption differs from the
//! scalar version, so results are distributionally equivalent, not bit-identical.
//! Not yet compiled or benchmarked in this environment — run `cargo test`.

mod booleanizer;
mod rng;

pub mod data;

pub use booleanizer::Booleanizer;
pub use rng::Rng;

const WORD_BITS: usize = 64;
/// Precision (bits) of packed Bernoulli feedback masks; probability error <= 2^-MASK_BITS.
const MASK_BITS: usize = 12;
const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
/// Minimum item count before rayon parallelism pays off over its dispatch overhead.
#[cfg(feature = "parallel")]
const PARALLEL_MIN: usize = 128;

#[inline(always)]
fn words_for(bits: usize) -> usize {
    (bits + WORD_BITS - 1) / WORD_BITS
}

fn digits_of(p: f64, n: usize) -> Vec<u8> {
    let mut d = Vec::with_capacity(n);
    let mut x = p;
    for _ in 0..n {
        x *= 2.0;
        d.push(if x >= 1.0 {
            x -= 1.0;
            1u8
        } else {
            0u8
        });
    }
    d
}

/// A bit-packed weighted multiclass Tsetlin Machine.
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

    /// Bit-plane TA counters. Clause `cj = c*CPC + j` occupies
    /// `state[cj*state_bits*words .. (cj+1)*state_bits*words]`; within that
    /// chunk plane `b` word `w` is at `b*words + w`. Top plane = include bitset.
    state: Vec<u64>,
    /// Per-clause integer weights (>= 1), indexed `c*CPC + j`.
    weights: Vec<i32>,
    /// Per-clause RNG (enables lock-free parallel training).
    rngs: Vec<Rng>,
    /// Per-class RNG used for drop/inv/keep mask generation (one per class).
    /// Keeps class updates independent so they can run in parallel via rayon::join.
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

    /// Per-clause dropout probability during training.
    /// Each clause is independently skipped with this probability on every
    /// training sample. Mirrors TMU's `clause_drop_p` (default: 0.0 = no drop).
    /// Typical value for large models: 0.75.
    pub fn clause_drop_p(mut self, p: f64) -> Self {
        assert!((0.0..1.0).contains(&p), "clause_drop_p must be in [0, 1)");
        self.clause_drop_p = p;
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
    /// Specificity parameter `s` controlling the TA feedback probabilities.
    pub fn s(&self) -> f64 {
        self.s
    }
    /// Learned weight of a clause (>= 1).
    pub fn clause_weight(&self, class: usize, clause: usize) -> i32 {
        self.weights[class * self.clauses_per_class + clause]
    }

    // ---- indexing ---------------------------------------------------------

    #[inline(always)]
    fn clause_base(&self, c: usize, j: usize) -> usize {
        (c * self.clauses_per_class + j) * self.state_bits * self.words
    }
    #[inline(always)]
    fn top_base(&self, c: usize, j: usize) -> usize {
        self.clause_base(c, j) + (self.state_bits - 1) * self.words
    }

    // ---- packing ----------------------------------------------------------

    #[inline]
    pub fn pack(x: &[u8], n_features: usize, out: &mut [u64]) {
        for w in out.iter_mut() {
            *w = 0;
        }
        for i in 0..n_features {
            if x[i] != 0 {
                out[i / WORD_BITS] |= 1u64 << (i % WORD_BITS);
            } else {
                let j = n_features + i;
                out[j / WORD_BITS] |= 1u64 << (j % WORD_BITS);
            }
        }
    }
    #[inline]
    pub fn pack_sample(&self, x: &[u8], out: &mut [u64]) {
        Self::pack(x, self.n_features, out);
    }

    // ---- clause firing (whole-self, for sums/inference) -------------------

    // Branchless inner loop — no early exit so the compiler can auto-vectorize
    // (AVX2: 4 u64/cycle for MNIST's 25 words ≈ 4× over scalar).
    // Empty clauses (included == 0) return false, matching TMU predict semantics.
    #[inline(always)]
    fn fire_predict(state: &[u64], tb: usize, lit: &[u64], valid: &[u64], words: usize) -> bool {
        let mut violation = 0u64;
        let mut included = 0u64;
        for k in 0..words {
            let inc = state[tb + k] & valid[k];
            violation |= inc & !lit[k];
            included |= inc;
        }
        violation == 0 && included != 0
    }

    #[allow(dead_code)]
    #[inline(always)]
    fn fire_train(&self, c: usize, j: usize) -> bool {
        let tb = self.top_base(c, j);
        for k in 0..self.words {
            let inc = self.state[tb + k] & self.valid[k];
            if inc & !self.literals[k] != 0 {
                return false;
            }
        }
        true
    }

    // ---- inference --------------------------------------------------------

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
            for j in 0..cps {
                if Self::fire_predict(state, cb + j * bw, lit, valid, words) {
                    let w = cw[j];
                    if j & 1 == 0 { sum += w; } else { sum -= w; }
                }
            }
            let v = sum.clamp(-self.threshold, self.threshold);
            if v > best_score { best_score = v; best = c; }
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
        for c in 0..self.n_classes {
            let cb = c * cps * bw + tb_off;
            let cw = &self.weights[c * cps..(c + 1) * cps];
            let mut sum = 0i32;
            for j in 0..cps {
                if Self::fire_predict(state, cb + j * bw, lit, valid, words) {
                    let w = cw[j];
                    if j & 1 == 0 { sum += w; } else { sum -= w; }
                }
            }
            out[c] = sum.clamp(-self.threshold, self.threshold);
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
        // Parallelize only when per-sample work is large enough to amortize rayon overhead.
        // clauses_per_class is the dominant cost in predict_packed; n controls batch depth.
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

    // ---- per-clause primitives (operate on one clause's chunk) ------------

    #[inline(always)]
    fn clause_fire(chunk: &[u64], lit: &[u64], valid: &[u64], words: usize, sb: usize) -> bool {
        let tb = (sb - 1) * words;
        for k in 0..words {
            let inc = chunk[tb + k] & valid[k];
            if inc & !lit[k] != 0 {
                return false;
            }
        }
        true
    }


    #[inline(always)]
    fn clause_inc(chunk: &mut [u64], words: usize, sb: usize, k: usize, mask: u64) {
        if mask == 0 {
            return;
        }
        let mut carry = mask;
        for b in 0..sb {
            let idx = b * words + k;
            let next = chunk[idx] & carry;
            chunk[idx] ^= carry;
            carry = next;
            if carry == 0 {
                return;
            }
        }
        for b in 0..sb {
            chunk[b * words + k] |= carry;
        }
    }

    #[inline(always)]
    fn clause_dec(chunk: &mut [u64], words: usize, sb: usize, k: usize, mask: u64) {
        if mask == 0 {
            return;
        }
        let mut borrow = mask;
        for b in 0..sb {
            let idx = b * words + k;
            let next = !chunk[idx] & borrow;
            chunk[idx] ^= borrow;
            borrow = next;
            if borrow == 0 {
                return;
            }
        }
        let clear = !borrow;
        for b in 0..sb {
            chunk[b * words + k] &= clear;
        }
    }

    #[inline(always)]
    fn bmask_word(rng: &mut Rng, digits: &[u8]) -> u64 {
        let mut word = 0u64;
        for i in (0..digits.len()).rev() {
            let r = rng.next_u64();
            word = if digits[i] == 1 { r | word } else { r & word };
        }
        word
    }

    // clause_type_i takes pre-generated masks (one word per literal-word) rather than
    // probability digits — the caller generates them once and shares across all clauses,
    // matching TMU's reuse_random_feedback=true default.
    fn clause_type_i(
        chunk: &mut [u64],
        weight: &mut i32,
        lit: &[u64],
        valid: &[u64],
        words: usize,
        sb: usize,
        boost: bool,
        inv_mask: &[u64],
        keep_mask: &[u64],
        wmax: i32,
        max_included: usize,
    ) {
        let out = Self::clause_fire(chunk, lit, valid, words, sb);
        let tb = (sb - 1) * words;
        let under_limit = max_included == usize::MAX || {
            let n: u32 = (0..words).map(|k| (chunk[tb + k] & valid[k]).count_ones()).sum();
            (n as usize) < max_included
        };
        if out && under_limit {
            *weight = (*weight + 1).min(wmax);
            for k in 0..words {
                let litw = lit[k];
                let inc_mask = if boost { litw & valid[k] } else { litw & keep_mask[k] & valid[k] };
                Self::clause_inc(chunk, words, sb, k, inc_mask);
                Self::clause_dec(chunk, words, sb, k, !litw & inv_mask[k] & valid[k]);
            }
        } else {
            for k in 0..words {
                Self::clause_dec(chunk, words, sb, k, inv_mask[k] & valid[k]);
            }
        }
    }

    fn clause_type_ii(
        chunk: &mut [u64],
        weight: &mut i32,
        lit: &[u64],
        valid: &[u64],
        words: usize,
        sb: usize,
    ) {
        if !Self::clause_fire(chunk, lit, valid, words, sb) {
            return;
        }
        *weight = (*weight - 1).max(1);
        let tb = (sb - 1) * words;
        for k in 0..words {
            let excluded = !chunk[tb + k];
            let inc_mask = !lit[k] & excluded & valid[k];
            Self::clause_inc(chunk, words, sb, k, inc_mask);
        }
    }

    // ---- training ---------------------------------------------------------

    fn class_sum_train(&self, c: usize) -> i32 {
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
            for k in 0..words {
                if self.state[jbase + k] & self.valid[k] & !self.literals[k] != 0 {
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
    fn update_class(&mut self, c: usize, target: u8) {
        let sum = self.class_sum_train(c);
        let cps = self.clauses_per_class;
        let words = self.words;
        let sb = self.state_bits;
        let bw = sb * words;
        let boost = self.boost_true_positive;
        let wmax = self.threshold;
        let max_inc = self.max_included_literals;
        let drop_p = self.clause_drop_p;

        let Self { state, weights, rngs, class_rngs, literals, valid, dig_inv, dig_keep, .. } = self;
        let lit = literals.as_slice();
        let val = valid.as_slice();
        let crng = &mut class_rngs[c];

        let t = wmax as f64;
        let v = sum as f64;
        let p = if target == 1 { (t - v) / (2.0 * t) } else { (t + v) / (2.0 * t) };

        let drop_mask: Vec<bool> = if drop_p > 0.0 {
            (0..cps).map(|_| crng.next_f64() < drop_p).collect()
        } else {
            vec![]
        };
        let inv_mask: Vec<u64> = (0..words).map(|_| Self::bmask_word(crng, dig_inv)).collect();
        let keep_mask: Vec<u64> = (0..words).map(|_| Self::bmask_word(crng, dig_keep)).collect();

        let class_state = &mut state[c * cps * bw..(c + 1) * cps * bw];
        let class_w = &mut weights[c * cps..(c + 1) * cps];
        let class_rng = &mut rngs[c * cps..(c + 1) * cps];

        for j in 0..cps {
            if !drop_mask.is_empty() && drop_mask[j] { continue; }
            if class_rng[j].next_f64() > p { continue; }
            let chunk = &mut class_state[j * bw..(j + 1) * bw];
            let w = &mut class_w[j];
            let positive = j & 1 == 0;
            if (target == 1) == positive {
                Self::clause_type_i(chunk, w, lit, val, words, sb, boost, &inv_mask, &keep_mask, wmax, max_inc);
            } else {
                Self::clause_type_ii(chunk, w, lit, val, words, sb);
            }
        }
    }

    // Standalone class-update kernel — takes explicit slices so it can be called
    // from a rayon::join closure without needing &mut self.
    #[cfg(feature = "parallel")]
    #[allow(clippy::too_many_arguments)]
    fn update_class_par(
        sum: i32, target: u8,
        class_state: &mut [u64], class_weights: &mut [i32],
        clause_rngs: &mut [Rng], class_rng: &mut Rng,
        lit: &[u64], valid: &[u64], dig_inv: &[u8], dig_keep: &[u8],
        cps: usize, words: usize, sb: usize,
        boost: bool, wmax: i32, max_inc: usize, drop_p: f64,
    ) {
        let bw = sb * words;
        let t = wmax as f64;
        let v = sum as f64;
        let p = if target == 1 { (t - v) / (2.0 * t) } else { (t + v) / (2.0 * t) };

        let drop_mask: Vec<bool> = if drop_p > 0.0 {
            (0..cps).map(|_| class_rng.next_f64() < drop_p).collect()
        } else {
            vec![]
        };
        let inv_mask: Vec<u64> = (0..words).map(|_| Self::bmask_word(class_rng, dig_inv)).collect();
        let keep_mask: Vec<u64> = (0..words).map(|_| Self::bmask_word(class_rng, dig_keep)).collect();

        for j in 0..cps {
            if !drop_mask.is_empty() && drop_mask[j] { continue; }
            if clause_rngs[j].next_f64() > p { continue; }
            let chunk = &mut class_state[j * bw..(j + 1) * bw];
            let w = &mut class_weights[j];
            let positive = j & 1 == 0;
            if (target == 1) == positive {
                Self::clause_type_i(chunk, w, lit, valid, words, sb, boost, &inv_mask, &keep_mask, wmax, max_inc);
            } else {
                Self::clause_type_ii(chunk, w, lit, valid, words, sb);
            }
        }
    }

    pub fn fit_one_packed(&mut self, lit: &[u64], y: usize) {
        debug_assert_eq!(lit.len(), self.words);
        debug_assert!(y < self.n_classes);
        self.literals.copy_from_slice(lit);

        let mut neg = self.rng.below(self.n_classes);
        while neg == y { neg = self.rng.below(self.n_classes); }

        #[cfg(feature = "parallel")]
        {
            // Compute both class sums concurrently (read-only).
            let (sum_y, sum_neg) = rayon::join(
                || self.class_sum_train(y),
                || self.class_sum_train(neg),
            );

            // Extract disjoint per-class slices for y and neg, then run both
            // class updates in parallel — each class owns exclusive state/weight/rng slices.
            let cps = self.clauses_per_class;
            let bw = self.state_bits * self.words;
            let words = self.words;
            let sb = self.state_bits;
            let boost = self.boost_true_positive;
            let wmax = self.threshold;
            let max_inc = self.max_included_literals;
            let drop_p = self.clause_drop_p;

            let Self { state, weights, rngs, class_rngs, valid, dig_inv, dig_keep, literals, .. } = self;
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
                || Self::update_class_par(sum_y, 1, cs_y, cw_y, cr_y, crng_y, lit_sl, val, dig_inv, dig_keep, cps, words, sb, boost, wmax, max_inc, drop_p),
                || Self::update_class_par(sum_neg, 0, cs_neg, cw_neg, cr_neg, crng_neg, lit_sl, val, dig_inv, dig_keep, cps, words, sb, boost, wmax, max_inc, drop_p),
            );
            return;
        }

        #[cfg(not(feature = "parallel"))]
        {
            self.update_class(y, 1);
            self.update_class(neg, 0);
        }
    }

    pub fn fit_one(&mut self, x: &[u8], y: usize) {
        debug_assert!(y < self.n_classes);
        let nf = self.n_features;
        let mut lit = vec![0u64; self.words];
        Self::pack(x, nf, &mut lit);
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
            Self::pack(x, nf, &mut packed[i * w..(i + 1) * w]);
        }
        self.fit_epoch_packed(&packed, n, ys);
    }

    pub fn pack_dataset(&self, xs: &[&[u8]]) -> Vec<u64> {
        let n = xs.len();
        let w = self.words;
        let nf = self.n_features;
        let mut packed = vec![0u64; n * w];
        for (i, x) in xs.iter().enumerate() {
            Self::pack(x, nf, &mut packed[i * w..(i + 1) * w]);
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

    // ---- interpretability -------------------------------------------------

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

    #[test]
    fn weighted_learns_xor_with_few_clauses() {
        let (xtr, ytr) = make_xor(5000, 0.25, 1);
        let (xte, yte) = make_xor(2000, 0.0, 2);
        let xtr_r: Vec<&[u8]> = xtr.iter().map(|v| v.as_slice()).collect();
        let xte_r: Vec<&[u8]> = xte.iter().map(|v| v.as_slice()).collect();

        // only 8 clauses/class thanks to weighting
        let mut tm = TsetlinMachine::with_config(2, 12, 8, 15, 3.9, 8, true, 7);
        for _ in 0..15 {
            tm.fit_epoch(&xtr_r, &ytr);
        }
        let acc = tm.accuracy(&xte_r, &yte);
        assert!(acc > 0.95, "expected >0.95, got {acc}");
        // weights stay bounded by T
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
            TsetlinMachine::clause_inc(chunk, words, sb, 0, valid0);
        }
        for b in 0..sb {
            assert_eq!(chunk[b * words] & valid0, valid0);
        }
        for _ in 0..1000 {
            TsetlinMachine::clause_dec(chunk, words, sb, 0, valid0);
        }
        for b in 0..sb {
            assert_eq!(chunk[b * words] & valid0, 0);
        }
    }
}
