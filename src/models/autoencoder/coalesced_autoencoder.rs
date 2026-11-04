//! Coalesced Tsetlin Machine autoencoder: TMU's `TMAutoEncoder`
//! (`tmu/models/autoencoder/autoencoder.py`).
//!
//! A **single shared clause bank** of `n_clauses` clauses is scored by every output
//! bit position via signed per-output weights, mirroring the coalesced architecture
//! from [`crate::CoalescedTsetlinMachine`].

use crate::clause_bank::dense::rule_from_include;
#[cfg(feature = "parallel")]
use crate::clause_bank::dense::{DENSE_TRAIN_PARALLEL_MIN, PARALLEL_MIN, use_parallel};
use crate::clause_bank::dense::{
    GOLDEN, HeldOut, HeldOutIndex, MASK_BITS, WORD_BITS, bmask_word, clause_fire, digits_of,
    expand_bits_to_bytes, rebuild_include, type_i_update_bytes, type_ii_update_bytes,
    type_iii_update, words_for,
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
///
/// Each shared clause also carries a per-sample confidence gate (mirroring TMU's
/// `update_clause` mechanism): as a clause's average `|weight|` across all outputs
/// approaches `threshold`, it becomes progressively less likely to receive further
/// feedback that sample. This protects clauses that have already specialised from
/// being repeatedly pulled in conflicting directions by other outputs' training signal.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TMCoalescedAutoEncoder {
    pub(crate) n_features: usize,
    pub(crate) n_literals: usize,
    pub(crate) words: usize,
    /// Number of clauses in the single shared clause bank.
    pub(crate) n_clauses: usize,
    pub(crate) threshold: i32,
    pub(crate) s: f64,
    pub(crate) boost_true_positive: bool,
    pub(crate) max_included_literals: usize,
    pub(crate) clause_drop_p: f64,
    pub(crate) literal_drop_p: f64,
    pub(crate) literal_rng: Rng,
    pub(crate) dig_lit_active: Vec<u8>,

    /// u8 TA counters. Clause `j` occupies `ta[j * n_literals .. (j+1) * n_literals]`.
    pub(crate) ta: Vec<u8>,
    /// Include bitset. Clause `j` occupies `include[j * words .. (j+1) * words]`.
    pub(crate) include: Vec<u64>,
    /// Inference map from literals to the clauses that include them (TMU's
    /// `literal_clause_map`), built on first prediction and dropped by every
    /// `&mut self` / `mut self` method, so it never outlives the include
    /// bitsets it was built from.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) literal_index: std::sync::OnceLock<InferenceIndex>,
    pub(crate) half: u8,
    pub(crate) max_state: u8,

    /// Signed per-output weights indexed `output * n_clauses + clause`.
    /// Initialised to random ±1; unbounded growth during training.
    pub(crate) weights: Vec<i32>,
    /// Per-clause RNGs (n_clauses total).
    pub(crate) rngs: Vec<Rng>,
    /// Per-output RNGs for inv/keep Bernoulli masks and clause dropout.
    pub(crate) output_rngs: Vec<Rng>,
    pub(crate) valid: Vec<u64>,
    pub(crate) dig_inv: Vec<u8>,
    pub(crate) dig_keep: Vec<u8>,

    pub(crate) ind: Vec<u8>,
    pub(crate) cat: Vec<u64>,
    pub(crate) d: f64,
    pub(crate) type_iii: bool,

    /// RNG stream for the per-sample confidence-based clause gate (see
    /// [`Self::confidence_gate`]). Kept separate from `rngs`/`output_rngs` so this
    /// gate's draws don't perturb the per-clause feedback sequence.
    pub(crate) clause_gate_rng: Rng,

    pub(crate) literals: Vec<u64>,
    pub(crate) rng: Rng,

    /// Train outputs in a fresh random order for every sample (see
    /// [`Self::shuffle_outputs`]). A training setting: not saved.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) shuffle_outputs: bool,
    /// [`Self::fit_examples`]' target probability; 0 = each feature's frequency.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) output_balancing: f64,
    /// Rows combined into one [`Self::fit_examples`] example.
    #[cfg_attr(feature = "serde", serde(skip, default = "one_row"))]
    pub(crate) accumulation: usize,
}

/// One sample's held-out votes from a [`TMCoalescedAutoEncoder`]
/// ([`TMCoalescedAutoEncoder::held_out_votes`]): each output's clamped vote
/// with its own feature hidden, and the clauses behind each vote.
#[derive(Clone, Debug)]
pub struct HeldOutVotes<'a> {
    ae: &'a TMCoalescedAutoEncoder,
    classes: Vec<HeldOut>,
    scores: Vec<i32>,
}

impl HeldOutVotes<'_> {
    /// Each output's clamped held-out vote (`held_out_scores`).
    pub fn scores(&self) -> &[i32] {
        &self.scores
    }

    /// The shared clauses that vote on `output` with its feature hidden.
    pub fn fired_clauses(&self, output: usize) -> Vec<usize> {
        fired_for(&self.classes, output)
    }

    /// The clause with the strongest vote on `output` in one direction among
    /// those voting on it, and its weight: the most negative weight when
    /// `against`, else the most positive (the first such clause on ties).
    /// `None` when no clause votes that way.
    pub fn strongest_clause(&self, output: usize, against: bool) -> Option<(usize, i32)> {
        let nc = self.ae.n_clauses;
        let ws = &self.ae.weights[output * nc..(output + 1) * nc];
        let o = output as u32;
        let mut best: Option<(usize, i32)> = None;
        for (j, &class) in self.classes.iter().enumerate() {
            if !votes_on(class, o) {
                continue;
            }
            let w = ws[j];
            let wanted = if against { w < 0 } else { w > 0 };
            let better = |b: i32| if against { w < b } else { w > b };
            if wanted && best.is_none_or(|(_, b)| better(b)) {
                best = Some((j, w));
            }
        }
        best
    }
}

/// Whether a clause with this verdict votes on output `o`.
#[inline]
fn votes_on(class: HeldOut, o: u32) -> bool {
    match class {
        HeldOut::All => true,
        HeldOut::AllBut(f) => f != o,
        HeldOut::Only(f) => f == o,
        HeldOut::Never => false,
    }
}

/// The clauses whose verdict lets them vote on `output`.
fn fired_for(classes: &[HeldOut], output: usize) -> Vec<usize> {
    let o = output as u32;
    (0..classes.len())
        .filter(|&j| votes_on(classes[j], o))
        .collect()
}

/// What inference reads, built from the clause bank on first prediction: the
/// literal → clause map (with each clause's single feature, for held-out
/// scoring) and the weights clause-major (`weights_t[j * n_features + o]`).
#[derive(Clone, Debug, Default)]
pub(crate) struct InferenceIndex {
    held: HeldOutIndex,
    weights_t: Vec<i32>,
}

#[cfg(feature = "serde")]
fn one_row() -> usize {
    1
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
        let clause_gate_rng = Rng::new(seed ^ 0x434C_4155_5345_4754u64); // "CLAUSEGT"

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
            literal_index: Default::default(),
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
            clause_gate_rng,
            literals: vec![0u64; words],
            rng,
            shuffle_outputs: false,
            output_balancing: 0.0,
            accumulation: 1,
        }
    }

    // ---- builders -----------------------------------------------------------

    /// Limit how many literals each clause may include (Type Ia guard).
    pub fn max_included_literals(mut self, max: usize) -> Self {
        self.reset_literal_index();
        self.max_included_literals = max;
        self
    }

    /// Per-clause dropout probability during training (default: 0.0).
    pub fn clause_drop_p(mut self, p: f64) -> Self {
        self.reset_literal_index();
        assert!((0.0..1.0).contains(&p), "clause_drop_p must be in [0, 1)");
        self.clause_drop_p = p;
        self
    }

    /// Per-literal dropout probability during training (default: 0.0).
    pub fn literal_drop_p(mut self, p: f64) -> Self {
        self.reset_literal_index();
        assert!((0.0..1.0).contains(&p), "literal_drop_p must be in [0, 1)");
        self.literal_drop_p = p;
        self.dig_lit_active = digits_of(1.0 - p, MASK_BITS);
        self
    }

    /// Enable Type III feedback with indicator strength `d` (must be > 1.0).
    pub fn type_iii_feedback(mut self, d: f64) -> Self {
        self.reset_literal_index();
        assert!(d > 1.0, "d must be > 1.0");
        self.d = d;
        self.type_iii = true;
        self
    }

    /// Start with every literal excluded, as TMU initialises its clause bank
    /// (each automaton one step below the include threshold). By default
    /// each automaton starts on either side of it at random, so about half the
    /// literals begin included. Call before training.
    pub fn start_excluded(mut self) -> Self {
        self.reset_literal_index();
        self.ta.iter_mut().for_each(|t| *t = self.half - 1);
        let (n_literals, words) = (self.n_literals, self.words);
        for j in 0..self.n_clauses {
            rebuild_include(
                &self.ta[j * n_literals..(j + 1) * n_literals],
                &mut self.include[j * words..(j + 1) * words],
                &self.valid,
                words,
                n_literals,
                self.half,
            );
        }
        self
    }

    /// Train the outputs of each sample in a fresh random order, as TMU does,
    /// instead of `0..n_features` (default off). Outputs share the clause bank,
    /// so the order changes what is learned. CPU only: `to_cuda` refuses it.
    /// [`fit_examples`](Self::fit_examples) always shuffles.
    pub fn shuffle_outputs(mut self, on: bool) -> Self {
        self.reset_literal_index();
        self.shuffle_outputs = on;
        self
    }

    /// TMU's `output_balancing` for [`fit_examples`](Self::fit_examples): the
    /// probability that an output's target is "present". 0 (the default, as in
    /// TMU) uses each feature's frequency in the training rows; 0.5 trains
    /// rare features on rows that have them half the time.
    pub fn output_balancing(mut self, p: f64) -> Self {
        self.reset_literal_index();
        assert!(
            (0.0..=1.0).contains(&p),
            "output_balancing must be in [0, 1]"
        );
        self.output_balancing = p;
        self
    }

    /// TMU's `accumulation` for [`fit_examples`](Self::fit_examples): each
    /// example is the union of this many rows with the drawn target (default 1).
    pub fn accumulation(mut self, rows: usize) -> Self {
        self.reset_literal_index();
        assert!(rows >= 1, "accumulation must be at least 1");
        self.accumulation = rows;
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

    /// The shared clauses that fire for `lit` (inference semantics: an empty
    /// clause does not fire), through the literal → clause map. A clause's
    /// output does not depend on the output being voted on, so it is computed
    /// once per sample and shared by every output's vote.
    fn fired_predict(&self, lit: &[u64]) -> Vec<usize> {
        self.index()
            .held
            .map
            .clause_outputs(lit)
            .into_iter()
            .enumerate()
            .filter_map(|(j, fires)| fires.then_some(j))
            .collect()
    }

    fn index(&self) -> &InferenceIndex {
        self.literal_index.get_or_init(|| {
            let (nc, nf) = (self.n_clauses, self.n_features);
            let mut weights_t = vec![0i32; nf * nc];
            for o in 0..nf {
                for j in 0..nc {
                    weights_t[j * nf + o] = self.weights[o * nc + j];
                }
            }
            InferenceIndex {
                held: HeldOutIndex::build(&self.include, &self.valid, nc, self.words, nf),
                weights_t,
            }
        })
    }

    /// Like [`reconstruct_scores`](Self::reconstruct_scores), but each
    /// output's vote is taken with that output's own feature hidden (its two
    /// literals left out of every clause), as training hides it.
    ///
    /// Plain reconstruction lets a clause look at the very bit it votes on: a
    /// clause that other outputs taught to include literal `o` echoes bit `o`
    /// back. Hidden, the vote says how well the *rest* of the sample predicts
    /// each bit — what an anomaly score wants. One pass through the literal →
    /// clause map gives every clause's verdict for every output.
    pub fn held_out_scores(&self, sample: &EncodedSample, out: &mut [i32]) {
        debug_assert_eq!(out.len(), self.n_features);
        out.copy_from_slice(self.held_out_votes(sample).scores());
    }

    /// The held-out votes for `sample` (as [`held_out_scores`](Self::held_out_scores)),
    /// kept with every clause's verdict, so the clauses behind any output's vote
    /// can be read without evaluating the clauses again: what an explanation of
    /// several outputs needs.
    pub fn held_out_votes(&self, sample: &EncodedSample) -> HeldOutVotes<'_> {
        let nf = self.n_features;
        let index = self.index();
        let classes = index.held.classify(&sample.0);
        let mut scores = vec![0i32; nf];
        for (j, &class) in classes.iter().enumerate() {
            let row = &index.weights_t[j * nf..(j + 1) * nf];
            match class {
                HeldOut::All => scores.iter_mut().zip(row).for_each(|(s, w)| *s += w),
                HeldOut::AllBut(f) => {
                    scores.iter_mut().zip(row).for_each(|(s, w)| *s += w);
                    scores[f as usize] -= row[f as usize];
                }
                HeldOut::Only(f) => scores[f as usize] += row[f as usize],
                HeldOut::Never => {}
            }
        }
        let t = self.threshold;
        scores.iter_mut().for_each(|s| *s = (*s).clamp(-t, t));
        HeldOutVotes {
            ae: self,
            classes,
            scores,
        }
    }

    /// The shared clauses that vote on `output` with `output`'s own feature
    /// hidden (see [`held_out_scores`](Self::held_out_scores)). For several
    /// outputs of one sample, [`held_out_votes`](Self::held_out_votes)
    /// evaluates the clauses once.
    pub fn held_out_fired_clauses(&self, sample: &EncodedSample, output: usize) -> Vec<usize> {
        fired_for(&self.index().held.classify(&sample.0), output)
    }

    /// Drop the inference map; called by every method that can change the
    /// include bitsets or the valid mask.
    fn reset_literal_index(&mut self) {
        self.literal_index = std::sync::OnceLock::new();
    }

    /// Each output's weighted sum of the clauses in `fired` (unclamped).
    fn sums_of(&self, fired: &[usize], out: &mut [i32]) {
        let nc = self.n_clauses;
        for (o, out_o) in out.iter_mut().enumerate() {
            let ws = &self.weights[o * nc..(o + 1) * nc];
            *out_o = fired.iter().map(|&j| ws[j]).sum();
        }
    }

    fn reconstruct_lit(&self, lit: &[u64]) -> Vec<u8> {
        debug_assert_eq!(lit.len(), self.words);
        let mut sums = vec![0i32; self.n_features];
        self.sums_of(&self.fired_predict(lit), &mut sums);
        sums.into_iter().map(|s| (s > 0) as u8).collect()
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
        if n >= PARALLEL_MIN && use_parallel(self.n_clauses, w) {
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
        debug_assert_eq!(out.len(), self.n_features);
        self.sums_of(&self.fired_predict(&sample.0), out);
        let t = self.threshold;
        out.iter_mut().for_each(|s| *s = (*s).clamp(-t, t));
    }

    /// Return the indices of all shared clauses that fire for `sample`.
    /// The `_output` parameter is accepted for API uniformity but ignored — clauses are shared.
    pub fn fired_clauses(&self, sample: &EncodedSample, _output: usize) -> Vec<usize> {
        self.fired_predict(&sample.0)
    }

    // ---- training helpers ---------------------------------------------------

    /// Update all shared clauses for output `o`.
    ///
    /// `clause_outputs[j]` must already be computed (under the output-`o` feature mask)
    /// before this call.  `update_clause[j]` is the per-sample confidence gate (see
    /// [`Self::confidence_gate`]), shared across every output for this sample.  Clause
    /// dropout and Bernoulli inv/keep masks are drawn from `output_rngs[o]`.
    #[allow(clippy::too_many_arguments)]
    fn update_output_coalesced(
        &mut self,
        o: usize,
        target: u8,
        sum: i32,
        clause_outputs: &[bool],
        update_clause: &[bool],
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

        // Empty means "all active" (checked via `is_empty()` below) — matches the
        // `drop_mask` convention in vanilla_autoencoder.rs so the default (no dropout)
        // path allocates nothing instead of materialising an all-true Vec every call.
        let clause_active: Vec<bool> = if drop_p > 0.0 {
            (0..nc).map(|_| orng.next_f64() >= drop_p).collect()
        } else {
            Vec::new()
        };

        let inv_mask: Vec<u64> = (0..words).map(|_| bmask_word(orng, dig_inv)).collect();
        let keep_mask: Vec<u64> = (0..words).map(|_| bmask_word(orng, dig_keep)).collect();
        let inv_b = expand_bits_to_bytes(&inv_mask, n_literals);
        let keep_b = expand_bits_to_bytes(&keep_mask, n_literals);

        let out_w = &mut weights[o * nc..(o + 1) * nc];

        #[cfg(feature = "parallel")]
        if nc >= DENSE_TRAIN_PARALLEL_MIN {
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
                        let active_j =
                            (clause_active.is_empty() || clause_active[j]) && update_clause[j];
                        apply_one_clause_coalesced(
                            ta_j,
                            inc_j,
                            w,
                            rng,
                            target,
                            p,
                            clause_outputs[j],
                            active_j,
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
                        if active_j
                            && type_iii_update(
                                ta_j,
                                ind_j,
                                cat_j,
                                inc_j,
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
                            )
                        {
                            rebuild_include(ta_j, inc_j, val, words, n_literals, half);
                        }
                    });
            } else {
                ta.par_chunks_mut(n_literals)
                    .zip(include.par_chunks_mut(words))
                    .zip(out_w.par_iter_mut())
                    .zip(rngs.par_iter_mut())
                    .enumerate()
                    .for_each(|(j, (((ta_j, inc_j), w), rng))| {
                        let active_j =
                            (clause_active.is_empty() || clause_active[j]) && update_clause[j];
                        apply_one_clause_coalesced(
                            ta_j,
                            inc_j,
                            w,
                            rng,
                            target,
                            p,
                            clause_outputs[j],
                            active_j,
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
                    });
            }
            return;
        }

        for j in 0..nc {
            let active_j = (clause_active.is_empty() || clause_active[j]) && update_clause[j];
            apply_one_clause_coalesced(
                &mut ta[j * n_literals..(j + 1) * n_literals],
                &mut include[j * words..(j + 1) * words],
                &mut out_w[j],
                &mut rngs[j],
                target,
                p,
                clause_outputs[j],
                active_j,
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
            if type_iii_en
                && active_j
                && type_iii_update(
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
                )
            {
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

    /// Confidence-based clause gate (ported from TMU's `update_clause`): for each shared
    /// clause, average `|weight|` across all outputs, then draw a Bernoulli with
    /// probability `(threshold - avg_abs_weight) / threshold`. A clause with high average
    /// magnitude (confident/specialised across outputs) becomes progressively less likely
    /// to receive further feedback this sample; a fresh, unspecialised clause stays highly
    /// eligible. Writes into `out`, sized `n_clauses`, reused across samples by the caller.
    fn confidence_gate(&mut self, out: &mut [bool]) {
        debug_assert_eq!(out.len(), self.n_clauses);
        let nc = self.n_clauses;
        let n_features = self.n_features;
        let t = self.threshold as f64;
        for (j, slot) in out.iter_mut().enumerate() {
            let mut avg_abs = 0.0f64;
            for o in 0..n_features {
                avg_abs += (self.weights[o * nc + j] as f64).abs();
            }
            avg_abs /= n_features as f64;
            let p_update = (t - avg_abs.min(t)) / t;
            *slot = self.clause_gate_rng.next_f64() < p_update;
        }
    }

    /// Literal-dropout mask for one sample (all active without dropout).
    fn draw_literal_active(&mut self) -> Vec<u64> {
        let words = self.words;
        if self.literal_drop_p > 0.0 {
            let rng = &mut self.literal_rng;
            let dig = &self.dig_lit_active;
            (0..words).map(|_| bmask_word(rng, dig)).collect()
        } else {
            vec![!0u64; words]
        }
    }

    /// The order outputs are trained in this round: `0..n_features`, or a
    /// fresh shuffle (from the model's RNG) when `shuffled`.
    fn output_order(&mut self, shuffled: bool) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.n_features).collect();
        if shuffled {
            for i in (1..order.len()).rev() {
                let k = self.rng.below(i + 1);
                order.swap(i, k);
            }
        }
        order
    }

    /// Train output `o` towards `target` on the sample in `self.literals`
    /// (`lit_b` its bytes), with `o`'s own literals hidden. `lit_active` /
    /// `active_b` are the sample's active literals; they are masked for `o`
    /// and restored before returning.
    #[allow(clippy::too_many_arguments)]
    fn train_output(
        &mut self,
        o: usize,
        target: u8,
        lit_b: &[u8],
        lit_active: &mut [u64],
        active_b: &mut [u8],
        clause_outputs: &mut [bool],
        update_clause: &[bool],
    ) {
        let nc = self.n_clauses;
        let n_features = self.n_features;
        let words = self.words;

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

        // Compute clause fires under this output's feature mask (into the reused buffer).
        {
            let include = self.include.as_slice();
            let valid = self.valid.as_slice();
            let litv = self.literals.as_slice();
            for (j, out) in clause_outputs.iter_mut().enumerate() {
                *out = clause_fire(
                    &include[j * words..(j + 1) * words],
                    litv,
                    valid,
                    words,
                    lit_active,
                );
            }
        }

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

        let lit = std::mem::take(&mut self.literals);
        self.update_output_coalesced(
            o,
            target,
            sum,
            clause_outputs,
            update_clause,
            lit_b,
            active_b,
            &lit,
            lit_active,
        );
        self.literals = lit;

        // Restore feature mask for the next output's iteration.
        lit_active[word_o] = save_word_o;
        lit_active[word_neg_o] = save_word_neg_o;
        active_b[o] = save_ab_o;
        active_b[neg_o] = save_ab_neg_o;
    }

    fn fit_one_lit(&mut self, lit: &[u64]) {
        debug_assert_eq!(lit.len(), self.words);
        self.literals.copy_from_slice(lit);
        let nc = self.n_clauses;
        let n_literals = self.n_literals;

        let lit_b = expand_bits_to_bytes(lit, n_literals);
        let mut lit_active = self.draw_literal_active();
        let mut active_b = expand_bits_to_bytes(&lit_active, n_literals);

        // Reused across every output below.
        let mut clause_outputs = vec![false; nc];

        // Confidence gate is drawn once per sample and shared by every output's
        // update this round (matches TMU's `update_clause` timing).
        let mut update_clause = vec![true; nc];
        self.confidence_gate(&mut update_clause);

        for o in self.output_order(self.shuffle_outputs) {
            let target = ((lit[o / WORD_BITS] >> (o % WORD_BITS)) & 1) as u8;
            self.train_output(
                o,
                target,
                &lit_b,
                &mut lit_active,
                &mut active_b,
                &mut clause_outputs,
                &update_clause,
            );
        }
    }

    /// Train on a single encoded sample (unsupervised — no label required).
    pub fn fit_one(&mut self, sample: &EncodedSample) {
        self.reset_literal_index();
        self.fit_one_lit(&sample.0);
    }

    /// Run one training epoch over an encoded batch, shuffling the order each epoch.
    pub fn fit_epoch(&mut self, batch: &EncodedBatch) {
        self.reset_literal_index();
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

    /// Train as TMU's `TMAutoEncoder.fit(X, number_of_examples)`: every
    /// example trains each output, in a fresh random order, on an example
    /// made for that output.
    ///
    /// For output `o`, the target is drawn first: present with probability
    /// [`output_balancing`](Self::output_balancing) if set, else with bit `o`'s
    /// frequency in `batch`. The example is then a random row of `batch` with
    /// bit `o` equal to the target, or the union of
    /// [`accumulation`](Self::accumulation) such rows. If bit `o` is in every
    /// row or in none, the rows are drawn at random and the target is kept as
    /// drawn (as TMU does).
    ///
    /// [`fit_epoch`](Self::fit_epoch) instead trains every output on the same
    /// row, towards that row's own bits. Here each output sees examples chosen
    /// for it, so a rare feature is trained on rows that have it as often as
    /// its target is drawn present — with `output_balancing(0.5)`, half the
    /// time. The confidence gate is drawn once per example, the literal
    /// dropout mask once per output example.
    pub fn fit_examples(&mut self, batch: &EncodedBatch, number_of_examples: usize) {
        self.reset_literal_index();
        debug_assert_eq!(batch.words, self.words);
        let n = batch.n;
        if n == 0 {
            return;
        }
        let nf = self.n_features;
        let nc = self.n_clauses;
        let words = self.words;
        let n_literals = self.n_literals;
        let data = batch.data.as_slice();
        let bit =
            |row: usize, f: usize| (data[row * words + f / WORD_BITS] >> (f % WORD_BITS)) & 1 == 1;

        // Rows that have each feature (sorted), TMU's CSC columns.
        let mut with: Vec<Vec<u32>> = vec![Vec::new(); nf];
        for row in 0..n {
            for (f, rows) in with.iter_mut().enumerate() {
                if bit(row, f) {
                    rows.push(row as u32);
                }
            }
        }

        let mut clause_outputs = vec![false; nc];
        let mut update_clause = vec![true; nc];
        let mut union = vec![false; nf];
        for _ in 0..number_of_examples {
            let order = self.output_order(true);
            self.confidence_gate(&mut update_clause);
            for o in order {
                let rows = &with[o];
                let p_true = if self.output_balancing > 0.0 {
                    self.output_balancing
                } else {
                    rows.len() as f64 / n as f64
                };
                let target = self.rng.next_f64() <= p_true;

                // The example: the union of `accumulation` rows.
                union.iter_mut().for_each(|b| *b = false);
                let degenerate = rows.is_empty() || rows.len() == n;
                for _ in 0..self.accumulation {
                    let row = if degenerate {
                        self.rng.below(n)
                    } else if target {
                        rows[self.rng.below(rows.len())] as usize
                    } else {
                        loop {
                            let r = self.rng.below(n);
                            if rows.binary_search(&(r as u32)).is_err() {
                                break r;
                            }
                        }
                    };
                    for (f, b) in union.iter_mut().enumerate() {
                        *b |= bit(row, f);
                    }
                }
                self.literals.iter_mut().for_each(|w| *w = 0);
                for (f, &b) in union.iter().enumerate() {
                    let l = if b { f } else { f + nf };
                    self.literals[l / WORD_BITS] |= 1u64 << (l % WORD_BITS);
                }
                let lit_b = expand_bits_to_bytes(&self.literals, n_literals);
                let mut lit_active = self.draw_literal_active();
                let mut active_b = expand_bits_to_bytes(&lit_active, n_literals);
                self.train_output(
                    o,
                    u8::from(target),
                    &lit_b,
                    &mut lit_active,
                    &mut active_b,
                    &mut clause_outputs,
                    &update_clause,
                );
            }
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
        if n >= PARALLEL_MIN && use_parallel(self.n_clauses, w) {
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
        let inc = &self.include[clause * self.words..(clause + 1) * self.words];
        rule_from_include(inc, self.n_literals, self.n_features)
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
            TMCoalescedAutoEncoder::with_config(12, 10, 10, 3.0, 8, true, 42).type_iii_feedback(0.5)
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

    #[test]
    fn start_excluded_leaves_every_clause_empty() {
        let ae = TMCoalescedAutoEncoder::with_config(12, 20, 15, 3.9, 8, true, 42).start_excluded();
        assert!(ae.ta.iter().all(|&t| t == ae.half - 1));
        assert!(ae.include.iter().all(|&w| w == 0));
        // Empty clauses never fire at prediction, so every vote is 0.
        let sample = enc(12).encode_one(&[1; 12]);
        let mut out = vec![1i32; 12];
        ae.reconstruct_scores(&sample, &mut out);
        assert_eq!(out, vec![0; 12]);
    }

    #[test]
    fn shuffled_outputs_change_training_and_still_learn() {
        let xs = make_mirrored(2000, 6, 1);
        let e = enc(12);
        let batch = e.encode_batch(&as_slices(&xs));
        let mut fixed = TMCoalescedAutoEncoder::with_config(12, 20, 15, 3.9, 8, true, 42);
        let mut shuffled =
            TMCoalescedAutoEncoder::with_config(12, 20, 15, 3.9, 8, true, 42).shuffle_outputs(true);
        for _ in 0..20 {
            fixed.fit_epoch(&batch);
            shuffled.fit_epoch(&batch);
        }
        assert_ne!(fixed.ta, shuffled.ta, "the output order should matter");
        let acc = shuffled.reconstruction_accuracy(&batch);
        assert!(acc > 0.80, "shuffled outputs: accuracy {acc:.4}");
    }

    #[test]
    fn fit_examples_learns_like_tmu() {
        let xs = make_mirrored(2000, 6, 1);
        let e = enc(12);
        let batch = e.encode_batch(&as_slices(&xs));
        for (name, ae) in [
            (
                "frequency",
                TMCoalescedAutoEncoder::with_config(12, 20, 15, 3.9, 8, true, 42),
            ),
            (
                "balanced, excluded start",
                TMCoalescedAutoEncoder::with_config(12, 20, 15, 3.9, 8, true, 42)
                    .output_balancing(0.5)
                    .start_excluded(),
            ),
            (
                "accumulation 2",
                TMCoalescedAutoEncoder::with_config(12, 20, 15, 3.9, 8, true, 42).accumulation(2),
            ),
        ] {
            let mut ae = ae;
            for _ in 0..20 {
                ae.fit_examples(&batch, 2000);
            }
            let acc = ae.reconstruction_accuracy(&batch);
            // Two rows' union no longer mirrors exactly, so accumulation
            // learns a looser rule.
            let bar = if name.starts_with("accumulation") {
                0.70
            } else {
                0.80
            };
            assert!(acc > bar, "{name}: accuracy {acc:.4}");
        }
    }

    #[test]
    fn fit_examples_is_deterministic_and_handles_constant_features() {
        // Feature 0 is always on and feature 1 always off: no row has the
        // other value, so examples for them are drawn at random (no endless
        // search for a matching row).
        let mut rng = Rng::new(3);
        let xs: Vec<Vec<u8>> = (0..300)
            .map(|_| {
                let mut v: Vec<u8> = (0..8).map(|_| (rng.next_u64() & 1) as u8).collect();
                v[0] = 1;
                v[1] = 0;
                v
            })
            .collect();
        let batch = enc(8).encode_batch(&as_slices(&xs));
        let train = || {
            let mut ae = TMCoalescedAutoEncoder::with_config(8, 10, 8, 3.0, 8, true, 7)
                .output_balancing(0.5);
            ae.fit_examples(&batch, 300);
            ae
        };
        let (a, b) = (train(), train());
        assert_eq!(a.ta, b.ta);
        assert_eq!(a.weights, b.weights);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn training_settings_are_not_saved() {
        use crate::serial::SaveLoad;
        let ae = TMCoalescedAutoEncoder::with_config(6, 4, 5, 3.0, 8, true, 1)
            .shuffle_outputs(true)
            .output_balancing(0.5)
            .accumulation(3)
            .start_excluded();
        let dir = std::env::temp_dir().join(format!("tmu_cae_settings_{}", std::process::id()));
        let path = dir.with_extension("bin");
        ae.save(&path).unwrap();
        let back = TMCoalescedAutoEncoder::load(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(!back.shuffle_outputs);
        assert_eq!(back.output_balancing, 0.0);
        assert_eq!(back.accumulation, 1);
        // The excluded start is state, so it is kept.
        assert_eq!(back.ta, ae.ta);
    }

    /// `reconstruct_scores` by brute force: every clause, `fire_predict`.
    fn scores_by_scan(ae: &TMCoalescedAutoEncoder, sample: &EncodedSample) -> Vec<i32> {
        let (nc, w) = (ae.n_clauses, ae.words);
        (0..ae.n_features)
            .map(|o| {
                let s: i32 = (0..nc)
                    .filter(|&j| {
                        crate::clause_bank::dense::fire_predict(
                            &ae.include[j * w..(j + 1) * w],
                            &sample.0,
                            &ae.valid,
                            w,
                        )
                    })
                    .map(|j| ae.weights[o * nc + j])
                    .sum();
                s.clamp(-ae.threshold, ae.threshold)
            })
            .collect()
    }

    #[test]
    fn literal_map_scores_match_a_full_scan_across_training() {
        // 70 features: literals span three words.
        let mut rng = Rng::new(4);
        let xs: Vec<Vec<u8>> = (0..400)
            .map(|_| {
                (0..70)
                    .map(|_| rng.next_u64().is_multiple_of(5) as u8)
                    .collect()
            })
            .collect();
        let e = enc(70);
        let batch = e.encode_batch(&as_slices(&xs));
        let samples: Vec<EncodedSample> = xs.iter().take(40).map(|x| e.encode_one(x)).collect();
        let check = |ae: &TMCoalescedAutoEncoder, when: &str| {
            let mut got = vec![0i32; 70];
            for s in &samples {
                ae.reconstruct_scores(s, &mut got);
                assert_eq!(got, scores_by_scan(ae, s), "{when}");
                let recon: Vec<u8> = got.iter().map(|&v| (v > 0) as u8).collect();
                assert_eq!(ae.reconstruct(s), recon, "{when}");
                let fired: Vec<usize> = (0..ae.n_clauses)
                    .filter(|&j| {
                        crate::clause_bank::dense::fire_predict(
                            &ae.include[j * ae.words..(j + 1) * ae.words],
                            &s.0,
                            &ae.valid,
                            ae.words,
                        )
                    })
                    .collect();
                assert_eq!(ae.fired_clauses(s, 0), fired, "{when}");
            }
        };
        let mut ae = TMCoalescedAutoEncoder::with_config(70, 30, 10, 3.0, 8, true, 2);
        check(&ae, "untrained");
        ae.fit_epoch(&batch);
        check(&ae, "after one epoch"); // the map is built here ...
        ae.fit_epoch(&batch);
        check(&ae, "after more training"); // ... and must be rebuilt here
        ae.fit_examples(&batch, 100);
        check(&ae, "after fit_examples");
        let ae = ae.start_excluded();
        check(&ae, "after start_excluded");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn literal_map_scores_match_after_save_load() {
        use crate::serial::SaveLoad;
        let xs = make_mirrored(300, 6, 3);
        let e = enc(12);
        let batch = e.encode_batch(&as_slices(&xs));
        let mut ae = TMCoalescedAutoEncoder::with_config(12, 20, 15, 3.9, 8, true, 42);
        ae.fit_epoch(&batch);
        let s = e.encode_one(&xs[0]);
        let mut before = vec![0i32; 12];
        ae.reconstruct_scores(&s, &mut before); // builds the map
        let path = std::env::temp_dir().join(format!("tmu_cae_map_{}.bin", std::process::id()));
        ae.save(&path).unwrap();
        let back = TMCoalescedAutoEncoder::load(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let mut after = vec![0i32; 12];
        back.reconstruct_scores(&s, &mut after);
        assert_eq!(before, after);
        assert_eq!(after, scores_by_scan(&back, &s));
    }

    #[test]
    fn held_out_scores_hide_each_outputs_own_feature() {
        let mut rng = Rng::new(9);
        let xs: Vec<Vec<u8>> = (0..300)
            .map(|_| {
                (0..70)
                    .map(|_| rng.next_u64().is_multiple_of(5) as u8)
                    .collect()
            })
            .collect();
        let e = enc(70);
        let batch = e.encode_batch(&as_slices(&xs));
        let mut ae = TMCoalescedAutoEncoder::with_config(70, 40, 8, 3.0, 8, true, 3);
        for round in 0..2 {
            ae.fit_epoch(&batch); // the second round must rebuild the index
            let (nc, w, nf) = (ae.n_clauses, ae.words, ae.n_features);
            let mut got = vec![0i32; nf];
            for x in xs.iter().take(30) {
                let s = e.encode_one(x);
                ae.held_out_scores(&s, &mut got);
                for (o, &vote) in got.iter().enumerate() {
                    let fired: Vec<usize> = (0..nc)
                        .filter(|&j| {
                            crate::clause_bank::dense::fires_hidden(
                                &ae.include,
                                &ae.valid,
                                w,
                                nf,
                                &s.0,
                                j,
                                o,
                            )
                        })
                        .collect();
                    let sum: i32 = fired.iter().map(|&j| ae.weights[o * nc + j]).sum();
                    assert_eq!(
                        vote,
                        sum.clamp(-ae.threshold, ae.threshold),
                        "round {round}, output {o}"
                    );
                    assert_eq!(
                        ae.held_out_fired_clauses(&s, o),
                        fired,
                        "round {round}, output {o}"
                    );
                    let votes = ae.held_out_votes(&s);
                    assert_eq!(votes.scores()[o], vote);
                    assert_eq!(votes.fired_clauses(o), fired);
                    for against in [true, false] {
                        // The first clause with the most extreme weight that way.
                        let mut want: Option<(usize, i32)> = None;
                        for &j in &fired {
                            let w = ae.weights[o * nc + j];
                            let wanted = if against { w < 0 } else { w > 0 };
                            let better =
                                want.is_none_or(|(_, b)| if against { w < b } else { w > b });
                            if wanted && better {
                                want = Some((j, w));
                            }
                        }
                        assert_eq!(votes.strongest_clause(o, against), want);
                    }
                }
            }
        }
    }

    #[test]
    fn held_out_scores_ignore_a_clause_on_the_hidden_bit_alone() {
        // A clause including only literal 0 echoes bit 0 when reconstructing;
        // held out, it is empty and says nothing about output 0.
        let mut ae = TMCoalescedAutoEncoder::with_config(3, 1, 5, 3.0, 8, true, 1).start_excluded();
        ae.ta[0] = ae.half;
        let (n_lit, words, half) = (ae.n_literals, ae.words, ae.half);
        let valid = ae.valid.clone();
        rebuild_include(
            &ae.ta[..n_lit],
            &mut ae.include[..words],
            &valid,
            words,
            n_lit,
            half,
        );
        ae.weights[..3].copy_from_slice(&[3, 3, 3]);
        ae.reset_literal_index();
        let sample = enc(3).encode_one(&[1, 1, 0]);
        let (mut plain, mut held) = (vec![0i32; 3], vec![0i32; 3]);
        ae.reconstruct_scores(&sample, &mut plain);
        ae.held_out_scores(&sample, &mut held);
        assert_eq!(plain, vec![3, 3, 3]);
        assert_eq!(held, vec![0, 3, 3]);
        // Bit 0 off: the clause fails, except for output 0, where it is empty.
        ae.held_out_scores(&enc(3).encode_one(&[0, 1, 0]), &mut held);
        assert_eq!(held, vec![0, 0, 0]);
    }
}
