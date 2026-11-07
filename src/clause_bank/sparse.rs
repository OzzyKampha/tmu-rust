//! Sparse clause bank with absorbing actions.
//!
//! Mirrors TMU's `clause_bank/clause_bank_sparse.py` / `ClauseBankSparse`.
//!
//! Where the [`dense`](crate::clause_bank::dense) bank stores a full
//! `2 * n_features` TA counter array per clause and sweeps *every* literal on
//! each fire-check, the sparse bank stores, per clause, three **index lists**:
//!
//! * `included`   — literals currently in the conjunction (each with a TA state);
//! * `excluded`   — candidate literals parked near the inclusion boundary;
//! * `unallocated` — literals that have been *absorbed out* (permanently dropped
//!   from candidacy) or are otherwise not tracked.
//!
//! A clause fires iff every literal in its (usually tiny) `included` list is
//! present in the input, so per-clause work scales with the number of *active*
//! literals rather than the literal count.  **Absorbing actions** are what make
//! the lists shrink: with TMU's `absorbing = k` (`k >= 0`), once a candidate
//! literal's TA state falls to `k` or below it is dropped from the clause for
//! good (counted in `absorbed`). `k = -1` (the default, as in TMU) turns this
//! off: candidate states then stop at 1. With `literal_sampling < 1` each
//! clause starts with only a random share of the literals as candidates (the
//! rest wait in `unallocated`), and `literal_insertion_state` refills an
//! absorbed slot from there. As in TMU, included literals are not locked at
//! the maximum state.
//!
//! ## Initialisation policy
//!
//! Every clause starts with all literals in `excluded` at the *least-forgotten*
//! exclude state (`half - 1`, i.e. just below the inclusion threshold) and empty
//! `included` / `unallocated`.  This matches the dense bank's "all literals start
//! just-excluded" initialisation, so the two banks begin from the same logical
//! state.
//!
//! ## Scope
//!
//! Consumed by [`TMSparseClassifier`](crate::models::TMSparseClassifier). Per-clause
//! feedback lives in methods on [`SparseClause`] so the model can run the per-class
//! clause loop in parallel (Rayon, `--features parallel`) over disjoint clause state.
//! AVX2 is intentionally not used: the per-clause excluded-list scan is dominated by
//! per-index bit gathers, scalar RNG draws, and `swap_remove` mutation, which don't
//! vectorise (upstream `cair/tmu`'s sparse C bank is also scalar). Type III feedback
//! is likewise unsupported here (its indicator array conflicts with literal removal).

use crate::clause_bank::dense::WORD_BITS;
use crate::rng::Rng;

/// `absorbing` value that turns absorbing off (TMU's default).
pub(crate) const NO_ABSORBING: i32 = -1;

#[cfg(feature = "serde")]
fn no_absorbing() -> i32 {
    NO_ABSORBING
}

#[cfg(feature = "serde")]
fn one_u8() -> u8 {
    1
}

#[cfg(feature = "serde")]
fn one_f64() -> f64 {
    1.0
}

#[cfg(feature = "serde")]
fn yes() -> bool {
    true
}

/// One sparse clause: literal indices partitioned into included / excluded, each
/// carrying its TA automaton state; everything else lives in `unallocated`.
///
/// The `*_state` vectors run parallel to their index vectors (`included_state[i]`
/// is the state of literal `included[i]`), which keeps the index vectors as plain
/// `u32` for fast firing scans and lets absorbing removal move an index and its
/// state together with an O(1) `swap_remove`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct SparseClause {
    /// Literal indices currently in the conjunction.
    pub(crate) included: Vec<u32>,
    /// TA state for each included literal (parallel to `included`).
    pub(crate) included_state: Vec<u8>,
    /// Candidate literal indices parked near the inclusion boundary.
    pub(crate) excluded: Vec<u32>,
    /// TA state for each excluded literal (parallel to `excluded`).
    pub(crate) excluded_state: Vec<u8>,
    /// Literals not (yet) candidates of this clause: those left out by TMU's
    /// `literal_sampling`, handed out by `literal_insertion_state` when a
    /// candidate is absorbed. Popped from the end.
    pub(crate) unallocated: Vec<u32>,
    /// Literals absorbed out of this clause (dropped for good, as TMU).
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) absorbed: u32,
}

/// TMU's sparse-bank training options (see the classifier's builders).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SparseOpts {
    /// `absorbing`: absorb candidates whose state falls to this or below; -1 off.
    pub(crate) absorbing: i32,
    /// `feedback_rate_excluded_literals`: candidates are updated with
    /// probability 1/rate per feedback, by `rate` states at a time.
    pub(crate) feedback_rate: u8,
    /// `literal_insertion_state`: an absorbed candidate is replaced by an
    /// unallocated literal at this state; -1 = not replaced.
    pub(crate) insertion_state: i32,
}

/// A bank of [`SparseClause`]s sharing the same literal geometry and TA bounds.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct SparseClauseBank {
    pub(crate) n_clauses: usize,
    pub(crate) n_literals: usize,
    /// Inclusion threshold: a literal with `state >= half` is included.
    pub(crate) half: u8,
    /// Absorbing include state (maximum TA counter value).
    pub(crate) max_state: u8,
    /// Absorbing exclude threshold, as TMU's `absorbing`: an excluded literal
    /// whose state falls to this value or below is removed from the clause's
    /// candidates for good. -1 (default) = off. A training setting, not saved.
    #[cfg_attr(feature = "serde", serde(skip, default = "no_absorbing"))]
    pub(crate) absorbing: i32,
    /// TMU's `feedback_rate_excluded_literals` (>= 1). Training setting.
    #[cfg_attr(feature = "serde", serde(skip, default = "one_u8"))]
    pub(crate) feedback_rate: u8,
    /// TMU's `literal_insertion_state` (-1 = off). Training setting.
    #[cfg_attr(feature = "serde", serde(skip, default = "no_absorbing"))]
    pub(crate) insertion_state: i32,
    /// TMU's `literal_sampling`: the fraction of literals each clause starts
    /// with as candidates. Applied when set (re-initialising the clauses).
    #[cfg_attr(feature = "serde", serde(skip, default = "one_f64"))]
    pub(crate) literal_sampling: f64,
    /// TMU's `feature_negation`: whether negated literals may be used. Saved
    /// with the model, so `grow` keeps it.
    #[cfg_attr(feature = "serde", serde(default = "yes"))]
    pub(crate) feature_negation: bool,
    pub(crate) clauses: Vec<SparseClause>,
}

/// Returns `true` if literal `l` is present in the packed input `lit`.
///
/// Uses the exact [`pack`](crate::clause_bank::dense::pack) layout: positive
/// literal `i` → bit `i`; negated literal `i` → bit `n_features + i`.
#[inline(always)]
fn lit_present(lit: &[u64], l: u32) -> bool {
    let l = l as usize;
    (lit[l / WORD_BITS] >> (l % WORD_BITS)) & 1 != 0
}

/// Returns `true` if literal `l` is active under the dropout mask `lit_active`.
#[inline(always)]
fn lit_active(lit_active: &[u64], l: u32) -> bool {
    let l = l as usize;
    (lit_active[l / WORD_BITS] >> (l % WORD_BITS)) & 1 != 0
}

impl SparseClause {
    /// Predict-semantics fire check: an **empty** clause returns `false`.
    #[inline]
    fn fire_predict(&self, lit: &[u64]) -> bool {
        if self.included.is_empty() {
            return false;
        }
        self.included.iter().all(|&l| lit_present(lit, l))
    }

    /// Train-semantics fire check: an **empty** clause returns `true`.
    #[inline]
    pub(crate) fn fire_train(&self, lit: &[u64], lit_active_mask: &[u64]) -> bool {
        self.included
            .iter()
            .all(|&l| !lit_active(lit_active_mask, l) || lit_present(lit, l))
    }

    /// Number of included literals (for the `max_included` gate).
    #[inline]
    pub(crate) fn n_included(&self) -> usize {
        self.included.len()
    }

    /// Apply Type I feedback to this clause, as TMU's `cbs_type_i_feedback`.
    /// See [`SparseClauseBank::type_i`].
    ///
    /// Type Ia (`fired_under_limit`): included literals present in the input
    /// are rewarded, absent ones decremented with probability 1/s; then (with
    /// probability 1/rate) candidates present are rewarded by `rate` and
    /// promoted at `half`, absent ones above `rate` decremented by `rate` with
    /// probability 1/s and absorbed at or below `absorbing`.
    /// Type Ib: (with probability 1/rate) every active candidate above `rate` is
    /// decremented by `rate` with probability 1/s (and absorbed as above), then
    /// every active included literal is decremented with probability 1/s.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn type_i(
        &mut self,
        lit: &[u64],
        lit_active_mask: &[u64],
        fired_under_limit: bool,
        boost: bool,
        rng: &mut Rng,
        inv_p: f64,
        keep_p: f64,
        max_included: usize,
        half: u8,
        max_state: u8,
        opts: SparseOpts,
    ) {
        let rate = opts.feedback_rate.max(1);
        let inv_rate = 1.0 / f64::from(rate);
        // TMU draws this gate every time; with rate 1 it always passes, so the
        // draw is skipped (keeping rate-1 training identical to before).
        let candidates_gate = |rng: &mut Rng| rate == 1 || rng.next_f64() <= inv_rate;
        if fired_under_limit {
            self.decrement_or_reward_included(
                lit,
                lit_active_mask,
                true,
                boost,
                rng,
                inv_p,
                keep_p,
                half,
                max_state,
            );
            if candidates_gate(rng) {
                let mut i = 0;
                while i < self.excluded.len() {
                    let l = self.excluded[i];
                    if !lit_active(lit_active_mask, l) {
                        i += 1;
                        continue;
                    }
                    let state = self.excluded_state[i];
                    if lit_present(lit, l) {
                        if self.included.len() < max_included && (boost || rng.next_f64() <= keep_p)
                        {
                            let new_state = state.saturating_add(rate).min(max_state);
                            if new_state >= half {
                                promote_to_included(self, i, new_state);
                                continue;
                            }
                            self.excluded_state[i] = new_state;
                        }
                    } else if state > rate
                        && rng.next_f64() <= inv_p
                        && self.lower_candidate(i, state - rate, opts)
                    {
                        continue;
                    }
                    i += 1;
                }
            }
        } else {
            if candidates_gate(rng) {
                let mut i = 0;
                while i < self.excluded.len() {
                    let l = self.excluded[i];
                    let state = self.excluded_state[i];
                    if lit_active(lit_active_mask, l)
                        && state > rate
                        && rng.next_f64() <= inv_p
                        && self.lower_candidate(i, state - rate, opts)
                    {
                        continue;
                    }
                    i += 1;
                }
            }
            self.decrement_or_reward_included(
                lit,
                lit_active_mask,
                false,
                boost,
                rng,
                inv_p,
                keep_p,
                half,
                max_state,
            );
        }
    }

    /// Included literals under Type Ia (`reward`: present ones move up, absent
    /// ones down with probability 1/s) or Type Ib (every active one down with
    /// probability 1/s). A literal falling below `half` becomes a candidate.
    #[allow(clippy::too_many_arguments)]
    fn decrement_or_reward_included(
        &mut self,
        lit: &[u64],
        lit_active_mask: &[u64],
        reward: bool,
        boost: bool,
        rng: &mut Rng,
        inv_p: f64,
        keep_p: f64,
        half: u8,
        max_state: u8,
    ) {
        // Scan by index; demotions swap-remove, so do NOT advance `i` after one.
        let mut i = 0;
        while i < self.included.len() {
            let l = self.included[i];
            if !lit_active(lit_active_mask, l) {
                i += 1;
                continue;
            }
            let state = self.included_state[i];
            if reward && lit_present(lit, l) {
                if state < max_state && (boost || rng.next_f64() <= keep_p) {
                    self.included_state[i] = state + 1;
                }
            } else if rng.next_f64() <= inv_p {
                let new_state = state - 1;
                if new_state < half {
                    demote_to_excluded(self, i, new_state);
                    continue;
                }
                self.included_state[i] = new_state;
            }
            i += 1;
        }
    }

    /// Set candidate `i` to `new_state`, absorbing it if at or below the
    /// threshold. Returns `true` when slot `i` now holds a literal that has not
    /// been visited this pass (it was swap-removed), so the caller re-visits it.
    fn lower_candidate(&mut self, i: usize, new_state: u8, opts: SparseOpts) -> bool {
        if i32::from(new_state) > opts.absorbing {
            self.excluded_state[i] = new_state;
            return false;
        }
        self.absorbed += 1;
        match (opts.insertion_state >= 0, self.unallocated.pop()) {
            // TMU's literal insertion: an unallocated literal takes the slot.
            (true, Some(l)) => {
                self.excluded[i] = l;
                self.excluded_state[i] = opts.insertion_state as u8;
                false
            }
            (_, popped) => {
                if let Some(l) = popped {
                    self.unallocated.push(l);
                }
                self.excluded.swap_remove(i);
                self.excluded_state.swap_remove(i);
                true
            }
        }
    }

    /// Apply Type II feedback to this clause. See [`SparseClauseBank::type_ii`].
    pub(crate) fn type_ii(
        &mut self,
        lit: &[u64],
        lit_active_mask: &[u64],
        half: u8,
        max_state: u8,
        rng: &mut Rng,
        feedback_rate: u8,
    ) {
        // TMU: candidates take Type II with probability 1/rate, by `rate`.
        let rate = feedback_rate.max(1);
        if rate > 1 && rng.next_f64() > 1.0 / f64::from(rate) {
            return;
        }
        let mut i = 0;
        while i < self.excluded.len() {
            let l = self.excluded[i];
            let active = lit_active(lit_active_mask, l);
            let state = self.excluded_state[i];
            // Absent and active (as TMU: no state guard — excluded states
            // never fall below 1, and absorbed literals have left the pool).
            if active && !lit_present(lit, l) {
                let new_state = state.saturating_add(rate).min(max_state);
                if new_state >= half {
                    promote_to_included(self, i, new_state);
                    continue;
                } else {
                    self.excluded_state[i] = new_state;
                }
            }
            i += 1;
        }
    }
}

impl SparseClauseBank {
    /// Create a bank of `n_clauses` clauses over `n_literals` literals.
    ///
    /// `state_bits` (2..=8) sets the TA counter precision exactly as in the dense
    /// bank: `half = 1 << (state_bits - 1)`, `max_state = (1 << state_bits) - 1`.
    /// `seed` is accepted for API symmetry with the dense bank but the deterministic
    /// initialisation below does not consume it (per-clause RNGs live in the model).
    pub(crate) fn new(n_clauses: usize, n_literals: usize, state_bits: usize, _seed: u64) -> Self {
        let half = 1u8 << (state_bits - 1);
        let max_state = ((1u16 << state_bits) - 1) as u8;
        let least_forgotten = half - 1;

        // Every clause starts with all literals just-excluded (state = half - 1),
        // matching the dense bank's initialisation.
        let proto = SparseClause {
            included: Vec::new(),
            included_state: Vec::new(),
            excluded: (0..n_literals as u32).collect(),
            excluded_state: vec![least_forgotten; n_literals],
            unallocated: Vec::new(),
            absorbed: 0,
        };
        let clauses = vec![proto; n_clauses];

        SparseClauseBank {
            absorbing: NO_ABSORBING,
            feedback_rate: 1,
            insertion_state: NO_ABSORBING,
            literal_sampling: 1.0,
            feature_negation: true,
            n_clauses,
            n_literals,
            half,
            max_state,
            clauses,
        }
    }

    /// The training options as one value for the per-clause updates.
    pub(crate) fn opts(&self) -> SparseOpts {
        SparseOpts {
            absorbing: self.absorbing,
            feedback_rate: self.feedback_rate,
            insertion_state: self.insertion_state,
        }
    }

    /// Re-initialise every clause from `literal_sampling` and
    /// `feature_negation`, as TMU's ClauseBankSparse does at construction: each
    /// clause gets a random `literal_sampling` share of the usable literals as
    /// candidates at the least-forgotten state; the rest wait in `unallocated`.
    /// Discards anything learned, so it is for configuring a fresh model.
    pub(crate) fn reinit(&mut self) {
        let n_features = self.n_literals / 2;
        let usable: Vec<u32> = if self.feature_negation {
            (0..self.n_literals as u32).collect()
        } else {
            (0..n_features as u32).collect()
        };
        let least_forgotten = self.half - 1;
        let k = ((usable.len() as f64) * self.literal_sampling) as usize;
        for (j, c) in self.clauses.iter_mut().enumerate() {
            let mut order = usable.clone();
            if k < order.len() {
                // Per-clause shuffle, deterministic per clause index.
                let mut rng = Rng::new(
                    0x5350_4152_5345_u64 ^ (j as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                );
                for i in (1..order.len()).rev() {
                    let r = (rng.next_u64() % (i as u64 + 1)) as usize;
                    order.swap(i, r);
                }
            }
            let rest: Vec<u32> = order[k.min(order.len())..].iter().rev().copied().collect();
            order.truncate(k);
            c.included.clear();
            c.included_state.clear();
            c.excluded_state = vec![least_forgotten; order.len()];
            c.excluded = order;
            // Popped from the end, so the next sampled-out literal comes first.
            c.unallocated = rest;
            c.absorbed = 0;
        }
    }

    // ---- growing ---------------------------------------------------------

    /// Grow the bank to `new_n_literals`, preserving every learned clause.
    ///
    /// The literal layout is `[positives 0..n_features | negateds n_features..2*n_features]`
    /// (see [`pack`](crate::clause_bank::dense::pack)), so growing `n_features`
    /// shifts the negated block up. Per clause: every stored index `>= old_n_features`
    /// (a negated literal) is remapped by `+ (new_n_features - old_n_features)` in all
    /// three pools, and the new positive/negated literals are appended to `excluded`
    /// at the least-forgotten state (`half - 1`), exactly matching [`SparseClauseBank::new`].
    ///
    /// Because new literals enter as just-excluded candidates (never `included`),
    /// firing on inputs whose new features are all 0 is unchanged — the same
    /// zero-forgetting guarantee as the dense bank.
    pub(crate) fn grow(&mut self, new_n_literals: usize) {
        debug_assert!(new_n_literals >= self.n_literals);
        debug_assert_eq!(new_n_literals % 2, 0);
        if new_n_literals == self.n_literals {
            return;
        }
        let old_nf = (self.n_literals / 2) as u32;
        let new_nf = (new_n_literals / 2) as u32;
        let shift = new_nf - old_nf;
        let least_forgotten = self.half - 1;
        let n_new = new_n_literals - self.n_literals;

        for c in &mut self.clauses {
            // Remap negated-literal indices (positives, being < old_nf, are untouched).
            let remap = |l: &mut u32| {
                if *l >= old_nf {
                    *l += shift;
                }
            };
            c.included.iter_mut().for_each(remap);
            c.excluded.iter_mut().for_each(remap);
            c.unallocated.iter_mut().for_each(remap);

            // Append the new literals as just-excluded candidates: new positives
            // [old_nf, new_nf) and new negateds [new_nf + old_nf, 2*new_nf).
            c.excluded.reserve(n_new);
            c.excluded_state.reserve(n_new);
            for f in old_nf..new_nf {
                c.excluded.push(f);
                c.excluded_state.push(least_forgotten);
            }
            if self.feature_negation {
                for l in (new_nf + old_nf)..(new_n_literals as u32) {
                    c.excluded.push(l);
                    c.excluded_state.push(least_forgotten);
                }
            }
        }

        self.n_literals = new_n_literals;
    }

    // ---- parallel access -------------------------------------------------

    /// Mutable slice of all clauses, for clause-parallel feedback (Rayon).
    pub(crate) fn clauses_mut(&mut self) -> &mut [SparseClause] {
        &mut self.clauses
    }

    /// `(half, max_state)` — the inclusion threshold and the maximum state,
    /// needed by per-clause feedback when operating on a bare [`SparseClause`].
    pub(crate) fn dims(&self) -> (u8, u8) {
        (self.half, self.max_state)
    }

    // ---- firing ----------------------------------------------------------

    /// Predict-semantics fire check for clause `cj`: an **empty** clause returns
    /// `false`.  Mirrors dense [`fire_predict`](crate::clause_bank::dense::fire_predict).
    pub(crate) fn fire_predict(&self, cj: usize, lit: &[u64]) -> bool {
        self.clauses[cj].fire_predict(lit)
    }

    /// Train-semantics fire check for clause `cj`: an **empty** clause returns
    /// `true` (so it still receives Type Ib feedback).  Mirrors dense
    /// [`clause_fire`](crate::clause_bank::dense::clause_fire); an included literal
    /// inactive under `lit_active` is skipped.
    pub(crate) fn fire_train(&self, cj: usize, lit: &[u64], lit_active_mask: &[u64]) -> bool {
        self.clauses[cj].fire_train(lit, lit_active_mask)
    }

    // ---- feedback --------------------------------------------------------

    /// Apply Type I feedback to clause `cj`.
    ///
    /// `fired_under_limit` is `true` for **Type Ia** (the clause fired *and* is
    /// under the include limit) and `false` for **Type Ib**.  `boost` forces
    /// inclusion increments (skips the stochastic keep draw), mirroring
    /// `boost_true_positive_feedback`.  `inv_p` is the decrement probability
    /// (`1/s`) and `keep_p` the increment-boost probability (`(s-1)/s`).
    /// `max_included` caps the number of included literals (promotions are gated
    /// on the live included count).
    ///
    /// Test-only convenience wrapper; production training calls
    /// [`SparseClause::type_i`] directly on a `&mut SparseClause` (see
    /// `clauses_mut`) so the per-clause loop can run in parallel.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn type_i(
        &mut self,
        cj: usize,
        lit: &[u64],
        lit_active_mask: &[u64],
        fired_under_limit: bool,
        boost: bool,
        rng: &mut Rng,
        inv_p: f64,
        keep_p: f64,
        max_included: usize,
    ) {
        let (half, max_state, opts) = (self.half, self.max_state, self.opts());
        self.clauses[cj].type_i(
            lit,
            lit_active_mask,
            fired_under_limit,
            boost,
            rng,
            inv_p,
            keep_p,
            max_included,
            half,
            max_state,
            opts,
        );
    }

    /// Apply Type II feedback to clause `cj` (called only when the clause fired on
    /// a negative-class sample).
    ///
    /// For each excluded literal that is absent (and active) and above the
    /// absorbing floor, increment its state by 1; if it reaches `half`, promote it
    /// to the included pool (making the clause harder to fire on negatives).
    /// Included literals are never touched, mirroring dense Type II.
    ///
    /// Test-only convenience wrapper; see [`SparseClauseBank::type_i`].
    #[cfg(test)]
    pub(crate) fn type_ii(&mut self, cj: usize, lit: &[u64], lit_active_mask: &[u64]) {
        let (half, max_state, rate) = (self.half, self.max_state, self.feedback_rate);
        let mut rng = Rng::new(0);
        self.clauses[cj].type_ii(lit, lit_active_mask, half, max_state, &mut rng, rate);
    }

    // ---- introspection ---------------------------------------------------

    /// The included literal indices of clause `cj` (the learned conjunction).
    pub(crate) fn included_literals(&self, cj: usize) -> &[u32] {
        &self.clauses[cj].included
    }

    /// Number of included literals in clause `cj`.
    ///
    /// Test-only; production code uses [`SparseClause::n_included`] on a bare clause.
    #[cfg(test)]
    pub(crate) fn n_included(&self, cj: usize) -> usize {
        self.clauses[cj].included.len()
    }

    /// `(at_max, total_tracked)` counts of literals at the absorbing include state
    /// across the whole bank, where `total_tracked = included + excluded` literals
    /// (i.e. excludes already-absorbed `unallocated` literals).
    pub(crate) fn count_absorbing_include(&self) -> (u64, u64) {
        let mut at_max = 0u64;
        let mut total = 0u64;
        for c in &self.clauses {
            total += (c.included.len() + c.excluded.len()) as u64;
            at_max += c
                .included_state
                .iter()
                .filter(|&&s| s == self.max_state)
                .count() as u64;
        }
        (at_max, total)
    }

    /// Total number of literals that have been absorbed out (the size of every
    /// clause's `unallocated` pool, which under the v1 init policy only grows via
    /// absorbing removal).
    pub(crate) fn count_absorbed_exclude(&self) -> u64 {
        self.clauses.iter().map(|c| u64::from(c.absorbed)).sum()
    }

    /// Total number of (clause, literal) slots, for fraction denominators.
    pub(crate) fn total_literals(&self) -> u64 {
        (self.n_clauses * self.n_literals) as u64
    }
}

/// Move included literal at `idx` to the excluded pool with state `state`.
#[inline]
fn demote_to_excluded(c: &mut SparseClause, idx: usize, state: u8) {
    let l = c.included.swap_remove(idx);
    c.included_state.swap_remove(idx);
    c.excluded.push(l);
    c.excluded_state.push(state);
}

impl SparseClauseBank {
    /// Include literal `l` in clause `cj` at `state` (at least the inclusion
    /// threshold), wherever it was: a candidate, unallocated, already included
    /// (its state is raised), or absorbed (it comes back). Every literal stays
    /// accounted for exactly once.
    pub(crate) fn include_literal(&mut self, cj: usize, l: u32, state: u8) {
        let state = state.max(self.half);
        let c = &mut self.clauses[cj];
        if let Some(i) = c.included.iter().position(|&x| x == l) {
            c.included_state[i] = c.included_state[i].max(state);
            return;
        }
        if let Some(i) = c.excluded.iter().position(|&x| x == l) {
            promote_to_included(c, i, state);
            return;
        }
        if let Some(i) = c.unallocated.iter().position(|&x| x == l) {
            c.unallocated.swap_remove(i);
        } else {
            c.absorbed = c.absorbed.saturating_sub(1);
        }
        c.included.push(l);
        c.included_state.push(state);
    }
}

/// Move excluded literal at `idx` to the included pool with state `state`.
#[inline]
fn promote_to_included(c: &mut SparseClause, idx: usize, state: u8) {
    let l = c.excluded.swap_remove(idx);
    c.excluded_state.swap_remove(idx);
    c.included.push(l);
    c.included_state.push(state);
}

#[cfg(test)]
mod tests {
    use super::*;

    // 4 features → 8 literals; state_bits = 8 → half = 128, max_state = 255.
    fn bank(n_clauses: usize, n_features: usize) -> SparseClauseBank {
        SparseClauseBank::new(n_clauses, 2 * n_features, 8, 1)
    }

    /// Pack a 0/1 feature vector into the literal bit layout (pos i, neg n_feat+i).
    fn pack_feats(feats: &[u8]) -> Vec<u64> {
        let n = feats.len();
        let words = (2 * n).div_ceil(WORD_BITS);
        let mut out = vec![0u64; words];
        for (i, &f) in feats.iter().enumerate() {
            if f != 0 {
                out[i / WORD_BITS] |= 1u64 << (i % WORD_BITS);
            } else {
                let j = n + i;
                out[j / WORD_BITS] |= 1u64 << (j % WORD_BITS);
            }
        }
        out
    }

    fn all_active(words: usize) -> Vec<u64> {
        vec![!0u64; words]
    }

    /// Manually force a literal into the included pool at `state`.
    fn force_include(b: &mut SparseClauseBank, cj: usize, l: u32, state: u8) {
        let c = &mut b.clauses[cj];
        if let Some(pos) = c.excluded.iter().position(|&x| x == l) {
            c.excluded.swap_remove(pos);
            c.excluded_state.swap_remove(pos);
        }
        c.included.push(l);
        c.included_state.push(state);
    }

    #[test]
    fn fire_predict_empty_is_false() {
        let b = bank(1, 4);
        let lit = pack_feats(&[1, 0, 1, 0]);
        // No literals included yet → predict fire is false.
        assert!(!b.fire_predict(0, &lit));
    }

    #[test]
    fn fire_train_empty_is_true() {
        let b = bank(1, 4);
        let lit = pack_feats(&[1, 0, 1, 0]);
        let act = all_active(lit.len());
        // Empty include set → train fire is true (so it gets Type Ib feedback).
        assert!(b.fire_train(0, &lit, &act));
    }

    #[test]
    fn fire_predict_all_included_present_true_else_false() {
        let mut b = bank(1, 4);
        // Include literal 0 (feature 0 positive) and literal 5 (feature 1 negated).
        force_include(&mut b, 0, 0, 200);
        force_include(&mut b, 0, 4 + 1, 200);
        // feats: f0=1 (lit0 present), f1=0 (lit5 = neg f1 present)
        let good = pack_feats(&[1, 0, 0, 0]);
        assert!(b.fire_predict(0, &good));
        // f1=1 → negated literal 5 absent → clause must not fire.
        let bad = pack_feats(&[1, 1, 0, 0]);
        assert!(!b.fire_predict(0, &bad));
    }

    #[test]
    fn type_i_promotes_excluded_present_to_included() {
        let mut b = bank(1, 4);
        // Bring literal 0's excluded state to just below half so one boosted
        // increment promotes it.
        let half = b.half;
        let c = &mut b.clauses[0];
        let pos = c.excluded.iter().position(|&x| x == 0).unwrap();
        c.excluded_state[pos] = half - 1;

        let lit = pack_feats(&[1, 0, 0, 0]); // feature 0 present
        let act = all_active(lit.len());
        let mut rng = Rng::new(7);
        // fired_under_limit = true, boost = true → deterministic increment.
        b.type_i(0, &lit, &act, true, true, &mut rng, 1.0, 1.0, usize::MAX);
        assert!(
            b.included_literals(0).contains(&0),
            "present literal at boundary should be promoted to included"
        );
    }

    #[test]
    fn type_i_demotes_included_absent_below_half() {
        let mut b = bank(1, 4);
        // Include literal 0 exactly at half; one decrement drops it below.
        let half = b.half;
        force_include(&mut b, 0, 0, half);
        let lit = pack_feats(&[0, 0, 0, 0]); // feature 0 ABSENT → literal 0 absent
        let act = all_active(lit.len());
        let mut rng = Rng::new(3);
        // inv_p = 1.0 → deterministic decrement.
        b.type_i(0, &lit, &act, true, true, &mut rng, 1.0, 1.0, usize::MAX);
        assert!(
            !b.included_literals(0).contains(&0),
            "included literal that fell below half should be demoted"
        );
    }

    #[test]
    fn absorbing_exclude_removes_from_pool_permanently() {
        let mut b = bank(1, 4);
        b.absorbing = 1;
        // Drive excluded literal 0 to the threshold: from state 2, one
        // decrement reaches 1 = `absorbing` and absorbs it out.
        let c = &mut b.clauses[0];
        let pos = c.excluded.iter().position(|&x| x == 0).unwrap();
        c.excluded_state[pos] = 2;

        let lit = pack_feats(&[0, 0, 0, 0]); // literal 0 absent
        let act = all_active(lit.len());
        let mut rng = Rng::new(11);

        let before = b.clauses[0].excluded.len();
        b.type_i(0, &lit, &act, false, false, &mut rng, 1.0, 1.0, usize::MAX);
        let after = b.clauses[0].excluded.len();
        assert_eq!(after, before - 1, "literal should leave the excluded pool");
        assert_eq!(b.clauses[0].absorbed, 1, "counted as absorbed");

        // Dropped for good, as TMU: never in any pool again.
        for _ in 0..1000 {
            b.type_i(0, &lit, &act, false, false, &mut rng, 1.0, 1.0, usize::MAX);
        }
        let c = &b.clauses[0];
        assert!(
            !c.excluded.contains(&0) && !c.included.contains(&0) && !c.unallocated.contains(&0)
        );
    }

    #[test]
    fn absorbing_off_never_removes_and_floors_at_one() {
        let mut b = bank(1, 4);
        assert_eq!(b.absorbing, NO_ABSORBING, "off by default, as TMU");
        let lit = pack_feats(&[0, 0, 0, 0]);
        let act = all_active(lit.len());
        let mut rng = Rng::new(3);
        let n = b.clauses[0].excluded.len();
        for _ in 0..1000 {
            b.type_i(0, &lit, &act, false, false, &mut rng, 1.0, 1.0, usize::MAX);
        }
        assert_eq!(b.clauses[0].excluded.len(), n, "nothing absorbed");
        assert!(b.clauses[0].unallocated.is_empty());
        // The absent (positive) literals were decremented to the floor of 1.
        let c = &b.clauses[0];
        for (&l, &s) in c.excluded.iter().zip(&c.excluded_state) {
            if l < 4 {
                assert_eq!(s, 1, "literal {l} floors at 1");
            }
        }
    }

    #[test]
    fn a_literal_at_the_top_state_can_still_be_forgotten_as_tmu() {
        // TMU's sparse bank has no include lock (its absorbing_include is unused):
        // Type Ib keeps lowering an included literal until it leaves.
        let mut b = bank(1, 4);
        let max_state = b.max_state;
        force_include(&mut b, 0, 0, max_state);
        let lit = pack_feats(&[1, 0, 0, 0]); // literal 0 present
        let act = all_active(lit.len());
        let mut rng = Rng::new(5);
        for _ in 0..1000 {
            b.type_i(0, &lit, &act, false, false, &mut rng, 1.0, 1.0, usize::MAX);
        }
        assert!(
            !b.included_literals(0).contains(&0),
            "Type Ib lowers present literals too"
        );
    }

    #[test]
    fn literal_sampling_and_insertion_as_tmu() {
        let mut b = SparseClauseBank::new(3, 2 * 10, 8, 1);
        b.literal_sampling = 0.25;
        b.reinit();
        for c in &b.clauses {
            assert_eq!(c.excluded.len(), 5, "int(20 * 0.25) candidates");
            assert_eq!(c.unallocated.len(), 15);
            let mut all: Vec<u32> = c.excluded.iter().chain(&c.unallocated).copied().collect();
            all.sort_unstable();
            assert_eq!(
                all,
                (0..20).collect::<Vec<u32>>(),
                "a partition of the literals"
            );
        }
        assert_ne!(
            b.clauses[0].excluded, b.clauses[1].excluded,
            "sampled per clause"
        );

        // Insertion: an absorbed candidate is replaced by the next unallocated one.
        b.absorbing = 50;
        b.insertion_state = 40;
        let next = *b.clauses[0].unallocated.last().unwrap();
        let victim = b.clauses[0].excluded[0];
        b.clauses[0].excluded_state[0] = 51;
        let mut feats = [0u8; 10];
        // make every candidate except the victim present, so only it is lowered
        for &l in &b.clauses[0].excluded[1..] {
            if (l as usize) < 10 {
                feats[l as usize] = 1
            }
        }
        let lit = pack_feats(&feats);
        let act = all_active(lit.len());
        let mut rng = Rng::new(1);
        let c = &mut b.clauses[0];
        c.excluded_state.iter_mut().skip(1).for_each(|s| *s = 100);
        b.type_i(0, &lit, &act, false, false, &mut rng, 1.0, 1.0, usize::MAX);
        let c = &b.clauses[0];
        assert!(
            !c.excluded.contains(&victim) || (victim as usize) < 10 && feats[victim as usize] == 1
        );
        assert!(
            c.excluded.contains(&next),
            "the next unallocated literal was inserted"
        );
        assert_eq!(c.absorbed, 1);
        assert_eq!(
            c.excluded_state[c.excluded.iter().position(|&x| x == next).unwrap()],
            40
        );
    }

    #[test]
    fn without_feature_negation_no_negated_literal_is_a_candidate_even_after_grow() {
        let mut b = SparseClauseBank::new(2, 2 * 6, 8, 1);
        b.feature_negation = false;
        b.reinit();
        assert!(b.clauses.iter().all(|c| c.excluded.iter().all(|&l| l < 6)));
        b.grow(2 * 9);
        assert!(
            b.clauses.iter().all(|c| c.excluded.iter().all(|&l| l < 9)),
            "no negated after grow"
        );
        assert_eq!(b.clauses[0].excluded.len(), 9);
    }

    #[test]
    fn max_included_blocks_promotion() {
        let mut b = bank(1, 4);
        // Already at the limit (1 included literal, limit = 1).
        let max_state = b.max_state;
        let half = b.half;
        force_include(&mut b, 0, 0, max_state);
        // Bring literal 1's excluded state to the boundary.
        let c = &mut b.clauses[0];
        let pos = c.excluded.iter().position(|&x| x == 1).unwrap();
        c.excluded_state[pos] = half - 1;

        let lit = pack_feats(&[1, 1, 0, 0]); // literals 0 and 1 present
        let act = all_active(lit.len());
        let mut rng = Rng::new(9);
        b.type_i(0, &lit, &act, true, true, &mut rng, 1.0, 1.0, 1);
        assert_eq!(
            b.n_included(0),
            1,
            "at the include limit, no new literal should be promoted"
        );
    }

    #[test]
    fn grow_remaps_negated_indices_and_adds_excluded() {
        // 4 features (8 literals) -> 7 features (14 literals). half = 128.
        let mut b = bank(1, 4);
        let half = b.half;
        // Include a positive literal (feat 1 → index 1) and a negated literal
        // (feat 2 → index 4 + 2 = 6). Absorb out one excluded literal so the
        // unallocated pool is also exercised.
        force_include(&mut b, 0, 1, 200);
        force_include(&mut b, 0, 6, 200);
        {
            let c = &mut b.clauses[0];
            let pos = c.excluded.iter().position(|&x| x == 5).unwrap(); // neg of feat 1
            c.excluded.swap_remove(pos);
            c.excluded_state.swap_remove(pos);
            c.unallocated.push(5);
        }

        b.grow(14); // 7 features

        let c = &b.clauses[0];
        // Positive index 1 unchanged; negated indices shifted by (7 - 4) = 3.
        assert!(c.included.contains(&1), "positive index must be unchanged");
        assert!(
            c.included.contains(&(6 + 3)),
            "negated index must shift by 3"
        );
        assert!(
            c.unallocated.contains(&(5 + 3)),
            "unallocated negated index must shift"
        );
        // New literals present as just-excluded: new positives 4,5,6 and new
        // negateds 7+4..14 = 11,12,13.
        for l in [4u32, 5, 6, 11, 12, 13] {
            let pos = c.excluded.iter().position(|&x| x == l);
            assert!(pos.is_some(), "new literal {l} must be in excluded");
            assert_eq!(c.excluded_state[pos.unwrap()], half - 1);
        }
        // n_literals bookkeeping updated.
        assert_eq!(b.n_literals, 14);
    }

    #[test]
    fn grow_noop_when_equal() {
        let mut b = bank(1, 4);
        force_include(&mut b, 0, 2, 200);
        let before = b.clauses[0].clone();
        b.grow(8);
        assert_eq!(b.n_literals, 8);
        assert_eq!(b.clauses[0].included, before.included);
        assert_eq!(b.clauses[0].excluded, before.excluded);
    }

    #[test]
    fn determinism_same_seed_same_lists() {
        let run = || {
            let mut b = bank(2, 4);
            let lit = pack_feats(&[1, 0, 1, 1]);
            let act = all_active(lit.len());
            let mut rng = Rng::new(42);
            for _ in 0..50 {
                b.type_i(0, &lit, &act, true, false, &mut rng, 0.3, 0.7, usize::MAX);
                b.type_ii(1, &lit, &act);
            }
            (
                b.included_literals(0).to_vec(),
                b.included_literals(1).to_vec(),
            )
        };
        assert_eq!(run(), run());
    }
}
