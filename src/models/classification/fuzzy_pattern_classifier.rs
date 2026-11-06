//! Fuzzy-Pattern Tsetlin Machine (FPTM) classifier.
//!
//! A port of Artem Hnilov's Fuzzy-Pattern Tsetlin Machine
//! ([arXiv:2508.08350](https://arxiv.org/abs/2508.08350); reference code
//! [`BooBSD/FuzzyPatternTM`](https://github.com/BooBSD/FuzzyPatternTM),
//! `src/FuzzyPatternTM.jl`, MIT License, Copyright (c) 2025 Artem Hnilov).
//!
//! ## What differs from a vanilla Tsetlin machine
//!
//! * **Fuzzy clause evaluation.** A clause with `n` included literals starts
//!   from `min(n, LF)` (`LF` when it is empty) and loses one point per included
//!   literal that is false in the input, never going below 0. Up to `LF - 1`
//!   failures leave a reduced vote instead of silencing the clause, so each
//!   clause acts as a set of sub-patterns. `LF` is
//!   [`literal_failures`](TMFuzzyPatternClassifier::literal_failures).
//! * **Unweighted votes.** Each class has `clauses_per_class / 2` positive and as
//!   many negative clauses (even index positive, odd negative, as elsewhere in
//!   tmu-rs); its vote is the sum of positive scores minus the sum of negative
//!   ones. The vote is clamped to `±T` only for the feedback probability, so
//!   [`scores`](TMFuzzyPatternClassifier::scores) returns it unclamped.
//! * **Feedback**, per training sample: the true class's clauses get positive
//!   feedback and *every* other class's clauses negative feedback, each clause
//!   with probability `(T - v) / 2T` (positive) or `(T + v) / 2T` (negative),
//!   `v` being the class's clamped vote.
//!   - *Feedback 1* (the clauses that vote for the feedback's direction): if the
//!     clause scores above 0, every literal true in the input steps up — only
//!     while the clause has at most `L` included literals
//!     ([`max_literals`](TMFuzzyPatternClassifier::max_literals)) — and every
//!     excluded literal false in the input steps down, with no randomness.
//!     Otherwise `s = round(features / S)` random positive literals and as many
//!     random negated ones step down (`S`, the specificity, is the constructor's
//!     `specificity`).
//!   - *Feedback 2* (the other clauses): if the clause scores above 0, every
//!     excluded literal false in the input steps up.
//! * **States** run 0..=255; a literal is included at
//!   [`include_limit`](TMFuzzyPatternClassifier::include_limit) or above and
//!   starts one below it. The reference's text examples use 220.
//!
//! The reference trains samples in parallel threads without synchronisation;
//! here each class owns its state and RNG, so training is deterministic for a
//! seed, and with `--features parallel` the classes of one sample are trained
//! in parallel with the same result.

use crate::clause_bank::dense::{GOLDEN, WORD_BITS, words_for};
#[cfg(feature = "parallel")]
use crate::clause_bank::dense::{PARALLEL_MIN, use_parallel};
use crate::encoder::{EncodedBatch, EncodedSample};
use crate::rng::Rng;

/// One class's clauses: per clause a state for each of the `2 * n_features`
/// literals (positive literals first, then negated, as in the encoder) and the
/// literals currently included.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct Team {
    /// `clauses_per_class * n_literals` states, clause-major.
    pub(crate) states: Vec<u8>,
    /// Included literals of each clause, in literal order.
    pub(crate) included: Vec<Vec<u32>>,
    /// This class's feedback RNG.
    pub(crate) rng: Rng,
}

/// A multiclass Fuzzy-Pattern Tsetlin Machine.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TMFuzzyPatternClassifier {
    pub(crate) n_classes: usize,
    pub(crate) n_features: usize,
    pub(crate) n_literals: usize,
    pub(crate) words: usize,
    pub(crate) clauses_per_class: usize,
    pub(crate) threshold: i32,
    /// `S`: the number of random decrements per unmatched Feedback 1 is
    /// `round(n_features / S)`.
    pub(crate) specificity: f64,
    /// `round(n_features / S)`, ties to even as Julia's `round`.
    pub(crate) s_draws: usize,
    /// `L`: Feedback 1 raises literals only while a clause has at most this
    /// many included literals.
    pub(crate) max_literals: usize,
    /// `LF`: how many literal failures silence a clause.
    pub(crate) literal_failures: usize,
    /// States at or above this are included.
    pub(crate) include_limit: u8,
    pub(crate) teams: Vec<Team>,
    /// Shuffle RNG.
    pub(crate) rng: Rng,
}

#[cfg(feature = "serde")]
impl crate::serial::SaveLoad for TMFuzzyPatternClassifier {
    const TAG: u8 = crate::serial::TAG_FUZZY_PATTERN;
}

const STATE_MAX: u8 = u8::MAX;
const STATE_MIN: u8 = 0;

#[inline]
fn bit(lit: &[u64], l: usize) -> bool {
    (lit[l / WORD_BITS] >> (l % WORD_BITS)) & 1 == 1
}

/// The fuzzy score of a clause whose included literals are `included`.
#[inline]
fn clause_score(lit: &[u64], included: &[u32], lf: usize) -> i32 {
    let n = included.len();
    let mut c = if n > 0 && n < lf { n as i32 } else { lf as i32 };
    for &l in included {
        if c <= 0 {
            return 0;
        }
        c -= !bit(lit, l as usize) as i32;
    }
    c
}

impl TMFuzzyPatternClassifier {
    /// A classifier with the reference's defaults: `L = 16`, `LF = 4`,
    /// include limit 128, seed 42.
    ///
    /// `clauses_per_class` must be even (half positive, half negative);
    /// `specificity` is the reference's `S` (> 0).
    pub fn new(
        n_classes: usize,
        n_features: usize,
        clauses_per_class: usize,
        threshold: i32,
        specificity: f64,
    ) -> Self {
        assert!(n_classes >= 1, "n_classes must be >= 1");
        assert!(n_features >= 1, "n_features must be >= 1");
        assert!(
            clauses_per_class >= 2 && clauses_per_class.is_multiple_of(2),
            "clauses_per_class must be even and >= 2 (half positive, half negative)"
        );
        assert!(threshold >= 1, "threshold must be >= 1");
        assert!(specificity > 0.0, "specificity must be > 0");
        let n_literals = 2 * n_features;
        let mut tm = Self {
            n_classes,
            n_features,
            n_literals,
            words: words_for(n_literals),
            clauses_per_class,
            threshold,
            specificity,
            s_draws: (n_features as f64 / specificity).round_ties_even() as usize,
            max_literals: 16,
            literal_failures: 4,
            include_limit: 128,
            teams: Vec::new(),
            rng: Rng::new(42),
        };
        tm.reset(42);
        tm
    }

    /// Reset every clause to the initial state (all literals one below the
    /// include limit, nothing included) and reseed.
    fn reset(&mut self, seed: u64) {
        let init = self.include_limit - 1;
        let cps = self.clauses_per_class;
        let nl = self.n_literals;
        self.teams = (0..self.n_classes)
            .map(|c| Team {
                states: vec![init; cps * nl],
                included: vec![Vec::new(); cps],
                rng: Rng::new(seed ^ GOLDEN.wrapping_mul(c as u64 + 1)),
            })
            .collect();
        self.rng = Rng::new(seed);
    }

    /// `L`: Feedback 1 raises literals only while a clause has at most `l`
    /// included literals. Default 16.
    pub fn max_literals(mut self, l: usize) -> Self {
        self.max_literals = l;
        self
    }

    /// `LF`: a clause scores `min(included, LF)` minus its failing literals,
    /// floored at 0. Default 4; must be >= 1.
    pub fn literal_failures(mut self, lf: usize) -> Self {
        assert!(lf >= 1, "literal_failures must be >= 1");
        self.literal_failures = lf;
        self
    }

    /// States at or above `limit` are included; every literal starts at
    /// `limit - 1`. Default 128; must be in 1..=255. Resets the clauses: set it
    /// before training.
    pub fn include_limit(mut self, limit: u8) -> Self {
        assert!(limit >= 1, "include_limit must be in 1..=255");
        self.include_limit = limit;
        let seed = self.rng.clone().next_u64();
        self.reset(seed);
        self
    }

    /// Reseed (resets the clauses: set it before training).
    pub fn seed(mut self, seed: u64) -> Self {
        self.reset(seed);
        self
    }

    // ---- accessors --------------------------------------------------------

    /// Number of classes.
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }
    /// Number of input features.
    pub fn n_features(&self) -> usize {
        self.n_features
    }
    /// Clauses per class (half positive, half negative).
    pub fn clauses_per_class(&self) -> usize {
        self.clauses_per_class
    }
    /// The voting threshold `T`.
    pub fn threshold(&self) -> i32 {
        self.threshold
    }
    /// `round(n_features / S)`: random decrements per unmatched Feedback 1.
    pub fn s_draws(&self) -> usize {
        self.s_draws
    }

    /// Included literals of clause `j` of `class`, as literal indices
    /// (`< n_features` positive, `>= n_features` negated).
    pub fn included_literals(&self, class: usize, j: usize) -> &[u32] {
        &self.teams[class].included[j]
    }

    /// The state of literal `l` in clause `j` of `class`.
    pub fn literal_state(&self, class: usize, j: usize, l: usize) -> u8 {
        self.teams[class].states[j * self.n_literals + l]
    }

    /// The fuzzy score of clause `j` of `class` for `sample` (0..=LF).
    pub fn clause_score(&self, sample: &EncodedSample, class: usize, j: usize) -> i32 {
        clause_score(
            &sample.0,
            &self.teams[class].included[j],
            self.literal_failures,
        )
    }

    // ---- inference --------------------------------------------------------

    /// A class's vote: positive clause scores minus negative ones.
    fn vote(&self, team: &Team, lit: &[u64]) -> i32 {
        vote(team, lit, self.literal_failures)
    }

    fn predict_lit(&self, lit: &[u64]) -> usize {
        let mut best = 0;
        let mut best_v = i32::MIN;
        for (c, team) in self.teams.iter().enumerate() {
            let v = self.vote(team, lit);
            if v > best_v {
                best_v = v;
                best = c;
            }
        }
        best
    }

    /// Predict the class for an encoded sample (ties go to the lowest class).
    pub fn predict(&self, sample: &EncodedSample) -> usize {
        self.predict_lit(&sample.0)
    }

    /// Fill `out` with each class's vote, **unclamped** (the reference predicts
    /// on the unclamped vote; clamping would only add ties).
    pub fn scores(&self, sample: &EncodedSample, out: &mut [i32]) {
        debug_assert_eq!(out.len(), self.n_classes);
        for (o, team) in out.iter_mut().zip(&self.teams) {
            *o = self.vote(team, &sample.0);
        }
    }

    /// Clauses of `class` that score above 0 for `sample`.
    pub fn fired_clauses(&self, sample: &EncodedSample, class: usize) -> Vec<usize> {
        let team = &self.teams[class];
        (0..self.clauses_per_class)
            .filter(|&j| clause_score(&sample.0, &team.included[j], self.literal_failures) > 0)
            .collect()
    }

    /// Predict every sample of a batch.
    pub fn predict_batch(&self, batch: &EncodedBatch) -> Vec<usize> {
        let w = self.words;
        let n = batch.len();
        let data = batch.data.as_slice();
        #[cfg(feature = "parallel")]
        if n >= PARALLEL_MIN && use_parallel(self.clauses_per_class * self.n_classes, w) {
            use rayon::prelude::*;
            return (0..n)
                .into_par_iter()
                .map(|i| self.predict_lit(&data[i * w..(i + 1) * w]))
                .collect();
        }
        (0..n)
            .map(|i| self.predict_lit(&data[i * w..(i + 1) * w]))
            .collect()
    }

    /// Fraction of a batch predicted as `ys`.
    pub fn accuracy(&self, batch: &EncodedBatch, ys: &[usize]) -> f64 {
        assert_eq!(batch.len(), ys.len());
        if ys.is_empty() {
            return 0.0;
        }
        let p = self.predict_batch(batch);
        p.iter().zip(ys).filter(|(a, b)| a == b).count() as f64 / ys.len() as f64
    }

    // ---- training ---------------------------------------------------------

    /// Train on one sample with true class `y`.
    pub fn fit_one(&mut self, sample: &EncodedSample, y: usize) {
        let lit = sample.0.clone();
        self.fit_one_lit(&lit, y);
    }

    fn fit_one_lit(&mut self, lit: &[u64], y: usize) {
        assert!(y < self.n_classes, "label {y} out of range");
        let p = Params::of(self);
        #[cfg(feature = "parallel")]
        if self.n_classes > 1 && use_parallel(self.clauses_per_class * self.n_classes, self.words) {
            use rayon::prelude::*;
            self.teams
                .par_iter_mut()
                .enumerate()
                .for_each(|(c, team)| feedback(&p, team, lit, c == y));
            return;
        }
        for (c, team) in self.teams.iter_mut().enumerate() {
            feedback(&p, team, lit, c == y);
        }
    }

    /// One epoch over a batch in a shuffled order.
    pub fn fit_epoch(&mut self, batch: &EncodedBatch, ys: &[usize]) {
        let n = batch.len();
        assert_eq!(n, ys.len());
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let k = self.rng.below(i + 1);
            order.swap(i, k);
        }
        let w = self.words;
        for &i in &order {
            let lit = &batch.data[i * w..(i + 1) * w];
            self.fit_one_lit(lit, ys[i]);
        }
    }

    /// Widen the input to `n` features (`n >= n_features`), keeping everything
    /// learned: the new literals start one below the include limit, as at
    /// initialisation, and are included nowhere, so every clause scores old
    /// inputs (with the new features absent) exactly as before.
    pub fn grow_features(&mut self, n: usize) {
        let old = self.n_features;
        assert!(n >= old, "cannot shrink {old} features to {n}");
        if n == old {
            return;
        }
        let init = self.include_limit - 1;
        let (old_nl, new_nl) = (2 * old, 2 * n);
        for team in &mut self.teams {
            let mut states = vec![init; self.clauses_per_class * new_nl];
            for j in 0..self.clauses_per_class {
                let (src, dst) = (&team.states[j * old_nl..(j + 1) * old_nl], j * new_nl);
                states[dst..dst + old].copy_from_slice(&src[..old]);
                states[dst + n..dst + n + old].copy_from_slice(&src[old..]);
            }
            team.states = states;
            for inc in &mut team.included {
                for l in inc.iter_mut() {
                    if *l as usize >= old {
                        *l += (n - old) as u32;
                    }
                }
            }
        }
        self.n_features = n;
        self.n_literals = new_nl;
        self.words = words_for(new_nl);
        self.s_draws = (n as f64 / self.specificity).round_ties_even() as usize;
    }

    /// How many literal automata sit at each state `0..=255`, over every
    /// clause of every class (as the other classifiers'
    /// `ta_state_histogram`). States from
    /// [`inclusion_state`](Self::inclusion_state) up include their literal.
    pub fn ta_state_histogram(&self) -> Vec<u64> {
        let mut h = vec![0u64; usize::from(STATE_MAX) + 1];
        for team in &self.teams {
            for &s in &team.states {
                h[usize::from(s)] += 1;
            }
        }
        h
    }

    /// The state from which a literal is included (the include limit).
    pub fn inclusion_state(&self) -> u8 {
        self.include_limit
    }
}

/// A class's vote: positive (even) clause scores minus negative (odd) ones.
fn vote(team: &Team, lit: &[u64], lf: usize) -> i32 {
    team.included
        .iter()
        .enumerate()
        .map(|(j, inc)| {
            let s = clause_score(lit, inc, lf);
            if j.is_multiple_of(2) { s } else { -s }
        })
        .sum()
}

/// The settings feedback needs, copied out so classes can train in parallel.
struct Params {
    n_features: usize,
    n_literals: usize,
    cps: usize,
    t: i32,
    s_draws: usize,
    l: usize,
    lf: usize,
    limit: u8,
}

impl Params {
    fn of(tm: &TMFuzzyPatternClassifier) -> Self {
        Self {
            n_features: tm.n_features,
            n_literals: tm.n_literals,
            cps: tm.clauses_per_class,
            t: tm.threshold,
            s_draws: tm.s_draws,
            l: tm.max_literals,
            lf: tm.literal_failures,
            limit: tm.include_limit,
        }
    }
}

/// The reference's `feedback!` for one class: `positive` when `y` is this
/// class. Feedback 1 goes to the clauses voting in the feedback's direction
/// (positive clauses for the true class, negative clauses otherwise),
/// Feedback 2 to the others. Clauses are visited Feedback 1 first, then
/// Feedback 2, each in index order, as in the reference.
fn feedback(p: &Params, team: &mut Team, lit: &[u64], positive: bool) {
    let v = vote(team, lit, p.lf).clamp(-p.t, p.t);
    let update = if positive {
        f64::from(p.t - v)
    } else {
        f64::from(p.t + v)
    } / f64::from(2 * p.t);
    // Feedback 1 clauses have parity `first`, Feedback 2 the other.
    let first = if positive { 0 } else { 1 };
    for pass in [first, 1 - first] {
        for j in (pass..p.cps).step_by(2) {
            if team.rng.next_f64() >= update {
                continue;
            }
            let matched = clause_score(lit, &team.included[j], p.lf) > 0;
            let states = &mut team.states[j * p.n_literals..(j + 1) * p.n_literals];
            if pass == first {
                if matched {
                    feedback_1_matched(states, lit, team.included[j].len() <= p.l, p.limit);
                } else {
                    // Only the drawn literals change: drop those that fell
                    // below the limit instead of rescanning every state.
                    let included = &mut team.included[j];
                    let drop = |l: usize| {
                        if let Ok(k) = included.binary_search(&(l as u32)) {
                            included.remove(k);
                        }
                    };
                    feedback_1_unmatched(
                        states,
                        &mut team.rng,
                        p.n_features,
                        p.s_draws,
                        p.limit,
                        drop,
                    );
                    continue;
                }
            } else if matched {
                feedback_2(states, lit, p.limit);
            } else {
                continue;
            }
            team.included[j] = included_of(states, p.limit);
        }
    }
}

/// Feedback 1, clause matched: raise every literal true in the input (when
/// `grow`), then lower every excluded literal false in the input.
fn feedback_1_matched(states: &mut [u8], lit: &[u64], grow: bool, limit: u8) {
    if grow {
        for (l, s) in states.iter_mut().enumerate() {
            if bit(lit, l) && *s < STATE_MAX {
                *s += 1;
            }
        }
    }
    for (l, s) in states.iter_mut().enumerate() {
        if !bit(lit, l) && *s < limit && *s > STATE_MIN {
            *s -= 1;
        }
    }
}

/// Feedback 1, clause unmatched: `s_draws` times, lower one random positive
/// literal and one random negated literal (drawn in that order, as the
/// reference). `excluded(l)` is called for each literal that falls from the
/// include limit to below it.
fn feedback_1_unmatched(
    states: &mut [u8],
    rng: &mut Rng,
    n_features: usize,
    s_draws: usize,
    limit: u8,
    mut excluded: impl FnMut(usize),
) {
    for _ in 0..s_draws {
        for i in [rng.below(n_features), n_features + rng.below(n_features)] {
            if states[i] > STATE_MIN {
                states[i] -= 1;
                if states[i] + 1 == limit {
                    excluded(i);
                }
            }
        }
    }
}

/// Feedback 2, clause matched: raise every excluded literal false in the input.
fn feedback_2(states: &mut [u8], lit: &[u64], limit: u8) {
    for (l, s) in states.iter_mut().enumerate() {
        if !bit(lit, l) && *s < limit {
            *s += 1;
        }
    }
}

fn included_of(states: &[u8], limit: u8) -> Vec<u32> {
    states
        .iter()
        .enumerate()
        .filter(|&(_, &s)| s >= limit)
        .map(|(l, _)| l as u32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Encoder;

    fn lits(bits: &[u8]) -> Vec<u64> {
        EncodedSample::from_bits(bits, bits.len()).0
    }

    #[test]
    fn clause_score_is_min_of_size_and_lf_minus_failures() {
        // features x0=1, x1=0, x2=1, x3=0; literals 0..4 positive, 4..8 negated.
        let x = lits(&[1, 0, 1, 0]);
        assert_eq!(clause_score(&x, &[], 4), 4, "an empty clause scores LF");
        assert_eq!(clause_score(&x, &[0, 2], 4), 2, "2 literals, both true");
        assert_eq!(clause_score(&x, &[0, 1, 2], 4), 2, "3 literals, one fails");
        assert_eq!(
            clause_score(&x, &[0, 2, 5, 7], 4),
            4,
            "4 true literals cap at LF"
        );
        assert_eq!(
            clause_score(&x, &[0, 2, 5, 7, 1, 3], 4),
            2,
            "6 literals, 2 fail, LF 4"
        );
        assert_eq!(clause_score(&x, &[1, 3, 4, 6], 4), 0, "all fail");
        assert_eq!(clause_score(&x, &[1, 3, 4, 6, 0], 4), 0, "never below 0");
        assert_eq!(
            clause_score(&x, &[1], 1),
            0,
            "LF 1 is the all-or-nothing clause"
        );
        assert_eq!(clause_score(&x, &[0, 2, 5], 1), 1);
    }

    #[test]
    fn feedback_rules_follow_the_reference() {
        let x = lits(&[1, 0]); // literals: x0=1, x1=0, !x0=0, !x1=1
        let mut s = vec![127u8, 127, 127, 200];
        feedback_1_matched(&mut s, &x, true, 128);
        assert_eq!(
            s,
            vec![128, 126, 126, 201],
            "true literals up, excluded false ones down"
        );
        let mut s = vec![127u8, 130, 127, 127];
        feedback_1_matched(&mut s, &x, false, 128);
        assert_eq!(
            s,
            vec![127, 130, 126, 127],
            "over L: no raise; included false literal kept"
        );
        let mut s = vec![127u8, 127, 130, 127];
        feedback_2(&mut s, &x, 128);
        assert_eq!(
            s,
            vec![127, 128, 130, 127],
            "excluded false literals up, included ones left"
        );
        let mut s = vec![1u8, 0, 1, 0];
        let mut rng = Rng::new(1);
        let mut dropped = Vec::new();
        feedback_1_unmatched(&mut s, &mut rng, 2, 50, 1, |l| dropped.push(l));
        dropped.sort();
        assert_eq!(
            dropped,
            vec![0, 2],
            "the literals that fell below the limit"
        );
        assert!(
            s.iter().all(|&v| v == 0),
            "random decrements floor at 0: {s:?}"
        );
    }

    fn noisy_xor(n: usize, noise: f64, seed: u64) -> (Vec<Vec<u8>>, Vec<usize>) {
        let mut rng = Rng::new(seed);
        (0..n)
            .map(|_| {
                let f: Vec<u8> = (0..12).map(|_| (rng.next_u64() & 1) as u8).collect();
                let mut y = (f[0] ^ f[1]) as usize;
                if rng.next_f64() < noise {
                    y = 1 - y;
                }
                (f, y)
            })
            .unzip()
    }

    fn train(seed: u64) -> (TMFuzzyPatternClassifier, f64) {
        let (xtr, ytr) = noisy_xor(3000, 0.1, 1);
        let (xte, yte) = noisy_xor(2000, 0.0, 2);
        let enc = Encoder::for_binary(12);
        let rows =
            |v: &[Vec<u8>]| enc.encode_batch(&v.iter().map(|x| x.as_slice()).collect::<Vec<_>>());
        let (btr, bte) = (rows(&xtr), rows(&xte));
        let mut tm = TMFuzzyPatternClassifier::new(2, 12, 20, 10, 2.0)
            .max_literals(12)
            .literal_failures(2)
            .seed(seed);
        for _ in 0..30 {
            tm.fit_epoch(&btr, &ytr);
        }
        let acc = tm.accuracy(&bte, &yte);
        (tm, acc)
    }

    #[test]
    fn learns_noisy_xor_and_is_deterministic() {
        let (a, acc) = train(7);
        assert!(acc > 0.9, "noisy XOR test accuracy {acc}");
        let (b, _) = train(7);
        for c in 0..2 {
            assert_eq!(
                a.teams[c].states, b.teams[c].states,
                "same seed, same model"
            );
        }
        let (c, _) = train(8);
        assert_ne!(
            a.teams[0].states, c.teams[0].states,
            "another seed, another model"
        );
    }

    #[test]
    fn scores_are_unclamped_votes_and_predict_takes_the_first_maximum() {
        let mut tm = TMFuzzyPatternClassifier::new(3, 2, 2, 1, 1.0);
        let x = EncodedSample::from_bits(&[1, 0], 2);
        let mut out = [0i32; 3];
        tm.scores(&x, &mut out);
        assert_eq!(out, [0, 0, 0], "empty clauses cancel: LF - LF");
        assert_eq!(tm.predict(&x), 0, "a tie goes to the lowest class");
        // Class 2: positive clause needs x0 (true), negative clause needs x1 (false).
        tm.teams[2].included = vec![vec![0], vec![1]];
        tm.scores(&x, &mut out);
        assert_eq!(out, [0, 0, 1]);
        assert_eq!(tm.predict(&x), 2);
        assert_eq!(tm.fired_clauses(&x, 2), vec![0]);
    }

    #[test]
    fn training_the_classes_together_matches_training_them_one_by_one() {
        // 3 x 50 clauses: above the parallel threshold with `--features parallel`.
        let (xs, ys) = noisy_xor(300, 0.1, 5);
        let ys: Vec<usize> = ys
            .iter()
            .zip(&xs)
            .map(|(y, x)| (y + x[2] as usize) % 3)
            .collect();
        let tm0 = TMFuzzyPatternClassifier::new(3, 12, 50, 8, 3.0)
            .literal_failures(2)
            .seed(4);
        let mut a = tm0.clone();
        let mut b = tm0;
        for (x, &y) in xs.iter().zip(&ys) {
            let lit = EncodedSample::from_bits(x, 12).0;
            a.fit_one_lit(&lit, y);
            let p = Params::of(&b);
            for (c, team) in b.teams.iter_mut().enumerate() {
                feedback(&p, team, &lit, c == y);
            }
        }
        for c in 0..3 {
            assert_eq!(a.teams[c].states, b.teams[c].states, "class {c}");
            assert_eq!(a.teams[c].included, b.teams[c].included, "class {c}");
        }
    }

    #[test]
    fn the_histogram_counts_every_automaton() {
        let tm = TMFuzzyPatternClassifier::new(3, 5, 4, 10, 2.0).include_limit(200);
        let h = tm.ta_state_histogram();
        assert_eq!(h.len(), 256);
        assert_eq!(h.iter().sum::<u64>(), 3 * 4 * 10);
        assert_eq!(
            h[199],
            3 * 4 * 10,
            "every literal starts one below the limit"
        );
        assert_eq!(tm.inclusion_state(), 200);
    }

    #[test]
    fn grow_features_keeps_what_was_learned() {
        let (mut tm, _) = train(5);
        let (xte, _) = noisy_xor(300, 0.0, 6);
        let before: Vec<usize> = xte
            .iter()
            .map(|x| tm.predict(&EncodedSample::from_bits(x, 12)))
            .collect();
        let rules: Vec<Vec<u32>> = (0..20)
            .map(|j| tm.included_literals(1, j).to_vec())
            .collect();
        tm.grow_features(20);
        assert_eq!(tm.n_features(), 20);
        assert_eq!(tm.s_draws(), 10, "round(20 / 2)");
        for (x, &p) in xte.iter().zip(&before) {
            let mut wide = x.clone();
            wide.resize(20, 0);
            assert_eq!(tm.predict(&EncodedSample::from_bits(&wide, 20)), p);
        }
        for (j, r) in rules.iter().enumerate() {
            let moved: Vec<u32> = r.iter().map(|&l| if l >= 12 { l + 8 } else { l }).collect();
            assert_eq!(tm.included_literals(1, j), moved.as_slice(), "clause {j}");
            assert_eq!(
                tm.literal_state(1, j, 12),
                tm.include_limit - 1,
                "new feature starts fresh"
            );
        }
        // It keeps training on the wider input.
        let mut wide = xte[0].clone();
        wide.resize(20, 1);
        tm.fit_one(&EncodedSample::from_bits(&wide, 20), 1);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn save_and_load_round_trip() {
        use crate::serial::SaveLoad;
        let (tm, _) = train(3);
        let mut buf = Vec::new();
        tm.write_to(&mut buf).unwrap();
        let back = TMFuzzyPatternClassifier::read_from(&mut buf.as_slice()).unwrap();
        let (xte, _) = noisy_xor(200, 0.0, 9);
        let enc = Encoder::for_binary(12);
        let b = enc.encode_batch(&xte.iter().map(|x| x.as_slice()).collect::<Vec<_>>());
        assert_eq!(tm.predict_batch(&b), back.predict_batch(&b));
        assert_eq!(tm.s_draws, back.s_draws);
    }
}
