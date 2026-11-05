use crate::encoder::EncodedSample;
use crate::models::{
    CoalescedTsetlinMachine, TMAutoEncoder, TMCoalescedAutoEncoder, TMFuzzyPatternClassifier,
    TMRegressor, TMSparseClassifier, TsetlinMachine,
};

/// Uniform clause-inspection interface across all TM model types.
///
/// Normalises the differing concrete APIs (no `output` arg for coalesced/regressor
/// types, `clauses_per_class` vs `n_clauses`, etc.) behind a single trait so that
/// generic stat functions can operate on any model.
pub trait ClauseInspect {
    /// Total number of clauses in the model.
    fn n_clauses(&self) -> usize;

    /// Number of logical outputs: classes for classifiers, features for
    /// autoencoders, `1` for regressors.
    fn n_outputs(&self) -> usize;

    /// Signed weight of `clause` for `output`.
    fn clause_weight(&self, output: usize, clause: usize) -> i32;

    /// Included literals for `clause` of `output` as `(feature_index, is_negated)` pairs.
    fn clause_rule(&self, output: usize, clause: usize) -> Vec<(usize, bool)>;

    /// Return the indices of all clauses (local to `output`) that fire for `sample`.
    fn fired_clauses(&self, sample: &EncodedSample, output: usize) -> Vec<usize>;
}

impl ClauseInspect for CoalescedTsetlinMachine {
    fn n_clauses(&self) -> usize {
        self.n_clauses()
    }
    fn n_outputs(&self) -> usize {
        self.n_classes()
    }
    fn clause_weight(&self, output: usize, clause: usize) -> i32 {
        self.clause_weight(output, clause)
    }
    fn clause_rule(&self, _output: usize, clause: usize) -> Vec<(usize, bool)> {
        self.clause_rule(clause)
    }
    fn fired_clauses(&self, sample: &EncodedSample, output: usize) -> Vec<usize> {
        self.fired_clauses(sample, output)
    }
}

impl ClauseInspect for TsetlinMachine {
    // Returns clauses *per output* so `clause` always ranges 0..n_clauses()
    // regardless of TM type (coalesced banks have one shared set; non-coalesced
    // have one set per output).
    fn n_clauses(&self) -> usize {
        self.clauses_per_class()
    }
    fn n_outputs(&self) -> usize {
        self.n_classes()
    }
    // Weights are stored unsigned with polarity by index (even clauses vote for
    // the class, odd ones against); report the signed vote, as the trait says.
    fn clause_weight(&self, output: usize, clause: usize) -> i32 {
        let w = self.clause_weight(output, clause);
        if self.clause_is_positive(clause) {
            w
        } else {
            -w
        }
    }
    fn clause_rule(&self, output: usize, clause: usize) -> Vec<(usize, bool)> {
        self.clause_rule(output, clause)
    }
    fn fired_clauses(&self, sample: &EncodedSample, output: usize) -> Vec<usize> {
        self.fired_clauses(sample, output)
    }
}

impl ClauseInspect for TMSparseClassifier {
    // Clauses are per class (like `TsetlinMachine`), with polarity by index:
    // even clauses vote for the class, odd ones against. The weight is reported
    // signed accordingly, so a fired clause adds exactly `clause_weight`.
    fn n_clauses(&self) -> usize {
        self.clauses_per_class()
    }
    fn n_outputs(&self) -> usize {
        self.n_classes()
    }
    fn clause_weight(&self, output: usize, clause: usize) -> i32 {
        let w = self.clause_weight(output, clause);
        if self.clause_is_positive(clause) {
            w
        } else {
            -w
        }
    }
    fn clause_rule(&self, output: usize, clause: usize) -> Vec<(usize, bool)> {
        self.clause_rule(output, clause)
    }
    fn fired_clauses(&self, sample: &EncodedSample, output: usize) -> Vec<usize> {
        self.fired_clauses(sample, output)
    }
}

/// Fuzzy-pattern clauses are unweighted: positive (even) clauses weigh +1 and
/// negative (odd) ones -1. "Fired" means a fuzzy score above 0.
impl ClauseInspect for TMFuzzyPatternClassifier {
    fn n_clauses(&self) -> usize {
        self.clauses_per_class()
    }
    fn n_outputs(&self) -> usize {
        self.n_classes()
    }
    fn clause_weight(&self, _output: usize, clause: usize) -> i32 {
        if clause.is_multiple_of(2) { 1 } else { -1 }
    }
    fn clause_rule(&self, output: usize, clause: usize) -> Vec<(usize, bool)> {
        let nf = self.n_features();
        self.included_literals(output, clause)
            .iter()
            .map(|&l| {
                let l = l as usize;
                if l < nf { (l, false) } else { (l - nf, true) }
            })
            .collect()
    }
    fn fired_clauses(&self, sample: &EncodedSample, output: usize) -> Vec<usize> {
        self.fired_clauses(sample, output)
    }
}

impl ClauseInspect for TMAutoEncoder {
    fn n_clauses(&self) -> usize {
        self.clauses_per_output()
    }
    fn n_outputs(&self) -> usize {
        self.n_features()
    }
    fn clause_weight(&self, output: usize, clause: usize) -> i32 {
        self.clause_weight(output, clause)
    }
    fn clause_rule(&self, output: usize, clause: usize) -> Vec<(usize, bool)> {
        self.clause_rule(output, clause)
    }
    fn fired_clauses(&self, sample: &EncodedSample, output: usize) -> Vec<usize> {
        self.fired_clauses(sample, output)
    }
}

impl ClauseInspect for TMCoalescedAutoEncoder {
    fn n_clauses(&self) -> usize {
        self.n_clauses()
    }
    fn n_outputs(&self) -> usize {
        self.n_features()
    }
    fn clause_weight(&self, output: usize, clause: usize) -> i32 {
        self.clause_weight(output, clause)
    }
    fn clause_rule(&self, _output: usize, clause: usize) -> Vec<(usize, bool)> {
        self.clause_rule(clause)
    }
    fn fired_clauses(&self, sample: &EncodedSample, output: usize) -> Vec<usize> {
        self.fired_clauses(sample, output)
    }
}

impl ClauseInspect for TMRegressor {
    fn n_clauses(&self) -> usize {
        self.n_clauses()
    }
    fn n_outputs(&self) -> usize {
        1
    }
    fn clause_weight(&self, _output: usize, clause: usize) -> i32 {
        self.clause_weight(clause)
    }
    fn clause_rule(&self, _output: usize, clause: usize) -> Vec<(usize, bool)> {
        self.clause_rule(clause)
    }
    fn fired_clauses(&self, sample: &EncodedSample, output: usize) -> Vec<usize> {
        self.fired_clauses(sample, output)
    }
}
