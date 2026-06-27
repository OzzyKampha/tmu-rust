use crate::models::{
    CoalescedTsetlinMachine, TMAutoEncoder, TMCoalescedAutoEncoder, TMRegressor, TsetlinMachine,
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
    fn clause_weight(&self, output: usize, clause: usize) -> i32 {
        self.clause_weight(output, clause)
    }
    fn clause_rule(&self, output: usize, clause: usize) -> Vec<(usize, bool)> {
        self.clause_rule(output, clause)
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
}
