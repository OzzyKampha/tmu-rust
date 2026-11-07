//! Classification models.
//!
//! Mirrors TMU's `tmu/models/classification/` package.

pub(crate) mod coalesced_classifier;
mod composite_classifier;
mod convolutional_classifier;
mod fuzzy_pattern_classifier;
mod sparse_classifier;
mod vanilla_classifier;

pub use coalesced_classifier::CoalescedTsetlinMachine;
pub use composite_classifier::TMCompositeClassifier;
pub use convolutional_classifier::ConvolutionalTsetlinMachine;
pub use fuzzy_pattern_classifier::TMFuzzyPatternClassifier;
pub use sparse_classifier::{ClauseKind, RuleLearning, TMSparseClassifier};
pub use vanilla_classifier::TsetlinMachine;
