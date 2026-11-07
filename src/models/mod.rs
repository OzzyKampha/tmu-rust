//! Tsetlin Machine model implementations.
//!
//! Mirrors TMU's `tmu/models/` package.

pub mod autoencoder;
pub mod classification;
pub mod regression;

pub use autoencoder::{HeldOutVotes, TMAutoEncoder, TMCoalescedAutoEncoder};
pub use classification::{
    ClauseKind, CoalescedTsetlinMachine, ConvolutionalTsetlinMachine, TMCompositeClassifier,
    TMFuzzyPatternClassifier, TMSparseClassifier, TsetlinMachine,
};
pub use regression::TMRegressor;
