//! Bit-packed, **weighted** multiclass Tsetlin Machine with bit-parallel and
//! optionally multi-threaded training.
//!
//! ## Module layout
//!
//! | Module | Role |
//! |---|---|
//! | `clause_bank::dense` | Bit-manipulation primitives (fire, inc/dec, type I/II feedback, pack). Mirrors TMU's `clause_bank_dense.py`. |
//! | `clause_bank::sparse` | Sparse clause bank with absorbing actions (per-clause index lists). Mirrors TMU's `clause_bank_sparse.py`. |
//! | [`models`] | TM model implementations. |
//! | [`TsetlinMachine`] | Vanilla weighted multiclass TM. Mirrors TMU's `vanilla_classifier.py`. |
//! | [`CoalescedTsetlinMachine`] | Coalesced TM: shared clause bank + signed per-class weights. Mirrors TMU's `coalesced_classifier.py`. |
//! | [`TMSparseClassifier`] | Sparse-clause-bank TM: absorbing actions remove literals as training converges. |
//! | [`data`] | CSV loaders for binary-valued datasets. |
//! | [`Booleanizer`] | Feature booleanizer (thresholding / thermometer encoding). |
//! | [`Rng`] | Fast xoshiro256** RNG used throughout. |
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use tmu_rs::TsetlinMachine;
//!
//! let mut tm = TsetlinMachine::with_config(2, 12, 8, 15, 3.9, 8, true, 42);
//! // tm.fit_epoch(&train_x, &train_y);
//! // let acc = tm.accuracy(&test_x, &test_y);
//! ```

mod booleanizer;
pub(crate) mod clause_bank;
mod clause_inspect;
pub mod data;
pub mod encoder;
pub mod models;
mod rng;
#[cfg(feature = "serde")]
mod serial;

pub use booleanizer::Booleanizer;
pub use clause_inspect::ClauseInspect;
pub use encoder::{EncodedBatch, EncodedSample, Encoder};
pub use models::{
    CoalescedTsetlinMachine, ConvolutionalTsetlinMachine, TMAutoEncoder, TMCoalescedAutoEncoder,
    TMCompositeClassifier, TMRegressor, TMSparseClassifier, TsetlinMachine,
};
pub use rng::Rng;
#[cfg(feature = "serde")]
pub use serial::SaveLoad;
