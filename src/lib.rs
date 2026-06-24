//! Bit-packed, **weighted** multiclass Tsetlin Machine with bit-parallel and
//! optionally multi-threaded training.
//!
//! ## Module layout
//!
//! | Module | Role |
//! |---|---|
//! | `clause_bank::dense` | Bit-manipulation primitives (fire, inc/dec, type I/II feedback, pack). Mirrors TMU's `clause_bank_dense.py`. |
//! | [`models`] | TM model implementations. |
//! | [`TsetlinMachine`] | Vanilla weighted multiclass TM. Mirrors TMU's `vanilla_classifier.py`. |
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
pub mod data;
pub mod encoder;
pub mod models;
mod rng;

pub use booleanizer::Booleanizer;
pub use encoder::{EncodedBatch, EncodedSample, Encoder};
pub use models::TMAutoEncoder;
pub use models::TsetlinMachine;
pub use rng::Rng;
