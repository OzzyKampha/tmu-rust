//! Tsetlin Machine model implementations.
//!
//! Mirrors TMU's `tmu/models/` package.

pub mod autoencoder;
pub mod classification;

pub use autoencoder::{TMAutoEncoder, TMCoalescedAutoEncoder};
pub use classification::{CoalescedTsetlinMachine, TsetlinMachine};
