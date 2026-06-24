//! Classification models.
//!
//! Mirrors TMU's `tmu/models/classification/` package.

mod coalesced_classifier;
mod vanilla_classifier;

pub use coalesced_classifier::CoalescedTsetlinMachine;
pub use vanilla_classifier::TsetlinMachine;
