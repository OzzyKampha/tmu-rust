//! Classification models.
//!
//! Mirrors TMU's `tmu/models/classification/` package.

mod coalesced_classifier;
mod composite_classifier;
mod convolutional_classifier;
mod vanilla_classifier;

pub use coalesced_classifier::CoalescedTsetlinMachine;
pub use composite_classifier::TMCompositeClassifier;
pub use convolutional_classifier::ConvolutionalTsetlinMachine;
pub use vanilla_classifier::TsetlinMachine;
