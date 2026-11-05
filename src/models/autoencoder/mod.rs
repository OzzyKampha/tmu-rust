//! Tsetlin Machine autoencoder model.
//!
//! Mirrors TMU's `tmu/models/autoencoder/` package.

mod coalesced_autoencoder;
mod vanilla_autoencoder;

pub use coalesced_autoencoder::{HeldOutVotes, TMCoalescedAutoEncoder};
pub use vanilla_autoencoder::TMAutoEncoder;
