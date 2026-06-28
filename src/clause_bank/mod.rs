//! Clause bank implementations.
//!
//! Mirrors TMU's `tmu/clause_bank/` package.  Each sub-module is a distinct
//! hardware/memory layout; further variants (coalesced, CUDA) can live here
//! alongside `dense` and `sparse`.

pub(crate) mod dense;
pub(crate) mod sparse;
