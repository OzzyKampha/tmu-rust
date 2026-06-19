//! Clause bank implementations.
//!
//! Mirrors TMU's `tmu/clause_bank/` package.  Each sub-module is a distinct
//! hardware/memory layout; future variants (sparse, coalesced, CUDA) live here
//! alongside `dense`.

pub(crate) mod dense;
