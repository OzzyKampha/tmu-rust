//! GPU backend for the vanilla [`TsetlinMachine`](crate::TsetlinMachine).
//!
//! Portable GPU **training** and **inference** built on [`wgpu`] / WGSL compute
//! shaders — it runs on any Vulkan / Metal / DX12 adapter (NVIDIA, AMD, Intel,
//! Apple), and on a software Vulkan driver (mesa llvmpipe) for CI. No CUDA
//! toolkit is required.
//!
//! # Guarantees
//!
//! GPU training is **bitwise identical** to CPU training for the same seed and
//! configuration: it reproduces the per-clause SplitMix64 RNG streams and the
//! exact feedback logic. A model trained on the GPU is therefore fully
//! interchangeable with a CPU model — you can train on the GPU and run inference
//! on the CPU (or vice versa), and [`save`](crate::SaveLoad::save) /
//! [`load`](crate::SaveLoad::load) work unchanged because the model state lives
//! in the same host-side [`TsetlinMachine`] struct; the GPU merely holds a
//! device-side copy that is synced at boundaries.
//!
//! # Quick start
//!
//! ```no_run
//! use std::sync::Arc;
//! use tmu_rs::{TsetlinMachine, GpuContext};
//! # use tmu_rs::{Encoder, EncodedBatch};
//! # fn demo(batch: &EncodedBatch, ys: &[usize], test: &EncodedBatch) -> Result<(), Box<dyn std::error::Error>> {
//! let ctx = Arc::new(GpuContext::new()?);          // Err if no adapter is available
//! let tm = TsetlinMachine::with_config(2, 12, 64, 15, 3.9, 8, true, 42);
//! let mut gpu = tm.to_gpu(&ctx)?;
//! for _ in 0..20 {
//!     gpu.fit_epoch(batch, ys);                    // train on the GPU
//! }
//! let preds = gpu.predict_batch(test);             // infer on the GPU
//! let cpu_model = gpu.into_cpu();                  // ...or fall back to the CPU
//! let _ = cpu_model.predict_batch(test);
//! # let _ = preds; Ok(())
//! # }
//! ```
//!
//! # Unsupported options (v1)
//!
//! [`to_gpu`](crate::TsetlinMachine::to_gpu) returns [`GpuError::Unsupported`]
//! rather than silently falling back when the model uses
//! [`type_iii_feedback`](crate::TsetlinMachine::type_iii_feedback) or
//! `clause_drop_p > 0`. Grow features on the CPU model, then move it to the GPU.

mod context;

pub use context::{GpuContext, GpuError};

mod buffers;
mod vanilla;

pub use vanilla::GpuTsetlinMachine;
pub(crate) use vanilla::GpuEpochPlan;

#[cfg(test)]
mod gpu_tests;
