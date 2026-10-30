//! CUDA (GPU) training and inference for every model (`--features cuda`).
//!
//! Each CPU model has a GPU counterpart that owns the model's learned state in
//! device memory:
//!
//! | CPU model | GPU model |
//! |---|---|
//! | [`TsetlinMachine`](crate::TsetlinMachine) | [`CudaTsetlinMachine`] |
//! | [`CoalescedTsetlinMachine`](crate::CoalescedTsetlinMachine) | [`CudaCoalescedTsetlinMachine`] |
//! | [`ConvolutionalTsetlinMachine`](crate::ConvolutionalTsetlinMachine) | [`CudaConvolutionalTsetlinMachine`] |
//! | [`TMSparseClassifier`](crate::TMSparseClassifier) | [`CudaTMSparseClassifier`] |
//! | [`TMCompositeClassifier`](crate::TMCompositeClassifier) | [`CudaTMCompositeClassifier`] |
//! | [`TMRegressor`](crate::TMRegressor) | [`CudaTMRegressor`] |
//! | [`TMAutoEncoder`](crate::TMAutoEncoder) | [`CudaTMAutoEncoder`] |
//! | [`TMCoalescedAutoEncoder`](crate::TMCoalescedAutoEncoder) | [`CudaTMCoalescedAutoEncoder`] |
//!
//! Configure a model on the CPU as usual, move it to the GPU with `to_cuda`,
//! train / predict there, and bring it back with `to_cpu` (for saving,
//! interpretability, or further CPU use):
//!
//! ```rust,no_run
//! use tmu_rs::cuda::CudaDevice;
//! use tmu_rs::{Encoder, TsetlinMachine};
//!
//! # let (raw_x, ys): (Vec<Vec<u8>>, Vec<usize>) = (vec![], vec![]);
//! let dev = CudaDevice::new(0)?;
//! let enc = Encoder::for_binary(12);
//! let rows: Vec<&[u8]> = raw_x.iter().map(|r| r.as_slice()).collect();
//! let batch = enc.encode_batch(&rows);
//!
//! let mut tm = TsetlinMachine::new(2, 12, 1000, 500, 3.9).to_cuda(&dev)?;
//! for _ in 0..10 {
//!     tm.fit_epoch(&batch, &ys)?;
//! }
//! println!("accuracy {}", tm.accuracy(&batch, &ys)?);
//! let cpu_model = tm.to_cpu()?; // e.g. to save or inspect clauses
//! # Ok::<(), tmu_rs::cuda::CudaError>(())
//! ```
//!
//! ## Exactness
//!
//! GPU training is **bit-identical** to the CPU model's exact (default)
//! training path: the kernels consume the same per-clause / per-class random
//! streams in the same order and apply the same saturating TA arithmetic, so a
//! model trained on the GPU and copied back with `to_cpu` equals the CPU-trained
//! model for the same seed, and predictions agree exactly. (The CPU's opt-in
//! `data_parallel` mode is approximate and has no GPU equivalent; the GPU always
//! trains exactly.)
//!
//! ## Requirements
//!
//! An NVIDIA GPU with driver, plus NVRTC (`libnvrtc.so` / `nvrtc64_*.dll` from the
//! CUDA toolkit, or the `nvidia-cuda-nvrtc-cu12` pip wheel). Both are loaded at
//! runtime, so the crate *builds* without them. Override library locations with
//! `TMU_CUDA_DRIVER_PATH` / `TMU_NVRTC_PATH`.
//!
//! Without a GPU, `--features cuda-emu` provides [`CudaDevice::emulator`], which
//! runs the identical kernel source on the CPU (slowly) — used by the test suite.
//!
//! ## Performance notes
//!
//! Training is inherently sequential over samples (each update depends on the
//! previous one), so the GPU parallelises *within* a sample: across clauses and,
//! inside each clause, across 64-literal words. Expect large speed-ups for
//! models with many clauses and/or wide inputs, and little or none for tiny
//! models where kernel-launch latency (~2 launches per sample) dominates.
//! Batch inference parallelises over samples *and* clauses.

mod common;
mod device;
mod driver;
mod params;

mod autoencoder;
mod coalesced;
mod coalesced_autoencoder;
mod composite;
mod convolutional;
mod regressor;
mod resident;
mod sparse;
mod vanilla;

// `TMU_TEST_GPU=1 cargo test --features cuda` runs these on CUDA device 0;
// `cargo test --features cuda-emu` runs them on the CPU kernel emulator.
#[cfg(test)]
mod tests;

pub use autoencoder::CudaTMAutoEncoder;
pub use coalesced::CudaCoalescedTsetlinMachine;
pub use coalesced_autoencoder::CudaTMCoalescedAutoEncoder;
pub use composite::CudaTMCompositeClassifier;
pub use convolutional::CudaConvolutionalTsetlinMachine;
pub use device::CudaDevice;
pub use regressor::CudaTMRegressor;
pub use resident::CudaEncodedBatch;
pub use sparse::CudaTMSparseClassifier;
pub use vanilla::CudaTsetlinMachine;

/// Errors from the CUDA backend.
#[derive(Debug, Clone)]
pub enum CudaError {
    /// The CUDA driver or NVRTC library could not be loaded.
    Load(String),
    /// No (or not enough) CUDA devices.
    NoDevice(String),
    /// NVRTC failed to compile the kernels (includes the compiler log).
    Compile(String),
    /// A CUDA driver call failed.
    Driver {
        /// The failing call (or kernel name).
        call: String,
        /// The `CUresult` code.
        code: i32,
        /// The driver's error string.
        msg: String,
    },
    /// The operation is not valid for this model / device combination.
    Invalid(String),
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::Load(m) => write!(f, "CUDA library load failed: {m}"),
            CudaError::NoDevice(m) => write!(f, "no CUDA device: {m}"),
            CudaError::Compile(m) => write!(f, "CUDA kernel compilation failed: {m}"),
            CudaError::Driver { call, code, msg } => {
                write!(f, "CUDA error in {call}: {msg} (code {code})")
            }
            CudaError::Invalid(m) => write!(f, "invalid CUDA operation: {m}"),
        }
    }
}

impl std::error::Error for CudaError {}
