# tmu-rs

A Rust port of the [cair/tmu](https://github.com/cair/tmu) Tsetlin Machine library.

Implements the core Tsetlin Machine variants — multiclass classifier, coalesced classifier, regressor, convolutional (1-D and 2-D), autoencoder, composite classifier, and a sparse classifier with absorbing actions — with bit-packed clause banks, bit-parallel training, optional Rayon multi-threading, and a fast type-safe booleanizer.

For a full breakdown of what has been ported and what is missing, see [PORTING_STATUS.md](PORTING_STATUS.md). For a Python vs Rust throughput and accuracy comparison, see [BENCHMARKS.md](BENCHMARKS.md).

---

## Use as a library dependency

Add to your project's `Cargo.toml`:

```toml
[dependencies]
tmu-rs = { git = "https://github.com/ozzykampha/tmu-rust" }

# Optional: pin to a specific tag for reproducible builds
# tmu-rs = { git = "https://github.com/ozzykampha/tmu-rust", tag = "v1.0.0" }

# Optional: enable multi-threaded training
# tmu-rs = { git = "https://github.com/ozzykampha/tmu-rust", features = ["parallel"] }
```

Then use it:

```rust
use tmu_rs::{TsetlinMachine, Encoder};

// Build encoder from training data (binary features in this example)
let encoder = Encoder::binary(n_features);
let train_x = encoder.encode_batch(&raw_train_x);

// Create and train the classifier
let mut tm = TsetlinMachine::with_config(
    n_classes, clauses_per_class, n_features,
    threshold, specificity, max_states, boost_true_positive, seed,
);
for _ in 0..epochs {
    tm.fit_epoch(&train_x, &train_y);
}
let accuracy = tm.accuracy(&test_x, &test_y);
```

---

## Features

- Bit-packed clause bank for cache-efficient inference and training
- **Five model types**:
  - `TMClassifier` — weighted multiclass classification
  - `TMCoalescedClassifier` — one shared clause bank with signed per-class weights
  - `TMRegressor` — continuous output from binary features
  - `ConvolutionalTsetlinMachine` — 1-D and 2-D sliding-window clause banks (weight-tied patches)
  - `TMCompositeClassifier` — ensemble of per-class Tsetlin Machines with independent clause banks
  - `TMAutoEncoder` — binary reconstruction via positive-only clause banks
  - `TMSparseClassifier` — sparse clause bank with **absorbing actions**: literals are permanently dropped from each clause as training converges, so memory and per-clause evaluation scale with the number of *active* literals (a big win in high-dimensional, sparsely-relevant feature spaces)
- Optional multi-threaded training via [Rayon](https://github.com/rayon-rs/rayon) (`--features parallel`)
- Optional **CUDA GPU training and inference for every model** (`--features cuda`) — bit-identical to CPU training; see [GPU (CUDA)](#gpu-cuda)
- AVX2 fast paths for clause update loops with runtime dispatch — u8 TA counters processed 32-wide (4× smaller working set vs u32; scalar fallback on non-AVX2 targets)
- Type-safe `Encoder` for binary, numeric (quantile booleanization), and categorical inputs
- Fast booleanizer for continuous-valued inputs
- Save/load trained models and encoders to disk via the `SaveLoad` trait (on by default through the `serde` feature) — serde + bincode preserve all learned state **and** RNG streams, so a reloaded model predicts identically and can resume training deterministically. Build with `--no-default-features` for a dependency-free build without save/load
- Ports of the core TMU demos including classification, regression, convolution, autoencoder, and composite

---

## Getting started

```sh
git clone --recurse-submodules https://github.com/OzzyKampha/tmu-rust.git
cd tmu-rust
cargo build --release
```

Run a self-contained example:

```sh
cargo run --release --example noisy_xor
```

For multi-threaded training:

```sh
cargo run --release --features parallel --example mnist
```

For maximum performance, compile with native CPU optimizations (enables AVX2 and other extensions at compile time):

```sh
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

AVX2 fast paths are also activated at runtime automatically when the CPU supports it, even without `target-cpu=native`.

---

## GPU (CUDA)

Every model can train and predict on an NVIDIA GPU. Configure it on the CPU as usual, move it with `to_cuda`, and bring it back with `to_cpu` (e.g. to save it or inspect clauses):

```rust
use tmu_rs::cuda::CudaDevice;

let dev = CudaDevice::new(0)?;                                   // first GPU
let mut tm = TsetlinMachine::new(10, 784, 2000, 50, 10.0).to_cuda(&dev)?;
for _ in 0..epochs {
    tm.fit_epoch(&train_x, &train_y)?;
}
let accuracy = tm.accuracy(&test_x, &test_y)?;
let cpu_model = tm.to_cpu()?;
```

| CPU model | GPU model |
|---|---|
| `TsetlinMachine` | `CudaTsetlinMachine` |
| `CoalescedTsetlinMachine` | `CudaCoalescedTsetlinMachine` |
| `ConvolutionalTsetlinMachine` (1-D and 2-D) | `CudaConvolutionalTsetlinMachine` |
| `TMSparseClassifier` | `CudaTMSparseClassifier` |
| `TMCompositeClassifier` | `CudaTMCompositeClassifier` |
| `TMRegressor` | `CudaTMRegressor` |
| `TMAutoEncoder` | `CudaTMAutoEncoder` |
| `TMCoalescedAutoEncoder` | `CudaTMCoalescedAutoEncoder` |

- **Exact.** GPU training is bit-identical to the CPU model's (default, exact) training for the same seed — every option is supported, including dropout, `max_included_literals`, class weights, focused negative sampling and Type III feedback.
- **No build-time CUDA dependency.** The driver (`libcuda`) and NVRTC (`libnvrtc`, from the CUDA toolkit or the `nvidia-cuda-nvrtc-cu12` pip wheel) are loaded at runtime and the kernels are JIT-compiled for your GPU on `CudaDevice::new`. Set `TMU_CUDA_DRIVER_PATH` / `TMU_NVRTC_PATH` for non-standard install locations.
- **Where it helps.** Training is sequential over samples, so the GPU parallelises within each sample (across clauses and 64-literal words). Expect the biggest wins for many clauses and/or wide inputs; tiny models are dominated by kernel-launch latency. Batch inference parallelises over samples and clauses.
- **Testing without a GPU.** `--features cuda-emu` adds `CudaDevice::emulator()`, which runs the identical kernel source on the CPU (slowly). `cargo test --features cuda-emu` checks every GPU model is bit-identical to its CPU counterpart; on a GPU machine, `TMU_TEST_GPU=1 cargo test --features cuda-emu` runs the same checks on the real device.

```sh
cargo run --release --features cuda --example cuda              # CPU vs GPU timing + identity check
cargo run --release --features cuda-emu --example cuda -- --emu # same, on the kernel emulator
```

---

## Examples

The examples reproduce the [`cair/tmu`](https://github.com/cair/tmu) demos with matching hyperparameters (e.g. MNIST: 2000 clauses, T=50, s=10.0; IMDb: 2000 clauses, T=80, s=10.0). See [PORTING_STATUS.md](PORTING_STATUS.md) for the full status.

**Classification demos**

| TMU demo | Example | Data required | Command |
|---|---|---|---|
| `XORDemo` | `xor` | — | `cargo run --release --example xor` |
| `NoisyXORDemo` | `noisy_xor` | — | `cargo run --release --example noisy_xor` |
| `InterpretabilityDemo` | `interpretability` | — | `cargo run --release --example interpretability` |
| `TMCoalescedClassifier` | `coalesced` | — | `cargo run --release --example coalesced` |
| `TMSparseClassifier` | `sparse` | — | `cargo run --release --example sparse` |
| `BreastCancerDemo` | `breast_cancer` | scikit-learn | see [Data preparation](#data-preparation) |
| `MNISTDemo` / `MNISTDemoWeightedClauses` | `mnist` | MNIST | see [Data preparation](#data-preparation) |
| `IMDbTextCategorizationDemo` | `imdb` | Keras IMDb | see [Data preparation](#data-preparation) |

**New model types** (run `python scripts/gen_shared_data.py` once first to generate shared data)

| Model | Example | Command |
|---|---|---|
| `TMRegressor` | `regression` | `cargo run --release --example regression` |
| `ConvolutionalTM` 1-D | `convolutional` | `cargo run --release --example convolutional` |
| `ConvolutionalTM` 2-D | `convolutional_2d` | `cargo run --release --example convolutional_2d` |
| `TMCompositeClassifier` | `composite` | `cargo run --release --example composite` |
| `TMAutoEncoder` | `autoencoder` | `cargo run --release --example autoencoder` |
| `TMCoalescedAutoEncoder` | `coalesced_autoencoder` | `cargo run --release --example coalesced_autoencoder` |

**Extras**

| Example | Description |
|---|---|
| `sparse_vs_dense` | Dense vs sparse head-to-head: accuracy parity, memory footprint, train/inference time |
| `save_load` | Train → save → load → predict/resume round-trip |
| `ndr_flows` | Synthetic network-flow detection (booleanizer + rule extraction) |
| `sysmon` / `sysmon_windows` / `sysmon_mordor` | Sysmon event classification |
| `bench_training` | Training throughput benchmark (sequential vs parallel, IMDB-scale) |
| `bench_autoencoder` | AutoEncoder throughput + accuracy vs Python TMU |
| `absorb_timing` | Per-epoch accuracy and absorbing-state fraction at various `state_bits` |
| `cuda` | GPU vs CPU training time and bit-identity check (`--features cuda`) |

`bench_training` uses a synthetic dataset — no download required. Compare with and without `--features parallel`.

---

## Data preparation

Three examples require datasets generated by the Python scripts in `scripts/`. Generated files are written to `data/` and are not tracked by git.

**Breast Cancer** (requires `scikit-learn`):
```sh
pip install scikit-learn
python scripts/prepare_breast_cancer.py
cargo run --release --example breast_cancer
```

**MNIST** (requires `tensorflow` or `scikit-learn`):
```sh
python scripts/prepare_mnist.py
cargo run --release --features parallel --example mnist
```

**IMDb** (requires `tensorflow`):
```sh
python scripts/prepare_imdb.py
cargo run --release --features parallel --example imdb
```

**Regression / Convolutional / Composite shared data** (requires `numpy`):
```sh
pip install numpy
python scripts/gen_shared_data.py        # writes 14 binary files to data/
cargo run --release --example regression
cargo run --release --example convolutional
cargo run --release --example convolutional_2d
cargo run --release --example composite
```

---

## Project structure

```
src/
  encoder.rs              # Type-safe Encoder (binary / numeric / categorical)
  booleanizer.rs          # Quantile booleanization (used by Encoder)
  clause_bank/            # Bit-packed clause storage and update logic
  models/
    classification/
      vanilla_classifier.rs      # TMClassifier
      coalesced_classifier.rs    # TMCoalescedClassifier
      convolutional_classifier.rs # ConvolutionalTsetlinMachine (1-D and 2-D)
      composite_classifier.rs    # TMCompositeClassifier
    regression/
      vanilla_regressor.rs       # TMRegressor
    autoencoder/
      vanilla_autoencoder.rs     # TMAutoEncoder
      coalesced_autoencoder.rs   # TMCoalescedAutoEncoder
  rng.rs                  # Fast SplitMix64 RNG
  cuda/                   # GPU backend (--features cuda)
    kernels.cu            # CUDA kernels for every model (NVRTC-compiled at runtime)
    emu.h                 # Shim to run the same kernels on the CPU (--features cuda-emu)
    driver.rs, device.rs  # Runtime-loaded CUDA driver / NVRTC, device buffers, launches
    *.rs                  # One GPU model wrapper per CPU model
examples/               # Demo programs (ports of TMU + extras)
benches/                # Criterion throughput benchmarks
scripts/                # Python data preparation and comparison scripts
data/tmu/               # cair/tmu submodule (reference implementation)
```

---

## License

MIT

Original TMU library: [cair/tmu](https://github.com/cair/tmu) (MIT).
