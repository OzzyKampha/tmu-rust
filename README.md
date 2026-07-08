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
- Optional **GPU training and inference** for `TMClassifier` via portable [wgpu](https://github.com/gfx-rs/wgpu)/WGSL compute (`--features gpu`) — runs on any Vulkan/Metal/DX12 adapter, **bitwise-identical to CPU training** so models trained on the GPU can be run (or resumed) on the CPU and vice versa. See [GPU acceleration](#gpu-acceleration-training--inference)
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

## GPU acceleration (training + inference)

The `gpu` feature adds a portable GPU backend for the vanilla `TMClassifier`
(`TsetlinMachine`), built on [wgpu](https://github.com/gfx-rs/wgpu) / WGSL
compute shaders. It runs on any Vulkan, Metal, or DX12 adapter — NVIDIA, AMD,
Intel, or Apple — with **no CUDA toolkit required**, and falls back cleanly to a
software Vulkan driver (mesa llvmpipe) for CI.

```toml
tmu-rs = { git = "https://github.com/ozzykampha/tmu-rust", features = ["gpu"] }
```

```rust
use std::sync::Arc;
use tmu_rs::{TsetlinMachine, GpuContext};

let ctx = Arc::new(GpuContext::new()?);          // Err if no adapter is available
let tm = TsetlinMachine::with_config(2, n_features, 64, 15, 3.9, 8, true, 42);

let mut gpu = tm.to_gpu(&ctx)?;                   // move a copy onto the GPU
for _ in 0..epochs {
    gpu.fit_epoch(&train, &ytr);                 // train on the GPU
}
let preds = gpu.predict_batch(&test);            // infer on the GPU
let cpu_model = gpu.into_cpu();                   // ...or download and use the CPU
let _ = cpu_model.predict_batch(&test);          // (save/load work unchanged)
```

**Train anywhere, infer anywhere.** GPU training reproduces the per-clause
SplitMix64 RNG streams and feedback logic exactly, so for a given seed and
configuration the trained model (`ta`, `include`, `weights`, and all RNG state)
is **bit-for-bit identical** to CPU training. A model can be trained on the GPU
and run for inference on the CPU (or the reverse), and `save`/`load` are
unchanged — the model state lives in the same host struct; the GPU holds a
device-side copy synced at boundaries (`sync` / `into_cpu`).

Supported today: `boost_true_positive`, `max_included_literals`, `class_weights`,
`state_bits` (2–8), and `literal_drop_p`. `to_gpu` returns
`GpuError::Unsupported` (rather than silently falling back) for
`type_iii_feedback` and `clause_drop_p > 0`. To grow features, `into_cpu()`,
grow, then `to_gpu()` again.

**Too big for VRAM? It still trains.** `to_gpu` never fails because a model is
too large — if the model doesn't fit in GPU memory it stays on the CPU and
`fit_epoch` / `predict_batch` run there transparently (check
`GpuTsetlinMachine::is_gpu_resident()`). Data-parallel likewise scales the
replica count down to what fits, falling back to the exact GPU path (or, if even
the single model doesn't fit, the CPU). The only hard error is
`GpuError::Unsupported` for the options listed above.

**Two training modes.** The default GPU path is *exact* (bitwise-identical to
CPU) but processes samples sequentially — latency-bound, so it only beats a
multi-threaded CPU on large models. For a bigger speedup, build the model with
[`data_parallel(true)`](crate::TsetlinMachine::data_parallel): the GPU then
trains `R` model replicas in lockstep on sample shards and averages them
(mirroring the CPU `data_parallel` path), which cuts kernel launches ~`R×` and
fills the device. Like the CPU flag, this is **approximate** (accuracy tracks
exact within noise) and replica-count dependent, but deterministic for a given
seed and `R`. The replica count is chosen automatically from available VRAM;
override it with `GpuTsetlinMachine::set_replicas(Some(r))` (dynamic — takes
effect on the next epoch). Inference is identical in both modes.

```rust
let tm = TsetlinMachine::with_config(10, n_features, 2000, 50, 5.0, 8, true, 42)
    .data_parallel(true);                 // opt into the fast (approximate) GPU path
let mut gpu = tm.to_gpu(&ctx)?;
gpu.set_replicas(Some(16));               // optional: pin the replica count
for _ in 0..epochs { gpu.fit_epoch(&train, &ys); }
```

### Try it / benchmark it

```sh
# First: confirm your GPU (not a software fallback) is selected, and see limits:
cargo run --release --features gpu --example gpu_probe

# Train noisy XOR on the GPU, then evaluate on both GPU and CPU:
cargo run --release --features gpu,serde --example gpu_xor

# CPU-vs-GPU training/inference throughput comparison (checks parity first):
cargo run --release --features gpu          --example gpu_vs_cpu_bench
cargo run --release --features gpu,parallel --example gpu_vs_cpu_bench
```

The benchmark prints a per-stage table (`train` / `infer`, ms and samples/s)
with the GPU speedup, and reports the one-time upload cost separately. The GPU
wins on **large** models (many clauses × many features × large batches), where
there is enough parallel work to hide launch and transfer overhead; on tiny
models the CPU is faster. Real speedups require real hardware — on the mesa
llvmpipe software driver the "GPU" is CPU emulation (no AVX2), so treat those
numbers as a functional check, not a hardware comparison.

**Testing:** GPU tests live in `tests/gpu_parity.rs` (plus RNG bit-exactness
unit tests) and **skip cleanly** when no adapter is present. To exercise them in
a headless environment, install a software Vulkan driver (e.g.
`mesa-vulkan-drivers`) and run:

```sh
cargo test --features gpu,serde --test gpu_parity
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
| `gpu_probe` | GPU: print the selected adapter, driver, and limits — confirm your GPU is used (needs `--features gpu`) |
| `gpu_xor` | GPU: train noisy XOR on the GPU, evaluate on GPU and CPU, save/load (needs `--features gpu,serde`) |
| `gpu_vs_cpu_bench` | CPU-vs-GPU training/inference throughput comparison (needs `--features gpu`) |

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
examples/               # Demo programs (ports of TMU + extras)
benches/                # Criterion throughput benchmarks
scripts/                # Python data preparation and comparison scripts
data/tmu/               # cair/tmu submodule (reference implementation)
```

---

## License

MIT

Original TMU library: [cair/tmu](https://github.com/cair/tmu) (MIT).
