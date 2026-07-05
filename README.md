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
- **Growing feature space** — extend a trained model with new features as the encoder learns new symbols, *without discarding learned automata*: `Encoder::extend_categorical` / `grow_binary` paired with `grow_features` on the vanilla, coalesced, and sparse classifiers. Predictions on previously-seen inputs stay bit-identical across the grow; the new features are immediately learnable
- Optional multi-threaded training via [Rayon](https://github.com/rayon-rs/rayon) (`--features parallel`): work-aware parallel inference, plus an opt-in approximate **data-parallel** training mode (`.data_parallel(true)`, ~2–3× on multiple cores) — while `fit_epoch` stays exact and deterministic by default
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

## Growing a trained model

When new data introduces symbols the encoder has never seen, grow the vocabulary and the machine instead of retraining from scratch. Existing feature indices stay stable and every learned clause is preserved, so predictions on previously-seen inputs are unchanged:

```rust
// New events arrive with tokens not in the original vocabulary.
let added = encoder.extend_categorical(&new_samples); // append new tokens as new features
if added > 0 {
    tm.grow_features(encoder.n_features());            // widen the machine to match
}
// Re-encode with the grown encoder, then keep training — old detections intact,
// new features learned. Works on TsetlinMachine, CoalescedTsetlinMachine, TMSparseClassifier.
```

See `examples/growing.rs` (a Sysmon detector that learns new adversary tooling incrementally) and `examples/grow_scaling.rs` (dense vs sparse as the literal space grows to 1M features).

## Faster training (optional, approximate)

`fit_epoch` is **exact and deterministic by default**. On a multi-core build (`--features parallel`) you can opt into an approximate **data-parallel** epoch — samples are sharded across cores, each trains a replica, and the replicas are merged — typically ~2–3× faster, by setting one flag; you still call `fit_epoch`:

```rust
let mut tm = TsetlinMachine::with_config(/* … */).data_parallel(true);
tm.fit_epoch(&train_x, &train_y); // shards across cores when the model is large enough
```

Trade-off: results are no longer bit-identical to sequential training (accuracy tracks exact within noise). Leave it off for exact, reproducible runs. Training a large model on the exact path prints a one-time hint suggesting the flag. Validate accuracy with `examples/dp_vs_exact.rs`; measure throughput with `examples/parallel_scaling.rs`.

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
| `growing` | Grow a trained detector's vocabulary + literal space on new data, keeping learned automata (Sysmon story) |
| `grow_scaling` | Dense vs sparse literal-space growth to 1M features; grow latency, memory, inference throughput |
| `parallel_scaling` | Training ms/epoch vs clauses × literals; exact vs `data_parallel`, scalar vs `--features parallel` |
| `dp_vs_exact` | Validate `data_parallel(true)` accuracy against exact training on real data (breast cancer) |
| `sparse_vs_dense` | Dense vs sparse head-to-head: accuracy parity, memory footprint, train/inference time |
| `save_load` | Train → save → load → predict/resume round-trip |
| `ndr_flows` | Synthetic network-flow detection (booleanizer + rule extraction) |
| `sysmon` / `sysmon_windows` / `sysmon_mordor` | Sysmon event classification |
| `bench_training` | Training throughput benchmark (sequential vs parallel, IMDB-scale) |
| `bench_autoencoder` | AutoEncoder throughput + accuracy vs Python TMU |
| `absorb_timing` | Per-epoch accuracy and absorbing-state fraction at various `state_bits` |

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
