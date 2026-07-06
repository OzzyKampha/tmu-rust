# Changelog

All notable changes to tmu-rs are documented here.

---

## [Unreleased]

### Added
- **TMSparseClassifier** — sparse clause bank with **absorbing actions** (port of
  TMU's `ClauseBankSparse`). Each clause stores included / excluded literal index
  lists instead of a dense per-literal counter array; literals that reach the
  absorbing exclude floor are permanently removed from the candidate pool, so
  memory and per-clause evaluation scale with the number of *active* literals.
  Vanilla-classifier API parity (weighted clauses, Type I/II feedback, dropout,
  `max_included_literals`, class weights, interpretability, save/load with
  `TAG_SPARSE = 9`).
- **`TMSparseClassifier` Rayon support** — `--features parallel` parallelises
  training over clauses (lock-free; bit-identical to scalar) and inference over
  samples, gated by `PARALLEL_MIN` like the dense model. As with dense, parallel
  training pays off only at large clause counts. AVX2 is intentionally not used for
  the sparse bank (the gather/RNG/`swap_remove` hot path doesn't vectorise; upstream
  `cair/tmu`'s sparse C is also scalar). Type III feedback remains unsupported.
- **`examples/sparse.rs`** — noisy-XOR demo showing absorbing literal removal.
- **`examples/sparse_vs_dense.rs`** — dense vs sparse head-to-head (accuracy,
  memory footprint, train/inference time); see the "Dense vs Sparse" section in
  `BENCHMARKS.md`.

### Changed
- **Rust 2024 edition** — the crate now builds with `edition = "2024"` and declares
  `rust-version = "1.96"` (the latest stable toolchain; 2024 edition itself needs
  only ≥ 1.85).
  Migration notes: bodies of AVX2 `unsafe fn`s are wrapped in explicit `unsafe {}`
  blocks (2024 no longer treats an `unsafe fn` body as one big unsafe block), the
  `gen` identifier (now a reserved keyword) was renamed, and two closure patterns
  were adjusted for the 2024 match-ergonomics rules. No behavior changes.

---

## [1.0.0] — 2026-06-25

### Added
- **TMRegressor** — continuous output Tsetlin Machine; clause votes sum to a weighted
  score clipped to `[−T, T]`; feedback matches TMU's `vanilla_regressor.py` exactly.
- **ConvolutionalTsetlinMachine** — sliding-window clause bank with weight tying:
  - 1-D convolution: `with_config(n_classes, n_features, kernel, stride, ...)`.
  - 2-D convolution: `with_config_2d(n_classes, input_rows, input_cols, patch_rows, patch_cols, stride, ...)`.
  - OR semantics: a clause fires if it fires on any patch position.
  - New accessors: `patch_rows()`, `patch_cols()`, `input_rows()`, `input_cols()`.
- **TMCompositeClassifier** — ensemble of independent per-class Tsetlin Machines,
  each with its own clause bank and hyperparameters.
- **Shared binary test data** — `scripts/gen_shared_data.py` writes 14 raw
  little-endian binary files to `data/` (fixed numpy seeds). Rust examples and the
  Python comparison script load the same files, giving bit-identical train/test splits.
- **`scripts/compare_new_models.py`** — side-by-side accuracy comparison for all four
  new model types vs Python TMU; `--conv`, `--conv2d`, `--regressor`, `--composite` flags.
- **`scripts/compare_new.sh`** — shell wrapper for the full comparison; now covers
  all four models with `--conv2d` support and a gen_shared_data prerequisite note.
- **`examples/regression.rs`** — TMRegressor demo (count function, MAE reporting).
- **`examples/convolutional.rs`** — ConvolutionalTM 1-D demo (4-feat XOR, 3 patches).
- **`examples/convolutional_2d.rs`** — ConvolutionalTM 2-D demo (2×4 image, vertical XOR).
- **`examples/composite.rs`** — TMCompositeClassifier demo (4-class XOR).

### Changed
- README updated: Features section now lists all five model types; examples table
  expanded with regression, convolutional, autoencoder, and composite entries;
  outdated "not yet ported" note removed; project structure reflects actual layout.
- `ConvolutionalTsetlinMachine` struct gains `patch_rows`, `patch_cols`, `input_rows`,
  `input_cols`, `n_patch_cols` fields; `with_config()` sets 1-D defaults (backward-
  compatible); `pack_patch()` branches on `patch_rows == 1` for the 1-D fast path.

### Fixed
- Removed temporary `examples/conv_test.rs` diagnostic file.
- One `mut` warning in convolutional classifier (`firing` Vec is read-only after
  construction).

---

## [0.9.1] — prior release

### Added
- `SaveLoad` trait for persisting trained models and encoders (serde + bincode);
  enabled by default via the `serde` feature.  Reloaded models predict identically
  and can resume training deterministically (RNG state preserved).
- `TMCoalescedAutoEncoder` — shared clause bank with signed per-output weights.
- `save_load` example: train → save → load → predict/resume round-trip.

---

## [0.9.0]

### Added
- `TMAutoEncoder` — unsupervised Tsetlin Machine for binary reconstruction.
- `bench_autoencoder` example: throughput + accuracy vs Python TMU.
- `coalesced_autoencoder` example.

---

## [0.8.0]

### Added
- `TMCoalescedClassifier` — one shared clause bank with signed per-class weights.
- `coalesced` example.
- Imbalanced-class weighting support.

---

## [0.7.x and earlier]

- Initial port of `TMClassifier` with bit-packed clause bank, weighted multiclass
  training, AVX2 fast paths, type-safe `Encoder`, and ports of XOR, NoisyXOR,
  Interpretability, BreastCancer, MNIST, and IMDb demos.
- `--features parallel` (Rayon) for multi-threaded clause-bank training.
- `bench_training`, `absorb_timing`, `ndr_flows`, `sysmon*` extras.
- CI: GitHub Actions for test and cross-platform release builds.
