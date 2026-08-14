# Changelog

All notable changes to tmu-rs are documented here.

---

## [Unreleased]

### Changed
- **Data-parallel training is now automatic** for `TsetlinMachine`. Built with
  `--features parallel` on more than one core, `fit_epoch` picks a data-parallel path
  once an epoch has ≥ 256 samples and each sample touches ≥ 8 KiB of TA state. On that
  path training is **approximate**: results are not bit-identical to sequential
  training and depend on the shard count. Smaller models, single-core builds and
  Type III feedback continue to train exactly.
  Three ways to opt out: `.data_parallel(false)`, the new `fit_epoch_exact()`, or
  `TMU_RS_DATA_PARALLEL=0`. A one-line notice is printed to stderr the first time auto
  engages (silence it with `TMU_RS_DP_QUIET=1`).
- **Data-parallel training merges per mini-batch instead of once per epoch**, and
  reuses cached replicas rather than cloning the model every epoch. Rayon is now
  dispatched roughly twice per mini-batch instead of twice per *sample*, and the merge
  streams each replica sequentially (fusing the weight merge and `rebuild_include`
  into one cache-resident pass) instead of gathering across replicas per element.
- **`data_parallel(true)` is now honoured at any model size.** It previously also
  required `use_parallel(n_clauses, words)`, which silently ignored the opt-in for
  small models — the `data_parallel_learns_xor` tests had never actually exercised the
  data-parallel path.
- **Type III feedback no longer panics under `data_parallel(true)`**; it declines the
  path and trains exactly. The two `assert!(!self.type_iii, ...)` calls are gone, so
  automatic selection can never panic on a flag the user did not set.
- **`TMCompositeClassifier` trains its ensemble members concurrently.** Unlike the
  in-model data-parallel path this is **exact** — members share no mutable state and
  each draws its shuffle from its own RNG — and it is asserted clause-by-clause
  against serial training.

### Performance
- Dense TM training is memory-bandwidth bound, so multicore speedups are modest
  regardless of strategy. On a 4-core Xeon (33 MiB L3, min of 9 epochs) the
  data-parallel path lands within ~10% of the exact clause-parallel path across model
  sizes from 0.03 MiB to 33.6 MiB of TA state. The merge cadence is the setting that
  actually moves the number: at a 2.1 MiB model, merging every 256 samples measures
  1.14× versus exact, while merging every 32 samples measures **0.46×**.
- `examples/parallel_scaling.rs` now reports the automatic decision alongside the
  exact and forced columns, adds a merge-cadence sweep, and takes the **minimum** of 9
  epochs rather than the median of 3 — the previous sampling was not reproducible on
  shared hardware (±20% between runs, wider than most effects it measured).
- Removed a full copy of the encoded batch that `TMSparseClassifier::fit_epoch` made
  on **every epoch**, and a per-sample copy in its `fit_one`.
- `TMCompositeClassifier::predict_batch` no longer allocates an `EncodedSample` per
  sample per member, and runs sample-parallel.

### Added
- `TsetlinMachine::fit_epoch_exact`, `data_parallel_auto`, `data_parallel_shards`,
  `data_parallel_batch` and `clear_replica_cache`. Pinning shards and batch makes the
  approximate path bit-reproducible across machines with different core counts.
- `TMCompositeClassifier::data_parallel` to force member-parallel training on or off.
- Environment overrides: `TMU_RS_DATA_PARALLEL`, `TMU_RS_DP_SHARDS`,
  `TMU_RS_DP_BATCH`, `TMU_RS_DP_MERGE`, `TMU_RS_DP_QUIET`.
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
