# Changelog

All notable changes to tmu-rs are documented here.

---

## [Unreleased]

### Added
- **Clause injection:** `TMSparseClassifier::inject_clause(class, j, literals,
  weight)` seeds a clause with prior knowledge (its literals included at the
  maximum state, its weight set) before or between training; feedback then
  refines it like any other clause. Every literal stays accounted for once
  (candidate, unallocated, included or absorbed).
- **`Encoder::token_bit(token)`:** a categorical encoder's bit for a token,
  the inverse of `vocab_token`.
- **`TMFuzzyPatternClassifier::ta_state_histogram()`** (the other classifiers'
  name and type, replacing `state_histogram()`) and **`inclusion_state()`**,
  the state from which a literal is included: the fuzzy machine's include
  limit need not be the middle state.

---

## [1.1.0] — 2026-11-06

### Added
- **Fuzzy-Pattern Tsetlin Machine:** `TMFuzzyPatternClassifier`, a port of
  Artem Hnilov's FPTM ([arXiv:2508.08350](https://arxiv.org/abs/2508.08350);
  reference code `BooBSD/FuzzyPatternTM`, MIT). A clause with `n` included
  literals scores `min(n, LF)` minus its failing literals, floored at 0, so a
  partly matching clause still votes; votes are unweighted (positive minus
  negative clauses) and predictions use them unclamped. Settings: `T`,
  specificity `S` (`round(features / S)` random decrements), `max_literals`
  (`L`), `literal_failures` (`LF`), `include_limit`. Deterministic per seed;
  with `--features parallel` the classes of a sample train in parallel with
  the same result. `scripts/check_fuzzy_pattern.py` retrains the reference
  algorithm (transliterated from `src/FuzzyPatternTM.jl`, same RNG) and
  compares every state: identical over four settings. Save/load (tag 10) and
  `ClauseInspect`, `grow_features`. Example: `fuzzy_pattern`.
- **TA state histograms:** `ta_state_histogram()` on `TsetlinMachine`,
  `CoalescedTsetlinMachine` and `TMSparseClassifier` counts the automata at
  each state `0..=max_state`, for watching a machine converge (how many sit at
  the exclude floor, how many include). On the sparse bank it counts the
  included and candidate lists; `TMSparseClassifier::absorbed_literals()`
  gives the exact number of literals absorbed out of them.
- **Held-out autoencoder scoring:** `held_out_scores(&sample, &mut out)` and
  `held_out_fired_clauses(&sample, output)` on `TMCoalescedAutoEncoder` and
  `TMAutoEncoder`. Each output's vote is taken with that output's own feature
  hidden, as training hides it; plain `reconstruct_scores` lets a clause look
  at the bit it votes on and echo it back. Through the literal → clause map,
  extended with each clause's single feature: a clause fires for output `o`
  when all its false literals are on `o` and it has a literal on another
  feature, so one pass gives every clause's verdict for every output. Tested
  against removing each output's literals from every clause and evaluating
  it (random banks with and without feature negation; trained models).
  `TMCoalescedAutoEncoder::held_out_votes(&sample)` returns the votes with
  every clause's verdict kept (`HeldOutVotes`: `scores()`,
  `fired_clauses(output)`, `strongest_clause(output, against)`), so
  explaining several outputs of one sample evaluates the clauses once.
- **`CudaTMCoalescedAutoEncoder::fit_examples`**: TMU's autoencoder training on
  the GPU, bit-identical to the CPU's `fit_examples` (every random stream
  included). The examples are drawn on the host with the CPU model's code; the
  GPU evaluates and updates the clauses, through the same kernels as
  `fit_epoch` (`cae_update` takes the drawn target). TMU's own CUDA clause
  bank builds the examples in a kernel, the same algorithm, but draws every
  target present or absent 50/50 (`rng_gen.choice(2)`), ignoring the
  frequency its CPU path uses: `output_balancing(0.5)` reproduces that. (As
  released in TMU 0.8.3, `TMAutoEncoder.fit` passes `target_true_p=` to the
  CUDA bank's `produce_autoencoder_example`, which does not take it.)
- **Autoencoder training progress:** `fit_epoch_progress` and
  `fit_examples_progress` on `TMCoalescedAutoEncoder` and its CUDA twin call
  back with the rows (examples) trained so far, for a progress line inside a
  long epoch. They train the same model as `fit_epoch` / `fit_examples`.
- Examples `cuda_coalesced_sparse` and `cuda_sparse_bank`: GPU vs CPU training
  speed on wide sparse inputs (44,428 features, a few dozen set per sample),
  each checked bit-identical to the CPU.

### Changed
- **Coalesced GPU training ~20× faster on wide inputs**, still bit-identical to
  the CPU. The y / neg feedback masks were drawn by one thread of `co_plan`
  (2 × 2 × words × 12 sequential SplitMix64 draws per sample); a new `co_masks`
  kernel draws them a thread per mask word, from the counter-based stream
  (`rng_peek`). `shared_fire` evaluates a clause per block with coalesced
  reads instead of a clause per thread (also for the coalesced autoencoder,
  whose masks are drawn across a block). `co_update` counts a clause's
  included literals only for Type Ia under a literal cap, and moves a word's
  64 TA states as eight u64s when the clause is 8-byte aligned. 44,428
  features, 100 clauses, RTX A500 Laptop: 316 → 6,576 samples/s (one CPU
  thread: ~1,700).
- **Sparse clause bank on the GPU:** `sp_update` decides its per-literal draws
  on the draw's 53-bit integer (`m <= floor(p · 2^53)`, exactly `rng_f64() <=
  p`) instead of in double precision, which most GPUs run at a fraction of
  single-precision speed. The list walk stays one thread per clause, in the
  CPU's order, so the bank remains slower on the GPU than on the CPU for
  small models.
- **Autoencoder inference through TMU's literal → clause map**
  (`TMCoalescedAutoEncoder`, `TMAutoEncoder`), as the classifiers since #42:
  a sample's clause outputs come from its false literals and the clauses that
  include them, and are computed once per sample for every output. The
  coalesced autoencoder evaluated every shared clause once per output before.
  Same results (tests against a full `fire_predict` scan across training,
  `fit_examples`, `start_excluded` and save/load; saved outputs of 1,000
  samples compared before/after). 384 features, 256 shared clauses:
  `reconstruct_scores` 1,190 → 10 µs per sample; vanilla, 8 clauses per
  output: 45 → 26 µs. The map is built on first prediction and dropped by
  every `&mut self` / `mut self` method and the CUDA `to_cpu` copies.

### Added
- **TMU's autoencoder training for `TMCoalescedAutoEncoder`** (opt-in; the
  default `fit_epoch` is bit-identical to before):
  - `fit_examples(&batch, number_of_examples)` trains like TMU's
    `TMAutoEncoder.fit`: each example trains every output, in a fresh random
    order, on an example made for that output — target drawn first (each
    feature's frequency, or `output_balancing(p)`), then a random row with
    that value, or the union of `accumulation(n)` such rows. `fit_epoch`
    trains every output on the same row.
  - `start_excluded()`: start with every literal excluded, as TMU's clause
    bank does (by default about half start included).
  - `shuffle_outputs(true)`: a fresh output order per sample in `fit_epoch` /
    `fit_one` too. CPU only: `to_cuda` refuses it.
  Training settings, not saved (the file format is unchanged).
  Remaining differences from TMU, kept by design: the training vote hides
  the output's own literals (TMU hides them only from feedback), every firing
  clause counts in it (TMU drops clauses the confidence gate skips), and a
  vote of exactly 0 reconstructs as 0 (TMU: 1).

- **`EncodedBatch::select(rows)`** and **`EncodedBatch::words()`** — copy chosen
  samples (any order, repeats allowed) into a new batch. Lets a training loop
  encode its data once and draw every epoch's shuffled or class-balanced batches
  by copying words instead of re-encoding.

- **GPU-resident training data** (`--features cuda`): `EncodedBatch::to_cuda`
  uploads a batch once as a `CudaEncodedBatch`;
  `CudaCoalescedTsetlinMachine::fit_epoch_rows(&data, rows, ys)` trains on chosen
  rows of it and `predict_batch_resident(&data)` scores it, reading each sample
  in place — no per-epoch copy or upload. `fit_epoch_rows` is bit-identical to
  `fit_epoch(&batch.select(rows), ys)` (same shuffle and random draws).

### Changed
- **`Encoder::encode_batch_categorical` encodes rows in parallel** with the
  `parallel` feature (identical output). Encoding 210k samples × 54k features
  drops from ~47 s to the time a single core needs divided by the core count.
- **CUDA GPU support for every model, training and inference** (`--features cuda`).
  `to_cuda(&CudaDevice)` moves any model (`TsetlinMachine`,
  `CoalescedTsetlinMachine`, `ConvolutionalTsetlinMachine` 1-D/2-D,
  `TMSparseClassifier`, `TMCompositeClassifier`, `TMRegressor`, `TMAutoEncoder`,
  `TMCoalescedAutoEncoder`) to the GPU; `fit_epoch` / `fit_one` / `predict_batch` /
  `scores_batch` / `accuracy` (and model-specific metrics) run there, and
  `to_cpu()` brings the learned state back. GPU training is **bit-identical** to the
  CPU's exact training path — same RNG streams, draw order and TA arithmetic —
  including dropout, `max_included_literals`, class weights, focused negative
  sampling, absorbing sparse lists, and Type III feedback. The CUDA driver and
  NVRTC are loaded at runtime, so the crate builds without the CUDA toolkit.
- **`cuda-emu` feature** — `CudaDevice::emulator()` runs the same CUDA kernel source
  (compiled as C++) on the CPU, so the GPU path is tested in CI without a GPU:
  GPU-vs-CPU equivalence tests for every model (`cargo test --features cuda-emu`;
  on a GPU machine, `TMU_TEST_GPU=1 cargo test --features cuda` runs them on the
  real device without the emulator).
- **`examples/cuda.rs`** — CPU vs GPU training time and a bit-identity check.
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
