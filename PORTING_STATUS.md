# Porting Status

This document tracks the porting status of [cair/tmu](https://github.com/cair/tmu) to Rust.

---

## Machine types

| TMU type | Rust status | Notes |
|---|---|---|
| `TMClassifier` | ✅ Ported | Weighted multiclass; full training + inference API |
| `TMCoalesced` | ✅ Ported | Shared clause bank + signed per-class weight matrix; focused negative sampling |
| `TMRegressor` | ✅ Ported | Continuous-output weighted clauses; feedback probability driven by current prediction vs target |
| `TMAutoEncoder` | ✅ Ported | Unsupervised; dedicated per-output clause banks |
| `TMCoalescedAutoEncoder` | ✅ Ported | Coalesced variant: shared clause bank + signed per-output weights |
| `TMCompositeClassifier` | ✅ Ported | Ensemble of `TsetlinMachine` models; class scores summed at inference |
| Convolutional TM | ✅ Ported | 1-D receptive-field clauses; weight tying across patch positions |
| `ClauseBankSparse` / `TMSparseClassifier` | ✅ Ported | Per-clause included/excluded index lists with **absorbing actions** (literals removed at the exclude floor); vanilla-only, scalar, no Type III / parallel / AVX2 in v1 |

---

## TMClassifier features

| Feature | Status | Notes |
|---|---|---|
| Bit-packed clause bank | ✅ | 64-bit word packing, interleaved literal layout |
| Weighted clauses | ✅ | Integer weights per clause, >= 1 |
| Type Ia / Ib feedback | ✅ | Include/exclude update with absorbing state guard |
| Type II feedback | ✅ | Weight decrement on false positives |
| Boost true positives | ✅ | `boost_true_positive_feedback` option |
| Literal dropout | ✅ | `literal_drop_p` per sample |
| Clause dropout | ✅ | `clause_drop_p` per epoch |
| Max included literals | ✅ | Type Ia guard on dense clauses |
| Configurable TA state bits | ✅ | 2–8 bits per automaton counter (u8 storage) |
| Absorbing state tracking | ✅ | `absorbed_include_fraction()`, `absorbed_exclude_fraction()` |
| Clause rule extraction | ✅ | `clause_rule()`, `clause_is_positive()` |
| Booleanizer | ✅ | Quantile-based continuous-to-binary encoder |
| `Encoder` type | ✅ | Type-safe input encoding: binary, numeric (booleanizer), categorical |
| Multi-threaded training | ✅ | `--features parallel` (Rayon) |
| AVX2 SIMD acceleration | ✅ | u8 TA counters, 32-wide AVX2 (rebuild\_include, type\_i, type\_ii); runtime dispatch; scalar fallback retained |
| Pre-packed dataset API | ✅ | `pack_dataset()` + `fit_epoch_packed()` |
| Batch prediction | ✅ | `predict_batch_packed()` |
| Raw class scores | ✅ | `scores_packed()` |
| GPU / CUDA acceleration | ❌ Not planned | |
| Imbalanced-class weighting | ✅ | Per-class feedback scaling via `class_weights()` builder method |

---

## TMRegressor features

| Feature | Status | Notes |
|---|---|---|
| Bit-packed clause bank | ✅ | Same 64-bit word packing as classifier; even clauses positive, odd negative |
| Weighted clauses | ✅ | Integer weights per clause, >= 1; max weight = threshold |
| Continuous-output prediction | ✅ | Vote sum clamped to `[0, threshold]`, returned as `f64` |
| Type Ia / Ib feedback | ✅ | Feedback probability `(T − v) / (2T)` when pushing output up |
| Type II feedback | ✅ | Feedback probability `v / (2T)` when pushing output down |
| Boost true positives | ✅ | `boost_true_positive` option |
| Literal dropout | ✅ | `literal_drop_p` builder |
| Clause dropout | ✅ | `clause_drop_p` builder |
| Max included literals | ✅ | `max_included_literals` Type Ia guard |
| Configurable TA state bits | ✅ | 2–8 bits per counter |
| Clause rule extraction | ✅ | `clause_rule()`, `clause_is_positive()` |
| Batch prediction | ✅ | `predict_batch()` |
| MAE / RMSE metrics | ✅ | `mae()`, `rmse()` over encoded batches |
| Multi-threaded training | ✅ | `--features parallel` (Rayon), clause-parallel feedback |
| Save / load | ✅ | `serde` feature; file tag `TAG_REGRESSOR = 6` |
| GPU / CUDA acceleration | ❌ Not planned | |

---

## ConvolutionalTM features

| Feature | Status | Notes |
|---|---|---|
| 1-D receptive-field clauses | ✅ | Kernel slides over consecutive feature positions; `kernel_size` features per patch |
| Patch extraction | ✅ | `pack_patch()` extracts and bit-packs any contiguous window of the input |
| Multi-patch inference | ✅ | Clause votes summed over all `n_patches` positions |
| Weight tying (training) | ✅ | Each clause update uses one random patch per sample; same weights applied everywhere |
| Weighted clauses | ✅ | Integer weights per clause, >= 1; max weight = threshold |
| Type Ia / Ib feedback | ✅ | Reuses `type_i_update_bytes` from clause bank |
| Type II feedback | ✅ | Reuses `type_ii_update_bytes` from clause bank |
| Boost true positives | ✅ | `boost_true_positive` option |
| Literal dropout | ✅ | `literal_drop_p` builder |
| Clause dropout | ✅ | `clause_drop_p` builder |
| Max included literals | ✅ | `max_included_literals` Type Ia guard |
| Configurable TA state bits | ✅ | 2–8 bits per counter |
| Clause rule extraction | ✅ | `clause_rule(class, clause)` returns patch-relative feature indices |
| Batch prediction | ✅ | `predict_batch()`, `accuracy()` |
| Multi-threaded training | ✅ | `--features parallel` (Rayon), clause-parallel feedback |
| Save / load | ✅ | `serde` feature; file tag `TAG_CONVOLUTIONAL = 7` |
| 2-D (image) convolution | ❌ Not ported | TMU also supports 2-D kernels; pre-flatten rows as a workaround |
| GPU / CUDA acceleration | ❌ Not planned | |

---

## TMCompositeClassifier features

| Feature | Status | Notes |
|---|---|---|
| Constituent model ensemble | ✅ | Owns `Vec<TsetlinMachine>`; add models with `.add()` |
| Score aggregation | ✅ | Class scores summed across all constituents; argmax → predicted class |
| Independent training | ✅ | `fit_epoch()` trains each constituent in turn on the same batch |
| Constituent validation | ✅ | Panics if a newly added model has a different `n_classes()` |
| `len()` / `is_empty()` | ✅ | Query constituent count |
| Batch prediction | ✅ | `predict_batch()`, `accuracy()` |
| Save / load | ✅ | `serde` feature; file tag `TAG_COMPOSITE = 8` |
| Mixed constituent types | ❌ Not ported | TMU allows heterogeneous ensembles; Rust variant holds only `TsetlinMachine` for now |

---

## TMCoalesced features

| Feature | Status | Notes |
|---|---|---|
| Single shared clause bank | ✅ | `n_clauses` clauses shared across all classes (vs per-class pools) |
| Signed per-class weight matrix | ✅ | `weights[class][clause]`, initialised to ±1, may go negative; polarity = sign |
| Type Ia / Ib / II feedback | ✅ | Reuses the dense bit primitives; feedback type chosen by weight sign |
| Boost true positives | ✅ | `boost_true_positive_feedback` option |
| Clause / literal dropout | ✅ | `clause_drop_p`, `literal_drop_p` builders |
| Max included literals | ✅ | `max_included_literals` Type Ia guard |
| Focused negative sampling | ✅ | `focused_negative_sampling()` builder (proportional to per-class update probability) |
| Multi-threaded training | ✅ | `--features parallel` (Rayon), clause-parallel feedback |
| Configurable TA state bits | ✅ | 2–8 bits per counter |
| Clause rule extraction | ✅ | `clause_rule()`, `clause_weight()`, `clause_is_positive()` |

---

## TMSparseClassifier features

Sparse clause bank with absorbing actions: each clause stores included / excluded
literal **index lists** instead of a dense per-literal counter array, and literals
that reach the absorbing exclude floor are permanently removed from the pool.

| Feature | Status | Notes |
|---|---|---|
| Per-clause index lists | ✅ | `included` / `excluded` / `unallocated` indices + parallel state arrays |
| Absorbing exclude (removal) | ✅ | Excluded literal at state 0 is swap-removed into `unallocated`, never revisited |
| Absorbing include (lock) | ✅ | Included literal at `max_state` is immune to decrement |
| Weighted clauses | ✅ | Integer weights per clause, >= 1; max weight = threshold |
| Type Ia / Ib feedback | ✅ | Promote excluded→included at `half`, demote included→excluded below `half` |
| Type II feedback | ✅ | Excluded-only increments on fired negative-class clauses |
| Boost true positives | ✅ | `boost_true_positive` option |
| Clause / literal dropout | ✅ | `clause_drop_p`, `literal_drop_p` builders |
| Max included literals | ✅ | `max_included_literals` Type Ia growth guard |
| Configurable TA state bits | ✅ | 2–8 bits per counter (same `half` / `max_state` as dense) |
| Clause rule extraction | ✅ | `clause_rule()`, `clause_is_positive()` |
| Absorbing introspection | ✅ | `absorbed_include_fraction()`, `absorbed_exclude_fraction()` (latter = removed fraction) |
| Save / load | ✅ | `serde` feature; file tag `TAG_SPARSE = 9` |
| Multi-threaded training | ✅ | `--features parallel` (Rayon); clause-parallel feedback over disjoint per-clause state, bit-identical to scalar. Like the dense model, pays off only at large clause counts (≈1000+) |
| Multi-threaded inference | ✅ | `--features parallel`; `predict_batch` / `accuracy` parallelise over samples |
| Type III feedback | ❌ Not in v1 | Indicator array conflicts with literal removal |
| AVX2 SIMD | ❌ Not applicable | The excluded-list scan is per-index bit gathers + scalar RNG + `swap_remove`, none of which vectorise; upstream `cair/tmu`'s sparse C bank is also scalar |

---

## Examples (TMU demo ports)

| TMU demo | Rust example | Status | Notes |
|---|---|---|---|
| `XORDemo` | `xor` | ✅ Validated | 100% accuracy |
| `NoisyXORDemo` | `noisy_xor` | ✅ Validated | Noisy labels, converges cleanly |
| `InterpretabilityDemo` | `interpretability` | ✅ Validated | Prints extracted clause rules |
| *(extra)* `ndr_flows` | `ndr_flows` | ✅ Complete | Synthetic network-flow detection; not part of TMU |
| *(extra)* `bench_training` | `bench_training` | ✅ Complete | Throughput benchmark: sequential vs parallel, IMDB-scale, synthetic data |
| *(extra)* `absorb_timing` | `absorb_timing` | ✅ Complete | TA absorbing-state fractions at varying `state_bits` |
| `BreastCancerDemo` | `breast_cancer` | ✅ Validated | ~99–100% test accuracy |
| `MNISTDemo` / `MNISTDemoWeightedClauses` | `mnist` | ✅ Validated | ~93% (2000 clauses, T=50, s=10.0) |
| `IMDbTextCategorizationDemo` | `imdb` | ✅ Validated | 2000 clauses, T=80, s=10.0 |
| Convolutional demo | `convolutional` | ✅ Ported | 4 features, kernel=2, stride=1 (3 patches); learns XOR of features 0,1 despite 2 noisy patches; ~77% test accuracy |
| Composite demo | `composite` | ✅ Ported | 3×20-clause ensemble vs 60-clause single model on 4-class XOR |
| Sparse demo | `sparse` | ✅ Validated | 12-feature noisy XOR; 100% test accuracy, absorbing fraction climbs as irrelevant literals are dropped |
| *(extra)* Dense vs sparse | `sparse_vs_dense` | ✅ Complete | Head-to-head accuracy / memory / throughput; at high feature counts sparse reaches ~2.4× smaller memory at near-parity accuracy |
| Regression demo | `regression` | ✅ Ported | Continuous target (count function scaled to `[0, T]`); MAE + RMSE metrics |
| Autoencoder demos | `autoencoder`, `coalesced_autoencoder` | ✅ Ported | `TMAutoEncoder` (vanilla) + `TMCoalescedAutoEncoder` (shared-bank) |
| Coalesced demo | `coalesced` | ✅ Validated | 4-class shared-bank demo; 100% accuracy |
| *(extra)* Save/load round-trip | `save_load` | ✅ Complete | Train → save → load → predict/resume; serde feature |

