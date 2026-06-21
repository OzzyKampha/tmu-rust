# Porting Status

This document tracks the porting status of [cair/tmu](https://github.com/cair/tmu) to Rust.

---

## Machine types

| TMU type | Rust status | Notes |
|---|---|---|
| `TMClassifier` | ✅ Ported | Weighted multiclass; full training + inference API |
| `TMCoalesced` | ❌ Not ported | Requires different memory layout |
| `TMRegressor` | ❌ Not ported | Requires continuous-output learning rule |
| `TMAutoEncoder` | ❌ Not ported | Unsupervised; different clause update logic |
| `TMCompositeClassifier` | ❌ Not ported | Hybrid architecture |
| Convolutional TM | ❌ Not ported | Requires receptive-field clause structure |

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
| Configurable TA state bits | ✅ | 2–16 bits per automaton counter |
| Absorbing state tracking | ✅ | `absorbed_include_fraction()`, `absorbed_exclude_fraction()` |
| Clause rule extraction | ✅ | `clause_rule()`, `clause_is_positive()` |
| Booleanizer | ✅ | Quantile-based continuous-to-binary encoder |
| `Encoder` type | ✅ | Type-safe input encoding: binary, numeric (booleanizer), categorical |
| Multi-threaded training | ✅ | `--features parallel` (Rayon) |
| Pre-packed dataset API | ✅ | `pack_dataset()` + `fit_epoch_packed()` |
| Batch prediction | ✅ | `predict_batch_packed()` |
| Raw class scores | ✅ | `scores_packed()` |
| GPU / CUDA acceleration | ❌ Not planned | |
| Imbalanced-class weighting | ❌ Not planned | Per-class clause weights |

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
| Convolutional demos | — | ❌ Not ported | Requires `ConvolutionalTM` |
| Regression demos | — | ❌ Not ported | Requires `TMRegressor` |
| Autoencoder demos | — | ❌ Not ported | Requires `TMAutoEncoder` |
| Coalesced demos | — | ❌ Not ported | Requires `TMCoalesced` |

