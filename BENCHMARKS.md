# Benchmarks

Compares training throughput and accuracy between this Rust implementation and the Python [cair/tmu](https://github.com/cair/tmu) reference library.

---

## Setup

### Rust

```sh
cargo build --release
```

For Rayon clause-parallel training:

```sh
cargo build --release --features parallel
```

For maximum performance with native CPU extensions (enables AVX2 at compile time):

```sh
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

Note: AVX2 fast paths in the clause update loops are also activated at **runtime** via
`is_x86_feature_detected!("avx2")`, even without `target-cpu=native`.

### Python (tmu)

```sh
pip install tmu numpy
```

`tmu` requires a C compiler and libffi to build its CFFI extension. On Linux:

```sh
sudo apt install build-essential libffi-dev
```

On macOS, Xcode command-line tools provide both. The Python benchmark uses the
`"CPU"` platform (CFFI C extension), not the pure-Python fallback.

---

## Running the benchmark

### Quick: Rust only (no Python required)

```sh
# Sequential
cargo run --release --example bench_training

# Parallel (Rayon)
cargo run --release --features parallel --example bench_training

# Native + parallel (maximum performance)
RUSTFLAGS="-C target-cpu=native" cargo run --release --features parallel --example bench_training
```

### Quick: Python only

```sh
# Large config (IMDb-scale, speed comparison)
python scripts/compare_tmu.py

# Small config (NoisyXOR-scale, accuracy check — completes in seconds)
python scripts/compare_tmu.py --small

# Both
python scripts/compare_tmu.py --both
```

### Full side-by-side comparison

```sh
# Rust sequential + Python large
bash scripts/compare.sh

# Add Rayon parallel Rust run
bash scripts/compare.sh --parallel

# All variants with native CPU extensions
bash scripts/compare.sh --parallel --native

# Accuracy check (small config, Python only; Rust always runs large)
bash scripts/compare.sh --small
```

---

## Configuration

Two benchmark configs are used:

| Parameter | Small (NoisyXOR) | Large (IMDb-scale) |
|---|---|---|
| n\_features | 20 | 1 000 |
| n\_clauses / class | 10 | 10 000 |
| n\_classes | 2 | 2 |
| T (threshold) | 10 | 8 000 |
| s | 3.0 | 2.0 |
| state\_bits | 8 | 8 |
| n\_train | 2 000 | 2 000 |
| weighted\_clauses | true | true |
| timed epochs | 20 | 8 |
| warmup epochs | 0 | 2 |
| clause updates / epoch | 400 K | 40 M |

**Small** is used for **accuracy parity** — both implementations converge to ~100% on
clean XOR labels within 20 epochs.

**Large** is used for **throughput comparison** — the 40 M clause-updates-per-epoch
workload exercises the hot SIMD loops and the Rayon parallel path at realistic scale.

---

## Sample output (4-core cloud VM, `compare.sh --parallel --native`)

Results after the u8 TA counter optimisation (PR #7/#8 — 32-wide AVX2, 4× smaller array):

```
══════════════════════════════════════════════════════════════════════
  COMPARISON SUMMARY
  (config: 2 classes · 1000 features · 10000 clauses/class · IMDb-scale)
══════════════════════════════════════════════════════════════════════
  Runner                               Median ms     Mclause-ups/s
  ────────────────────────────────────────────────────────────────────
  Rust (sequential)                    4163.7 ms               9.6
  Rust (parallel, Rayon)               2902.1 ms              13.8
  Python TMU (C extension)            28677.2 ms               1.4
══════════════════════════════════════════════════════════════════════
  Rust sequential speedup over Python: 6.9x
  Rust parallel  speedup over Python: 9.9x
```

**Notes on these numbers:**

- The Rayon gain is modest (1.4×) on a 4-core VM; expect proportionally larger
  gains on many-core hardware.
- Per-epoch time decreases sharply as clauses absorb — for Rust sequential the
  range is 6 652 ms (epoch 0) → 3 035 ms (epoch 7). The median captures the
  mid-training cost, not the steady-state cost.
- Python TMU uses a CFFI C extension (`tmu.tmulib`). Its slowness relative to the
  "150–400 ms" reference in bench\_training.rs is expected: that reference was for
  the cair/tmu IMDb config which uses **2 000 total clauses** (1 000/class).
  This benchmark uses **20 000 total** (10 000/class) — 10× more work per epoch.

---

## Methodology

### What "Mclause-updates/s" measures

Each training epoch processes `n_train × n_classes × n_clauses_per_class` clause
update operations (the inner TA feedback loop). This normalised metric lets you
compare configs with different clause counts.

Clause update cost is not identical across implementations:
- Rust applies Type I / II feedback with scalar u8-expanded TA loops, plus an
  AVX2 fast path that processes 8 literals per instruction (runtime-dispatched via
  `is_x86_feature_detected!("avx2")`).
- Python TMU uses a C extension with a similar loop structure but different
  SIMD strategy and memory layout (bit-planes per state bit vs u32 per TA).

Both count each clause-per-sample pass through the feedback function as one update.

### Why per-epoch time decreases over training

As TA counters converge to absorbing states (include: all bits set, exclude: all
zero) they hit an early-exit path (the `rng.next_f64() > p` check before the
feedback loop). Later epochs are therefore faster than early ones. The **median**
across timed epochs is the standard summary metric; **min** approximates the
fully-converged cost.

### Why data is not bit-identical between Rust and Python

Rust uses **SplitMix64** (seeded via `Rng::new(42)` in `rng.rs`).
Python uses **PCG64** via `numpy.random.default_rng(42)`.

These are different algorithms — the training data has identical *statistical*
properties (i.i.d. Bernoulli(0.5) features, balanced XOR labels) but different
bit sequences. Throughput numbers are not affected. Accuracy numbers on the small
config are comparable but expect ±2–3% variation between individual runs.

If exact data reproducibility is needed, write the Rust dataset to CSV and load
it from Python — the scripts intentionally do not do this because the goal is
throughput, not bit-for-bit reproducibility.

### TMClassifier clause count convention

`TMClassifier(number_of_clauses=N)` in Python allocates **N total** clauses,
split evenly across classes. `compare_tmu.py` passes
`number_of_clauses = n_clauses_per_class × n_classes` to match the Rust
`with_config(..., clauses_per_class, ...)` argument, ensuring both run the same
per-class clause count.

### TMAutoEncoder clause count convention

The Rust and Python autoencoder architectures differ:

- **Rust** (`TMAutoEncoder::with_config(n, clauses_per_output, ...)`) gives each output
  its own **dedicated** clause bank of `clauses_per_output` clauses. Total clauses =
  `n_features × clauses_per_output`.
- **Python** (`TMAutoEncoder(number_of_clauses=N, ...)`) uses a single **shared** clause
  bank of `N` total clauses across all outputs. Each output has its own weight bank but
  reads the same `N` clause outputs.

`bench_autoencoder.py` passes `number_of_clauses = n_features × clauses_per_output` so
the total clause count matches the Rust configuration. The update cost per epoch differs
because Python's inner loop iterates over all `N` clauses for each of the `n_features`
outputs, whereas Rust iterates over only `clauses_per_output` clauses per output.

---

## Autoencoder benchmark

Compares `TMAutoEncoder` throughput between this Rust implementation and Python `tmu`.
Reconstruction accuracy is reported for the Rust implementation only — see the note on
Python TMU accuracy below.

### Running the autoencoder benchmark

```sh
# Rust (both configs: accuracy then throughput)
cargo run --release --example bench_autoencoder

# Rust with Rayon clause-parallel
cargo run --release --features parallel --example bench_autoencoder

# Python TMU (large throughput config only)
python scripts/bench_autoencoder.py

# Python TMU (small accuracy config only)
python scripts/bench_autoencoder.py --small

# Python TMU (both)
python scripts/bench_autoencoder.py --both
```

### Autoencoder benchmark configs

| Parameter | Small (accuracy) | Large (throughput) |
|---|---|---|
| n\_features | 20 | 200 |
| clauses\_per\_output | 40 | 50 |
| T (threshold) | 20 | 200 |
| s | 3.9 | 2.0 |
| state\_bits | 8 | 8 |
| n\_train | 2 000 | 2 000 |
| timed epochs | 20 | 8 |
| warmup epochs | 0 | 2 |
| clause updates / epoch (Rust) | 1.6 M | 20 M |
| clause updates / epoch (Python) | 32 M | 4 000 M |

**Small** is used to verify **Rust accuracy** — the Rust implementation converges from
~50% to >98% reconstruction accuracy within 20 epochs on structured binary data
(mirrored-half: bits `n/2+i` = bits `i`). Python TMU does not converge on this config
(see the tmu accuracy note below).

**Large** is used for **throughput comparison** — exercises the clause update hot loop
at scale. Both Rust and Python produce valid speed numbers here.

### Autoencoder clause updates formula

The two implementations have different per-epoch work:

**Rust** — dedicated per-output clause banks:
```
clause_updates_per_epoch = n_train × n_features × clauses_per_output
```

**Python** — shared clause bank iterated per output:
```
clause_updates_per_epoch = n_train × n_features × total_clauses
                         = n_train × n_features × (n_features × clauses_per_output)
```

For the large config this gives Rust 20 M and Python 4 000 M clause updates per epoch —
a 200× difference. Both throughput numbers (Mclause-updates/s) measure clause-update
kernel throughput but over architecturally different inner loops.

### Sample results (4-core cloud VM)

**Rust** (sequential, `cargo run --release --example bench_autoencoder`):

```
Mode   : SEQUENTIAL  [small (accuracy check)]
Config : 20 features · 40 clauses/output · T=20 · s=3.9 · 2000 training samples
Workload: 1 M clause updates per epoch

epoch         ms      samples/s    Mclause-ups/s  recon-acc
    0      44.7          44704             35.8   0.9755
    ...
   19      37.5          53346             42.7   0.9834

── Summary (20 timed epochs) ──────────────────────────────────────
  median    38.1 ms  |  mean    40.1 ms  |  min    37.5 ms  |  max    57.8 ms
  throughput  :     52431 samples/s          41.9 Mclause-updates/s

Mode   : SEQUENTIAL  [large (throughput)]
Config : 200 features · 50 clauses/output · T=200 · s=2 · 2000 training samples
Workload: 20 M clause updates per epoch

  median  1025.9 ms  | throughput: 1950 samples/s  19.5 Mclause-updates/s
```

**Python TMU** (`python scripts/bench_autoencoder.py --both`):

```
Mode   : Python TMU  [small (accuracy check)]
Config : 20 features · 40 clauses/output · T=20 · s=3.9 · 2000 training samples
Workload: 32 M clause updates per epoch

  median  6135.8 ms  | throughput: 326 samples/s   5.2 Mclause-updates/s
  recon-acc: 0.5267 (does not converge — see tmu accuracy note below)

Mode   : Python TMU  [large (throughput)]
Config : 200 features · 50 clauses/output · T=200 · s=2.0 · 2000 training samples
Workload: 4000 M clause updates per epoch

  median 270247 ms  | throughput: 7 samples/s   14.8 Mclause-updates/s
```

---

## New model type benchmarks (v1.0)

Measured on a 4-core cloud VM, single-threaded release build (`cargo build --release`),
`RUSTFLAGS="-C target-cpu=native"` for the Rust column.

| Model | Python TMU | Rust (default) | Speedup |
|---|---|---|---|
| TMRegressor (60 ep, 5000×20, T=100) | 13.7 s | 0.39 s | 35× |
| ConvolutionalTM 1-D (60 ep, 5000×4, 3 patches) | 127 s | 5.2 s | 24× |
| ConvolutionalTM 2-D (60 ep, 5000×8, 3 patches) | 135 s | 5.1 s | 27× |
| TMCompositeClassifier (30 ep, 5000×8, 4-class) | 63 s | 1.0 s | 63× |

**Notes:**

- Python conv times include 3× patch expansion (15 000 samples/epoch); Rust processes patches
  internally — same effective work per epoch.
- `--features parallel` (Rayon) is counterproductive at this scale: with only 100–200 clauses
  and 5 000 samples, Rayon thread scheduling overhead dominates and can slow training by 10–25×.
  Parallel shines at ≥1 000 clauses and large sample counts (MNIST, IMDb scale).
- All four models use shared binary data (`data/cmp_*.bin`) generated by
  `scripts/gen_shared_data.py` so Rust and Python train on identical samples.
- Run with `bash scripts/compare_new.sh --rust-only` (Rust) or
  `python scripts/compare_all.py` (Python side-by-side).

---

## Dense vs Sparse clause bank

`TMSparseClassifier` stores each clause as included / excluded literal **index
lists** rather than a dense per-literal counter array, and **absorbing actions**
permanently remove literals once they reach the exclude floor. This trades
constant-factor overhead (a 4-byte index + 1-byte state per *tracked* literal,
plus scalar list iteration instead of bit-parallel / AVX2 sweeps) for a footprint
that shrinks as irrelevant literals are absorbed away. It therefore wins only when
the feature space is large and most literals get absorbed.

Run it yourself: `cargo run --release --example sparse_vs_dense`.

Representative numbers (noisy XOR, 2 relevant features, 20 clauses/class, T=15,
s=3.9, `max_included=8`, single-threaded release build):

| features | epochs | dense acc | sparse acc | dense mem | sparse mem | mem ratio | absorbed |
|---|---|---|---|---|---|---|---|
| 32  | 40  | 0.995 | 0.994 | 2 880 B  | 5 955 B | 0.5× (dense smaller) | 54% |
| 128 | 150 | 0.988 | 0.978 | 11 520 B | 4 900 B | **2.4× smaller** | 90% |

**Takeaways:**

- **Accuracy is at parity** in both regimes — the sparse port is algorithmically
  correct, not just smaller. (The `sparse_matches_dense_accuracy` test enforces
  this within a 0.05 margin.)
- **Memory crossover** happens around ~78% absorption: below it, dense's 1-byte
  bit-packed counters beat the sparse 5-byte-per-literal index storage; above it,
  sparse pulls ahead (2.4× smaller at 128 features / 90% absorbed).
- **Sparse single-thread is slower** than dense at these sizes — dense uses AVX2 on
  contiguous counters, while the sparse hot path is per-index bit gathers + scalar
  RNG + list mutation. Sparse's advantage is memory and asymptotic per-clause
  evaluation cost in high-dimensional, sparsely-relevant problems, not raw
  single-thread speed at small scale.

### Sparse parallelism (`--features parallel`)

Training parallelises over clauses and inference over samples (Rayon), gated by
`PARALLEL_MIN` like the dense model. Each clause owns disjoint state, so parallel
training is **bit-identical** to scalar (verified by a weight/rule checksum). As
with dense, it only pays off at **large clause counts** — at small counts the
per-clause work is too light to amortise Rayon's per-region dispatch overhead
(training enters a parallel region twice per sample).

Sparse training, 64 features, 5000 samples, single-thread vs Rayon (4-core VM):

| clauses/class | epochs | scalar | parallel | speedup |
|---|---|---|---|---|
| 200  | 30 | 8.8 s  | 15.9 s | **0.55× (slower)** |
| 2000 | 6  | 20.5 s | 10.5 s | **1.9× faster** |

So enable `--features parallel` for sparse only with large models; for small/medium
clause counts the scalar build is faster. **AVX2 is not implemented for the sparse
bank** (the gather/RNG/`swap_remove` hot path doesn't vectorise; upstream `cair/tmu`'s
sparse C is also scalar) — see `PORTING_STATUS.md`.

---

### tmu Python accuracy note

Python `tmu` versions tested (0.7.9, 0.8.3) do not converge on the small accuracy config
due to known implementation issues:

- **tmu 0.8.3**: `produce_autoencoder_example` returns a `target_value` computed by
  Python's RNG independently from the C extension's internal sample selection (which
  uses `rand()`). This creates ~35% label-sample mismatch and prevents learning.
- **tmu 0.7.9**: Labels are consistent (batch C function handles both sample and target),
  but the shared clause bank fails to converge with the benchmark parameters — only the
  first output converges while others stay at ~50%.

Throughput numbers from the Python benchmark are unaffected by these accuracy issues.
