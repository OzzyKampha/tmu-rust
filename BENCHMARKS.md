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

```
══════════════════════════════════════════════════════════════════════
  COMPARISON SUMMARY
  (config: 2 classes · 1000 features · 10000 clauses/class · IMDb-scale)
══════════════════════════════════════════════════════════════════════
  Runner                               Median ms     Mclause-ups/s
  ────────────────────────────────────────────────────────────────────
  Rust (sequential)                    5899.2 ms               6.8
  Rust (parallel, Rayon)               3985.2 ms              10.0
  Python TMU (C extension)            20884.4 ms               1.9
══════════════════════════════════════════════════════════════════════
  Rust sequential speedup over Python: 3.5x
  Rust parallel  speedup over Python: 5.2x
```

**Notes on these numbers:**

- The Rayon gain is modest (1.5×) on a 4-core VM; expect proportionally larger
  gains on many-core hardware.
- Per-epoch time decreases sharply as clauses absorb — for Rust sequential the
  range is 12 357 ms (epoch 0) → 3 911 ms (epoch 7). The median captures the
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
