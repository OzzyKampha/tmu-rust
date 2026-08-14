//! One training epoch at a fixed, representative config — a stable target for
//! **memory-traffic** profiling rather than timing.
//!
//! Wall-clock on shared/virtualised hardware is not reproducible enough to resolve
//! the effects this crate cares about (the same cell has been observed spanning
//! ±20% between runs, and absolute timings drifting >2× over an afternoon). Counting
//! memory references instead is deterministic:
//!
//! ```sh
//! cargo build --release --example traffic_profile
//! valgrind --tool=cachegrind --cache-sim=yes \
//!     target/release/examples/traffic_profile
//! ```
//!
//! Read `D refs` (total data references) and especially **`LLd misses`** — last-level
//! data misses are the direct proxy for main-memory traffic, which is what bounds
//! dense Tsetlin Machine training. Compare the counts before and after a change; they
//! move deterministically even when wall-clock cannot resolve the difference.
//!
//! The config matches Sweep A/C of `examples/parallel_scaling.rs` (1024 clauses/class,
//! 512 features, 1000 samples ≈ 2.1 MiB of TA state) so numbers are comparable across
//! the two harnesses. Training is forced onto the **exact** path: this measures the
//! per-sample kernels, not the threading strategy, and keeps the run single-threaded
//! so cachegrind's counts are not interleaved across cores.
//!
//! Pass an epoch count as argv[1] (default 1) to amortise setup when comparing.

use tmu_rs::{Encoder, Rng, TsetlinMachine};

const N_CLASSES: usize = 2;
const N_FEATURES: usize = 512;
const CLAUSES_PER_CLASS: usize = 1024;
const SAMPLES: usize = 1000;

fn main() {
    let epochs: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(1);

    let mut rng = Rng::new(1);
    let xs: Vec<Vec<u8>> = (0..SAMPLES)
        .map(|_| {
            (0..N_FEATURES)
                .map(|_| (rng.next_u64() & 1) as u8)
                .collect()
        })
        .collect();
    let ys: Vec<usize> = xs.iter().map(|f| (f[0] ^ f[1]) as usize).collect();
    let rows: Vec<&[u8]> = xs.iter().map(|v| v.as_slice()).collect();
    let batch = Encoder::for_binary(N_FEATURES).encode_batch(&rows);

    let mut tm = TsetlinMachine::with_config(
        N_CLASSES,
        N_FEATURES,
        CLAUSES_PER_CLASS,
        20,
        3.9,
        8,
        true,
        7,
    );

    for _ in 0..epochs {
        // Exact path: this harness profiles the per-sample kernels, not threading.
        tm.fit_epoch_exact(&batch, &ys);
    }

    // Consume the model so nothing above can be optimised away.
    println!(
        "epochs={epochs} clauses={} features={N_FEATURES} accuracy={:.4}",
        N_CLASSES * CLAUSES_PER_CLASS,
        tm.accuracy(&batch, &ys)
    );
}
