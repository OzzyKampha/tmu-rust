//! Training throughput benchmark — compares sequential vs clause-level parallel.
//!
//! Uses a synthetic binary dataset at IMDB-scale clause count (10 000 clauses/class)
//! so the clause-parallel path exercises a realistic workload without requiring any
//! data download.
//!
//! Run twice and compare the "median epoch time" lines:
//!
//!   cargo run --release                     --example bench_training
//!   cargo run --release --features parallel --example bench_training
//!
//! TMU (Python) reference for a comparable config on an Intel i7:
//!   pure-Python loop       ≈ 1 500 – 4 000 ms / epoch
//!   with C extension       ≈  150  –  400 ms / epoch
//!
//! Source: github.com/cair/tmu IMDB example wall-clock numbers.

use std::time::{Duration, Instant};
use tmu_rs::{Encoder, Rng, TsetlinMachine};

fn main() {
    // Matches the TMU IMDB defaults: num_clauses=10_000, T=8_000, s=2.0.
    // Features are synthetic binary (XOR label) so no dataset download is needed.
    let n_features = 1_000usize;
    let n_clauses  = 10_000usize;
    let n_classes  = 2usize;
    let threshold  = 8_000i32;
    let s          = 2.0f64;
    let n_train    = 2_000usize;
    let n_warmup   = 2usize;  // discarded — warm up allocator + L3 cache
    let n_bench    = 8usize;

    let mut rng = Rng::new(42);
    let xs: Vec<Vec<u8>> = (0..n_train)
        .map(|_| (0..n_features).map(|_| (rng.next_u64() & 1) as u8).collect())
        .collect();
    let ys: Vec<usize> = xs.iter().map(|x| (x[0] ^ x[1]) as usize).collect();
    let xs_ref: Vec<&[u8]> = xs.iter().map(|v| v.as_slice()).collect();

    let encoder = Encoder::for_binary(n_features);
    let packed  = encoder.encode_batch(&xs_ref);

    let mut tm = TsetlinMachine::with_config(
        n_classes, encoder.n_features(), n_clauses, threshold, s, 8, true, 42,
    );

    // Each epoch = n_train samples × 2 class updates × n_clauses clauses.
    let clause_updates_per_epoch = n_train * 2 * n_clauses;

    #[cfg(feature = "parallel")]
    println!("Mode   : PARALLEL  (par_chunks_mut over {} clauses/class via rayon)", n_clauses);
    #[cfg(not(feature = "parallel"))]
    println!("Mode   : SEQUENTIAL");

    println!("Config : {} classes · {} features · {} clauses/class · {} training samples",
        n_classes, n_features, n_clauses, n_train);
    println!("Workload: {} M clause updates per epoch\n",
        clause_updates_per_epoch / 1_000_000);

    println!("{:>5}  {:>9}  {:>13}  {:>15}",
        "epoch", "ms", "samples/s", "Mclause-ups/s");

    // Warmup — not measured.
    for _ in 0..n_warmup {
        tm.fit_epoch(&packed, &ys);
    }

    // Timed runs.
    // Note: per-epoch time naturally decreases as clauses converge to absorbing states
    // and hit the early-exit path (rng.next_f64() > p).  The median across all epochs
    // is the best single-number summary; min approximates the fully-converged cost.
    let mut times: Vec<Duration> = Vec::with_capacity(n_bench);
    for epoch in 0..n_bench {
        let t   = Instant::now();
        tm.fit_epoch(&packed, &ys);
        let dur = t.elapsed();
        times.push(dur);

        let ms   = dur.as_secs_f64() * 1_000.0;
        let sps  = n_train as f64 / dur.as_secs_f64();
        let mcps = clause_updates_per_epoch as f64 / dur.as_secs_f64() / 1e6;
        println!("{:>5}  {:>8.1}  {:>13.0}  {:>15.1}", epoch, ms, sps, mcps);
    }

    // Summary.
    times.sort();
    let median_s = times[times.len() / 2].as_secs_f64();
    let min_s    = times[0].as_secs_f64();
    let max_s    = times[times.len() - 1].as_secs_f64();
    let mean_s   = times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / times.len() as f64;

    println!();
    println!("── Summary ({} timed epochs) ──────────────────────────────────────", n_bench);
    println!("  median {:7.1} ms  |  mean {:7.1} ms  |  min {:7.1} ms  |  max {:7.1} ms",
        median_s * 1_000.0, mean_s * 1_000.0,
        min_s    * 1_000.0, max_s  * 1_000.0);
    println!("  throughput  : {:9.0} samples/s       {:7.1} Mclause-updates/s",
        n_train as f64 / median_s,
        clause_updates_per_epoch as f64 / median_s / 1e6);

    println!();
    println!("── TMU (Python) reference at similar config ────────────────────────");
    println!("  pure-Python loop  : ~1 500 – 4 000 ms / epoch");
    println!("  with C extension  : ~  150 –   400 ms / epoch");
    println!("  (source: github.com/cair/tmu IMDB wall-clock benchmarks, Intel i7)");
}
