//! Training throughput vs. clause count and literal width — and whether the
//! work-aware Rayon gate helps the "few clauses, many literals" case.
//!
//! Finding from this benchmark: the work-aware gate (`items >= 128 || items ×
//! words >= 256`) helps **sparse** training and all **inference**, but **dense
//! training keeps a count-only gate** — its per-clause work is AVX2-fast and
//! Rayon is dispatched per sample, so parallelising a few wide clauses is pure
//! overhead (a measured regression). This example measures `fit_epoch` wall-time
//! (ms/epoch) for dense `TsetlinMachine` and sparse `TMSparseClassifier` in two
//! sweeps:
//!
//!   A. vary clauses/class at a fixed moderate width  (the classic count axis)
//!   B. FIX a low clause count (below the old 128 threshold) and widen features
//!      (the axis the old count-only gate ignored)
//!
//! Run both and compare:
//!   cargo run --release --example parallel_scaling
//!   cargo run --release --features parallel --example parallel_scaling
//!
//! Sweep B is the headline: with the old `clauses >= 128` gate its cells ran
//! single-threaded regardless of width; the work-aware gate parallelises them.

use std::time::Instant;
use tmu_rs::{Encoder, Rng, TMSparseClassifier, TsetlinMachine};

const N_CLASSES: usize = 2;
const SAMPLES: usize = 1000;
const WARMUP: usize = 1;
const TIMED: usize = 3;

fn make(n: usize, n_features: usize, seed: u64) -> (Vec<Vec<u8>>, Vec<usize>) {
    let mut rng = Rng::new(seed);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let f: Vec<u8> = (0..n_features).map(|_| (rng.next_u64() & 1) as u8).collect();
        ys.push((f[0] ^ f[1]) as usize);
        xs.push(f);
    }
    (xs, ys)
}

fn as_slices(xs: &[Vec<u8>]) -> Vec<&[u8]> {
    xs.iter().map(|v| v.as_slice()).collect()
}

/// Median ms/epoch of `fit_epoch` over TIMED epochs (after WARMUP), for a freshly
/// built classifier of the given kind at (features, clauses/class).
fn time_epoch_dense(features: usize, cpc: usize, batch: &tmu_rs::EncodedBatch, ys: &[usize]) -> f64 {
    let mut tm = TsetlinMachine::with_config(N_CLASSES, features, cpc, 20, 3.9, 8, true, 7);
    median_epoch(|| tm.fit_epoch(batch, ys))
}

/// Dense **data-parallel** (approximate) training via the data_parallel flag.
fn time_epoch_dense_dp(features: usize, cpc: usize, batch: &tmu_rs::EncodedBatch, ys: &[usize]) -> f64 {
    let mut tm =
        TsetlinMachine::with_config(N_CLASSES, features, cpc, 20, 3.9, 8, true, 7).data_parallel(true);
    median_epoch(|| tm.fit_epoch(batch, ys))
}

fn time_epoch_sparse(features: usize, cpc: usize, batch: &tmu_rs::EncodedBatch, ys: &[usize]) -> f64 {
    let mut tm = TMSparseClassifier::with_config(N_CLASSES, features, cpc, 20, 3.9, 8, true, 7)
        .max_included_literals(32);
    median_epoch(|| tm.fit_epoch(batch, ys))
}

fn median_epoch(mut epoch: impl FnMut()) -> f64 {
    for _ in 0..WARMUP {
        epoch();
    }
    let mut times: Vec<f64> = (0..TIMED)
        .map(|_| {
            let t = Instant::now();
            epoch();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

fn main() {
    #[cfg(feature = "parallel")]
    let mode = "PARALLEL (rayon)";
    #[cfg(not(feature = "parallel"))]
    let mode = "SCALAR (single-thread)";
    println!("=== Training throughput — ms/epoch === [{mode}]");
    println!("{N_CLASSES} classes, {SAMPLES} samples/epoch, median of {TIMED} timed epochs\n");

    // ── Sweep A: vary clauses at fixed moderate width ───────────────────────
    let feats_a = 512;
    let (xtr, ytr) = make(SAMPLES, feats_a, 1);
    let e = Encoder::for_binary(feats_a);
    let btr = e.encode_batch(&as_slices(&xtr));
    println!("Sweep A — {feats_a} features (words={}), vary clauses/class:", feats_a / 32);
    println!(
        "  {:>7} | {:>12} | {:>12} | {:>10}",
        "clauses", "dense(exact)", "dense(DP)", "sparse"
    );
    println!("  {}", "-".repeat(52));
    for &cpc in &[16usize, 64, 256, 1024, 4096, 8192, 16384] {
        let d = time_epoch_dense(feats_a, cpc, &btr, &ytr);
        let dp = time_epoch_dense_dp(feats_a, cpc, &btr, &ytr);
        let s = time_epoch_sparse(feats_a, cpc, &btr, &ytr);
        let sp = d / dp;
        println!("  {cpc:>7} | {d:>10.1}ms | {dp:>8.1}ms {sp:>3.1}x | {s:>8.1}ms");
    }

    // ── Sweep B: FIX low clause count, widen features (the headline) ────────
    let cpc_b = 32; // below PARALLEL_MIN=128
    println!("\nSweep B — FIXED {cpc_b} clauses/class (< 128), widen features:");
    println!("  {:>8} | {:>6} | {:>10} | {:>10}", "features", "words", "dense", "sparse");
    println!("  {}", "-".repeat(50));
    for &feats in &[128usize, 1_000, 5_000, 20_000] {
        let (xtr, ytr) = make(SAMPLES, feats, 2);
        let e = Encoder::for_binary(feats);
        let btr = e.encode_batch(&as_slices(&xtr));
        let words = feats.div_ceil(32);
        let work = cpc_b * words;
        let d = time_epoch_dense(feats, cpc_b, &btr, &ytr);
        let s = time_epoch_sparse(feats, cpc_b, &btr, &ytr);
        // Dense training is count-only (32 < 128 → scalar). Sparse training is
        // work-aware (parallelises once cps×words ≥ 256).
        let sparse_par = if work >= 256 { "sparse∥" } else { "sparse scalar" };
        println!(
            "  {feats:>8} | {words:>6} | {d:>8.1}ms | {s:>8.1}ms   (cps×words={work}: dense scalar, {sparse_par})"
        );
    }

    println!("\nCompare SCALAR vs PARALLEL runs:");
    println!("- dense(exact): bit-identical clause-parallel; memory-bandwidth bound, so it");
    println!("  only helps very large models (gated at DENSE_TRAIN_PARALLEL_MIN clauses).");
    println!("- dense(DP): fit_epoch + data_parallel(true) — approximate data-parallel,");
    println!("  ~2-4x faster than scalar at any size (see the xN column in Sweep A).");
    println!("- sparse training + all inference use the work-aware gate (Follow-up 2).");
}
