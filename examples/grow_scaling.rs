//! Growing at scale: dense vs sparse as the literal space explodes, and whether
//! Rayon (`--features parallel`) helps.
//!
//! Both a dense `TsetlinMachine` and a sparse `TMSparseClassifier` are trained on
//! a small base problem so their clauses carry real learned literals, then grown
//! to progressively larger vocabularies. At each size we measure:
//!
//!   * grow latency          — cost of the grow_features call itself
//!   * model memory (est.)    — bytes of clause state
//!   * inference throughput   — events/sec on a re-encoded batch
//!
//! The point: dense cost scales with the *total* literal count (it sweeps every
//! word of every clause per event); sparse inference scales with the *included*
//! literals per clause (a handful), so it stays cheap as the vocab grows.
//!
//! Run both:
//!   cargo run --release --example grow_scaling
//!   cargo run --release --features parallel --example grow_scaling

use std::time::Instant;
use tmu_rs::{Encoder, Rng, TMSparseClassifier, TsetlinMachine};

const BASE_FEATURES: usize = 16;
const CLAUSES_PER_CLASS: usize = 200; // > PARALLEL_MIN (128) so the Rayon paths engage
const N_CLASSES: usize = 2;
const INFER_BATCH: usize = 20_000;
/// Grow targets (feature counts). Dense is skipped past DENSE_LIMIT to stay in RAM.
const SIZES: &[usize] = &[1_000, 10_000, 100_000, 1_000_000];
const DENSE_LIMIT: usize = 100_000;

fn make_xor(n: usize, noise: f64, seed: u64) -> (Vec<Vec<u8>>, Vec<usize>) {
    let mut rng = Rng::new(seed);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let f: Vec<u8> = (0..BASE_FEATURES).map(|_| (rng.next_u64() & 1) as u8).collect();
        let mut y = (f[0] ^ f[1]) as usize;
        if rng.next_f64() <= noise {
            y = 1 - y;
        }
        xs.push(f);
        ys.push(y);
    }
    (xs, ys)
}

fn as_slices(xs: &[Vec<u8>]) -> Vec<&[u8]> {
    xs.iter().map(|v| v.as_slice()).collect()
}

/// How many events to time at a given width. Raw padded rows cost `n_features`
/// bytes each, so cap the count by a memory budget (a 1M-feature × 20k-event raw
/// batch would be 20 GB). events/sec is normalised, so fewer samples is fine.
fn sample_count(n_features: usize) -> usize {
    let by_mem = 200_000_000 / n_features.max(1);
    INFER_BATCH.min(by_mem).max(256)
}

/// A set of base-problem events, zero-padded to `n_features`, for timing
/// inference at the grown geometry. Returns the encoded batch and its length.
fn infer_batch(n_features: usize, seed: u64) -> (tmu_rs::EncodedBatch, usize) {
    let n = sample_count(n_features);
    let (xs, _) = make_xor(n, 0.0, seed);
    let padded: Vec<Vec<u8>> = xs
        .iter()
        .map(|x| {
            let mut p = x.clone();
            p.resize(n_features, 0);
            p
        })
        .collect();
    let enc = Encoder::for_binary(n_features);
    let batch = enc.encode_batch(&as_slices(&padded));
    (batch, n)
}

/// Dense clause-state bytes: ta + ind (u8/literal) + include + cat (u64/word).
fn dense_bytes(n_features: usize) -> usize {
    let n_clauses = N_CLASSES * CLAUSES_PER_CLASS;
    let n_literals = 2 * n_features;
    let words = n_literals.div_ceil(64);
    n_clauses * n_literals * 2 + n_clauses * words * 8 * 2
}

fn human_bytes(b: usize) -> String {
    const U: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut v = b as f64;
    let mut i = 0;
    while v >= 1024.0 && i < 3 {
        v /= 1024.0;
        i += 1;
    }
    format!("{v:.1} {}", U[i])
}

fn main() {
    #[cfg(feature = "parallel")]
    let mode = "PARALLEL (rayon)";
    #[cfg(not(feature = "parallel"))]
    let mode = "SCALAR (single-thread)";
    println!("=== Growing at scale — dense vs sparse === [{mode}]\n");
    println!(
        "base: {BASE_FEATURES} features, {CLAUSES_PER_CLASS} clauses/class × {N_CLASSES} classes \
         = {} clauses; inference batch = {INFER_BATCH} events\n",
        N_CLASSES * CLAUSES_PER_CLASS
    );

    // ── train both on the base problem so clauses learn real literals ───────
    let (xtr, ytr) = make_xor(4000, 0.1, 1);
    let e = Encoder::for_binary(BASE_FEATURES);
    let btr = e.encode_batch(&as_slices(&xtr));

    let mut dense =
        TsetlinMachine::with_config(N_CLASSES, BASE_FEATURES, CLAUSES_PER_CLASS, 20, 3.9, 8, true, 7);
    let mut sparse =
        TMSparseClassifier::with_config(N_CLASSES, BASE_FEATURES, CLAUSES_PER_CLASS, 20, 3.9, 8, true, 7)
            .max_included_literals(8);
    for _ in 0..20 {
        dense.fit_epoch(&btr, &ytr);
        sparse.fit_epoch(&btr, &ytr);
    }
    println!("trained base models (dense acc {:.3}, sparse acc {:.3})\n",
        dense.accuracy(&btr, &ytr), sparse.accuracy(&btr, &ytr));

    println!(
        "{:>9} | {:>7} {:>11} {:>9} | {:>7} {:>11} {:>9}",
        "features", "d.grow", "d.infer", "d.mem", "s.grow", "s.infer", "s.mem"
    );
    println!("{}", "-".repeat(74));

    for &size in SIZES {
        // ---- dense (only while it fits in RAM) --------------------------
        let (dense_grow_ms, dense_eps, dense_mem) = if size <= DENSE_LIMIT {
            let t = Instant::now();
            dense.grow_features(size);
            let g = t.elapsed().as_secs_f64() * 1e3;
            let (batch, n) = infer_batch(size, 100 + size as u64);
            let t = Instant::now();
            let _ = dense.predict_batch(&batch);
            let eps = n as f64 / t.elapsed().as_secs_f64();
            (Some(g), Some(eps), Some(dense_bytes(size)))
        } else {
            (None, None, None)
        };

        // ---- sparse -----------------------------------------------------
        let t = Instant::now();
        sparse.grow_features(size);
        let sparse_grow_ms = t.elapsed().as_secs_f64() * 1e3;
        let (batch, n) = infer_batch(size, 500 + size as u64);
        let t = Instant::now();
        let _ = sparse.predict_batch(&batch);
        let sparse_eps = n as f64 / t.elapsed().as_secs_f64();

        // Sparse clause-state bytes: tracked literals (included+excluded) cost a
        // 4-byte index + 1-byte state each. Right after a grow nothing is absorbed,
        // so this is ~all literals — sparse only wins memory once training absorbs.
        let tracked = (N_CLASSES * CLAUSES_PER_CLASS * 2 * size) as f64
            * (1.0 - sparse.absorbed_exclude_fraction());
        let sparse_mem = (tracked * 5.0) as usize;

        let d_grow = dense_grow_ms.map(|g| format!("{g:.0}ms")).unwrap_or("—".into());
        let d_inf = dense_eps.map(|e| format!("{}/s", (e as u64))).unwrap_or("too big".into());
        let d_mem = dense_mem.map(human_bytes).unwrap_or_else(|| human_bytes(dense_bytes(size)) + "*");
        println!(
            "{:>9} | {:>7} {:>11} {:>9} | {:>7} {:>11} {:>9}",
            size,
            d_grow,
            d_inf,
            d_mem,
            format!("{sparse_grow_ms:.0}ms"),
            format!("{}/s", sparse_eps as u64),
            human_bytes(sparse_mem),
        );
    }
    println!("\n* dense memory past {DENSE_LIMIT} features is the estimate it *would* need (not allocated).");
    println!("  s.mem is freshly-grown (untrained) sparse state; absorbing actions shrink it as training converges.");
    println!("Compare the two runs (scalar vs --features parallel) to see the Rayon effect.");
}
