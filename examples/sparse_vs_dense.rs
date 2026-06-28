//! Dense vs sparse Tsetlin Machine — head-to-head on the same data.
//!
//! Trains the dense [`TsetlinMachine`] and the [`TMSparseClassifier`] with
//! *identical* hyperparameters and seed on a wide noisy-XOR task (only features 0
//! and 1 are relevant; the rest are noise) and prints a side-by-side comparison of
//! accuracy, average clause size, an estimated memory footprint, and wall-clock
//! training / inference time.
//!
//! The point: the sparse bank should match dense accuracy while storing far fewer
//! literals, because absorbing actions permanently drop the irrelevant ones.
//!
//! `cargo run --release --example sparse_vs_dense`

use std::time::Instant;

use tmu_rs::{Encoder, Rng, TMSparseClassifier, TsetlinMachine};

const N_FEATURES: usize = 128;
const NOISE: f64 = 0.1;
const CLAUSES_PER_CLASS: usize = 20;
const THRESHOLD: i32 = 15;
const S: f64 = 3.9;
const STATE_BITS: u8 = 8;
const MAX_INCLUDED: usize = 8;
const EPOCHS: usize = 150;
const SEED: u64 = 42;

fn make(n: usize, noise: f64, seed: u64) -> (Vec<Vec<u8>>, Vec<usize>) {
    let mut rng = Rng::new(seed);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let f: Vec<u8> = (0..N_FEATURES)
            .map(|_| (rng.next_u64() & 1) as u8)
            .collect();
        let mut y = (f[0] ^ f[1]) as usize;
        if rng.next_f64() <= noise {
            y = 1 - y;
        }
        xs.push(f);
        ys.push(y);
    }
    (xs, ys)
}

fn main() {
    let (xtr, ytr) = make(5000, NOISE, 1);
    let (xte, yte) = make(5000, 0.0, 2);

    let enc = Encoder::for_binary(N_FEATURES);
    let xtr_r: Vec<&[u8]> = xtr.iter().map(|v| v.as_slice()).collect();
    let xte_r: Vec<&[u8]> = xte.iter().map(|v| v.as_slice()).collect();
    let tr = enc.encode_batch(&xtr_r);
    let te = enc.encode_batch(&xte_r);

    let n_clauses = 2 * CLAUSES_PER_CLASS;
    let n_literals = 2 * N_FEATURES;
    let words = n_literals.div_ceil(64);

    // ---- dense ----------------------------------------------------------
    let mut dense = TsetlinMachine::with_config(
        2,
        N_FEATURES,
        CLAUSES_PER_CLASS,
        THRESHOLD,
        S,
        STATE_BITS,
        true,
        SEED,
    )
    .max_included_literals(MAX_INCLUDED);
    let t0 = Instant::now();
    for _ in 0..EPOCHS {
        dense.fit_epoch(&tr, &ytr);
    }
    let dense_train = t0.elapsed();
    let t0 = Instant::now();
    let dense_acc = dense.accuracy(&te, &yte);
    let dense_infer = t0.elapsed();
    let dense_avg = avg_clause_size(2, CLAUSES_PER_CLASS, |c, j| dense.clause_rule(c, j).len());
    // Dense always stores a full u8 counter per literal + the include bitset (u64 words).
    let dense_bytes = n_clauses * n_literals /* ta: 1 byte/literal */
        + n_clauses * words * 8 /* include bitset */;

    // ---- sparse ---------------------------------------------------------
    let mut sparse = TMSparseClassifier::with_config(
        2,
        N_FEATURES,
        CLAUSES_PER_CLASS,
        THRESHOLD,
        S,
        STATE_BITS,
        true,
        SEED,
    )
    .max_included_literals(MAX_INCLUDED);
    let t0 = Instant::now();
    for _ in 0..EPOCHS {
        sparse.fit_epoch(&tr, &ytr);
    }
    let sparse_train = t0.elapsed();
    let t0 = Instant::now();
    let sparse_acc = sparse.accuracy(&te, &yte);
    let sparse_infer = t0.elapsed();
    let sparse_avg = avg_clause_size(2, CLAUSES_PER_CLASS, |c, j| sparse.clause_rule(c, j).len());
    // Sparse stores only still-tracked literals (4-byte index + 1-byte state).
    let total_literals = (n_clauses * n_literals) as f64;
    let tracked = (total_literals * (1.0 - sparse.absorbed_exclude_fraction())).round() as usize;
    let sparse_bytes = tracked * (4 + 1);

    // ---- report ---------------------------------------------------------
    println!(
        "Noisy XOR — {N_FEATURES} features ({} relevant), {CLAUSES_PER_CLASS} clauses/class, \
         T={THRESHOLD}, s={S}, max_included={MAX_INCLUDED}, {EPOCHS} epochs\n",
        2
    );
    println!("{:<22}{:>14}{:>14}", "metric", "dense", "sparse");
    println!("{}", "-".repeat(50));
    println!(
        "{:<22}{:>14.4}{:>14.4}",
        "test accuracy", dense_acc, sparse_acc
    );
    println!(
        "{:<22}{:>14.2}{:>14.2}",
        "avg literals/clause", dense_avg, sparse_avg
    );
    println!(
        "{:<22}{:>14}{:>14}",
        "est. memory (bytes)", dense_bytes, sparse_bytes
    );
    println!(
        "{:<22}{:>13.1}x{:>14}",
        "memory ratio",
        dense_bytes as f64 / sparse_bytes.max(1) as f64,
        ""
    );
    println!(
        "{:<22}{:>13.0}ms{:>12.0}ms",
        "train time",
        dense_train.as_secs_f64() * 1e3,
        sparse_train.as_secs_f64() * 1e3
    );
    println!(
        "{:<22}{:>13.2}ms{:>12.2}ms",
        "inference (5k)",
        dense_infer.as_secs_f64() * 1e3,
        sparse_infer.as_secs_f64() * 1e3
    );
    println!(
        "\nsparse absorbed-out fraction: {:.3}",
        sparse.absorbed_exclude_fraction()
    );
}

/// Average included-literal count across all clauses, given a per-clause size fn.
fn avg_clause_size(n_classes: usize, cps: usize, size: impl Fn(usize, usize) -> usize) -> f64 {
    let mut total = 0usize;
    for c in 0..n_classes {
        for j in 0..cps {
            total += size(c, j);
        }
    }
    total as f64 / (n_classes * cps) as f64
}
