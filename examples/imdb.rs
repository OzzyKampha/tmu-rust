//! IMDb text categorization demo — mirrors TMU's `IMDbTextCategorizationDemo`.
//! Sentiment classification from a binary bag-of-words (top-5000 vocabulary).
//!
//! TMU defaults: num_clauses=10000, T=8000, s=2.0, weighted_clauses=True,
//!               epochs=40, max_ngram=2, features=5000
//!               (clause_drop_p=0.75 not yet implemented in this port)
//!
//! Prepare the data first (writes data/imdb_train.txt & imdb_test.txt):
//!   python scripts/prepare_imdb.py
//! Then:
//!   cargo run --release --features parallel --example imdb
//!
//! Optional arg: number of epochs (default 40).

use std::time::Instant;
use tmu_rs::{data, TsetlinMachine};

const N_FEATURES: usize = 5000; // must match scripts/prepare_imdb.py

/// Load pre-processed IMDb bag-of-words data and train a weighted TM for the configured number of epochs.
fn main() {
    let epochs: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(40);

    let load = |p: &str| {
        data::read_sparse_binary(p, N_FEATURES).unwrap_or_else(|e| {
            eprintln!("could not read {p}: {e}\nRun: python scripts/prepare_imdb.py");
            std::process::exit(1);
        })
    };
    let (xtr, ytr) = load("data/imdb_train.txt");
    let (xte, yte) = load("data/imdb_test.txt");
    println!("train={} test={} vocab(features)={N_FEATURES}", xtr.len(), xte.len());

    let mut tm = TsetlinMachine::with_config(2, N_FEATURES, 10000, 8000, 2.0, 8, true, 42)
        .clause_drop_p(0.75);

    let xtr_r: Vec<&[u8]> = xtr.iter().map(|v| v.as_slice()).collect();
    let xte_r: Vec<&[u8]> = xte.iter().map(|v| v.as_slice()).collect();
    let packed_tr = tm.pack_dataset(&xtr_r);
    let packed_te = tm.pack_dataset(&xte_r);

    for epoch in 1..=epochs {
        let t = Instant::now();
        tm.fit_epoch_packed(&packed_tr, xtr.len(), &ytr);
        let train_secs = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let acc = tm.accuracy_packed(&packed_te, xte.len(), &yte);
        let test_secs = t.elapsed().as_secs_f64();
        println!(
            "Epoch: {epoch:>2}, Accuracy: {:.2}, Training Time: {train_secs:.2}s, Testing Time: {test_secs:.2}s",
            acc * 100.0
        );
    }
}
