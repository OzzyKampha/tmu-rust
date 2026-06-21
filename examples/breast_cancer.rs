//! Breast Cancer demo — mirrors TMU's `BreastCancerDemo`. Loads the Wisconsin
//! Breast Cancer dataset (numeric features), booleanizes it, and trains the TM.
//!
//! Prepare the data first:
//!   python scripts/prepare_breast_cancer.py     # writes data/breast_cancer.csv
//! Then:
//!   cargo run --release --example breast_cancer
//!
//! Optional arg: number of epochs (default 25).

use tmu_rs::{data, Encoder, Rng, TsetlinMachine};

/// Load the Breast Cancer CSV, booleanize numeric features, and train a TM with 80/20 split.
fn main() {
    let epochs: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(50);
    let path = "data/breast_cancer.csv";

    let (xs, ys) = data::read_numeric_csv(path).unwrap_or_else(|e| {
        eprintln!("could not read {path}: {e}\nRun: python scripts/prepare_breast_cancer.py");
        std::process::exit(1);
    });
    let n_classes = ys.iter().copied().max().unwrap() + 1;
    println!("{} samples, {} numeric features, {n_classes} classes", xs.len(), xs[0].len());

    let mut idx: Vec<usize> = (0..xs.len()).collect();
    let mut rng = Rng::new(1);
    for i in (1..idx.len()).rev() {
        idx.swap(i, rng.below(i + 1));
    }
    let cut = idx.len() * 4 / 5;
    let (tr_idx, te_idx) = idx.split_at(cut);

    let tr_rows: Vec<&[f64]> = tr_idx.iter().map(|&i| xs[i].as_slice()).collect();
    let te_rows: Vec<&[f64]> = te_idx.iter().map(|&i| xs[i].as_slice()).collect();
    let ytr: Vec<usize> = tr_idx.iter().map(|&i| ys[i]).collect();
    let yte: Vec<usize> = te_idx.iter().map(|&i| ys[i]).collect();

    let encoder = Encoder::fit_numeric(&tr_rows, 10);
    println!("booleanized to {} binary features\n", encoder.n_features());

    let mut tm = TsetlinMachine::with_config(n_classes, encoder.n_features(), 300, 100, 5.0, 8, true, 7);

    let packed_tr = encoder.encode_batch_numeric(&tr_rows);
    let packed_te = encoder.encode_batch_numeric(&te_rows);

    for epoch in 1..=epochs {
        tm.fit_epoch(&packed_tr, &ytr);
        println!(
            "epoch {epoch:>2}  train={:.4}  test={:.4}",
            tm.accuracy(&packed_tr, &ytr),
            tm.accuracy(&packed_te, &yte)
        );
    }
}
