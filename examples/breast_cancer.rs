//! Breast Cancer demo — mirrors TMU's `BreastCancerDemo`. Loads the Wisconsin
//! Breast Cancer dataset (numeric features), booleanizes it, and trains the TM.
//!
//! Prepare the data first:
//!   python scripts/prepare_breast_cancer.py     # writes data/breast_cancer.csv
//! Then:
//!   cargo run --release --example breast_cancer
//!
//! Optional arg: number of epochs (default 25).

use tmu_rs::{data, Booleanizer, Rng, TsetlinMachine};

/// Load the Breast Cancer CSV, booleanize numeric features, and train a TM with 80/20 split.
fn main() {
    let epochs: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(50);
    let path = "data/breast_cancer.csv";

    let (xs, ys) = data::read_numeric_csv(path).unwrap_or_else(|e| {
        eprintln!("could not read {path}: {e}\nRun: python scripts/prepare_breast_cancer.py");
        std::process::exit(1);
    });
    let n_features = xs[0].len();
    let n_classes = ys.iter().copied().max().unwrap() + 1;
    println!("{} samples, {n_features} numeric features, {n_classes} classes", xs.len());

    let mut idx: Vec<usize> = (0..xs.len()).collect();
    let mut rng = Rng::new(1);
    for i in (1..idx.len()).rev() {
        idx.swap(i, rng.below(i + 1));
    }
    let cut = idx.len() * 4 / 5;
    let (tr_idx, te_idx) = idx.split_at(cut);

    let tr_rows: Vec<&[f64]> = tr_idx.iter().map(|&i| xs[i].as_slice()).collect();
    let booleanizer = Booleanizer::fit(&tr_rows, n_features, 10);
    let n_bin = booleanizer.n_output_features();
    println!("booleanized to {n_bin} binary features\n");

    let booleanize = |i: usize| {
        let mut o = vec![0u8; n_bin];
        booleanizer.transform_row(&xs[i], &mut o);
        o
    };
    let btr: Vec<Vec<u8>> = tr_idx.iter().map(|&i| booleanize(i)).collect();
    let bte: Vec<Vec<u8>> = te_idx.iter().map(|&i| booleanize(i)).collect();
    let ytr: Vec<usize> = tr_idx.iter().map(|&i| ys[i]).collect();
    let yte: Vec<usize> = te_idx.iter().map(|&i| ys[i]).collect();

    let mut tm = TsetlinMachine::with_config(n_classes, n_bin, 300, 100, 5.0, 8, true, 7);

    let btr_r: Vec<&[u8]> = btr.iter().map(|v| v.as_slice()).collect();
    let bte_r: Vec<&[u8]> = bte.iter().map(|v| v.as_slice()).collect();
    let packed_tr = tm.pack_dataset(&btr_r);
    let packed_te = tm.pack_dataset(&bte_r);

    for epoch in 1..=epochs {
        tm.fit_epoch_packed(&packed_tr, btr.len(), &ytr);
        println!(
            "epoch {epoch:>2}  train={:.4}  test={:.4}",
            tm.accuracy_packed(&packed_tr, btr.len(), &ytr),
            tm.accuracy_packed(&packed_te, bte.len(), &yte)
        );
    }
}
