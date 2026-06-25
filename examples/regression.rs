//! TMRegressor demo — predict a continuous target from binary features.
//!
//! Dataset: 20 binary features; target y = (number of 1s in features 0..4) × 20,
//! giving y ∈ {0, 20, 40, 60, 80, 100} with threshold = 100.
//!
//! Loads shared data from data/cmp_regressor_*.bin (run scripts/gen_shared_data.py
//! once to create those files) so that Rust and Python train on identical samples.
//!
//! `cargo run --release --example regression`

use tmu_rs::{Encoder, TMRegressor};

const N_FEATURES: usize = 20;
const THRESHOLD: i32 = 100;
const N_CLAUSES: usize = 200;
const S: f64 = 3.0;
const N_EPOCHS: usize = 60;

fn load_u8(path: &str, n: usize, d: usize) -> Vec<Vec<u8>> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|_| panic!("Missing {path} — run: python scripts/gen_shared_data.py"));
    assert_eq!(bytes.len(), n * d, "unexpected file size in {path}");
    bytes.chunks_exact(d).map(|r| r.to_vec()).collect()
}

fn load_f64(path: &str, n: usize) -> Vec<f64> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|_| panic!("Missing {path} — run: python scripts/gen_shared_data.py"));
    assert_eq!(bytes.len(), n * 8, "unexpected file size in {path}");
    bytes
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn main() {
    let xtr = load_u8("data/cmp_regressor_X_train.bin", 5000, N_FEATURES);
    let ytr = load_f64("data/cmp_regressor_y_train.bin", 5000);
    let xte = load_u8("data/cmp_regressor_X_test.bin",  1000, N_FEATURES);
    let yte = load_f64("data/cmp_regressor_y_test.bin",  1000);

    let encoder = Encoder::for_binary(N_FEATURES);
    let xtr_r: Vec<&[u8]> = xtr.iter().map(|v| v.as_slice()).collect();
    let xte_r: Vec<&[u8]> = xte.iter().map(|v| v.as_slice()).collect();
    let packed_tr = encoder.encode_batch(&xtr_r);
    let packed_te = encoder.encode_batch(&xte_r);

    let mut tm = TMRegressor::with_config(N_FEATURES, N_CLAUSES, THRESHOLD, S, 8, true, 42);

    println!("Training TMRegressor: {N_FEATURES} features, {N_CLAUSES} clauses, T={THRESHOLD}, s={S}");
    println!("Target: count of 1s in features 0..4, scaled to [0, {THRESHOLD}]");
    println!("(shared data: data/cmp_regressor_*.bin — identical to Python side)");
    println!("{:>5}  {:>10}  {:>10}", "epoch", "train MAE", "test MAE");

    for epoch in 1..=N_EPOCHS {
        tm.fit_epoch(&packed_tr, &ytr);
        if epoch % 5 == 0 || epoch == 1 {
            let tr_mae = tm.mae(&packed_tr, &ytr);
            let te_mae = tm.mae(&packed_te, &yte);
            println!("{epoch:>5}  {tr_mae:>10.3}  {te_mae:>10.3}");
        }
    }

    let final_mae = tm.mae(&packed_te, &yte);
    let final_rmse = tm.rmse(&packed_te, &yte);
    println!("\nFinal test MAE:  {final_mae:.3}");
    println!("Final test RMSE: {final_rmse:.3}");

    println!("\nSample predictions (first 10 test samples):");
    println!("{:>6}  {:>8}  {:>8}", "true y", "pred", "error");
    let xte10: Vec<&[u8]> = xte[..10].iter().map(|v| v.as_slice()).collect();
    let b10 = encoder.encode_batch(&xte10);
    let preds = tm.predict_batch(&b10);
    for (i, (p, t)) in preds.iter().zip(&yte[..10]).enumerate() {
        println!(
            "{:>6.1}  {:>8.1}  {:>+8.1}  {}",
            t, p, p - t,
            xte[i][0..5].iter().map(|b| b.to_string()).collect::<Vec<_>>().join("")
        );
    }
}
