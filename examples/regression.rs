//! TMRegressor demo — predict a continuous target from binary features.
//!
//! Dataset: 20 binary features; target y = (number of 1s in features 0..4) × 20,
//! giving y ∈ {0, 20, 40, 60, 80, 100} with threshold = 100.
//!
//! Reports MAE each epoch; the TM should converge to a low MAE after ~50 epochs.
//!
//! `cargo run --release --example regression`

use tmu_rs::{Encoder, Rng, TMRegressor};

const N_FEATURES: usize = 20;
const THRESHOLD: i32 = 100;
const N_CLAUSES: usize = 200;
const S: f64 = 3.0;
const N_EPOCHS: usize = 60;

/// Generate `n` samples.  Target = count of 1s in features 0..4, scaled to [0, THRESHOLD].
fn make(n: usize, seed: u64) -> (Vec<Vec<u8>>, Vec<f64>) {
    let mut rng = Rng::new(seed);
    let scale = THRESHOLD as f64 / 5.0; // 5 counting features → max count = 5
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let f: Vec<u8> = (0..N_FEATURES).map(|_| (rng.next_u64() & 1) as u8).collect();
        let count = f[0..5].iter().map(|&b| b as usize).sum::<usize>();
        ys.push(count as f64 * scale);
        xs.push(f);
    }
    (xs, ys)
}

fn main() {
    let (xtr, ytr) = make(5000, 1);
    let (xte, yte) = make(1000, 2);

    let encoder = Encoder::for_binary(N_FEATURES);
    let xtr_r: Vec<&[u8]> = xtr.iter().map(|v| v.as_slice()).collect();
    let xte_r: Vec<&[u8]> = xte.iter().map(|v| v.as_slice()).collect();
    let packed_tr = encoder.encode_batch(&xtr_r);
    let packed_te = encoder.encode_batch(&xte_r);

    let mut tm = TMRegressor::with_config(N_FEATURES, N_CLAUSES, THRESHOLD, S, 8, true, 42);

    println!("Training TMRegressor: {N_FEATURES} features, {N_CLAUSES} clauses, T={THRESHOLD}, s={S}");
    println!("Target: count of 1s in features 0..4, scaled to [0, {THRESHOLD}]");
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

    // Show a few predictions vs ground truth
    println!("\nSample predictions (first 10 test samples):");
    println!("{:>6}  {:>8}  {:>8}", "true y", "pred", "error");
    let xte10: Vec<&[u8]> = xte[..10].iter().map(|v| v.as_slice()).collect();
    let b10 = encoder.encode_batch(&xte10);
    let preds = tm.predict_batch(&b10);
    for (i, (p, t)) in preds.iter().zip(&yte[..10]).enumerate() {
        println!("{:>6.1}  {:>8.1}  {:>+8.1}  {}", t, p, p - t, xte[i][0..5].iter().map(|b| b.to_string()).collect::<Vec<_>>().join(""));
    }
}
