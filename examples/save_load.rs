//! Save / load demo — train a TM on XOR, persist it to disk, reload, and verify
//! that the reloaded model predicts identically and can keep training.
//!
//! `cargo run --release --features serde --example save_load`

use tmu_rs::{Encoder, Rng, SaveLoad, TsetlinMachine};

/// Generate `n` noise-free 2-bit XOR samples.
fn make(n: usize, seed: u64) -> (Vec<Vec<u8>>, Vec<usize>) {
    let mut rng = Rng::new(seed);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let x0 = (rng.next_u64() & 1) as u8;
        let x1 = (rng.next_u64() & 1) as u8;
        xs.push(vec![x0, x1]);
        ys.push((x0 ^ x1) as usize);
    }
    (xs, ys)
}

fn main() -> std::io::Result<()> {
    let (xtr, ytr) = make(1000, 1);
    let (xte, yte) = make(1000, 2);

    let encoder = Encoder::for_binary(2);
    let xtr_r: Vec<&[u8]> = xtr.iter().map(|v| v.as_slice()).collect();
    let xte_r: Vec<&[u8]> = xte.iter().map(|v| v.as_slice()).collect();
    let packed_tr = encoder.encode_batch(&xtr_r);
    let packed_te = encoder.encode_batch(&xte_r);

    // Train a small model.
    let mut tm = TsetlinMachine::with_config(2, encoder.n_features(), 4, 10, 10.0, 8, true, 42)
        .max_included_literals(32);
    for _ in 0..30 {
        tm.fit_epoch(&packed_tr, &ytr);
    }
    let acc_before = tm.accuracy(&packed_te, &yte);
    println!("Trained accuracy:        {:.2}%", acc_before * 100.0);

    // Persist the model and its encoder side-by-side.
    let dir = std::env::temp_dir();
    let model_path = dir.join("xor_model.tmrs");
    let encoder_path = dir.join("xor_encoder.tmrs");
    tm.save(&model_path)?;
    encoder.save(&encoder_path)?;
    println!("Saved model to:          {}", model_path.display());
    println!("Saved encoder to:        {}", encoder_path.display());

    // Reload from disk — no retraining, no reinitialisation.
    let loaded_encoder = Encoder::load(&encoder_path)?;
    let mut loaded = TsetlinMachine::load(&model_path)?;
    let bte = loaded_encoder.encode_batch(&xte_r);
    let acc_after = loaded.accuracy(&bte, &yte);
    println!("Reloaded accuracy:       {:.2}%", acc_after * 100.0);
    assert_eq!(
        acc_before, acc_after,
        "reloaded model must predict identically"
    );

    // The reloaded model can continue training where the original left off.
    let btr = loaded_encoder.encode_batch(&xtr_r);
    for _ in 0..30 {
        loaded.fit_epoch(&btr, &ytr);
    }
    println!(
        "After resumed training:  {:.2}%",
        loaded.accuracy(&bte, &yte) * 100.0
    );

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&encoder_path);
    println!("\nSave/load round-trip succeeded.");
    Ok(())
}
