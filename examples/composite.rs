//! TMCompositeClassifier demo — ensemble of TsetlinMachine models.
//!
//! Three small `TsetlinMachine` classifiers (different seeds) are combined
//! into a composite whose class scores are summed at inference time.
//!
//! The dataset is a noisy 4-class XOR problem:
//!   y = 2*(x[0]^x[1]) + (x[2]^x[3])  ∈ {0, 1, 2, 3}
//! (5% label noise in training, clean test set)
//!
//! Loads shared data from data/cmp_composite_*.bin (run scripts/gen_shared_data.py
//! once to create those files) so Rust and Python train on identical samples.
//!
//! `cargo run --release --example composite`

use tmu_rs::{Encoder, TMCompositeClassifier, TsetlinMachine};

const N_FEATURES: usize = 8;
const N_CLASSES: usize = 4;
const CLAUSES_EACH: usize = 20;
const TOTAL_CLAUSES: usize = 60;
const THRESHOLD: i32 = 20;
const S: f64 = 3.9;
const EPOCHS: usize = 30;

fn load_u8(path: &str, n: usize, d: usize) -> Vec<Vec<u8>> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|_| panic!("Missing {path} — run: python scripts/gen_shared_data.py"));
    assert_eq!(bytes.len(), n * d, "unexpected file size in {path}");
    bytes.chunks_exact(d).map(|r| r.to_vec()).collect()
}

fn load_u32_as_usize(path: &str, n: usize) -> Vec<usize> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|_| panic!("Missing {path} — run: python scripts/gen_shared_data.py"));
    assert_eq!(bytes.len(), n * 4, "unexpected file size in {path}");
    bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
        .collect()
}

fn main() {
    let xtr = load_u8("data/cmp_composite_X_train.bin", 5000, N_FEATURES);
    let ytr = load_u32_as_usize("data/cmp_composite_y_train.bin", 5000);
    let xte = load_u8("data/cmp_composite_X_test.bin", 1000, N_FEATURES);
    let yte = load_u32_as_usize("data/cmp_composite_y_test.bin", 1000);

    let enc = Encoder::for_binary(N_FEATURES);
    let btr = enc.encode_batch(&xtr.iter().map(|v| v.as_slice()).collect::<Vec<_>>());
    let bte = enc.encode_batch(&xte.iter().map(|v| v.as_slice()).collect::<Vec<_>>());

    // Composite: 3 × 20 clauses/class
    let mut composite = TMCompositeClassifier::new();
    for seed in [10u64, 20, 30] {
        composite.add(TsetlinMachine::with_config(
            N_CLASSES,
            N_FEATURES,
            CLAUSES_EACH,
            THRESHOLD,
            S,
            8,
            true,
            seed,
        ));
    }

    // Single model: 60 clauses/class (same total budget)
    let mut single = TsetlinMachine::with_config(
        N_CLASSES,
        N_FEATURES,
        TOTAL_CLAUSES,
        THRESHOLD,
        S,
        8,
        true,
        42,
    );

    println!("(shared data: data/cmp_composite_*.bin — identical to Python side)");
    println!(
        "Comparison: composite (3×{CLAUSES_EACH} clauses/class) vs \
         single ({TOTAL_CLAUSES} clauses/class)"
    );
    println!(
        "{:>5}  {:>14}  {:>14}",
        "epoch", "composite acc", "single acc"
    );

    for epoch in 1..=EPOCHS {
        composite.fit_epoch(&btr, &ytr);
        single.fit_epoch(&btr, &ytr);
        if epoch % 5 == 0 || epoch == 1 {
            let ca = composite.accuracy(&bte, &yte);
            let sa = single.accuracy(&bte, &yte);
            println!("{epoch:>5}  {ca:>14.4}  {sa:>14.4}");
        }
    }

    let ca = composite.accuracy(&bte, &yte);
    let sa = single.accuracy(&bte, &yte);
    println!("\nFinal composite accuracy: {ca:.4}");
    println!("Final single accuracy:    {sa:.4}");
    println!(
        "\nComposite has {} constituent classifiers, {} clauses/class each.",
        composite.len(),
        CLAUSES_EACH
    );
}
