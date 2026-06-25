//! TMCompositeClassifier demo — ensemble of TsetlinMachine models.
//!
//! Three small `TsetlinMachine` classifiers (different seeds) are combined
//! into a composite whose class scores are summed at inference time.
//!
//! The dataset is a noisy 4-class XOR problem:
//!   y = 2*(x[0]^x[1]) + (x[2]^x[3])  ∈ {0, 1, 2, 3}
//!
//! The composite is compared against a single model with the same total
//! clause budget to demonstrate the ensemble benefit.
//!
//! `cargo run --release --example composite`

use tmu_rs::{Encoder, Rng, TMCompositeClassifier, TsetlinMachine};

const N_FEATURES: usize = 8;
const N_CLASSES: usize = 4;
const CLAUSES_EACH: usize = 20;   // per constituent model
const TOTAL_CLAUSES: usize = 60;  // single model with same budget
const THRESHOLD: i32 = 20;
const S: f64 = 3.9;
const EPOCHS: usize = 30;

fn make(n: usize, noise: f64, seed: u64) -> (Vec<Vec<u8>>, Vec<usize>) {
    let mut rng = Rng::new(seed);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let f: Vec<u8> = (0..N_FEATURES).map(|_| (rng.next_u64() & 1) as u8).collect();
        let mut y = 2 * (f[0] ^ f[1]) as usize + (f[2] ^ f[3]) as usize;
        if rng.next_f64() < noise {
            y = rng.below(N_CLASSES);
        }
        xs.push(f);
        ys.push(y);
    }
    (xs, ys)
}

fn main() {
    let (xtr, ytr) = make(5000, 0.05, 1);
    let (xte, yte) = make(1000, 0.0, 2);

    let enc = Encoder::for_binary(N_FEATURES);
    let btr = enc.encode_batch(&xtr.iter().map(|v| v.as_slice()).collect::<Vec<_>>());
    let bte = enc.encode_batch(&xte.iter().map(|v| v.as_slice()).collect::<Vec<_>>());

    // Composite: 3 × 20 clauses/class
    let mut composite = TMCompositeClassifier::new();
    for seed in [10u64, 20, 30] {
        composite.add(TsetlinMachine::with_config(
            N_CLASSES, N_FEATURES, CLAUSES_EACH, THRESHOLD, S, 8, true, seed,
        ));
    }

    // Single model: 60 clauses/class (same total budget)
    let mut single = TsetlinMachine::with_config(
        N_CLASSES, N_FEATURES, TOTAL_CLAUSES, THRESHOLD, S, 8, true, 42,
    );

    println!(
        "Comparison: composite (3×{CLAUSES_EACH} clauses/class) vs \
         single ({TOTAL_CLAUSES} clauses/class)"
    );
    println!("{:>5}  {:>14}  {:>14}", "epoch", "composite acc", "single acc");

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
