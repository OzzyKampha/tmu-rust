//! Fuzzy-Pattern Tsetlin Machine on noisy XOR (12 features, 10% label noise
//! on training, noise-free test set), next to the vanilla machine with the
//! same clause budget.
//!
//! `cargo run --release --example fuzzy_pattern`

use tmu_rs::{Encoder, Rng, TMFuzzyPatternClassifier, TsetlinMachine};

const N_FEATURES: usize = 12;

fn make(n: usize, noise: f64, seed: u64) -> (Vec<Vec<u8>>, Vec<usize>) {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let f: Vec<u8> = (0..N_FEATURES)
                .map(|_| (rng.next_u64() & 1) as u8)
                .collect();
            let mut y = (f[0] ^ f[1]) as usize;
            if rng.next_f64() < noise {
                y = 1 - y;
            }
            (f, y)
        })
        .unzip()
}

fn main() {
    let (xtr, ytr) = make(3000, 0.1, 1);
    let (xte, yte) = make(2000, 0.0, 2);
    let enc = Encoder::for_binary(N_FEATURES);
    let rows =
        |v: &[Vec<u8>]| enc.encode_batch(&v.iter().map(|x| x.as_slice()).collect::<Vec<_>>());
    let (btr, bte) = (rows(&xtr), rows(&xte));

    // 20 clauses per class, T 10, S 2 (s = 12 / 2 = 6 random decrements), L 12, LF 2.
    let mut fptm = TMFuzzyPatternClassifier::new(2, N_FEATURES, 20, 10, 2.0)
        .max_literals(12)
        .literal_failures(2);
    let mut vanilla = TsetlinMachine::with_config(2, N_FEATURES, 20, 10, 3.0, 8, false, 42);
    println!("epoch  fuzzy-pattern  vanilla");
    for epoch in 1..=30 {
        fptm.fit_epoch(&btr, &ytr);
        vanilla.fit_epoch(&btr, &ytr);
        if epoch % 5 == 0 {
            println!(
                "{epoch:>5}  {:>13.4}  {:>7.4}",
                fptm.accuracy(&bte, &yte),
                vanilla.accuracy(&bte, &yte)
            );
        }
    }
    println!("\nfuzzy-pattern clauses of class 1 (literal l < 12: x_l, else NOT x_(l-12)):");
    for j in 0..fptm.clauses_per_class() {
        let sign = if j.is_multiple_of(2) { '+' } else { '-' };
        println!("  {sign} {:?}", fptm.included_literals(1, j));
    }
}
