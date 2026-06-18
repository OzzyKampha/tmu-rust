//! Noisy-XOR sanity check — uses the same data setup as TMU's `InterpretabilityDemo`
//! (20 features, 10% label noise on training, noise-free test set).
//!
//! Reports accuracy each epoch; the TM should converge to ~90%+ despite noise.
//!
//! `cargo run --release --example noisy_xor`

use tmu_rs::{Rng, TsetlinMachine};

const N_FEATURES: usize = 20;
const NOISE: f64 = 0.1;

fn make(n: usize, noise: f64, seed: u64) -> (Vec<Vec<u8>>, Vec<usize>) {
    let mut rng = Rng::new(seed);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let f: Vec<u8> = (0..N_FEATURES).map(|_| (rng.next_u64() & 1) as u8).collect();
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
    let (xte, yte) = make(5000, 0.0, 2); // clean test set

    // Mirrors InterpretabilityDemo: clauses=10, T=10, s=3.0, boost=false
    let mut tm = TsetlinMachine::with_config(2, N_FEATURES, 10, 10, 3.0, 8, false, 42);

    let xtr_r: Vec<&[u8]> = xtr.iter().map(|v| v.as_slice()).collect();
    let xte_r: Vec<&[u8]> = xte.iter().map(|v| v.as_slice()).collect();
    let packed_tr = tm.pack_dataset(&xtr_r);
    let packed_te = tm.pack_dataset(&xte_r);

    for epoch in 1..=20 {
        tm.fit_epoch_packed(&packed_tr, xtr.len(), &ytr);
        println!(
            "epoch {epoch:>2}  test accuracy = {:.4}",
            tm.accuracy_packed(&packed_te, xte.len(), &yte)
        );
    }
    println!("\nfinal test accuracy: {:.4}", tm.accuracy_packed(&packed_te, xte.len(), &yte));
}
