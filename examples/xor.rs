//! XOR demo — mirrors TMU's `XORDemo`.
//! Clean (noise-free) 2-bit XOR: the TM should reach 100% with only 4 clauses.
//!
//! TMU defaults: num_clauses=4, T=10, s=10.0, max_included_literals=32, epochs=60
//!
//! `cargo run --release --example xor`

use tmu_rs::{Rng, TsetlinMachine};

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

fn main() {
    let (xtr, ytr) = make(1000, 1);
    let (xte, yte) = make(1000, 2);

    let mut tm = TsetlinMachine::with_config(2, 2, 4, 10, 10.0, 8, true, 42)
        .max_included_literals(32);

    let xtr_r: Vec<&[u8]> = xtr.iter().map(|v| v.as_slice()).collect();
    let xte_r: Vec<&[u8]> = xte.iter().map(|v| v.as_slice()).collect();
    let packed_tr = tm.pack_dataset(&xtr_r);
    let packed_te = tm.pack_dataset(&xte_r);

    for epoch in 1..=60 {
        tm.fit_epoch_packed(&packed_tr, xtr.len(), &ytr);
        let acc = tm.accuracy_packed(&packed_te, xte.len(), &yte);
        println!("Epoch: {epoch:>2}, Accuracy: {:.2}", acc * 100.0);
    }
}
