use std::time::Instant;
use tmu_rs::{Encoder, Rng, TsetlinMachine};

/// Train a TM for `n_epochs` with the given `state_bits` and print per-epoch accuracy and absorbing fractions.
fn run(label: &str, state_bits: u8, n_epochs: usize) {
    let n_features = 30usize;
    let n_train    = 2000usize;
    let n_clauses  = 200usize;
    let threshold  = 20i32;
    let s          = 3.0f64;

    let mut rng = Rng::new(42);
    let xs: Vec<Vec<u8>> = (0..n_train)
        .map(|_| (0..n_features).map(|_| (rng.next_u64() & 1) as u8).collect())
        .collect();
    let ys: Vec<usize> = xs.iter().map(|x| (x[0] ^ x[1]) as usize).collect();
    let xs_ref: Vec<&[u8]> = xs.iter().map(|v| v.as_slice()).collect();

    let encoder = Encoder::for_binary(n_features);
    let packed = encoder.encode_batch(&xs_ref);

    let mut tm = TsetlinMachine::with_config(
        2, encoder.n_features(), n_clauses, threshold, s, state_bits, true, 1,
    );

    println!("\n── {} (state_bits={}) ──", label, state_bits);
    println!("{:>5}  {:>7}  {:>6}  {:>8}  {:>8}",
        "epoch", "time", "acc%", "abs_inc%", "abs_exc%");

    for epoch in 0..n_epochs {
        let t = Instant::now();
        tm.fit_epoch(&packed, &ys);
        let us = t.elapsed().as_micros();

        let acc = 100.0 * tm.accuracy(&packed, &ys);
        let abs_inc = tm.absorbed_include_fraction() * 100.0;
        let abs_exc = tm.absorbed_exclude_fraction() * 100.0;

        println!("{:>5}  {:>5}µs  {:>5.1}%  {:>7.1}%  {:>7.1}%",
            epoch, us, acc, abs_inc, abs_exc);
    }
}

/// Compare absorbing convergence speed between low and high `state_bits` configurations.
fn main() {
    // state_bits=2: max=3, min=0 — absorbs fast (needs only 3 consecutive incs/decs)
    run("fast absorb", 2, 200);

    // state_bits=8: max=255 — matches default config, needs hundreds of epochs to absorb
    run("slow absorb", 8, 15);
}
