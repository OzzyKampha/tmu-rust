//! ConvolutionalTsetlinMachine demo.
//!
//! Dataset: 4 binary features.  The label is the XOR of features 0 and 1 — the
//! pattern lives in the first of the 3 patch positions.
//!
//! With kernel_size=2 and stride=1 the clause bank slides over 3 patch positions
//! ([0,1], [1,2], [2,3]).  Weight tying lets a single set of clause weights apply
//! at every position; the model must discover that patch 0 carries the label while
//! the other two patches add noise.
//!
//! `cargo run --release --example convolutional`

use tmu_rs::{ConvolutionalTsetlinMachine, Rng};

const N_FEATURES: usize = 4;
const KERNEL: usize = 2;
const STRIDE: usize = 1;
const CLAUSES: usize = 100;
const THRESHOLD: i32 = 50;
const S: f64 = 3.5;
const EPOCHS: usize = 60;

fn make(n: usize, seed: u64) -> (Vec<Vec<u8>>, Vec<usize>) {
    let mut rng = Rng::new(seed);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let f: Vec<u8> = (0..N_FEATURES).map(|_| (rng.next_u64() & 1) as u8).collect();
        let y = (f[0] ^ f[1]) as usize;
        xs.push(f);
        ys.push(y);
    }
    (xs, ys)
}

fn main() {
    let (xtr, ytr) = make(5000, 1);
    let (xte, yte) = make(1000, 2);

    let tr: Vec<&[u8]> = xtr.iter().map(|v| v.as_slice()).collect();
    let te: Vec<&[u8]> = xte.iter().map(|v| v.as_slice()).collect();

    let n_patches = (N_FEATURES - KERNEL) / STRIDE + 1;
    println!(
        "ConvolutionalTM: {N_FEATURES} features, kernel={KERNEL}, stride={STRIDE}, \
         {n_patches} patches, {CLAUSES} clauses/class, T={THRESHOLD}, s={S}"
    );
    println!("Pattern: y = x[0] XOR x[1]  (patch 0; patches 1 and 2 carry noise)");
    println!("{:>5}  {:>10}  {:>10}", "epoch", "train acc", "test acc");

    let mut ctm =
        ConvolutionalTsetlinMachine::with_config(2, N_FEATURES, KERNEL, STRIDE, CLAUSES, THRESHOLD, S, 8, true, 42);

    for epoch in 1..=EPOCHS {
        ctm.fit_epoch(&tr, &ytr);
        if epoch % 10 == 0 || epoch == 1 {
            let tr_acc = ctm.accuracy(&tr, &ytr);
            let te_acc = ctm.accuracy(&te, &yte);
            println!("{epoch:>5}  {tr_acc:>10.4}  {te_acc:>10.4}");
        }
    }

    let final_acc = ctm.accuracy(&te, &yte);
    println!("\nFinal test accuracy: {final_acc:.4}");

    // Show the most informative learned clause rules.
    // Indices are patch-relative (0 = first kernel feature, 1 = second kernel feature).
    println!("\nTop clause rules (first 4 positive clauses of class 0, patch-relative indices):");
    let mut shown = 0;
    for j in (0..CLAUSES).step_by(2) {
        let rule = ctm.clause_rule(0, j);
        if rule.is_empty() {
            continue;
        }
        let features: Vec<String> = rule
            .iter()
            .map(|&(f, neg)| format!("{}x{}", if neg { "¬" } else { "" }, f))
            .collect();
        println!("  clause {:>3}  {}", j, features.join(" ∧ "));
        shown += 1;
        if shown >= 4 {
            break;
        }
    }
}
