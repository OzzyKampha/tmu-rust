//! Coalesced Tsetlin Machine demo — TMU's `TMCoalescedClassifier`.
//!
//! A 4-class problem (`y = 2*(b0^b1) + (b2^b3)` over 8 bits) solved with a **single
//! shared bank of just 40 clauses** voted on by all four classes through a signed
//! per-class weight matrix.  A vanilla machine would allocate `n_classes *
//! clauses_per_class` dedicated clauses; the coalesced machine shares one pool.
//!
//! Prints per-epoch accuracy, then the learned sign pattern of a few shared clauses
//! across the classes (showing how one clause can vote `+` for some classes and `-`
//! for others).
//!
//! `cargo run --release --example coalesced`

use tmu_rs::{CoalescedTsetlinMachine, Encoder, Rng};

const N_FEATURES: usize = 8;
const N_CLASSES: usize = 4;
const N_CLAUSES: usize = 40;

/// Generate `n` samples of the 4-class XOR-of-pairs problem over `N_FEATURES` bits.
fn make(n: usize, seed: u64) -> (Vec<Vec<u8>>, Vec<usize>) {
    let mut rng = Rng::new(seed);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let f: Vec<u8> = (0..N_FEATURES).map(|_| (rng.next_u64() & 1) as u8).collect();
        let y = ((f[0] ^ f[1]) as usize) * 2 + (f[2] ^ f[3]) as usize;
        xs.push(f);
        ys.push(y);
    }
    (xs, ys)
}

fn main() {
    let (xtr, ytr) = make(4000, 1);
    let (xte, yte) = make(1000, 2);

    let encoder = Encoder::for_binary(N_FEATURES);
    let xtr_r: Vec<&[u8]> = xtr.iter().map(|v| v.as_slice()).collect();
    let xte_r: Vec<&[u8]> = xte.iter().map(|v| v.as_slice()).collect();
    let packed_tr = encoder.encode_batch(&xtr_r);
    let packed_te = encoder.encode_batch(&xte_r);

    let mut tm = CoalescedTsetlinMachine::with_config(
        N_CLASSES,
        encoder.n_features(),
        N_CLAUSES,
        30,  // threshold T
        3.9, // s
        8,   // state bits
        true,
        42,
    )
    .focused_negative_sampling(true);

    println!(
        "Coalesced TM: {N_CLASSES} classes share one bank of {N_CLAUSES} clauses \
         (vanilla would need {N_CLASSES}×clauses_per_class dedicated clauses)."
    );
    for epoch in 1..=40 {
        tm.fit_epoch(&packed_tr, &ytr);
        if epoch % 5 == 0 || epoch == 1 {
            println!(
                "epoch {epoch:>2}  test accuracy = {:.4}",
                tm.accuracy(&packed_te, &yte)
            );
        }
    }
    println!("\nfinal test accuracy: {:.4}", tm.accuracy(&packed_te, &yte));

    println!("\nLearned signed weights for the first 8 shared clauses (rows = clause):");
    print!("{:>8}", "clause");
    for c in 0..N_CLASSES {
        print!("{:>8}", format!("class{c}"));
    }
    println!();
    for j in 0..8.min(tm.n_clauses()) {
        print!("{j:>8}");
        for c in 0..N_CLASSES {
            print!("{:>8}", tm.clause_weight(c, j));
        }
        println!();
    }
}
