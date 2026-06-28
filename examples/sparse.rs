//! Sparse Tsetlin Machine with absorbing actions on noisy XOR.
//!
//! Trains [`TMSparseClassifier`] on a 12-feature noisy-XOR task (only features 0
//! and 1 are relevant; the other 10 are noise). As training converges, absorbing
//! actions permanently remove the irrelevant literals from each clause's candidate
//! pool — so the average clause shrinks and the absorbed-out fraction climbs.
//!
//! `cargo run --release --example sparse`

use tmu_rs::{Encoder, Rng, TMSparseClassifier};

const N_FEATURES: usize = 12;
const NOISE: f64 = 0.1;

/// Generate `n` XOR(f0, f1) samples over `N_FEATURES` random bits with label noise.
fn make(n: usize, noise: f64, seed: u64) -> (Vec<Vec<u8>>, Vec<usize>) {
    let mut rng = Rng::new(seed);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let f: Vec<u8> = (0..N_FEATURES)
            .map(|_| (rng.next_u64() & 1) as u8)
            .collect();
        let mut y = (f[0] ^ f[1]) as usize;
        if rng.next_f64() <= noise {
            y = 1 - y;
        }
        xs.push(f);
        ys.push(y);
    }
    (xs, ys)
}

fn avg_clause_size(tm: &TMSparseClassifier) -> f64 {
    let cps = tm.clauses_per_class();
    let mut total = 0usize;
    for c in 0..tm.n_classes() {
        for j in 0..cps {
            total += tm.clause_rule(c, j).len();
        }
    }
    total as f64 / (tm.n_classes() * cps) as f64
}

fn main() {
    let (xtr, ytr) = make(5000, NOISE, 1);
    let (xte, yte) = make(5000, 0.0, 2); // clean test set

    let encoder = Encoder::for_binary(N_FEATURES);
    let xtr_r: Vec<&[u8]> = xtr.iter().map(|v| v.as_slice()).collect();
    let xte_r: Vec<&[u8]> = xte.iter().map(|v| v.as_slice()).collect();
    let packed_tr = encoder.encode_batch(&xtr_r);
    let packed_te = encoder.encode_batch(&xte_r);

    let mut tm = TMSparseClassifier::with_config(2, N_FEATURES, 10, 15, 3.9, 8, true, 42)
        .max_included_literals(8);

    println!("epoch  test_acc  avg_clause  absorbed_out");
    for epoch in 1..=20 {
        tm.fit_epoch(&packed_tr, &ytr);
        println!(
            "{epoch:>5}  {:>8.4}  {:>10.2}  {:>11.3}",
            tm.accuracy(&packed_te, &yte),
            avg_clause_size(&tm),
            tm.absorbed_exclude_fraction(),
        );
    }

    println!(
        "\nfinal test accuracy: {:.4}",
        tm.accuracy(&packed_te, &yte)
    );
    println!("avg included literals/clause: {:.2}", avg_clause_size(&tm));
    println!(
        "absorbed-out fraction: {:.3}  (irrelevant literals dropped from the pool)",
        tm.absorbed_exclude_fraction()
    );

    // Show a couple of learned rules — they should pick out features 0 and 1.
    for j in 0..tm.clauses_per_class().min(4) {
        let sign = if tm.clause_is_positive(j) { "+" } else { "-" };
        let rule: Vec<String> = tm
            .clause_rule(0, j)
            .iter()
            .map(|&(f, neg)| {
                if neg {
                    format!("¬x{f}")
                } else {
                    format!("x{f}")
                }
            })
            .collect();
        println!("class0 clause {j:>2} ({sign}): {}", rule.join(" ∧ "));
    }
}
