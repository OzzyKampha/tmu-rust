//! Interpretability demo — mirrors TMU's `InterpretabilityDemo`.
//! Trains on noisy XOR (20 features, 10% label noise) then prints the learned
//! clauses for each class and polarity as human-readable propositional rules.
//!
//! TMU defaults: num_clauses=10, T=10, s=3.0, features=20, noise=0.1, epochs=20,
//!               boost_true_positive_feedback=0 (False)
//!
//! `cargo run --release --example interpretability`

use tmu_rs::{Encoder, Rng, TsetlinMachine};

const N_FEATURES: usize = 20;
const NOISE: f64 = 0.1;

/// Generate `n` XOR samples over `N_FEATURES` random bits with label noise probability `noise`.
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

/// Generate `n` noise-free XOR samples over `N_FEATURES` random bits.
fn make_clean(n: usize, seed: u64) -> (Vec<Vec<u8>>, Vec<usize>) {
    make(n, 0.0, seed)
}

/// Format a clause rule as a human-readable conjunction string (e.g. `"x0 ∧ ¬x3"`).
fn render(rule: &[(usize, bool)]) -> String {
    if rule.is_empty() {
        return "(empty)".to_string();
    }
    rule.iter()
        .map(|&(f, neg)| if neg { format!("¬x{f}") } else { format!("x{f}") })
        .collect::<Vec<_>>()
        .join(" ∧ ")
}

/// Train on noisy XOR, then print learned clause rules and literal frequencies for each class.
fn main() {
    let (xtr, ytr) = make(5000, NOISE, 1);
    let (xte, yte) = make_clean(5000, 2);

    let encoder = Encoder::for_binary(N_FEATURES);
    // boost_true_positive_feedback=0 (False) — matches TMU InterpretabilityDemo default
    let mut tm = TsetlinMachine::with_config(2, encoder.n_features(), 10, 10, 3.0, 8, false, 42);

    let xtr_r: Vec<&[u8]> = xtr.iter().map(|v| v.as_slice()).collect();
    let xte_r: Vec<&[u8]> = xte.iter().map(|v| v.as_slice()).collect();
    let packed_tr = encoder.encode_batch(&xtr_r);
    let packed_te = encoder.encode_batch(&xte_r);

    for _epoch in 1..=20 {
        tm.fit_epoch(&packed_tr, &ytr);
        let acc = tm.accuracy(&packed_te, &yte);
        println!("Accuracy: {:.2}", acc * 100.0);
    }

    let cpc = tm.clauses_per_class();

    for class in 0..2usize {
        println!("\nClass {class} Positive Clauses:\n");
        for j in 0..cpc {
            if !tm.clause_is_positive(j) {
                continue;
            }
            let w = tm.clause_weight(class, j);
            let rule = tm.clause_rule(class, j);
            println!("Clause #{j} W:{w}  {}", render(&rule));
        }

        println!("\nClass {class} Negative Clauses:\n");
        for j in 0..cpc {
            if tm.clause_is_positive(j) {
                continue;
            }
            let w = tm.clause_weight(class, j);
            let rule = tm.clause_rule(class, j);
            println!("Clause #{j} W:{w}  {}", render(&rule));
        }
    }

    println!("\nLiteral Frequency (count of clauses including each literal):\n");
    let n_lits = N_FEATURES * 2;
    let mut freq = vec![0usize; n_lits];
    for class in 0..2usize {
        for j in 0..cpc {
            for (lit, _) in tm.clause_rule(class, j) {
                freq[lit] += 1;
            }
        }
    }
    for (i, &f) in freq.iter().enumerate() {
        if f > 0 {
            let name = if i < N_FEATURES {
                format!("x{i}")
            } else {
                format!("¬x{}", i - N_FEATURES)
            };
            println!("  {name}: {f}");
        }
    }
}
