//! Type III feedback demo.
//!
//! Trains a 2-class XOR TM with and without Type III feedback and prints
//! the learned clause rules side by side, showing how Type III produces
//! smaller, more focused conjunctions.
//!
//! `cargo run --release --example type_iii_demo`

use tmu_rs::{Encoder, Rng, TsetlinMachine};

const N_FEATURES: usize = 20;
const N_TRAIN: usize = 5000;
const N_TEST: usize = 1000;
const EPOCHS: usize = 60;
const CLAUSES: usize = 10;
const THRESHOLD: i32 = 10;
const S: f64 = 3.0;
const MAX_STATE: u8 = 8;

fn make_xor(n: usize, seed: u64) -> (Vec<Vec<u8>>, Vec<usize>) {
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

fn as_slices(xs: &[Vec<u8>]) -> Vec<&[u8]> {
    xs.iter().map(|v| v.as_slice()).collect()
}

fn render(rule: &[(usize, bool)]) -> String {
    if rule.is_empty() {
        return "(empty — clause always fires)".to_string();
    }
    rule.iter()
        .map(|&(f, neg)| if neg { format!("¬x{f}") } else { format!("x{f}") })
        .collect::<Vec<_>>()
        .join(" ∧ ")
}

fn print_rules(tm: &TsetlinMachine) {
    let cpc = tm.clauses_per_class();
    let mut total_literals = 0usize;
    let mut n_clauses = 0usize;
    for class in 0..2 {
        println!("  Class {} (XOR = {}):", class, class);
        for c in 0..cpc {
            let rule = tm.clause_rule(class, c);
            let polarity = if tm.clause_is_positive(c) { "+" } else { "−" };
            let weight = tm.clause_weight(class, c);
            let len = rule.len();
            total_literals += len;
            n_clauses += 1;
            println!(
                "    [{polarity}] w={weight:3}  ({len:2} literals)  {}",
                render(&rule)
            );
        }
    }
    let avg = total_literals as f64 / n_clauses as f64;
    println!("  Avg literals/clause: {avg:.1}  (total: {total_literals})");
}

fn main() {
    let (xtr, ytr) = make_xor(N_TRAIN, 1);
    let (xte, yte) = make_xor(N_TEST, 2);

    let enc = Encoder::for_binary(N_FEATURES);
    let btr = enc.encode_batch(&as_slices(&xtr));
    let bte = enc.encode_batch(&as_slices(&xte));

    // --- Baseline: no Type III ---
    let mut base = TsetlinMachine::with_config(
        2, N_FEATURES, CLAUSES, THRESHOLD, S, MAX_STATE, true, 42,
    );
    for _ in 0..EPOCHS {
        base.fit_epoch(&btr, &ytr);
    }
    let base_acc = base.accuracy(&bte, &yte);

    // --- Type III: d = 200 ---
    let mut t3 = TsetlinMachine::with_config(
        2, N_FEATURES, CLAUSES, THRESHOLD, S, MAX_STATE, true, 42,
    )
    .type_iii_feedback(200.0);
    for _ in 0..EPOCHS {
        t3.fit_epoch(&btr, &ytr);
    }
    let t3_acc = t3.accuracy(&bte, &yte);

    println!("=== Without Type III  (acc = {:.1}%) ===", base_acc * 100.0);
    print_rules(&base);

    println!();
    println!("=== With Type III d=200  (acc = {:.1}%) ===", t3_acc * 100.0);
    print_rules(&t3);

    println!();
    println!("XOR uses only x0 and x1.  Type III should yield clauses with fewer");
    println!("spurious features (x2..x{}) while maintaining accuracy.", N_FEATURES - 1);
}
