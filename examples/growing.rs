//! Growing TM demo — extend the encoder with new tokens and grow the machine,
//! keeping every learned automaton.
//!
//! Scenario: a classifier is trained on categorical `col::val` events (phase A).
//! Later, new data arrives containing tokens (and a whole column) never seen in
//! training (phase B). Instead of retraining from scratch:
//!
//! 1. [`Encoder::extend_categorical`] appends the new tokens as new features —
//!    all existing feature indices stay stable;
//! 2. [`TsetlinMachine::grow_features`] grows the literal space to match —
//!    learned clauses are preserved bit-for-bit and the new literals start
//!    excluded, so behaviour on old data is unchanged;
//! 3. training simply continues on the combined data.
//!
//! `cargo run --release --example growing`

use tmu_rs::{Encoder, Rng, TsetlinMachine};

/// Processes seen in phase A. Label = 1 (suspicious) iff the process is bad.
const PROCS_A: &[(&str, usize)] = &[
    ("proc::explorer.exe", 0),
    ("proc::winword.exe", 0),
    ("proc::chrome.exe", 0),
    ("proc::mimikatz.exe", 1),
    ("proc::psexec.exe", 1),
];

/// Processes that only appear in phase B — unseen tokens for the encoder.
const PROCS_B: &[(&str, usize)] = &[
    ("proc::outlook.exe", 0),
    ("proc::teams.exe", 0),
    ("proc::rundll32.exe", 1),
    ("proc::certutil.exe", 1),
];

const USERS: &[&str] = &["user::alice", "user::bob", "user::carol"];

/// Parent-process column: only present in phase B events.
const PARENTS_B: &[&str] = &["parent::services.exe", "parent::explorer.exe"];

/// Generate `n` samples drawing processes from `procs`; phase-B samples also
/// carry the new `parent::` column.
fn make_data(
    n: usize,
    procs: &[(&str, usize)],
    with_parent: bool,
    seed: u64,
) -> (Vec<Vec<String>>, Vec<usize>) {
    let mut rng = Rng::new(seed);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let (proc_tok, label) = procs[(rng.next_u64() % procs.len() as u64) as usize];
        let user = USERS[(rng.next_u64() % USERS.len() as u64) as usize];
        let mut sample = vec![proc_tok.to_string(), user.to_string()];
        if with_parent {
            sample.push(PARENTS_B[(rng.next_u64() % PARENTS_B.len() as u64) as usize].to_string());
        }
        xs.push(sample);
        ys.push(label);
    }
    (xs, ys)
}

fn as_token_slices(xs: &[Vec<String>]) -> Vec<Vec<&str>> {
    xs.iter()
        .map(|s| s.iter().map(|t| t.as_str()).collect())
        .collect()
}

fn encode(enc: &Encoder, xs: &[Vec<String>]) -> tmu_rs::EncodedBatch {
    let toks = as_token_slices(xs);
    let refs: Vec<&[&str]> = toks.iter().map(|t| t.as_slice()).collect();
    enc.encode_batch_categorical(&refs)
}

fn main() {
    // ── Phase A: fit encoder + train on the initial data ────────────────────
    let (xa_tr, ya_tr) = make_data(4000, PROCS_A, false, 1);
    let (xa_te, ya_te) = make_data(1000, PROCS_A, false, 2);

    let toks = as_token_slices(&xa_tr);
    let refs: Vec<&[&str]> = toks.iter().map(|t| t.as_slice()).collect();
    let mut enc = Encoder::fit_categorical(&refs);
    println!("phase A vocabulary: {} features", enc.n_features());

    let mut tm = TsetlinMachine::new(2, enc.n_features(), 20, 15, 3.9);
    let ba_tr = encode(&enc, &xa_tr);
    for _ in 0..20 {
        tm.fit_epoch(&ba_tr, &ya_tr);
    }
    let acc_a_before = tm.accuracy(&encode(&enc, &xa_te), &ya_te);
    println!("accuracy on A after phase-A training: {acc_a_before:.3}");

    // ── Phase B: new data with unseen tokens and a new column ───────────────
    let (xb_tr, yb_tr) = make_data(4000, PROCS_B, true, 3);
    let (xb_te, yb_te) = make_data(1000, PROCS_B, true, 4);

    // Before growing, phase-B processes all collapse into the proc::<UNK> slot —
    // the model cannot tell them apart.
    let acc_b_ungrown = tm.accuracy(&encode(&enc, &xb_te), &yb_te);
    println!("accuracy on B before growing (all new tokens -> <UNK>): {acc_b_ungrown:.3}");

    // Extend the encoder and grow the machine to match.
    let toks_b = as_token_slices(&xb_tr);
    let refs_b: Vec<&[&str]> = toks_b.iter().map(|t| t.as_slice()).collect();
    let added = enc.extend_categorical(&refs_b);
    if added > 0 {
        tm.grow_features(enc.n_features());
    }
    println!(
        "extended encoder with {added} new features -> {} total; TM grown to match",
        enc.n_features()
    );

    // Learned automata are untouched: accuracy on A is preserved exactly
    // (every phase-A token was already in the vocabulary, so A encodes
    // bit-identically under the grown encoder).
    let acc_a_grown = tm.accuracy(&encode(&enc, &xa_te), &ya_te);
    println!("accuracy on A after grow, before any new training: {acc_a_grown:.3}");
    assert_eq!(
        acc_a_before, acc_a_grown,
        "grow must not change behaviour on previously-known data"
    );

    // ── Continue training on A ∪ B ──────────────────────────────────────────
    let (ba_tr, bb_tr) = (encode(&enc, &xa_tr), encode(&enc, &xb_tr));
    for _ in 0..20 {
        tm.fit_epoch(&ba_tr, &ya_tr);
        tm.fit_epoch(&bb_tr, &yb_tr);
    }

    let acc_a_final = tm.accuracy(&encode(&enc, &xa_te), &ya_te);
    let acc_b_final = tm.accuracy(&encode(&enc, &xb_te), &yb_te);
    println!("after continued training:  A = {acc_a_final:.3}   B = {acc_b_final:.3}");

    assert!(acc_a_final > 0.95, "old task degraded: {acc_a_final}");
    assert!(acc_b_final > 0.95, "new task not learned: {acc_b_final}");
    println!("grown TM learned the new tokens without forgetting the old ones ✓");
}
