//! Growing TM demo — a cyber-defence detector that learns new adversary tooling
//! incrementally, without retraining and without forgetting what it already knows.
//!
//! Scenario (SOC / EDR): a Tsetlin Machine classifies process-creation events
//! (Sysmon Event ID 1) as **benign** or **malicious** from `col::val` tokens
//! (`proc::…`, `user::…`, later `parent::…`). It is trained on the tooling seen
//! so far. Then attackers rotate in new LOLBins the model has never observed.
//!
//! A fixed-vocabulary model collapses every unseen process into one `proc::<UNK>`
//! bucket, so it is blind to novel tooling. Instead of a full retrain:
//!
//! 1. [`Encoder::extend_categorical`] mints features for the new tokens —
//!    existing feature indices stay stable;
//! 2. [`TsetlinMachine::grow_features`] widens the machine to match —
//!    learned clauses are preserved bit-for-bit, new literals start excluded,
//!    so detection on old tooling is unchanged;
//! 3. a short fine-tune teaches the new tooling; old detections are untouched.
//!
//! Being a Tsetlin Machine, the detector shows its work: it prints the exact
//! token conjunctions it flags as malicious, before and after growing.
//!
//! `cargo run --release --example growing`

use std::time::Instant;
use tmu_rs::{Encoder, Rng, TsetlinMachine};

/// Malicious class index (2-class model: 0 = benign, 1 = malicious).
const MAL: usize = 1;

/// Processes seen on Day 0. Label = 1 (malicious) iff the process is attacker tooling.
const PROCS_A: &[(&str, usize)] = &[
    ("proc::explorer.exe", 0),
    ("proc::winword.exe", 0),
    ("proc::chrome.exe", 0),
    ("proc::mimikatz.exe", 1),
    ("proc::psexec.exe", 1),
];

/// Processes that only appear later — unseen tokens for the deployed detector.
const PROCS_B: &[(&str, usize)] = &[
    ("proc::outlook.exe", 0),
    ("proc::teams.exe", 0),
    ("proc::rundll32.exe", 1),
    ("proc::certutil.exe", 1),
];

const USERS: &[&str] = &["user::alice", "user::bob", "user::carol"];

/// Parent-process column: a whole new field that only appears in later events.
const PARENTS_B: &[&str] = &["parent::services.exe", "parent::explorer.exe"];

/// Generate `n` events drawing processes from `procs`; later events also carry
/// the new `parent::` column.
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

/// Classify a single event and print it as a SOC verdict with a confidence margin.
///
/// The margin is `score[MAL] - score[BENIGN]`; near-zero means the detector is
/// effectively guessing (e.g. the process collapsed into `proc::<UNK>`).
fn triage(enc: &Encoder, tm: &TsetlinMachine, tokens: &[&str], truth: usize) {
    let s = enc.encode_one_categorical(tokens);
    let mut scores = [0i32; 2];
    tm.scores(&s, &mut scores);
    let verdict = if tm.predict(&s) == MAL { "MALICIOUS" } else { "BENIGN   " };
    let margin = scores[MAL] - scores[1 - MAL];
    let truth_s = if truth == MAL { "malicious" } else { "benign" };
    let hit = if tm.predict(&s) == truth { "✓" } else { "✗ MISS" };
    println!(
        "  [{verdict}] margin {margin:+5}  (truth: {truth_s:<9}) {hit}   {}",
        tokens.join(", ")
    );
}

/// Print the token conjunctions the detector flags as malicious.
///
/// Walks the positive clauses of the malicious class, maps each included literal
/// back to its `col::val` token via the encoder, and renders `A ∧ ¬B` rules —
/// mirrors the interpretability example's `render`/frequency roll-up.
fn print_detection_logic(enc: &Encoder, tm: &TsetlinMachine, top_n: usize) {
    let cpc = tm.clauses_per_class();
    let mut shown = 0;
    for j in 0..cpc {
        if !tm.clause_is_positive(j) {
            continue;
        }
        let rule = tm.clause_rule(MAL, j);
        if rule.is_empty() {
            continue;
        }
        let body = rule
            .iter()
            .map(|&(feat, neg)| {
                let tok = enc.vocab_token(feat);
                if neg {
                    format!("¬{tok}")
                } else {
                    tok.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(" ∧ ");
        println!("    flag malicious when:  {body}   (weight {})", tm.clause_weight(MAL, j));
        shown += 1;
        if shown >= top_n {
            break;
        }
    }

    // Token-frequency roll-up: which tokens most drive malicious verdicts.
    let mut freq: std::collections::BTreeMap<String, usize> = Default::default();
    for j in 0..cpc {
        if !tm.clause_is_positive(j) {
            continue;
        }
        for (feat, neg) in tm.clause_rule(MAL, j) {
            if !neg {
                *freq.entry(enc.vocab_token(feat).to_string()).or_default() += 1;
            }
        }
    }
    let mut ranked: Vec<_> = freq.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let summary = ranked
        .iter()
        .take(4)
        .map(|(t, c)| format!("{t}×{c}"))
        .collect::<Vec<_>>()
        .join("  ");
    if !summary.is_empty() {
        println!("    top malicious indicators:  {summary}");
    }
}

fn main() {
    println!("=== Growing cyber-defence detector (Sysmon process-creation triage) ===\n");
    println!("(release build; throughput numbers are meaningful only with --release)\n");

    // ── Day 0: fit encoder + train the detector on known tooling ────────────
    let (xa_tr, ya_tr) = make_data(4000, PROCS_A, false, 1);
    let (xa_te, ya_te) = make_data(2000, PROCS_A, false, 2);

    let toks = as_token_slices(&xa_tr);
    let refs: Vec<&[&str]> = toks.iter().map(|t| t.as_slice()).collect();
    let mut enc = Encoder::fit_categorical(&refs);

    let mut tm = TsetlinMachine::new(2, enc.n_features(), 20, 15, 3.9);
    let ba_tr = encode(&enc, &xa_tr);
    let t_train0 = Instant::now();
    for _ in 0..20 {
        tm.fit_epoch(&ba_tr, &ya_tr);
    }
    let train0_secs = t_train0.elapsed().as_secs_f64();

    let acc_a_before = tm.accuracy(&encode(&enc, &xa_te), &ya_te);
    println!("Day 0 — trained on known tooling ({} vocab features)", enc.n_features());
    println!("  detection accuracy on known tooling: {acc_a_before:.3}");
    println!("  initial training: {:.0} ms for 4k events × 20 epochs\n", train0_secs * 1e3);

    println!("  what the detector learned to flag:");
    print_detection_logic(&enc, &tm, 6);

    // ── Day N: new adversary tooling appears in telemetry ───────────────────
    let (xb_tr, yb_tr) = make_data(4000, PROCS_B, true, 3);
    let (xb_te, yb_te) = make_data(2000, PROCS_B, true, 4);

    println!("\nDay N — new tooling observed in the wild (never seen in training).");
    println!("  live triage BEFORE growing (all new binaries collapse into proc::<UNK>):");
    triage(&enc, &tm, &["proc::certutil.exe", "parent::services.exe", "user::bob"], MAL);
    triage(&enc, &tm, &["proc::rundll32.exe", "parent::services.exe", "user::alice"], MAL);
    triage(&enc, &tm, &["proc::outlook.exe", "parent::explorer.exe", "user::carol"], 0);
    let acc_b_ungrown = tm.accuracy(&encode(&enc, &xb_te), &yb_te);
    println!("  accuracy on new tooling before growing: {acc_b_ungrown:.3}  (≈ coin flip — blind)");

    // ── Grow: extend the vocabulary and widen the detector ──────────────────
    let toks_b = as_token_slices(&xb_tr);
    let refs_b: Vec<&[&str]> = toks_b.iter().map(|t| t.as_slice()).collect();
    let added = enc.extend_categorical(&refs_b);
    let t_grow = Instant::now();
    if added > 0 {
        tm.grow_features(enc.n_features());
    }
    let grow_us = t_grow.elapsed().as_secs_f64() * 1e6;
    println!(
        "\n  extended vocabulary by {added} features -> {} total; grew detector in {grow_us:.1} µs",
        enc.n_features()
    );

    // Zero-forgetting: detection on known tooling is byte-identical after the grow.
    let acc_a_grown = tm.accuracy(&encode(&enc, &xa_te), &ya_te);
    println!("  accuracy on known tooling immediately after grow: {acc_a_grown:.3}  (unchanged)");
    assert_eq!(
        acc_a_before, acc_a_grown,
        "grow must not change behaviour on previously-known tooling"
    );

    // ── Fine-tune on the new tooling (old detections stay put) ──────────────
    let (ba_tr, bb_tr) = (encode(&enc, &xa_tr), encode(&enc, &xb_tr));
    let t_ft = Instant::now();
    for _ in 0..20 {
        tm.fit_epoch(&ba_tr, &ya_tr);
        tm.fit_epoch(&bb_tr, &yb_tr);
    }
    let ft_secs = t_ft.elapsed().as_secs_f64();

    let acc_a_final = tm.accuracy(&encode(&enc, &xa_te), &ya_te);
    let acc_b_final = tm.accuracy(&encode(&enc, &xb_te), &yb_te);
    println!("\n  fine-tuned on new tooling in {:.0} ms (vs. a full retrain from scratch).", ft_secs * 1e3);
    println!("  live triage AFTER growing + fine-tune:");
    triage(&enc, &tm, &["proc::certutil.exe", "parent::services.exe", "user::bob"], MAL);
    triage(&enc, &tm, &["proc::rundll32.exe", "parent::services.exe", "user::alice"], MAL);
    triage(&enc, &tm, &["proc::outlook.exe", "parent::explorer.exe", "user::carol"], 0);
    println!("  detection accuracy — known tooling: {acc_a_final:.3}   new tooling: {acc_b_final:.3}");

    println!("\n  what the detector flags now (old rules survive, new rules added):");
    print_detection_logic(&enc, &tm, 8);

    // ── Throughput: can it keep up with event volume? ───────────────────────
    let (xbench, ybench) = {
        // Mix of old + new tooling, large batch, to measure line-rate capacity.
        let (mut xs, mut ys) = make_data(50_000, PROCS_A, true, 10);
        let (xb, yb) = make_data(50_000, PROCS_B, true, 11);
        xs.extend(xb);
        ys.extend(yb);
        (xs, ys)
    };
    let bbench = encode(&enc, &xbench);
    let t_infer = Instant::now();
    let preds = tm.predict_batch(&bbench);
    let infer_secs = t_infer.elapsed().as_secs_f64();
    let eps = xbench.len() as f64 / infer_secs;
    let correct = preds.iter().zip(&ybench).filter(|(p, y)| p == y).count();
    println!("\nThroughput — classified {} events in {:.1} ms", xbench.len(), infer_secs * 1e3);
    println!("  {eps:.0} events/sec   ({:.3} accuracy on the mixed stream)", correct as f64 / xbench.len() as f64);

    assert!(acc_a_final > 0.95, "known-tooling detection degraded: {acc_a_final}");
    assert!(acc_b_final > 0.95, "new tooling not learned: {acc_b_final}");
    println!("\n✓ Detector learned new adversary tooling with zero forgetting of prior detections.");
}
