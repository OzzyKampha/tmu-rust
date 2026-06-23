//! Network flow detection demo (benign vs. malicious) end to end:
//! synthetic numeric flow features -> quantile booleanizer -> bit-packed TM,
//! reporting accuracy/precision/recall per epoch and printing a few learned
//! clauses as human-readable detection rules.
//!
//! ```text
//! cargo run --release --example ndr_flows
//! ```
//!
//! NOTE: the data here is synthetic, purely to exercise the pipeline. In
//! production you'd feed real features (e.g. parsed from Zeek conn.log /
//! Suricata EVE / NetFlow) into the same `Encoder` + `TsetlinMachine`.

use tmu_rs::{Encoder, Rng, TsetlinMachine};

const FEATURES: [&str; 8] = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "src_pkts",
    "dst_pkts",
    "dport",
    "iat_var",
    "syn_ratio",
];

/// Sample a uniform random value in `[lo, hi)`.
fn uniform(r: &mut Rng, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * r.next_f64()
}
/// Sample from an exponential distribution with the given `mean`.
fn exponential(r: &mut Rng, mean: f64) -> f64 {
    -mean * (1.0 - r.next_f64()).ln()
}
/// Sample from a standard normal distribution using the Box–Muller transform.
fn normal(r: &mut Rng) -> f64 {
    let u1 = (1.0 - r.next_f64()).max(1e-12);
    let u2 = r.next_f64();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}
/// Sample from a log-normal distribution with log-space mean `mu` and std-dev `sigma`.
fn lognormal(r: &mut Rng, mu: f64, sigma: f64) -> f64 {
    (mu + sigma * normal(r)).exp()
}

/// Generate `n` synthetic network-flow records; `mal_frac` controls the fraction of malicious flows.
fn gen(n: usize, mal_frac: f64, seed: u64) -> (Vec<Vec<f64>>, Vec<usize>) {
    let mut r = Rng::new(seed);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    let scan_ports = [4444.0, 1337.0, 31337.0, 6667.0, 9001.0, 53.0];
    let svc_ports = [80.0, 443.0, 53.0, 22.0, 25.0, 8080.0, 3306.0];
    for _ in 0..n {
        let mal = r.next_f64() < mal_frac;
        let row = if !mal {
            vec![
                exponential(&mut r, 30.0),
                lognormal(&mut r, 7.0, 1.2),
                lognormal(&mut r, 8.0, 1.3),
                uniform(&mut r, 5.0, 200.0),
                uniform(&mut r, 5.0, 300.0),
                svc_ports[r.below(svc_ports.len())],
                uniform(&mut r, 0.5, 5.0),
                uniform(&mut r, 0.0, 0.2),
            ]
        } else {
            vec![
                exponential(&mut r, 3.0),
                lognormal(&mut r, 5.0, 1.0),
                lognormal(&mut r, 10.0, 1.0),
                uniform(&mut r, 1.0, 20.0),
                uniform(&mut r, 1.0, 15.0),
                scan_ports[r.below(scan_ports.len())],
                uniform(&mut r, 0.0, 0.3),
                uniform(&mut r, 0.6, 1.0),
            ]
        };
        xs.push(row);
        ys.push(mal as usize);
    }
    (xs, ys)
}

/// Run the full NDR pipeline: generate synthetic flows, encode, train TM, report metrics and rules.
fn main() {
    let (xtr, ytr) = gen(4000, 0.3, 1);
    let (xte, yte) = gen(1500, 0.3, 2);

    let xtr_ref: Vec<&[f64]> = xtr.iter().map(|r| r.as_slice()).collect();
    let xte_ref: Vec<&[f64]> = xte.iter().map(|r| r.as_slice()).collect();

    let encoder = Encoder::fit_numeric(&xtr_ref, 8);
    println!(
        "numeric features: 8  ->  binary features: {}\n",
        encoder.n_features()
    );

    let mut tm = TsetlinMachine::with_config(2, encoder.n_features(), 40, 50, 4.0, 8, true, 7);

    let packed_tr = encoder.encode_batch_numeric(&xtr_ref);
    let packed_te = encoder.encode_batch_numeric(&xte_ref);

    for epoch in 1..=8 {
        tm.fit_epoch(&packed_tr, &ytr);
        let preds = tm.predict_batch(&packed_te);
        let (mut tp, mut fp, mut tn, mut fn_) = (0usize, 0, 0, 0);
        for (&pred, &y) in preds.iter().zip(yte.iter()) {
            match (y, pred) {
                (1, 1) => tp += 1,
                (0, 1) => fp += 1,
                (0, 0) => tn += 1,
                _ => fn_ += 1,
            }
        }
        let acc = (tp + tn) as f64 / xte.len() as f64;
        let prec = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let rec = if tp + fn_ > 0 {
            tp as f64 / (tp + fn_) as f64
        } else {
            0.0
        };
        println!("epoch {epoch}  acc={acc:.4}  precision={prec:.3}  recall={rec:.3}");
    }

    println!("\nExample learned rules for class = malicious (positive clauses):");
    let mut shown = 0;
    for j in 0..tm.clauses_per_class() {
        if !tm.clause_is_positive(j) {
            continue;
        }
        let rule = tm.clause_rule(1, j);
        if rule.is_empty() || rule.len() > 5 {
            continue;
        }
        let parts: Vec<String> = rule
            .iter()
            .map(|&(bit, negated)| {
                let (f, thr) = encoder.bit_origin(bit);
                let op = if negated { "<=" } else { ">" };
                format!("{} {} {:.1}", FEATURES[f], op, thr)
            })
            .collect();
        println!(
            "  (w={}) IF {}",
            tm.clause_weight(1, j),
            parts.join(" AND ")
        );
        shown += 1;
        if shown == 6 {
            break;
        }
    }
}
