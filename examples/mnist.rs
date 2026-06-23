//! MNIST demo — mirrors TMU's `MNISTDemo` / `MNISTDemoWeightedClauses`. Loads
//! binarized MNIST (pixels already thresholded to 0/1) and trains the weighted
//! TM. Reports per-epoch test accuracy, throughput, and timing.
//!
//! TMU defaults: num_clauses=2000, T=5000, s=10.0, max_included_literals=32,
//!               weighted_clauses=True, epochs=60, seed=42
//!
//! Prepare the data first:
//!   python scripts/prepare_mnist.py
//!
//! Run (TMU defaults — matches reference accuracy ~98.5%):
//!   cargo run --release --features parallel --example mnist
//! Best speed — enable AVX2 auto-vectorisation:
//!   RUSTFLAGS="-C target-cpu=native" cargo run --release --features parallel --example mnist
//!
//! Args: [epochs=60] [clauses_per_class=2000]

use std::io::Write;
use std::time::Instant;
use tmu_rs::{data, Encoder, TsetlinMachine};

/// Load binarized MNIST, encode it once, and run the training loop reporting per-epoch accuracy and throughput.
fn main() {
    let mut args = std::env::args().skip(1);
    let epochs: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(2);
    let clauses: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(2000);

    // ---- load data ----------------------------------------------------------
    print!("Loading data... ");
    std::io::stdout().flush().ok();
    let t_load = Instant::now();

    let load = |p: &str| {
        data::read_binary_csv(p).unwrap_or_else(|e| {
            eprintln!("could not read {p}: {e}\nRun: python scripts/prepare_mnist.py");
            std::process::exit(1);
        })
    };
    let (xtr, ytr) = load("data/mnist_train_bin.csv");
    let (xte, yte) = load("data/mnist_test_bin.csv");
    let n_features = xtr[0].len(); // 784

    println!("done ({:.2}s)", t_load.elapsed().as_secs_f64());
    println!(
        "  train={:<6}  test={:<5}  features={n_features}",
        xtr.len(),
        xte.len()
    );

    // ---- model config -------------------------------------------------------
    let threshold = 5000i32;
    let s = 10.0f64;
    println!("  clauses/class={clauses}  T={threshold}  s={s}  epochs={epochs}  classes=10");
    #[cfg(feature = "parallel")]
    println!("  parallel=ON  (rayon)");
    #[cfg(not(feature = "parallel"))]
    println!("  parallel=OFF — add --features parallel for multi-threaded training");
    println!("  tip: prefix with RUSTFLAGS=\"-C target-cpu=native\" for AVX2 vectorisation");
    println!();

    let encoder = Encoder::for_binary(n_features);
    let mut tm =
        TsetlinMachine::with_config(10, encoder.n_features(), clauses, threshold, s, 8, true, 42)
            .max_included_literals(32);

    // ---- encode once --------------------------------------------------------
    print!("Encoding data... ");
    std::io::stdout().flush().ok();
    let t_pack = Instant::now();
    let xtr_r: Vec<&[u8]> = xtr.iter().map(|v| v.as_slice()).collect();
    let xte_r: Vec<&[u8]> = xte.iter().map(|v| v.as_slice()).collect();
    let packed_tr = encoder.encode_batch(&xtr_r);
    let packed_te = encoder.encode_batch(&xte_r);
    println!("done ({:.2}s)", t_pack.elapsed().as_secs_f64());
    println!();

    // ---- training loop ------------------------------------------------------
    let mut eta_secs: Option<f64> = None;
    let t_total = Instant::now();

    let mut accuracy_log: Vec<f64> = Vec::with_capacity(epochs);
    let mut train_time_log: Vec<f64> = Vec::with_capacity(epochs);
    let mut test_time_log: Vec<f64> = Vec::with_capacity(epochs);

    for epoch in 1..=epochs {
        if let Some(eta) = eta_secs {
            print!("epoch {epoch:>2}/{epochs}  training…  (ETA ~{eta:.0}s)   ");
        } else {
            print!("epoch {epoch:>2}/{epochs}  training…");
        }
        std::io::stdout().flush().ok();

        let t_train = Instant::now();
        tm.fit_epoch(&packed_tr, &ytr);
        let train_secs = t_train.elapsed().as_secs_f64();
        let throughput = xtr.len() as f64 / train_secs;

        eta_secs = Some(train_secs * (epochs - epoch) as f64);

        print!("\repoch {epoch:>2}/{epochs}  evaluating…                                        ");
        std::io::stdout().flush().ok();

        let t_eval = Instant::now();
        let acc = tm.accuracy(&packed_te, &yte);
        let eval_secs = t_eval.elapsed().as_secs_f64();

        println!(
            "\repoch {epoch:>2}/{epochs}  test={acc:.4}  \
             train={train_secs:.1}s ({throughput:.0} samp/s)  eval={eval_secs:.1}s"
        );

        accuracy_log.push(acc * 100.0);
        train_time_log.push(train_secs);
        test_time_log.push(eval_secs);
    }

    println!("\ntotal: {:.1}s", t_total.elapsed().as_secs_f64());

    // ---- TMU-compatible results summary -------------------------------------
    let fmt_list = |v: &[f64]| {
        let inner: Vec<String> = v.iter().map(|x| format!("{x}")).collect();
        format!("[{}]", inner.join(", "))
    };
    println!(
        "{{'accuracy': {}, 'train_time': {}, 'test_time': {}, \
         'args': {{'num_clauses': {clauses}, 'T': {threshold}, 's': {s}, \
         'max_included_literals': 32, 'platform': 'CPU', 'weighted_clauses': True, 'epochs': {epochs}}}}}",
        fmt_list(&accuracy_log),
        fmt_list(&train_time_log),
        fmt_list(&test_time_log),
    );
}
