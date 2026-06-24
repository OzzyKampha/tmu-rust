//! TMAutoEncoder throughput and accuracy benchmark.
//!
//! Runs two configs sequentially:
//!   1. Small (accuracy): 20-feature, prints reconstruction accuracy per epoch
//!   2. Large (throughput): 200-feature, measures Mclause-updates/s
//!
//! Run:
//!   cargo run --release                     --example bench_autoencoder
//!   cargo run --release --features parallel --example bench_autoencoder

use std::time::{Duration, Instant};
use tmu_rs::{Encoder, Rng, TMAutoEncoder};

struct Config {
    label: &'static str,
    n_features: usize,
    clauses_per_output: usize,
    threshold: i32,
    s: f64,
    n_train: usize,
    n_warmup: usize,
    n_bench: usize,
    print_accuracy: bool,
}

const SMALL: Config = Config {
    label: "small (accuracy check)",
    n_features: 20,
    clauses_per_output: 40,
    threshold: 20,
    s: 3.9,
    n_train: 2_000,
    n_warmup: 0,
    n_bench: 20,
    print_accuracy: true,
};

const LARGE: Config = Config {
    label: "large (throughput)",
    n_features: 200,
    clauses_per_output: 50,
    threshold: 200,
    s: 2.0,
    n_train: 2_000,
    n_warmup: 2,
    n_bench: 8,
    print_accuracy: false,
};

fn run_bench(cfg: &Config) {
    let Config {
        label,
        n_features,
        clauses_per_output,
        threshold,
        s,
        n_train,
        n_warmup,
        n_bench,
        print_accuracy,
    } = cfg;

    let mut rng = Rng::new(42);
    let xs: Vec<Vec<u8>> = (0..*n_train)
        .map(|_| (0..*n_features).map(|_| (rng.next_u64() & 1) as u8).collect())
        .collect();
    let xs_ref: Vec<&[u8]> = xs.iter().map(|v| v.as_slice()).collect();

    let encoder = Encoder::for_binary(*n_features);
    let packed = encoder.encode_batch(&xs_ref);

    let mut ae = TMAutoEncoder::with_config(
        *n_features,
        *clauses_per_output,
        *threshold,
        *s,
        8,
        true,
        42,
    );

    // Each epoch = n_train samples × n_features outputs × clauses_per_output clauses.
    let clause_updates_per_epoch = n_train * n_features * clauses_per_output;

    #[cfg(feature = "parallel")]
    let mode = "PARALLEL";
    #[cfg(not(feature = "parallel"))]
    let mode = "SEQUENTIAL";

    println!("\nMode   : {mode}  [{label}]");
    println!(
        "Config : {} features · {} clauses/output · T={} · s={} · {} training samples",
        n_features, clauses_per_output, threshold, s, n_train
    );
    println!(
        "Workload: {} M clause updates per epoch\n",
        clause_updates_per_epoch / 1_000_000
    );

    let header = if *print_accuracy {
        format!(
            "{:>5}  {:>9}  {:>13}  {:>15}  {:>8}",
            "epoch", "ms", "samples/s", "Mclause-ups/s", "recon-acc"
        )
    } else {
        format!(
            "{:>5}  {:>9}  {:>13}  {:>15}",
            "epoch", "ms", "samples/s", "Mclause-ups/s"
        )
    };
    println!("{header}");

    for _ in 0..*n_warmup {
        ae.fit_epoch(&packed);
    }

    let mut times: Vec<Duration> = Vec::with_capacity(*n_bench);
    for epoch in 0..*n_bench {
        let t = Instant::now();
        ae.fit_epoch(&packed);
        let dur = t.elapsed();
        times.push(dur);

        let ms = dur.as_secs_f64() * 1_000.0;
        let sps = *n_train as f64 / dur.as_secs_f64();
        let mcps = clause_updates_per_epoch as f64 / dur.as_secs_f64() / 1e6;

        if *print_accuracy {
            let acc = ae.reconstruction_accuracy(&packed);
            println!(
                "{:>5}  {:>8.1}  {:>13.0}  {:>15.1}  {:>7.4}",
                epoch, ms, sps, mcps, acc
            );
        } else {
            println!("{:>5}  {:>8.1}  {:>13.0}  {:>15.1}", epoch, ms, sps, mcps);
        }
    }

    times.sort();
    let median_s = times[times.len() / 2].as_secs_f64();
    let min_s = times[0].as_secs_f64();
    let max_s = times[times.len() - 1].as_secs_f64();
    let mean_s = times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / times.len() as f64;

    println!();
    println!(
        "── Summary ({} timed epochs) ──────────────────────────────────────",
        n_bench
    );
    println!(
        "  median {:7.1} ms  |  mean {:7.1} ms  |  min {:7.1} ms  |  max {:7.1} ms",
        median_s * 1_000.0,
        mean_s * 1_000.0,
        min_s * 1_000.0,
        max_s * 1_000.0
    );
    println!(
        "  throughput  : {:9.0} samples/s       {:7.1} Mclause-updates/s",
        *n_train as f64 / median_s,
        clause_updates_per_epoch as f64 / median_s / 1e6
    );
}

fn main() {
    run_bench(&SMALL);
    run_bench(&LARGE);
}
