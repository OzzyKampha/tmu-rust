//! Validate data-parallel training against exact training on a **real** dataset.
//!
//! Trains two identically-configured, identically-seeded Tsetlin Machines on the
//! Wisconsin Breast Cancer dataset (30 numeric features, 2 classes) — one with
//! the exact [`TsetlinMachine::fit_epoch`], one with the approximate
//! [`TsetlinMachine::fit_epoch_parallel`] (data-parallel over samples) — and
//! reports test accuracy for each so their gap is visible. Averaged over a few
//! seeds to smooth the data-parallel path's thread-count nondeterminism.
//!
//! Prepare the data first:
//!   pip install scikit-learn && python scripts/prepare_breast_cancer.py
//! Then (data-parallel only actually parallelises with the feature):
//!   cargo run --release --features parallel --example dp_vs_exact
//!
//! Optional args: <epochs> <seeds> (defaults 40 5).

use tmu_rs::{data, Encoder, Rng, TsetlinMachine};

fn split_encode(
    xs: &[Vec<f64>],
    ys: &[usize],
    seed: u64,
) -> (Encoder, tmu_rs::EncodedBatch, Vec<usize>, tmu_rs::EncodedBatch, Vec<usize>) {
    let mut idx: Vec<usize> = (0..xs.len()).collect();
    let mut rng = Rng::new(seed);
    for i in (1..idx.len()).rev() {
        idx.swap(i, rng.below(i + 1));
    }
    let cut = idx.len() * 4 / 5;
    let (tr_idx, te_idx) = idx.split_at(cut);
    let tr_rows: Vec<&[f64]> = tr_idx.iter().map(|&i| xs[i].as_slice()).collect();
    let te_rows: Vec<&[f64]> = te_idx.iter().map(|&i| xs[i].as_slice()).collect();
    let ytr: Vec<usize> = tr_idx.iter().map(|&i| ys[i]).collect();
    let yte: Vec<usize> = te_idx.iter().map(|&i| ys[i]).collect();
    let enc = Encoder::fit_numeric(&tr_rows, 10);
    let btr = enc.encode_batch_numeric(&tr_rows);
    let bte = enc.encode_batch_numeric(&te_rows);
    (enc, btr, ytr, bte, yte)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let epochs: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(40);
    let seeds: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);
    let path = "data/breast_cancer.csv";

    let (xs, ys) = data::read_numeric_csv(path).unwrap_or_else(|e| {
        eprintln!("could not read {path}: {e}\nRun: python scripts/prepare_breast_cancer.py");
        std::process::exit(1);
    });
    let n_classes = ys.iter().copied().max().unwrap() + 1;

    #[cfg(feature = "parallel")]
    let mode = format!("PARALLEL ({} threads)", rayon::current_num_threads());
    #[cfg(not(feature = "parallel"))]
    let mode = "SCALAR (data-parallel falls back to exact)".to_string();

    println!("Breast Cancer — {} samples, {} features, {n_classes} classes [{mode}]", xs.len(), xs[0].len());
    println!("exact fit_epoch  vs  approximate fit_epoch_parallel, {epochs} epochs, {seeds} seeds\n");
    println!("  {:>4} | {:>12} | {:>12} | {:>7}", "seed", "exact test", "DP test", "gap");
    println!("  {}", "-".repeat(46));

    let (mut sum_exact, mut sum_dp) = (0.0f64, 0.0f64);
    for seed in 0..seeds as u64 {
        let (enc, btr, ytr, bte, yte) = split_encode(&xs, &ys, seed + 1);

        // Two identically-seeded models: only the epoch method differs.
        let nf = enc.n_features();
        let mut exact = TsetlinMachine::with_config(n_classes, nf, 300, 100, 5.0, 8, true, 7);
        let mut dp = TsetlinMachine::with_config(n_classes, nf, 300, 100, 5.0, 8, true, 7);
        for _ in 0..epochs {
            exact.fit_epoch(&btr, &ytr);
            dp.fit_epoch_parallel(&btr, &ytr);
        }
        let ea = exact.accuracy(&bte, &yte);
        let da = dp.accuracy(&bte, &yte);
        sum_exact += ea;
        sum_dp += da;
        println!("  {seed:>4} | {ea:>11.4} | {da:>11.4} | {:>+7.4}", da - ea);
    }

    let (me, md) = (sum_exact / seeds as f64, sum_dp / seeds as f64);
    println!("  {}", "-".repeat(46));
    println!("  {:>4} | {me:>11.4} | {md:>11.4} | {:>+7.4}", "mean", md - me);
    println!(
        "\nData-parallel mean test accuracy is {:+.2} pts vs exact.",
        (md - me) * 100.0
    );
}
