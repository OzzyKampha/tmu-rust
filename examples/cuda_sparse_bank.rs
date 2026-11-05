//! GPU training speed of the sparse clause bank (absorbing, literal sampling)
//! on wide, sparse inputs — the shape of log-event models.
//!
//! ```sh
//! cargo run --release --features cuda --example cuda_sparse_bank
//! cargo run --release --features cuda --example cuda_sparse_bank -- 44428 2000 1500
//! ```
//!
//! Arguments: features, clauses per class, samples. Trains three epochs on the
//! CPU and on the GPU from the same seed (the first is the slow one: every
//! clause starts with all its sampled candidate literals), reports samples/s
//! for both, and checks that the GPU-trained model is bit-identical to the
//! CPU-trained one.

use std::time::Instant;

use tmu_rs::cuda::CudaDevice;
use tmu_rs::{Encoder, Rng, TMSparseClassifier};

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("features clauses samples"))
        .collect();
    let n_features = args.first().copied().unwrap_or(44_428);
    let clauses = args.get(1).copied().unwrap_or(2000);
    let n_train = args.get(2).copied().unwrap_or(1500);
    let dev = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("No usable CUDA GPU: {e}");
            return;
        }
    };
    println!("device: {}", dev.name());
    println!("{n_features} features, {clauses} clauses per class, {n_train} samples");

    // ~40 features set per sample; class 1 carries one of a few marker features.
    let mut r = Rng::new(7);
    let mut xs = Vec::with_capacity(n_train);
    let mut ys = Vec::with_capacity(n_train);
    for _ in 0..n_train {
        let mut x = vec![0u8; n_features];
        for _ in 0..40 {
            x[(r.next_u64() % n_features as u64) as usize] = 1;
        }
        let y = r.next_u64().is_multiple_of(5) as usize;
        if y == 1 {
            x[(r.next_u64() % 8) as usize] = 1;
        }
        xs.push(x);
        ys.push(y);
    }
    let rows: Vec<&[u8]> = xs.iter().map(|x| x.as_slice()).collect();
    let batch = Encoder::for_binary(n_features).encode_batch(&rows);
    drop(xs);

    let make = || {
        TMSparseClassifier::with_config(2, n_features, clauses, 4000, 2.0, 8, true, 42)
            .max_included_literals(20)
            .absorbing(100)
            .literal_sampling(0.1)
    };
    let mut cpu = make();
    let mut gpu = make().to_cuda(&dev).expect("upload model");

    for epoch in 1..=3 {
        let t = Instant::now();
        cpu.fit_epoch(&batch, &ys);
        let cpu_s = t.elapsed().as_secs_f64();

        let t = Instant::now();
        gpu.fit_epoch(&batch, &ys).expect("GPU training");
        dev.synchronize().expect("sync");
        let gpu_s = t.elapsed().as_secs_f64();
        println!(
            "epoch {epoch}: CPU {:.0} samples/s  GPU {:.0} samples/s  ({:.2}x)",
            n_train as f64 / cpu_s,
            n_train as f64 / gpu_s,
            cpu_s / gpu_s
        );
    }

    let back = gpu.to_cpu().expect("download model");
    let same = back.predict_batch(&batch) == cpu.predict_batch(&batch)
        && (0..2).all(|c| {
            (0..clauses).all(|j| {
                back.clause_weight(c, j) == cpu.clause_weight(c, j)
                    && back.clause_rule(c, j) == cpu.clause_rule(c, j)
            })
        });
    println!("GPU model identical to CPU model: {same}");
}
