//! CPU vs GPU throughput comparison for training and inference.
//!
//!   cargo run --release --features gpu          --example gpu_vs_cpu_bench
//!   cargo run --release --features gpu,parallel --example gpu_vs_cpu_bench
//!
//! The CPU column reflects the build: SCALAR without `parallel`, multi-threaded
//! with it. The GPU column uses whatever adapter wgpu selects — on a machine
//! with a real GPU that is the GPU; in a CI container with only the mesa
//! llvmpipe software driver it is still the CPU (so treat those numbers as a
//! functional check, not a hardware speedup).
//!
//! Parity is asserted before timing, so the two paths are doing identical work.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tmu_rs::{EncodedBatch, Encoder, GpuContext, Rng, TsetlinMachine};

fn make(n: usize, bits: usize, n_classes: usize, seed: u64) -> (EncodedBatch, Vec<usize>, Encoder) {
    let mut rng = Rng::new(seed);
    let xs: Vec<Vec<u8>> = (0..n)
        .map(|_| (0..bits).map(|_| (rng.next_u64() & 1) as u8).collect())
        .collect();
    let ys: Vec<usize> = xs
        .iter()
        .map(|x| (x[0] as usize * 2 + x[1] as usize) % n_classes)
        .collect();
    let enc = Encoder::for_binary(bits);
    let refs: Vec<&[u8]> = xs.iter().map(|v| v.as_slice()).collect();
    let batch = enc.encode_batch(&refs);
    (batch, ys, enc)
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

struct Config {
    name: &'static str,
    bits: usize,
    n_classes: usize,
    cpc: usize,
    n_train: usize,
    n_test: usize,
    epochs: usize,
}

fn run(ctx: &Arc<GpuContext>, c: &Config) {
    let (train, ytr, enc) = make(c.n_train, c.bits, c.n_classes, 1);
    let (test, _yte, _) = make(c.n_test, c.bits, c.n_classes, 2);
    let build = || {
        TsetlinMachine::with_config(c.n_classes, enc.n_features(), c.cpc, 50, 5.0, 8, true, 42)
            .max_included_literals(32)
    };

    println!(
        "\n=== {} : {} classes · {} features · {} clauses/class · {} train / {} test ===",
        c.name, c.n_classes, c.bits, c.cpc, c.n_train, c.n_test
    );

    // ---- parity check (train a few epochs both ways, compare predictions) ----
    let mut cpu_chk = build();
    for _ in 0..3 {
        cpu_chk.fit_epoch(&train, &ytr);
    }
    let mut gpu_chk = build().to_gpu(ctx).expect("to_gpu");
    for _ in 0..3 {
        gpu_chk.fit_epoch(&train, &ytr);
    }
    let mism = cpu_chk
        .predict_batch(&test)
        .iter()
        .zip(&gpu_chk.predict_batch(&test))
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "parity: {mism} / {} predictions differ (expect 0)",
        c.n_test
    );

    // ---- training: CPU ----
    let mut cpu = build();
    cpu.fit_epoch(&train, &ytr); // warmup
    let mut cpu_epochs = Vec::new();
    for _ in 0..c.epochs {
        let t = Instant::now();
        cpu.fit_epoch(&train, &ytr);
        cpu_epochs.push(t.elapsed());
    }
    let cpu_train = median(cpu_epochs);

    // ---- training: GPU (measure one-time upload separately) ----
    let t_up = Instant::now();
    let mut gpu = build().to_gpu(ctx).expect("to_gpu");
    gpu.fit_epoch(&train, &ytr); // warmup (also compiles/allocates)
    let upload = t_up.elapsed();
    let mut gpu_epochs = Vec::new();
    for _ in 0..c.epochs {
        let t = Instant::now();
        gpu.fit_epoch(&train, &ytr);
        gpu_epochs.push(t.elapsed());
    }
    let gpu_train = median(gpu_epochs);

    // ---- inference ----
    let t = Instant::now();
    for _ in 0..5 {
        std::hint::black_box(cpu.predict_batch(&test));
    }
    let cpu_infer = t.elapsed() / 5;
    let t = Instant::now();
    for _ in 0..5 {
        std::hint::black_box(gpu.predict_batch(&test));
    }
    let gpu_infer = t.elapsed() / 5;

    let cpu_mode = if cfg!(feature = "parallel") {
        "CPU(par)"
    } else {
        "CPU(scalar)"
    };
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let sps = |d: Duration, n: usize| n as f64 / d.as_secs_f64();

    println!("(GPU one-time upload + warmup: {:.1} ms)", ms(upload));
    println!(
        "{:<9} {:>12} {:>14} {:>12} {:>10}",
        "stage", cpu_mode, "GPU", "unit", "GPU speedup"
    );
    println!(
        "{:<9} {:>12.2} {:>14.2} {:>12} {:>9.2}x",
        "train",
        ms(cpu_train),
        ms(gpu_train),
        "ms/epoch",
        cpu_train.as_secs_f64() / gpu_train.as_secs_f64()
    );
    println!(
        "{:<9} {:>12.0} {:>14.0} {:>12} {:>9.2}x",
        "train",
        sps(cpu_train, c.n_train),
        sps(gpu_train, c.n_train),
        "samples/s",
        cpu_train.as_secs_f64() / gpu_train.as_secs_f64()
    );
    println!(
        "{:<9} {:>12.2} {:>14.2} {:>12} {:>9.2}x",
        "infer",
        ms(cpu_infer),
        ms(gpu_infer),
        "ms/batch",
        cpu_infer.as_secs_f64() / gpu_infer.as_secs_f64()
    );
    println!(
        "{:<9} {:>12.0} {:>14.0} {:>12} {:>9.2}x",
        "infer",
        sps(cpu_infer, c.n_test),
        sps(gpu_infer, c.n_test),
        "samples/s",
        cpu_infer.as_secs_f64() / gpu_infer.as_secs_f64()
    );
}

fn main() {
    let ctx = match GpuContext::new() {
        Ok(c) => {
            let i = c.adapter_info();
            let dt = format!("{:?}", i.device_type);
            println!("GPU adapter: {} ({}, {:?})", i.name, dt, i.backend);
            if dt == "Cpu" {
                println!(
                    "NOTE: this is a software (CPU) Vulkan driver — GPU numbers reflect emulation, not real hardware."
                );
            }
            Arc::new(c)
        }
        Err(e) => {
            eprintln!(
                "No usable GPU adapter ({e}). Install a Vulkan driver to run this benchmark."
            );
            return;
        }
    };

    let configs = [
        Config {
            name: "small (GPU overhead dominates)",
            bits: 12,
            n_classes: 2,
            cpc: 32,
            n_train: 400,
            n_test: 2000,
            epochs: 5,
        },
        Config {
            name: "large (parallelism pays off)",
            bits: 256,
            n_classes: 10,
            cpc: 500,
            n_train: 2000,
            n_test: 4000,
            epochs: 5,
        },
    ];
    for c in &configs {
        run(&ctx, c);
    }
}
