//! Head-to-head training/inference throughput for all four modes:
//!   CPU exact · CPU data-parallel · GPU exact · GPU data-parallel
//! plus CPU-vs-GPU inference. Final test accuracy is shown per training variant
//! so the approximate (data-parallel) modes are judged on speed AND quality.
//!
//!   cargo run --release --features gpu          --example gpu_vs_cpu_bench
//!   cargo run --release --features gpu,parallel --example gpu_vs_cpu_bench
//!
//! CPU data-parallel only differs from CPU exact when built with `parallel`
//! (multi-threaded shards). The GPU uses whatever adapter wgpu selects — on a
//! real GPU that's the GPU; in a CI container with only the mesa llvmpipe
//! software driver it is still the CPU, so treat those numbers as a functional
//! check, not a hardware speedup. Exact GPU training is bitwise-identical to CPU;
//! data-parallel (both CPU and GPU) is approximate.

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

/// One trained-model result: median epoch time + final test accuracy.
struct TrainResult {
    per_epoch: Duration,
    acc: f64,
}

fn run(ctx: &Arc<GpuContext>, c: &Config) {
    let (train, ytr, enc) = make(c.n_train, c.bits, c.n_classes, 1);
    let (test, yte, _) = make(c.n_test, c.bits, c.n_classes, 2);
    let build = || {
        TsetlinMachine::with_config(c.n_classes, enc.n_features(), c.cpc, 50, 5.0, 8, true, 42)
            .max_included_literals(32)
    };

    println!(
        "\n=== {} : {} classes · {} features · {} clauses/class · {} train / {} test ===",
        c.name, c.n_classes, c.bits, c.cpc, c.n_train, c.n_test
    );

    // Time `c.epochs` training epochs (median), then report test accuracy. The
    // first epoch of each variant absorbs warmup/allocation; the median ignores it.
    let train_cpu = |dp: bool| -> TrainResult {
        let mut m = if dp {
            build().data_parallel(true)
        } else {
            build()
        };
        let mut times = Vec::new();
        for _ in 0..c.epochs {
            let t = Instant::now();
            m.fit_epoch(&train, &ytr);
            times.push(t.elapsed());
        }
        TrainResult {
            per_epoch: median(times),
            acc: m.accuracy(&test, &yte),
        }
    };
    let train_gpu = |dp: bool| -> TrainResult {
        let base = if dp {
            build().data_parallel(true)
        } else {
            build()
        };
        let mut m = base.to_gpu(ctx).expect("to_gpu");
        let mut times = Vec::new();
        for _ in 0..c.epochs {
            let t = Instant::now();
            m.fit_epoch(&train, &ytr);
            times.push(t.elapsed());
        }
        TrainResult {
            per_epoch: median(times),
            acc: m.accuracy(&test, &yte),
        }
    };

    // Data-parallel only differs from exact on the CPU when built with `parallel`.
    let cpu_dp_label = if cfg!(feature = "parallel") {
        "CPU data-parallel"
    } else {
        "CPU data-parallel*"
    };

    let r_cpu_exact = train_cpu(false);
    let r_cpu_dp = train_cpu(true);
    let r_gpu_exact = train_gpu(false);
    let r_gpu_dp = train_gpu(true);

    // Inference: time CPU vs GPU predict_batch on a trained model.
    let mut cpu_inf_model = build();
    for _ in 0..c.epochs {
        cpu_inf_model.fit_epoch(&train, &ytr);
    }
    let mut gpu_inf_model = build().to_gpu(ctx).expect("to_gpu");
    for _ in 0..c.epochs {
        gpu_inf_model.fit_epoch(&train, &ytr);
    }
    let time_infer = |mut f: Box<dyn FnMut()>| {
        let t = Instant::now();
        for _ in 0..5 {
            f();
        }
        t.elapsed() / 5
    };
    let cpu_infer = time_infer(Box::new(|| {
        std::hint::black_box(cpu_inf_model.predict_batch(&test));
    }));
    let gpu_infer = time_infer(Box::new(|| {
        std::hint::black_box(gpu_inf_model.predict_batch(&test));
    }));

    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let sps = |d: Duration, n: usize| n as f64 / d.as_secs_f64();
    let base = r_cpu_exact.per_epoch.as_secs_f64();
    let speedup = |d: Duration| base / d.as_secs_f64();

    // ---- training: all four variants ----
    println!(
        "\nTRAINING ({} epochs, {} train samples):",
        c.epochs, c.n_train
    );
    println!(
        "{:<20} {:>11} {:>12} {:>10} {:>7}",
        "variant", "ms/epoch", "samples/s", "vs CPU-ex", "acc"
    );
    let row = |label: &str, r: &TrainResult| {
        println!(
            "{:<20} {:>11.2} {:>12.0} {:>9.2}x {:>7.3}",
            label,
            ms(r.per_epoch),
            sps(r.per_epoch, c.n_train),
            speedup(r.per_epoch),
            r.acc
        );
    };
    row("CPU exact", &r_cpu_exact);
    row(cpu_dp_label, &r_cpu_dp);
    row("GPU exact", &r_gpu_exact);
    row("GPU data-parallel", &r_gpu_dp);
    if !cfg!(feature = "parallel") {
        println!("  * CPU data-parallel needs --features parallel to differ from exact.");
    }

    // ---- inference: CPU vs GPU ----
    println!("\nINFERENCE ({} test samples):", c.n_test);
    println!(
        "{:<20} {:>11} {:>12} {:>10}",
        "variant", "ms/batch", "samples/s", "vs CPU"
    );
    println!(
        "{:<20} {:>11.2} {:>12.0} {:>9.2}x",
        "CPU",
        ms(cpu_infer),
        sps(cpu_infer, c.n_test),
        1.0
    );
    println!(
        "{:<20} {:>11.2} {:>12.0} {:>9.2}x",
        "GPU",
        ms(gpu_infer),
        sps(gpu_infer, c.n_test),
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
