//! GPU training + inference with CUDA.
//!
//! ```sh
//! cargo run --release --features cuda --example cuda            # real GPU
//! cargo run --release --features cuda-emu --example cuda -- --emu  # CPU kernel emulator
//! ```
//!
//! Trains the same Tsetlin Machine on the CPU and on the GPU from the same seed,
//! reports time per epoch for both, and checks that the GPU-trained model is
//! bit-identical to the CPU-trained one.

use std::time::Instant;

use tmu_rs::cuda::CudaDevice;
use tmu_rs::{Encoder, Rng, TsetlinMachine};

fn main() {
    let emu = std::env::args().any(|a| a == "--emu");
    let dev = if emu {
        #[cfg(feature = "cuda-emu")]
        {
            CudaDevice::emulator()
        }
        #[cfg(not(feature = "cuda-emu"))]
        {
            eprintln!("--emu needs `--features cuda-emu`");
            return;
        }
    } else {
        match CudaDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("No usable CUDA GPU: {e}");
                eprintln!(
                    "(try `--features cuda-emu --example cuda -- --emu` to run the kernels on the CPU)"
                );
                return;
            }
        }
    };
    println!("device: {}", dev.name());

    // Noisy XOR over the first two features, the rest is noise. The emulator is
    // slow, so it gets a much smaller problem.
    let (n_features, n_train, clauses, epochs) = if emu {
        (64, 400, 40, 2)
    } else {
        (1000, 5000, 2000, 3)
    };
    let mut r = Rng::new(7);
    let mut xs = Vec::with_capacity(n_train);
    let mut ys = Vec::with_capacity(n_train);
    for _ in 0..n_train {
        let x: Vec<u8> = (0..n_features).map(|_| (r.next_u64() & 1) as u8).collect();
        let mut y = (x[0] ^ x[1]) as usize;
        if r.next_f64() < 0.1 {
            y ^= 1;
        }
        xs.push(x);
        ys.push(y);
    }
    let rows: Vec<&[u8]> = xs.iter().map(|x| x.as_slice()).collect();
    let batch = Encoder::for_binary(n_features).encode_batch(&rows);

    let make = || {
        TsetlinMachine::with_config(2, n_features, clauses, clauses as i32 / 2, 3.9, 8, true, 42)
    };
    let mut cpu = make();
    let mut gpu = make().to_cuda(&dev).expect("upload model");

    for epoch in 1..=epochs {
        let t = Instant::now();
        cpu.fit_epoch(&batch, &ys);
        let cpu_s = t.elapsed().as_secs_f64();

        let t = Instant::now();
        gpu.fit_epoch(&batch, &ys).expect("GPU training");
        dev.synchronize().expect("sync");
        let gpu_s = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let acc = gpu.accuracy(&batch, &ys).expect("GPU inference");
        let inf_s = t.elapsed().as_secs_f64();
        println!(
            "epoch {epoch}: CPU {cpu_s:.2}s  GPU {gpu_s:.2}s  ({:.1}x)  train acc {acc:.3}  GPU inference {inf_s:.3}s",
            cpu_s / gpu_s
        );
    }

    let back = gpu.to_cpu().expect("download model");
    let same = back.predict_batch(&batch) == cpu.predict_batch(&batch)
        && (0..clauses).all(|j| back.clause_weight(0, j) == cpu.clause_weight(0, j))
        && (0..clauses).all(|j| back.clause_rule(1, j) == cpu.clause_rule(1, j));
    println!("GPU model identical to CPU model: {same}");
}
