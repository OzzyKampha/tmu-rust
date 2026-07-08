//! GPU demo: train noisy XOR on the GPU, evaluate on both the GPU and the CPU,
//! and confirm the trained model round-trips through save/load.
//!
//! Requires the `gpu` feature and a Vulkan/Metal/DX12 adapter (the mesa llvmpipe
//! software driver works too). Run with:
//!
//!   cargo run --release --features gpu,serde --example gpu_xor
//!
//! A model trained on the GPU is bit-for-bit identical to one trained on the CPU
//! with the same seed, so it can be used interchangeably for CPU or GPU
//! inference.

use std::sync::Arc;

use tmu_rs::{Encoder, GpuContext, Rng, TsetlinMachine};

fn make(n: usize, bits: usize, noise: f64, seed: u64) -> (Vec<Vec<u8>>, Vec<usize>) {
    let mut rng = Rng::new(seed);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let x: Vec<u8> = (0..bits).map(|_| (rng.next_u64() & 1) as u8).collect();
        let mut y = (x[0] ^ x[1]) as usize;
        if rng.next_f64() < noise {
            y ^= 1;
        }
        xs.push(x);
        ys.push(y);
    }
    (xs, ys)
}

fn main() {
    let ctx = match GpuContext::new() {
        Ok(c) => {
            let info = c.adapter_info();
            println!(
                "GPU adapter: {} ({:?}, {:?})",
                info.name, info.device_type, info.backend
            );
            Arc::new(c)
        }
        Err(e) => {
            eprintln!(
                "No usable GPU adapter ({e}). Install a Vulkan driver (e.g. mesa-vulkan-drivers) to run this example."
            );
            return;
        }
    };

    let bits = 12;
    let (xtr, ytr) = make(2000, bits, 0.1, 1);
    let (xte, yte) = make(2000, bits, 0.0, 2);
    let encoder = Encoder::for_binary(bits);
    let refs_tr: Vec<&[u8]> = xtr.iter().map(|v| v.as_slice()).collect();
    let refs_te: Vec<&[u8]> = xte.iter().map(|v| v.as_slice()).collect();
    let train = encoder.encode_batch(&refs_tr);
    let test = encoder.encode_batch(&refs_te);

    let tm = TsetlinMachine::with_config(2, encoder.n_features(), 40, 15, 3.9, 8, true, 42)
        .max_included_literals(32);

    let mut gpu = tm.to_gpu(&ctx).expect("move model to GPU");
    println!("\nTraining noisy XOR on the GPU:");
    for epoch in 1..=20 {
        gpu.fit_epoch(&train, &ytr);
        if epoch % 5 == 0 {
            let acc = gpu.accuracy(&test, &yte);
            println!("  epoch {epoch:2}: GPU test accuracy = {:.3}", acc);
        }
    }

    // Bring the trained model back to the CPU and evaluate there too.
    let cpu = gpu.into_cpu();
    let correct = cpu
        .predict_batch(&test)
        .iter()
        .zip(&yte)
        .filter(|(p, y)| p == y)
        .count();
    println!(
        "\nSame model, CPU inference: accuracy = {:.3}",
        correct as f64 / yte.len() as f64
    );

    // Save / load round-trip (proves GPU-trained models serialize normally).
    #[cfg(feature = "serde")]
    {
        use tmu_rs::SaveLoad;
        let path = std::env::temp_dir().join("tmu_gpu_xor.bin");
        cpu.save(&path).expect("save");
        let loaded = TsetlinMachine::load(&path).expect("load");
        let same = loaded.predict_batch(&test) == cpu.predict_batch(&test);
        println!(
            "Saved to {} and reloaded — predictions identical: {same}",
            path.display()
        );
        let _ = std::fs::remove_file(&path);
    }
}
