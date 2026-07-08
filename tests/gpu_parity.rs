//! GPU-vs-CPU parity tests for the vanilla TsetlinMachine.
//!
//! Every test skips cleanly (returns) when no GPU adapter is available, so the
//! suite passes in environments without a GPU or software Vulkan driver. To
//! actually exercise the GPU install a Vulkan driver (e.g. `mesa-vulkan-drivers`
//! provides the llvmpipe software driver) and run:
//!
//! ```text
//! cargo test --features gpu,serde --test gpu_parity
//! ```

#![cfg(feature = "gpu")]

use std::sync::Arc;

use tmu_rs::{Encoder, GpuContext, GpuError, Rng, TsetlinMachine};

fn ctx() -> Option<Arc<GpuContext>> {
    match GpuContext::new() {
        Ok(c) => {
            eprintln!("gpu parity tests on adapter: {}", c.adapter_info().name);
            Some(Arc::new(c))
        }
        Err(GpuError::NoAdapter) => {
            eprintln!("no GPU adapter available; skipping GPU parity test");
            None
        }
        Err(e) => panic!("adapter present but device creation failed: {e}"),
    }
}

/// Noisy XOR on `bits` binary features (only the first two decide the label).
fn make_xor(n: usize, bits: usize, noise: f64, seed: u64) -> (Vec<Vec<u8>>, Vec<usize>) {
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

/// Multiclass toy problem: label = (x[0]*2 + x[1]) mod n_classes.
fn make_multiclass(
    n: usize,
    bits: usize,
    n_classes: usize,
    seed: u64,
) -> (Vec<Vec<u8>>, Vec<usize>) {
    let mut rng = Rng::new(seed);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let x: Vec<u8> = (0..bits).map(|_| (rng.next_u64() & 1) as u8).collect();
        let y = (x[0] as usize * 2 + x[1] as usize) % n_classes;
        xs.push(x);
        ys.push(y);
    }
    (xs, ys)
}

fn encode(encoder: &Encoder, xs: &[Vec<u8>]) -> tmu_rs::EncodedBatch {
    let refs: Vec<&[u8]> = xs.iter().map(|v| v.as_slice()).collect();
    encoder.encode_batch(&refs)
}

/// CPU-trained model, then assert GPU inference matches CPU inference exactly.
fn assert_predict_parity(bits: usize, n_classes: usize, cpc: usize, epochs: usize, seed: u64) {
    let Some(ctx) = ctx() else { return };

    let (xtr, ytr) = make_multiclass(800, bits, n_classes, seed);
    let (xte, _) = make_multiclass(1500, bits, n_classes, seed + 1);
    let encoder = Encoder::for_binary(bits);
    let train = encode(&encoder, &xtr);
    let test = encode(&encoder, &xte);

    let mut tm =
        TsetlinMachine::with_config(n_classes, encoder.n_features(), cpc, 20, 5.0, 8, true, seed)
            .max_included_literals(32);
    for _ in 0..epochs {
        tm.fit_epoch(&train, &ytr);
    }

    let cpu_pred = tm.predict_batch(&test);
    let mut gpu = tm.to_gpu(&ctx).expect("to_gpu");
    let gpu_pred = gpu.predict_batch(&test);

    assert_eq!(cpu_pred.len(), gpu_pred.len());
    let mism = cpu_pred
        .iter()
        .zip(&gpu_pred)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        mism, 0,
        "bits={bits} n_classes={n_classes}: {mism} prediction mismatches"
    );
}

#[test]
fn predict_parity_binary_2class() {
    assert_predict_parity(12, 2, 40, 10, 7);
}

#[test]
fn predict_parity_multiword_padding() {
    // 100 features -> 200 literals -> words not a multiple of 64 (padding bits).
    assert_predict_parity(100, 2, 40, 8, 11);
}

#[test]
fn predict_parity_4class() {
    assert_predict_parity(13, 4, 32, 8, 21);
}

#[test]
fn predict_parity_untrained_empty_clauses() {
    // A fresh (untrained) model exercises the empty-clause fire_predict path.
    let Some(ctx) = ctx() else { return };
    let (xte, _) = make_xor(500, 16, 0.0, 3);
    let encoder = Encoder::for_binary(16);
    let test = encode(&encoder, &xte);

    let tm = TsetlinMachine::with_config(2, encoder.n_features(), 20, 15, 3.9, 8, true, 5);
    let cpu = tm.predict_batch(&test);
    let mut gpu = tm.to_gpu(&ctx).expect("to_gpu");
    let g = gpu.predict_batch(&test);
    assert_eq!(cpu, g);
}
