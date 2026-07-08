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

/// Serialize the full model state (ta, include, weights, all RNG streams) to
/// bytes — a single value capturing everything that must match bit-for-bit.
#[cfg(feature = "serde")]
fn state_bytes(tm: &TsetlinMachine) -> Vec<u8> {
    use tmu_rs::SaveLoad;
    let mut buf = Vec::new();
    tm.write_to(&mut buf).unwrap();
    buf
}

/// A configurable model builder so parity tests can sweep options.
struct Cfg {
    bits: usize,
    n_classes: usize,
    cpc: usize,
    threshold: i32,
    s: f64,
    state_bits: u8,
    boost: bool,
    max_inc: usize,
    literal_drop_p: f64,
    class_weights: Option<Vec<f64>>,
    seed: u64,
}

impl Cfg {
    fn build(&self) -> (TsetlinMachine, Encoder) {
        let enc = Encoder::for_binary(self.bits);
        let mut tm = TsetlinMachine::with_config(
            self.n_classes,
            enc.n_features(),
            self.cpc,
            self.threshold,
            self.s,
            self.state_bits,
            self.boost,
            self.seed,
        )
        .max_included_literals(self.max_inc)
        .literal_drop_p(self.literal_drop_p);
        if let Some(cw) = &self.class_weights {
            tm = tm.class_weights(cw.clone());
        }
        (tm, enc)
    }
}

#[cfg(feature = "serde")]
fn assert_train_parity(cfg: Cfg, epochs: usize) {
    let Some(ctx) = ctx() else { return };

    let (xtr, ytr) = make_multiclass(600, cfg.bits, cfg.n_classes, cfg.seed);
    let (cpu_tm, enc) = cfg.build();
    let (gpu_seed_tm, _) = cfg.build();
    let train = encode(&enc, &xtr);

    // CPU reference.
    let mut cpu_tm = cpu_tm;
    for _ in 0..epochs {
        cpu_tm.fit_epoch(&train, &ytr);
    }

    // GPU: train then bring the state back to the host.
    let mut gpu = gpu_seed_tm.to_gpu(&ctx).expect("to_gpu");
    for _ in 0..epochs {
        gpu.fit_epoch(&train, &ytr);
    }
    let gpu_tm = gpu.into_cpu();

    assert_eq!(
        state_bytes(&cpu_tm),
        state_bytes(&gpu_tm),
        "GPU-trained state differs from CPU-trained state (bitwise)"
    );
}

#[cfg(feature = "serde")]
#[test]
fn train_parity_default() {
    assert_train_parity(
        Cfg {
            bits: 12,
            n_classes: 2,
            cpc: 32,
            threshold: 15,
            s: 3.9,
            state_bits: 8,
            boost: true,
            max_inc: usize::MAX,
            literal_drop_p: 0.0,
            class_weights: None,
            seed: 42,
        },
        5,
    );
}

#[cfg(feature = "serde")]
#[test]
fn train_parity_no_boost_low_statebits() {
    assert_train_parity(
        Cfg {
            bits: 20,
            n_classes: 3,
            cpc: 24,
            threshold: 12,
            s: 6.0,
            state_bits: 4,
            boost: false,
            max_inc: usize::MAX,
            literal_drop_p: 0.0,
            class_weights: None,
            seed: 7,
        },
        5,
    );
}

#[cfg(feature = "serde")]
#[test]
fn train_parity_max_included_literals() {
    assert_train_parity(
        Cfg {
            bits: 16,
            n_classes: 2,
            cpc: 40,
            threshold: 15,
            s: 3.9,
            state_bits: 8,
            boost: true,
            max_inc: 8,
            literal_drop_p: 0.0,
            class_weights: None,
            seed: 99,
        },
        6,
    );
}

#[cfg(feature = "serde")]
#[test]
fn train_parity_literal_dropout() {
    assert_train_parity(
        Cfg {
            bits: 100, // multiword + padding
            n_classes: 4,
            cpc: 24,
            threshold: 20,
            s: 5.0,
            state_bits: 8,
            boost: true,
            max_inc: usize::MAX,
            literal_drop_p: 0.3,
            class_weights: None,
            seed: 123,
        },
        5,
    );
}

#[cfg(feature = "serde")]
#[test]
fn train_parity_class_weights() {
    assert_train_parity(
        Cfg {
            bits: 14,
            n_classes: 3,
            cpc: 32,
            threshold: 15,
            s: 3.9,
            state_bits: 8,
            boost: true,
            max_inc: usize::MAX,
            literal_drop_p: 0.0,
            class_weights: Some(vec![1.0, 1.5, 0.7]),
            seed: 321,
        },
        5,
    );
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

#[test]
fn gpu_train_converges_on_noisy_xor() {
    let Some(ctx) = ctx() else { return };
    let (xtr, ytr) = make_xor(2000, 12, 0.1, 1);
    let (xte, yte) = make_xor(2000, 12, 0.0, 2);
    let encoder = Encoder::for_binary(12);
    let train = encode(&encoder, &xtr);
    let test = encode(&encoder, &xte);

    let tm = TsetlinMachine::with_config(2, encoder.n_features(), 40, 15, 3.9, 8, true, 42)
        .max_included_literals(32);
    let mut gpu = tm.to_gpu(&ctx).expect("to_gpu");
    for _ in 0..30 {
        gpu.fit_epoch(&train, &ytr);
    }
    let acc = gpu.accuracy(&test, &yte);
    assert!(acc > 0.95, "GPU-trained noisy-XOR accuracy too low: {acc}");
}

#[cfg(feature = "serde")]
#[test]
fn gpu_train_save_load_cpu_roundtrip() {
    use tmu_rs::SaveLoad;
    let Some(ctx) = ctx() else { return };
    let (xtr, ytr) = make_xor(1500, 12, 0.05, 1);
    let (xte, yte) = make_xor(1500, 12, 0.0, 2);
    let encoder = Encoder::for_binary(12);
    let train = encode(&encoder, &xtr);
    let test = encode(&encoder, &xte);

    let tm = TsetlinMachine::with_config(2, encoder.n_features(), 40, 15, 3.9, 8, true, 42)
        .max_included_literals(32);
    let mut gpu = tm.to_gpu(&ctx).expect("to_gpu");
    for _ in 0..12 {
        gpu.fit_epoch(&train, &ytr);
    }
    // GPU predictions before syncing to the host.
    let gpu_pred = gpu.predict_batch(&test);

    // Save the synced host model, reload it, and run CPU inference.
    let host = gpu.into_cpu();
    let mut buf = Vec::new();
    host.write_to(&mut buf).unwrap();
    let loaded = TsetlinMachine::read_from(&mut buf.as_slice()).unwrap();
    let cpu_pred = loaded.predict_batch(&test);

    assert_eq!(gpu_pred, cpu_pred, "GPU vs reloaded-CPU predictions differ");

    // The loaded model should keep training bit-identically to a pure-CPU run.
    let mut pure_cpu =
        TsetlinMachine::with_config(2, encoder.n_features(), 40, 15, 3.9, 8, true, 42)
            .max_included_literals(32);
    for _ in 0..12 {
        pure_cpu.fit_epoch(&train, &ytr);
    }
    let mut a = loaded;
    a.fit_epoch(&train, &ytr);
    pure_cpu.fit_epoch(&train, &ytr);
    let mut ba = Vec::new();
    let mut bb = Vec::new();
    a.write_to(&mut ba).unwrap();
    pure_cpu.write_to(&mut bb).unwrap();
    assert_eq!(ba, bb, "resumed CPU training diverged from pure-CPU");
    let _ = yte;
}

#[cfg(feature = "serde")]
#[test]
fn interleaved_cpu_gpu_cpu_matches_pure_cpu() {
    let Some(ctx) = ctx() else { return };
    let (xtr, ytr) = make_multiclass(700, 16, 3, 5);
    let encoder = Encoder::for_binary(16);
    let train = encode(&encoder, &xtr);

    // Pure CPU: 6 epochs.
    let mut cpu = TsetlinMachine::with_config(3, encoder.n_features(), 24, 15, 3.9, 8, true, 9);
    for _ in 0..6 {
        cpu.fit_epoch(&train, &ytr);
    }

    // Interleaved: 2 CPU, 2 GPU, 2 CPU.
    let mut m = TsetlinMachine::with_config(3, encoder.n_features(), 24, 15, 3.9, 8, true, 9);
    m.fit_epoch(&train, &ytr);
    m.fit_epoch(&train, &ytr);
    let mut g = m.to_gpu(&ctx).expect("to_gpu");
    g.fit_epoch(&train, &ytr);
    g.fit_epoch(&train, &ytr);
    let mut m = g.into_cpu();
    m.fit_epoch(&train, &ytr);
    m.fit_epoch(&train, &ytr);

    assert_eq!(
        state_bytes(&cpu),
        state_bytes(&m),
        "interleaved CPU/GPU/CPU training diverged from pure CPU"
    );
}

#[test]
fn to_gpu_rejects_unsupported_options() {
    let Some(ctx) = ctx() else { return };
    let encoder = Encoder::for_binary(8);

    let t3 = TsetlinMachine::with_config(2, encoder.n_features(), 8, 15, 3.9, 8, true, 1)
        .type_iii_feedback(200.0);
    match t3.to_gpu(&ctx) {
        Err(GpuError::Unsupported(what)) => assert_eq!(what, "type_iii_feedback"),
        Err(e) => panic!("expected Unsupported(type_iii_feedback), got {e:?}"),
        Ok(_) => panic!("expected Unsupported(type_iii_feedback), got Ok"),
    }

    let cd = TsetlinMachine::with_config(2, encoder.n_features(), 8, 15, 3.9, 8, true, 1)
        .clause_drop_p(0.3);
    match cd.to_gpu(&ctx) {
        Err(GpuError::Unsupported(what)) => assert_eq!(what, "clause_drop_p"),
        Err(e) => panic!("expected Unsupported(clause_drop_p), got {e:?}"),
        Ok(_) => panic!("expected Unsupported(clause_drop_p), got Ok"),
    }

    // The original models are untouched and still usable on the CPU.
    let (xte, _) = make_xor(50, 8, 0.0, 1);
    let test = encode(&encoder, &xte);
    let _ = t3.predict_batch(&test);
}
