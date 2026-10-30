//! GPU-vs-CPU equivalence tests: every GPU model must end up bit-identical to
//! its CPU counterpart after the same training. Run on the CPU kernel emulator
//! (`cargo test --features cuda-emu`); on a machine with a GPU, set
//! `TMU_TEST_GPU=1` to run the same checks on CUDA device 0 instead.

use super::CudaDevice;
use crate::Encoder;
use crate::encoder::EncodedBatch;
use crate::rng::Rng;

fn device() -> CudaDevice {
    if std::env::var("TMU_TEST_GPU").is_ok() {
        return CudaDevice::new(0).expect("TMU_TEST_GPU set but no usable CUDA device");
    }
    CudaDevice::emulator()
}

/// Serial emulator: fastest, for the larger runs.
fn device_serial() -> CudaDevice {
    if std::env::var("TMU_TEST_GPU").is_err() {
        return CudaDevice::emulator_with_block(1);
    }
    device()
}

/// Noisy XOR on the first two features, other features random.
fn xor_data(n: usize, n_features: usize, noise: f64, seed: u64) -> (Vec<Vec<u8>>, Vec<usize>) {
    let mut r = Rng::new(seed);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for _ in 0..n {
        let x: Vec<u8> = (0..n_features).map(|_| (r.next_u64() & 1) as u8).collect();
        let mut y = (x[0] ^ x[1]) as usize;
        if r.next_f64() < noise {
            y ^= 1;
        }
        xs.push(x);
        ys.push(y);
    }
    (xs, ys)
}

fn encode(xs: &[Vec<u8>], n_features: usize) -> EncodedBatch {
    let rows: Vec<&[u8]> = xs.iter().map(|r| r.as_slice()).collect();
    Encoder::for_binary(n_features).encode_batch(&rows)
}

fn states(r: &[Rng]) -> Vec<u64> {
    r.iter().map(|r| r.0).collect()
}

mod vanilla {
    use super::*;
    use crate::TsetlinMachine;

    fn check(make: impl Fn() -> TsetlinMachine, dev: &CudaDevice, epochs: usize) {
        let nf = 12;
        let (xs, ys) = xor_data(160, nf, 0.1, 7);
        let batch = encode(&xs, nf);
        let mut cpu = make();
        let mut gpu = make().to_cuda(dev).unwrap();
        for _ in 0..epochs {
            cpu.fit_epoch(&batch, &ys);
            gpu.fit_epoch(&batch, &ys).unwrap();
        }
        let back = gpu.to_cpu().unwrap();
        assert_eq!(back.ta, cpu.ta, "TA states differ");
        assert_eq!(back.include, cpu.include);
        assert_eq!(back.weights, cpu.weights);
        assert_eq!(states(&back.rngs), states(&cpu.rngs));
        assert_eq!(states(&back.class_rngs), states(&cpu.class_rngs));
        assert_eq!(back.rng.0, cpu.rng.0);
        assert_eq!(back.literal_rng.0, cpu.literal_rng.0);
        assert_eq!(back.ind, cpu.ind);
        assert_eq!(back.cat, cpu.cat);
        assert_eq!(
            gpu.predict_batch(&batch).unwrap(),
            cpu.predict_batch(&batch)
        );
        for i in [0usize, 5, 17] {
            let s = crate::encoder::EncodedSample(
                batch.data[i * batch.words..(i + 1) * batch.words].to_vec(),
            );
            let mut want = vec![0i32; cpu.n_classes()];
            cpu.scores(&s, &mut want);
            assert_eq!(gpu.scores(&s).unwrap(), want);
        }
        assert_eq!(
            gpu.accuracy(&batch, &ys).unwrap(),
            cpu.accuracy(&batch, &ys)
        );
    }

    #[test]
    fn matches_cpu_default() {
        check(
            || TsetlinMachine::with_config(2, 12, 10, 8, 3.9, 8, true, 1),
            &device(),
            3,
        );
    }

    #[test]
    fn matches_cpu_all_options() {
        check(
            || {
                TsetlinMachine::with_config(3, 12, 8, 6, 2.5, 6, false, 9)
                    .clause_drop_p(0.25)
                    .literal_drop_p(0.2)
                    .max_included_literals(3)
                    .class_weights(vec![1.0, 2.0, 0.5])
            },
            &device_serial(),
            3,
        );
    }

    #[test]
    fn matches_cpu_type_iii() {
        check(
            || TsetlinMachine::with_config(2, 12, 6, 8, 3.9, 8, true, 3).type_iii_feedback(30.0),
            &device(),
            2,
        );
    }
}

macro_rules! assert_dense_eq {
    ($gpu:expr, $cpu:expr) => {{
        let (g, c) = (&$gpu, &$cpu);
        assert_eq!(g.ta, c.ta, "TA states differ");
        assert_eq!(g.include, c.include, "include bitsets differ");
        assert_eq!(g.weights, c.weights, "weights differ");
        assert_eq!(
            states(&g.rngs),
            states(&c.rngs),
            "clause RNG streams differ"
        );
        assert_eq!(g.ind, c.ind, "Type III indicator states differ");
        assert_eq!(g.cat, c.cat, "Type III clause-and-target bits differ");
        assert_eq!(g.rng.0, c.rng.0, "master RNG differs");
        assert_eq!(g.literal_rng.0, c.literal_rng.0, "literal RNG differs");
    }};
}

fn sample(batch: &EncodedBatch, i: usize) -> crate::encoder::EncodedSample {
    crate::encoder::EncodedSample(batch.data[i * batch.words..(i + 1) * batch.words].to_vec())
}

mod vanilla_more {
    use super::*;
    use crate::TsetlinMachine;

    #[test]
    fn low_state_bits_hit_absorbing_states() {
        // 3-bit TAs saturate quickly, exercising the max-state guards.
        let nf = 10;
        let (xs, ys) = xor_data(120, nf, 0.05, 11);
        let batch = encode(&xs, nf);
        let make = || TsetlinMachine::with_config(2, nf, 8, 6, 3.0, 3, false, 5);
        let mut cpu = make();
        let mut gpu = make().to_cuda(&device_serial()).unwrap();
        for _ in 0..6 {
            cpu.fit_epoch(&batch, &ys);
            gpu.fit_epoch(&batch, &ys).unwrap();
        }
        let back = gpu.to_cpu().unwrap();
        assert!(
            cpu.ta.contains(&cpu.max_state),
            "test should reach max state"
        );
        assert_dense_eq!(back, cpu);
        assert_eq!(states(&back.class_rngs), states(&cpu.class_rngs));
    }

    #[test]
    fn fit_one_matches_and_learns() {
        let nf = 12;
        let (xs, ys) = xor_data(200, nf, 0.0, 21);
        let batch = encode(&xs, nf);
        let make = || TsetlinMachine::with_config(2, nf, 10, 8, 3.9, 8, true, 2);
        let mut cpu = make();
        let mut gpu = make().to_cuda(&device_serial()).unwrap();
        for (i, &y) in ys.iter().enumerate() {
            cpu.fit_one(&sample(&batch, i), y);
            gpu.fit_one(&sample(&batch, i), y).unwrap();
        }
        for _ in 0..8 {
            cpu.fit_epoch(&batch, &ys);
            gpu.fit_epoch(&batch, &ys).unwrap();
        }
        assert_dense_eq!(gpu.to_cpu().unwrap(), cpu);
        assert!(
            gpu.accuracy(&batch, &ys).unwrap() > 0.9,
            "GPU model should learn XOR"
        );
    }

    #[test]
    fn resumes_from_trained_cpu_model() {
        let nf = 12;
        let (xs, ys) = xor_data(100, nf, 0.1, 3);
        let batch = encode(&xs, nf);
        let mut cpu = TsetlinMachine::with_config(2, nf, 6, 8, 3.9, 8, true, 4);
        cpu.fit_epoch(&batch, &ys);
        let mut gpu = cpu.to_cuda(&device_serial()).unwrap();
        cpu.fit_epoch(&batch, &ys);
        gpu.fit_epoch(&batch, &ys).unwrap();
        assert_dense_eq!(gpu.to_cpu().unwrap(), cpu);
    }
}

mod coalesced {
    use super::*;
    use crate::CoalescedTsetlinMachine;

    fn check(make: impl Fn() -> CoalescedTsetlinMachine, n_classes: usize, dev: &CudaDevice) {
        let nf = 12;
        let (xs, mut ys) = xor_data(150, nf, 0.1, 13);
        if n_classes > 2 {
            for (y, x) in ys.iter_mut().zip(&xs) {
                *y += 2 * x[2] as usize % n_classes.max(1);
                *y %= n_classes;
            }
        }
        let batch = encode(&xs, nf);
        let mut cpu = make();
        let mut gpu = make().to_cuda(dev).unwrap();
        for _ in 0..3 {
            cpu.fit_epoch(&batch, &ys);
            gpu.fit_epoch(&batch, &ys).unwrap();
        }
        let back = gpu.to_cpu().unwrap();
        assert_dense_eq!(back, cpu);
        assert_eq!(states(&back.class_rngs), states(&cpu.class_rngs));
        assert_eq!(
            gpu.predict_batch(&batch).unwrap(),
            cpu.predict_batch(&batch)
        );
        for i in [0usize, 9] {
            let mut want = vec![0i32; n_classes];
            cpu.scores(&sample(&batch, i), &mut want);
            assert_eq!(gpu.scores(&sample(&batch, i)).unwrap(), want);
        }
    }

    #[test]
    fn matches_cpu_default() {
        check(
            || CoalescedTsetlinMachine::with_config(2, 12, 16, 10, 3.9, 8, true, 1),
            2,
            &device(),
        );
    }

    #[test]
    fn matches_cpu_all_options() {
        check(
            || {
                CoalescedTsetlinMachine::with_config(4, 12, 12, 8, 2.5, 5, false, 8)
                    .clause_drop_p(0.3)
                    .literal_drop_p(0.15)
                    .max_included_literals(4)
                    .focused_negative_sampling(true)
            },
            4,
            &device_serial(),
        );
    }

    #[test]
    fn matches_cpu_type_iii() {
        check(
            || {
                CoalescedTsetlinMachine::with_config(3, 12, 10, 8, 3.9, 8, true, 6)
                    .type_iii_feedback(20.0)
            },
            3,
            &device(),
        );
    }
}

mod convolutional {
    use super::*;
    use crate::ConvolutionalTsetlinMachine;

    fn check(make: impl Fn() -> ConvolutionalTsetlinMachine, nf: usize, dev: &CudaDevice) {
        let (xs, ys) = xor_data(100, nf, 0.05, 17);
        let rows: Vec<&[u8]> = xs.iter().map(|r| r.as_slice()).collect();
        let mut cpu = make();
        let mut gpu = make().to_cuda(dev).unwrap();
        for _ in 0..3 {
            cpu.fit_epoch(&rows, &ys);
            gpu.fit_epoch(&rows, &ys).unwrap();
        }
        let back = gpu.to_cpu().unwrap();
        assert_eq!(back.ta, cpu.ta);
        assert_eq!(back.include, cpu.include);
        assert_eq!(back.weights, cpu.weights);
        assert_eq!(states(&back.rngs), states(&cpu.rngs));
        assert_eq!(states(&back.class_rngs), states(&cpu.class_rngs));
        assert_eq!(back.ind, cpu.ind);
        assert_eq!(back.cat, cpu.cat);
        assert_eq!(back.rng.0, cpu.rng.0);
        assert_eq!(back.literal_rng.0, cpu.literal_rng.0);
        assert_eq!(gpu.predict_batch(&rows).unwrap(), cpu.predict_batch(&rows));
        let mut want = vec![0i32; cpu.n_classes()];
        cpu.scores(rows[3], &mut want);
        assert_eq!(gpu.scores(rows[3]).unwrap(), want);
    }

    #[test]
    fn matches_cpu_1d() {
        check(
            || ConvolutionalTsetlinMachine::with_config(2, 12, 4, 2, 10, 8, 3.9, 8, true, 1),
            12,
            &device(),
        );
    }

    #[test]
    fn matches_cpu_2d_all_options() {
        check(
            || {
                ConvolutionalTsetlinMachine::with_config_2d(
                    3, 4, 4, 2, 3, 1, 8, 6, 3.0, 4, false, 2,
                )
                .clause_drop_p(0.2)
                .literal_drop_p(0.1)
                .max_included_literals(3)
            },
            16,
            &device_serial(),
        );
    }

    #[test]
    fn matches_cpu_type_iii() {
        check(
            || {
                ConvolutionalTsetlinMachine::with_config(2, 12, 3, 1, 8, 8, 3.9, 8, true, 3)
                    .type_iii_feedback(25.0)
            },
            12,
            &device(),
        );
    }
}

mod regressor {
    use super::*;
    use crate::TMRegressor;

    fn data(n: usize, nf: usize, t: i32, seed: u64) -> (EncodedBatch, Vec<f64>) {
        let mut r = Rng::new(seed);
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for _ in 0..n {
            let x: Vec<u8> = (0..nf).map(|_| (r.next_u64() & 1) as u8).collect();
            let ones = x[..4].iter().filter(|&&b| b == 1).count() as f64;
            ys.push(ones * t as f64 / 4.0);
            xs.push(x);
        }
        (encode(&xs, nf), ys)
    }

    fn check(make: impl Fn() -> TMRegressor, dev: &CudaDevice) {
        let (batch, ys) = data(120, 10, 12, 5);
        let mut cpu = make();
        let mut gpu = make().to_cuda(dev).unwrap();
        for _ in 0..3 {
            cpu.fit_epoch(&batch, &ys);
            gpu.fit_epoch(&batch, &ys).unwrap();
        }
        assert_dense_eq!(gpu.to_cpu().unwrap(), cpu);
        assert_eq!(
            gpu.predict_batch(&batch).unwrap(),
            cpu.predict_batch(&batch)
        );
        assert_eq!(gpu.mae(&batch, &ys).unwrap(), cpu.mae(&batch, &ys));
        assert_eq!(gpu.rmse(&batch, &ys).unwrap(), cpu.rmse(&batch, &ys));
    }

    #[test]
    fn matches_cpu_default() {
        check(
            || TMRegressor::with_config(10, 16, 12, 3.9, 8, true, 1),
            &device(),
        );
    }

    #[test]
    fn matches_cpu_all_options() {
        check(
            || {
                TMRegressor::with_config(10, 12, 12, 2.5, 4, false, 7)
                    .clause_drop_p(0.25)
                    .literal_drop_p(0.2)
                    .max_included_literals(3)
            },
            &device_serial(),
        );
    }

    #[test]
    fn matches_cpu_type_iii() {
        check(
            || TMRegressor::with_config(10, 12, 12, 3.9, 8, true, 2).type_iii_feedback(15.0),
            &device(),
        );
    }
}

mod autoencoder {
    use super::*;
    use crate::TMAutoEncoder;

    fn check(make: impl Fn() -> TMAutoEncoder, dev: &CudaDevice) {
        let nf = 8;
        let (xs, _) = xor_data(60, nf, 0.0, 23);
        let batch = encode(&xs, nf);
        let mut cpu = make();
        let mut gpu = make().to_cuda(dev).unwrap();
        for _ in 0..2 {
            cpu.fit_epoch(&batch);
            gpu.fit_epoch(&batch).unwrap();
        }
        let back = gpu.to_cpu().unwrap();
        assert_dense_eq!(back, cpu);
        assert_eq!(states(&back.output_rngs), states(&cpu.output_rngs));
        assert_eq!(
            gpu.reconstruct_batch(&batch).unwrap(),
            cpu.reconstruct_batch(&batch)
        );
        assert_eq!(
            gpu.reconstruction_accuracy(&batch).unwrap(),
            cpu.reconstruction_accuracy(&batch)
        );
        let mut want = vec![0i32; nf];
        cpu.reconstruct_scores(&sample(&batch, 2), &mut want);
        assert_eq!(
            &gpu.reconstruct_scores_batch(&batch).unwrap()[2 * nf..3 * nf],
            &want[..]
        );
    }

    #[test]
    fn matches_cpu_default() {
        check(
            || TMAutoEncoder::with_config(8, 6, 8, 3.9, 8, true, 1),
            &device(),
        );
    }

    #[test]
    fn matches_cpu_all_options() {
        check(
            || {
                TMAutoEncoder::with_config(8, 6, 6, 2.5, 4, false, 4)
                    .clause_drop_p(0.2)
                    .literal_drop_p(0.2)
                    .max_included_literals(3)
            },
            &device_serial(),
        );
    }

    #[test]
    fn matches_cpu_type_iii() {
        check(
            || TMAutoEncoder::with_config(8, 4, 8, 3.9, 8, true, 9).type_iii_feedback(20.0),
            &device_serial(),
        );
    }
}

mod coalesced_autoencoder {
    use super::*;
    use crate::TMCoalescedAutoEncoder;

    fn check(make: impl Fn() -> TMCoalescedAutoEncoder, dev: &CudaDevice) {
        let nf = 8;
        let (xs, _) = xor_data(50, nf, 0.0, 29);
        let batch = encode(&xs, nf);
        let mut cpu = make();
        let mut gpu = make().to_cuda(dev).unwrap();
        for _ in 0..2 {
            cpu.fit_epoch(&batch);
            gpu.fit_epoch(&batch).unwrap();
        }
        let back = gpu.to_cpu().unwrap();
        assert_dense_eq!(back, cpu);
        assert_eq!(states(&back.output_rngs), states(&cpu.output_rngs));
        assert_eq!(back.clause_gate_rng.0, cpu.clause_gate_rng.0);
        assert_eq!(
            gpu.reconstruct_batch(&batch).unwrap(),
            cpu.reconstruct_batch(&batch)
        );
        let mut want = vec![0i32; nf];
        cpu.reconstruct_scores(&sample(&batch, 1), &mut want);
        assert_eq!(
            &gpu.reconstruct_scores_batch(&batch).unwrap()[nf..2 * nf],
            &want[..]
        );
    }

    #[test]
    fn matches_cpu_default() {
        check(
            || TMCoalescedAutoEncoder::with_config(8, 12, 8, 3.9, 8, true, 1),
            &device_serial(),
        );
    }

    #[test]
    fn matches_cpu_all_options() {
        check(
            || {
                TMCoalescedAutoEncoder::with_config(8, 10, 6, 2.5, 4, false, 5)
                    .clause_drop_p(0.25)
                    .literal_drop_p(0.2)
                    .max_included_literals(3)
            },
            &device_serial(),
        );
    }

    #[test]
    fn matches_cpu_type_iii() {
        check(
            || {
                TMCoalescedAutoEncoder::with_config(8, 8, 8, 3.9, 8, true, 3)
                    .type_iii_feedback(20.0)
            },
            &device(),
        );
    }
}

mod sparse {
    use super::*;
    use crate::TMSparseClassifier;

    fn check(
        make: impl Fn() -> TMSparseClassifier,
        n_classes: usize,
        epochs: usize,
        dev: &CudaDevice,
    ) {
        let nf = 12;
        let (xs, mut ys) = xor_data(150, nf, 0.1, 31);
        for (y, x) in ys.iter_mut().zip(&xs) {
            *y = (*y + x[3] as usize) % n_classes;
        }
        let batch = encode(&xs, nf);
        let mut cpu = make();
        let mut gpu = make().to_cuda(dev).unwrap();
        for _ in 0..epochs {
            cpu.fit_epoch(&batch, &ys);
            gpu.fit_epoch(&batch, &ys).unwrap();
        }
        let back = gpu.to_cpu().unwrap();
        for (a, b) in back.bank.clauses.iter().zip(&cpu.bank.clauses) {
            assert_eq!(a.included, b.included);
            assert_eq!(a.included_state, b.included_state);
            assert_eq!(a.excluded, b.excluded);
            assert_eq!(a.excluded_state, b.excluded_state);
            assert_eq!(a.unallocated, b.unallocated);
        }
        assert_eq!(back.weights, cpu.weights);
        assert_eq!(states(&back.rngs), states(&cpu.rngs));
        assert_eq!(states(&back.class_rngs), states(&cpu.class_rngs));
        assert_eq!(back.rng.0, cpu.rng.0);
        assert_eq!(back.literal_rng.0, cpu.literal_rng.0);
        assert_eq!(
            gpu.predict_batch(&batch).unwrap(),
            cpu.predict_batch(&batch)
        );
        let mut want = vec![0i32; n_classes];
        cpu.scores(&sample(&batch, 4), &mut want);
        assert_eq!(gpu.scores(&sample(&batch, 4)).unwrap(), want);
    }

    #[test]
    fn matches_cpu_default() {
        check(
            || TMSparseClassifier::with_config(2, 12, 10, 8, 3.9, 8, true, 1),
            2,
            3,
            &device(),
        );
    }

    #[test]
    fn matches_cpu_absorbing_and_options() {
        // 3-bit states absorb literals out of the excluded pool within a few epochs.
        let make = || {
            TMSparseClassifier::with_config(3, 12, 8, 6, 3.0, 3, false, 7)
                .clause_drop_p(0.2)
                .literal_drop_p(0.1)
                .max_included_literals(4)
                .class_weights(vec![1.0, 1.5, 0.75])
        };
        let mut probe = make();
        let (xs, ys) = xor_data(150, 12, 0.1, 31);
        let batch = encode(&xs, 12);
        for _ in 0..6 {
            probe.fit_epoch(&batch, &ys.iter().map(|y| y % 3).collect::<Vec<_>>());
        }
        assert!(
            probe.bank.count_absorbed_exclude() > 0,
            "test should absorb literals"
        );
        check(make, 3, 6, &device_serial());
    }
}

mod composite {
    use super::*;
    use crate::{TMCompositeClassifier, TsetlinMachine};

    #[test]
    fn matches_cpu() {
        let nf = 12;
        let (xs, ys) = xor_data(100, nf, 0.1, 37);
        let batch = encode(&xs, nf);
        let make = || {
            let mut c = TMCompositeClassifier::new();
            c.add(TsetlinMachine::with_config(2, nf, 6, 8, 3.9, 8, true, 1));
            c.add(TsetlinMachine::with_config(2, nf, 8, 10, 3.0, 8, true, 2));
            c
        };
        let mut cpu = make();
        let mut gpu = make().to_cuda(&device_serial()).unwrap();
        for _ in 0..2 {
            cpu.fit_epoch(&batch, &ys);
            gpu.fit_epoch(&batch, &ys).unwrap();
        }
        let back = gpu.to_cpu().unwrap();
        for (g, c) in back.classifiers().iter().zip(cpu.classifiers()) {
            assert_dense_eq!(g, c);
        }
        assert_eq!(
            gpu.predict_batch(&batch).unwrap(),
            cpu.predict_batch(&batch)
        );
        let mut want = vec![0i32; 2];
        cpu.scores(&sample(&batch, 7), &mut want);
        assert_eq!(gpu.scores(&sample(&batch, 7)).unwrap(), want);
        assert_eq!(
            gpu.accuracy(&batch, &ys).unwrap(),
            cpu.accuracy(&batch, &ys)
        );
    }

    #[test]
    fn empty_composite_is_rejected() {
        assert!(
            TMCompositeClassifier::new()
                .to_cuda(&device_serial())
                .is_err()
        );
    }
}

/// Inputs spanning several 64-bit words, so clause blocks run with multiple
/// cooperating threads (word striding, block reductions, Type III `block_min`).
mod wide_inputs {
    use super::*;
    use crate::{
        CoalescedTsetlinMachine, ConvolutionalTsetlinMachine, TMAutoEncoder,
        TMCoalescedAutoEncoder, TMRegressor, TsetlinMachine,
    };

    fn dev() -> CudaDevice {
        if std::env::var("TMU_TEST_GPU").is_err() {
            return CudaDevice::emulator_with_block(4);
        }
        device()
    }

    #[test]
    fn vanilla_type_iii_and_limit() {
        let nf = 150; // 300 literals → 5 words
        let (xs, ys) = xor_data(80, nf, 0.1, 41);
        let batch = encode(&xs, nf);
        let make = || {
            TsetlinMachine::with_config(2, nf, 6, 8, 3.9, 8, true, 1)
                .type_iii_feedback(20.0)
                .max_included_literals(40)
                .literal_drop_p(0.1)
        };
        let mut cpu = make();
        let mut gpu = make().to_cuda(&dev()).unwrap();
        for _ in 0..2 {
            cpu.fit_epoch(&batch, &ys);
            gpu.fit_epoch(&batch, &ys).unwrap();
        }
        assert_dense_eq!(gpu.to_cpu().unwrap(), cpu);
        assert_eq!(
            gpu.predict_batch(&batch).unwrap(),
            cpu.predict_batch(&batch)
        );
    }

    #[test]
    fn coalesced_type_iii() {
        let nf = 100;
        let (xs, ys) = xor_data(60, nf, 0.1, 43);
        let batch = encode(&xs, nf);
        let make = || {
            CoalescedTsetlinMachine::with_config(2, nf, 8, 8, 3.9, 8, true, 2)
                .type_iii_feedback(20.0)
                .max_included_literals(30)
        };
        let mut cpu = make();
        let mut gpu = make().to_cuda(&dev()).unwrap();
        for _ in 0..2 {
            cpu.fit_epoch(&batch, &ys);
            gpu.fit_epoch(&batch, &ys).unwrap();
        }
        assert_dense_eq!(gpu.to_cpu().unwrap(), cpu);
        assert_eq!(
            gpu.predict_batch(&batch).unwrap(),
            cpu.predict_batch(&batch)
        );
    }

    #[test]
    fn conv_many_patches() {
        // 2-D 12×12 image, 7×7 patches, stride 1 → 36 patches of 98 literals (2 words);
        // 1-D 400 features, kernel 40, stride 1 → 361 patches (6 patch-bitset words,
        // more than the 4 threads per block).
        let (xs, ys) = xor_data(40, 144, 0.0, 47);
        let rows: Vec<&[u8]> = xs.iter().map(|r| r.as_slice()).collect();
        let make = || {
            ConvolutionalTsetlinMachine::with_config_2d(2, 12, 12, 7, 7, 1, 6, 8, 3.9, 8, true, 3)
                .type_iii_feedback(15.0)
        };
        let mut cpu = make();
        let mut gpu = make().to_cuda(&dev()).unwrap();
        cpu.fit_epoch(&rows, &ys);
        gpu.fit_epoch(&rows, &ys).unwrap();
        let back = gpu.to_cpu().unwrap();
        assert_eq!(
            (back.ta, back.weights, back.ind),
            (cpu.ta.clone(), cpu.weights.clone(), cpu.ind.clone())
        );
        assert_eq!(gpu.predict_batch(&rows).unwrap(), cpu.predict_batch(&rows));

        let (xs, ys) = xor_data(30, 400, 0.0, 53);
        let rows: Vec<&[u8]> = xs.iter().map(|r| r.as_slice()).collect();
        let make =
            || ConvolutionalTsetlinMachine::with_config(2, 400, 40, 1, 6, 8, 3.9, 8, true, 4);
        let mut cpu = make();
        let mut gpu = make().to_cuda(&dev()).unwrap();
        for _ in 0..2 {
            cpu.fit_epoch(&rows, &ys);
            gpu.fit_epoch(&rows, &ys).unwrap();
        }
        let back = gpu.to_cpu().unwrap();
        assert_eq!(back.ta, cpu.ta);
        assert_eq!(states(&back.rngs), states(&cpu.rngs));
        assert_eq!(gpu.predict_batch(&rows).unwrap(), cpu.predict_batch(&rows));
    }

    #[test]
    fn regressor_type_iii() {
        let nf = 130;
        let (xs, _) = xor_data(60, nf, 0.0, 59);
        let ys: Vec<f64> = xs
            .iter()
            .map(|x| (x[0] + x[1] + x[2]) as f64 * 3.0)
            .collect();
        let batch = encode(&xs, nf);
        let make = || TMRegressor::with_config(nf, 8, 9, 3.9, 8, true, 5).type_iii_feedback(10.0);
        let mut cpu = make();
        let mut gpu = make().to_cuda(&dev()).unwrap();
        for _ in 0..2 {
            cpu.fit_epoch(&batch, &ys);
            gpu.fit_epoch(&batch, &ys).unwrap();
        }
        assert_dense_eq!(gpu.to_cpu().unwrap(), cpu);
    }

    #[test]
    fn autoencoders() {
        let nf = 70; // 140 literals → 3 words
        let (xs, _) = xor_data(12, nf, 0.0, 61);
        let batch = encode(&xs, nf);

        let make = || TMAutoEncoder::with_config(nf, 2, 6, 3.9, 8, true, 6).type_iii_feedback(10.0);
        let mut cpu = make();
        let mut gpu = make().to_cuda(&dev()).unwrap();
        cpu.fit_epoch(&batch);
        gpu.fit_epoch(&batch).unwrap();
        assert_dense_eq!(gpu.to_cpu().unwrap(), cpu);

        let make = || {
            TMCoalescedAutoEncoder::with_config(nf, 6, 6, 3.9, 8, true, 7).max_included_literals(10)
        };
        let mut cpu = make();
        let mut gpu = make().to_cuda(&dev()).unwrap();
        cpu.fit_epoch(&batch);
        gpu.fit_epoch(&batch).unwrap();
        assert_dense_eq!(gpu.to_cpu().unwrap(), cpu);
        assert_eq!(
            gpu.reconstruct_batch(&batch).unwrap(),
            cpu.reconstruct_batch(&batch)
        );
    }
}

#[test]
fn global_accumulator_scores_match() {
    // Inference normally accumulates per-output sums in shared memory; very wide
    // autoencoders fall back to global atomics. Both must agree with the CPU.
    use crate::{CoalescedTsetlinMachine, TMAutoEncoder};
    use std::sync::atomic::Ordering;
    let nf = 10;
    let (xs, ys) = xor_data(60, nf, 0.0, 71);
    let batch = encode(&xs, nf);
    let mut ae = TMAutoEncoder::with_config(nf, 4, 8, 3.9, 8, true, 1);
    let mut co = CoalescedTsetlinMachine::with_config(3, nf, 10, 8, 3.9, 8, true, 2);
    for _ in 0..3 {
        ae.fit_epoch(&batch);
        co.fit_epoch(&batch, &ys);
    }
    let dev = device_serial();
    let (gae, gco) = (ae.to_cuda(&dev).unwrap(), co.to_cuda(&dev).unwrap());
    super::common::FORCE_GLOBAL_ACC.store(true, Ordering::Relaxed);
    let (r1, p1) = (gae.reconstruct_batch(&batch), gco.predict_batch(&batch));
    super::common::FORCE_GLOBAL_ACC.store(false, Ordering::Relaxed);
    assert_eq!(r1.unwrap(), ae.reconstruct_batch(&batch));
    assert_eq!(p1.unwrap(), co.predict_batch(&batch));
}
