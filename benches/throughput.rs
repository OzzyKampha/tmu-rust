use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tmu_rs::{EncodedSample, Encoder, TsetlinMachine};

// ---- dataset helpers -------------------------------------------------------

/// Generate `n` XOR samples with 12 features; label = feature0 XOR feature1.
fn xor_dataset(n: usize) -> (Vec<Vec<u8>>, Vec<usize>) {
    let xs: Vec<Vec<u8>> = (0..n)
        .map(|i| (0..12u8).map(|j| ((i + j as usize) & 1) as u8).collect())
        .collect();
    let ys: Vec<usize> = xs.iter().map(|x| (x[0] ^ x[1]) as usize).collect();
    (xs, ys)
}

/// Generate `n` synthetic multiclass samples with `n_features` binary features and `n_classes` classes.
fn multiclass_dataset(n: usize, n_features: usize, n_classes: usize) -> (Vec<Vec<u8>>, Vec<usize>) {
    let xs: Vec<Vec<u8>> = (0..n)
        .map(|i| (0..n_features).map(|j| ((i + j) & 1) as u8).collect())
        .collect();
    let ys: Vec<usize> = xs.iter().map(|x| x[0] as usize % n_classes).collect();
    (xs, ys)
}

/// Convert owned vectors to byte slices for API calls.
fn as_slices(xs: &[Vec<u8>]) -> Vec<&[u8]> {
    xs.iter().map(|v| v.as_slice()).collect()
}

// ---- training benchmarks ---------------------------------------------------

/// Benchmark encode + `fit_epoch` (pack + train) across varying clause counts and
/// multiclass configurations.
fn bench_fit_epoch(c: &mut Criterion) {
    let mut g = c.benchmark_group("fit_epoch");

    // vary clause count; 1000 samples, 12 features, 2 classes
    for &cpc in &[10usize, 40, 100, 200, 500] {
        let (xs, ys) = xor_dataset(1000);
        let xr = as_slices(&xs);
        let encoder = Encoder::for_binary(12);
        g.throughput(Throughput::Elements(1000));
        g.bench_with_input(BenchmarkId::new("xor_clauses", cpc), &cpc, |b, &cpc| {
            b.iter(|| {
                let batch = encoder.encode_batch(&xr);
                let mut tm = TsetlinMachine::with_config(2, 12, cpc, 15, 3.9, 8, true, 42);
                tm.fit_epoch(&batch, &ys);
            })
        });
    }

    // multiclass: 10 classes, 64 features, 20 clauses/class, 2000 samples
    {
        let (xs, ys) = multiclass_dataset(2000, 64, 10);
        let xr = as_slices(&xs);
        let encoder = Encoder::for_binary(64);
        g.throughput(Throughput::Elements(2000));
        g.bench_function("multiclass_10x64", |b| {
            b.iter(|| {
                let batch = encoder.encode_batch(&xr);
                let mut tm = TsetlinMachine::with_config(10, 64, 20, 25, 5.0, 8, true, 42);
                tm.fit_epoch(&batch, &ys);
            })
        });
    }

    g.finish();
}

// ---- pre-encoded training benchmarks (encode once, reuse across epochs) ----

/// Benchmark `fit_epoch` on a pre-encoded batch to isolate training throughput from
/// the encoding (packing) cost.
fn bench_fit_epoch_packed(c: &mut Criterion) {
    let mut g = c.benchmark_group("fit_epoch_packed");

    for &cpc in &[10usize, 40, 100, 200, 500] {
        let (xs, ys) = xor_dataset(1000);
        let encoder = Encoder::for_binary(12);
        let batch = encoder.encode_batch(&as_slices(&xs));
        g.throughput(Throughput::Elements(1000));
        g.bench_with_input(BenchmarkId::new("xor_clauses", cpc), &cpc, |b, &cpc| {
            b.iter(|| {
                let mut tm = TsetlinMachine::with_config(2, 12, cpc, 15, 3.9, 8, true, 42);
                tm.fit_epoch(&batch, &ys);
            })
        });
    }

    {
        let (xs, ys) = multiclass_dataset(2000, 64, 10);
        let encoder = Encoder::for_binary(64);
        let batch = encoder.encode_batch(&as_slices(&xs));
        g.throughput(Throughput::Elements(2000));
        g.bench_function("multiclass_10x64", |b| {
            b.iter(|| {
                let mut tm = TsetlinMachine::with_config(10, 64, 20, 25, 5.0, 8, true, 42);
                tm.fit_epoch(&batch, &ys);
            })
        });
    }

    g.finish();
}

// ---- inference benchmarks --------------------------------------------------

/// Benchmark single-sample and batch inference latency/throughput.
fn bench_predict(c: &mut Criterion) {
    let mut g = c.benchmark_group("predict");

    // single-sample latency — pre-trained TM, one predict call on a pre-encoded sample
    {
        let (xs, ys) = xor_dataset(500);
        let encoder = Encoder::for_binary(12);
        let batch = encoder.encode_batch(&as_slices(&xs));
        let mut tm = TsetlinMachine::with_config(2, 12, 40, 15, 3.9, 8, true, 42);
        for _ in 0..10 {
            tm.fit_epoch(&batch, &ys);
        }
        let sample = encoder.encode_one(&xs[0]);
        g.throughput(Throughput::Elements(1));
        g.bench_function("single_xor", |b| b.iter(|| tm.predict(&sample)));
    }

    // batch throughput — pre-encoded batch of N samples
    for &n in &[100usize, 1000, 10_000] {
        let (xs, ys) = xor_dataset(500);
        let encoder = Encoder::for_binary(12);
        let train = encoder.encode_batch(&as_slices(&xs));
        let mut tm = TsetlinMachine::with_config(2, 12, 40, 15, 3.9, 8, true, 42);
        for _ in 0..10 {
            tm.fit_epoch(&train, &ys);
        }
        let rows: Vec<&[u8]> = (0..n).map(|i| xs[i % xs.len()].as_slice()).collect();
        let batch = encoder.encode_batch(&rows);
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("batch_xor", n), &n, |b, _| {
            b.iter(|| tm.predict_batch(&batch))
        });
    }

    g.finish();
}

// ---- encoding (packing) benchmark ------------------------------------------

/// Benchmark the public packing primitive `EncodedSample::from_bits` across varying
/// feature counts.  Unlike the lower-level internal `pack`, this allocates the output
/// `EncodedSample` per call, so the measurement includes that allocation.
fn bench_pack(c: &mut Criterion) {
    let mut g = c.benchmark_group("pack");

    for &n_features in &[12usize, 64, 256, 1024] {
        let sample: Vec<u8> = (0..n_features).map(|i| (i & 1) as u8).collect();
        g.throughput(Throughput::Elements(n_features as u64));
        g.bench_with_input(
            BenchmarkId::new("features", n_features),
            &n_features,
            |b, &nf| b.iter(|| EncodedSample::from_bits(&sample, nf)),
        );
    }

    g.finish();
}

criterion_group!(
    benches,
    bench_fit_epoch,
    bench_fit_epoch_packed,
    bench_predict,
    bench_pack
);
criterion_main!(benches);
