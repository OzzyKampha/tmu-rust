use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use tmu_rs::TsetlinMachine;

// ---- dataset helpers -------------------------------------------------------

fn xor_dataset(n: usize) -> (Vec<Vec<u8>>, Vec<usize>) {
    let xs: Vec<Vec<u8>> = (0..n)
        .map(|i| (0..12u8).map(|j| ((i + j as usize) & 1) as u8).collect())
        .collect();
    let ys: Vec<usize> = xs.iter().map(|x| (x[0] ^ x[1]) as usize).collect();
    (xs, ys)
}

fn multiclass_dataset(n: usize, n_features: usize, n_classes: usize) -> (Vec<Vec<u8>>, Vec<usize>) {
    let xs: Vec<Vec<u8>> = (0..n)
        .map(|i| (0..n_features).map(|j| ((i + j) & 1) as u8).collect())
        .collect();
    let ys: Vec<usize> = xs.iter().map(|x| x[0] as usize % n_classes).collect();
    (xs, ys)
}

fn as_slices(xs: &[Vec<u8>]) -> Vec<&[u8]> {
    xs.iter().map(|v| v.as_slice()).collect()
}

// ---- training benchmarks ---------------------------------------------------

fn bench_fit_epoch(c: &mut Criterion) {
    let mut g = c.benchmark_group("fit_epoch");

    // vary clause count; 1000 samples, 12 features, 2 classes
    for &cpc in &[10usize, 40, 100, 200, 500] {
        let (xs, ys) = xor_dataset(1000);
        let xr = as_slices(&xs);
        g.throughput(Throughput::Elements(1000));
        g.bench_with_input(BenchmarkId::new("xor_clauses", cpc), &cpc, |b, &cpc| {
            b.iter(|| {
                let mut tm = TsetlinMachine::with_config(2, 12, cpc, 15, 3.9, 8, true, 42);
                tm.fit_epoch(&xr, &ys);
            })
        });
    }

    // multiclass: 10 classes, 64 features, 20 clauses/class, 2000 samples
    {
        let (xs, ys) = multiclass_dataset(2000, 64, 10);
        let xr = as_slices(&xs);
        g.throughput(Throughput::Elements(2000));
        g.bench_function("multiclass_10x64", |b| {
            b.iter(|| {
                let mut tm = TsetlinMachine::with_config(10, 64, 20, 25, 5.0, 8, true, 42);
                tm.fit_epoch(&xr, &ys);
            })
        });
    }

    g.finish();
}

// ---- packed training benchmarks (pre-pack once, reuse across epochs) -------

fn bench_fit_epoch_packed(c: &mut Criterion) {
    let mut g = c.benchmark_group("fit_epoch_packed");

    for &cpc in &[10usize, 40, 100, 200, 500] {
        let (xs, ys) = xor_dataset(1000);
        let tm0 = TsetlinMachine::with_config(2, 12, cpc, 15, 3.9, 8, true, 42);
        let w = tm0.words_per_sample();
        let packed: Vec<u64> = xs
            .iter()
            .flat_map(|x| {
                let mut lit = vec![0u64; w];
                TsetlinMachine::pack(x, 12, &mut lit);
                lit
            })
            .collect();
        g.throughput(Throughput::Elements(1000));
        g.bench_with_input(BenchmarkId::new("xor_clauses", cpc), &cpc, |b, &cpc| {
            b.iter(|| {
                let mut tm = TsetlinMachine::with_config(2, 12, cpc, 15, 3.9, 8, true, 42);
                tm.fit_epoch_packed(&packed, 1000, &ys);
            })
        });
    }

    {
        let (xs, ys) = multiclass_dataset(2000, 64, 10);
        let tm0 = TsetlinMachine::with_config(10, 64, 20, 25, 5.0, 8, true, 42);
        let w = tm0.words_per_sample();
        let packed: Vec<u64> = xs
            .iter()
            .flat_map(|x| {
                let mut lit = vec![0u64; w];
                TsetlinMachine::pack(x, 64, &mut lit);
                lit
            })
            .collect();
        g.throughput(Throughput::Elements(2000));
        g.bench_function("multiclass_10x64", |b| {
            b.iter(|| {
                let mut tm = TsetlinMachine::with_config(10, 64, 20, 25, 5.0, 8, true, 42);
                tm.fit_epoch_packed(&packed, 2000, &ys);
            })
        });
    }

    g.finish();
}

// ---- inference benchmarks --------------------------------------------------

fn bench_predict(c: &mut Criterion) {
    let mut g = c.benchmark_group("predict");

    // single-sample latency — pre-trained TM, one pack + predict call
    {
        let (xs, ys) = xor_dataset(500);
        let xr = as_slices(&xs);
        let mut tm = TsetlinMachine::with_config(2, 12, 40, 15, 3.9, 8, true, 42);
        for _ in 0..10 {
            tm.fit_epoch(&xr, &ys);
        }
        let sample = xs[0].as_slice();
        g.throughput(Throughput::Elements(1));
        g.bench_function("single_xor", |b| b.iter(|| tm.predict(sample)));
    }

    // batch throughput — packed batch of N samples
    for &n in &[100usize, 1000, 10_000] {
        let (xs, ys) = xor_dataset(500);
        let xr = as_slices(&xs);
        let mut tm = TsetlinMachine::with_config(2, 12, 40, 15, 3.9, 8, true, 42);
        for _ in 0..10 {
            tm.fit_epoch(&xr, &ys);
        }
        let w = tm.words_per_sample();
        let batch: Vec<u64> = xs
            .iter()
            .cycle()
            .take(n)
            .flat_map(|x| {
                let mut lit = vec![0u64; w];
                tm.pack_sample(x, &mut lit);
                lit
            })
            .collect();
        g.throughput(Throughput::Elements(n as u64));
        g.bench_with_input(BenchmarkId::new("batch_packed_xor", n), &n, |b, &n| {
            b.iter(|| tm.predict_batch_packed(&batch, n))
        });
    }

    g.finish();
}

// ---- packing benchmark -----------------------------------------------------

fn bench_pack(c: &mut Criterion) {
    let mut g = c.benchmark_group("pack");

    for &n_features in &[12usize, 64, 256, 1024] {
        let sample: Vec<u8> = (0..n_features).map(|i| (i & 1) as u8).collect();
        let mut out = vec![0u64; (2 * n_features + 63) / 64];
        g.throughput(Throughput::Elements(n_features as u64));
        g.bench_with_input(BenchmarkId::new("features", n_features), &n_features, |b, &nf| {
            b.iter(|| TsetlinMachine::pack(&sample, nf, &mut out))
        });
    }

    g.finish();
}

criterion_group!(benches, bench_fit_epoch, bench_fit_epoch_packed, bench_predict, bench_pack);
criterion_main!(benches);
