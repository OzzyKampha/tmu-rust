//! Train a small Fuzzy-Pattern TM and print its data, settings and final
//! states as JSON, for `scripts/check_fuzzy_pattern.py` (a transliteration of
//! the reference Julia code) to retrain and compare state by state.
//!
//! `fuzzy_pattern_dump SEED EPOCHS FEATURES CLAUSES T S L LF INCLUDE_LIMIT`
use tmu_rs::{Encoder, Rng, TMFuzzyPatternClassifier};
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (seed, epochs, nf, cl, t, sp, l, lf, lim): (
        u64,
        usize,
        usize,
        usize,
        i32,
        f64,
        usize,
        usize,
        u8,
    ) = (
        a[0].parse().unwrap(),
        a[1].parse().unwrap(),
        a[2].parse().unwrap(),
        a[3].parse().unwrap(),
        a[4].parse().unwrap(),
        a[5].parse().unwrap(),
        a[6].parse().unwrap(),
        a[7].parse().unwrap(),
        a[8].parse().unwrap(),
    );
    let mut rng = Rng::new(99);
    let (xs, ys): (Vec<Vec<u8>>, Vec<usize>) = (0..400)
        .map(|_| {
            let f: Vec<u8> = (0..nf)
                .map(|_| rng.next_u64().is_multiple_of(3) as u8)
                .collect();
            let mut y = (f[0] as usize + 2 * f[1] as usize + f[2] as usize) % 3;
            if rng.next_f64() < 0.1 {
                y = (y + 1) % 3;
            }
            (f, y)
        })
        .unzip();
    let enc = Encoder::for_binary(nf);
    let b = enc.encode_batch(&xs.iter().map(|x| x.as_slice()).collect::<Vec<_>>());
    let mut tm = TMFuzzyPatternClassifier::new(3, nf, cl, t, sp)
        .max_literals(l)
        .literal_failures(lf)
        .include_limit(lim)
        .seed(seed);
    for _ in 0..epochs {
        tm.fit_epoch(&b, &ys);
    }
    let states: Vec<Vec<Vec<u8>>> = (0..3)
        .map(|c| {
            (0..cl)
                .map(|j| (0..2 * nf).map(|k| tm.literal_state(c, j, k)).collect())
                .collect()
        })
        .collect();
    let pred = tm.predict_batch(&b);
    println!(
        "{}",
        serde_json::json!({"X": xs, "y": ys, "states": states, "pred": pred, "s_draws": tm.s_draws()})
    );
}
