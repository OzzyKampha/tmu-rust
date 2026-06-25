//! ConvolutionalTsetlinMachine demo (2-D).
//!
//! Dataset: 2×4 binary image (8 features, row-major), 2×2 patches, stride=1
//! → 1×3 = 3 patch positions: (0,0), (0,1), (0,2).
//!
//! Signal at patch (0,0): label = x[0,0] XOR x[1,0]  (vertical XOR of the first
//! column of the 2×2 block — a genuinely 2-D signal spanning both image rows).
//! Features not in the top-left 2×2 block are independent noise.
//!
//! Using 3 patches keeps the noise ratio identical to the 1-D comparison, so
//! both algorithms converge to ~75% (the theoretical ceiling for 1-signal / 2-noise
//! OR-semantics convolution, matching Python TMU patch-expansion + majority vote).
//!
//! Loads shared data from data/cmp_conv2d_*.bin (run scripts/gen_shared_data.py
//! once to create those files) so Rust and Python train on identical samples.
//!
//! `cargo run --release --example convolutional_2d`

use tmu_rs::ConvolutionalTsetlinMachine;

const INPUT_ROWS: usize = 2;
const INPUT_COLS: usize = 4;
const PATCH_ROWS: usize = 2;
const PATCH_COLS: usize = 2;
const STRIDE: usize = 1;
const CLAUSES: usize = 100;
const THRESHOLD: i32 = 50;
const S: f64 = 3.5;
const EPOCHS: usize = 60;

fn load_u8(path: &str, n: usize, d: usize) -> Vec<Vec<u8>> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|_| panic!("Missing {path} — run: python scripts/gen_shared_data.py"));
    assert_eq!(bytes.len(), n * d, "unexpected file size in {path}");
    bytes.chunks_exact(d).map(|r| r.to_vec()).collect()
}

fn load_u32_as_usize(path: &str, n: usize) -> Vec<usize> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|_| panic!("Missing {path} — run: python scripts/gen_shared_data.py"));
    assert_eq!(bytes.len(), n * 4, "unexpected file size in {path}");
    bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
        .collect()
}

fn main() {
    let n_features = INPUT_ROWS * INPUT_COLS;
    let n_patch_rows = (INPUT_ROWS - PATCH_ROWS) / STRIDE + 1;
    let n_patch_cols = (INPUT_COLS - PATCH_COLS) / STRIDE + 1;
    let n_patches = n_patch_rows * n_patch_cols;

    let xtr = load_u8("data/cmp_conv2d_X_train.bin", 5000, n_features);
    let ytr = load_u32_as_usize("data/cmp_conv2d_y_train.bin", 5000);
    let xte = load_u8("data/cmp_conv2d_X_test.bin",  1000, n_features);
    let yte = load_u32_as_usize("data/cmp_conv2d_y_test.bin",  1000);

    let tr: Vec<&[u8]> = xtr.iter().map(|v| v.as_slice()).collect();
    let te: Vec<&[u8]> = xte.iter().map(|v| v.as_slice()).collect();

    println!(
        "ConvolutionalTM 2-D: {INPUT_ROWS}×{INPUT_COLS} image, \
         patch={PATCH_ROWS}×{PATCH_COLS}, stride={STRIDE}, {n_patches} patches"
    );
    println!("  {CLAUSES} clauses/class, T={THRESHOLD}, s={S}");
    println!("Pattern: x[0,0] XOR x[1,0]  (vertical XOR, signal at patch (0,0); patches (0,1),(0,2) = noise)");
    println!("(shared data: data/cmp_conv2d_*.bin — identical to Python side)");
    println!("{:>5}  {:>10}  {:>10}", "epoch", "train acc", "test acc");

    let mut ctm = ConvolutionalTsetlinMachine::with_config_2d(
        2, INPUT_ROWS, INPUT_COLS, PATCH_ROWS, PATCH_COLS, STRIDE,
        CLAUSES, THRESHOLD, S, 8, true, 42,
    );

    for epoch in 1..=EPOCHS {
        ctm.fit_epoch(&tr, &ytr);
        if epoch % 10 == 0 || epoch == 1 {
            let tr_acc = ctm.accuracy(&tr, &ytr);
            let te_acc = ctm.accuracy(&te, &yte);
            println!("{epoch:>5}  {tr_acc:>10.4}  {te_acc:>10.4}");
        }
    }

    let final_acc = ctm.accuracy(&te, &yte);
    println!("\nFinal test accuracy: {final_acc:.4}");

    println!("\nTop clause rules (first 4 positive clauses of class 0, patch-relative indices):");
    println!("  Patch layout: px0=x[0,0], px1=x[0,1], px2=x[1,0], px3=x[1,1]");
    println!("  Signal: px0 XOR px2 (vertical XOR of first patch column)");
    let mut shown = 0;
    for j in (0..CLAUSES).step_by(2) {
        let rule = ctm.clause_rule(0, j);
        if rule.is_empty() { continue; }
        let features: Vec<String> = rule
            .iter()
            .map(|&(f, neg)| format!("{}px{}", if neg { "¬" } else { "" }, f))
            .collect();
        println!("  clause {:>3}  {}", j, features.join(" ∧ "));
        shown += 1;
        if shown >= 4 { break; }
    }
}
