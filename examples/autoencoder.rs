//! TMAutoEncoder demo: learn to reconstruct random 20-bit binary vectors.
//!
//! Demonstrates that reconstruction accuracy improves over epochs, starting
//! near chance (50%) and converging toward high accuracy.

use tmu_rs::{Encoder, TMAutoEncoder};

fn main() {
    let n_features = 20usize;
    let n_train = 2000usize;
    let n_test = 500usize;

    // Generate random binary training and test data.
    let mut seed = 0x1234_5678u64;
    let mut next_bit = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed & 1) as u8
    };

    let xs_train: Vec<Vec<u8>> = (0..n_train)
        .map(|_| (0..n_features).map(|_| next_bit()).collect())
        .collect();
    let xs_test: Vec<Vec<u8>> = (0..n_test)
        .map(|_| (0..n_features).map(|_| next_bit()).collect())
        .collect();

    let enc = Encoder::for_binary(n_features);
    let batch_train = enc.encode_batch(&xs_train.iter().map(|v| v.as_slice()).collect::<Vec<_>>());
    let batch_test = enc.encode_batch(&xs_test.iter().map(|v| v.as_slice()).collect::<Vec<_>>());

    let mut ae = TMAutoEncoder::new(n_features, 40, 20, 3.9);

    println!("TMAutoEncoder demo — {n_features}-bit binary reconstruction");
    println!("{:>6}  {:>12}  {:>12}", "Epoch", "Train acc", "Test acc");

    for epoch in 0..=25 {
        if epoch > 0 {
            ae.fit_epoch(&batch_train);
        }
        if epoch == 0 || epoch % 5 == 0 {
            let tr = ae.reconstruction_accuracy(&batch_train);
            let te = ae.reconstruction_accuracy(&batch_test);
            println!("{epoch:>6}  {tr:>12.4}  {te:>12.4}");
        }
    }

    // Show a reconstruction example.
    let sample_x = &xs_test[0];
    let sample = enc.encode_one(sample_x);
    let recon = ae.reconstruct(&sample);
    println!("\nExample reconstruction (first test sample):");
    println!("  Input:  {:?}", sample_x);
    println!("  Output: {:?}", recon);
    let correct: usize = sample_x.iter().zip(&recon).filter(|(a, b)| *a == *b).count();
    println!("  {correct}/{n_features} bits correct");
}
