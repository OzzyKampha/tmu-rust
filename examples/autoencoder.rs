//! TMAutoEncoder demo: learn to reconstruct 20-bit binary vectors with structure.
//!
//! The second half of each vector mirrors the first half, so each output bit
//! can be predicted from the other bits (proper autoencoder — feature o is
//! masked from the input when predicting output o). Demonstrates that
//! reconstruction accuracy improves over epochs, starting near 50% and
//! converging toward high accuracy.

use tmu_rs::{Encoder, TMAutoEncoder};

fn main() {
    let n_features = 20usize;
    let n_train = 2000usize;
    let n_test = 500usize;

    // Generate structured binary data: first half random, second half = first half.
    // This gives the autoencoder something to learn (bit n/2+i correlates with bit i).
    // Random i.i.d. data would stay near 50% — there's nothing inter-bit to discover.
    let half = n_features / 2;
    let mut rng = tmu_rs::Rng::new(0x1234_5678);
    let mut make_sample = || {
        let first: Vec<u8> = (0..half).map(|_| (rng.next_u64() & 1) as u8).collect();
        let mut v = first.clone();
        v.extend_from_slice(&first);
        v
    };

    let xs_train: Vec<Vec<u8>> = (0..n_train).map(|_| make_sample()).collect();
    let xs_test: Vec<Vec<u8>> = (0..n_test).map(|_| make_sample()).collect();

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
    let correct: usize = sample_x
        .iter()
        .zip(&recon)
        .filter(|(a, b)| *a == *b)
        .count();
    println!("  {correct}/{n_features} bits correct");
}
