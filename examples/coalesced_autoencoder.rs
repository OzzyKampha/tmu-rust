//! TMCoalescedAutoEncoder demo — shared clause bank for binary reconstruction.
//!
//! The same structured mirrored-half data as `autoencoder.rs`, but the model uses
//! **one shared bank of clauses** voted on by every output bit via a signed per-output
//! weight matrix.  This lets a single learned clause contribute positively to some
//! output bits and negatively to others.
//!
//! Prints accuracy per epoch, an example reconstruction, and a slice of the signed
//! weight matrix so you can see how clauses specialise across outputs.
//!
//! Compare with `autoencoder.rs` (vanilla dedicated banks) and `coalesced.rs`
//! (coalesced classifier).
//!
//! `cargo run --release --example coalesced_autoencoder`

use tmu_rs::{Encoder, Rng, TMCoalescedAutoEncoder};

const N_FEATURES: usize = 20;
const N_CLAUSES: usize = 40; // shared across all 20 outputs
const THRESHOLD: i32 = 20;
const S: f64 = 3.9;
const N_TRAIN: usize = 2_000;
const N_TEST: usize = 500;
const N_EPOCHS: usize = 25;

fn main() {
    // Structured data: second half mirrors first half (bit n/2+i = bit i).
    // Random i.i.d. data has no inter-bit correlations, so reconstruction would
    // stay near 50%.  Mirrored data gives the autoencoder something to learn.
    let half = N_FEATURES / 2;
    let mut rng = Rng::new(0x1234_5678);
    let mut sample = || {
        let first: Vec<u8> = (0..half).map(|_| (rng.next_u64() & 1) as u8).collect();
        let mut v = first.clone();
        v.extend_from_slice(&first);
        v
    };

    let xs_train: Vec<Vec<u8>> = (0..N_TRAIN).map(|_| sample()).collect();
    let xs_test: Vec<Vec<u8>> = (0..N_TEST).map(|_| sample()).collect();

    let enc = Encoder::for_binary(N_FEATURES);
    let batch_train =
        enc.encode_batch(&xs_train.iter().map(|v| v.as_slice()).collect::<Vec<_>>());
    let batch_test = enc.encode_batch(&xs_test.iter().map(|v| v.as_slice()).collect::<Vec<_>>());

    let mut ae = TMCoalescedAutoEncoder::new(N_FEATURES, N_CLAUSES, THRESHOLD, S);

    // Vanilla AE would allocate N_FEATURES × N_CLAUSES = 800 dedicated clauses.
    // Coalesced shares N_CLAUSES = 40 across all outputs.
    println!(
        "TMCoalescedAutoEncoder — {N_FEATURES}-bit reconstruction \
         with {N_CLAUSES} shared clauses"
    );
    println!(
        "(Vanilla AE would use {} dedicated clauses; coalesced shares {N_CLAUSES})\n",
        N_FEATURES * N_CLAUSES
    );
    println!("{:>6}  {:>12}  {:>12}", "Epoch", "Train acc", "Test acc");

    for epoch in 0..=N_EPOCHS {
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
    let sample_enc = enc.encode_one(sample_x);
    let recon = ae.reconstruct(&sample_enc);
    let correct: usize = sample_x.iter().zip(&recon).filter(|(a, b)| *a == *b).count();

    println!("\nExample reconstruction (first test sample):");
    println!("  Input:  {:?}", sample_x);
    println!("  Output: {:?}", recon);
    println!("  {correct}/{N_FEATURES} bits correct");

    // Show the signed weight table for a handful of clauses and outputs.
    // A positive weight means the clause votes for output bit = 1;
    // negative means it votes for output bit = 0.
    // Seeing both signs in the same clause row shows how one shared clause
    // specialises differently across output positions.
    let show_clauses = 6.min(N_CLAUSES);
    let show_outputs = 10.min(N_FEATURES);

    println!("\nSigned weight matrix (first {show_clauses} shared clauses × first {show_outputs} outputs):");
    print!("{:>8}", "clause");
    for o in 0..show_outputs {
        print!("{:>6}", format!("out{o}"));
    }
    println!();

    for j in 0..show_clauses {
        print!("{j:>8}");
        for o in 0..show_outputs {
            let w = ae.clause_weight(o, j);
            print!("{w:>6}");
        }
        println!();
    }

    // Count how many clauses have mixed polarity across outputs (some + and some -).
    let mixed = (0..N_CLAUSES)
        .filter(|&j| {
            let weights: Vec<i32> = (0..N_FEATURES).map(|o| ae.clause_weight(o, j)).collect();
            weights.iter().any(|&w| w > 0) && weights.iter().any(|&w| w < 0)
        })
        .count();
    println!(
        "\n{mixed}/{N_CLAUSES} shared clauses have mixed polarity across outputs \
         (vote + for some bits, - for others)."
    );
}
