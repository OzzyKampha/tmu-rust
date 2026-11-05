//! GPU [`TMCoalescedAutoEncoder`].

use super::CudaError;
use super::autoencoder::reconstruction_accuracy;
use super::common::{
    DenseState, N_DIGITS, Result, STAGE_BYTES, TRAIN_CHUNK, blocks, dense_scores, digits_mask,
    max_inc, restore_rngs, rng_states, shuffle,
};
use super::device::{CudaDevice, DevBuf};
use super::params::{CaeGateParams, CaeUpdateParams, SCORE_SHARED, SharedFireParams};
use crate::clause_bank::dense::{GOLDEN, bmask_word};
use crate::encoder::{EncodedBatch, EncodedSample};
use crate::models::TMCoalescedAutoEncoder;

/// A [`TMCoalescedAutoEncoder`] whose shared clause bank lives on a CUDA device.
///
/// Outputs are trained one after another on the shared bank (each pass depends
/// on the previous one), so a sample costs `1 + 2 * n_features` kernel launches;
/// each launch is parallel over all clauses and literal words. Bit-identical to
/// the CPU model.
pub struct CudaTMCoalescedAutoEncoder {
    host: TMCoalescedAutoEncoder,
    dev: CudaDevice,
    st: DenseState,
    output_rngs: DevBuf<u64>,
    chunk: usize,
    lits: DevBuf<u64>,
    acts: DevBuf<u64>,
    sums: DevBuf<i32>,
    fire: DevBuf<u8>,
    gate: DevBuf<u8>,
    drop_s0: DevBuf<u64>,
    inv: DevBuf<u64>,
    keep: DevBuf<u64>,
}

impl TMCoalescedAutoEncoder {
    /// Copy this model to a CUDA device for GPU training / inference.
    pub fn to_cuda(&self, device: &CudaDevice) -> Result<CudaTMCoalescedAutoEncoder> {
        if self.shuffle_outputs {
            return Err(CudaError::Invalid(
                "shuffle_outputs(true) is not available on the GPU; train it on the CPU".into(),
            ));
        }
        CudaTMCoalescedAutoEncoder::new(self.clone(), device)
    }
}

impl CudaTMCoalescedAutoEncoder {
    fn new(host: TMCoalescedAutoEncoder, dev: &CudaDevice) -> Result<Self> {
        let st = DenseState::new(
            dev,
            &host.ta,
            &host.include,
            &host.weights,
            &host.rngs,
            &host.ind,
            &host.cat,
            &host.valid,
            host.type_iii,
        )?;
        let (w, nc, nf) = (host.words, host.n_clauses, host.n_features);
        let chunk = (STAGE_BYTES / (w * 8)).clamp(1, TRAIN_CHUNK);
        let literal_drop = host.literal_drop_p > 0.0;
        Ok(CudaTMCoalescedAutoEncoder {
            output_rngs: DevBuf::from_slice(dev, &rng_states(&host.output_rngs))?,
            chunk,
            lits: DevBuf::zeros(dev, chunk * w)?,
            acts: if literal_drop {
                DevBuf::zeros(dev, chunk * w)?
            } else {
                DevBuf::from_slice(dev, &vec![!0u64; w])?
            },
            sums: DevBuf::zeros(dev, chunk * nf)?,
            fire: DevBuf::zeros(dev, nc)?,
            gate: DevBuf::zeros(dev, nc)?,
            drop_s0: DevBuf::zeros(dev, 1)?,
            inv: DevBuf::zeros(dev, w)?,
            keep: DevBuf::zeros(dev, w)?,
            host,
            dev: dev.clone(),
            st,
        })
    }

    /// The device this model lives on.
    pub fn device(&self) -> &CudaDevice {
        &self.dev
    }

    /// Copy the learned state back into a CPU [`TMCoalescedAutoEncoder`].
    pub fn to_cpu(&self) -> Result<TMCoalescedAutoEncoder> {
        let mut m = self.host.clone();
        let TMCoalescedAutoEncoder {
            ta,
            include,
            weights,
            rngs,
            ind,
            cat,
            type_iii,
            output_rngs,
            ..
        } = &mut m;
        self.st
            .download(ta, include, weights, rngs, ind, cat, *type_iii)?;
        restore_rngs(output_rngs, &self.output_rngs.download()?);
        m.literal_index = Default::default();
        Ok(m)
    }

    /// Run one shuffled training epoch, exactly like [`TMCoalescedAutoEncoder::fit_epoch`].
    pub fn fit_epoch(&mut self, batch: &EncodedBatch) -> Result<()> {
        self.fit_epoch_progress(batch, &mut |_| {})
    }

    /// [`fit_epoch`](Self::fit_epoch), calling `progress` with the number of
    /// rows queued on the device so far after each row. Trains the same model.
    pub fn fit_epoch_progress(
        &mut self,
        batch: &EncodedBatch,
        progress: &mut dyn FnMut(usize),
    ) -> Result<()> {
        assert_eq!(batch.words, self.host.words);
        let order = shuffle(&mut self.host.rng, batch.n);
        self.train(batch, &order, progress)
    }

    /// Train on a single sample, exactly like [`TMCoalescedAutoEncoder::fit_one`].
    pub fn fit_one(&mut self, sample: &EncodedSample) -> Result<()> {
        let batch = EncodedBatch::from_words(sample.0.clone(), 1, self.host.words);
        self.train(&batch, &[0], &mut |_| {})
    }

    fn train(
        &mut self,
        batch: &EncodedBatch,
        order: &[usize],
        progress: &mut dyn FnMut(usize),
    ) -> Result<()> {
        let (words, nf) = (self.host.words, self.host.n_features);
        let literal_drop = self.host.literal_drop_p > 0.0;
        let mut lits = vec![0u64; self.chunk * words];
        let mut acts = vec![0u64; if literal_drop { self.chunk * words } else { 0 }];

        let mut done = 0;
        for chunk in order.chunks(self.chunk) {
            let h = &mut self.host;
            for (s, &i) in chunk.iter().enumerate() {
                lits[s * words..(s + 1) * words]
                    .copy_from_slice(&batch.data[i * words..(i + 1) * words]);
                if literal_drop {
                    for w in &mut acts[s * words..(s + 1) * words] {
                        *w = bmask_word(&mut h.literal_rng, &h.dig_lit_active);
                    }
                }
            }
            let n = chunk.len();
            self.lits.upload(&lits[..n * words])?;
            if literal_drop {
                self.acts.upload(&acts[..n * words])?;
            }
            self.sums.zero_prefix(n * nf)?;
            for s in 0..n {
                let lit = self.lits.at(s * words);
                let act = if literal_drop {
                    self.acts.at(s * words)
                } else {
                    self.acts.ptr()
                };
                self.launch_gate()?;
                for o in 0..nf {
                    self.launch_output(o, lit, act, self.sums.at(s * nf + o), -1)?;
                }
                done += 1;
                progress(done);
            }
        }
        Ok(())
    }

    /// Train like TMU's `TMAutoEncoder.fit`, exactly like
    /// [`TMCoalescedAutoEncoder::fit_examples`]: every example trains each
    /// output, in a fresh random order, on an example made for that output.
    ///
    /// The examples are drawn on the host with the CPU model's own code and
    /// random streams; the GPU evaluates and updates the clauses, so the
    /// result is bit-identical to the CPU's. (TMU's CUDA clause bank builds
    /// the examples in a kernel instead, the same algorithm.)
    pub fn fit_examples(&mut self, batch: &EncodedBatch, number_of_examples: usize) -> Result<()> {
        self.fit_examples_progress(batch, number_of_examples, &mut |_| {})
    }

    /// [`fit_examples`](Self::fit_examples), calling `progress` with the
    /// number of examples queued on the device so far after each example.
    /// Trains the same model.
    pub fn fit_examples_progress(
        &mut self,
        batch: &EncodedBatch,
        number_of_examples: usize,
        progress: &mut dyn FnMut(usize),
    ) -> Result<()> {
        assert_eq!(batch.words, self.host.words);
        if batch.n == 0 {
            return Ok(());
        }
        let (words, nf) = (self.host.words, self.host.n_features);
        let literal_drop = self.host.literal_drop_p > 0.0;
        let with = self.host.rows_with_features(batch);
        let mut union = vec![false; nf];
        let mut lits = vec![0u64; nf * words];
        let mut acts = vec![!0u64; if literal_drop { nf * words } else { 0 }];
        let mut targets = vec![0i32; nf];
        let ex_lits = DevBuf::zeros(&self.dev, nf * words)?;
        let ex_acts = if literal_drop {
            Some(DevBuf::zeros(&self.dev, nf * words)?)
        } else {
            None
        };
        let ex_sums: DevBuf<i32> = DevBuf::zeros(&self.dev, nf)?;
        for done in 1..=number_of_examples {
            // Host draws in the CPU's order: the output order, then per output
            // its example (master stream) and literal mask (literal stream).
            let order = self.host.output_order(true);
            for &o in &order {
                targets[o] = i32::from(self.host.draw_example(batch, &with, o, &mut union));
                lits[o * words..(o + 1) * words].copy_from_slice(&self.host.literals);
                if literal_drop {
                    let a = self.host.draw_literal_active();
                    acts[o * words..(o + 1) * words].copy_from_slice(&a);
                }
            }
            ex_lits.upload(&lits)?;
            if let Some(a) = &ex_acts {
                a.upload(&acts)?;
            }
            ex_sums.zero_prefix(nf)?;
            self.launch_gate()?;
            for &o in &order {
                let act = match &ex_acts {
                    Some(a) => a.at(o * words),
                    None => self.acts.ptr(),
                };
                self.launch_output(o, ex_lits.at(o * words), act, ex_sums.at(o), targets[o])?;
            }
            progress(done);
        }
        Ok(())
    }

    /// The confidence gate for one sample: `n_clauses` unconditional draws
    /// from the gate stream, so the host can track that stream itself.
    fn launch_gate(&mut self) -> Result<()> {
        let h = &mut self.host;
        let nc = h.n_clauses;
        let ib = self.dev.inner.item_block();
        let gp = CaeGateParams {
            weights: self.st.weights.ptr(),
            gate: self.gate.ptr(),
            gate_rng: h.clause_gate_rng.0,
            n_clauses: nc as u32,
            n_features: h.n_features as u32,
            threshold: h.threshold,
            _pad: 0,
        };
        self.dev
            .inner
            .launch("cae_gate", blocks(nc, ib), ib, 0, &gp)?;
        h.clause_gate_rng.0 = h
            .clause_gate_rng
            .0
            .wrapping_add((nc as u64).wrapping_mul(GOLDEN));
        Ok(())
    }

    /// Train output `o` on the sample at `lit` (active literals `act`), its
    /// vote summed into `sum`: towards `target`, or with `target = -1`
    /// towards bit `o` of the sample.
    fn launch_output(&self, o: usize, lit: u64, act: u64, sum: u64, target: i32) -> Result<()> {
        let h = &self.host;
        let (words, nc, nf) = (h.words, h.n_clauses, h.n_features);
        let cb = self.dev.inner.clause_block(words);
        let n_fire_blocks = nc as u32;
        let fp = SharedFireParams {
            include: self.st.include.ptr(),
            valid: self.st.valid.ptr(),
            lit,
            act,
            w_out: self.st.weights.at(o * nc),
            sum,
            fire: self.fire.ptr(),
            out_rng: self.output_rngs.at(o),
            drop_s0: self.drop_s0.ptr(),
            inv: self.inv.ptr(),
            keep: self.keep.ptr(),
            drop_p: h.clause_drop_p,
            n_clauses: nc as u32,
            words: words as u32,
            n_fire_blocks,
            dig_inv: digits_mask(&h.dig_inv),
            dig_keep: digits_mask(&h.dig_keep),
            n_digits: N_DIGITS,
            ex_a: o as i32,
            ex_b: (o + nf) as i32,
        };
        self.dev
            .inner
            .launch("shared_fire", n_fire_blocks + 1, cb, 0, &fp)?;
        let up = CaeUpdateParams {
            ta: self.st.ta.ptr(),
            include: self.st.include.ptr(),
            weights: fp.w_out,
            rngs: self.st.rngs.ptr(),
            ind: self.st.ind.ptr(),
            cat: self.st.cat.ptr(),
            valid: self.st.valid.ptr(),
            lit,
            act,
            fire: self.fire.ptr(),
            gate: self.gate.ptr(),
            sum: fp.sum,
            drop_s0: fp.drop_s0,
            inv: fp.inv,
            keep: fp.keep,
            drop_p: h.clause_drop_p,
            d: h.d,
            max_inc: max_inc(h.max_included_literals),
            n_clauses: nc as u32,
            words: words as u32,
            n_literals: h.n_literals as u32,
            o: o as u32,
            threshold: h.threshold,
            nf: nf as i32,
            boost: h.boost_true_positive as u32,
            half: h.half as u32,
            max_state: h.max_state as u32,
            type_iii: h.type_iii as u32,
            target,
        };
        self.dev.inner.launch("cae_update", nc as u32, cb, 0, &up)?;
        Ok(())
    }

    fn raw_scores(&self, batch: &EncodedBatch) -> Result<Vec<i32>> {
        let h = &self.host;
        let w = h.words;
        assert_eq!(batch.words, w);
        dense_scores(
            &self.dev,
            &self.st,
            h.n_clauses,
            w,
            h.n_features,
            h.n_clauses,
            SCORE_SHARED,
            batch.n,
            1,
            |i, buf| buf.copy_from_slice(&batch.data[i * w..(i + 1) * w]),
        )
    }

    /// Clamped per-output vote sums for every sample (`batch.n * n_features`).
    pub fn reconstruct_scores_batch(&self, batch: &EncodedBatch) -> Result<Vec<i32>> {
        let t = self.host.threshold;
        Ok(self
            .raw_scores(batch)?
            .into_iter()
            .map(|v| v.clamp(-t, t))
            .collect())
    }

    /// Reconstruct every sample (bit = 1 where the output's vote is positive).
    pub fn reconstruct_batch(&self, batch: &EncodedBatch) -> Result<Vec<Vec<u8>>> {
        let nf = self.host.n_features;
        Ok(self
            .raw_scores(batch)?
            .chunks(nf)
            .map(|r| r.iter().map(|&v| (v > 0) as u8).collect())
            .collect())
    }

    /// Reconstruct one sample.
    pub fn reconstruct(&self, sample: &EncodedSample) -> Result<Vec<u8>> {
        let b = EncodedBatch::from_words(sample.0.clone(), 1, self.host.words);
        Ok(self.reconstruct_batch(&b)?.pop().unwrap())
    }

    /// Fraction of (sample, bit) pairs reconstructed correctly.
    pub fn reconstruction_accuracy(&self, batch: &EncodedBatch) -> Result<f64> {
        reconstruction_accuracy(batch, self.host.n_features, &self.raw_scores(batch)?)
    }
}
