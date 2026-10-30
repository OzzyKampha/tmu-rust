//! GPU [`TMCoalescedAutoEncoder`].

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
        Ok(m)
    }

    /// Run one shuffled training epoch, exactly like [`TMCoalescedAutoEncoder::fit_epoch`].
    pub fn fit_epoch(&mut self, batch: &EncodedBatch) -> Result<()> {
        assert_eq!(batch.words, self.host.words);
        let order = shuffle(&mut self.host.rng, batch.n);
        self.train(batch, &order)
    }

    /// Train on a single sample, exactly like [`TMCoalescedAutoEncoder::fit_one`].
    pub fn fit_one(&mut self, sample: &EncodedSample) -> Result<()> {
        let batch = EncodedBatch::from_words(sample.0.clone(), 1, self.host.words);
        self.train(&batch, &[0])
    }

    fn train(&mut self, batch: &EncodedBatch, order: &[usize]) -> Result<()> {
        let h = &mut self.host;
        let (words, nc, nf) = (h.words, h.n_clauses, h.n_features);
        let literal_drop = h.literal_drop_p > 0.0;
        let mut lits = vec![0u64; self.chunk * words];
        let mut acts = vec![0u64; if literal_drop { self.chunk * words } else { 0 }];
        let ib = self.dev.inner.item_block();
        let cb = self.dev.inner.clause_block(words);
        let n_fire_blocks = blocks(nc, ib);

        for chunk in order.chunks(self.chunk) {
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
                // Confidence gate: `n_clauses` unconditional draws from the gate
                // stream, so the host can track that stream itself.
                let gp = CaeGateParams {
                    weights: self.st.weights.ptr(),
                    gate: self.gate.ptr(),
                    gate_rng: h.clause_gate_rng.0,
                    n_clauses: nc as u32,
                    n_features: nf as u32,
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

                for o in 0..nf {
                    let fp = SharedFireParams {
                        include: self.st.include.ptr(),
                        valid: self.st.valid.ptr(),
                        lit,
                        act,
                        w_out: self.st.weights.at(o * nc),
                        sum: self.sums.at(s * nf + o),
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
                        .launch("shared_fire", n_fire_blocks + 1, ib, 0, &fp)?;
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
                    };
                    self.dev.inner.launch("cae_update", nc as u32, cb, 0, &up)?;
                }
            }
        }
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
