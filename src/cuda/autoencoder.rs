//! GPU [`TMAutoEncoder`].

use super::common::{
    DenseState, N_DIGITS, Result, blocks, dense_scores, digits_mask, max_inc, restore_rngs,
    rng_states, shuffle,
};
use super::device::{CudaDevice, DevBuf};
use super::params::{GroupedFireParams, GroupedUpdateParams, SCORE_GROUPED};
use super::vanilla::Scratch;
use crate::clause_bank::dense::{WORD_BITS, bmask_word};
use crate::encoder::{EncodedBatch, EncodedSample};
use crate::models::TMAutoEncoder;

/// A [`TMAutoEncoder`] whose clause banks live on a CUDA device.
///
/// Every output's clause bank is updated in the same pair of kernel launches
/// per sample (outputs own disjoint clauses). Training is bit-identical to the
/// CPU model.
pub struct CudaTMAutoEncoder {
    host: TMAutoEncoder,
    dev: CudaDevice,
    st: DenseState,
    output_rngs: DevBuf<u64>,
    s: Scratch,
}

impl TMAutoEncoder {
    /// Copy this model to a CUDA device for GPU training / inference.
    pub fn to_cuda(&self, device: &CudaDevice) -> Result<CudaTMAutoEncoder> {
        CudaTMAutoEncoder::new(self.clone(), device)
    }
}

impl CudaTMAutoEncoder {
    fn new(host: TMAutoEncoder, dev: &CudaDevice) -> Result<Self> {
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
        let output_rngs = DevBuf::from_slice(dev, &rng_states(&host.output_rngs))?;
        let nf = host.n_features;
        let s = Scratch::new(
            dev,
            host.words,
            host.words,
            nf,
            nf * host.clauses_per_output,
            host.literal_drop_p > 0.0,
        )?;
        Ok(CudaTMAutoEncoder {
            host,
            dev: dev.clone(),
            st,
            output_rngs,
            s,
        })
    }

    /// The device this model lives on.
    pub fn device(&self) -> &CudaDevice {
        &self.dev
    }

    /// Copy the learned state back into a CPU [`TMAutoEncoder`].
    pub fn to_cpu(&self) -> Result<TMAutoEncoder> {
        let mut m = self.host.clone();
        let TMAutoEncoder {
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

    /// Run one shuffled training epoch, exactly like [`TMAutoEncoder::fit_epoch`].
    pub fn fit_epoch(&mut self, batch: &EncodedBatch) -> Result<()> {
        assert_eq!(batch.words, self.host.words);
        let order = shuffle(&mut self.host.rng, batch.n);
        self.train(batch, &order)
    }

    /// Train on a single sample, exactly like [`TMAutoEncoder::fit_one`].
    pub fn fit_one(&mut self, sample: &EncodedSample) -> Result<()> {
        let batch = EncodedBatch::from_words(sample.0.clone(), 1, self.host.words);
        self.train(&batch, &[0])
    }

    fn train(&mut self, batch: &EncodedBatch, order: &[usize]) -> Result<()> {
        let h = &mut self.host;
        let (words, cpo, nf) = (h.words, h.clauses_per_output, h.n_features);
        let literal_drop = h.literal_drop_p > 0.0;
        let chunk_len = self.s.chunk;
        let mut lits = vec![0u64; chunk_len * words];
        let mut acts = vec![0u64; if literal_drop { chunk_len * words } else { 0 }];
        let ib = self.dev.inner.item_block();
        let n_fire_blocks = blocks(nf * cpo, ib);
        let grid_fire = n_fire_blocks + blocks(nf, ib);
        let cb = self.dev.inner.clause_block(words);

        for chunk in order.chunks(chunk_len) {
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
            self.s.lits.upload(&lits[..n * words])?;
            if literal_drop {
                self.s.acts.upload(&acts[..n * words])?;
            }
            self.s.sums.zero_prefix(n * nf)?;

            for s in 0..n {
                let lit = self.s.lits.at(s * words);
                let act = self.s.act(s, words, literal_drop);
                let fp = GroupedFireParams {
                    include: self.st.include.ptr(),
                    valid: self.st.valid.ptr(),
                    lit,
                    act,
                    weights: self.st.weights.ptr(),
                    sums: self.s.sums.at(s * nf),
                    fire: self.s.fire.ptr(),
                    group_ids: 0,
                    group_rngs: self.output_rngs.ptr(),
                    drop_s0: self.s.drop_s0.ptr(),
                    inv: self.s.inv.ptr(),
                    keep: self.s.keep.ptr(),
                    drop_p: h.clause_drop_p,
                    n_groups: nf as u32,
                    cpg: cpo as u32,
                    words: words as u32,
                    n_literals: h.n_literals as u32,
                    dig_inv: digits_mask(&h.dig_inv),
                    dig_keep: digits_mask(&h.dig_keep),
                    n_digits: N_DIGITS,
                    n_fire_blocks,
                    ae_nf: nf as i32,
                    _pad: 0,
                };
                self.dev
                    .inner
                    .launch("tm_grouped_fire", grid_fire, ib, 0, &fp)?;
                let up = GroupedUpdateParams {
                    ta: self.st.ta.ptr(),
                    include: self.st.include.ptr(),
                    weights: self.st.weights.ptr(),
                    rngs: self.st.rngs.ptr(),
                    ind: self.st.ind.ptr(),
                    cat: self.st.cat.ptr(),
                    valid: self.st.valid.ptr(),
                    lit,
                    act,
                    sums: fp.sums,
                    fire: fp.fire,
                    group_ids: 0,
                    drop_s0: fp.drop_s0,
                    inv: fp.inv,
                    keep: fp.keep,
                    class_weights: 0,
                    drop_p: h.clause_drop_p,
                    d: h.d,
                    max_inc: max_inc(h.max_included_literals),
                    n_groups: nf as u32,
                    cpg: cpo as u32,
                    words: words as u32,
                    n_literals: h.n_literals as u32,
                    threshold: h.threshold,
                    ae_nf: nf as i32,
                    boost: h.boost_true_positive as u32,
                    half: h.half as u32,
                    max_state: h.max_state as u32,
                    type_iii: h.type_iii as u32,
                };
                self.dev
                    .inner
                    .launch("tm_grouped_update", (nf * cpo) as u32, cb, 0, &up)?;
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
            h.n_features * h.clauses_per_output,
            w,
            h.n_features,
            h.clauses_per_output,
            SCORE_GROUPED,
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

pub(crate) fn reconstruction_accuracy(batch: &EncodedBatch, nf: usize, raw: &[i32]) -> Result<f64> {
    let w = batch.words;
    let mut correct = 0usize;
    for (i, row) in raw.chunks(nf).enumerate() {
        let lit = &batch.data[i * w..(i + 1) * w];
        for (o, &v) in row.iter().enumerate() {
            let actual = (lit[o / WORD_BITS] >> (o % WORD_BITS)) & 1;
            correct += ((v > 0) as u64 == actual) as usize;
        }
    }
    Ok(correct as f64 / (batch.n * nf) as f64)
}
