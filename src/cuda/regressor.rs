//! GPU [`TMRegressor`].

use super::common::{
    DenseState, N_DIGITS, Result, STAGE_BYTES, TRAIN_CHUNK, blocks, dense_scores, digits_mask,
    max_inc, shuffle,
};
use super::device::{CudaDevice, DevBuf};
use super::params::{RgFireParams, RgPlanParams, RgUpdateParams, SCORE_SHARED};
use crate::encoder::{EncodedBatch, EncodedSample};
use crate::models::TMRegressor;

/// A [`TMRegressor`] whose clause bank lives on a CUDA device.
///
/// Whether a sample triggers feedback depends on the current prediction, so the
/// master and literal-dropout RNG streams live on the device during training
/// (no per-sample host round-trip). Training is bit-identical to the CPU model.
pub struct CudaTMRegressor {
    host: TMRegressor,
    dev: CudaDevice,
    st: DenseState,
    master_rng: DevBuf<u64>,
    literal_rng: DevBuf<u64>,
    chunk: usize,
    lits: DevBuf<u64>,
    pred_sums: DevBuf<i32>,
    act: DevBuf<u64>,
    plan: DevBuf<i32>,
    plan_p: DevBuf<f64>,
    drop_s0: DevBuf<u64>,
    inv: DevBuf<u64>,
    keep: DevBuf<u64>,
}

impl TMRegressor {
    /// Copy this model to a CUDA device for GPU training / inference.
    pub fn to_cuda(&self, device: &CudaDevice) -> Result<CudaTMRegressor> {
        CudaTMRegressor::new(self.clone(), device)
    }
}

impl CudaTMRegressor {
    fn new(host: TMRegressor, dev: &CudaDevice) -> Result<Self> {
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
        let w = host.words;
        let chunk = (STAGE_BYTES / (w * 8)).clamp(1, TRAIN_CHUNK);
        Ok(CudaTMRegressor {
            master_rng: DevBuf::from_slice(dev, &[host.rng.0])?,
            literal_rng: DevBuf::from_slice(dev, &[host.literal_rng.0])?,
            chunk,
            lits: DevBuf::zeros(dev, chunk * w)?,
            pred_sums: DevBuf::zeros(dev, chunk)?,
            act: DevBuf::zeros(dev, w)?,
            plan: DevBuf::zeros(dev, 2)?,
            plan_p: DevBuf::zeros(dev, 1)?,
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

    /// Copy the learned state back into a CPU [`TMRegressor`].
    pub fn to_cpu(&self) -> Result<TMRegressor> {
        let mut m = self.host.clone();
        let TMRegressor {
            ta,
            include,
            weights,
            rngs,
            ind,
            cat,
            type_iii,
            ..
        } = &mut m;
        self.st
            .download(ta, include, weights, rngs, ind, cat, *type_iii)?;
        m.rng.0 = self.master_rng.download()?[0];
        m.literal_rng.0 = self.literal_rng.download()?[0];
        Ok(m)
    }

    /// Run one shuffled training epoch, exactly like [`TMRegressor::fit_epoch`].
    pub fn fit_epoch(&mut self, batch: &EncodedBatch, ys: &[f64]) -> Result<()> {
        assert_eq!(batch.words, self.host.words);
        assert_eq!(batch.n, ys.len());
        // The shuffle consumes the master stream on the host; it then continues
        // on the device for the epoch.
        self.host.rng.0 = self.master_rng.download()?[0];
        let order = shuffle(&mut self.host.rng, batch.n);
        self.master_rng.upload(&[self.host.rng.0])?;
        self.train(batch, ys, &order)
    }

    /// Train on a single sample, exactly like [`TMRegressor::fit_one`].
    pub fn fit_one(&mut self, sample: &EncodedSample, y: f64) -> Result<()> {
        let batch = EncodedBatch::from_words(sample.0.clone(), 1, self.host.words);
        self.train(&batch, &[y], &[0])
    }

    fn train(&mut self, batch: &EncodedBatch, ys: &[f64], order: &[usize]) -> Result<()> {
        let h = &self.host;
        let (words, nc) = (h.words, h.n_clauses);
        let mut lits = vec![0u64; self.chunk * words];
        let ib = self.dev.inner.item_block();
        let cb = self.dev.inner.clause_block(words);
        let literal_drop = h.literal_drop_p > 0.0;

        for chunk in order.chunks(self.chunk) {
            for (s, &i) in chunk.iter().enumerate() {
                lits[s * words..(s + 1) * words]
                    .copy_from_slice(&batch.data[i * words..(i + 1) * words]);
            }
            let n = chunk.len();
            self.lits.upload(&lits[..n * words])?;
            self.pred_sums.zero_prefix(n)?;
            for (s, &i) in chunk.iter().enumerate() {
                let lit = self.lits.at(s * words);
                let fp = RgFireParams {
                    include: self.st.include.ptr(),
                    valid: self.st.valid.ptr(),
                    lit,
                    weights: self.st.weights.ptr(),
                    pred_sum: self.pred_sums.at(s),
                    n_clauses: nc as u32,
                    words: words as u32,
                };
                self.dev
                    .inner
                    .launch("rg_fire", blocks(nc, ib), ib, 0, &fp)?;
                let pp = RgPlanParams {
                    master_rng: self.master_rng.ptr(),
                    literal_rng: self.literal_rng.ptr(),
                    pred_sum: fp.pred_sum,
                    plan_p: self.plan_p.ptr(),
                    plan: self.plan.ptr(),
                    act: self.act.ptr(),
                    drop_s0: self.drop_s0.ptr(),
                    inv: self.inv.ptr(),
                    keep: self.keep.ptr(),
                    y: ys[i],
                    drop_p: h.clause_drop_p,
                    n_clauses: nc as u32,
                    words: words as u32,
                    threshold: h.threshold,
                    dig_inv: digits_mask(&h.dig_inv),
                    dig_keep: digits_mask(&h.dig_keep),
                    dig_lit: digits_mask(&h.dig_lit_active),
                    n_digits: N_DIGITS,
                    literal_drop: literal_drop as u32,
                };
                self.dev.inner.launch("rg_plan", 1, 1, 0, &pp)?;
                let up = RgUpdateParams {
                    ta: self.st.ta.ptr(),
                    include: self.st.include.ptr(),
                    weights: self.st.weights.ptr(),
                    rngs: self.st.rngs.ptr(),
                    ind: self.st.ind.ptr(),
                    cat: self.st.cat.ptr(),
                    valid: self.st.valid.ptr(),
                    lit,
                    act: self.act.ptr(),
                    plan: self.plan.ptr(),
                    plan_p: self.plan_p.ptr(),
                    drop_s0: self.drop_s0.ptr(),
                    inv: self.inv.ptr(),
                    keep: self.keep.ptr(),
                    drop_p: h.clause_drop_p,
                    d: h.d,
                    max_inc: max_inc(h.max_included_literals),
                    n_clauses: nc as u32,
                    words: words as u32,
                    n_literals: h.n_literals as u32,
                    threshold: h.threshold,
                    boost: h.boost_true_positive as u32,
                    half: h.half as u32,
                    max_state: h.max_state as u32,
                    type_iii: h.type_iii as u32,
                };
                self.dev.inner.launch("rg_update", nc as u32, cb, 0, &up)?;
            }
        }
        Ok(())
    }

    /// Predict the continuous output of every sample (in `[0, threshold]`).
    pub fn predict_batch(&self, batch: &EncodedBatch) -> Result<Vec<f64>> {
        let h = &self.host;
        let w = h.words;
        assert_eq!(batch.words, w);
        let s = dense_scores(
            &self.dev,
            &self.st,
            h.n_clauses,
            w,
            1,
            h.n_clauses,
            SCORE_SHARED,
            batch.n,
            1,
            |i, buf| buf.copy_from_slice(&batch.data[i * w..(i + 1) * w]),
        )?;
        Ok(s.into_iter()
            .map(|v| v.clamp(0, h.threshold) as f64)
            .collect())
    }

    /// Predict the continuous output of one sample.
    pub fn predict(&self, sample: &EncodedSample) -> Result<f64> {
        let b = EncodedBatch::from_words(sample.0.clone(), 1, self.host.words);
        Ok(self.predict_batch(&b)?[0])
    }

    /// Mean absolute error over a batch.
    pub fn mae(&self, batch: &EncodedBatch, ys: &[f64]) -> Result<f64> {
        assert_eq!(batch.n, ys.len());
        let p = self.predict_batch(batch)?;
        Ok(p.iter().zip(ys).map(|(a, b)| (a - b).abs()).sum::<f64>() / ys.len() as f64)
    }

    /// Root mean squared error over a batch.
    pub fn rmse(&self, batch: &EncodedBatch, ys: &[f64]) -> Result<f64> {
        assert_eq!(batch.n, ys.len());
        let p = self.predict_batch(batch)?;
        let mse = p
            .iter()
            .zip(ys)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            / ys.len() as f64;
        Ok(mse.sqrt())
    }
}
