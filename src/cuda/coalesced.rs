//! GPU [`CoalescedTsetlinMachine`].

use super::common::{
    DenseState, N_DIGITS, Result, STAGE_BYTES, TRAIN_CHUNK, argmax_first, blocks, check_smem_ints,
    dense_scores, digits_mask, max_inc, restore_rngs, rng_states, shuffle,
};
use super::device::{CudaDevice, DevBuf};
use super::params::{CoPlanParams, CoUpdateParams, SCORE_SHARED, SharedFireParams};
use crate::clause_bank::dense::bmask_word;
use crate::encoder::{EncodedBatch, EncodedSample};
use crate::models::CoalescedTsetlinMachine;

/// A [`CoalescedTsetlinMachine`] whose shared clause bank lives on a CUDA device.
///
/// Per sample: one launch evaluates every shared clause, a single-block launch
/// plans the step (clause dropout, negative class — including focused negative
/// sampling — and feedback masks), and one launch applies both class passes.
/// The master RNG stays on the device during training. Bit-identical to the CPU
/// model.
pub struct CudaCoalescedTsetlinMachine {
    host: CoalescedTsetlinMachine,
    dev: CudaDevice,
    st: DenseState,
    class_rngs: DevBuf<u64>,
    master_rng: DevBuf<u64>,
    chunk: usize,
    lits: DevBuf<u64>,
    acts: DevBuf<u64>,
    fire: DevBuf<u8>,
    active: DevBuf<u8>,
    plan: DevBuf<i32>,
    inv: DevBuf<u64>,
    keep: DevBuf<u64>,
}

impl CoalescedTsetlinMachine {
    /// Copy this model to a CUDA device for GPU training / inference.
    pub fn to_cuda(&self, device: &CudaDevice) -> Result<CudaCoalescedTsetlinMachine> {
        CudaCoalescedTsetlinMachine::new(self.clone(), device)
    }
}

impl CudaCoalescedTsetlinMachine {
    fn new(host: CoalescedTsetlinMachine, dev: &CudaDevice) -> Result<Self> {
        check_smem_ints(host.n_classes, "classes")?;
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
        let (w, nc) = (host.words, host.n_clauses);
        let chunk = (STAGE_BYTES / (w * 8)).clamp(1, TRAIN_CHUNK);
        let literal_drop = host.literal_drop_p > 0.0;
        Ok(CudaCoalescedTsetlinMachine {
            class_rngs: DevBuf::from_slice(dev, &rng_states(&host.class_rngs))?,
            master_rng: DevBuf::from_slice(dev, &[host.rng.0])?,
            chunk,
            lits: DevBuf::zeros(dev, chunk * w)?,
            acts: if literal_drop {
                DevBuf::zeros(dev, chunk * w)?
            } else {
                DevBuf::from_slice(dev, &vec![!0u64; w])?
            },
            fire: DevBuf::zeros(dev, nc)?,
            active: DevBuf::zeros(dev, nc)?,
            plan: DevBuf::zeros(dev, 4)?,
            inv: DevBuf::zeros(dev, 2 * w)?,
            keep: DevBuf::zeros(dev, 2 * w)?,
            host,
            dev: dev.clone(),
            st,
        })
    }

    /// The device this model lives on.
    pub fn device(&self) -> &CudaDevice {
        &self.dev
    }

    /// Number of classes.
    pub fn n_classes(&self) -> usize {
        self.host.n_classes
    }

    /// Copy the learned state back into a CPU [`CoalescedTsetlinMachine`].
    pub fn to_cpu(&self) -> Result<CoalescedTsetlinMachine> {
        let mut m = self.host.clone();
        let CoalescedTsetlinMachine {
            ta,
            include,
            weights,
            rngs,
            ind,
            cat,
            type_iii,
            class_rngs,
            ..
        } = &mut m;
        self.st
            .download(ta, include, weights, rngs, ind, cat, *type_iii)?;
        restore_rngs(class_rngs, &self.class_rngs.download()?);
        m.rng.0 = self.master_rng.download()?[0];
        Ok(m)
    }

    /// Run one shuffled training epoch, exactly like [`CoalescedTsetlinMachine::fit_epoch`].
    pub fn fit_epoch(&mut self, batch: &EncodedBatch, ys: &[usize]) -> Result<()> {
        assert_eq!(batch.words, self.host.words);
        assert_eq!(batch.n, ys.len());
        self.host.rng.0 = self.master_rng.download()?[0];
        let order = shuffle(&mut self.host.rng, batch.n);
        self.master_rng.upload(&[self.host.rng.0])?;
        self.train(batch, ys, &order)
    }

    /// Train on a single sample, exactly like [`CoalescedTsetlinMachine::fit_one`].
    pub fn fit_one(&mut self, sample: &EncodedSample, y: usize) -> Result<()> {
        let batch = EncodedBatch::from_words(sample.0.clone(), 1, self.host.words);
        self.train(&batch, &[y], &[0])
    }

    fn train(&mut self, batch: &EncodedBatch, ys: &[usize], order: &[usize]) -> Result<()> {
        let h = &mut self.host;
        let (words, nc, ncls) = (h.words, h.n_clauses, h.n_classes);
        let literal_drop = h.literal_drop_p > 0.0;
        let mut lits = vec![0u64; self.chunk * words];
        let mut acts = vec![0u64; if literal_drop { self.chunk * words } else { 0 }];
        let ib = self.dev.inner.item_block();
        let cb = self.dev.inner.clause_block(words);

        for chunk in order.chunks(self.chunk) {
            for (s, &i) in chunk.iter().enumerate() {
                assert!(ys[i] < ncls, "label out of range");
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
            for (s, &i) in chunk.iter().enumerate() {
                let lit = self.lits.at(s * words);
                let act = if literal_drop {
                    self.acts.at(s * words)
                } else {
                    self.acts.ptr()
                };
                let fp = SharedFireParams {
                    include: self.st.include.ptr(),
                    valid: self.st.valid.ptr(),
                    lit,
                    act,
                    fire: self.fire.ptr(),
                    n_clauses: nc as u32,
                    words: words as u32,
                    n_fire_blocks: blocks(nc, ib),
                    ex_a: -1,
                    ex_b: -1,
                    ..Default::default()
                };
                self.dev
                    .inner
                    .launch("shared_fire", fp.n_fire_blocks, ib, 0, &fp)?;
                let pp = CoPlanParams {
                    master_rng: self.master_rng.ptr(),
                    class_rngs: self.class_rngs.ptr(),
                    weights: self.st.weights.ptr(),
                    fire: self.fire.ptr(),
                    active: self.active.ptr(),
                    plan: self.plan.ptr(),
                    inv: self.inv.ptr(),
                    keep: self.keep.ptr(),
                    drop_p: h.clause_drop_p,
                    n_clauses: nc as u32,
                    n_classes: ncls as u32,
                    words: words as u32,
                    y: ys[i] as u32,
                    threshold: h.threshold,
                    focused: h.focused_negative_sampling as u32,
                    dig_inv: digits_mask(&h.dig_inv),
                    dig_keep: digits_mask(&h.dig_keep),
                    n_digits: N_DIGITS,
                    _pad: 0,
                };
                self.dev
                    .inner
                    .launch("co_plan", 1, ib, (ncls * 4) as u32, &pp)?;
                let up = CoUpdateParams {
                    ta: self.st.ta.ptr(),
                    include: self.st.include.ptr(),
                    weights: self.st.weights.ptr(),
                    rngs: self.st.rngs.ptr(),
                    ind: self.st.ind.ptr(),
                    cat: self.st.cat.ptr(),
                    valid: self.st.valid.ptr(),
                    lit,
                    act,
                    fire: self.fire.ptr(),
                    active: self.active.ptr(),
                    plan: self.plan.ptr(),
                    inv: self.inv.ptr(),
                    keep: self.keep.ptr(),
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
                self.dev.inner.launch("co_update", nc as u32, cb, 0, &up)?;
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
            h.n_classes,
            h.n_clauses,
            SCORE_SHARED,
            batch.n,
            1,
            |i, buf| buf.copy_from_slice(&batch.data[i * w..(i + 1) * w]),
        )
    }

    /// Clamped per-class scores for every sample (`batch.n * n_classes`, row-major).
    pub fn scores_batch(&self, batch: &EncodedBatch) -> Result<Vec<i32>> {
        let t = self.host.threshold;
        Ok(self
            .raw_scores(batch)?
            .into_iter()
            .map(|v| v.clamp(-t, t))
            .collect())
    }

    /// Clamped class scores for one sample.
    pub fn scores(&self, sample: &EncodedSample) -> Result<Vec<i32>> {
        self.scores_batch(&EncodedBatch::from_words(
            sample.0.clone(),
            1,
            self.host.words,
        ))
    }

    /// Predict the class of every sample (argmax of the unclamped class sums,
    /// like the CPU model).
    pub fn predict_batch(&self, batch: &EncodedBatch) -> Result<Vec<usize>> {
        let s = self.raw_scores(batch)?;
        Ok(s.chunks(self.host.n_classes).map(argmax_first).collect())
    }

    /// Predict the class of one sample.
    pub fn predict(&self, sample: &EncodedSample) -> Result<usize> {
        let b = EncodedBatch::from_words(sample.0.clone(), 1, self.host.words);
        Ok(self.predict_batch(&b)?[0])
    }

    /// Fraction of correctly predicted samples.
    pub fn accuracy(&self, batch: &EncodedBatch, ys: &[usize]) -> Result<f64> {
        assert_eq!(batch.n, ys.len());
        let p = self.predict_batch(batch)?;
        Ok(p.iter().zip(ys).filter(|(a, b)| a == b).count() as f64 / ys.len().max(1) as f64)
    }
}
