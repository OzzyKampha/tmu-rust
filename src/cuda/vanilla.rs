//! GPU [`TsetlinMachine`].

use super::common::{
    DenseState, N_DIGITS, Result, STAGE_BYTES, TRAIN_CHUNK, argmax_first, blocks, dense_scores,
    digits_mask, max_inc, restore_rngs, rng_states, shuffle,
};
use super::device::{CudaDevice, DevBuf};
use super::params::{GroupedFireParams, GroupedUpdateParams, SCORE_GROUPED};
use crate::clause_bank::dense::bmask_word;
use crate::encoder::{EncodedBatch, EncodedSample};
use crate::models::TsetlinMachine;

/// A [`TsetlinMachine`] whose clause bank lives on a CUDA device.
///
/// Create with [`TsetlinMachine::to_cuda`]; training is bit-identical to the
/// CPU model's exact path (see the [module docs](super)).
pub struct CudaTsetlinMachine {
    /// Hyper-parameters and host-side RNG streams (shuffle / negative class /
    /// literal dropout). Its learned state is stale until [`to_cpu`](Self::to_cpu).
    host: TsetlinMachine,
    dev: CudaDevice,
    st: DenseState,
    class_rngs: DevBuf<u64>,
    class_weights: DevBuf<f64>,
    s: Scratch,
}

/// Per-step staging and scratch buffers for the {y, neg} grouped kernels.
pub(crate) struct Scratch {
    /// Training steps staged per upload.
    pub chunk: usize,
    pub lits: DevBuf<u64>,
    pub acts: DevBuf<u64>,
    pub group_ids: DevBuf<u32>,
    pub sums: DevBuf<i32>,
    pub fire: DevBuf<u8>,
    pub drop_s0: DevBuf<u64>,
    pub inv: DevBuf<u64>,
    pub keep: DevBuf<u64>,
}

impl Scratch {
    pub(crate) fn new(
        dev: &CudaDevice,
        lit_words: usize,
        words: usize,
        n_groups: usize,
        fire_len: usize,
        literal_drop: bool,
    ) -> Result<Self> {
        // Bound the staged literal data (convolution stages every patch).
        let chunk = (STAGE_BYTES / (lit_words * 8)).clamp(1, TRAIN_CHUNK);
        Ok(Scratch {
            chunk,
            lits: DevBuf::zeros(dev, chunk * lit_words)?,
            acts: if literal_drop {
                DevBuf::zeros(dev, chunk * words)?
            } else {
                DevBuf::from_slice(dev, &vec![!0u64; words])?
            },
            group_ids: DevBuf::zeros(dev, chunk * n_groups)?,
            sums: DevBuf::zeros(dev, chunk * n_groups)?,
            fire: DevBuf::zeros(dev, fire_len.max(1))?,
            drop_s0: DevBuf::zeros(dev, n_groups)?,
            inv: DevBuf::zeros(dev, n_groups * words)?,
            keep: DevBuf::zeros(dev, n_groups * words)?,
        })
    }

    /// Device address of step `s`'s literal-active mask.
    pub(crate) fn act(&self, s: usize, words: usize, literal_drop: bool) -> u64 {
        if literal_drop {
            self.acts.at(s * words)
        } else {
            self.acts.ptr()
        }
    }
}

impl TsetlinMachine {
    /// Copy this model to a CUDA device for GPU training / inference.
    pub fn to_cuda(&self, device: &CudaDevice) -> Result<CudaTsetlinMachine> {
        CudaTsetlinMachine::new(self.clone(), device)
    }
}

impl CudaTsetlinMachine {
    fn new(host: TsetlinMachine, dev: &CudaDevice) -> Result<Self> {
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
        let class_rngs = DevBuf::from_slice(dev, &rng_states(&host.class_rngs))?;
        let class_weights = DevBuf::from_slice(dev, &host.class_weights)?;
        let s = Scratch::new(
            dev,
            host.words,
            host.words,
            2,
            2 * host.clauses_per_class,
            host.literal_drop_p > 0.0,
        )?;
        Ok(CudaTsetlinMachine {
            host,
            dev: dev.clone(),
            st,
            class_rngs,
            class_weights,
            s,
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

    /// Copy the learned state back into a CPU [`TsetlinMachine`] (for saving,
    /// clause inspection, or continued CPU training).
    pub fn to_cpu(&self) -> Result<TsetlinMachine> {
        let mut m = self.host.clone();
        let TsetlinMachine {
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
        Ok(m)
    }

    /// Run one training epoch (shuffled), exactly like [`TsetlinMachine::fit_epoch`].
    pub fn fit_epoch(&mut self, batch: &EncodedBatch, ys: &[usize]) -> Result<()> {
        assert_eq!(batch.words, self.host.words);
        assert_eq!(batch.n, ys.len());
        let order = shuffle(&mut self.host.rng, batch.n);
        self.train(batch, ys, &order)
    }

    /// Train on a single sample, exactly like [`TsetlinMachine::fit_one`].
    pub fn fit_one(&mut self, sample: &EncodedSample, y: usize) -> Result<()> {
        let batch = EncodedBatch::from_words(sample.0.clone(), 1, self.host.words);
        self.train(&batch, &[y], &[0])
    }

    fn train(&mut self, batch: &EncodedBatch, ys: &[usize], order: &[usize]) -> Result<()> {
        let h = &mut self.host;
        let (words, cps, nc) = (h.words, h.clauses_per_class, h.n_classes);
        let literal_drop = h.literal_drop_p > 0.0;
        let dig_lit = h.dig_lit_active.clone();
        let chunk_len = self.s.chunk;
        let mut lits = vec![0u64; chunk_len * words];
        let mut acts = vec![0u64; if literal_drop { chunk_len * words } else { 0 }];
        let mut gids = vec![0u32; chunk_len * 2];
        let ib = self.dev.inner.item_block();
        let n_fire_blocks = blocks(2 * cps, ib);
        let grid_fire = n_fire_blocks + blocks(2, ib);
        let cb = self.dev.inner.clause_block(words);

        for chunk in order.chunks(chunk_len) {
            for (s, &i) in chunk.iter().enumerate() {
                assert!(ys[i] < nc, "label out of range");
                lits[s * words..(s + 1) * words]
                    .copy_from_slice(&batch.data[i * words..(i + 1) * words]);
                let y = ys[i];
                let mut neg = h.rng.below(nc);
                while neg == y {
                    neg = h.rng.below(nc);
                }
                gids[2 * s] = y as u32;
                gids[2 * s + 1] = neg as u32;
                if literal_drop {
                    for w in &mut acts[s * words..(s + 1) * words] {
                        *w = bmask_word(&mut h.literal_rng, &dig_lit);
                    }
                }
            }
            let n = chunk.len();
            self.s.lits.upload(&lits[..n * words])?;
            self.s.group_ids.upload(&gids[..n * 2])?;
            if literal_drop {
                self.s.acts.upload(&acts[..n * words])?;
            }
            self.s.sums.zero_prefix(n * 2)?;

            for s in 0..n {
                let lit = self.s.lits.at(s * words);
                let act = self.s.act(s, words, literal_drop);
                let fp = GroupedFireParams {
                    include: self.st.include.ptr(),
                    valid: self.st.valid.ptr(),
                    lit,
                    act,
                    weights: self.st.weights.ptr(),
                    sums: self.s.sums.at(2 * s),
                    fire: self.s.fire.ptr(),
                    group_ids: self.s.group_ids.at(2 * s),
                    group_rngs: self.class_rngs.ptr(),
                    drop_s0: self.s.drop_s0.ptr(),
                    inv: self.s.inv.ptr(),
                    keep: self.s.keep.ptr(),
                    drop_p: h.clause_drop_p,
                    n_groups: 2,
                    cpg: cps as u32,
                    words: words as u32,
                    n_literals: h.n_literals as u32,
                    dig_inv: digits_mask(&h.dig_inv),
                    dig_keep: digits_mask(&h.dig_keep),
                    n_digits: N_DIGITS,
                    n_fire_blocks,
                    ae_nf: 0,
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
                    group_ids: fp.group_ids,
                    drop_s0: fp.drop_s0,
                    inv: fp.inv,
                    keep: fp.keep,
                    class_weights: self.class_weights.ptr(),
                    drop_p: h.clause_drop_p,
                    d: h.d,
                    max_inc: max_inc(h.max_included_literals),
                    n_groups: 2,
                    cpg: cps as u32,
                    words: words as u32,
                    n_literals: h.n_literals as u32,
                    threshold: h.threshold,
                    ae_nf: 0,
                    boost: h.boost_true_positive as u32,
                    half: h.half as u32,
                    max_state: h.max_state as u32,
                    type_iii: h.type_iii as u32,
                };
                self.dev
                    .inner
                    .launch("tm_grouped_update", (2 * cps) as u32, cb, 0, &up)?;
            }
        }
        Ok(())
    }

    /// Clamped per-class scores for every sample (`batch.n * n_classes`, row-major).
    pub fn scores_batch(&self, batch: &EncodedBatch) -> Result<Vec<i32>> {
        let h = &self.host;
        let w = h.words;
        assert_eq!(batch.words, w);
        let mut out = dense_scores(
            &self.dev,
            &self.st,
            h.n_classes * h.clauses_per_class,
            w,
            h.n_classes,
            h.clauses_per_class,
            SCORE_GROUPED,
            batch.n,
            1,
            |i, buf| buf.copy_from_slice(&batch.data[i * w..(i + 1) * w]),
        )?;
        for v in &mut out {
            *v = (*v).clamp(-h.threshold, h.threshold);
        }
        Ok(out)
    }

    /// Clamped class scores for one sample.
    pub fn scores(&self, sample: &EncodedSample) -> Result<Vec<i32>> {
        self.scores_batch(&EncodedBatch::from_words(
            sample.0.clone(),
            1,
            self.host.words,
        ))
    }

    /// Predict the class of every sample.
    pub fn predict_batch(&self, batch: &EncodedBatch) -> Result<Vec<usize>> {
        let s = self.scores_batch(batch)?;
        Ok(s.chunks(self.host.n_classes).map(argmax_first).collect())
    }

    /// Predict the class of one sample.
    pub fn predict(&self, sample: &EncodedSample) -> Result<usize> {
        Ok(argmax_first(&self.scores(sample)?))
    }

    /// Fraction of correctly predicted samples.
    pub fn accuracy(&self, batch: &EncodedBatch, ys: &[usize]) -> Result<f64> {
        assert_eq!(batch.n, ys.len());
        let p = self.predict_batch(batch)?;
        Ok(p.iter().zip(ys).filter(|(a, b)| a == b).count() as f64 / ys.len().max(1) as f64)
    }
}
