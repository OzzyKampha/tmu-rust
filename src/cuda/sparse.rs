//! GPU [`TMSparseClassifier`].

use super::common::{
    Result, STAGE_BYTES, TRAIN_CHUNK, argmax_first, blocks, check_smem_ints, max_inc, restore_rngs,
    rng_states, shuffle,
};
use super::device::{CudaDevice, DevBuf};
use super::params::{SpFireParams, SpUpdateParams, SparseBank, SparseScoresParams};
use crate::clause_bank::dense::WORD_BITS;
use crate::clause_bank::sparse::SparseClause;
use crate::encoder::{EncodedBatch, EncodedSample};
use crate::models::TMSparseClassifier;

/// A [`TMSparseClassifier`] whose sparse clause bank lives on a CUDA device.
///
/// Each clause keeps its included / excluded / absorbed index lists in
/// fixed-capacity device arrays, and one GPU thread per clause runs the same
/// sequential absorbing-action algorithm as the CPU (so results are
/// bit-identical). Parallelism is across clauses; device memory is
/// `~14 bytes × n_clauses × n_literals`.
pub struct CudaTMSparseClassifier {
    host: TMSparseClassifier,
    dev: CudaDevice,
    bank: DevBank,
    weights: DevBuf<i32>,
    rngs: DevBuf<u64>,
    class_rngs: DevBuf<u64>,
    class_weights: DevBuf<f64>,
    chunk: usize,
    lits: DevBuf<u64>,
    acts: DevBuf<u64>,
    group_ids: DevBuf<u32>,
    sums: DevBuf<i32>,
    fire: DevBuf<u8>,
    drop_s0: DevBuf<u64>,
}

struct DevBank {
    inc_idx: DevBuf<u32>,
    exc_idx: DevBuf<u32>,
    una_idx: DevBuf<u32>,
    inc_st: DevBuf<u8>,
    exc_st: DevBuf<u8>,
    n_inc: DevBuf<u32>,
    n_exc: DevBuf<u32>,
    n_una: DevBuf<u32>,
}

impl DevBank {
    fn params(&self) -> SparseBank {
        SparseBank {
            inc_idx: self.inc_idx.ptr(),
            exc_idx: self.exc_idx.ptr(),
            una_idx: self.una_idx.ptr(),
            inc_st: self.inc_st.ptr(),
            exc_st: self.exc_st.ptr(),
            n_inc: self.n_inc.ptr(),
            n_exc: self.n_exc.ptr(),
            n_una: self.n_una.ptr(),
        }
    }
}

impl TMSparseClassifier {
    /// Copy this model to a CUDA device for GPU training / inference.
    pub fn to_cuda(&self, device: &CudaDevice) -> Result<CudaTMSparseClassifier> {
        CudaTMSparseClassifier::new(self.clone(), device)
    }
}

impl CudaTMSparseClassifier {
    fn new(host: TMSparseClassifier, dev: &CudaDevice) -> Result<Self> {
        check_smem_ints(host.n_classes, "classes")?;
        let nl = host.n_literals;
        let clauses = &host.bank.clauses;
        let n = clauses.len();
        let mut inc_idx = vec![0u32; n * nl];
        let mut exc_idx = vec![0u32; n * nl];
        let mut una_idx = vec![0u32; n * nl];
        let mut inc_st = vec![0u8; n * nl];
        let mut exc_st = vec![0u8; n * nl];
        let (mut n_inc, mut n_exc, mut n_una) = (vec![0u32; n], vec![0u32; n], vec![0u32; n]);
        for (c, cl) in clauses.iter().enumerate() {
            let b = c * nl;
            inc_idx[b..b + cl.included.len()].copy_from_slice(&cl.included);
            inc_st[b..b + cl.included.len()].copy_from_slice(&cl.included_state);
            exc_idx[b..b + cl.excluded.len()].copy_from_slice(&cl.excluded);
            exc_st[b..b + cl.excluded.len()].copy_from_slice(&cl.excluded_state);
            una_idx[b..b + cl.unallocated.len()].copy_from_slice(&cl.unallocated);
            n_inc[c] = cl.included.len() as u32;
            n_exc[c] = cl.excluded.len() as u32;
            n_una[c] = cl.unallocated.len() as u32;
        }
        let bank = DevBank {
            inc_idx: DevBuf::from_slice(dev, &inc_idx)?,
            exc_idx: DevBuf::from_slice(dev, &exc_idx)?,
            una_idx: DevBuf::from_slice(dev, &una_idx)?,
            inc_st: DevBuf::from_slice(dev, &inc_st)?,
            exc_st: DevBuf::from_slice(dev, &exc_st)?,
            n_inc: DevBuf::from_slice(dev, &n_inc)?,
            n_exc: DevBuf::from_slice(dev, &n_exc)?,
            n_una: DevBuf::from_slice(dev, &n_una)?,
        };
        let w = host.words;
        let chunk = (STAGE_BYTES / (w * 8)).clamp(1, TRAIN_CHUNK);
        let literal_drop = host.literal_drop_p > 0.0;
        Ok(CudaTMSparseClassifier {
            bank,
            weights: DevBuf::from_slice(dev, &host.weights)?,
            rngs: DevBuf::from_slice(dev, &rng_states(&host.rngs))?,
            class_rngs: DevBuf::from_slice(dev, &rng_states(&host.class_rngs))?,
            class_weights: DevBuf::from_slice(dev, &host.class_weights)?,
            chunk,
            lits: DevBuf::zeros(dev, chunk * w)?,
            acts: if literal_drop {
                DevBuf::zeros(dev, chunk * w)?
            } else {
                DevBuf::from_slice(dev, &vec![!0u64; w])?
            },
            group_ids: DevBuf::zeros(dev, chunk * 2)?,
            sums: DevBuf::zeros(dev, chunk * 2)?,
            fire: DevBuf::zeros(dev, 2 * host.clauses_per_class)?,
            drop_s0: DevBuf::zeros(dev, 2)?,
            host,
            dev: dev.clone(),
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

    /// Copy the learned state back into a CPU [`TMSparseClassifier`].
    pub fn to_cpu(&self) -> Result<TMSparseClassifier> {
        let mut m = self.host.clone();
        let nl = m.n_literals;
        let b = &self.bank;
        let (inc_idx, exc_idx, una_idx) = (
            b.inc_idx.download()?,
            b.exc_idx.download()?,
            b.una_idx.download()?,
        );
        let (inc_st, exc_st) = (b.inc_st.download()?, b.exc_st.download()?);
        let (n_inc, n_exc, n_una) = (
            b.n_inc.download()?,
            b.n_exc.download()?,
            b.n_una.download()?,
        );
        for (c, cl) in m.bank.clauses.iter_mut().enumerate() {
            let base = c * nl;
            let (ni, ne, nu) = (n_inc[c] as usize, n_exc[c] as usize, n_una[c] as usize);
            *cl = SparseClause {
                included: inc_idx[base..base + ni].to_vec(),
                included_state: inc_st[base..base + ni].to_vec(),
                excluded: exc_idx[base..base + ne].to_vec(),
                excluded_state: exc_st[base..base + ne].to_vec(),
                unallocated: una_idx[base..base + nu].to_vec(),
            };
        }
        self.weights.download_into(&mut m.weights)?;
        restore_rngs(&mut m.rngs, &self.rngs.download()?);
        restore_rngs(&mut m.class_rngs, &self.class_rngs.download()?);
        Ok(m)
    }

    /// Run one shuffled training epoch, exactly like [`TMSparseClassifier::fit_epoch`].
    pub fn fit_epoch(&mut self, batch: &EncodedBatch, ys: &[usize]) -> Result<()> {
        assert_eq!(batch.words, self.host.words);
        assert_eq!(batch.n, ys.len());
        let order = shuffle(&mut self.host.rng, batch.n);
        self.train(batch, ys, &order)
    }

    /// Train on a single sample, exactly like [`TMSparseClassifier::fit_one`].
    pub fn fit_one(&mut self, sample: &EncodedSample, y: usize) -> Result<()> {
        let batch = EncodedBatch::from_words(sample.0.clone(), 1, self.host.words);
        self.train(&batch, &[y], &[0])
    }

    fn train(&mut self, batch: &EncodedBatch, ys: &[usize], order: &[usize]) -> Result<()> {
        let h = &mut self.host;
        let (words, cps, nc) = (h.words, h.clauses_per_class, h.n_classes);
        let literal_drop = h.literal_drop_p > 0.0;
        let keep_p = 1.0 - h.literal_drop_p;
        let mut lits = vec![0u64; self.chunk * words];
        let mut acts = vec![0u64; if literal_drop { self.chunk * words } else { 0 }];
        let mut gids = vec![0u32; self.chunk * 2];
        let ib = self.dev.inner.item_block();
        let n_fire_blocks = blocks(2 * cps, ib);

        for chunk in order.chunks(self.chunk) {
            for (s, &i) in chunk.iter().enumerate() {
                let y = ys[i];
                assert!(y < nc, "label out of range");
                lits[s * words..(s + 1) * words]
                    .copy_from_slice(&batch.data[i * words..(i + 1) * words]);
                let mut neg = h.rng.below(nc);
                while neg == y {
                    neg = h.rng.below(nc);
                }
                gids[2 * s] = y as u32;
                gids[2 * s + 1] = neg as u32;
                if literal_drop {
                    // Same per-literal draws as TMSparseClassifier::fit_one_lit.
                    let mask = &mut acts[s * words..(s + 1) * words];
                    mask.fill(0);
                    for l in 0..h.n_literals {
                        if h.literal_rng.next_f64() < keep_p {
                            mask[l / WORD_BITS] |= 1u64 << (l % WORD_BITS);
                        }
                    }
                }
            }
            let n = chunk.len();
            self.lits.upload(&lits[..n * words])?;
            self.group_ids.upload(&gids[..n * 2])?;
            if literal_drop {
                self.acts.upload(&acts[..n * words])?;
            }
            self.sums.zero_prefix(n * 2)?;
            for s in 0..n {
                let lit = self.lits.at(s * words);
                let act = if literal_drop {
                    self.acts.at(s * words)
                } else {
                    self.acts.ptr()
                };
                let fp = SpFireParams {
                    bank: self.bank.params(),
                    lit,
                    act,
                    weights: self.weights.ptr(),
                    sums: self.sums.at(2 * s),
                    fire: self.fire.ptr(),
                    group_ids: self.group_ids.at(2 * s),
                    class_rngs: self.class_rngs.ptr(),
                    drop_s0: self.drop_s0.ptr(),
                    drop_p: h.clause_drop_p,
                    n_groups: 2,
                    cpg: cps as u32,
                    n_literals: h.n_literals as u32,
                    n_fire_blocks,
                };
                self.dev
                    .inner
                    .launch("sp_fire", n_fire_blocks + blocks(2, ib), ib, 0, &fp)?;
                let up = SpUpdateParams {
                    bank: fp.bank,
                    weights: self.weights.ptr(),
                    rngs: self.rngs.ptr(),
                    lit,
                    act,
                    sums: fp.sums,
                    fire: fp.fire,
                    group_ids: fp.group_ids,
                    drop_s0: fp.drop_s0,
                    class_weights: self.class_weights.ptr(),
                    drop_p: h.clause_drop_p,
                    inv_p: 1.0 / h.s,
                    keep_p: (h.s - 1.0) / h.s,
                    max_inc: max_inc(h.max_included_literals),
                    n_groups: 2,
                    cpg: cps as u32,
                    n_literals: h.n_literals as u32,
                    threshold: h.threshold,
                    boost: h.boost_true_positive as u32,
                    half: h.bank.half as u32,
                    max_state: h.bank.max_state as u32,
                    _pad: 0,
                };
                self.dev
                    .inner
                    .launch("sp_update", blocks(2 * cps, ib), ib, 0, &up)?;
            }
        }
        Ok(())
    }

    /// Clamped per-class scores for every sample (`batch.n * n_classes`, row-major).
    pub fn scores_batch(&self, batch: &EncodedBatch) -> Result<Vec<i32>> {
        let h = &self.host;
        let (w, ncls) = (h.words, h.n_classes);
        assert_eq!(batch.words, w);
        let chunk = (STAGE_BYTES / (w * 8)).clamp(1, 65_536).min(batch.n.max(1));
        let lits = DevBuf::<u64>::zeros(&self.dev, chunk * w)?;
        let out = DevBuf::<i32>::zeros(&self.dev, chunk * ncls)?;
        let mut res = vec![0i32; batch.n * ncls];
        let mut start = 0;
        while start < batch.n {
            let len = chunk.min(batch.n - start);
            lits.upload(&batch.data[start * w..(start + len) * w])?;
            let p = SparseScoresParams {
                bank: self.bank.params(),
                lits: lits.ptr(),
                weights: self.weights.ptr(),
                out: out.ptr(),
                n_clauses: (ncls * h.clauses_per_class) as u32,
                words: w as u32,
                n_classes: ncls as u32,
                cpg: h.clauses_per_class as u32,
                n_literals: h.n_literals as u32,
                _pad: 0,
            };
            self.dev.inner.launch(
                "sparse_scores",
                len as u32,
                self.dev.inner.item_block(),
                (ncls * 4) as u32,
                &p,
            )?;
            out.download_into(&mut res[start * ncls..(start + len) * ncls])?;
            start += len;
        }
        for v in &mut res {
            *v = (*v).clamp(-h.threshold, h.threshold);
        }
        Ok(res)
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
