//! GPU [`ConvolutionalTsetlinMachine`].

use super::common::{
    DenseState, N_DIGITS, Result, argmax_last, blocks, dense_scores, digits_mask, max_inc,
    restore_rngs, rng_states, shuffle,
};
use super::device::{CudaDevice, DevBuf};
use super::params::{ConvFireParams, ConvUpdateParams, SCORE_GROUPED};
use super::vanilla::Scratch;
use crate::clause_bank::dense::bmask_word;
use crate::models::ConvolutionalTsetlinMachine;

/// A [`ConvolutionalTsetlinMachine`] whose clause bank lives on a CUDA device.
///
/// Patches are packed on the host and evaluated on the GPU (every clause ×
/// every patch in parallel). Training is bit-identical to the CPU model.
pub struct CudaConvolutionalTsetlinMachine {
    host: ConvolutionalTsetlinMachine,
    dev: CudaDevice,
    st: DenseState,
    class_rngs: DevBuf<u64>,
    s: Scratch,
}

impl ConvolutionalTsetlinMachine {
    /// Copy this model to a CUDA device for GPU training / inference.
    pub fn to_cuda(&self, device: &CudaDevice) -> Result<CudaConvolutionalTsetlinMachine> {
        CudaConvolutionalTsetlinMachine::new(self.clone(), device)
    }
}

impl CudaConvolutionalTsetlinMachine {
    fn new(host: ConvolutionalTsetlinMachine, dev: &CudaDevice) -> Result<Self> {
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
        let s = Scratch::new(
            dev,
            host.n_patches * host.words,
            host.words,
            2,
            0,
            host.literal_drop_p > 0.0,
        )?;
        Ok(CudaConvolutionalTsetlinMachine {
            host,
            dev: dev.clone(),
            st,
            class_rngs,
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

    /// Copy the learned state back into a CPU [`ConvolutionalTsetlinMachine`].
    pub fn to_cpu(&self) -> Result<ConvolutionalTsetlinMachine> {
        let mut m = self.host.clone();
        let ConvolutionalTsetlinMachine {
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

    /// Run one shuffled training epoch over raw inputs, exactly like
    /// [`ConvolutionalTsetlinMachine::fit_epoch`].
    pub fn fit_epoch(&mut self, xs: &[&[u8]], ys: &[usize]) -> Result<()> {
        assert_eq!(xs.len(), ys.len());
        let order = shuffle(&mut self.host.rng, xs.len());
        self.train(xs, ys, &order)
    }

    /// Train on one raw input, exactly like [`ConvolutionalTsetlinMachine::fit_one`].
    pub fn fit_one(&mut self, x: &[u8], y: usize) -> Result<()> {
        self.train(&[x], &[y], &[0])
    }

    fn train(&mut self, xs: &[&[u8]], ys: &[usize], order: &[usize]) -> Result<()> {
        let h = &mut self.host;
        let (words, cps, nc, np) = (h.words, h.clauses_per_class, h.n_classes, h.n_patches);
        let per = np * words;
        let literal_drop = h.literal_drop_p > 0.0;
        let chunk_len = self.s.chunk;
        let mut lits = vec![0u64; chunk_len * per];
        let mut acts = vec![0u64; if literal_drop { chunk_len * words } else { 0 }];
        let mut gids = vec![0u32; chunk_len * 2];
        let ib = self.dev.inner.item_block();
        let n_fire_blocks = blocks(2 * cps, ib);
        let grid_fire = n_fire_blocks + blocks(2, ib);
        let cb = self.dev.inner.clause_block(words.max(np.div_ceil(64)));
        let smem = (np.div_ceil(64) * 8) as u32;

        for chunk in order.chunks(chunk_len) {
            for (s, &i) in chunk.iter().enumerate() {
                let (x, y) = (xs[i], ys[i]);
                assert_eq!(x.len(), h.n_input_features);
                assert!(y < nc, "label out of range");
                h.pack_all_patches(x, &mut lits[s * per..(s + 1) * per]);
                let mut neg = h.rng.below(nc);
                while neg == y {
                    neg = h.rng.below(nc);
                }
                gids[2 * s] = y as u32;
                gids[2 * s + 1] = neg as u32;
                if literal_drop {
                    for w in &mut acts[s * words..(s + 1) * words] {
                        *w = bmask_word(&mut h.literal_rng, &h.dig_lit_active);
                    }
                }
            }
            let n = chunk.len();
            self.s.lits.upload(&lits[..n * per])?;
            self.s.group_ids.upload(&gids[..n * 2])?;
            if literal_drop {
                self.s.acts.upload(&acts[..n * words])?;
            }
            self.s.sums.zero_prefix(n * 2)?;

            for s in 0..n {
                let patches = self.s.lits.at(s * per);
                let act = self.s.act(s, words, literal_drop);
                let fp = ConvFireParams {
                    include: self.st.include.ptr(),
                    valid: self.st.valid.ptr(),
                    patches,
                    act,
                    weights: self.st.weights.ptr(),
                    sums: self.s.sums.at(2 * s),
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
                    n_patches: np as u32,
                    dig_inv: digits_mask(&h.dig_inv),
                    dig_keep: digits_mask(&h.dig_keep),
                    n_digits: N_DIGITS,
                    n_fire_blocks,
                    _pad: 0,
                };
                self.dev.inner.launch("conv_fire", grid_fire, ib, 0, &fp)?;
                let up = ConvUpdateParams {
                    ta: self.st.ta.ptr(),
                    include: self.st.include.ptr(),
                    weights: self.st.weights.ptr(),
                    rngs: self.st.rngs.ptr(),
                    ind: self.st.ind.ptr(),
                    cat: self.st.cat.ptr(),
                    valid: self.st.valid.ptr(),
                    patches,
                    act,
                    sums: fp.sums,
                    group_ids: fp.group_ids,
                    drop_s0: fp.drop_s0,
                    inv: fp.inv,
                    keep: fp.keep,
                    drop_p: h.clause_drop_p,
                    d: h.d,
                    max_inc: max_inc(h.max_included_literals),
                    n_groups: 2,
                    cpg: cps as u32,
                    words: words as u32,
                    n_literals: h.n_literals as u32,
                    n_patches: np as u32,
                    threshold: h.threshold,
                    boost: h.boost_true_positive as u32,
                    half: h.half as u32,
                    max_state: h.max_state as u32,
                    type_iii: h.type_iii as u32,
                };
                self.dev
                    .inner
                    .launch("conv_update", (2 * cps) as u32, cb, smem, &up)?;
            }
        }
        Ok(())
    }

    /// Clamped per-class scores for every input (`xs.len() * n_classes`, row-major).
    pub fn scores_batch(&self, xs: &[&[u8]]) -> Result<Vec<i32>> {
        let h = &self.host;
        let mut out = dense_scores(
            &self.dev,
            &self.st,
            h.n_classes * h.clauses_per_class,
            h.words,
            h.n_classes,
            h.clauses_per_class,
            SCORE_GROUPED,
            xs.len(),
            h.n_patches,
            |i, buf| {
                assert_eq!(xs[i].len(), h.n_input_features);
                h.pack_all_patches(xs[i], buf)
            },
        )?;
        for v in &mut out {
            *v = (*v).clamp(-h.threshold, h.threshold);
        }
        Ok(out)
    }

    /// Clamped class scores for one input.
    pub fn scores(&self, x: &[u8]) -> Result<Vec<i32>> {
        self.scores_batch(&[x])
    }

    /// Predict the class of every input.
    pub fn predict_batch(&self, xs: &[&[u8]]) -> Result<Vec<usize>> {
        let s = self.scores_batch(xs)?;
        Ok(s.chunks(self.host.n_classes).map(argmax_last).collect())
    }

    /// Predict the class of one input.
    pub fn predict(&self, x: &[u8]) -> Result<usize> {
        Ok(argmax_last(&self.scores(x)?))
    }

    /// Fraction of correctly predicted inputs.
    pub fn accuracy(&self, xs: &[&[u8]], ys: &[usize]) -> Result<f64> {
        assert_eq!(xs.len(), ys.len());
        let p = self.predict_batch(xs)?;
        Ok(p.iter().zip(ys).filter(|(a, b)| a == b).count() as f64 / ys.len().max(1) as f64)
    }
}
