//! Host-side helpers shared by the GPU models.

use super::CudaError;
use super::device::{CudaDevice, DevBuf};
use super::params::ScoresParams;
use crate::rng::Rng;

/// Training samples staged to the device per upload.
pub(crate) const TRAIN_CHUNK: usize = 1024;
/// Upper bound on staged literal data per training upload (bytes).
pub(crate) const STAGE_BYTES: usize = 64 << 20;
/// Upper bound on staged literal data per inference launch (bytes).
const PREDICT_BYTES: usize = 64 << 20;
/// Dynamic shared memory available to a block without opting in to more.
pub(crate) const SMEM_LIMIT: usize = 48 << 10;
/// Digits per Bernoulli mask word (`clause_bank::dense::MASK_BITS`).
pub(crate) const N_DIGITS: u32 = crate::clause_bank::dense::MASK_BITS as u32;

pub(crate) type Result<T> = std::result::Result<T, CudaError>;

/// Pack the binary digits of a probability (as produced by `digits_of`) into a
/// bitmask: bit `i` = digit `i`.
pub(crate) fn digits_mask(d: &[u8]) -> u32 {
    d.iter()
        .enumerate()
        .fold(0u32, |m, (i, &b)| m | ((b as u32 & 1) << i))
}

pub(crate) fn rng_states(r: &[Rng]) -> Vec<u64> {
    r.iter().map(|r| r.0).collect()
}

pub(crate) fn restore_rngs(r: &mut [Rng], states: &[u64]) {
    for (r, &s) in r.iter_mut().zip(states) {
        r.0 = s;
    }
}

/// Fisher–Yates order — identical to the CPU models' epoch shuffle.
pub(crate) fn shuffle(rng: &mut Rng, n: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let k = rng.below(i + 1);
        order.swap(i, k);
    }
    order
}

pub(crate) fn max_inc(m: usize) -> u64 {
    if m == usize::MAX { u64::MAX } else { m as u64 }
}

pub(crate) fn blocks(n: usize, block: u32) -> u32 {
    n.div_ceil(block as usize) as u32
}

/// Device-resident dense clause bank (TA states, include bitsets, weights,
/// per-clause RNGs, Type III state).
pub(crate) struct DenseState {
    pub ta: DevBuf<u8>,
    pub include: DevBuf<u64>,
    pub weights: DevBuf<i32>,
    pub rngs: DevBuf<u64>,
    pub ind: DevBuf<u8>,
    pub cat: DevBuf<u64>,
    pub valid: DevBuf<u64>,
}

impl DenseState {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        dev: &CudaDevice,
        ta: &[u8],
        include: &[u64],
        weights: &[i32],
        rngs: &[Rng],
        ind: &[u8],
        cat: &[u64],
        valid: &[u64],
        type_iii: bool,
    ) -> Result<Self> {
        Ok(DenseState {
            ta: DevBuf::from_slice(dev, ta)?,
            include: DevBuf::from_slice(dev, include)?,
            weights: DevBuf::from_slice(dev, weights)?,
            rngs: DevBuf::from_slice(dev, &rng_states(rngs))?,
            // Type III state is only touched when enabled.
            ind: if type_iii {
                DevBuf::from_slice(dev, ind)?
            } else {
                DevBuf::zeros(dev, 1)?
            },
            cat: if type_iii {
                DevBuf::from_slice(dev, cat)?
            } else {
                DevBuf::zeros(dev, 1)?
            },
            valid: DevBuf::from_slice(dev, valid)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn download(
        &self,
        ta: &mut [u8],
        include: &mut [u64],
        weights: &mut [i32],
        rngs: &mut [Rng],
        ind: &mut [u8],
        cat: &mut [u64],
        type_iii: bool,
    ) -> Result<()> {
        self.ta.download_into(ta)?;
        self.include.download_into(include)?;
        self.weights.download_into(weights)?;
        restore_rngs(rngs, &self.rngs.download()?);
        if type_iii {
            self.ind.download_into(ind)?;
            self.cat.download_into(cat)?;
        }
        Ok(())
    }
}

/// Batch inference over a dense clause bank: raw (unclamped) per-output scores,
/// `n_samples * n_outputs`, row-major. `fill(i, buf)` writes sample `i`'s
/// literals (`n_patches * words` words) into `buf`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dense_scores(
    dev: &CudaDevice,
    st: &DenseState,
    n_clauses: usize,
    words: usize,
    n_outputs: usize,
    cpg: usize,
    mode: u32,
    n_samples: usize,
    n_patches: usize,
    mut fill: impl FnMut(usize, &mut [u64]),
) -> Result<Vec<i32>> {
    let per_sample = n_patches * words;
    let chunk = (PREDICT_BYTES / (per_sample * 8).max(1))
        .clamp(1, 65_536)
        .min(n_samples.max(1));
    let lits = DevBuf::<u64>::zeros(dev, chunk * per_sample)?;
    let out = DevBuf::<i32>::zeros(dev, chunk * n_outputs)?;
    let mut host_lits = vec![0u64; chunk * per_sample];
    let mut host_out = vec![0i32; chunk * n_outputs];
    let mut result = Vec::with_capacity(n_samples * n_outputs);
    // Per-output accumulators live in shared memory unless there are too many.
    let global_acc = n_outputs * 4 > SMEM_LIMIT || force_global_acc();
    let mut start = 0;
    while start < n_samples {
        let len = chunk.min(n_samples - start);
        for i in 0..len {
            fill(
                start + i,
                &mut host_lits[i * per_sample..(i + 1) * per_sample],
            );
        }
        lits.upload(&host_lits[..len * per_sample])?;
        if global_acc {
            out.zero_prefix(len * n_outputs)?;
        }
        let p = ScoresParams {
            include: st.include.ptr(),
            valid: st.valid.ptr(),
            lits: lits.ptr(),
            weights: st.weights.ptr(),
            out: out.ptr(),
            n_clauses: n_clauses as u32,
            words: words as u32,
            n_outputs: n_outputs as u32,
            cpg: cpg as u32,
            n_patches: n_patches as u32,
            mode,
            global_acc: global_acc as u32,
            _pad: 0,
        };
        let smem = if global_acc {
            0
        } else {
            (n_outputs * 4) as u32
        };
        dev.inner
            .launch("dense_scores", len as u32, dev.inner.item_block(), smem, &p)?;
        out.download_into(&mut host_out[..len * n_outputs])?;
        result.extend_from_slice(&host_out[..len * n_outputs]);
        start += len;
    }
    Ok(result)
}

#[cfg(test)]
pub(crate) static FORCE_GLOBAL_ACC: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

fn force_global_acc() -> bool {
    #[cfg(test)]
    return FORCE_GLOBAL_ACC.load(std::sync::atomic::Ordering::Relaxed);
    #[cfg(not(test))]
    false
}

/// Error unless `n` per-block `i32` accumulators fit in shared memory.
pub(crate) fn check_smem_ints(n: usize, what: &str) -> Result<()> {
    if n * 4 > SMEM_LIMIT {
        return Err(CudaError::Invalid(format!(
            "{n} {what} exceed the GPU shared-memory limit ({} max)",
            SMEM_LIMIT / 4
        )));
    }
    Ok(())
}

/// Index of the last maximum (matches `Iterator::max_by_key`, used by the
/// convolutional and composite models).
pub(crate) fn argmax_last(v: &[i32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x >= v[best] {
            best = i;
        }
    }
    best
}

/// Index of the first maximum (matches the CPU models' strict `>` tie rule).
pub(crate) fn argmax_first(v: &[i32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}
