//! Training data held in GPU memory.

use super::common::Result;
use super::device::{CudaDevice, DevBuf};
use crate::encoder::EncodedBatch;

/// An [`EncodedBatch`] uploaded once to GPU memory.
///
/// Train on any rows of it with
/// [`CudaCoalescedTsetlinMachine::fit_epoch_rows`](super::CudaCoalescedTsetlinMachine::fit_epoch_rows)
/// and score it with
/// [`predict_batch_resident`](super::CudaCoalescedTsetlinMachine::predict_batch_resident):
/// the GPU reads each sample where it already lives, so a training loop that
/// draws many (shuffled, class-balanced) batches from one data set uploads it
/// once instead of every epoch.
pub struct CudaEncodedBatch {
    pub(crate) buf: DevBuf<u64>,
    pub(crate) n: usize,
    pub(crate) words: usize,
    pub(crate) dev: CudaDevice,
}

impl CudaEncodedBatch {
    pub fn len(&self) -> usize {
        self.n
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
    /// 64-bit words per sample.
    pub fn words(&self) -> usize {
        self.words
    }
}

impl EncodedBatch {
    /// Upload the batch to `dev` (once), for training and scoring there without
    /// further host-to-device copies.
    pub fn to_cuda(&self, dev: &CudaDevice) -> Result<CudaEncodedBatch> {
        // A zero-length allocation is not portable; keep one word for empties.
        let buf = if self.data.is_empty() {
            DevBuf::zeros(dev, 1)?
        } else {
            DevBuf::from_slice(dev, &self.data)?
        };
        Ok(CudaEncodedBatch {
            buf,
            n: self.n,
            words: self.words,
            dev: dev.clone(),
        })
    }
}
