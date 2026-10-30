//! GPU [`TMCompositeClassifier`].

use super::CudaError;
use super::common::{Result, argmax_last};
use super::device::CudaDevice;
use super::vanilla::CudaTsetlinMachine;
use crate::encoder::{EncodedBatch, EncodedSample};
use crate::models::TMCompositeClassifier;

/// A [`TMCompositeClassifier`] whose constituent machines all live on one CUDA
/// device. Each constituent trains exactly like its CPU counterpart; inference
/// sums the constituents' clamped class scores.
pub struct CudaTMCompositeClassifier {
    members: Vec<CudaTsetlinMachine>,
    dev: CudaDevice,
}

impl TMCompositeClassifier {
    /// Copy every constituent to a CUDA device for GPU training / inference.
    pub fn to_cuda(&self, device: &CudaDevice) -> Result<CudaTMCompositeClassifier> {
        if self.classifiers.is_empty() {
            return Err(CudaError::Invalid(
                "TMCompositeClassifier has no constituent models".into(),
            ));
        }
        let members = self
            .classifiers
            .iter()
            .map(|m| m.to_cuda(device))
            .collect::<Result<Vec<_>>>()?;
        Ok(CudaTMCompositeClassifier {
            members,
            dev: device.clone(),
        })
    }
}

impl CudaTMCompositeClassifier {
    /// The device this model lives on.
    pub fn device(&self) -> &CudaDevice {
        &self.dev
    }

    /// The constituent GPU machines.
    pub fn classifiers(&self) -> &[CudaTsetlinMachine] {
        &self.members
    }

    /// Number of classes.
    pub fn n_classes(&self) -> usize {
        self.members[0].n_classes()
    }

    /// Copy every constituent back into a CPU [`TMCompositeClassifier`].
    pub fn to_cpu(&self) -> Result<TMCompositeClassifier> {
        let mut c = TMCompositeClassifier::new();
        for m in &self.members {
            c.add(m.to_cpu()?);
        }
        Ok(c)
    }

    /// Train every constituent for one epoch, like [`TMCompositeClassifier::fit_epoch`].
    pub fn fit_epoch(&mut self, batch: &EncodedBatch, ys: &[usize]) -> Result<()> {
        for m in &mut self.members {
            m.fit_epoch(batch, ys)?;
        }
        Ok(())
    }

    /// Summed clamped class scores for every sample (`batch.n * n_classes`).
    pub fn scores_batch(&self, batch: &EncodedBatch) -> Result<Vec<i32>> {
        let mut total = vec![0i32; batch.n * self.n_classes()];
        for m in &self.members {
            for (t, s) in total.iter_mut().zip(m.scores_batch(batch)?) {
                *t += s;
            }
        }
        Ok(total)
    }

    /// Summed class scores for one sample.
    pub fn scores(&self, sample: &EncodedSample) -> Result<Vec<i32>> {
        let w = sample.0.len();
        self.scores_batch(&EncodedBatch::from_words(sample.0.clone(), 1, w))
    }

    /// Predict the class of every sample.
    pub fn predict_batch(&self, batch: &EncodedBatch) -> Result<Vec<usize>> {
        let s = self.scores_batch(batch)?;
        Ok(s.chunks(self.n_classes()).map(argmax_last).collect())
    }

    /// Predict the class of one sample.
    pub fn predict(&self, sample: &EncodedSample) -> Result<usize> {
        Ok(argmax_last(&self.scores(sample)?))
    }

    /// Fraction of correctly predicted samples.
    pub fn accuracy(&self, batch: &EncodedBatch, ys: &[usize]) -> Result<f64> {
        assert_eq!(batch.n, ys.len());
        let p = self.predict_batch(batch)?;
        Ok(p.iter().zip(ys).filter(|(a, b)| a == b).count() as f64 / ys.len().max(1) as f64)
    }
}
