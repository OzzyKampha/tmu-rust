//! Composite Tsetlin Machine classifier (ensemble of `TsetlinMachine` models).
//!
//! Mirrors TMU's `TMCompositeClassifier`.
//!
//! Multiple `TsetlinMachine` instances are trained independently on the same
//! dataset and combined at inference time by summing their class scores.
//! This ensemble approach improves robustness over a single model, especially
//! with fewer clauses per constituent.
//!
//! All constituent models must have the same `n_classes` and be trained on
//! inputs with the same feature count.

use crate::encoder::{EncodedBatch, EncodedSample};
use crate::models::classification::TsetlinMachine;

/// An ensemble of [`TsetlinMachine`] classifiers whose class scores are summed
/// at inference time.
///
/// ## Example
///
/// ```rust,no_run
/// use tmu_rs::{TMCompositeClassifier, TsetlinMachine};
///
/// let mut composite = TMCompositeClassifier::new();
/// composite.add(TsetlinMachine::new(2, 12, 8, 15, 3.9));
/// composite.add(TsetlinMachine::new(2, 12, 8, 15, 3.9));
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TMCompositeClassifier {
    classifiers: Vec<TsetlinMachine>,
    /// Whether to train members concurrently: `None` = auto, `Some(_)` = forced.
    /// Not serialised. Member-parallel training is exact, so this only affects speed.
    #[cfg_attr(feature = "serde", serde(skip))]
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    member_parallel: Option<bool>,
}

impl TMCompositeClassifier {
    /// Create an empty composite classifier.  Add constituents with [`add`](Self::add).
    pub fn new() -> Self {
        TMCompositeClassifier {
            classifiers: Vec::new(),
            member_parallel: None,
        }
    }

    /// Force concurrent (or serial) training of the ensemble members.
    ///
    /// Member-parallel training is **exact** — bit-identical to serial — so this is a
    /// pure speed control, unlike `TsetlinMachine::data_parallel`. Auto by default.
    pub fn data_parallel(&mut self, on: bool) -> &mut Self {
        self.member_parallel = Some(on);
        self
    }

    /// Add a `TsetlinMachine` to the ensemble.
    ///
    /// All models must have the same `n_classes`.  The first model added sets
    /// the canonical `n_classes`; subsequent models must match.
    pub fn add(&mut self, model: TsetlinMachine) -> &mut Self {
        if let Some(first) = self.classifiers.first() {
            assert_eq!(
                model.n_classes(),
                first.n_classes(),
                "all constituent classifiers must have the same n_classes"
            );
        }
        self.classifiers.push(model);
        self
    }

    /// Number of constituent classifiers.
    pub fn len(&self) -> usize {
        self.classifiers.len()
    }

    /// Return `true` if no classifiers have been added.
    pub fn is_empty(&self) -> bool {
        self.classifiers.is_empty()
    }

    /// Number of output classes (taken from the first constituent).
    pub fn n_classes(&self) -> usize {
        self.classifiers.first().map(|m| m.n_classes()).unwrap_or(0)
    }

    /// Immutable access to the constituent classifiers.
    pub fn classifiers(&self) -> &[TsetlinMachine] {
        &self.classifiers
    }

    /// Mutable access to the constituent classifiers (e.g. for manual tuning).
    pub fn classifiers_mut(&mut self) -> &mut [TsetlinMachine] {
        &mut self.classifiers
    }

    // ---- inference -----------------------------------------------------------

    fn assert_nonempty(&self) {
        assert!(
            !self.classifiers.is_empty(),
            "TMCompositeClassifier has no constituent models"
        );
    }

    /// Sum class scores from all constituents and return the argmax class.
    pub fn predict(&self, sample: &EncodedSample) -> usize {
        self.assert_nonempty();
        let n_classes = self.n_classes();
        let mut combined = vec![0i32; n_classes];
        let mut buf = vec![0i32; n_classes];
        for model in &self.classifiers {
            model.scores(sample, &mut buf);
            for (c, &s) in buf.iter().enumerate() {
                combined[c] += s;
            }
        }
        combined
            .iter()
            .enumerate()
            .max_by_key(|&(_, &v)| v)
            .map(|(i, _)| i)
            .unwrap()
    }

    /// Fill `out` with the combined (summed) class scores from all constituents.
    pub fn scores(&self, sample: &EncodedSample, out: &mut [i32]) {
        self.assert_nonempty();
        assert_eq!(out.len(), self.n_classes());
        out.fill(0);
        let mut buf = vec![0i32; self.n_classes()];
        for model in &self.classifiers {
            model.scores(sample, &mut buf);
            for (c, &s) in buf.iter().enumerate() {
                out[c] += s;
            }
        }
    }

    /// Predict classes for all samples in an encoded batch.
    pub fn predict_batch(&self, batch: &EncodedBatch) -> Vec<usize> {
        self.assert_nonempty();
        let n = batch.len();
        let words = batch.words;
        let packed = batch.data.as_slice();
        #[cfg(feature = "parallel")]
        if n >= crate::clause_bank::dense::PARALLEL_MIN {
            use rayon::prelude::*;
            return (0..n)
                .into_par_iter()
                .map(|i| self.predict_lit(&packed[i * words..(i + 1) * words]))
                .collect();
        }
        (0..n)
            .map(|i| self.predict_lit(&packed[i * words..(i + 1) * words]))
            .collect()
    }

    /// [`predict`](Self::predict) on an already-packed literal slice.
    ///
    /// Avoids copying a row of an [`EncodedBatch`] into an
    /// [`EncodedSample`] just to score it.
    fn predict_lit(&self, lit: &[u64]) -> usize {
        let n_classes = self.n_classes();
        let mut combined = vec![0i32; n_classes];
        let mut buf = vec![0i32; n_classes];
        for model in &self.classifiers {
            model.scores_lit(lit, &mut buf);
            for (c, &s) in buf.iter().enumerate() {
                combined[c] += s;
            }
        }
        combined
            .iter()
            .enumerate()
            .max_by_key(|&(_, &v)| v)
            .map(|(i, _)| i)
            .unwrap()
    }

    // ---- training ------------------------------------------------------------

    /// Run one training epoch on each constituent classifier independently.
    ///
    /// Each model trains on the same batch in a fresh random order (determined
    /// by its own internal RNG, so they diverge as expected for an ensemble).
    ///
    /// With `--features parallel` the members train concurrently. Unlike the
    /// data-parallel path inside a single model, this is **exact**: members share no
    /// mutable state and each draws its shuffle from its own RNG, so the result is
    /// bit-identical to training them one after another.
    pub fn fit_epoch(&mut self, batch: &EncodedBatch, ys: &[usize]) {
        self.assert_nonempty();
        #[cfg(feature = "parallel")]
        if self.member_parallel_enabled(batch.len()) {
            use rayon::prelude::*;
            // Suppress each member's *own* data-parallel path for the duration.
            // `rayon::current_num_threads()` does not account for nesting, so every
            // member would otherwise fork a full set of replicas — `members ×
            // threads` copies of the model. Their clause-parallel path is left
            // enabled: it is bit-identical, and it lets Rayon work-steal to fill the
            // pool when there are fewer members than cores.
            let saved: Vec<Option<bool>> = self
                .classifiers
                .iter_mut()
                .map(|m| m.suppress_data_parallel())
                .collect();
            self.classifiers
                .par_iter_mut()
                .for_each(|model| model.fit_epoch(batch, ys));
            for (model, s) in self.classifiers.iter_mut().zip(saved) {
                model.restore_data_parallel(s);
            }
            return;
        }
        for model in &mut self.classifiers {
            model.fit_epoch(batch, ys);
        }
    }

    /// Whether to train members concurrently. Costs one Rayon dispatch per epoch (no
    /// replicas, no merge), so the bar is just "more than one member, more than one
    /// core, and enough total work to notice".
    #[cfg(feature = "parallel")]
    fn member_parallel_enabled(&self, n: usize) -> bool {
        if let Some(on) = self.member_parallel {
            return on && self.classifiers.len() > 1 && rayon::current_num_threads() > 1;
        }
        self.classifiers.len() > 1
            && rayon::current_num_threads() > 1
            && n.saturating_mul(self.classifiers.len())
                >= crate::clause_bank::dense::PARALLEL_WORK_MIN
    }

    // ---- metrics -------------------------------------------------------------

    /// Fraction of correctly predicted samples in an encoded batch.
    pub fn accuracy(&self, batch: &EncodedBatch, ys: &[usize]) -> f64 {
        let preds = self.predict_batch(batch);
        let correct = preds.iter().zip(ys).filter(|&(&p, &y)| p == y).count();
        correct as f64 / ys.len() as f64
    }
}

#[cfg(feature = "serde")]
impl crate::serial::SaveLoad for TMCompositeClassifier {
    const TAG: u8 = crate::serial::TAG_COMPOSITE;
}

impl Default for TMCompositeClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::Encoder;
    use crate::rng::Rng;

    fn make_xor(n: usize, seed: u64) -> (Vec<Vec<u8>>, Vec<usize>) {
        let mut rng = Rng::new(seed);
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for _ in 0..n {
            let f: Vec<u8> = (0..12).map(|_| (rng.next_u64() & 1) as u8).collect();
            let y = (f[0] ^ f[1]) as usize;
            xs.push(f);
            ys.push(y);
        }
        (xs, ys)
    }

    fn as_slices(xs: &[Vec<u8>]) -> Vec<&[u8]> {
        xs.iter().map(|v| v.as_slice()).collect()
    }

    #[test]
    fn composite_empty() {
        let c = TMCompositeClassifier::new();
        assert!(c.is_empty());
        assert_eq!(c.n_classes(), 0);
    }

    #[test]
    fn composite_add_returns_self() {
        let mut c = TMCompositeClassifier::new();
        c.add(TsetlinMachine::new(2, 12, 8, 15, 3.9));
        assert_eq!(c.len(), 1);
        assert_eq!(c.n_classes(), 2);
    }

    #[test]
    #[should_panic(expected = "same n_classes")]
    fn composite_mismatched_n_classes_panics() {
        let mut c = TMCompositeClassifier::new();
        c.add(TsetlinMachine::new(2, 12, 8, 15, 3.9));
        c.add(TsetlinMachine::new(3, 12, 8, 15, 3.9));
    }

    #[test]
    fn composite_predict_valid_class() {
        let (xs, _) = make_xor(10, 1);
        let enc = Encoder::for_binary(12);
        let rows: Vec<&[u8]> = xs.iter().map(|v| v.as_slice()).collect();
        let batch = enc.encode_batch(&rows);

        let mut c = TMCompositeClassifier::new();
        c.add(TsetlinMachine::new(2, 12, 8, 15, 3.9));
        c.add(TsetlinMachine::new(2, 12, 8, 15, 3.9));

        let preds = c.predict_batch(&batch);
        assert!(preds.iter().all(|&p| p < 2));
    }

    /// Member-parallel training must be **bit-identical** to serial training.
    ///
    /// This is the anchor test for the whole multicore effort: unlike the
    /// data-parallel path inside a single model, ensemble members share no mutable
    /// state, so there is no averaging and no approximation to excuse a difference.
    /// If this ever fails, member-parallelism has introduced real cross-talk.
    #[cfg(feature = "parallel")]
    #[test]
    fn composite_member_parallel_matches_serial() {
        let (xtr, ytr) = make_xor(2000, 7);
        let enc = Encoder::for_binary(12);
        let btr = enc.encode_batch(&as_slices(&xtr));

        let build = || {
            let mut c = TMCompositeClassifier::new();
            for seed in [11u64, 22, 33, 44] {
                c.add(TsetlinMachine::with_config(
                    2, 12, 10, 12, 3.9, 8, true, seed,
                ));
            }
            c
        };
        let mut par = build();
        let mut ser = build();
        par.data_parallel(true);
        ser.data_parallel(false);

        for _ in 0..8 {
            par.fit_epoch(&btr, &ytr);
            ser.fit_epoch(&btr, &ytr);
        }

        // Compare the learned rules clause by clause, not just the predictions —
        // identical predictions could hide diverged internal state.
        assert_eq!(par.len(), ser.len());
        for (m_par, m_ser) in par.classifiers().iter().zip(ser.classifiers()) {
            for c in 0..m_par.n_classes() {
                for j in 0..m_par.clauses_per_class() {
                    assert_eq!(
                        m_par.clause_rule(c, j),
                        m_ser.clause_rule(c, j),
                        "clause ({c}, {j}) diverged between member-parallel and serial"
                    );
                    assert_eq!(m_par.clause_weight(c, j), m_ser.clause_weight(c, j));
                }
            }
        }
        assert_eq!(par.predict_batch(&btr), ser.predict_batch(&btr));
    }

    #[test]
    fn composite_learns_xor() {
        let (xtr, ytr) = make_xor(3000, 1);
        let (xte, yte) = make_xor(1000, 2);
        let enc = Encoder::for_binary(12);
        let btr = enc.encode_batch(&as_slices(&xtr));
        let bte = enc.encode_batch(&as_slices(&xte));

        let mut composite = TMCompositeClassifier::new();
        // Three small models that together can solve XOR
        for seed in [10u64, 20, 30] {
            composite.add(TsetlinMachine::with_config(
                2, 12, 6, 10, 3.9, 8, true, seed,
            ));
        }

        for _ in 0..20 {
            composite.fit_epoch(&btr, &ytr);
        }
        let acc = composite.accuracy(&bte, &yte);
        assert!(
            acc > 0.75,
            "ensemble accuracy {acc:.3} should beat chance significantly"
        );
    }

    #[test]
    fn composite_scores_argmax_matches_predict() {
        let (xs, _) = make_xor(50, 4);
        let enc = Encoder::for_binary(12);
        let rows: Vec<&[u8]> = xs.iter().map(|v| v.as_slice()).collect();
        let batch = enc.encode_batch(&rows);

        let mut c = TMCompositeClassifier::new();
        c.add(TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 1));
        c.add(TsetlinMachine::with_config(2, 12, 8, 10, 3.0, 8, true, 2));

        let words = batch.words;
        let mut buf = vec![0i32; 2];
        for (i, &_y) in (0..xs.len()).zip(vec![0usize; xs.len()].iter()) {
            let sample =
                crate::encoder::EncodedSample(batch.data[i * words..(i + 1) * words].to_vec());
            c.scores(&sample, &mut buf);
            let argmax = buf
                .iter()
                .enumerate()
                .max_by_key(|&(_, &v)| v)
                .map(|(j, _)| j)
                .unwrap();
            assert_eq!(argmax, c.predict(&sample));
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn composite_save_load_roundtrip() {
        use crate::serial::SaveLoad;
        let (xs, ys) = make_xor(500, 7);
        let enc = Encoder::for_binary(12);
        let rows: Vec<&[u8]> = xs.iter().map(|v| v.as_slice()).collect();
        let batch = enc.encode_batch(&rows);

        let mut c = TMCompositeClassifier::new();
        c.add(TsetlinMachine::with_config(2, 12, 6, 10, 3.9, 8, true, 1));
        c.add(TsetlinMachine::with_config(2, 12, 6, 10, 3.9, 8, true, 2));
        for _ in 0..5 {
            c.fit_epoch(&batch, &ys);
        }

        let tmp = std::env::temp_dir().join("test_composite.tmrs");
        c.save(&tmp).unwrap();
        let loaded = TMCompositeClassifier::load(&tmp).unwrap();
        let _ = std::fs::remove_file(&tmp);

        let preds_orig = c.predict_batch(&batch);
        let preds_loaded = loaded.predict_batch(&batch);
        assert_eq!(preds_orig, preds_loaded);
    }
}
