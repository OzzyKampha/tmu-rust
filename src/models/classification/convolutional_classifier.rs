//! Convolutional Tsetlin Machine classifier.
//!
//! Mirrors TMU's convolutional TM variant.
//!
//! Each clause operates on a fixed-size receptive field (*patch*) of
//! `kernel_size` consecutive input features rather than the full input.
//! During prediction the clause is applied to every patch position and
//! weighted votes are summed across positions.  During training a single
//! random patch is chosen per clause per sample for feedback, implementing
//! weight tying across spatial positions.
//!
//! Both 1-D (sequence) and 2-D (image) receptive fields are supported.
//! Use `with_config` for 1-D; `with_config_2d` for image patches.

#[cfg(feature = "parallel")]
use crate::clause_bank::dense::DENSE_TRAIN_PARALLEL_MIN;
use crate::clause_bank::dense::{
    bmask_word, clause_fire, digits_of, expand_bits_to_bytes, fire_predict, pack, rebuild_include,
    type_i_update_bytes, type_ii_update_bytes, type_iii_update, words_for, GOLDEN, MASK_BITS,
    WORD_BITS,
};
use crate::rng::Rng;

/// Convolutional Tsetlin Machine for multiclass classification over structured inputs.
///
/// ## Layout
///
/// Given `n_input_features` input features, `kernel_size`, and `stride`:
/// - **1-D**: `n_patches = (n_input_features − kernel_size) / stride + 1`
/// - **2-D**: `n_patches = n_patch_rows × n_patch_cols` where each axis uses the same formula
/// - Each clause has `n_literals = 2 * patch_size` literals (one positive + one negated per pixel)
/// - Clause weight tying: the same clause fires at every position; votes are summed
///
/// ## Training
///
/// For each sample the feedback probability for class `c` is computed from the
/// clamped vote sum across all patches.  Then for each clause a *single random
/// patch* is drawn to apply Type I / II TA updates (weight-tied learning).
///
/// ## Example
///
/// ```rust,no_run
/// use tmu_rs::ConvolutionalTsetlinMachine;
///
/// // 1-D: 16 features, kernel=4, stride=2  →  7 patch positions
/// let mut ctm = ConvolutionalTsetlinMachine::new(2, 16, 4, 2, 20, 100, 3.9);
///
/// // 2-D: 28×28 image, 10×10 patch, stride=2
/// let mut ctm2d = ConvolutionalTsetlinMachine::new_2d(2, 28, 28, 10, 10, 2, 20, 100, 3.9);
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConvolutionalTsetlinMachine {
    n_classes: usize,
    n_input_features: usize,
    kernel_size: usize,
    stride: usize,
    n_patches: usize,
    n_literals: usize,
    words: usize,
    // 2-D convolution fields (patch_rows == 1 means 1-D mode)
    patch_rows: usize,
    patch_cols: usize,
    input_rows: usize,
    input_cols: usize,
    n_patch_cols: usize, // patch positions along the column axis
    clauses_per_class: usize,
    threshold: i32,
    s: f64,
    boost_true_positive: bool,
    max_included_literals: usize,
    clause_drop_p: f64,
    literal_drop_p: f64,
    literal_rng: Rng,
    dig_lit_active: Vec<u8>,

    /// TA counters: clause `cj = c*CPC + j` occupies `ta[cj*n_literals..(cj+1)*n_literals]`.
    ta: Vec<u8>,
    /// Include bitset: clause `cj` occupies `include[cj*words..(cj+1)*words]`.
    include: Vec<u64>,
    half: u8,
    max_state: u8,

    weights: Vec<i32>,
    rngs: Vec<Rng>,
    class_rngs: Vec<Rng>,
    valid: Vec<u64>,
    dig_inv: Vec<u8>,
    dig_keep: Vec<u8>,

    ind: Vec<u8>,
    cat: Vec<u64>,
    d: f64,
    type_iii: bool,

    rng: Rng,
}

#[cfg(feature = "serde")]
impl crate::serial::SaveLoad for ConvolutionalTsetlinMachine {
    const TAG: u8 = crate::serial::TAG_CONVOLUTIONAL;
}

/// Apply TA feedback for one clause of a ConvolutionalTM.
///
/// Differs from the vanilla-classifier helper in that `fires` is pre-computed
/// by OR-ing the clause output across all patch positions (matching TMU's
/// `cb_calculate_clause_output_feedback` semantics).  The caller selects a patch
/// from among the FIRING patches (or any patch for the Ib-only path) and expands
/// its features into `lit_b` before calling this function.
#[allow(clippy::too_many_arguments)]
#[inline]
fn apply_one_clause_conv(
    j: usize,
    ta: &mut [u8],
    inc: &mut [u64],
    w: &mut i32,
    rng: &mut Rng,
    target: u8,
    p: f64,
    drop_mask: &[bool],
    val: &[u64],
    words: usize,
    lit_b: &[u8],
    inv_b: &[u8],
    keep_b: &[u8],
    active_b: &[u8],
    n_literals: usize,
    boost: bool,
    wmax: i32,
    max_inc: usize,
    half: u8,
    max_state: u8,
    fires: bool,
) {
    if !drop_mask.is_empty() && drop_mask[j] {
        return;
    }
    if rng.next_f64() > p {
        return;
    }
    let positive = j & 1 == 0;
    if (target == 1) == positive {
        // Type Ia / Ib: `fires` is the OR across all patches (TMU semantics).
        let under_limit = max_inc == usize::MAX || {
            let n: u32 = (0..words).map(|k| (inc[k] & val[k]).count_ones()).sum();
            (n as usize) < max_inc
        };
        let fired_under = fires && under_limit;
        if fired_under {
            *w = (*w + 1).min(wmax);
        }
        type_i_update_bytes(
            ta,
            n_literals,
            fired_under,
            boost,
            lit_b,
            inv_b,
            keep_b,
            active_b,
            max_state,
        );
    } else {
        // Type II: only runs when clause fires on at least one patch.
        if !fires {
            return;
        }
        *w = (*w - 1).max(1);
        type_ii_update_bytes(ta, n_literals, lit_b, active_b, half, max_state);
    }
    rebuild_include(ta, inc, val, words, n_literals, half);
}

impl ConvolutionalTsetlinMachine {
    /// Create a 1-D convolutional TM with default settings: 8 state bits, boost enabled, seed 42.
    ///
    /// * `n_input_features` — total number of input features.
    /// * `kernel_size` — width of each receptive field patch; must be ≤ `n_input_features`.
    /// * `stride` — step between patches (1 = fully overlapping).
    /// * `clauses_per_class` — must be even and ≥ 2.
    pub fn new(
        n_classes: usize,
        n_input_features: usize,
        kernel_size: usize,
        stride: usize,
        clauses_per_class: usize,
        threshold: i32,
        s: f64,
    ) -> Self {
        Self::with_config(
            n_classes,
            n_input_features,
            kernel_size,
            stride,
            clauses_per_class,
            threshold,
            s,
            8,
            true,
            42,
        )
    }

    /// Create a 2-D convolutional TM with default settings: 8 state bits, boost enabled, seed 42.
    ///
    /// * `input_rows`, `input_cols` — image dimensions (flattened row-major input).
    /// * `patch_rows`, `patch_cols` — patch (receptive field) dimensions.
    /// * `stride` — step between patches along both axes.
    #[allow(clippy::too_many_arguments)]
    pub fn new_2d(
        n_classes: usize,
        input_rows: usize,
        input_cols: usize,
        patch_rows: usize,
        patch_cols: usize,
        stride: usize,
        clauses_per_class: usize,
        threshold: i32,
        s: f64,
    ) -> Self {
        Self::with_config_2d(
            n_classes,
            input_rows,
            input_cols,
            patch_rows,
            patch_cols,
            stride,
            clauses_per_class,
            threshold,
            s,
            8,
            true,
            42,
        )
    }

    /// Create a convolutional TM with full configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn with_config(
        n_classes: usize,
        n_input_features: usize,
        kernel_size: usize,
        stride: usize,
        clauses_per_class: usize,
        threshold: i32,
        s: f64,
        state_bits: u8,
        boost_true_positive: bool,
        seed: u64,
    ) -> Self {
        assert!(n_classes >= 2);
        assert!(kernel_size >= 1 && kernel_size <= n_input_features);
        assert!(stride >= 1);
        assert!(clauses_per_class >= 2 && clauses_per_class.is_multiple_of(2));
        assert!(threshold >= 1);
        assert!(s > 1.0);
        assert!((2..=8).contains(&state_bits));

        let n_patches = (n_input_features - kernel_size) / stride + 1;
        assert!(
            n_patches >= 1,
            "kernel_size and stride must yield at least 1 patch"
        );

        let state_bits = state_bits as usize;
        let n_literals = 2 * kernel_size;
        let words = words_for(n_literals);
        let n_clauses = n_classes * clauses_per_class;
        let mut rng = Rng::new(seed);

        let mut valid = vec![0u64; words];
        for l in 0..n_literals {
            valid[l / WORD_BITS] |= 1u64 << (l % WORD_BITS);
        }

        let half = 1u8 << (state_bits - 1);
        let max_state = ((1u16 << state_bits) - 1) as u8;

        let mut ta = vec![0u8; n_clauses * n_literals];
        let mut include = vec![0u64; n_clauses * words];
        for cj in 0..n_clauses {
            let tb = cj * n_literals;
            for l in 0..n_literals {
                ta[tb + l] = if rng.next_u64() & 1 == 0 {
                    half - 1
                } else {
                    half
                };
            }
            rebuild_include(
                &ta[tb..tb + n_literals],
                &mut include[cj * words..(cj + 1) * words],
                &valid,
                words,
                n_literals,
                half,
            );
        }

        let rngs = (0..n_clauses)
            .map(|i| Rng::new(seed ^ (i as u64).wrapping_add(1).wrapping_mul(GOLDEN)))
            .collect();
        let class_rngs = (0..n_classes)
            .map(|c| Rng::new(seed ^ (c as u64 + n_clauses as u64 + 1).wrapping_mul(GOLDEN)))
            .collect();
        let literal_rng = Rng::new(seed ^ 0x4C49_5445_5241_4C21u64);

        ConvolutionalTsetlinMachine {
            n_classes,
            n_input_features,
            kernel_size,
            stride,
            n_patches,
            n_literals,
            words,
            // 1-D defaults for the 2-D fields
            patch_rows: 1,
            patch_cols: kernel_size,
            input_rows: 1,
            input_cols: n_input_features,
            n_patch_cols: n_patches,
            clauses_per_class,
            threshold,
            s,
            boost_true_positive,
            max_included_literals: usize::MAX,
            clause_drop_p: 0.0,
            literal_drop_p: 0.0,
            literal_rng,
            dig_lit_active: digits_of(1.0, MASK_BITS),
            ta,
            include,
            half,
            max_state,
            weights: vec![1i32; n_clauses],
            rngs,
            class_rngs,
            valid,
            dig_inv: digits_of(1.0 / s, MASK_BITS),
            dig_keep: digits_of((s - 1.0) / s, MASK_BITS),
            ind: vec![half; n_clauses * n_literals],
            cat: vec![0u64; n_clauses * words],
            d: 200.0,
            type_iii: false,
            rng,
        }
    }

    /// Create a 2-D convolutional TM with full configuration.
    ///
    /// The input is a flattened `input_rows × input_cols` binary image (row-major).
    /// Patches are `patch_rows × patch_cols` rectangles sliding with the given `stride`
    /// along both the row and column axes, matching TMU's `patch_dim=(H,W)` semantics.
    #[allow(clippy::too_many_arguments)]
    pub fn with_config_2d(
        n_classes: usize,
        input_rows: usize,
        input_cols: usize,
        patch_rows: usize,
        patch_cols: usize,
        stride: usize,
        clauses_per_class: usize,
        threshold: i32,
        s: f64,
        state_bits: u8,
        boost_true_positive: bool,
        seed: u64,
    ) -> Self {
        assert!(n_classes >= 2);
        assert!(patch_rows >= 1 && patch_rows <= input_rows);
        assert!(patch_cols >= 1 && patch_cols <= input_cols);
        assert!(stride >= 1);
        assert!(clauses_per_class >= 2 && clauses_per_class.is_multiple_of(2));
        assert!(threshold >= 1);
        assert!(s > 1.0);
        assert!((2..=8).contains(&state_bits));

        let n_patch_rows = (input_rows - patch_rows) / stride + 1;
        let n_patch_cols = (input_cols - patch_cols) / stride + 1;
        let n_patches = n_patch_rows * n_patch_cols;
        assert!(n_patches >= 1);

        let patch_size = patch_rows * patch_cols;
        let state_bits = state_bits as usize;
        let n_literals = 2 * patch_size;
        let words = words_for(n_literals);
        let n_clauses = n_classes * clauses_per_class;
        let mut rng = Rng::new(seed);

        let mut valid = vec![0u64; words];
        for l in 0..n_literals {
            valid[l / WORD_BITS] |= 1u64 << (l % WORD_BITS);
        }

        let half = 1u8 << (state_bits - 1);
        let max_state = ((1u16 << state_bits) - 1) as u8;

        let mut ta = vec![0u8; n_clauses * n_literals];
        let mut include = vec![0u64; n_clauses * words];
        for cj in 0..n_clauses {
            let tb = cj * n_literals;
            for l in 0..n_literals {
                ta[tb + l] = if rng.next_u64() & 1 == 0 {
                    half - 1
                } else {
                    half
                };
            }
            rebuild_include(
                &ta[tb..tb + n_literals],
                &mut include[cj * words..(cj + 1) * words],
                &valid,
                words,
                n_literals,
                half,
            );
        }

        let rngs = (0..n_clauses)
            .map(|i| Rng::new(seed ^ (i as u64).wrapping_add(1).wrapping_mul(GOLDEN)))
            .collect();
        let class_rngs = (0..n_classes)
            .map(|c| Rng::new(seed ^ (c as u64 + n_clauses as u64 + 1).wrapping_mul(GOLDEN)))
            .collect();
        let literal_rng = Rng::new(seed ^ 0x4C49_5445_5241_4C21u64);

        ConvolutionalTsetlinMachine {
            n_classes,
            n_input_features: input_rows * input_cols,
            kernel_size: patch_cols,
            stride,
            n_patches,
            n_literals,
            words,
            patch_rows,
            patch_cols,
            input_rows,
            input_cols,
            n_patch_cols,
            clauses_per_class,
            threshold,
            s,
            boost_true_positive,
            max_included_literals: usize::MAX,
            clause_drop_p: 0.0,
            literal_drop_p: 0.0,
            literal_rng,
            dig_lit_active: digits_of(1.0, MASK_BITS),
            ta,
            include,
            half,
            max_state,
            weights: vec![1i32; n_clauses],
            rngs,
            class_rngs,
            valid,
            dig_inv: digits_of(1.0 / s, MASK_BITS),
            dig_keep: digits_of((s - 1.0) / s, MASK_BITS),
            ind: vec![half; n_clauses * n_literals],
            cat: vec![0u64; n_clauses * words],
            d: 200.0,
            type_iii: false,
            rng,
        }
    }

    /// Limit how many literals each clause may include (Type Ia guard).
    pub fn max_included_literals(mut self, max: usize) -> Self {
        self.max_included_literals = max;
        self
    }

    /// Per-clause dropout probability during training.
    pub fn clause_drop_p(mut self, p: f64) -> Self {
        assert!((0.0..1.0).contains(&p));
        self.clause_drop_p = p;
        self
    }

    /// Per-literal dropout probability during training.
    pub fn literal_drop_p(mut self, p: f64) -> Self {
        assert!((0.0..1.0).contains(&p));
        self.literal_drop_p = p;
        self.dig_lit_active = digits_of(1.0 - p, MASK_BITS);
        self
    }

    /// Enable Type III feedback with indicator strength `d` (must be > 1.0).
    pub fn type_iii_feedback(mut self, d: f64) -> Self {
        assert!(d > 1.0, "d must be > 1.0");
        self.d = d;
        self.type_iii = true;
        self
    }

    // ---- accessors -----------------------------------------------------------

    pub fn n_classes(&self) -> usize {
        self.n_classes
    }
    pub fn n_input_features(&self) -> usize {
        self.n_input_features
    }
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }
    pub fn stride(&self) -> usize {
        self.stride
    }
    pub fn n_patches(&self) -> usize {
        self.n_patches
    }
    pub fn clauses_per_class(&self) -> usize {
        self.clauses_per_class
    }
    pub fn patch_rows(&self) -> usize {
        self.patch_rows
    }
    pub fn patch_cols(&self) -> usize {
        self.patch_cols
    }
    pub fn input_rows(&self) -> usize {
        self.input_rows
    }
    pub fn input_cols(&self) -> usize {
        self.input_cols
    }

    // ---- patch packing -------------------------------------------------------

    /// Pack patch `p_idx` from a raw feature vector into `out`.
    ///
    /// For 1-D (`patch_rows == 1`): extracts `kernel_size` consecutive bytes at offset `p_idx * stride`.
    /// For 2-D: extracts a `patch_rows × patch_cols` rectangle from the flattened image, then packs it.
    fn pack_patch(&self, x: &[u8], p_idx: usize, out: &mut [u64]) {
        if self.patch_rows == 1 {
            let start = p_idx * self.stride;
            pack(&x[start..start + self.patch_cols], self.patch_cols, out);
        } else {
            let r = (p_idx / self.n_patch_cols) * self.stride;
            let c = (p_idx % self.n_patch_cols) * self.stride;
            let patch_size = self.patch_rows * self.patch_cols;
            let mut flat = vec![0u8; patch_size];
            for pr in 0..self.patch_rows {
                let row_start = (r + pr) * self.input_cols + c;
                flat[pr * self.patch_cols..(pr + 1) * self.patch_cols]
                    .copy_from_slice(&x[row_start..row_start + self.patch_cols]);
            }
            pack(&flat, patch_size, out);
        }
    }

    /// Pack all patches from `x` into a pre-allocated flat buffer
    /// (row-major: patch 0 at `[0..words]`, patch 1 at `[words..2*words]`, …).
    fn pack_all_patches(&self, x: &[u8], buf: &mut [u64]) {
        let w = self.words;
        for p in 0..self.n_patches {
            self.pack_patch(x, p, &mut buf[p * w..(p + 1) * w]);
        }
    }

    // ---- inference -----------------------------------------------------------

    /// Compute clamped weighted clause sums for each class from pre-packed patches.
    ///
    /// `patches_buf` is a flat row-major array of packed patches
    /// (`n_patches * words` elements).
    ///
    /// Matches TMU's `cb_calculate_clause_output_predict`: a clause contributes
    /// its weight **once** if it fires on **any** patch (OR across positions),
    /// not once per firing patch.
    fn class_scores_from_patches(&self, patches_buf: &[u64], out: &mut [i32]) {
        let cps = self.clauses_per_class;
        let words = self.words;
        let inc = self.include.as_slice();
        let val = self.valid.as_slice();

        for (c, score) in out.iter_mut().enumerate() {
            let cw = &self.weights[c * cps..(c + 1) * cps];
            let mut sum = 0i32;
            for (j, &w) in cw.iter().enumerate() {
                let cj = c * cps + j;
                let clause_inc = &inc[cj * words..(cj + 1) * words];
                // OR semantics: fires if it fires on ANY patch position.
                let fires = (0..self.n_patches).any(|p| {
                    fire_predict(
                        clause_inc,
                        &patches_buf[p * words..(p + 1) * words],
                        val,
                        words,
                    )
                });
                if fires {
                    if j & 1 == 0 {
                        sum += w;
                    } else {
                        sum -= w;
                    }
                }
            }
            *score = sum.clamp(-self.threshold, self.threshold);
        }
    }

    /// Predict the class for a raw (unpacked) input vector.
    ///
    /// `x` must have exactly `n_input_features` 0/1 elements.
    pub fn predict(&self, x: &[u8]) -> usize {
        assert_eq!(x.len(), self.n_input_features);
        let mut patches_buf = vec![0u64; self.n_patches * self.words];
        self.pack_all_patches(x, &mut patches_buf);
        let mut scores = vec![0i32; self.n_classes];
        self.class_scores_from_patches(&patches_buf, &mut scores);
        scores
            .iter()
            .enumerate()
            .max_by_key(|&(_, &v)| v)
            .map(|(i, _)| i)
            .unwrap()
    }

    /// Fill `out` with the clamped weighted clause sums for each class.
    pub fn scores(&self, x: &[u8], out: &mut [i32]) {
        assert_eq!(x.len(), self.n_input_features);
        assert_eq!(out.len(), self.n_classes);
        let mut patches_buf = vec![0u64; self.n_patches * self.words];
        self.pack_all_patches(x, &mut patches_buf);
        self.class_scores_from_patches(&patches_buf, out);
    }

    /// Predict classes for a batch of raw input vectors.
    pub fn predict_batch(&self, xs: &[&[u8]]) -> Vec<usize> {
        xs.iter().map(|x| self.predict(x)).collect()
    }

    // ---- training ------------------------------------------------------------

    /// Compute the training-mode clause sum for class `c` over all patches.
    ///
    /// Matches TMU's `cb_calculate_clause_output_update`: OR semantics —
    /// a clause counts once in the sum if it fires on ANY patch position.
    fn class_sum_train(&self, c: usize, patches_buf: &[u64], lit_active: &[u64]) -> i32 {
        let cps = self.clauses_per_class;
        let words = self.words;
        let inc = self.include.as_slice();
        let val = self.valid.as_slice();
        let mut sum = 0i32;
        for j in 0..cps {
            let cj = c * cps + j;
            let clause_inc = &inc[cj * words..(cj + 1) * words];
            // OR semantics: fires if it fires on ANY patch.
            let fires = (0..self.n_patches).any(|p| {
                clause_fire(
                    clause_inc,
                    &patches_buf[p * words..(p + 1) * words],
                    val,
                    words,
                    lit_active,
                )
            });
            if fires {
                let w = self.weights[c * cps + j];
                if j & 1 == 0 {
                    sum += w;
                } else {
                    sum -= w;
                }
            }
        }
        sum.clamp(-self.threshold, self.threshold)
    }

    /// Apply feedback to all clauses of class `c`.
    ///
    /// Matches TMU's `cb_calculate_clause_output_feedback` + `cb_type_i/ii_feedback`:
    /// for each clause, first collect all patches where it fires (OR scan), then pick
    /// ONE of those firing patches for the TA update.  If no patch fires the clause
    /// receives only Type Ib feedback (which does not use patch features).
    fn update_class(
        &mut self,
        c: usize,
        target: u8,
        sum: i32,
        patches_buf: &[u64],
        lit_active: &[u64],
    ) {
        let cps = self.clauses_per_class;
        let words = self.words;
        let n_literals = self.n_literals;
        let boost = self.boost_true_positive;
        let wmax = self.threshold;
        let max_inc = self.max_included_literals;
        let drop_p = self.clause_drop_p;
        let half = self.half;
        let max_state = self.max_state;
        let n_patches = self.n_patches;
        let cw = self.class_weights_dummy();
        let type_iii_en = self.type_iii;
        let d_val = self.d;
        let target_bool = target != 0;

        let t = wmax as f64;
        let v = sum as f64;
        let p = if target == 1 {
            ((t - v) / (2.0 * t) * cw).min(1.0)
        } else {
            ((t + v) / (2.0 * t) * cw).min(1.0)
        };

        let Self {
            ta,
            include,
            weights,
            rngs,
            class_rngs,
            valid,
            dig_inv,
            dig_keep,
            rng: _,
            ind,
            cat,
            ..
        } = self;

        let crng = &mut class_rngs[c];

        let drop_mask: Vec<bool> = if drop_p > 0.0 {
            (0..cps).map(|_| crng.next_f64() < drop_p).collect()
        } else {
            vec![]
        };

        let inv_mask: Vec<u64> = (0..words).map(|_| bmask_word(crng, dig_inv)).collect();
        let keep_mask: Vec<u64> = (0..words).map(|_| bmask_word(crng, dig_keep)).collect();
        let inv_b = expand_bits_to_bytes(&inv_mask, n_literals);
        let keep_b = expand_bits_to_bytes(&keep_mask, n_literals);
        let active_b = expand_bits_to_bytes(lit_active, n_literals);

        let val = valid.as_slice();
        let class_ta = &mut ta[c * cps * n_literals..(c + 1) * cps * n_literals];
        let class_inc = &mut include[c * cps * words..(c + 1) * cps * words];
        let class_w = &mut weights[c * cps..(c + 1) * cps];
        let class_rng = &mut rngs[c * cps..(c + 1) * cps];
        let class_ind = &mut ind[c * cps * n_literals..(c + 1) * cps * n_literals];
        let class_cat = &mut cat[c * cps * words..(c + 1) * cps * words];

        #[cfg(feature = "parallel")]
        if cps >= DENSE_TRAIN_PARALLEL_MIN {
            use rayon::prelude::*;
            if type_iii_en {
                class_ta
                    .par_chunks_mut(n_literals)
                    .zip(class_inc.par_chunks_mut(words))
                    .zip(class_w.par_iter_mut())
                    .zip(class_rng.par_iter_mut())
                    .zip(class_ind.par_chunks_mut(n_literals))
                    .zip(class_cat.par_chunks_mut(words))
                    .enumerate()
                    .for_each(|(j, (((((ta_c, inc_c), w), rng_c), ind_c), cat_c))| {
                        let firing: Vec<usize> = (0..n_patches)
                            .filter(|&pp| {
                                clause_fire(
                                    inc_c,
                                    &patches_buf[pp * words..(pp + 1) * words],
                                    val,
                                    words,
                                    lit_active,
                                )
                            })
                            .collect();
                        let fires = !firing.is_empty();
                        let p_idx = if fires { firing[rng_c.below(firing.len())] } else { 0 };
                        let lit_b = expand_bits_to_bytes(
                            &patches_buf[p_idx * words..(p_idx + 1) * words],
                            n_literals,
                        );
                        apply_one_clause_conv(
                            j, ta_c, inc_c, w, rng_c, target, p, &drop_mask, val, words, &lit_b,
                            &inv_b, &keep_b, &active_b, n_literals, boost, wmax, max_inc, half,
                            max_state, fires,
                        );
                        if drop_mask.is_empty() || !drop_mask[j] {
                            let patch_lit = &patches_buf[p_idx * words..(p_idx + 1) * words];
                            if type_iii_update(
                                ta_c, ind_c, cat_c, inc_c, patch_lit, val, lit_active, &active_b, words,
                                n_literals, d_val, p, target_bool, rng_c, half, max_state,
                            ) {
                                rebuild_include(ta_c, inc_c, val, words, n_literals, half);
                            }
                        }
                    });
            } else {
                class_ta
                    .par_chunks_mut(n_literals)
                    .zip(class_inc.par_chunks_mut(words))
                    .zip(class_w.par_iter_mut())
                    .zip(class_rng.par_iter_mut())
                    .enumerate()
                    .for_each(|(j, (((ta_c, inc_c), w), rng_c))| {
                        let firing: Vec<usize> = (0..n_patches)
                            .filter(|&pp| {
                                clause_fire(
                                    inc_c,
                                    &patches_buf[pp * words..(pp + 1) * words],
                                    val,
                                    words,
                                    lit_active,
                                )
                            })
                            .collect();
                        let fires = !firing.is_empty();
                        let p_idx = if fires { firing[rng_c.below(firing.len())] } else { 0 };
                        let lit_b = expand_bits_to_bytes(
                            &patches_buf[p_idx * words..(p_idx + 1) * words],
                            n_literals,
                        );
                        apply_one_clause_conv(
                            j, ta_c, inc_c, w, rng_c, target, p, &drop_mask, val, words, &lit_b,
                            &inv_b, &keep_b, &active_b, n_literals, boost, wmax, max_inc, half,
                            max_state, fires,
                        );
                    });
            }
            return;
        }

        for j in 0..cps {
            // Find all patches where this clause fires (TMU: output_one_patches).
            let fires;
            let p_idx;
            {
                let clause_inc_j = &class_inc[j * words..(j + 1) * words];
                let firing: Vec<usize> = (0..n_patches)
                    .filter(|&pp| {
                        clause_fire(
                            clause_inc_j,
                            &patches_buf[pp * words..(pp + 1) * words],
                            val,
                            words,
                            lit_active,
                        )
                    })
                    .collect();
                fires = !firing.is_empty();
                p_idx = if fires { firing[class_rng[j].below(firing.len())] } else { 0 };
            }
            let lit_b =
                expand_bits_to_bytes(&patches_buf[p_idx * words..(p_idx + 1) * words], n_literals);
            apply_one_clause_conv(
                j,
                &mut class_ta[j * n_literals..(j + 1) * n_literals],
                &mut class_inc[j * words..(j + 1) * words],
                &mut class_w[j],
                &mut class_rng[j],
                target,
                p,
                &drop_mask,
                val,
                words,
                &lit_b,
                &inv_b,
                &keep_b,
                &active_b,
                n_literals,
                boost,
                wmax,
                max_inc,
                half,
                max_state,
                fires,
            );
            if type_iii_en && (drop_mask.is_empty() || !drop_mask[j]) {
                let patch_lit = &patches_buf[p_idx * words..(p_idx + 1) * words];
                if type_iii_update(
                    &mut class_ta[j * n_literals..(j + 1) * n_literals],
                    &mut class_ind[j * n_literals..(j + 1) * n_literals],
                    &mut class_cat[j * words..(j + 1) * words],
                    &class_inc[j * words..(j + 1) * words],
                    patch_lit,
                    val,
                    lit_active,
                    &active_b,
                    words,
                    n_literals,
                    d_val,
                    p,
                    target_bool,
                    &mut class_rng[j],
                    half,
                    max_state,
                ) {
                    rebuild_include(
                        &class_ta[j * n_literals..(j + 1) * n_literals],
                        &mut class_inc[j * words..(j + 1) * words],
                        val,
                        words,
                        n_literals,
                        half,
                    );
                }
            }
        }
    }

    fn class_weights_dummy(&self) -> f64 {
        1.0
    }

    /// Train on a single raw (unpacked) input vector with class label `y`.
    pub fn fit_one(&mut self, x: &[u8], y: usize) {
        assert_eq!(x.len(), self.n_input_features);
        assert!(y < self.n_classes);

        // Pack all patches once.
        let mut patches_buf = vec![0u64; self.n_patches * self.words];
        self.pack_all_patches(x, &mut patches_buf);

        // Sample a random negative class.
        let mut neg = self.rng.below(self.n_classes);
        while neg == y {
            neg = self.rng.below(self.n_classes);
        }

        // Generate per-sample literal-active mask (shared across patches and classes).
        let lit_active: Vec<u64> = if self.literal_drop_p > 0.0 {
            let rng = &mut self.literal_rng;
            let dig = &self.dig_lit_active;
            let w = self.words;
            (0..w).map(|_| bmask_word(rng, dig)).collect()
        } else {
            vec![!0u64; self.words]
        };

        let sum_y = self.class_sum_train(y, &patches_buf, &lit_active);
        let sum_neg = self.class_sum_train(neg, &patches_buf, &lit_active);

        self.update_class(y, 1, sum_y, &patches_buf, &lit_active);
        self.update_class(neg, 0, sum_neg, &patches_buf, &lit_active);
    }

    /// Run one training epoch over a dataset of raw inputs.
    ///
    /// `xs` — raw feature vectors (each must have `n_input_features` 0/1 elements).
    /// `ys` — class labels.
    pub fn fit_epoch(&mut self, xs: &[&[u8]], ys: &[usize]) {
        let n = xs.len();
        assert_eq!(n, ys.len());
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let k = self.rng.below(i + 1);
            order.swap(i, k);
        }
        for &i in &order {
            self.fit_one(xs[i], ys[i]);
        }
    }

    // ---- metrics -------------------------------------------------------------

    /// Fraction of correctly predicted samples.
    pub fn accuracy(&self, xs: &[&[u8]], ys: &[usize]) -> f64 {
        assert_eq!(xs.len(), ys.len());
        let correct = xs
            .iter()
            .zip(ys)
            .filter(|(&x, &y)| self.predict(x) == y)
            .count();
        correct as f64 / xs.len() as f64
    }

    // ---- interpretability ----------------------------------------------------

    /// Return the included literals for clause `clause` of `class`
    /// as `(patch_feature_index, is_negated)` pairs.
    ///
    /// Indices are relative to the patch (`0..patch_rows*patch_cols`).
    pub fn clause_rule(&self, class: usize, clause: usize) -> Vec<(usize, bool)> {
        let patch_features = self.patch_rows * self.patch_cols;
        let cj = class * self.clauses_per_class + clause;
        let inc = &self.include[cj * self.words..(cj + 1) * self.words];
        let mut rule = Vec::new();
        for l in 0..self.n_literals {
            if (inc[l / WORD_BITS] >> (l % WORD_BITS)) & 1 != 0 {
                if l < patch_features {
                    rule.push((l, false));
                } else {
                    rule.push((l - patch_features, true));
                }
            }
        }
        rule
    }

    /// Return `true` if `clause` is a positive clause (even index within class).
    pub fn clause_is_positive(&self, clause: usize) -> bool {
        clause & 1 == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_xor_sequence(n: usize, n_features: usize, seed: u64) -> (Vec<Vec<u8>>, Vec<usize>) {
        let mut rng = Rng::new(seed);
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for _ in 0..n {
            let f: Vec<u8> = (0..n_features)
                .map(|_| (rng.next_u64() & 1) as u8)
                .collect();
            let y = (f[0] ^ f[1]) as usize;
            ys.push(y);
            xs.push(f);
        }
        (xs, ys)
    }

    fn as_slices(xs: &[Vec<u8>]) -> Vec<&[u8]> {
        xs.iter().map(|v| v.as_slice()).collect()
    }

    #[test]
    fn convolutional_constructs() {
        let ctm = ConvolutionalTsetlinMachine::new(2, 16, 4, 2, 20, 100, 3.9);
        assert_eq!(ctm.n_classes(), 2);
        assert_eq!(ctm.kernel_size(), 4);
        assert_eq!(ctm.n_patches(), 7); // (16 - 4) / 2 + 1 = 7
    }

    #[test]
    fn convolutional_stride_1_n_patches() {
        let ctm = ConvolutionalTsetlinMachine::new(2, 10, 3, 1, 4, 10, 2.0);
        assert_eq!(ctm.n_patches(), 8); // (10 - 3) / 1 + 1 = 8
    }

    #[test]
    fn convolutional_full_kernel_one_patch() {
        let ctm = ConvolutionalTsetlinMachine::new(2, 8, 8, 1, 4, 10, 2.0);
        assert_eq!(ctm.n_patches(), 1);
    }

    #[test]
    fn convolutional_predict_returns_valid_class() {
        let ctm = ConvolutionalTsetlinMachine::new(3, 12, 4, 2, 8, 20, 3.0);
        let x = vec![0u8, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1];
        let pred = ctm.predict(&x);
        assert!(pred < 3);
    }

    #[test]
    fn convolutional_trains_without_panic() {
        let (xs, ys) = make_xor_sequence(200, 12, 1);
        let slices = as_slices(&xs);
        let mut ctm = ConvolutionalTsetlinMachine::new(2, 12, 4, 2, 10, 20, 3.0);
        for _ in 0..3 {
            ctm.fit_epoch(&slices, &ys);
        }
        let acc = ctm.accuracy(&slices, &ys);
        assert!((0.0..=1.0).contains(&acc));
    }

    #[test]
    fn convolutional_learns_xor_in_first_features() {
        // 4 features, kernel=2, stride=1 → 3 patches ([0,1], [1,2], [2,3]).
        // XOR of features 0,1 lives in patch 0; patches 1,2 add noise (2 features
        // irrelevant).  60 clauses and 40 epochs are sufficient to learn through
        // the signal dilution from the two irrelevant patches.
        let (xtr, ytr) = make_xor_sequence(3000, 4, 1);
        let (xte, yte) = make_xor_sequence(800, 4, 2);
        let tr = as_slices(&xtr);
        let te = as_slices(&xte);
        let mut ctm =
            ConvolutionalTsetlinMachine::with_config(2, 4, 2, 1, 60, 50, 3.5, 8, true, 42);
        for _ in 0..40 {
            ctm.fit_epoch(&tr, &ytr);
        }
        let acc = ctm.accuracy(&te, &yte);
        assert!(
            acc > 0.65,
            "accuracy {acc:.3} should exceed 65% for learnable XOR in 3-patch setup"
        );
    }

    #[test]
    fn convolutional_clause_rule_feature_indices_in_range() {
        let ctm = ConvolutionalTsetlinMachine::new(2, 12, 4, 2, 8, 20, 3.0);
        for rule_feat in ctm.clause_rule(0, 0).iter().map(|&(f, _)| f) {
            assert!(
                rule_feat < ctm.kernel_size(),
                "feature index out of patch range"
            );
        }
    }

    #[test]
    fn convolutional_scores_argmax_matches_predict() {
        let (xs, ys) = make_xor_sequence(200, 8, 3);
        let slices = as_slices(&xs);
        let mut ctm = ConvolutionalTsetlinMachine::new(2, 8, 2, 1, 10, 20, 3.0);
        for _ in 0..5 {
            ctm.fit_epoch(&slices, &ys);
        }
        let mut out = vec![0i32; 2];
        for x in &xs[..20] {
            ctm.scores(x, &mut out);
            let argmax = out
                .iter()
                .enumerate()
                .max_by_key(|&(_, &v)| v)
                .map(|(i, _)| i)
                .unwrap();
            assert_eq!(argmax, ctm.predict(x));
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn convolutional_save_load_roundtrip() {
        use crate::serial::SaveLoad;
        let (xs, ys) = make_xor_sequence(300, 8, 5);
        let slices = as_slices(&xs);
        let mut ctm = ConvolutionalTsetlinMachine::new(2, 8, 2, 1, 10, 20, 3.0);
        for _ in 0..5 {
            ctm.fit_epoch(&slices, &ys);
        }
        let tmp = std::env::temp_dir().join("test_ctm.tmrs");
        ctm.save(&tmp).unwrap();
        let loaded = ConvolutionalTsetlinMachine::load(&tmp).unwrap();
        let _ = std::fs::remove_file(&tmp);
        for x in &xs[..50] {
            assert_eq!(ctm.predict(x), loaded.predict(x));
        }
    }

    #[test]
    fn type_iii_constructs_without_panic() {
        let _ctm = ConvolutionalTsetlinMachine::new(2, 8, 2, 1, 10, 20, 3.0)
            .type_iii_feedback(200.0);
    }

    #[test]
    fn type_iii_d_must_be_greater_than_one() {
        let result = std::panic::catch_unwind(|| {
            ConvolutionalTsetlinMachine::new(2, 8, 2, 1, 10, 20, 3.0).type_iii_feedback(0.5)
        });
        assert!(result.is_err(), "expected panic for d <= 1.0");
    }

    #[test]
    fn type_iii_trains_without_panic() {
        let (xs, ys) = make_xor_sequence(300, 8, 7);
        let slices = as_slices(&xs);
        let mut ctm = ConvolutionalTsetlinMachine::new(2, 8, 2, 1, 10, 20, 3.0)
            .type_iii_feedback(200.0);
        for _ in 0..5 {
            ctm.fit_epoch(&slices, &ys);
        }
    }
}
