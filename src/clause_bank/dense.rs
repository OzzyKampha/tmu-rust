//! Bit-level primitives shared by all Tsetlin Machine variants.
//!
//! Mirrors TMU's `clause_bank_dense.py` — stateless free functions that
//! operate on raw slice views of TA state and literal bits.  Separating them
//! here lets future variants (coalesced, sparse, convolutional) reuse the same
//! building blocks without duplicating bit-manipulation logic.

use crate::rng::Rng;

pub(crate) const WORD_BITS: usize = 64;
/// Precision (bits) of packed Bernoulli feedback masks; probability error ≤ 2^-MASK_BITS.
pub(crate) const MASK_BITS: usize = 12;
pub(crate) const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
/// Minimum item count before rayon parallelism pays off over its dispatch overhead.
#[cfg(feature = "parallel")]
pub(crate) const PARALLEL_MIN: usize = 128;

/// Return the minimum number of 64-bit words needed to hold `bits` bits.
#[inline(always)]
pub(crate) fn words_for(bits: usize) -> usize {
    bits.div_ceil(WORD_BITS)
}

/// Encode probability `p` as the first `n` bits of its binary (base-2) expansion.
/// Used to build Bernoulli sampling masks with precision MASK_BITS.
pub(crate) fn digits_of(p: f64, n: usize) -> Vec<u8> {
    let mut d = Vec::with_capacity(n);
    let mut x = p;
    for _ in 0..n {
        x *= 2.0;
        d.push(if x >= 1.0 {
            x -= 1.0;
            1u8
        } else {
            0u8
        });
    }
    d
}

/// Pack a raw feature vector into the bit-interleaved literal representation.
///
/// Literal `i` (positive) is bit `i`; literal `n_features + i` (negated) is
/// bit `n_features + i`.  The two halves of each feature are complementary.
/// Branchless form: both positive and negated bits are always written (one is 0),
/// avoiding a branch-per-feature misprediction and enabling LLVM to auto-vectorize.
#[inline]
pub fn pack(x: &[u8], n_features: usize, out: &mut [u64]) {
    for w in out.iter_mut() {
        *w = 0;
    }
    for i in 0..n_features {
        let pos = (x[i] != 0) as u64;
        out[i / WORD_BITS] |= pos << (i % WORD_BITS);
        let j = n_features + i;
        out[j / WORD_BITS] |= (1 - pos) << (j % WORD_BITS);
    }
}

/// Expand the first `n` bits of a packed bit-array into a byte-per-element Vec (0 or 1).
///
/// Called once per sample/class-update to produce SIMD-friendly byte arrays for the
/// hot TA update loops, where bit extraction per iteration prevents auto-vectorization.
#[inline]
pub(crate) fn expand_bits_to_bytes(bits: &[u64], n: usize) -> Vec<u8> {
    (0..n)
        .map(|l| ((bits[l / WORD_BITS] >> (l % WORD_BITS)) & 1) as u8)
        .collect()
}

/// Evaluate whether a clause fires for inference.
///
/// `inc` is the clause include bitset (one u64 word per 64 literals).
/// Returns `false` for empty clauses (no included literals), matching TMU predict semantics.
#[inline(always)]
pub(crate) fn fire_predict(inc: &[u64], lit: &[u64], valid: &[u64], words: usize) -> bool {
    let mut violation = 0u64;
    let mut included = 0u64;
    for k in 0..words {
        let inc_k = inc[k] & valid[k];
        violation |= inc_k & !lit[k];
        included |= inc_k;
    }
    violation == 0 && included != 0
}

/// Fire check for training — returns `true` if no active included literal is violated.
///
/// Unlike `fire_predict`, an empty clause (no active included literals) returns `true`,
/// matching TMU's `cb_calculate_clause_output_feedback` semantics.
/// Pass all-ones (`&[!0u64; words]`) when no literal dropout is in effect.
#[inline(always)]
pub(crate) fn clause_fire(
    inc: &[u64],
    lit: &[u64],
    valid: &[u64],
    words: usize,
    lit_active: &[u64],
) -> bool {
    let mut violation = 0u64;
    for k in 0..words {
        violation |= inc[k] & valid[k] & lit_active[k] & !lit[k];
    }
    violation == 0
}

/// Rebuild the clause include bitset from u32 TA counters: `ta[l] >= half` → included.
///
/// Called after every clause update to keep `inc` in sync with `ta`.
/// Branchless: cast-to-u64 + shift avoids branch-per-TA and gives LLVM the best
/// chance to emit packed compare instructions.  Further vectorisation (movemask) would
/// require an unsafe `_mm256_movemask_ps` intrinsic — out of scope for now.
#[inline]
pub(crate) fn rebuild_include(
    ta: &[u32],
    inc: &mut [u64],
    valid: &[u64],
    words: usize,
    n_literals: usize,
    half: u32,
) {
    for k in 0..words {
        let base = k * WORD_BITS;
        let limit = (n_literals - base).min(WORD_BITS);
        let mut word = 0u64;
        for bit in 0..limit {
            word |= ((ta[base + bit] >= half) as u64) << bit;
        }
        inc[k] = word & valid[k];
    }
}

/// Generate one 64-bit packed Bernoulli sample with probability encoded in
/// `digits` (fixed-point binary expansion, length = MASK_BITS).
#[inline(always)]
pub(crate) fn bmask_word(rng: &mut Rng, digits: &[u8]) -> u64 {
    let mut word = 0u64;
    for i in (0..digits.len()).rev() {
        let r = rng.next_u64();
        word = if digits[i] == 1 { r | word } else { r & word };
    }
    word
}

/// Branchless Type Ia / Ib TA update using byte-expanded inputs.
///
/// `active_b[l]` encodes valid & lit_active for literal `l` (0 = skip, 1 = update).
/// `lit_b[l]` is 1 if literal `l` is present in the current sample, 0 if absent.
/// `inv_b[l]` / `keep_b[l]` are Bernoulli feedback masks (0 or 1 per literal).
///
/// This loop is fully branchless — LLVM auto-vectorizes it with AVX2 `vpminud` /
/// `vpsubd` when `target-cpu=native` is set.
///
/// `fired_under`: true → Ia path (weight already incremented by caller); false → Ib path.
#[inline(always)]
pub(crate) fn type_i_update_bytes(
    ta: &mut [u32],
    n_literals: usize,
    fired_under: bool,
    boost: bool,
    lit_b: &[u8],
    inv_b: &[u8],
    keep_b: &[u8],
    active_b: &[u8],
    max_state: u32,
) {
    let boost_u32 = boost as u32;
    if fired_under {
        // Ia: present literals → try increment; absent literals → try decrement.
        for l in 0..n_literals {
            let la = active_b[l] as u32;
            let t = ta[l];
            let present = lit_b[l] as u32;
            let inc = present & (boost_u32 | keep_b[l] as u32) & la;
            let not_at_max = (t < max_state) as u32;
            let dec = (1 - present) & inv_b[l] as u32 & not_at_max & la;
            ta[l] = (t + inc).min(max_state).saturating_sub(dec);
        }
    } else {
        // Ib: all literals → try decrement (push toward exclusion).
        for l in 0..n_literals {
            let la = active_b[l] as u32;
            let t = ta[l];
            let not_at_max = (t < max_state) as u32;
            let dec = inv_b[l] as u32 & not_at_max & la;
            ta[l] = t.saturating_sub(dec);
        }
    }
}

/// Branchless Type II TA update using byte-expanded inputs.
///
/// Increments excluded absent non-absorbing active literals — pushes the clause
/// toward including features that weren't present (making it harder to fire on
/// negative-class samples).
#[inline(always)]
pub(crate) fn type_ii_update_bytes(
    ta: &mut [u32],
    n_literals: usize,
    lit_b: &[u8],
    active_b: &[u8],
    half: u32,
    max_state: u32,
) {
    for l in 0..n_literals {
        let la = active_b[l] as u32;
        let t = ta[l];
        let absent = 1 - lit_b[l] as u32;    // 1 if feature absent, 0 if present
        let excluded = (t < half) as u32;     // 1 if TA is below include threshold
        let not_zero = (t > 0) as u32;        // 0 at absorbing exclude state
        let inc = absent & excluded & not_zero & la;
        ta[l] = (t + inc).min(max_state);
    }
}

/// Type Ia / Ib feedback for one clause — u32 per-TA encoding (matches TMU C extension).
/// Used directly by unit tests; production code calls `type_i_update_bytes` with pre-expanded arrays.
///
/// * Ia path (fires && under literal limit): weight++, include active present literals,
///   exclude active absent literals (with probability masks).
/// * Ib path (doesn't fire, or at/over limit): exclude active absent literals.
///
/// **Absorbing include state**: literals whose TA counter equals `max_state` are immune
/// to decrement feedback on both paths.
///
/// `lit_active` is the per-sample literal dropout mask (Bernoulli(1-p) per bit).
/// Pass all-ones when `literal_drop_p == 0`.
///
/// Note: `max_included` counts **all** included literals (not just active ones),
/// matching TMU's `cb_number_of_include_actions` check.
///
/// This function expands the packed bit inputs to bytes before calling
/// `type_i_update_bytes`.  For the hot training path, call `type_i_update_bytes`
/// directly with pre-expanded arrays (amortised over all clauses in an epoch).
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn clause_type_i_bytes(
    ta: &mut [u32],
    inc: &mut [u64],
    weight: &mut i32,
    lit: &[u64],
    valid: &[u64],
    words: usize,
    n_literals: usize,
    boost: bool,
    inv_mask: &[u64],
    keep_mask: &[u64],
    wmax: i32,
    max_included: usize,
    lit_active: &[u64],
    half: u32,
    max_state: u32,
) {
    let fired = clause_fire(inc, lit, valid, words, lit_active);
    // Include count uses all valid literals regardless of dropout — matches TMU.
    let under_limit = max_included == usize::MAX || {
        let n: u32 = (0..words).map(|k| (inc[k] & valid[k]).count_ones()).sum();
        (n as usize) < max_included
    };

    let lit_b = expand_bits_to_bytes(lit, n_literals);
    let inv_b = expand_bits_to_bytes(inv_mask, n_literals);
    let keep_b = expand_bits_to_bytes(keep_mask, n_literals);
    // Combine valid and lit_active into a single active mask.
    let active_b: Vec<u8> = (0..n_literals)
        .map(|l| {
            let k = l / WORD_BITS;
            let shift = l % WORD_BITS;
            let v = ((valid[k] >> shift) & 1) as u8;
            let la = ((lit_active[k] >> shift) & 1) as u8;
            v & la
        })
        .collect();

    if fired && under_limit {
        *weight = (*weight + 1).min(wmax);
    }
    type_i_update_bytes(ta, n_literals, fired && under_limit, boost, &lit_b, &inv_b, &keep_b, &active_b, max_state);
    rebuild_include(ta, inc, valid, words, n_literals, half);
}

/// Type II feedback for one clause — u32 per-TA encoding.
///
/// fires → weight--, include absent active excluded literals.
///
/// **Absorbing exclude state**: literals whose TA counter is 0 are immune to
/// increment feedback.
///
/// This function expands the packed bit inputs to bytes before calling
/// `type_ii_update_bytes`.  For the hot training path, call `type_ii_update_bytes`
/// directly with pre-expanded arrays.
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn clause_type_ii_bytes(
    ta: &mut [u32],
    inc: &mut [u64],
    weight: &mut i32,
    lit: &[u64],
    valid: &[u64],
    words: usize,
    n_literals: usize,
    lit_active: &[u64],
    half: u32,
    max_state: u32,
) {
    if !clause_fire(inc, lit, valid, words, lit_active) {
        return;
    }
    *weight = (*weight - 1).max(1);

    let lit_b = expand_bits_to_bytes(lit, n_literals);
    let active_b: Vec<u8> = (0..n_literals)
        .map(|l| {
            let k = l / WORD_BITS;
            let shift = l % WORD_BITS;
            let v = ((valid[k] >> shift) & 1) as u8;
            let la = ((lit_active[k] >> shift) & 1) as u8;
            v & la
        })
        .collect();

    type_ii_update_bytes(ta, n_literals, &lit_b, &active_b, half, max_state);
    rebuild_include(ta, inc, valid, words, n_literals, half);
}
