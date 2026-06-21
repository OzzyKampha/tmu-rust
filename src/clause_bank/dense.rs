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
#[inline]
pub fn pack(x: &[u8], n_features: usize, out: &mut [u64]) {
    for w in out.iter_mut() {
        *w = 0;
    }
    for i in 0..n_features {
        if x[i] != 0 {
            out[i / WORD_BITS] |= 1u64 << (i % WORD_BITS);
        } else {
            let j = n_features + i;
            out[j / WORD_BITS] |= 1u64 << (j % WORD_BITS);
        }
    }
}

/// Evaluate whether clause `j` fires for inference — branchless so the compiler can auto-vectorize.
///
/// Scans all `words` words of the include bitset at `state[tb..]` against `lit`.
/// Returns `false` for empty clauses (no included literals), matching TMU predict semantics.
#[inline(always)]
pub(crate) fn fire_predict(
    state: &[u64],
    tb: usize,
    lit: &[u64],
    valid: &[u64],
    words: usize,
) -> bool {
    let mut violation = 0u64;
    let mut included = 0u64;
    for k in 0..words {
        let inc = state[tb + k] & valid[k];
        violation |= inc & !lit[k];
        included |= inc;
    }
    violation == 0 && included != 0
}

/// Firing check for training — early exit on first violation.
///
/// `chunk` is the full `state_bits * words` slice for one clause; the top
/// bit-plane (offset `(sb-1)*words`) is the include bitset.
/// `lit_active` masks which literals participate: inactive bits are treated as
/// absent from the clause, so they cannot cause a violation.  Pass all-ones
/// (`&[!0u64; words]`) when no literal dropout is in effect.
///
/// Mirrors TMU's `cb_calculate_clause_output_feedback` use of `literal_active`.
#[inline(always)]
pub(crate) fn clause_fire(
    chunk: &[u64],
    lit: &[u64],
    valid: &[u64],
    words: usize,
    sb: usize,
    lit_active: &[u64],
) -> bool {
    let tb = (sb - 1) * words;
    for k in 0..words {
        // Only active included literals can cause a violation.
        let inc = chunk[tb + k] & valid[k] & lit_active[k];
        if inc & !lit[k] != 0 {
            return false;
        }
    }
    true
}

/// Saturating ripple-carry increment on bit-plane word `k` of `chunk`.
#[inline(always)]
pub(crate) fn clause_inc(chunk: &mut [u64], words: usize, sb: usize, k: usize, mask: u64) {
    if mask == 0 {
        return;
    }
    let mut carry = mask;
    for b in 0..sb {
        let idx = b * words + k;
        let next = chunk[idx] & carry;
        chunk[idx] ^= carry;
        carry = next;
    }
    // Overflow: saturate to all-ones for the overflowed bits.
    for b in 0..sb {
        chunk[b * words + k] |= carry;
    }
}

/// Saturating ripple-borrow decrement on bit-plane word `k` of `chunk`.
#[inline(always)]
pub(crate) fn clause_dec(chunk: &mut [u64], words: usize, sb: usize, k: usize, mask: u64) {
    if mask == 0 {
        return;
    }
    let mut borrow = mask;
    for b in 0..sb {
        let idx = b * words + k;
        let next = !chunk[idx] & borrow;
        chunk[idx] ^= borrow;
        borrow = next;
    }
    // Underflow: saturate to all-zeros for the underflowed bits.
    let clear = !borrow;
    for b in 0..sb {
        chunk[b * words + k] &= clear;
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

/// Type Ia / Ib feedback for one clause.
///
/// * Ia path (fires && under literal limit): weight++, include active literals
///   present, exclude active literals absent (with probability masks).
/// * Ib path (doesn't fire, or at/over limit): exclude active literals absent.
///
/// **Absorbing include state**: literals whose TA counter is at the maximum value
/// (all `sb` state-plane bits set) are immune to decrement feedback on both paths.
/// Mirrors TMU's `ClauseBank.c` absorbing mask.
///
/// The absorbing mask for word `k` is computed inline immediately before
/// `clause_inc`/`clause_dec` touch word `k`, so the state planes are
/// already in L1 cache — the same "nearly free as a side-effect of the
/// state read" property as TMU's C implementation.
///
/// `lit_active` is the per-sample literal dropout mask (Bernoulli(1-p) per bit).
/// Pass all-ones when `literal_drop_p == 0`.
///
/// Note: `max_included` counts **all** included literals (not just active ones),
/// matching TMU's `cb_number_of_include_actions` check.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn clause_type_i(
    chunk: &mut [u64],
    weight: &mut i32,
    lit: &[u64],
    valid: &[u64],
    words: usize,
    sb: usize,
    boost: bool,
    inv_mask: &[u64],
    keep_mask: &[u64],
    wmax: i32,
    max_included: usize,
    lit_active: &[u64],
) {
    let out = clause_fire(chunk, lit, valid, words, sb, lit_active);
    let tb = (sb - 1) * words;
    // Include count uses all valid literals regardless of dropout — matches TMU.
    let under_limit = max_included == usize::MAX || {
        let n: u32 = (0..words).map(|k| (chunk[tb + k] & valid[k]).count_ones()).sum();
        (n as usize) < max_included
    };

    if out && under_limit {
        *weight = (*weight + 1).min(wmax);
        for k in 0..words {
            let litw = lit[k];
            let la = lit_active[k];
            // AND all planes — state is cache-hot from clause_fire above.
            let at_max_k = (0..sb).fold(!0u64, |acc, b| acc & chunk[b * words + k]);
            let inc_mask = if boost {
                litw & valid[k] & la
            } else {
                litw & keep_mask[k] & valid[k] & la
            };
            clause_inc(chunk, words, sb, k, inc_mask);
            clause_dec(chunk, words, sb, k, !litw & inv_mask[k] & valid[k] & la & !at_max_k);
        }
    } else {
        for k in 0..words {
            let at_max_k = (0..sb).fold(!0u64, |acc, b| acc & chunk[b * words + k]);
            clause_dec(chunk, words, sb, k, inv_mask[k] & valid[k] & lit_active[k] & !at_max_k);
        }
    }
}

/// Type II feedback for one clause: fires → weight--, include absent active literals.
///
/// **Absorbing exclude state**: literals whose TA counter is at the minimum value
/// (all `sb` state-plane bits clear) are immune to increment feedback.
///
/// `not_at_min_k` is OR-reduced inline over all planes, cache-hot from the
/// preceding `clause_fire` call — same "nearly free" property as TMU.
///
/// `lit_active` is the per-sample literal dropout mask.
#[inline(always)]
pub(crate) fn clause_type_ii(
    chunk: &mut [u64],
    weight: &mut i32,
    lit: &[u64],
    valid: &[u64],
    words: usize,
    sb: usize,
    lit_active: &[u64],
) {
    if !clause_fire(chunk, lit, valid, words, sb, lit_active) {
        return;
    }
    *weight = (*weight - 1).max(1);

    let tb = (sb - 1) * words;
    for k in 0..words {
        // OR all planes — state is cache-hot from clause_fire above.
        let not_at_min_k = (0..sb).fold(0u64, |acc, b| acc | chunk[b * words + k]);
        let excluded = !chunk[tb + k];
        let inc_mask = !lit[k] & excluded & valid[k] & lit_active[k] & not_at_min_k;
        clause_inc(chunk, words, sb, k, inc_mask);
    }
}
