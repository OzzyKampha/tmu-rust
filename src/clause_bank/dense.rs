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

#[inline(always)]
pub(crate) fn words_for(bits: usize) -> usize {
    bits.div_ceil(WORD_BITS)
}

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

// Branchless inner loop — no early exit so the compiler can auto-vectorize
// (AVX2: 4 u64/cycle for MNIST's 25 words ≈ 4× over scalar).
// Empty clauses (included == 0) return false, matching TMU predict semantics.
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
/// `chunk` is the full `state_bits * words` slice for one clause; the top
/// bit-plane (offset `(sb-1)*words`) is the include bitset.
#[inline(always)]
pub(crate) fn clause_fire(
    chunk: &[u64],
    lit: &[u64],
    valid: &[u64],
    words: usize,
    sb: usize,
) -> bool {
    let tb = (sb - 1) * words;
    for k in 0..words {
        let inc = chunk[tb + k] & valid[k];
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
        if carry == 0 {
            return;
        }
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
        if borrow == 0 {
            return;
        }
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
/// * Ia path (fires && under literal limit): weight++, include literals present,
///   exclude literals absent (with probability masks).
/// * Ib path (doesn't fire, or at/over limit): exclude literals absent only.
#[allow(clippy::too_many_arguments)]
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
) {
    let out = clause_fire(chunk, lit, valid, words, sb);
    let tb = (sb - 1) * words;
    let under_limit = max_included == usize::MAX || {
        let n: u32 = (0..words).map(|k| (chunk[tb + k] & valid[k]).count_ones()).sum();
        (n as usize) < max_included
    };
    if out && under_limit {
        *weight = (*weight + 1).min(wmax);
        for k in 0..words {
            let litw = lit[k];
            let inc_mask = if boost {
                litw & valid[k]
            } else {
                litw & keep_mask[k] & valid[k]
            };
            clause_inc(chunk, words, sb, k, inc_mask);
            clause_dec(chunk, words, sb, k, !litw & inv_mask[k] & valid[k]);
        }
    } else {
        for k in 0..words {
            clause_dec(chunk, words, sb, k, inv_mask[k] & valid[k]);
        }
    }
}

/// Type II feedback for one clause: fires → weight--, include absent literals.
pub(crate) fn clause_type_ii(
    chunk: &mut [u64],
    weight: &mut i32,
    lit: &[u64],
    valid: &[u64],
    words: usize,
    sb: usize,
) {
    if !clause_fire(chunk, lit, valid, words, sb) {
        return;
    }
    *weight = (*weight - 1).max(1);
    let tb = (sb - 1) * words;
    for k in 0..words {
        let excluded = !chunk[tb + k];
        let inc_mask = !lit[k] & excluded & valid[k];
        clause_inc(chunk, words, sb, k, inc_mask);
    }
}
