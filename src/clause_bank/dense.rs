//! Bit-level primitives shared by all Tsetlin Machine variants.
//!
//! Mirrors TMU's `clause_bank_dense.py` — stateless free functions that operate
//! on raw slice views of TA state and literal bits.  Separating them here lets
//! future variants (coalesced, sparse, convolutional) reuse the same building
//! blocks without duplicating bit-manipulation logic.
//!
//! ## TA counter storage
//!
//! TA counters are stored as `u8` (one byte per literal per clause), matching TMU's
//! C extension default of 8-bit states.  This keeps the `ta` array 4× smaller than a
//! `u32` layout (so a 10 000-clause model fits in L3 cache) and lets AVX2 process
//! **32 counters per register** with native saturating arithmetic
//! (`vpaddusb` / `vpsubusb` / `vpminub`).  `state_bits` is therefore capped at 2..=8.
//!
//! ## SIMD dispatch pattern
//!
//! The three hot update functions ([`rebuild_include`], [`type_i_update_bytes`],
//! [`type_ii_update_bytes`]) each follow the same structure:
//!
//! 1. A public `#[inline]` dispatcher checks `is_x86_feature_detected!("avx2")`
//!    at runtime (constant `true` on `target-cpu=native` builds — zero overhead).
//! 2. On AVX2 targets the `#[target_feature(enable = "avx2")]` unsafe inner
//!    function is called; on other targets the branchless scalar fallback runs.
//! 3. All unsafe pointers are read/write within slice bounds verified by the
//!    surrounding loop guard — no alignment requirements (unaligned loads/stores).
//!
//! AVX2 requires no additional feature flags beyond `target-cpu=native`; the
//! scalar fallbacks are correct and fast on any target.

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
/// Bit layout: positive literal `i` → bit `i`; negated literal `i` → bit
/// `n_features + i`.  Each feature's positive and negated bits are complementary.
///
/// The loop is branchless — both positive and negated bits are always written
/// (one is zero) to avoid branch mispredictions and allow LLVM to auto-vectorize.
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

/// Expand the first `n` bits of a packed bit-array into a `Vec<u8>` of 0/1 values.
///
/// Called once per sample/class-update to produce flat byte arrays for the hot AVX2
/// update loops.  Loading from four separate byte arrays with `_mm256_cvtepu8_epi32`
/// is faster than extracting bits inside the inner loop (variable shifts block
/// auto-vectorization).
#[inline]
pub(crate) fn expand_bits_to_bytes(bits: &[u64], n: usize) -> Vec<u8> {
    (0..n)
        .map(|l| ((bits[l / WORD_BITS] >> (l % WORD_BITS)) & 1) as u8)
        .collect()
}

/// Returns `true` if a clause fires for inference (no included literal is violated).
///
/// `inc` is the clause include bitset (`words` u64s, 1 bit per literal).
/// Empty clauses (no included literals) return `false`, matching TMU predict semantics.
/// Use [`clause_fire`] during training where empty clauses must return `true`.
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

/// Returns `true` if a clause fires during training (no active included literal is violated).
///
/// Unlike [`fire_predict`], an empty clause returns `true` — matching TMU's
/// `cb_calculate_clause_output_feedback` semantics so that clauses with no active
/// included literals still receive Type Ib feedback.
/// Pass `&[!0u64; words]` for `lit_active` when literal dropout is disabled.
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

/// Recompute the include bitset from TA counters after a clause update.
///
/// A literal `l` is included when `ta[l] >= half`.  The result is ANDed with `valid`
/// so padding bits beyond `n_literals` are always zero.
///
/// **AVX2 path** (x86_64): processes 32 u8s per iteration.  Each lane is XOR-biased
/// by `0x80` to convert unsigned `>=` to signed `>`, then `_mm256_cmpgt_epi8`
/// produces a SIMD mask and `_mm256_movemask_epi8` collapses 32 lanes → 32 bits in
/// one instruction.  The scalar tail handles the remaining `< 32` values.
///
/// **Scalar fallback**: branchless `(ta[l] >= half) as u64 << bit` loop.
#[inline]
pub(crate) fn rebuild_include(
    ta: &[u8],
    inc: &mut [u64],
    valid: &[u64],
    words: usize,
    n_literals: usize,
    half: u8,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 presence verified by the runtime check above.
            return unsafe { rebuild_include_avx2(ta, inc, valid, words, n_literals, half) };
        }
    }
    rebuild_include_scalar(ta, inc, valid, words, n_literals, half);
}

/// Branchless scalar fallback for [`rebuild_include`] on non-AVX2 targets.
#[inline]
fn rebuild_include_scalar(
    ta: &[u8],
    inc: &mut [u64],
    valid: &[u64],
    words: usize,
    n_literals: usize,
    half: u8,
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

/// AVX2 implementation of [`rebuild_include`]: 32 comparisons → 32 bits per iteration.
///
/// # Safety
/// Caller must verify AVX2 is available.  All pointer reads stay within the `ta`
/// slice (loop guard: `bit + 32 <= limit <= n_literals - base`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rebuild_include_avx2(
    ta: &[u8],
    inc: &mut [u64],
    valid: &[u64],
    words: usize,
    n_literals: usize,
    half: u8,
) {
    use std::arch::x86_64::*;

    // Bias both sides by 0x80 to convert unsigned >= to signed >.
    // ta >= half  ↔  (ta ^ 0x80) > ((half-1) ^ 0x80)  [signed i8]
    let bias = _mm256_set1_epi8(i8::MIN);
    let threshold = _mm256_set1_epi8((half.wrapping_sub(1) ^ 0x80) as i8);

    for k in 0..words {
        let base = k * WORD_BITS;
        let limit = (n_literals - base).min(WORD_BITS);
        let mut word = 0u64;
        let mut bit = 0usize;

        // 32 literals per AVX2 iteration.
        while bit + 32 <= limit {
            // SAFETY: `bit + 32 <= limit <= n_literals - base`, so `base + bit + 32 <= n_literals`
            // which is within the `ta` slice. `_mm256_loadu_si256` is an unaligned load.
            let ptr = ta.as_ptr().add(base + bit) as *const __m256i;
            let chunk = _mm256_loadu_si256(ptr);
            let biased = _mm256_xor_si256(chunk, bias);
            let cmp = _mm256_cmpgt_epi8(biased, threshold);
            let bits32 = _mm256_movemask_epi8(cmp) as u32 as u64;
            word |= bits32 << bit;
            bit += 32;
        }
        // Scalar tail for remaining < 32 literals.
        while bit < limit {
            word |= ((ta[base + bit] >= half) as u64) << bit;
            bit += 1;
        }

        inc[k] = word & valid[k];
    }
}

/// Generate one 64-bit Bernoulli sample mask from a fixed-point probability.
///
/// `digits` holds the base-2 expansion of the probability `p` (length = `MASK_BITS`).
/// Returns a u64 where each bit is 1 with probability `p`, independently.
#[inline(always)]
pub(crate) fn bmask_word(rng: &mut Rng, digits: &[u8]) -> u64 {
    let mut word = 0u64;
    for i in (0..digits.len()).rev() {
        let r = rng.next_u64();
        word = if digits[i] == 1 { r | word } else { r & word };
    }
    word
}

/// Apply Type I TA feedback to one clause (byte-array inputs).
///
/// All four input byte arrays have one element per literal (`n_literals` elements each),
/// where 0 means "skip this literal" and 1 means "apply feedback":
///
/// - `lit_b`: 1 if the literal is present in the current sample, 0 if absent.
/// - `inv_b`: Bernoulli feedback mask for the decrement (1/s probability).
/// - `keep_b`: Bernoulli feedback mask for the increment boost (1 - 1/s probability).
/// - `active_b`: combined valid × literal-dropout mask; 0 skips the literal entirely.
///
/// `fired_under = true` → **Type Ia** (clause fired and is under the include limit):
/// present active literals are incremented (capped at `max_state`); absent active
/// literals are decremented with probability `inv_b`, unless already at `max_state`.
///
/// `fired_under = false` → **Type Ib** (clause did not fire, or is at/over the limit):
/// absent active literals are decremented; present literals are untouched.
///
/// Weight bookkeeping is the caller's responsibility; call this function after updating
/// the weight.
///
/// **AVX2 path**: 32 literals/iteration using native saturating byte arithmetic
/// (`_mm256_adds_epu8` / `_mm256_subs_epu8`) and `_mm256_min_epu8` clamping.
///
/// **Scalar fallback**: branchless per-literal arithmetic.
#[inline(always)]
pub(crate) fn type_i_update_bytes(
    ta: &mut [u8],
    n_literals: usize,
    fired_under: bool,
    boost: bool,
    lit_b: &[u8],
    inv_b: &[u8],
    keep_b: &[u8],
    active_b: &[u8],
    max_state: u8,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 presence verified by the runtime check above.
            return unsafe {
                type_i_update_bytes_avx2(
                    ta, n_literals, fired_under, boost, lit_b, inv_b, keep_b, active_b,
                    max_state,
                )
            };
        }
    }
    type_i_update_bytes_scalar(ta, n_literals, fired_under, boost, lit_b, inv_b, keep_b, active_b, max_state);
}

/// Branchless scalar fallback for [`type_i_update_bytes`] on non-AVX2 targets.
#[inline]
fn type_i_update_bytes_scalar(
    ta: &mut [u8],
    n_literals: usize,
    fired_under: bool,
    boost: bool,
    lit_b: &[u8],
    inv_b: &[u8],
    keep_b: &[u8],
    active_b: &[u8],
    max_state: u8,
) {
    let boost_u8 = boost as u8;
    if fired_under {
        for l in 0..n_literals {
            let la = active_b[l];
            let t = ta[l];
            let present = lit_b[l];
            let inc = present & (boost_u8 | keep_b[l]) & la;
            let not_at_max = (t < max_state) as u8;
            let dec = (1 - present) & inv_b[l] & not_at_max & la;
            ta[l] = t.saturating_add(inc).min(max_state).saturating_sub(dec);
        }
    } else {
        for l in 0..n_literals {
            let la = active_b[l];
            let t = ta[l];
            let not_at_max = (t < max_state) as u8;
            let dec = inv_b[l] & not_at_max & la;
            ta[l] = t.saturating_sub(dec);
        }
    }
}

/// AVX2 implementation of [`type_i_update_bytes`]: 32 literals per iteration.
///
/// All inputs are already 0/1 bytes, so no widening is needed.  Increment is
/// `_mm256_adds_epu8` then `_mm256_min_epu8` (caps at `max_state`); decrement is
/// the native `_mm256_subs_epu8` (saturating unsigned byte subtract).
///
/// # Safety
/// Caller must verify AVX2 is available.  Pointer reads/writes stay within the
/// `ta` slice (loop guard: `l + 32 <= n_literals`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn type_i_update_bytes_avx2(
    ta: &mut [u8],
    n_literals: usize,
    fired_under: bool,
    boost: bool,
    lit_b: &[u8],
    inv_b: &[u8],
    keep_b: &[u8],
    active_b: &[u8],
    max_state: u8,
) {
    use std::arch::x86_64::*;

    let max_state_v = _mm256_set1_epi8(max_state as i8);
    let ones = _mm256_set1_epi8(1);
    let all_ones = _mm256_set1_epi8(-1);
    let boost_v = _mm256_set1_epi8(boost as i8);

    // Load 32 bytes (one full AVX2 register) from a u8 slice.
    // SAFETY: caller ensures at least 32 bytes remain at ptr (checked by loop bound).
    macro_rules! load32 {
        ($slice:expr, $off:expr) => {{
            _mm256_loadu_si256($slice.as_ptr().add($off) as *const __m256i)
        }};
    }

    let mut l = 0usize;

    if fired_under {
        while l + 32 <= n_literals {
            // SAFETY: l + 32 <= n_literals, so all reads are in-bounds.
            let t = load32!(ta, l);
            let present = load32!(lit_b, l);
            let keep_v = load32!(keep_b, l);
            let inv_v = load32!(inv_b, l);
            let la = load32!(active_b, l);

            // inc = present & (boost | keep) & la  (all values 0 or 1)
            let inc = _mm256_and_si256(_mm256_and_si256(present, _mm256_or_si256(boost_v, keep_v)), la);
            // t_clamped = min(t + inc, max_state)
            let t_clamped = _mm256_min_epu8(_mm256_adds_epu8(t, inc), max_state_v);

            // absent = 1 - present
            let absent = _mm256_sub_epi8(ones, present);
            // not_at_max: 0xFF where t != max_state, 0 where t == max_state
            let not_at_max = _mm256_andnot_si256(_mm256_cmpeq_epi8(t, max_state_v), all_ones);
            // dec = absent & inv & la (0 or 1); zeroed out where t == max_state
            let dec = _mm256_and_si256(_mm256_and_si256(_mm256_and_si256(absent, inv_v), la), not_at_max);

            // native saturating unsigned byte subtract
            let result = _mm256_subs_epu8(t_clamped, dec);
            _mm256_storeu_si256(ta.as_mut_ptr().add(l) as *mut __m256i, result);
            l += 32;
        }
        // Scalar tail.
        let boost_u8 = boost as u8;
        while l < n_literals {
            let t = ta[l];
            let present = lit_b[l];
            let la = active_b[l];
            let inc = present & (boost_u8 | keep_b[l]) & la;
            let not_at_max = (t < max_state) as u8;
            let dec = (1 - present) & inv_b[l] & not_at_max & la;
            ta[l] = t.saturating_add(inc).min(max_state).saturating_sub(dec);
            l += 1;
        }
    } else {
        while l + 32 <= n_literals {
            // SAFETY: l + 32 <= n_literals, so all reads are in-bounds.
            let t = load32!(ta, l);
            let inv_v = load32!(inv_b, l);
            let la = load32!(active_b, l);

            let not_at_max = _mm256_andnot_si256(_mm256_cmpeq_epi8(t, max_state_v), all_ones);
            let dec = _mm256_and_si256(_mm256_and_si256(inv_v, la), not_at_max);

            let result = _mm256_subs_epu8(t, dec);
            _mm256_storeu_si256(ta.as_mut_ptr().add(l) as *mut __m256i, result);
            l += 32;
        }
        while l < n_literals {
            let t = ta[l];
            let not_at_max = (t < max_state) as u8;
            let dec = inv_b[l] & not_at_max & active_b[l];
            ta[l] = t.saturating_sub(dec);
            l += 1;
        }
    }
}

/// Apply Type II TA feedback to one clause (byte-array inputs).
///
/// Type II feedback fires only when the clause fires on a negative-class sample.
/// For each active literal that is **absent** in the sample and currently **excluded**
/// (`ta[l] < half`) and **not at the absorbing exclude state** (`ta[l] > 0`), the
/// TA counter is incremented by 1 (capped at `max_state`).  This pushes clauses
/// toward requiring features that were absent, making them harder to fire on
/// negative samples.
///
/// - `lit_b`: 1 if the literal is present in the current sample, 0 if absent.
/// - `active_b`: combined valid × literal-dropout mask.
///
/// **AVX2 path**: 32 literals/iteration; unsigned comparisons use the `0x80`
/// bias trick with `_mm256_cmpgt_epi8`.
///
/// **Scalar fallback**: branchless per-literal arithmetic.
#[inline(always)]
pub(crate) fn type_ii_update_bytes(
    ta: &mut [u8],
    n_literals: usize,
    lit_b: &[u8],
    active_b: &[u8],
    half: u8,
    max_state: u8,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 presence verified by the runtime check above.
            return unsafe { type_ii_update_bytes_avx2(ta, n_literals, lit_b, active_b, half, max_state) };
        }
    }
    type_ii_update_bytes_scalar(ta, n_literals, lit_b, active_b, half, max_state);
}

/// Branchless scalar fallback for [`type_ii_update_bytes`] on non-AVX2 targets.
#[inline]
fn type_ii_update_bytes_scalar(
    ta: &mut [u8],
    n_literals: usize,
    lit_b: &[u8],
    active_b: &[u8],
    half: u8,
    max_state: u8,
) {
    for l in 0..n_literals {
        let la = active_b[l];
        let t = ta[l];
        let absent = 1 - lit_b[l];
        let excluded = (t < half) as u8;
        let not_zero = (t > 0) as u8;
        let inc = absent & excluded & not_zero & la;
        ta[l] = t.saturating_add(inc).min(max_state);
    }
}

/// AVX2 implementation of [`type_ii_update_bytes`]: 32 literals per iteration.
///
/// `t < half` and `t > 0` use the unsigned-compare bias trick: XOR both sides
/// with `0x80` so unsigned `<` becomes signed `<` (`_mm256_cmpgt_epi8` with
/// arguments swapped) and `> 0` becomes signed `> i8::MIN`.
///
/// # Safety
/// Caller must verify AVX2 is available.  Pointer reads/writes stay within the
/// `ta` slice (loop guard: `l + 32 <= n_literals`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn type_ii_update_bytes_avx2(
    ta: &mut [u8],
    n_literals: usize,
    lit_b: &[u8],
    active_b: &[u8],
    half: u8,
    max_state: u8,
) {
    use std::arch::x86_64::*;

    let max_state_v = _mm256_set1_epi8(max_state as i8);
    let ones = _mm256_set1_epi8(1);
    let bias = _mm256_set1_epi8(i8::MIN);
    // t < half  ↔  (t ^ 0x80) < (half ^ 0x80)  ↔  (half ^ 0x80) > (t ^ 0x80)  [signed i8]
    let half_biased = _mm256_set1_epi8((half ^ 0x80) as i8);
    // t > 0  ↔  (t ^ 0x80) > (0 ^ 0x80) = 0x80 = i8::MIN  [signed i8]
    let zero_biased = bias;

    macro_rules! load32 {
        ($slice:expr, $off:expr) => {{
            _mm256_loadu_si256($slice.as_ptr().add($off) as *const __m256i)
        }};
    }

    let mut l = 0usize;
    while l + 32 <= n_literals {
        // SAFETY: l + 32 <= n_literals, all reads in-bounds.
        let t = load32!(ta, l);
        let lit_v = load32!(lit_b, l);
        let la = load32!(active_b, l);

        let absent = _mm256_sub_epi8(ones, lit_v); // 1 - lit (0 or 1)
        let t_biased = _mm256_xor_si256(t, bias);
        // excluded: 0xFF where t < half, 0 elsewhere
        let excluded = _mm256_cmpgt_epi8(half_biased, t_biased);
        // not_zero: 0xFF where t > 0, 0 where t == 0
        let not_zero = _mm256_cmpgt_epi8(t_biased, zero_biased);

        // inc = absent & la (0 or 1) masked by excluded & not_zero (0x00 or 0xFF)
        let inc = _mm256_and_si256(
            _mm256_and_si256(absent, la),
            _mm256_and_si256(excluded, not_zero),
        );

        // t + inc, clamped to max_state
        let result = _mm256_min_epu8(_mm256_adds_epu8(t, inc), max_state_v);
        _mm256_storeu_si256(ta.as_mut_ptr().add(l) as *mut __m256i, result);
        l += 32;
    }
    // Scalar tail.
    while l < n_literals {
        let t = ta[l];
        let la = active_b[l];
        let absent = 1 - lit_b[l];
        let excluded = (t < half) as u8;
        let not_zero = (t > 0) as u8;
        let inc = absent & excluded & not_zero & la;
        ta[l] = t.saturating_add(inc).min(max_state);
        l += 1;
    }
}

/// Type Ia / Ib feedback for one clause from packed bit inputs (test helper).
///
/// This is a convenience wrapper used by unit tests that expands packed bit arrays
/// to bytes and delegates to [`type_i_update_bytes`] + [`rebuild_include`].
/// Production code calls those functions directly with pre-expanded arrays amortised
/// over all clauses per epoch.
///
/// **Paths:**
/// - Ia (`fired && under_limit`): weight++; increment present active literals toward
///   inclusion, decrement absent active literals toward exclusion (probabilistic).
/// - Ib (`!fired || over_limit`): decrement absent active literals only.
///
/// **Absorbing include state**: `ta[l] == max_state` is immune to decrement on both paths.
///
/// `lit_active`: per-sample literal dropout mask (Bernoulli(1-p) per bit).
/// Pass all-ones when `literal_drop_p == 0`.
///
/// `max_included`: upper bound on included literal count (all valid literals, not just
/// active ones) — matches TMU's `cb_number_of_include_actions` check.
/// Pass `usize::MAX` to disable.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn clause_type_i_bytes(
    ta: &mut [u8],
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
    half: u8,
    max_state: u8,
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

/// Type II feedback for one clause from packed bit inputs (test helper).
///
/// Convenience wrapper used by unit tests that expands packed bit arrays to bytes
/// and delegates to [`type_ii_update_bytes`] + [`rebuild_include`].
/// Production code calls those functions directly with pre-expanded arrays.
///
/// Only runs when the clause fires; returns early otherwise.
/// On fire: weight--; increments each absent active excluded non-zero TA counter.
///
/// **Absorbing exclude state**: `ta[l] == 0` is immune to increment feedback.
#[allow(dead_code)]
#[inline(always)]
pub(crate) fn clause_type_ii_bytes(
    ta: &mut [u8],
    inc: &mut [u64],
    weight: &mut i32,
    lit: &[u64],
    valid: &[u64],
    words: usize,
    n_literals: usize,
    lit_active: &[u64],
    half: u8,
    max_state: u8,
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
