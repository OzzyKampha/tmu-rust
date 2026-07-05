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

/// Clause-count floor for the **exact** dense clause-parallel *training* path.
///
/// Dense training is memory-bandwidth bound: `bench_training` (10k clauses) shows
/// only ~1.3× on 4 cores, and `parallel_scaling` shows exact clause-parallel
/// training is *slower* than scalar at moderate clause counts (the model stays
/// L3-resident, so sharing it across cores just adds coherence traffic). So the
/// exact path only clause-parallelises for genuinely large models where it wins;
/// below this, `fit_epoch` runs scalar. For a real multicore speedup at any size,
/// use `.fast_training(true)` (approximate, data-parallel over samples).
#[cfg(feature = "parallel")]
pub(crate) const DENSE_TRAIN_PARALLEL_MIN: usize = 4096;

/// Total-work floor (items × per-item words) for the work-aware branch, for cases
/// where a few heavy items still amortise Rayon dispatch. Calibrated from
/// `examples/parallel_scaling.rs`: below ~256 work-units even sparse training and
/// sample-parallel inference lose to dispatch overhead; the win grows above it.
#[cfg(feature = "parallel")]
pub(crate) const PARALLEL_WORK_MIN: usize = 256;

/// Decide whether to take a Rayon path: parallelise when there are **many items**
/// (`items >= PARALLEL_MIN`, the original rule) **OR** the **total work is large**
/// (`items × words_per_item >= PARALLEL_WORK_MIN`) — the latter catches
/// **few-but-wide** workloads (e.g. 16 clauses over a million literals) the
/// count-only rule wrongly ran single-threaded.
///
/// Used for **sample-parallel inference** (all models) and **sparse training**,
/// which the `parallel_scaling` benchmark shows benefit from the work term. Dense
/// *training* deliberately keeps the count-only gate: its per-clause work is
/// AVX2-fast and Rayon is dispatched per sample, so parallelising a few wide
/// clauses is pure overhead (measured regression). Bit-identical either way
/// (disjoint per-item state), so this only selects a code path, never the result.
#[cfg(feature = "parallel")]
#[inline]
pub(crate) fn use_parallel(items: usize, words_per_item: usize) -> bool {
    items >= PARALLEL_MIN || items.saturating_mul(words_per_item) >= PARALLEL_WORK_MIN
}

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

/// Grow a dense clause bank from `old_n_features` to `new_n_features`, preserving
/// every learned per-clause state. Returns `(new_n_literals, new_words)`.
///
/// The literal layout is `[positives 0..n_features | negateds n_features..2*n_features]`
/// (see [`pack`]), so growing shifts the negated block: per clause, old positive
/// states stay at `[0..old_nf]`, old negated states move from `[old_nf..2*old_nf]`
/// to `[new_nf..new_nf + old_nf]`, and the gaps are new literal slots.
///
/// New TA slots are initialised to the deterministic `half - 1` (just-excluded)
/// rather than the constructor's random `half-1`/`half` split: excluded new
/// literals leave every clause's firing behaviour on the old feature space
/// bit-identical, whereas randomly *including* a new positive literal would block
/// the clause on all previously-seen samples. `half - 1` is one Type Ia increment
/// from inclusion and not the absorbing state `0`, so new features stay learnable.
/// New `ind` slots get `half` and new `cat` bits `0`, matching constructor init.
///
/// `include` is rebuilt from the new `ta` via [`rebuild_include`]; `valid` is
/// rebuilt for the new literal count.
#[allow(clippy::too_many_arguments)]
pub(crate) fn grow_dense_state(
    n_clauses: usize,
    old_n_features: usize,
    new_n_features: usize,
    half: u8,
    ta: &mut Vec<u8>,
    include: &mut Vec<u64>,
    ind: &mut Vec<u8>,
    cat: &mut Vec<u64>,
    valid: &mut Vec<u64>,
) -> (usize, usize) {
    debug_assert!(new_n_features > old_n_features);
    let old_nl = 2 * old_n_features;
    let new_nl = 2 * new_n_features;
    let old_words = words_for(old_nl);
    let new_words = words_for(new_nl);

    let mut new_valid = vec![0u64; new_words];
    for l in 0..new_nl {
        new_valid[l / WORD_BITS] |= 1u64 << (l % WORD_BITS);
    }

    // Copy a clause-major u8 array (ta / ind) into the new geometry:
    // positives stay at the front, negateds shift up by (new_nf - old_nf).
    let regrow_u8 = |old: &[u8], fill: u8| -> Vec<u8> {
        let mut new = vec![fill; n_clauses * new_nl];
        for j in 0..n_clauses {
            let ob = j * old_nl;
            let nb = j * new_nl;
            new[nb..nb + old_n_features].copy_from_slice(&old[ob..ob + old_n_features]);
            new[nb + new_n_features..nb + new_n_features + old_n_features]
                .copy_from_slice(&old[ob + old_n_features..ob + old_nl]);
        }
        new
    };

    let new_ta = regrow_u8(ta, half - 1);
    let new_ind = regrow_u8(ind, half);

    // Remap cat bits one at a time — grow is a cold path, and per-bit remapping
    // avoids word-shift arithmetic across the moving negated-block boundary.
    // Padding bits above old_nl are already zero (every cat write is valid-masked).
    let mut new_cat = vec![0u64; n_clauses * new_words];
    for j in 0..n_clauses {
        for l in 0..old_nl {
            if cat[j * old_words + l / WORD_BITS] >> (l % WORD_BITS) & 1 != 0 {
                let nl = if l < old_n_features {
                    l
                } else {
                    l + (new_n_features - old_n_features)
                };
                new_cat[j * new_words + nl / WORD_BITS] |= 1u64 << (nl % WORD_BITS);
            }
        }
    }

    // Rebuild include from the new ta rather than remapping bits, so
    // include ↔ ta consistency holds by construction.
    let mut new_include = vec![0u64; n_clauses * new_words];
    for j in 0..n_clauses {
        rebuild_include(
            &new_ta[j * new_nl..(j + 1) * new_nl],
            &mut new_include[j * new_words..(j + 1) * new_words],
            &new_valid,
            new_words,
            new_nl,
            half,
        );
    }

    *ta = new_ta;
    *include = new_include;
    *ind = new_ind;
    *cat = new_cat;
    *valid = new_valid;
    (new_nl, new_words)
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
#[allow(clippy::too_many_arguments)]
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
                    ta,
                    n_literals,
                    fired_under,
                    boost,
                    lit_b,
                    inv_b,
                    keep_b,
                    active_b,
                    max_state,
                )
            };
        }
    }
    type_i_update_bytes_scalar(
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
}

/// Branchless scalar fallback for [`type_i_update_bytes`] on non-AVX2 targets.
#[allow(clippy::too_many_arguments)]
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
#[allow(clippy::too_many_arguments)]
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
            let inc = _mm256_and_si256(
                _mm256_and_si256(present, _mm256_or_si256(boost_v, keep_v)),
                la,
            );
            // t_clamped = min(t + inc, max_state)
            let t_clamped = _mm256_min_epu8(_mm256_adds_epu8(t, inc), max_state_v);

            // absent = 1 - present
            let absent = _mm256_sub_epi8(ones, present);
            // not_at_max: 0xFF where t != max_state, 0 where t == max_state
            let not_at_max = _mm256_andnot_si256(_mm256_cmpeq_epi8(t, max_state_v), all_ones);
            // dec = absent & inv & la (0 or 1); zeroed out where t == max_state
            let dec = _mm256_and_si256(
                _mm256_and_si256(_mm256_and_si256(absent, inv_v), la),
                not_at_max,
            );

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
            return unsafe {
                type_ii_update_bytes_avx2(ta, n_literals, lit_b, active_b, half, max_state)
            };
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
    type_i_update_bytes(
        ta,
        n_literals,
        fired && under_limit,
        boost,
        &lit_b,
        &inv_b,
        &keep_b,
        &active_b,
        max_state,
    );
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
#[allow(clippy::too_many_arguments)]
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

/// Return the index of the first literal that is included, active, valid, and absent from sample.
///
/// Used by [`type_iii_update`] to locate the single blocking literal for `clause_and_target`
/// toggling when the clause does not fire.  Returns `None` if no such literal exists (clause fires).
#[inline]
pub(crate) fn first_false_literal(
    inc: &[u64],
    lit: &[u64],
    valid: &[u64],
    lit_active: &[u64],
    words: usize,
) -> Option<usize> {
    for k in 0..words {
        let blocking = inc[k] & valid[k] & lit_active[k] & !lit[k];
        if blocking != 0 {
            return Some(k * WORD_BITS + blocking.trailing_zeros() as usize);
        }
    }
    None
}

/// Update the per-literal indicator state (`ind`) for one Type III clause (byte-array inputs).
///
/// - `inc_b[l]`: 1 if the literal should be incremented (`active ∧ cat ∧ lit` when target prob
///   succeeds), else 0.
/// - `dec_b[l]`: 1 if the literal should be decremented (`active ∧ ¬cat ∧ lit ∧ valid`), else 0.
///
/// **AVX2 path**: 32 literals/iteration — `_mm256_adds_epu8` + `_mm256_min_epu8` for increment,
/// `_mm256_subs_epu8` for decrement.
///
/// **Scalar fallback**: branchless per-literal saturating arithmetic.
#[inline(always)]
/// Update per-literal indicator `ind` directly from bitset words — zero allocation.
///
/// Replaces the old expand-to-bytes pipeline: no `Vec` is allocated.
///
/// - Inc: `ind[l] += 1` (saturating, ≤ `max_ind`) where `lit_active & cat & lit`.
/// - Dec: `ind[l] -= 1` (saturating) where `lit_active & !cat & lit & valid`.
///
/// The AVX2 path expands 32 bits at a time to 32 byte lanes via `vpshufb` + bit isolation,
/// then applies the same saturating arithmetic as [`type_iii_ind_bytes_avx2`].
fn type_iii_ind_words(
    ind: &mut [u8],
    n_literals: usize,
    lit_active: &[u64],
    cat: &[u64],
    lit: &[u64],
    valid: &[u64],
    do_inc: bool,
    max_ind: u8,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe {
                type_iii_ind_words_avx2(ind, n_literals, lit_active, cat, lit, valid, do_inc, max_ind)
            };
        }
    }
    type_iii_ind_words_scalar(ind, n_literals, lit_active, cat, lit, valid, do_inc, max_ind);
}

#[inline]
fn type_iii_ind_words_scalar(
    ind: &mut [u8],
    n_literals: usize,
    lit_active: &[u64],
    cat: &[u64],
    lit: &[u64],
    valid: &[u64],
    do_inc: bool,
    max_ind: u8,
) {
    for l in 0..n_literals {
        let w = l / WORD_BITS;
        let b = l % WORD_BITS;
        let active = (lit_active[w] >> b) & 1;
        let cat_b = (cat[w] >> b) & 1;
        let lit_b = (lit[w] >> b) & 1;
        let valid_b = (valid[w] >> b) & 1;
        let inc = if do_inc { (active & cat_b & lit_b) as u8 } else { 0u8 };
        let dec = (active & (1 - cat_b) & lit_b & valid_b) as u8;
        ind[l] = ind[l].saturating_add(inc).min(max_ind).saturating_sub(dec);
    }
}

/// AVX2 path for [`type_iii_ind_words`]: expands 32 bits per iteration via `vpshufb`.
///
/// For each group of 32 literals, computes `inc_bits` and `dec_bits` as `u32` values from the
/// raw bitset words, then uses a shuffle-based bit-to-byte expansion to produce 32-byte SIMD
/// vectors, and applies saturating arithmetic — exactly like `type_iii_ind_bytes_avx2` but
/// without any intermediate `Vec` allocation.
///
/// # Safety
/// Caller must verify AVX2 is available.  Pointer reads/writes stay within the
/// `ind` slice (loop guard: `l + 32 <= n_literals`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn type_iii_ind_words_avx2(
    ind: &mut [u8],
    n_literals: usize,
    lit_active: &[u64],
    cat: &[u64],
    lit: &[u64],
    valid: &[u64],
    do_inc: bool,
    max_ind: u8,
) {
    use std::arch::x86_64::*;

    let max_ind_v = _mm256_set1_epi8(max_ind as i8);
    let zero = _mm256_setzero_si256();
    let ones_v = _mm256_set1_epi8(1i8);

    // Shuffle control: within each 128-bit lane, route source bytes so that
    // output bytes 0-7 all hold source byte 0, bytes 8-15 hold byte 1, etc.
    // _mm256_set_epi8 order: (e31, e30, ..., e1, e0) where e0 is byte 0.
    let shuf = _mm256_set_epi8(
        3, 3, 3, 3, 3, 3, 3, 3,
        2, 2, 2, 2, 2, 2, 2, 2,
        1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0,
    );
    // Bit-isolation masks: lane i gets the mask for bit (i % 8).
    let bit_masks = _mm256_set_epi8(
        0x80u8 as i8, 0x40u8 as i8, 0x20u8 as i8, 0x10u8 as i8,
        0x08u8 as i8, 0x04u8 as i8, 0x02u8 as i8, 0x01u8 as i8,
        0x80u8 as i8, 0x40u8 as i8, 0x20u8 as i8, 0x10u8 as i8,
        0x08u8 as i8, 0x04u8 as i8, 0x02u8 as i8, 0x01u8 as i8,
        0x80u8 as i8, 0x40u8 as i8, 0x20u8 as i8, 0x10u8 as i8,
        0x08u8 as i8, 0x04u8 as i8, 0x02u8 as i8, 0x01u8 as i8,
        0x80u8 as i8, 0x40u8 as i8, 0x20u8 as i8, 0x10u8 as i8,
        0x08u8 as i8, 0x04u8 as i8, 0x02u8 as i8, 0x01u8 as i8,
    );

    // Expand a u32 mask to a __m256i with 0x01 per set bit, 0x00 per clear bit.
    macro_rules! bits32_to_vec {
        ($bits:expr) => {{
            let v = _mm256_set1_epi32($bits as i32);
            let spread = _mm256_shuffle_epi8(v, shuf);
            let masked = _mm256_and_si256(spread, bit_masks);
            let is_zero = _mm256_cmpeq_epi8(masked, zero);
            _mm256_andnot_si256(is_zero, ones_v)
        }};
    }

    let mut l = 0usize;
    while l + 32 <= n_literals {
        let widx = l / WORD_BITS;
        let boff = l % WORD_BITS; // 0 for the low half, 32 for the high half of each word.

        let w_active = lit_active[widx];
        let w_cat = cat[widx];
        let w_lit = lit[widx];
        let w_valid = valid[widx];

        let inc_bits: u32 = if do_inc {
            ((w_active & w_cat & w_lit) >> boff) as u32
        } else {
            0
        };
        let dec_bits: u32 = ((w_active & !w_cat & w_lit & w_valid) >> boff) as u32;

        let inc_v = bits32_to_vec!(inc_bits);
        let dec_v = bits32_to_vec!(dec_bits);

        // SAFETY: l + 32 <= n_literals, all reads/writes in-bounds.
        let ptr = ind.as_mut_ptr().add(l) as *mut __m256i;
        let v = _mm256_loadu_si256(ptr as *const __m256i);
        let incremented = _mm256_min_epu8(_mm256_adds_epu8(v, inc_v), max_ind_v);
        let result = _mm256_subs_epu8(incremented, dec_v);
        _mm256_storeu_si256(ptr, result);

        l += 32;
    }
    // Scalar tail.
    while l < n_literals {
        let w = l / WORD_BITS;
        let b = l % WORD_BITS;
        let active = (lit_active[w] >> b) & 1;
        let cat_b = (cat[w] >> b) & 1;
        let lit_b_v = (lit[w] >> b) & 1;
        let valid_b = (valid[w] >> b) & 1;
        let inc = if do_inc { (active & cat_b & lit_b_v) as u8 } else { 0u8 };
        let dec = (active & (1 - cat_b) & lit_b_v & valid_b) as u8;
        ind[l] = ind[l].saturating_add(inc).min(max_ind).saturating_sub(dec);
        l += 1;
    }
}

/// Fused [`type_iii_ind_words`] + [`type_iii_ta_dec_bytes`] in a single pass over `ind`.
///
/// When the clause fires and the TA-decrement probability succeeds, calling this instead
/// of the two functions separately saves one full re-read of `ind` and `active_b` —
/// the new `ind` values computed by the AVX2 loop are used immediately to decrement `ta`
/// without writing and re-loading them.
///
/// For large `n_literals` (e.g. 16 384 for 8 192 features) this eliminates ~32 KB of
/// L1-cache pressure per clause update.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn type_iii_ind_words_and_ta_dec(
    ta: &mut [u8],
    ind: &mut [u8],
    n_literals: usize,
    lit_active: &[u64],
    cat: &[u64],
    lit: &[u64],
    valid: &[u64],
    active_b: &[u8],
    do_inc: bool,
    max_ind: u8,
    half_ind: u8,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe {
                type_iii_ind_words_and_ta_dec_avx2(
                    ta, ind, n_literals, lit_active, cat, lit, valid, active_b, do_inc, max_ind,
                    half_ind,
                )
            };
        }
    }
    type_iii_ind_words_and_ta_dec_scalar(
        ta, ind, n_literals, lit_active, cat, lit, valid, active_b, do_inc, max_ind, half_ind,
    );
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn type_iii_ind_words_and_ta_dec_scalar(
    ta: &mut [u8],
    ind: &mut [u8],
    n_literals: usize,
    lit_active: &[u64],
    cat: &[u64],
    lit: &[u64],
    valid: &[u64],
    active_b: &[u8],
    do_inc: bool,
    max_ind: u8,
    half_ind: u8,
) {
    for l in 0..n_literals {
        let w = l / WORD_BITS;
        let b = l % WORD_BITS;
        let active = (lit_active[w] >> b) & 1;
        let cat_b = (cat[w] >> b) & 1;
        let lit_bv = (lit[w] >> b) & 1;
        let valid_b = (valid[w] >> b) & 1;
        let inc = if do_inc { (active & cat_b & lit_bv) as u8 } else { 0u8 };
        let dec_ind = (active & (1 - cat_b) & lit_bv & valid_b) as u8;
        let new_ind = ind[l].saturating_add(inc).min(max_ind).saturating_sub(dec_ind);
        ind[l] = new_ind;
        ta[l] = ta[l].saturating_sub(active_b[l] & (new_ind < half_ind) as u8);
    }
}

/// AVX2 fused [`type_iii_ind_words`] + [`type_iii_ta_dec_bytes`].
///
/// Inner loop writes `ind` and `ta` in a single pass: after computing the new `ind` vector
/// the same registers are used to test `< half_ind` and decrement `ta`, with no round-trip
/// through memory.
///
/// # Safety
/// Caller must verify AVX2 is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn type_iii_ind_words_and_ta_dec_avx2(
    ta: &mut [u8],
    ind: &mut [u8],
    n_literals: usize,
    lit_active: &[u64],
    cat: &[u64],
    lit: &[u64],
    valid: &[u64],
    active_b: &[u8],
    do_inc: bool,
    max_ind: u8,
    half_ind: u8,
) {
    use std::arch::x86_64::*;

    let max_ind_v = _mm256_set1_epi8(max_ind as i8);
    let zero = _mm256_setzero_si256();
    let ones_v = _mm256_set1_epi8(1i8);
    let bias = _mm256_set1_epi8(i8::MIN);
    // `new_ind < half_ind` via unsigned-compare bias trick (identical to type_iii_ta_dec_bytes_avx2).
    let half_biased = _mm256_set1_epi8((half_ind ^ 0x80) as i8);

    let shuf = _mm256_set_epi8(
        3, 3, 3, 3, 3, 3, 3, 3,
        2, 2, 2, 2, 2, 2, 2, 2,
        1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0,
    );
    let bit_masks = _mm256_set_epi8(
        0x80u8 as i8, 0x40u8 as i8, 0x20u8 as i8, 0x10u8 as i8,
        0x08u8 as i8, 0x04u8 as i8, 0x02u8 as i8, 0x01u8 as i8,
        0x80u8 as i8, 0x40u8 as i8, 0x20u8 as i8, 0x10u8 as i8,
        0x08u8 as i8, 0x04u8 as i8, 0x02u8 as i8, 0x01u8 as i8,
        0x80u8 as i8, 0x40u8 as i8, 0x20u8 as i8, 0x10u8 as i8,
        0x08u8 as i8, 0x04u8 as i8, 0x02u8 as i8, 0x01u8 as i8,
        0x80u8 as i8, 0x40u8 as i8, 0x20u8 as i8, 0x10u8 as i8,
        0x08u8 as i8, 0x04u8 as i8, 0x02u8 as i8, 0x01u8 as i8,
    );

    macro_rules! bits32_to_vec {
        ($bits:expr) => {{
            let v = _mm256_set1_epi32($bits as i32);
            let spread = _mm256_shuffle_epi8(v, shuf);
            let masked = _mm256_and_si256(spread, bit_masks);
            let is_zero = _mm256_cmpeq_epi8(masked, zero);
            _mm256_andnot_si256(is_zero, ones_v)
        }};
    }

    macro_rules! load32 {
        ($slice:expr, $off:expr) => {
            _mm256_loadu_si256($slice.as_ptr().add($off) as *const __m256i)
        };
    }

    let mut l = 0usize;
    while l + 32 <= n_literals {
        let widx = l / WORD_BITS;
        let boff = l % WORD_BITS;

        let w_active = lit_active[widx];
        let w_cat = cat[widx];
        let w_lit = lit[widx];
        let w_valid = valid[widx];

        let inc_bits: u32 = if do_inc {
            ((w_active & w_cat & w_lit) >> boff) as u32
        } else {
            0
        };
        let dec_bits: u32 = ((w_active & !w_cat & w_lit & w_valid) >> boff) as u32;

        let inc_v = bits32_to_vec!(inc_bits);
        let dec_v = bits32_to_vec!(dec_bits);

        // Update ind.
        let ind_ptr = ind.as_mut_ptr().add(l) as *mut __m256i;
        let ind_v = _mm256_loadu_si256(ind_ptr as *const __m256i);
        let new_ind = _mm256_subs_epu8(
            _mm256_min_epu8(_mm256_adds_epu8(ind_v, inc_v), max_ind_v),
            dec_v,
        );
        _mm256_storeu_si256(ind_ptr, new_ind);

        // Fused ta decrement: active_b[l] && new_ind < half_ind → decrement ta[l].
        let active_v = load32!(active_b, l);
        let biased_ind = _mm256_xor_si256(new_ind, bias);
        let ind_lt_half = _mm256_cmpgt_epi8(half_biased, biased_ind);
        let ta_dec = _mm256_and_si256(active_v, ind_lt_half);
        let ta_ptr = ta.as_mut_ptr().add(l) as *mut __m256i;
        _mm256_storeu_si256(ta_ptr, _mm256_subs_epu8(_mm256_loadu_si256(ta_ptr as *const __m256i), ta_dec));

        l += 32;
    }
    // Scalar tail.
    while l < n_literals {
        let w = l / WORD_BITS;
        let b = l % WORD_BITS;
        let active = (lit_active[w] >> b) & 1;
        let cat_b = (cat[w] >> b) & 1;
        let lit_bv = (lit[w] >> b) & 1;
        let valid_b = (valid[w] >> b) & 1;
        let inc = if do_inc { (active & cat_b & lit_bv) as u8 } else { 0u8 };
        let dec_ind = (active & (1 - cat_b) & lit_bv & valid_b) as u8;
        let new_ind = ind[l].saturating_add(inc).min(max_ind).saturating_sub(dec_ind);
        ind[l] = new_ind;
        ta[l] = ta[l].saturating_sub(active_b[l] & (new_ind < half_ind) as u8);
        l += 1;
    }
}

/// Decrement `ta[l]` by 1 (saturating) for each literal that is active and has `ind[l] < half_ind`.
///
/// - `active_b[l]`: 1 if the literal is active (`lit_active`), else 0.
///
/// **AVX2 path**: 32 literals/iteration; `ind < half_ind` uses the unsigned-compare bias trick
/// (`_mm256_cmpgt_epi8` with XOR-`0x80` bias) identical to [`type_ii_update_bytes_avx2`].
///
/// **Scalar fallback**: branchless per-literal arithmetic.
#[inline(always)]
fn type_iii_ta_dec_bytes(ta: &mut [u8], ind: &[u8], n_literals: usize, active_b: &[u8], half_ind: u8) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 presence verified by the runtime check above.
            return unsafe { type_iii_ta_dec_bytes_avx2(ta, ind, n_literals, active_b, half_ind) };
        }
    }
    type_iii_ta_dec_bytes_scalar(ta, ind, n_literals, active_b, half_ind);
}

#[inline]
fn type_iii_ta_dec_bytes_scalar(
    ta: &mut [u8],
    ind: &[u8],
    n_literals: usize,
    active_b: &[u8],
    half_ind: u8,
) {
    for l in 0..n_literals {
        let dec = active_b[l] & (ind[l] < half_ind) as u8;
        ta[l] = ta[l].saturating_sub(dec);
    }
}

/// AVX2 implementation of [`type_iii_ta_dec_bytes`]: 32 literals per iteration.
///
/// `ind < half_ind` is tested with the unsigned-compare bias trick: XOR both sides with `0x80`
/// so unsigned `<` becomes signed `<` (`_mm256_cmpgt_epi8` with swapped arguments).
///
/// # Safety
/// Caller must verify AVX2 is available.  Pointer reads/writes stay within the
/// `ta` / `ind` slices (loop guard: `l + 32 <= n_literals`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn type_iii_ta_dec_bytes_avx2(
    ta: &mut [u8],
    ind: &[u8],
    n_literals: usize,
    active_b: &[u8],
    half_ind: u8,
) {
    use std::arch::x86_64::*;

    let ones = _mm256_set1_epi8(1);
    let bias = _mm256_set1_epi8(i8::MIN);
    // ind < half_ind  ↔  (ind ^ 0x80) < (half_ind ^ 0x80)  as signed i8
    //                 ↔  (half_ind ^ 0x80) > (ind ^ 0x80)   [cmpgt argument order]
    let half_biased = _mm256_set1_epi8((half_ind ^ 0x80) as i8);

    macro_rules! load32 {
        ($slice:expr, $off:expr) => {
            _mm256_loadu_si256($slice.as_ptr().add($off) as *const __m256i)
        };
    }

    let mut l = 0usize;
    while l + 32 <= n_literals {
        // SAFETY: l + 32 <= n_literals, all reads in-bounds.
        let t = load32!(ta, l);
        let ind_v = load32!(ind, l);
        let active = load32!(active_b, l);

        let ind_biased = _mm256_xor_si256(ind_v, bias);
        // below_half: 0xFF where ind < half_ind, 0x00 elsewhere
        let below_half = _mm256_cmpgt_epi8(half_biased, ind_biased);
        // dec: 0 or 1 per literal  (active_b is 0/1; below_half is 0/0xFF → mask to 0/1 via `& ones`)
        let dec = _mm256_and_si256(active, _mm256_and_si256(below_half, ones));
        let result = _mm256_subs_epu8(t, dec);
        _mm256_storeu_si256(ta.as_mut_ptr().add(l) as *mut __m256i, result);
        l += 32;
    }
    // Scalar tail.
    while l < n_literals {
        let dec = active_b[l] & (ind[l] < half_ind) as u8;
        ta[l] = ta[l].saturating_sub(dec);
        l += 1;
    }
}

/// Apply Type III TA feedback to one clause — mirrors TMU's `cb_type_iii_feedback`.
///
/// Maintains a per-literal *indicator state* (`ind`, 8-bit) alongside the primary TA state.
/// Literals confirmed relevant to the target accumulate indicator credit; those whose indicator
/// state stays below `half_ind` have their primary TA counter decremented, gradually excluding
/// them from the clause and producing smaller, more interpretable rules.
///
/// `active_b` is the byte-expanded form of `lit_active` — pass the value already computed
/// once per sample in the outer training loop to avoid a per-clause allocation.
///
/// Returns `true` when the primary TA state was modified; the caller must then call
/// [`rebuild_include`].
///
/// # Algorithm
///
/// **Clause fires:**
/// - If `target` and random ≤ 1 − 1/d: increment `ind[l]` (saturating at `max_ind`) for
///   each active literal present in the sample that is also in `cat`.
/// - Decrement `ind[l]` (saturating at 0) for each active present literal **not** in `cat`.
/// - Invert `cat`: if `target`, `cat ← ~cat & valid`; otherwise `cat ← valid` (all set).
///
/// **Clause does not fire:**
/// - Find the first blocking literal (included, active, valid, absent from sample).
/// - If its bit in `cat` is 0, set it; if it is 1 and `target`, clear it.
///
/// **TA decrement (probabilistic, AVX2-accelerated):**
/// - With probability `update_p`, decrement `ta[l]` for each active literal with
///   `ind[l] < half_ind`.
///
/// **AVX2 paths**: the `ind` update ([`type_iii_ind_words`]) and `ta` decrement
/// ([`type_iii_ta_dec_bytes`]) are fully vectorised with zero heap allocation.
/// The bitset operations (`cat` update, `first_false_literal`) remain scalar.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn type_iii_update(
    ta: &mut [u8],
    ind: &mut [u8],
    cat: &mut [u64],
    inc: &[u64],
    lit: &[u64],
    valid: &[u64],
    lit_active: &[u64],
    active_b: &[u8],
    words: usize,
    n_literals: usize,
    d: f64,
    update_p: f64,
    target: bool,
    rng: &mut Rng,
    half_ind: u8,
    max_ind: u8,
) -> bool {
    let fires = clause_fire(inc, lit, valid, words, lit_active);

    if fires {
        let do_inc = target && rng.next_f64() <= 1.0 - 1.0 / d;

        if rng.next_f64() <= update_p {
            // Fused path: update ind and decrement ta in a single pass — halves memory reads
            // over calling type_iii_ind_words then type_iii_ta_dec_bytes separately.
            type_iii_ind_words_and_ta_dec(
                ta, ind, n_literals, lit_active, cat, lit, valid, active_b, do_inc, max_ind,
                half_ind,
            );
            for k in 0..words {
                cat[k] = if target { !cat[k] & valid[k] } else { valid[k] };
            }
            return true;
        }

        // update_p check failed: still update ind (and cat), but skip ta decrement.
        type_iii_ind_words(ind, n_literals, lit_active, cat, lit, valid, do_inc, max_ind);
        for k in 0..words {
            cat[k] = if target { !cat[k] & valid[k] } else { valid[k] };
        }
    } else {
        if let Some(off) = first_false_literal(inc, lit, valid, lit_active, words) {
            let chunk = off / WORD_BITS;
            let pos = off % WORD_BITS;
            if (cat[chunk] >> pos) & 1 == 0 {
                cat[chunk] |= 1u64 << pos;
            } else if target {
                cat[chunk] &= !(1u64 << pos);
            }
        }

        if rng.next_f64() <= update_p {
            type_iii_ta_dec_bytes(ta, ind, n_literals, active_b, half_ind);
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- parallel_by_work -----------------------------------------------------

    #[cfg(feature = "parallel")]
    #[test]
    fn use_parallel_thresholds() {
        // Many items: original count rule still fires regardless of width.
        assert!(use_parallel(PARALLEL_MIN, 1));
        assert!(!use_parallel(PARALLEL_MIN - 1, 1));
        // Few-but-wide: crosses the total-work floor (PARALLEL_WORK_MIN = 256).
        assert!(use_parallel(4, 64)); // 256
        assert!(use_parallel(2, 200)); // 400
        assert!(!use_parallel(4, 32)); // 128 < 256, and 4 < 128 → scalar
        assert!(!use_parallel(4, 1)); // trivial → scalar
        // No overflow at extreme sizes.
        assert!(use_parallel(usize::MAX, usize::MAX));
    }

    // ---- grow_dense_state -----------------------------------------------------

    #[test]
    fn grow_dense_state_relayouts_across_word_boundary() {
        // old_nf = 40 (old_nl = 80, 2 words) -> new_nf = 70 (new_nl = 140, 3 words):
        // the stride change and the negated-block shift both cross word boundaries.
        let n_clauses = 2usize;
        let (old_nf, new_nf) = (40usize, 70usize);
        let (old_nl, old_words) = (2 * old_nf, words_for(2 * old_nf));
        let half = 128u8;

        // Distinctive TA/ind values encoding (clause, literal) so any misplacement
        // is detected: ta = 7*j + l mod values that straddle half on both sides.
        let mut ta = vec![0u8; n_clauses * old_nl];
        let mut ind = vec![0u8; n_clauses * old_nl];
        for j in 0..n_clauses {
            for l in 0..old_nl {
                ta[j * old_nl + l] = ((j * 89 + l * 3) % 256) as u8;
                ind[j * old_nl + l] = ((j * 53 + l * 7 + 11) % 256) as u8;
            }
        }
        let mut valid = vec![0u64; old_words];
        for l in 0..old_nl {
            valid[l / WORD_BITS] |= 1u64 << (l % WORD_BITS);
        }
        let mut include = vec![0u64; n_clauses * old_words];
        for j in 0..n_clauses {
            rebuild_include(
                &ta[j * old_nl..(j + 1) * old_nl],
                &mut include[j * old_words..(j + 1) * old_words],
                &valid,
                old_words,
                old_nl,
                half,
            );
        }
        // cat bits on a spread of positions in both the positive and negated blocks.
        let cat_bits = [0usize, 7, 39, 40, 63, 64, 79];
        let mut cat = vec![0u64; n_clauses * old_words];
        for (j, &l) in cat_bits.iter().enumerate().map(|(i, l)| (i % n_clauses, l)) {
            cat[j * old_words + l / WORD_BITS] |= 1u64 << (l % WORD_BITS);
        }
        let (old_ta, old_ind, old_cat) = (ta.clone(), ind.clone(), cat.clone());

        let (new_nl, new_words) = grow_dense_state(
            n_clauses, old_nf, new_nf, half, &mut ta, &mut include, &mut ind, &mut cat, &mut valid,
        );
        assert_eq!(new_nl, 2 * new_nf);
        assert_eq!(new_words, words_for(2 * new_nf));

        // Remap: positives stay, negateds shift by (new_nf - old_nf).
        let remap = |l: usize| if l < old_nf { l } else { l + (new_nf - old_nf) };
        let is_old_slot = |l: usize| l < old_nf || (new_nf..new_nf + old_nf).contains(&l);

        for j in 0..n_clauses {
            for l in 0..old_nl {
                assert_eq!(ta[j * new_nl + remap(l)], old_ta[j * old_nl + l], "ta j={j} l={l}");
                assert_eq!(ind[j * new_nl + remap(l)], old_ind[j * old_nl + l], "ind j={j} l={l}");
            }
            for l in 0..new_nl {
                if !is_old_slot(l) {
                    assert_eq!(ta[j * new_nl + l], half - 1, "new ta slot j={j} l={l}");
                    assert_eq!(ind[j * new_nl + l], half, "new ind slot j={j} l={l}");
                }
                // include consistent with ta everywhere.
                let inc_bit = include[j * new_words + l / WORD_BITS] >> (l % WORD_BITS) & 1;
                assert_eq!(inc_bit == 1, ta[j * new_nl + l] >= half, "include j={j} l={l}");
                // cat bits appear exactly at remapped positions.
                let cat_bit = cat[j * new_words + l / WORD_BITS] >> (l % WORD_BITS) & 1;
                let expected = if is_old_slot(l) {
                    let ol = if l < old_nf { l } else { l - (new_nf - old_nf) };
                    old_cat[j * old_words + ol / WORD_BITS] >> (ol % WORD_BITS) & 1
                } else {
                    0
                };
                assert_eq!(cat_bit, expected, "cat j={j} l={l}");
            }
        }

        // valid covers exactly the new literal range.
        for l in 0..new_words * WORD_BITS {
            let bit = valid[l / WORD_BITS] >> (l % WORD_BITS) & 1;
            assert_eq!(bit == 1, l < new_nl, "valid bit {l}");
        }
        // include padding bits above new_nl are zero.
        for j in 0..n_clauses {
            for l in new_nl..new_words * WORD_BITS {
                assert_eq!(include[j * new_words + l / WORD_BITS] >> (l % WORD_BITS) & 1, 0);
            }
        }
    }

    // ---- clause_fire vs fire_predict on an empty clause ----------------------

    #[test]
    fn clause_fire_empty_returns_true() {
        // An empty clause (no included literals) must fire during training so that
        // Type Ib feedback still reaches it and can push excluded TAs toward 0.
        let words = 2usize;
        let inc = vec![0u64; words];
        let lit = vec![0u64; words];
        let valid = vec![!0u64; words];
        let lit_active = vec![!0u64; words];
        assert!(clause_fire(&inc, &lit, &valid, words, &lit_active));
    }

    #[test]
    fn fire_predict_and_clause_fire_differ_on_empty() {
        // The same all-zero include bitset must return different values for the
        // inference path (fire_predict → false) vs the training path (clause_fire → true).
        let words = 1usize;
        let inc = vec![0u64];
        let lit = vec![!0u64]; // all features present
        let valid = vec![!0u64];
        let lit_active = vec![!0u64];
        assert!(
            !fire_predict(&inc, &lit, &valid, words),
            "inference: empty clause must not vote"
        );
        assert!(
            clause_fire(&inc, &lit, &valid, words, &lit_active),
            "training: empty clause must fire"
        );
    }

    // ---- expand_bits_to_bytes round-trip ------------------------------------

    #[test]
    fn expand_bits_to_bytes_round_trips_pack() {
        // pack() followed by expand_bits_to_bytes() must reproduce the original
        // feature vector as 0/1 bytes for both positive and negated literals.
        let n_features = 6usize;
        let x: Vec<u8> = vec![1, 0, 1, 1, 0, 0];
        let n_literals = 2 * n_features;
        let words = words_for(n_literals);
        let mut lit = vec![0u64; words];
        pack(&x, n_features, &mut lit);

        let bytes = expand_bits_to_bytes(&lit, n_literals);
        assert_eq!(bytes.len(), n_literals);
        for i in 0..n_features {
            assert_eq!(bytes[i], x[i], "positive literal {i}");
            assert_eq!(bytes[n_features + i], 1 - x[i], "negated literal {i}");
        }
    }

    // ---- rebuild_include correctness -----------------------------------------

    #[test]
    fn rebuild_include_boundary_at_half() {
        // ta[l] = half-1 must produce excluded (0); ta[l] = half must produce included (1).
        let half = 128u8;
        let n_literals = 2usize;
        let words = 1usize;
        let ta = vec![half - 1, half];
        let valid = vec![0b11u64];
        let mut inc = vec![0u64];
        rebuild_include(&ta, &mut inc, &valid, words, n_literals, half);
        assert_eq!(inc[0] & 1, 0, "ta = half-1 must be excluded");
        assert_eq!((inc[0] >> 1) & 1, 1, "ta = half must be included");
    }

    #[test]
    fn rebuild_include_all_excluded() {
        // When every ta value is below half the entire include bitset must be zero.
        let half = 4u8;
        let n_literals = 8usize;
        let words = 1usize;
        let ta = vec![0u8, 1, 2, 3, 0, 1, 2, 3]; // all < 4
        let valid = vec![0xFFu64];
        let mut inc = vec![!0u64]; // start with all bits set to verify they get cleared
        rebuild_include(&ta, &mut inc, &valid, words, n_literals, half);
        assert_eq!(
            inc[0], 0,
            "all ta < half must produce an empty include bitset"
        );
    }

    #[test]
    fn rebuild_include_non_multiple_of_32() {
        // n_literals not divisible by 32 exercises the scalar tail in both the AVX2
        // and scalar paths.  Four sizes cover all tail lengths:
        //   31 → 0 full AVX2 chunks + 31-element tail
        //   33 → 1 chunk + 1-element tail
        //   63 → 1 chunk + 31-element tail
        //   65 → 2 full words; within the second word: 0 chunks + 1-element tail
        for &n in &[31usize, 33, 63, 65] {
            let half = 128u8;
            let words = words_for(n);
            // Even-indexed literals → included (ta=half); odd → excluded (ta=half-1).
            let ta: Vec<u8> = (0..n)
                .map(|l| if l % 2 == 0 { half } else { half - 1 })
                .collect();
            let mut valid = vec![0u64; words];
            for l in 0..n {
                valid[l / WORD_BITS] |= 1u64 << (l % WORD_BITS);
            }
            let mut inc = vec![0u64; words];
            rebuild_include(&ta, &mut inc, &valid, words, n, half);

            for l in 0..n {
                let got: u64 = (inc[l / WORD_BITS] >> (l % WORD_BITS)) & 1;
                let expected: u64 = if l % 2 == 0 { 1 } else { 0 };
                assert_eq!(
                    got, expected,
                    "n={n} literal {l}: expected {expected} got {got}"
                );
            }
        }
    }

    // ---- state_bits boundary arithmetic -------------------------------------

    #[test]
    fn state_bits_2_min_constants() {
        // state_bits=2 → half=2, max_state=3; verify saturation behaves correctly.
        let state_bits: usize = 2;
        let half = 1u8 << (state_bits - 1);
        let max_state = ((1u16 << state_bits) - 1) as u8;
        assert_eq!(half, 2u8);
        assert_eq!(max_state, 3u8);
        assert_eq!(3u8.saturating_add(1).min(max_state), 3u8, "saturate at 3");
        assert_eq!(0u8.saturating_sub(1), 0u8, "saturate at 0");
    }

    #[test]
    fn state_bits_8_max_no_overflow() {
        // state_bits=8 → half=128, max_state=255.  Computing max_state via a u16
        // intermediate avoids the `1u8 << 8` overflow that would occur with a direct u8 shift.
        let state_bits: usize = 8;
        let half = 1u8 << (state_bits - 1); // 1 << 7 = 128; no overflow
        let max_state = ((1u16 << state_bits) - 1) as u8; // 255 via u16
        assert_eq!(half, 128u8);
        assert_eq!(max_state, 255u8);
        assert_eq!(255u8.saturating_add(1).min(max_state), 255u8);
    }

    // ---- type_i_update_bytes Ib path ----------------------------------------

    #[test]
    fn type_i_ib_decrements_absent_non_max() {
        // Ib path (fired_under=false): absent active literals are decremented
        // unless they are at max_state (absorbing include state).
        let n_literals = 4usize;
        let max_state = 15u8;
        let mut ta = vec![5u8, 1, 0, max_state];
        let lit_b = vec![0u8; 4]; // all absent
        let inv_b = vec![1u8; 4]; // always trigger decrement
        let keep_b = vec![0u8; 4];
        let active_b = vec![1u8; 4];
        type_i_update_bytes(
            &mut ta, n_literals, false, false, &lit_b, &inv_b, &keep_b, &active_b, max_state,
        );
        assert_eq!(ta[0], 4, "5 - 1 = 4");
        assert_eq!(ta[1], 0, "1 - 1 = 0");
        assert_eq!(ta[2], 0, "0 - 1 saturates at 0");
        assert_eq!(ta[3], max_state, "max_state is immune to Ib decrement");
    }

    // ---- type_ii_update_bytes logic ------------------------------------------

    #[test]
    fn type_ii_increments_absent_excluded_nonzero() {
        // Type II increments only absent, excluded (ta < half), non-zero literals.
        // Each of the five literals tests exactly one guard condition.
        let half = 8u8;
        let max_state = 15u8;
        let n_literals = 5usize;
        // lit 0: absent, excluded, non-zero  → incremented 3→4
        // lit 1: present, excluded, non-zero → skipped (present guard)
        // lit 2: absent, included (ta≥half)  → skipped (excluded guard)
        // lit 3: absent, excluded, zero      → skipped (absorbing-exclude guard)
        // lit 4: absent, excluded, non-zero  → incremented to the half boundary
        let mut ta = vec![3u8, 3, half, 0, half - 1];
        let lit_b = vec![0u8, 1, 0, 0, 0];
        let active_b = vec![1u8; 5];
        type_ii_update_bytes(&mut ta, n_literals, &lit_b, &active_b, half, max_state);
        assert_eq!(ta[0], 4, "absent excluded non-zero → incremented");
        assert_eq!(ta[1], 3, "present → unchanged");
        assert_eq!(ta[2], half, "included (ta≥half) → unchanged");
        assert_eq!(ta[3], 0, "ta=0 (absorbing exclude) → stays at 0");
        assert_eq!(ta[4], half, "absent excluded non-zero incremented to half");
    }

    // ---- first_false_literal ------------------------------------------------

    #[test]
    fn first_false_literal_returns_none_when_fires() {
        // Clause fires when all included, active literals are present.
        // Bit 0 included, bit 0 present → fires → None.
        let words = 1usize;
        let inc = vec![0b0001u64];
        let lit = vec![0b0111u64]; // bit 0 present
        let valid = vec![0xFu64];
        let lit_active = vec![!0u64];
        assert_eq!(first_false_literal(&inc, &lit, &valid, &lit_active, words), None);
    }

    #[test]
    fn first_false_literal_finds_blocking_literal() {
        // Bits 0 and 2 included; bit 0 present but bit 2 absent → literal 2 blocks.
        let words = 1usize;
        let inc = vec![0b0101u64];
        let lit = vec![0b0001u64];
        let valid = vec![0xFu64];
        let lit_active = vec![!0u64];
        assert_eq!(
            first_false_literal(&inc, &lit, &valid, &lit_active, words),
            Some(2)
        );
    }

    #[test]
    fn first_false_literal_respects_lit_active() {
        // Bit 2 included and absent, but inactive (dropped) → not the blocker.
        // No active blocking literal → None.
        let words = 1usize;
        let inc = vec![0b0101u64];
        let lit = vec![0b0001u64];
        let valid = vec![0xFu64];
        let lit_active = vec![0b0001u64]; // only bit 0 active; bit 2 inactive
        assert_eq!(first_false_literal(&inc, &lit, &valid, &lit_active, words), None);
    }

    // ---- type_iii_update ----------------------------------------------------

    #[test]
    fn type_iii_update_inverts_cat_on_target_fire() {
        // Empty clause fires; target=true → cat should be inverted to all-valid.
        let words = 1usize;
        let n_literals = 4usize;
        let valid = vec![0xFu64];
        let mut ta = vec![0u8; n_literals];
        let mut ind = vec![0u8; n_literals];
        let mut cat = vec![0u64; words]; // cat starts all-zero
        let inc = vec![0u64; words];     // empty clause always fires
        let lit = vec![0b0011u64];       // bits 0 and 1 present
        let lit_active = vec![!0u64];
        let mut rng = crate::rng::Rng::new(42);

        let active_b = expand_bits_to_bytes(&lit_active, n_literals);
        type_iii_update(
            &mut ta, &mut ind, &mut cat, &inc,
            &lit, &valid, &lit_active, &active_b,
            words, n_literals,
            200.0, 0.0, // update_p=0: TA decrement never fires
            true, &mut rng, 128, 255,
        );

        // cat was 0; target=true → cat = ~0 & 0xF = 0xF
        assert_eq!(cat[0] & 0xF, 0xF, "cat should be bitwise-NOT of original (all-valid)");
    }

    #[test]
    fn type_iii_update_sets_cat_to_valid_on_non_target_fire() {
        // Empty clause fires; target=false → cat should become all-valid (0xF).
        let words = 1usize;
        let n_literals = 4usize;
        let valid = vec![0xFu64];
        let mut ta = vec![0u8; n_literals];
        let mut ind = vec![0u8; n_literals];
        let mut cat = vec![0b0011u64]; // cat has bits 0 and 1 set
        let inc = vec![0u64; words];
        let lit = vec![0b0011u64];
        let lit_active = vec![!0u64];
        let mut rng = crate::rng::Rng::new(42);

        let active_b = expand_bits_to_bytes(&lit_active, n_literals);
        type_iii_update(
            &mut ta, &mut ind, &mut cat, &inc,
            &lit, &valid, &lit_active, &active_b,
            words, n_literals,
            200.0, 0.0, false, &mut rng, 128, 255,
        );

        // non-target fire → cat = valid = 0xF
        assert_eq!(cat[0] & 0xF, 0xF);
    }

    #[test]
    fn type_iii_update_toggles_cat_on_non_fire() {
        // Clause includes bit 0 but bit 0 is absent → clause does not fire.
        // cat bit 0 is 0 → it should be set to 1.
        let words = 1usize;
        let n_literals = 4usize;
        let valid = vec![0xFu64];
        let mut ta = vec![128u8; n_literals]; // all included (ta >= half)
        let mut ind = vec![0u8; n_literals];
        let mut cat = vec![0u64; words]; // cat starts all-zero
        let mut inc = vec![0u64; words];
        rebuild_include(&ta, &mut inc, &valid, words, n_literals, 128);

        let lit = vec![0b0000u64]; // bit 0 absent → clause fails
        let lit_active = vec![!0u64];
        let mut rng = crate::rng::Rng::new(42);

        let active_b = expand_bits_to_bytes(&lit_active, n_literals);
        type_iii_update(
            &mut ta, &mut ind, &mut cat, &inc,
            &lit, &valid, &lit_active, &active_b,
            words, n_literals,
            200.0, 0.0, true, &mut rng, 128, 255,
        );

        // bit 0 was blocking and cat[0] was 0 → cat bit 0 should be set
        assert_eq!((cat[0]) & 1, 1, "blocking literal cat bit should be set");
    }

    #[test]
    fn type_iii_update_ta_decrement_removes_low_ind_literals() {
        // When update_p=1, TA is decremented for literals with ind < half_ind.
        let words = 1usize;
        let n_literals = 2usize;
        let valid = vec![0x3u64];
        // Start ta above 0 so decrement is visible
        let mut ta = vec![5u8, 5u8];
        // ind[0] is high (above half_ind=4) → protected; ind[1] is low → decremented
        let mut ind = vec![6u8, 2u8];
        let mut cat = vec![0u64; words];
        let inc = vec![0u64; words]; // empty clause, fires
        let lit = vec![0u64]; // nothing present
        let lit_active = vec![!0u64];
        let mut rng = crate::rng::Rng::new(0);

        let active_b = expand_bits_to_bytes(&lit_active, n_literals);
        let changed = type_iii_update(
            &mut ta, &mut ind, &mut cat, &inc,
            &lit, &valid, &lit_active, &active_b,
            words, n_literals,
            200.0, 1.0, // update_p=1: always decrement
            false, &mut rng, 4, 7,
        );

        assert!(changed, "TA state should have been modified");
        assert_eq!(ta[0], 5, "ind[0] >= half_ind → ta[0] unchanged");
        assert_eq!(ta[1], 4, "ind[1] < half_ind → ta[1] decremented");
    }
}
