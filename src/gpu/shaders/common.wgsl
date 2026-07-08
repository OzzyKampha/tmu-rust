// Shared numeric core for the tmu-rs GPU kernels.
//
// WGSL has no 64-bit integers, so u64 values (RNG state, SplitMix64 output) are
// emulated as vec2<u32> with .x = low 32 bits, .y = high 32 bits. This matches
// the little-endian byte layout of the host `Vec<u64>` reinterpreted as u32, so
// the RNG streams are bit-identical to the CPU `Rng` (src/rng.rs).

// SplitMix64 constants (see src/rng.rs and src/clause_bank/dense.rs::GOLDEN).
const GOLDEN: vec2<u32> = vec2<u32>(0x7F4A7C15u, 0x9E3779B9u); // 0x9E3779B97F4A7C15
const SM_M1:  vec2<u32> = vec2<u32>(0x1CE4E5B9u, 0xBF58476Du); // 0xBF58476D1CE4E5B9
const SM_M2:  vec2<u32> = vec2<u32>(0x133111EBu, 0x94D049BBu); // 0x94D049BB133111EB

// MASK_BITS from dense.rs (Bernoulli mask precision).
const MASK_BITS: u32 = 12u;

// Model configuration shared by all kernels (a uniform buffer).
struct Config {
    n_classes: u32,
    cps: u32,          // clauses_per_class
    n_literals: u32,   // 2 * n_features
    w32: u32,          // u32 words per clause bitset = 2 * words(u64)
    threshold: u32,    // T
    half: u32,
    max_state: u32,
    boost: u32,        // 0/1
    max_inc: u32,      // max_included_literals; usize::MAX -> 0xFFFFFFFF
    dig_inv: u32,      // Bernoulli digits (bit i = digit i) for the 1/s decrement mask
    dig_keep: u32,     // digits for the (1 - 1/s) increment-keep mask
    has_lit_active: u32, // 0 -> literal dropout off (treat lit_active as all-ones)
    n_replicas: u32,     // data-parallel replica count (>=1; only the DP kernels use it)
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    let carry = select(0u, 1u, lo < a.x);
    return vec2<u32>(lo, a.y + b.y + carry);
}

fn u64_xor(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return a ^ b;
}

// Right shift by a constant k in 1..=31 (all uses are 11/27/30/31).
fn u64_shr(a: vec2<u32>, k: u32) -> vec2<u32> {
    return vec2<u32>((a.x >> k) | (a.y << (32u - k)), a.y >> k);
}

// Full 32x32 -> 64 unsigned multiply (no WGSL intrinsic), via 16-bit partials.
fn mul32(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xFFFFu;
    let a1 = a >> 16u;
    let b0 = b & 0xFFFFu;
    let b1 = b >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = p01 + (p00 >> 16u);          // <= 0xFFFFFFFF (no overflow)
    let mid2 = p10 + (mid & 0xFFFFu);
    let lo = (p00 & 0xFFFFu) | (mid2 << 16u);
    let hi = p11 + (mid >> 16u) + (mid2 >> 16u);
    return vec2<u32>(lo, hi);
}

// wrapping_mul, low 64 bits only (matches u64::wrapping_mul).
fn u64_mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    var r = mul32(a.x, b.x);
    // u32 arithmetic wraps by default in WGSL.
    r.y = r.y + a.x * b.y + a.y * b.x;
    return r;
}

// a > b for unsigned 64-bit.
fn u64_gt(a: vec2<u32>, b: vec2<u32>) -> bool {
    return (a.y > b.y) || (a.y == b.y && a.x > b.x);
}

struct RngStep {
    state: vec2<u32>,
    value: vec2<u32>,
}

// One SplitMix64 step: mirrors Rng::next_u64 exactly.
fn splitmix_next(state_in: vec2<u32>) -> RngStep {
    let state = u64_add(state_in, GOLDEN);
    var z = state;
    z = u64_mul(u64_xor(z, u64_shr(z, 30u)), SM_M1);
    z = u64_mul(u64_xor(z, u64_shr(z, 27u)), SM_M2);
    z = u64_xor(z, u64_shr(z, 31u));
    return RngStep(state, z);
}

struct MaskStep {
    state: vec2<u32>,
    word: vec2<u32>,   // 64-bit Bernoulli mask as vec2<u32>
}

// Mirrors dense.rs::bmask_word: for i in (0..MASK_BITS).rev(), draw r; if digit i
// is 1 -> word = r | word, else word = r & word. `digits` packs digit i in bit i.
fn bmask_word(state_in: vec2<u32>, digits: u32) -> MaskStep {
    var st = state_in;
    var word = vec2<u32>(0u, 0u);
    var i: i32 = i32(MASK_BITS) - 1;
    loop {
        if (i < 0) { break; }
        let step = splitmix_next(st);
        st = step.state;
        let r = step.value;
        if (((digits >> u32(i)) & 1u) == 1u) {
            word = r | word;
        } else {
            word = r & word;
        }
        i = i - 1;
    }
    return MaskStep(st, word);
}
