// CUDA kernels for every tmu-rs model (training + inference).
//
// Compiled at runtime with NVRTC for real GPUs, or as plain C++ (with emu.h)
// for the CPU emulator (`--features cuda-emu`). Keep to the common subset:
// no headers, no warp intrinsics, block size any value in 1..=256.
//
// Training kernels reproduce the Rust CPU training loop *bit for bit*: every
// per-clause / per-class random stream is the same SplitMix64 stream the CPU
// model would consume, draws happen in the same order, f64 feedback
// probabilities are evaluated with the same operations (no FMA contraction),
// and TA updates use the same saturating u8 arithmetic. The host side therefore
// only has to reproduce the *orchestration* (shuffle, negative sampling, ...).
//
// Conventions
// * Every kernel takes one parameter struct by value; each struct is mirrored by
//   a `#[repr(C)]` struct of the same name on the Rust side. Fields are ordered
//   8-byte, then 4-byte, then 1-byte so both layouts agree without padding games.
// * Literal layout matches `clause_bank::dense::pack`: positive literal i is bit
//   i, negated literal i is bit n_features + i; 64 literals per u64 word.
// * "Clause blocks" (one block per clause) process whole 64-literal words per
//   thread, so include words can be rebuilt without races.

#ifdef TMU_EMU
#include "emu.h"
#endif

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned char u8;

#define GOLDEN 0x9E3779B97F4A7C15ULL
#define MAX_BLOCK 256
#define NO_LIMIT 0xFFFFFFFFFFFFFFFFULL
#define BIG_INT 0x7FFFFFFF
#define F64_EPSILON 2.220446049250313e-16

#ifndef TMU_EMU
extern __shared__ unsigned long long tmu_dyn_smem[];
#endif

__device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ int imax(int a, int b) { return a > b ? a : b; }
__device__ __forceinline__ int iclamp(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// ---------------------------------------------------------------------------
// RNG — SplitMix64, identical to `crate::rng::Rng`.
// ---------------------------------------------------------------------------

__device__ __forceinline__ u64 sm_mix(u64 z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
__device__ __forceinline__ u64 rng_next(u64* s) {
    *s += GOLDEN;
    return sm_mix(*s);
}
__device__ __forceinline__ double u64_to_f64(u64 x) {
    return (double)(x >> 11) * (1.0 / 9007199254740992.0);
}
__device__ __forceinline__ double rng_f64(u64* s) { return u64_to_f64(rng_next(s)); }
// The k-th (0-based) upcoming draw of a stream at state `s`, without advancing.
// SplitMix64 is counter-based, so a run of unconditional draws can be split
// across threads.
__device__ __forceinline__ u64 rng_peek(u64 s, u64 k) { return sm_mix(s + (k + 1ULL) * GOLDEN); }

// `clause_bank::dense::bmask_word`: one 64-bit Bernoulli mask; bit i of
// `digits` is the i-th binary digit of the probability.
__device__ u64 bmask_word(u64* s, u32 digits, int n_digits) {
    u64 w = 0;
    for (int i = n_digits - 1; i >= 0; --i) {
        u64 r = rng_next(s);
        w = ((digits >> i) & 1u) ? (r | w) : (r & w);
    }
    return w;
}

// ---------------------------------------------------------------------------
// Block helpers (all threads of the block must call these).
// ---------------------------------------------------------------------------

__device__ int block_sum(int v) {
    __shared__ int red[MAX_BLOCK];
    __shared__ int res;
    red[threadIdx.x] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        int s = 0;
        for (u32 i = 0; i < blockDim.x; i++) s += red[i];
        res = s;
    }
    __syncthreads();
    int r = res;
    __syncthreads();
    return r;
}

__device__ int block_min(int v) {
    __shared__ int red[MAX_BLOCK];
    __shared__ int res;
    red[threadIdx.x] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        int m = red[0];
        for (u32 i = 1; i < blockDim.x; i++) m = imin(m, red[i]);
        res = m;
    }
    __syncthreads();
    int r = res;
    __syncthreads();
    return r;
}

// Literal-active word `k`, with up to two literals (the autoencoder's own
// output bit and its negation) masked out. ex < 0 means "none".
__device__ __forceinline__ u64 act_word(const u64* act, int k, int ex_a, int ex_b) {
    u64 a = act[k];
    if (ex_a >= 0 && (ex_a >> 6) == k) a &= ~(1ULL << (ex_a & 63));
    if (ex_b >= 0 && (ex_b >> 6) == k) a &= ~(1ULL << (ex_b & 63));
    return a;
}

// Training-semantics fire (empty clause fires; inactive literals ignored).
__device__ bool clause_fire_t(const u64* inc, const u64* lit, const u64* valid, const u64* act,
                              int ex_a, int ex_b, int words) {
    u64 v = 0;
    for (int k = 0; k < words; k++) v |= inc[k] & valid[k] & act_word(act, k, ex_a, ex_b) & ~lit[k];
    return v == 0;
}

// Predict-semantics fire (empty clause does not fire).
__device__ bool fire_predict_t(const u64* inc, const u64* lit, const u64* valid, int words) {
    u64 viol = 0, incl = 0;
    for (int k = 0; k < words; k++) {
        u64 i = inc[k] & valid[k];
        viol |= i & ~lit[k];
        incl |= i;
    }
    return viol == 0 && incl != 0;
}

// Number of included literals of a clause (block-cooperative).
__device__ int block_included(const u64* inc, const u64* valid, int words) {
    int n = 0;
    for (int k = threadIdx.x; k < words; k += blockDim.x) n += __popcll(inc[k] & valid[k]);
    return block_sum(n);
}

// ---------------------------------------------------------------------------
// TA feedback (block-cooperative). Mirrors type_i_update_bytes /
// type_ii_update_bytes followed by rebuild_include.
// action: 0 none, 1 Type Ib, 2 Type Ia, 3 Type II. Must be block-uniform.
// ---------------------------------------------------------------------------

#define ACT_NONE 0
#define ACT_IB 1
#define ACT_IA 2
#define ACT_II 3

__device__ void ta_feedback_block(u8* ta, u64* inc, const u64* valid, const u64* lit,
                                  const u64* act, int ex_a, int ex_b, const u64* inv,
                                  const u64* keep, int words, int n_literals, int action,
                                  u32 boost, u32 half, u32 max_state) {
    if (action != ACT_NONE) {
        for (int k = threadIdx.x; k < words; k += blockDim.x) {
            u64 a = act_word(act, k, ex_a, ex_b);
            u64 lw = lit[k];
            u64 iv = inv[k];
            u64 kp = keep[k];
            int base = k * 64;
            int lim = imin(64, n_literals - base);
            u8* t = ta + base;
            u64 newinc = 0;
            for (int b = 0; b < lim; b++) {
                u32 la = (u32)(a >> b) & 1u;
                u32 present = (u32)(lw >> b) & 1u;
                u32 tv = t[b];
                u32 x;
                if (action == ACT_IA) {
                    u32 incv = present & (boost | ((u32)(kp >> b) & 1u)) & la;
                    u32 dec = (1u - present) & ((u32)(iv >> b) & 1u) & (u32)(tv < max_state) & la;
                    x = tv + incv;
                    if (x > max_state) x = max_state;
                    x = x >= dec ? x - dec : 0;
                } else if (action == ACT_IB) {
                    u32 dec = ((u32)(iv >> b) & 1u) & (u32)(tv < max_state) & la;
                    x = tv >= dec ? tv - dec : 0;
                } else {
                    u32 incv = (1u - present) & (u32)(tv < half) & (u32)(tv > 0) & la;
                    x = tv + incv;
                    if (x > max_state) x = max_state;
                }
                t[b] = (u8)x;
                newinc |= (u64)(x >= half) << b;
            }
            inc[k] = newinc & valid[k];
        }
    }
    __syncthreads();
}

// Rebuild the include words of a clause from its TA states.
__device__ void rebuild_include_block(const u8* ta, u64* inc, const u64* valid, int words,
                                      int n_literals, u32 half) {
    for (int k = threadIdx.x; k < words; k += blockDim.x) {
        int base = k * 64;
        int lim = imin(64, n_literals - base);
        u64 w = 0;
        for (int b = 0; b < lim; b++) w |= (u64)(ta[base + b] >= half) << b;
        inc[k] = w & valid[k];
    }
}

// ---------------------------------------------------------------------------
// Type III feedback (block-cooperative). Mirrors clause_bank::dense::type_iii_update
// (indicator states `ind`, clause-and-target bitset `cat`), including its RNG draws.
// `rng` points at the clause's RNG state held in shared memory; only thread 0
// draws. Rebuilds the include words when the TA states changed.
// ---------------------------------------------------------------------------

__device__ void type_iii_block(u8* ta, u8* ind, u64* cat, u64* inc, const u64* lit,
                               const u64* valid, const u64* act, int ex_a, int ex_b, int words,
                               int n_literals, double d, double update_p, int target, u64* rng,
                               u32 half, u32 max_state) {
    __shared__ int s_fires, s_do_inc, s_upd;
    int viol = 0;
    int first = BIG_INT;
    for (int k = threadIdx.x; k < words; k += blockDim.x) {
        u64 blk = inc[k] & valid[k] & act_word(act, k, ex_a, ex_b) & ~lit[k];
        if (blk) {
            viol = 1;
            int off = k * 64 + (__ffsll((long long)blk) - 1);
            first = imin(first, off);
        }
    }
    int fires = block_sum(viol) == 0;
    first = block_min(first);
    if (threadIdx.x == 0) {
        s_fires = fires;
        s_do_inc = 0;
        if (fires) {
            s_do_inc = target && rng_f64(rng) <= 1.0 - 1.0 / d;
            s_upd = rng_f64(rng) <= update_p;
        } else {
            if (first != BIG_INT) {
                int chunk = first >> 6, pos = first & 63;
                if (((cat[chunk] >> pos) & 1ULL) == 0) {
                    cat[chunk] |= 1ULL << pos;
                } else if (target) {
                    cat[chunk] &= ~(1ULL << pos);
                }
            }
            s_upd = rng_f64(rng) <= update_p;
        }
    }
    __syncthreads();
    int do_inc = s_do_inc, upd = s_upd;
    if (s_fires) {
        for (int k = threadIdx.x; k < words; k += blockDim.x) {
            u64 a = act_word(act, k, ex_a, ex_b), c = cat[k], lw = lit[k], vw = valid[k];
            int base = k * 64;
            int lim = imin(64, n_literals - base);
            for (int b = 0; b < lim; b++) {
                u32 active = (u32)(a >> b) & 1u, cb = (u32)(c >> b) & 1u;
                u32 lb = (u32)(lw >> b) & 1u, vb = (u32)(vw >> b) & 1u;
                u32 incv = do_inc ? (active & cb & lb) : 0u;
                u32 dec = active & (1u - cb) & lb & vb;
                u32 x = (u32)ind[base + b] + incv;
                if (x > 255u) x = 255u;
                if (x > max_state) x = max_state;
                x = x >= dec ? x - dec : 0;
                ind[base + b] = (u8)x;
                if (upd) {
                    u32 tdec = active & (u32)(x < half);
                    u32 tv = ta[base + b];
                    ta[base + b] = (u8)(tv >= tdec ? tv - tdec : 0);
                }
            }
            cat[k] = target ? (~c & vw) : vw;
        }
    } else if (upd) {
        for (int k = threadIdx.x; k < words; k += blockDim.x) {
            u64 a = act_word(act, k, ex_a, ex_b);
            int base = k * 64;
            int lim = imin(64, n_literals - base);
            for (int b = 0; b < lim; b++) {
                u32 tdec = ((u32)(a >> b) & 1u) & (u32)(ind[base + b] < half);
                u32 tv = ta[base + b];
                ta[base + b] = (u8)(tv >= tdec ? tv - tdec : 0);
            }
        }
    }
    if (upd) rebuild_include_block(ta, inc, valid, words, n_literals, half);
    __syncthreads();
}

// Per-group feedback masks drawn from a class/output RNG, in CPU order:
// [clause dropout draws] → inv mask words → keep mask words.
// The dropout draws are not materialised: the stream state before them is saved
// to `drop_s0` and update kernels re-derive draw j with rng_peek.
__device__ void gen_group_masks(u64* rng_state, u64* drop_s0, u64* inv, u64* keep, int words,
                                u32 n_drop_draws, double drop_p, u32 dig_inv, u32 dig_keep,
                                int n_digits) {
    u64 s = *rng_state;
    *drop_s0 = s;
    if (drop_p > 0.0) s += (u64)n_drop_draws * GOLDEN;
    for (int k = 0; k < words; k++) inv[k] = bmask_word(&s, dig_inv, n_digits);
    for (int k = 0; k < words; k++) keep[k] = bmask_word(&s, dig_keep, n_digits);
    *rng_state = s;
}

__device__ __forceinline__ double feedback_p(int target, int sum, int threshold, double cw) {
    double t = (double)threshold, v = (double)sum;
    double p = target ? (t - v) / (2.0 * t) * cw : (t + v) / (2.0 * t) * cw;
    return p < 1.0 ? p : 1.0;
}

// ===========================================================================
// Grouped dense models: TsetlinMachine (groups = {y, neg}) and TMAutoEncoder
// (groups = every output). Group g owns clauses [gid*cpg, (gid+1)*cpg); clause
// polarity is the parity of its index within the group.
// ===========================================================================

struct GroupedFireParams {
    const u64* include;
    const u64* valid;
    const u64* lit;        // sample literals (words)
    const u64* act;        // literal-active mask (words)
    const int* weights;
    int* sums;             // [n_groups], pre-zeroed
    u8* fire;              // [n_groups * cpg]
    const u32* group_ids;  // [n_groups]; null => gid = g
    u64* group_rngs;       // per class/output RNG state
    u64* drop_s0;          // [n_groups]
    u64* inv;              // [n_groups * words]
    u64* keep;             // [n_groups * words]
    double drop_p;
    u32 n_groups, cpg, words, n_literals;
    u32 dig_inv, dig_keep, n_digits;
    u32 n_fire_blocks;
    int ae_nf;             // > 0: autoencoder (mask literals gid, gid+nf; target = lit bit gid)
    u32 _pad;
};

extern "C" __global__ void tm_grouped_fire(const GroupedFireParams p) {
    if (blockIdx.x < p.n_fire_blocks) {
        u32 t = blockIdx.x * blockDim.x + threadIdx.x;
        if (t >= p.n_groups * p.cpg) return;
        u32 g = t / p.cpg, j = t % p.cpg;
        u32 gid = p.group_ids ? p.group_ids[g] : g;
        u64 cj = (u64)gid * p.cpg + j;
        int ex_a = p.ae_nf > 0 ? (int)gid : -1, ex_b = p.ae_nf > 0 ? (int)gid + p.ae_nf : -1;
        bool f = clause_fire_t(p.include + cj * p.words, p.lit, p.valid, p.act, ex_a, ex_b, p.words);
        p.fire[t] = f;
        if (f) {
            int w = p.weights[cj];
            atomicAdd(&p.sums[g], (j & 1u) ? -w : w);
        }
    } else {
        u32 g = (blockIdx.x - p.n_fire_blocks) * blockDim.x + threadIdx.x;
        if (g >= p.n_groups) return;
        u32 gid = p.group_ids ? p.group_ids[g] : g;
        gen_group_masks(&p.group_rngs[gid], &p.drop_s0[g], p.inv + (u64)g * p.words,
                        p.keep + (u64)g * p.words, p.words, p.cpg, p.drop_p, p.dig_inv,
                        p.dig_keep, p.n_digits);
    }
}

struct GroupedUpdateParams {
    u8* ta;
    u64* include;
    int* weights;
    u64* rngs;              // per-clause RNG
    u8* ind;                // Type III (may be null when type_iii == 0)
    u64* cat;
    const u64* valid;
    const u64* lit;
    const u64* act;
    const int* sums;
    const u8* fire;
    const u32* group_ids;
    const u64* drop_s0;
    const u64* inv;
    const u64* keep;
    const double* class_weights;  // per gid; null => 1.0
    double drop_p;
    double d;
    u64 max_inc;
    u32 n_groups, cpg, words, n_literals;
    int threshold;
    int ae_nf;
    u32 boost, half, max_state, type_iii;
};

extern "C" __global__ void tm_grouped_update(const GroupedUpdateParams p) {
    __shared__ int s_action, s_skip;
    __shared__ u64 s_rng;
    u32 g = blockIdx.x / p.cpg, j = blockIdx.x % p.cpg;
    u32 gid = p.group_ids ? p.group_ids[g] : g;
    u64 cj = (u64)gid * p.cpg + j;
    u8* ta = p.ta + cj * p.n_literals;
    u64* inc = p.include + cj * p.words;
    int ex_a = p.ae_nf > 0 ? (int)gid : -1, ex_b = p.ae_nf > 0 ? (int)gid + p.ae_nf : -1;
    int target = p.ae_nf > 0 ? (int)((p.lit[gid >> 6] >> (gid & 63)) & 1ULL) : (g == 0);
    int sum = iclamp(p.sums[g], -p.threshold, p.threshold);
    double cw = p.class_weights ? p.class_weights[gid] : 1.0;
    double prob = feedback_p(target, sum, p.threshold, cw);
    int n_inc = p.max_inc != NO_LIMIT ? block_included(inc, p.valid, p.words) : 0;

    if (threadIdx.x == 0) {
        u64 s = p.rngs[cj];
        int dropped = p.drop_p > 0.0 && u64_to_f64(rng_peek(p.drop_s0[g], j)) < p.drop_p;
        int action = ACT_NONE;
        if (!dropped && rng_f64(&s) <= prob) {
            int positive = (j & 1u) == 0;
            int fired = p.fire[(u64)g * p.cpg + j];
            int* w = &p.weights[cj];
            if (target == positive) {
                int fu = fired && (p.max_inc == NO_LIMIT || (u64)n_inc < p.max_inc);
                if (fu) *w = imin(*w + 1, p.threshold);
                action = fu ? ACT_IA : ACT_IB;
            } else if (fired) {
                *w = imax(*w - 1, 1);
                action = ACT_II;
            }
        }
        s_action = action;
        s_skip = dropped;
        s_rng = s;
    }
    __syncthreads();
    ta_feedback_block(ta, inc, p.valid, p.lit, p.act, ex_a, ex_b, p.inv + (u64)g * p.words,
                      p.keep + (u64)g * p.words, p.words, p.n_literals, s_action, p.boost, p.half,
                      p.max_state);
    if (p.type_iii && !s_skip) {
        type_iii_block(ta, p.ind + cj * p.n_literals, p.cat + cj * p.words, inc, p.lit, p.valid,
                       p.act, ex_a, ex_b, p.words, p.n_literals, p.d, prob, target, &s_rng,
                       p.half, p.max_state);
    }
    if (threadIdx.x == 0) p.rngs[cj] = s_rng;
}

// ===========================================================================
// ConvolutionalTsetlinMachine: groups {y, neg}; a clause fires if it fires on
// any patch; feedback uses one uniformly drawn firing patch.
// ===========================================================================

struct ConvFireParams {
    const u64* include;
    const u64* valid;
    const u64* patches;    // [n_patches * words]
    const u64* act;
    const int* weights;
    int* sums;
    const u32* group_ids;
    u64* group_rngs;
    u64* drop_s0;
    u64* inv;
    u64* keep;
    double drop_p;
    u32 n_groups, cpg, words, n_literals;
    u32 n_patches, dig_inv, dig_keep, n_digits;
    u32 n_fire_blocks, _pad;
};

extern "C" __global__ void conv_fire(const ConvFireParams p) {
    if (blockIdx.x < p.n_fire_blocks) {
        u32 t = blockIdx.x * blockDim.x + threadIdx.x;
        if (t >= p.n_groups * p.cpg) return;
        u32 g = t / p.cpg, j = t % p.cpg;
        u64 cj = (u64)p.group_ids[g] * p.cpg + j;
        const u64* inc = p.include + cj * p.words;
        bool f = false;
        for (u32 q = 0; q < p.n_patches && !f; q++)
            f = clause_fire_t(inc, p.patches + (u64)q * p.words, p.valid, p.act, -1, -1, p.words);
        if (f) {
            int w = p.weights[cj];
            atomicAdd(&p.sums[g], (j & 1u) ? -w : w);
        }
    } else {
        u32 g = (blockIdx.x - p.n_fire_blocks) * blockDim.x + threadIdx.x;
        if (g >= p.n_groups) return;
        gen_group_masks(&p.group_rngs[p.group_ids[g]], &p.drop_s0[g], p.inv + (u64)g * p.words,
                        p.keep + (u64)g * p.words, p.words, p.cpg, p.drop_p, p.dig_inv,
                        p.dig_keep, p.n_digits);
    }
}

struct ConvUpdateParams {
    u8* ta;
    u64* include;
    int* weights;
    u64* rngs;
    u8* ind;
    u64* cat;
    const u64* valid;
    const u64* patches;
    const u64* act;
    const int* sums;
    const u32* group_ids;
    const u64* drop_s0;
    const u64* inv;
    const u64* keep;
    double drop_p;
    double d;
    u64 max_inc;
    u32 n_groups, cpg, words, n_literals;
    u32 n_patches;
    int threshold;
    u32 boost, half, max_state, type_iii;
};

// Dynamic shared memory: ceil(n_patches / 64) u64 words (per-patch fire bits).
extern "C" __global__ void conv_update(const ConvUpdateParams p) {
    __shared__ int s_action, s_skip, s_patch;
    __shared__ u64 s_rng;
    u64* pfire = tmu_dyn_smem;
    u32 g = blockIdx.x / p.cpg, j = blockIdx.x % p.cpg;
    u64 cj = (u64)p.group_ids[g] * p.cpg + j;
    u8* ta = p.ta + cj * p.n_literals;
    u64* inc = p.include + cj * p.words;
    int target = g == 0;
    int sum = iclamp(p.sums[g], -p.threshold, p.threshold);
    double prob = feedback_p(target, sum, p.threshold, 1.0);
    int n_inc = p.max_inc != NO_LIMIT ? block_included(inc, p.valid, p.words) : 0;

    int pw = (int)((p.n_patches + 63) / 64);
    int cnt = 0;
    for (int w = threadIdx.x; w < pw; w += blockDim.x) {
        u64 bits = 0;
        int lim = imin(64, (int)p.n_patches - w * 64);
        for (int b = 0; b < lim; b++) {
            u64 q = (u64)w * 64 + b;
            if (clause_fire_t(inc, p.patches + q * p.words, p.valid, p.act, -1, -1, p.words))
                bits |= 1ULL << b;
        }
        pfire[w] = bits;
        cnt += __popcll(bits);
    }
    cnt = block_sum(cnt);

    if (threadIdx.x == 0) {
        u64 s = p.rngs[cj];
        int fires = cnt > 0;
        int patch = 0;
        if (fires) {
            int k = (int)(rng_next(&s) % (u64)cnt);
            for (int w = 0; w < pw; w++) {
                int c = __popcll(pfire[w]);
                if (k < c) {
                    u64 bits = pfire[w];
                    for (int i = 0; i < k; i++) bits &= bits - 1;
                    patch = w * 64 + (__ffsll((long long)bits) - 1);
                    break;
                }
                k -= c;
            }
        }
        int dropped = p.drop_p > 0.0 && u64_to_f64(rng_peek(p.drop_s0[g], j)) < p.drop_p;
        int action = ACT_NONE;
        if (!dropped && rng_f64(&s) <= prob) {
            int positive = (j & 1u) == 0;
            int* w = &p.weights[cj];
            if (target == positive) {
                int fu = fires && (p.max_inc == NO_LIMIT || (u64)n_inc < p.max_inc);
                if (fu) *w = imin(*w + 1, p.threshold);
                action = fu ? ACT_IA : ACT_IB;
            } else if (fires) {
                *w = imax(*w - 1, 1);
                action = ACT_II;
            }
        }
        s_action = action;
        s_skip = dropped;
        s_patch = patch;
        s_rng = s;
    }
    __syncthreads();
    const u64* lit = p.patches + (u64)s_patch * p.words;
    ta_feedback_block(ta, inc, p.valid, lit, p.act, -1, -1, p.inv + (u64)g * p.words,
                      p.keep + (u64)g * p.words, p.words, p.n_literals, s_action, p.boost, p.half,
                      p.max_state);
    if (p.type_iii && !s_skip) {
        type_iii_block(ta, p.ind + cj * p.n_literals, p.cat + cj * p.words, inc, lit, p.valid,
                       p.act, -1, -1, p.words, p.n_literals, p.d, prob, target, &s_rng, p.half,
                       p.max_state);
    }
    if (threadIdx.x == 0) p.rngs[cj] = s_rng;
}

// ===========================================================================
// Shared-clause-bank models: CoalescedTsetlinMachine and TMCoalescedAutoEncoder.
// ===========================================================================

// Clause outputs for every shared clause (training semantics), optionally under
// an output mask (ex_a/ex_b), optionally accumulating sum += w_out[j] of firing
// clauses for one output (`sum` non-null).
struct SharedFireParams {
    const u64* include;
    const u64* valid;
    const u64* lit;
    const u64* act;
    const int* w_out;      // weights row for the summed output (if sum != null)
    int* sum;
    u8* fire;              // [n_clauses]
    u64* out_rng;          // coalesced AE: this output's RNG (mask generation), else null
    u64* drop_s0;
    u64* inv;
    u64* keep;
    double drop_p;
    u32 n_clauses, words, n_fire_blocks;
    u32 dig_inv, dig_keep, n_digits;
    int ex_a, ex_b;
};

extern "C" __global__ void shared_fire(const SharedFireParams p) {
    if (blockIdx.x < p.n_fire_blocks) {
        u32 j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= p.n_clauses) return;
        bool f = clause_fire_t(p.include + (u64)j * p.words, p.lit, p.valid, p.act, p.ex_a, p.ex_b,
                               p.words);
        p.fire[j] = f;
        if (f && p.sum) atomicAdd(p.sum, p.w_out[j]);
    } else if (threadIdx.x == 0 && p.out_rng) {
        gen_group_masks(p.out_rng, p.drop_s0, p.inv, p.keep, p.words, p.n_clauses, p.drop_p,
                        p.dig_inv, p.dig_keep, p.n_digits);
    }
}

// One feedback pass of a shared clause for one class/output. Mirrors
// apply_one_clause_coalesced (+ Type III when enabled). Block-cooperative;
// `active` and the pointers must be block-uniform.
__device__ void coalesced_pass(u8* ta, u64* inc, int* w, u64* rng_sh, u8* ind, u64* cat,
                               const u64* valid, const u64* lit, const u64* act, int ex_a,
                               int ex_b, const u64* inv, const u64* keep, int words,
                               int n_literals, int target, double prob, int out_j, int active,
                               u64 max_inc, double d, u32 boost, u32 half, u32 max_state,
                               u32 type_iii) {
    __shared__ int s_action;
    if (!active) return;
    int n_inc = max_inc != NO_LIMIT ? block_included(inc, valid, words) : 0;
    if (threadIdx.x == 0) {
        int positive = *w >= 0;
        int give_i = target ? positive : !positive;
        int action = ACT_NONE;
        if (rng_f64(rng_sh) <= prob) {
            if (give_i) {
                int fu = out_j && (max_inc == NO_LIMIT || (u64)n_inc < max_inc);
                action = fu ? ACT_IA : ACT_IB;
            } else if (out_j) {
                action = ACT_II;
            }
        }
        if (out_j && rng_f64(rng_sh) <= prob) {
            if (target) {
                if (*w != BIG_INT) *w += 1;
            } else {
                if (*w != -BIG_INT - 1) *w -= 1;
            }
        }
        s_action = action;
    }
    __syncthreads();
    ta_feedback_block(ta, inc, valid, lit, act, ex_a, ex_b, inv, keep, words, n_literals, s_action,
                      boost, half, max_state);
    if (type_iii)
        type_iii_block(ta, ind, cat, inc, lit, valid, act, ex_a, ex_b, words, n_literals, d, prob,
                       target, rng_sh, half, max_state);
}

// Per-sample planning for the coalesced classifier (single block): clause
// dropout from the master RNG, negative-class choice (uniform or focused),
// clamped class sums, and the y / neg feedback masks.
struct CoPlanParams {
    u64* master_rng;       // [1]
    u64* class_rngs;
    const int* weights;    // [n_classes * n_clauses]
    const u8* fire;
    u8* active;            // [n_clauses]
    int* plan;             // out: [y, neg, sum_y, sum_neg]
    u64* inv;              // [2 * words]: y then neg
    u64* keep;
    double drop_p;
    u32 n_clauses, n_classes, words, y;
    int threshold;
    u32 focused, dig_inv, dig_keep, n_digits, _pad;
};

__device__ int co_class_sum(const CoPlanParams& p, u32 c) {
    int s = 0;
    for (u32 j = threadIdx.x; j < p.n_clauses; j += blockDim.x)
        if (p.active[j] && p.fire[j]) s += p.weights[(u64)c * p.n_clauses + j];
    return iclamp(block_sum(s), -p.threshold, p.threshold);
}

// Dynamic shared memory: n_classes ints.
extern "C" __global__ void co_plan(const CoPlanParams p) {
    __shared__ int s_neg;
    int* sums = (int*)tmu_dyn_smem;
    u64 s0 = *p.master_rng;
    for (u32 j = threadIdx.x; j < p.n_clauses; j += blockDim.x)
        p.active[j] = p.drop_p > 0.0 ? (u8)(u64_to_f64(rng_peek(s0, j)) >= p.drop_p) : (u8)1;
    __syncthreads();
    if (p.focused) {
        for (u32 c = 0; c < p.n_classes; c++) {
            int v = c == p.y ? 0 : co_class_sum(p, c);
            if (threadIdx.x == 0) sums[c] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        u64 s = s0;
        if (p.drop_p > 0.0) s += (u64)p.n_clauses * GOLDEN;
        int neg = -1;
        if (p.focused) {
            double t = (double)p.threshold, total = 0.0;
            for (u32 c = 0; c < p.n_classes; c++) {
                if (c == p.y) continue;
                double pr = (t + (double)sums[c]) / (2.0 * t);
                pr = pr < 0.0 ? 0.0 : (pr > 1.0 ? 1.0 : pr);
                total += pr;
            }
            if (total > 0.0) {
                double r = rng_f64(&s) * total;
                for (u32 c = 0; c < p.n_classes && neg < 0; c++) {
                    if (c == p.y) continue;
                    double pr = (t + (double)sums[c]) / (2.0 * t);
                    pr = pr < 0.0 ? 0.0 : (pr > 1.0 ? 1.0 : pr);
                    r -= pr;
                    if (r <= 0.0) neg = (int)c;
                }
            }
        }
        if (neg < 0) {
            neg = (int)(rng_next(&s) % (u64)p.n_classes);
            while ((u32)neg == p.y) neg = (int)(rng_next(&s) % (u64)p.n_classes);
        }
        *p.master_rng = s;
        s_neg = neg;
        u64* ry = &p.class_rngs[p.y];
        for (u32 k = 0; k < p.words; k++) p.inv[k] = bmask_word(ry, p.dig_inv, p.n_digits);
        for (u32 k = 0; k < p.words; k++) p.keep[k] = bmask_word(ry, p.dig_keep, p.n_digits);
        u64* rn = &p.class_rngs[neg];
        for (u32 k = 0; k < p.words; k++) p.inv[p.words + k] = bmask_word(rn, p.dig_inv, p.n_digits);
        for (u32 k = 0; k < p.words; k++) p.keep[p.words + k] = bmask_word(rn, p.dig_keep, p.n_digits);
    }
    __syncthreads();
    int neg = s_neg;
    int sy = co_class_sum(p, p.y);
    int sn = co_class_sum(p, (u32)neg);
    if (threadIdx.x == 0) {
        p.plan[0] = (int)p.y;
        p.plan[1] = neg;
        p.plan[2] = sy;
        p.plan[3] = sn;
    }
}

struct CoUpdateParams {
    u8* ta;
    u64* include;
    int* weights;
    u64* rngs;
    u8* ind;
    u64* cat;
    const u64* valid;
    const u64* lit;
    const u64* act;
    const u8* fire;
    const u8* active;
    const int* plan;
    const u64* inv;
    const u64* keep;
    double d;
    u64 max_inc;
    u32 n_clauses, words, n_literals;
    int threshold;
    u32 boost, half, max_state, type_iii;
};

extern "C" __global__ void co_update(const CoUpdateParams p) {
    __shared__ u64 s_rng;
    __shared__ int s_w;
    u32 j = blockIdx.x;
    u8* ta = p.ta + (u64)j * p.n_literals;
    u64* inc = p.include + (u64)j * p.words;
    u8* ind = p.type_iii ? p.ind + (u64)j * p.n_literals : 0;
    u64* cat = p.type_iii ? p.cat + (u64)j * p.words : 0;
    int out_j = p.fire[j], active = p.active[j];
    if (threadIdx.x == 0) s_rng = p.rngs[j];
    for (int pass = 0; pass < 2; pass++) {
        int c = p.plan[pass];
        int target = pass == 0;
        double t = (double)p.threshold, v = (double)p.plan[2 + pass];
        double prob = target ? (t - v) / (2.0 * t) : (t + v) / (2.0 * t);
        if (threadIdx.x == 0) s_w = p.weights[(u64)c * p.n_clauses + j];
        __syncthreads();
        coalesced_pass(ta, inc, &s_w, &s_rng, ind, cat, p.valid, p.lit, p.act, -1, -1,
                       p.inv + (u64)pass * p.words, p.keep + (u64)pass * p.words, p.words,
                       p.n_literals, target, prob, out_j, active, p.max_inc, p.d, p.boost, p.half,
                       p.max_state, p.type_iii);
        __syncthreads();
        if (threadIdx.x == 0) p.weights[(u64)c * p.n_clauses + j] = s_w;
        __syncthreads();
    }
    if (threadIdx.x == 0) p.rngs[j] = s_rng;
}

// Coalesced autoencoder: per-sample confidence gate (TMU `update_clause`).
struct CaeGateParams {
    const int* weights;    // [n_features * n_clauses]
    u8* gate;              // [n_clauses]
    u64 gate_rng;          // state of clause_gate_rng before this sample
    u32 n_clauses, n_features;
    int threshold;
    u32 _pad;
};

extern "C" __global__ void cae_gate(const CaeGateParams p) {
    u32 j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= p.n_clauses) return;
    double avg = 0.0;
    for (u32 o = 0; o < p.n_features; o++) {
        int w = p.weights[(u64)o * p.n_clauses + j];
        avg += (double)(w < 0 ? -w : w);
    }
    avg /= (double)p.n_features;
    double t = (double)p.threshold;
    double pu = (t - (avg < t ? avg : t)) / t;
    p.gate[j] = u64_to_f64(rng_peek(p.gate_rng, j)) < pu;
}

struct CaeUpdateParams {
    u8* ta;
    u64* include;
    int* weights;          // row of output o: [n_clauses]
    u64* rngs;
    u8* ind;
    u64* cat;
    const u64* valid;
    const u64* lit;
    const u64* act;
    const u8* fire;
    const u8* gate;
    const int* sum;
    const u64* drop_s0;
    const u64* inv;
    const u64* keep;
    double drop_p;
    double d;
    u64 max_inc;
    u32 n_clauses, words, n_literals, o;
    int threshold;
    int nf;
    u32 boost, half, max_state, type_iii;
};

extern "C" __global__ void cae_update(const CaeUpdateParams p) {
    __shared__ u64 s_rng;
    __shared__ int s_w;
    u32 j = blockIdx.x;
    u8* ta = p.ta + (u64)j * p.n_literals;
    u64* inc = p.include + (u64)j * p.words;
    int target = (int)((p.lit[p.o >> 6] >> (p.o & 63)) & 1ULL);
    int sum = iclamp(*p.sum, -p.threshold, p.threshold);
    double prob = feedback_p(target, sum, p.threshold, 1.0);
    int not_dropped = !(p.drop_p > 0.0) || u64_to_f64(rng_peek(*p.drop_s0, j)) >= p.drop_p;
    int active = not_dropped && p.gate[j];
    if (threadIdx.x == 0) {
        s_rng = p.rngs[j];
        s_w = p.weights[j];
    }
    __syncthreads();
    coalesced_pass(ta, inc, &s_w, &s_rng, p.type_iii ? p.ind + (u64)j * p.n_literals : 0,
                   p.type_iii ? p.cat + (u64)j * p.words : 0, p.valid, p.lit, p.act, (int)p.o,
                   (int)p.o + p.nf, p.inv, p.keep, p.words, p.n_literals, target, prob, p.fire[j],
                   active, p.max_inc, p.d, p.boost, p.half, p.max_state, p.type_iii);
    __syncthreads();
    if (threadIdx.x == 0) {
        p.rngs[j] = s_rng;
        p.weights[j] = s_w;
    }
}

// ===========================================================================
// TMRegressor.
// ===========================================================================

struct RgFireParams {
    const u64* include;
    const u64* valid;
    const u64* lit;
    const int* weights;
    int* pred_sum;         // [1], pre-zeroed (predict semantics)
    u32 n_clauses, words;
};

extern "C" __global__ void rg_fire(const RgFireParams p) {
    u32 j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= p.n_clauses) return;
    if (fire_predict_t(p.include + (u64)j * p.words, p.lit, p.valid, p.words))
        atomicAdd(p.pred_sum, p.weights[j]);
}

// Single-thread planning, in CPU order: update probability (skip when ~0, in
// which case no RNG is consumed), literal-dropout mask (literal RNG), then the
// inv / keep masks and clause-dropout draws (master RNG).
struct RgPlanParams {
    u64* master_rng;
    u64* literal_rng;
    const int* pred_sum;
    double* plan_p;        // out: update_p
    int* plan;             // out: [skip, push_up]
    u64* act;              // out: literal-active mask (words)
    u64* drop_s0;
    u64* inv;
    u64* keep;
    double y;
    double drop_p;
    u32 n_clauses, words;
    int threshold;
    u32 dig_inv, dig_keep, dig_lit, n_digits, literal_drop;
};

extern "C" __global__ void rg_plan(const RgPlanParams p) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double v = (double)iclamp(*p.pred_sum, 0, p.threshold);
    double t = (double)p.threshold;
    double err = (v - p.y) / t;
    double up = err * err;
    up = up < 1.0 ? up : 1.0;
    if (up < F64_EPSILON) {
        p.plan[0] = 1;
        return;
    }
    p.plan[0] = 0;
    p.plan[1] = p.y > v;
    *p.plan_p = up;
    for (u32 k = 0; k < p.words; k++)
        p.act[k] = p.literal_drop ? bmask_word(p.literal_rng, p.dig_lit, p.n_digits) : ~0ULL;
    u64 s = *p.master_rng;
    for (u32 k = 0; k < p.words; k++) p.inv[k] = bmask_word(&s, p.dig_inv, p.n_digits);
    for (u32 k = 0; k < p.words; k++) p.keep[k] = bmask_word(&s, p.dig_keep, p.n_digits);
    *p.drop_s0 = s;
    if (p.drop_p > 0.0) s += (u64)p.n_clauses * GOLDEN;
    *p.master_rng = s;
}

struct RgUpdateParams {
    u8* ta;
    u64* include;
    int* weights;
    u64* rngs;
    u8* ind;
    u64* cat;
    const u64* valid;
    const u64* lit;
    const u64* act;
    const int* plan;
    const double* plan_p;
    const u64* drop_s0;
    const u64* inv;
    const u64* keep;
    double drop_p;
    double d;
    u64 max_inc;
    u32 n_clauses, words, n_literals;
    int threshold;
    u32 boost, half, max_state, type_iii;
};

extern "C" __global__ void rg_update(const RgUpdateParams p) {
    __shared__ int s_action, s_skip;
    __shared__ u64 s_rng;
    if (p.plan[0]) return;
    u32 j = blockIdx.x;
    u8* ta = p.ta + (u64)j * p.n_literals;
    u64* inc = p.include + (u64)j * p.words;
    int push_up = p.plan[1];
    double up = *p.plan_p;
    int n_inc = p.max_inc != NO_LIMIT ? block_included(inc, p.valid, p.words) : 0;
    int viol = 0;
    for (int k = threadIdx.x; k < (int)p.words; k += blockDim.x)
        viol |= (inc[k] & p.valid[k] & p.act[k] & ~p.lit[k]) != 0;
    int fired = block_sum(viol) == 0;
    if (threadIdx.x == 0) {
        u64 s = p.rngs[j];
        int dropped = p.drop_p > 0.0 && u64_to_f64(rng_peek(*p.drop_s0, j)) < p.drop_p;
        int action = ACT_NONE;
        if (!dropped && rng_f64(&s) <= up) {
            int* w = &p.weights[j];
            if (push_up) {
                int fu = fired && (p.max_inc == NO_LIMIT || (u64)n_inc < p.max_inc);
                if (fu) *w = imin(*w + 1, p.threshold);
                action = fu ? ACT_IA : ACT_IB;
            } else if (fired) {
                *w = imax(*w - 1, 0);
                action = ACT_II;
            }
        }
        s_action = action;
        s_skip = dropped;
        s_rng = s;
    }
    __syncthreads();
    ta_feedback_block(ta, inc, p.valid, p.lit, p.act, -1, -1, p.inv, p.keep, p.words, p.n_literals,
                      s_action, p.boost, p.half, p.max_state);
    if (p.type_iii && !s_skip) {
        type_iii_block(ta, p.ind + (u64)j * p.n_literals, p.cat + (u64)j * p.words, inc, p.lit,
                       p.valid, p.act, -1, -1, p.words, p.n_literals, p.d, up, push_up, &s_rng,
                       p.half, p.max_state);
    }
    if (threadIdx.x == 0) p.rngs[j] = s_rng;
}

// ===========================================================================
// TMSparseClassifier: per-clause index lists with absorbing actions. One thread
// per clause runs the exact sequential list algorithm of SparseClause.
// Clause cj's lists live at [cj * n_literals, (cj + 1) * n_literals).
// ===========================================================================

struct SparseBank {
    u32* inc_idx;
    u32* exc_idx;
    u32* una_idx;
    u8* inc_st;
    u8* exc_st;
    u32* n_inc;
    u32* n_exc;
    u32* n_una;
};

__device__ __forceinline__ bool bit_of(const u64* v, u32 l) { return (v[l >> 6] >> (l & 63)) & 1ULL; }

__device__ bool sparse_fire_train(const SparseBank& b, u64 cj, u32 nl, const u64* lit,
                                  const u64* act) {
    u32 n = b.n_inc[cj];
    const u32* idx = b.inc_idx + cj * nl;
    for (u32 i = 0; i < n; i++)
        if (bit_of(act, idx[i]) && !bit_of(lit, idx[i])) return false;
    return true;
}

__device__ bool sparse_fire_predict(const SparseBank& b, u64 cj, u32 nl, const u64* lit) {
    u32 n = b.n_inc[cj];
    if (n == 0) return false;
    const u32* idx = b.inc_idx + cj * nl;
    for (u32 i = 0; i < n; i++)
        if (!bit_of(lit, idx[i])) return false;
    return true;
}

struct SpFireParams {
    SparseBank bank;
    const u64* lit;
    const u64* act;
    const int* weights;
    int* sums;
    u8* fire;
    const u32* group_ids;  // [2] = {y, neg}
    u64* class_rngs;
    u64* drop_s0;          // [2]
    double drop_p;
    u32 n_groups, cpg, n_literals, n_fire_blocks;
};

extern "C" __global__ void sp_fire(const SpFireParams p) {
    if (blockIdx.x < p.n_fire_blocks) {
        u32 t = blockIdx.x * blockDim.x + threadIdx.x;
        if (t >= p.n_groups * p.cpg) return;
        u32 g = t / p.cpg, j = t % p.cpg;
        u64 cj = (u64)p.group_ids[g] * p.cpg + j;
        bool f = sparse_fire_train(p.bank, cj, p.n_literals, p.lit, p.act);
        p.fire[t] = f;
        if (f) {
            int w = p.weights[cj];
            atomicAdd(&p.sums[g], (j & 1u) ? -w : w);
        }
    } else {
        u32 g = (blockIdx.x - p.n_fire_blocks) * blockDim.x + threadIdx.x;
        if (g >= p.n_groups) return;
        u64* r = &p.class_rngs[p.group_ids[g]];
        p.drop_s0[g] = *r;
        if (p.drop_p > 0.0) *r += (u64)p.cpg * GOLDEN;
    }
}

__device__ __forceinline__ void sp_swap_remove(u32* idx, u8* st, u32* n, u32 i) {
    u32 last = *n - 1;
    idx[i] = idx[last];
    st[i] = st[last];
    *n = last;
}

struct SpUpdateParams {
    SparseBank bank;
    int* weights;
    u64* rngs;
    const u64* lit;
    const u64* act;
    const int* sums;
    const u8* fire;
    const u32* group_ids;
    const u64* drop_s0;
    const double* class_weights;
    double drop_p;
    double inv_p, keep_p;
    u64 max_inc;
    u32 n_groups, cpg, n_literals;
    int threshold;
    u32 boost, half, max_state, _pad;
};

extern "C" __global__ void sp_update(const SpUpdateParams p) {
    u32 t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= p.n_groups * p.cpg) return;
    u32 g = t / p.cpg, j = t % p.cpg;
    u32 gid = p.group_ids[g];
    u64 cj = (u64)gid * p.cpg + j;
    int target = g == 0;
    int sum = iclamp(p.sums[g], -p.threshold, p.threshold);
    double prob = feedback_p(target, sum, p.threshold, p.class_weights[gid]);
    if (p.drop_p > 0.0 && u64_to_f64(rng_peek(p.drop_s0[g], j)) < p.drop_p) return;
    u64 s = p.rngs[cj];
    if (rng_f64(&s) > prob) {
        p.rngs[cj] = s;
        return;
    }
    const SparseBank& b = p.bank;
    u32 nl = p.n_literals;
    u32* inc_idx = b.inc_idx + cj * nl;
    u8* inc_st = b.inc_st + cj * nl;
    u32* exc_idx = b.exc_idx + cj * nl;
    u8* exc_st = b.exc_st + cj * nl;
    u32* una_idx = b.una_idx + cj * nl;
    u32 n_inc = b.n_inc[cj], n_exc = b.n_exc[cj], n_una = b.n_una[cj];
    int fired = p.fire[t];
    int positive = (j & 1u) == 0;
    int* w = &p.weights[cj];
    if (target == positive) {
        int fu = fired && (p.max_inc == NO_LIMIT || (u64)n_inc < p.max_inc);
        if (fu) *w = imin(*w + 1, p.threshold);
        // Type I — included literals.
        u32 i = 0;
        while (i < n_inc) {
            u32 l = inc_idx[i];
            if (!bit_of(p.act, l)) {
                i++;
                continue;
            }
            u32 st = inc_st[i];
            if (bit_of(p.lit, l)) {
                if (fu && st < p.max_state && (p.boost || rng_f64(&s) <= p.keep_p)) inc_st[i] = (u8)(st + 1);
                i++;
            } else {
                if (st < p.max_state && rng_f64(&s) <= p.inv_p) {
                    u32 ns = st - 1;
                    if (ns < p.half) {
                        sp_swap_remove(inc_idx, inc_st, &n_inc, i);
                        exc_idx[n_exc] = l;
                        exc_st[n_exc] = (u8)ns;
                        n_exc++;
                        continue;
                    }
                    inc_st[i] = (u8)ns;
                }
                i++;
            }
        }
        // Type I — excluded literals.
        i = 0;
        while (i < n_exc) {
            u32 l = exc_idx[i];
            if (!bit_of(p.act, l)) {
                i++;
                continue;
            }
            u32 st = exc_st[i];
            if (bit_of(p.lit, l)) {
                if (fu && (p.max_inc == NO_LIMIT || (u64)n_inc < p.max_inc) &&
                    (p.boost || rng_f64(&s) <= p.keep_p)) {
                    u32 ns = st + 1 < p.max_state ? st + 1 : p.max_state;
                    if (ns >= p.half) {
                        sp_swap_remove(exc_idx, exc_st, &n_exc, i);
                        inc_idx[n_inc] = l;
                        inc_st[n_inc] = (u8)ns;
                        n_inc++;
                        continue;
                    }
                    exc_st[i] = (u8)ns;
                }
                i++;
            } else {
                if (rng_f64(&s) <= p.inv_p) {
                    if (st <= 1) {
                        sp_swap_remove(exc_idx, exc_st, &n_exc, i);
                        una_idx[n_una++] = l;
                        continue;
                    }
                    exc_st[i] = (u8)(st - 1);
                }
                i++;
            }
        }
    } else if (fired) {
        *w = imax(*w - 1, 1);
        // Type II — excluded literals only.
        u32 i = 0;
        while (i < n_exc) {
            u32 l = exc_idx[i];
            u32 st = exc_st[i];
            if (bit_of(p.act, l) && !bit_of(p.lit, l) && st > 0) {
                u32 ns = st + 1 < p.max_state ? st + 1 : p.max_state;
                if (ns >= p.half) {
                    sp_swap_remove(exc_idx, exc_st, &n_exc, i);
                    inc_idx[n_inc] = l;
                    inc_st[n_inc] = (u8)ns;
                    n_inc++;
                    continue;
                }
                exc_st[i] = (u8)ns;
            }
            i++;
        }
    }
    b.n_inc[cj] = n_inc;
    b.n_exc[cj] = n_exc;
    b.n_una[cj] = n_una;
    p.rngs[cj] = s;
}

// ===========================================================================
// Inference: per-sample output scores (raw, unclamped — the host applies the
// model-specific clamp / threshold / argmax).
// ===========================================================================

#define SCORE_GROUPED 0  // output c = Σ_{j in group c} ±w[j] (parity polarity)
#define SCORE_SHARED 1   // output o = Σ_{j fired} w[o * n_clauses + j]

struct ScoresParams {
    const u64* include;
    const u64* valid;
    const u64* lits;       // [n_samples * n_patches * words]
    const int* weights;
    int* out;              // [n_samples * n_outputs]; pre-zeroed when global_acc
    u32 n_clauses, words, n_outputs, cpg;
    u32 n_patches, mode;
    u32 global_acc;        // accumulate straight into `out` (too many outputs for shared memory)
    u32 _pad;
};

// One block per sample. Dynamic shared memory: n_outputs ints (unless global_acc).
extern "C" __global__ void dense_scores(const ScoresParams p) {
    u64 s = blockIdx.x;
    int* acc = p.global_acc ? p.out + s * p.n_outputs : (int*)tmu_dyn_smem;
    if (!p.global_acc)
        for (u32 o = threadIdx.x; o < p.n_outputs; o += blockDim.x) acc[o] = 0;
    __syncthreads();
    const u64* lits = p.lits + s * p.n_patches * p.words;
    for (u32 j = threadIdx.x; j < p.n_clauses; j += blockDim.x) {
        const u64* inc = p.include + (u64)j * p.words;
        bool f = false;
        for (u32 q = 0; q < p.n_patches && !f; q++)
            f = fire_predict_t(inc, lits + (u64)q * p.words, p.valid, p.words);
        if (!f) continue;
        if (p.mode == SCORE_GROUPED) {
            int w = p.weights[j];
            atomicAdd(&acc[j / p.cpg], ((j % p.cpg) & 1u) ? -w : w);
        } else {
            for (u32 o = 0; o < p.n_outputs; o++) atomicAdd(&acc[o], p.weights[(u64)o * p.n_clauses + j]);
        }
    }
    __syncthreads();
    if (!p.global_acc)
        for (u32 o = threadIdx.x; o < p.n_outputs; o += blockDim.x) p.out[s * p.n_outputs + o] = acc[o];
}

struct SparseScoresParams {
    SparseBank bank;
    const u64* lits;       // [n_samples * words]
    const int* weights;
    int* out;              // [n_samples * n_classes]
    u32 n_clauses, words, n_classes, cpg;
    u32 n_literals, _pad;
};

extern "C" __global__ void sparse_scores(const SparseScoresParams p) {
    int* acc = (int*)tmu_dyn_smem;
    u64 s = blockIdx.x;
    for (u32 o = threadIdx.x; o < p.n_classes; o += blockDim.x) acc[o] = 0;
    __syncthreads();
    const u64* lit = p.lits + s * p.words;
    for (u32 j = threadIdx.x; j < p.n_clauses; j += blockDim.x) {
        if (!sparse_fire_predict(p.bank, j, p.n_literals, lit)) continue;
        int w = p.weights[j];
        atomicAdd(&acc[j / p.cpg], ((j % p.cpg) & 1u) ? -w : w);
    }
    __syncthreads();
    for (u32 o = threadIdx.x; o < p.n_classes; o += blockDim.x) p.out[s * p.n_classes + o] = acc[o];
}

// ===========================================================================
// CPU emulator entry point.
// ===========================================================================

#ifdef TMU_EMU
#include <mutex>

thread_local tmu_dim3 threadIdx;
tmu_dim3 blockIdx, blockDim, gridDim;
unsigned long long* tmu_dyn_smem;
std::barrier<>* tmu_block_barrier;

#define TMU_K(name, P) {#name, [](const void* q) { name(*(const P*)q); }, sizeof(P)},
static const TmuEmuEntry tmu_emu_table[] = {
    TMU_K(tm_grouped_fire, GroupedFireParams)
    TMU_K(tm_grouped_update, GroupedUpdateParams)
    TMU_K(conv_fire, ConvFireParams)
    TMU_K(conv_update, ConvUpdateParams)
    TMU_K(shared_fire, SharedFireParams)
    TMU_K(co_plan, CoPlanParams)
    TMU_K(co_update, CoUpdateParams)
    TMU_K(cae_gate, CaeGateParams)
    TMU_K(cae_update, CaeUpdateParams)
    TMU_K(rg_fire, RgFireParams)
    TMU_K(rg_plan, RgPlanParams)
    TMU_K(rg_update, RgUpdateParams)
    TMU_K(sp_fire, SpFireParams)
    TMU_K(sp_update, SpUpdateParams)
    TMU_K(dense_scores, ScoresParams)
    TMU_K(sparse_scores, SparseScoresParams)
};

extern "C" int tmu_emu_launch(const char* name, unsigned grid, unsigned block, unsigned smem,
                              const void* params, size_t param_size) {
    static std::mutex launch_lock;  // one kernel at a time: emulator state is global
    for (const TmuEmuEntry& e : tmu_emu_table) {
        if (strcmp(e.name, name) != 0) continue;
        if (e.param_size != param_size) return 2;
        std::lock_guard<std::mutex> guard(launch_lock);
        std::vector<unsigned long long> dyn(smem / 8 + 2);
        gridDim = {grid, 1, 1};
        blockDim = {block, 1, 1};
        tmu_dyn_smem = dyn.data();
        for (unsigned b = 0; b < grid; b++) {
            blockIdx = {b, 0, 0};
            if (block == 1) {
                threadIdx = {0, 0, 0};
                e.fn(params);
            } else {
                std::barrier<> bar((std::ptrdiff_t)block);
                tmu_block_barrier = &bar;
                std::vector<std::thread> ts;
                ts.reserve(block);
                for (unsigned t = 0; t < block; t++) {
                    ts.emplace_back([&, t] {
                        threadIdx = {t, 0, 0};
                        e.fn(params);
                    });
                }
                for (auto& th : ts) th.join();
            }
        }
        return 0;
    }
    return 1;
}
#endif
