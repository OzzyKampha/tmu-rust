// GPU training kernels for the vanilla TsetlinMachine.
//
// Reproduces vanilla_classifier.rs::fit_one_lit exactly (bitwise), for the two
// class-slots touched per sample: slot 0 = true class y (target 1), slot 1 =
// sampled negative class (target 0). The host precomputes the shuffle order,
// negative-class samples, and literal-dropout masks (so the host `rng` /
// `literal_rng` streams stay bit-identical to CPU training); the per-clause
// `rngs` and per-class `class_rngs` streams live on-device and are advanced here.
//
// Two dispatches per sample:
//   train_prep     - 2 workgroups (one per slot): reduce the class sum
//                    (clause_fire, empty clause fires TRUE), look up the feedback
//                    threshold, and generate the inv/keep Bernoulli masks.
//   clause_update  - cps x 2 workgroups (one per clause): one RNG draw + skip
//                    test, fire check, weight update, Type I/II TA update, and
//                    include rebuild. Mirrors apply_one_clause.

@group(0) @binding(0)  var<uniform> cfg: Config;
@group(0) @binding(1)  var<storage, read_write> ta: array<u32>;
@group(0) @binding(2)  var<storage, read_write> include: array<u32>;
@group(0) @binding(3)  var<storage, read_write> weights: array<i32>;
@group(0) @binding(4)  var<storage, read_write> rngs: array<vec2<u32>>;
@group(0) @binding(5)  var<storage, read_write> class_rngs: array<vec2<u32>>;
@group(0) @binding(6)  var<storage, read> valid: array<u32>;
@group(0) @binding(7)  var<storage, read> prob_table: array<vec2<u32>>;
@group(0) @binding(8)  var<storage, read> batch_lits: array<u32>;
@group(0) @binding(9)  var<storage, read> lit_active: array<u32>;
@group(0) @binding(10) var<storage, read_write> scratch: array<u32>;
@group(1) @binding(0)  var<uniform> smp: SampleParams;

struct SampleParams {
    sample_idx: u32,  // row into batch_lits
    step_idx: u32,    // row into lit_active (processing order)
    y: u32,           // true class
    neg: u32,         // sampled negative class
}

const WG: u32 = 128u;
var<workgroup> partial: array<i32, WG>;
var<workgroup> wg_action: u32;      // 0 = none, 1 = Type I, 2 = Type II (fired)
var<workgroup> wg_fired_under: u32; // for Type I

// scratch layout (u32): [0,1]=thresh slot0 (lo,hi), [2,3]=thresh slot1,
// then 4*w32 mask words from index 4.
fn thr_lo(slot: u32) -> u32 { return 2u * slot; }
fn thr_hi(slot: u32) -> u32 { return 2u * slot + 1u; }
fn inv_off(slot: u32) -> u32 { return 4u + (2u * slot) * cfg.w32; }
fn keep_off(slot: u32) -> u32 { return 4u + (2u * slot + 1u) * cfg.w32; }

fn active_word(k: u32, la_base: u32) -> u32 {
    if (cfg.has_lit_active == 0u) {
        return 0xFFFFFFFFu;
    }
    return lit_active[la_base + k];
}

// clause_fire (training): empty clause fires TRUE; honours literal dropout.
fn clause_fire_train(cj: u32, lit_base: u32, la_base: u32) -> bool {
    let inc_base = cj * cfg.w32;
    var violation: u32 = 0u;
    for (var k: u32 = 0u; k < cfg.w32; k = k + 1u) {
        let la = active_word(k, la_base);
        violation = violation | (include[inc_base + k] & valid[k] & la & ~batch_lits[lit_base + k]);
    }
    return violation == 0u;
}

// ---- train_prep --------------------------------------------------------------

@compute @workgroup_size(WG)
fn train_prep(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let slot = wid.x; // 0 or 1
    var cls: u32;
    var tgt: u32;
    if (slot == 0u) { cls = smp.y; tgt = 1u; } else { cls = smp.neg; tgt = 0u; }

    let lit_base = smp.sample_idx * cfg.w32;
    let la_base = smp.step_idx * cfg.w32;
    let clause0 = cls * cfg.cps;

    // Reduce the (weighted) firing-clause sum over this class's clauses.
    var local: i32 = 0;
    for (var j: u32 = lid.x; j < cfg.cps; j = j + WG) {
        let cj = clause0 + j;
        if (clause_fire_train(cj, lit_base, la_base)) {
            let w = weights[cj];
            if ((j & 1u) == 0u) { local = local + w; } else { local = local - w; }
        }
    }
    partial[lid.x] = local;
    workgroupBarrier();
    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) { partial[lid.x] = partial[lid.x] + partial[lid.x + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lid.x == 0u) {
        let t = i32(cfg.threshold);
        let v = clamp(partial[0], -t, t);
        // prob_table[target][class][v + T]
        let n = cfg.n_classes;
        let span = 2u * cfg.threshold + 1u;
        let idx = ((tgt * n + cls) * span) + u32(v + t);
        let thr = prob_table[idx];
        scratch[thr_lo(slot)] = thr.x;
        scratch[thr_hi(slot)] = thr.y;

        // Generate inv then keep Bernoulli masks, advancing this class's RNG.
        // One bmask_word per u64 word (words = w32/2); stored lo,hi per word.
        let words = cfg.w32 / 2u;
        var st = class_rngs[cls];
        let ib = inv_off(slot);
        for (var k: u32 = 0u; k < words; k = k + 1u) {
            let m = bmask_word(st, cfg.dig_inv);
            st = m.state;
            scratch[ib + 2u * k] = m.word.x;
            scratch[ib + 2u * k + 1u] = m.word.y;
        }
        let kb = keep_off(slot);
        for (var k: u32 = 0u; k < words; k = k + 1u) {
            let m = bmask_word(st, cfg.dig_keep);
            st = m.state;
            scratch[kb + 2u * k] = m.word.x;
            scratch[kb + 2u * k + 1u] = m.word.y;
        }
        class_rngs[cls] = st;
    }
}

// ---- clause_update -----------------------------------------------------------

@compute @workgroup_size(WG)
fn clause_update(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let j = wid.x;         // clause within class
    let slot = wid.y;      // 0 or 1
    if (j >= cfg.cps) { return; }

    var cls: u32;
    var tgt: u32;
    if (slot == 0u) { cls = smp.y; tgt = 1u; } else { cls = smp.neg; tgt = 0u; }
    let cj = cls * cfg.cps + j;

    let lit_base = smp.sample_idx * cfg.w32;
    let la_base = smp.step_idx * cfg.w32;
    let ta_base = cj * cfg.n_literals;
    let inc_base = cj * cfg.w32;

    if (lid.x == 0u) {
        // Always advance this clause's RNG once (matches CPU: rng.next_f64() > p).
        let step = splitmix_next(rngs[cj]);
        rngs[cj] = step.state;
        let m = u64_shr(step.value, 11u);
        let thr = vec2<u32>(scratch[thr_lo(slot)], scratch[thr_hi(slot)]);
        let skip = u64_gt(m, thr);

        wg_action = 0u;
        wg_fired_under = 0u;
        if (!skip) {
            let positive = (j & 1u) == 0u;
            let is_type_i = ((tgt == 1u) == positive);
            if (is_type_i) {
                let fired = clause_fire_train(cj, lit_base, la_base);
                var n_inc: u32 = 0u;
                for (var k: u32 = 0u; k < cfg.w32; k = k + 1u) {
                    n_inc = n_inc + countOneBits(include[inc_base + k] & valid[k]);
                }
                let under = (cfg.max_inc == 0xFFFFFFFFu) || (n_inc < cfg.max_inc);
                let fired_under = fired && under;
                if (fired_under) {
                    weights[cj] = min(weights[cj] + 1, i32(cfg.threshold));
                }
                wg_action = 1u;
                wg_fired_under = select(0u, 1u, fired_under);
            } else {
                if (clause_fire_train(cj, lit_base, la_base)) {
                    weights[cj] = max(weights[cj] - 1, 1);
                    wg_action = 2u;
                }
            }
        }
    }
    workgroupBarrier();

    let action = wg_action;
    if (action == 0u) { return; }
    let fired_under = wg_fired_under == 1u;
    let ms = cfg.max_state;
    let half = cfg.half;
    let boost = cfg.boost;

    // Phase B: per-literal TA update.
    for (var l: u32 = lid.x; l < cfg.n_literals; l = l + WG) {
        let word = l / 32u;
        let bit = l % 32u;
        let present = (batch_lits[lit_base + word] >> bit) & 1u;
        var la: u32 = 1u;
        if (cfg.has_lit_active != 0u) {
            la = (lit_active[la_base + word] >> bit) & 1u;
        }
        let t = ta[ta_base + l];
        var nt = t;
        if (action == 1u) {
            let inv = (scratch[inv_off(slot) + word] >> bit) & 1u;
            let keep = (scratch[keep_off(slot) + word] >> bit) & 1u;
            if (fired_under) {
                let inc = present & (boost | keep) & la;
                let not_at_max = select(0u, 1u, t < ms);
                let dec = (1u - present) & inv & not_at_max & la;
                var v = min(t + inc, ms);
                nt = select(v - dec, 0u, v < dec);
            } else {
                let not_at_max = select(0u, 1u, t < ms);
                let dec = inv & not_at_max & la;
                nt = select(t - dec, 0u, t < dec);
            }
        } else { // action == 2 (Type II, fired)
            let absent = 1u - present;
            let excluded = select(0u, 1u, t < half);
            let not_zero = select(0u, 1u, t > 0u);
            let inc = absent & excluded & not_zero & la;
            nt = min(t + inc, ms);
        }
        ta[ta_base + l] = nt;
    }
    workgroupBarrier();

    // Phase C: rebuild the include bitset from the updated TA counters.
    for (var k: u32 = lid.x; k < cfg.w32; k = k + WG) {
        let base = k * 32u;
        var limit: u32 = 32u;
        if (cfg.n_literals < base + 32u) {
            if (cfg.n_literals > base) { limit = cfg.n_literals - base; } else { limit = 0u; }
        }
        var w: u32 = 0u;
        for (var bit: u32 = 0u; bit < limit; bit = bit + 1u) {
            if (ta[ta_base + base + bit] >= half) {
                w = w | (1u << bit);
            }
        }
        include[inc_base + k] = w & valid[k];
    }
}
