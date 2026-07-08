// Data-parallel (approximate) GPU training: R replicas trained in lockstep,
// then merged by averaging TA counters + weights and rebuilding include bitsets.
// Mirrors the CPU `fit_epoch_data_parallel` path. Each replica r owns its own
// slice of every state buffer (indexed by a replica offset); at super-step s
// replica r trains on its shard's s-th sample (from `dp`), advancing its own
// on-device per-clause `rngs` and per-class `class_rngs` (both live in `rngs_all`:
// clause rngs first, then class rngs at `n_replicas * n_clauses`).
//
// Replica state buffers (bindings 1-4, 9) hold R copies; the canonical merge
// targets (bindings 10-12) are the single-model buffers read back to the host.

@group(0) @binding(0) var<uniform> cfg: Config;
@group(0) @binding(1) var<storage, read_write> ta: array<u32>;          // replica ta (R x)
@group(0) @binding(2) var<storage, read_write> include: array<u32>;      // replica include (R x)
@group(0) @binding(3) var<storage, read_write> weights: array<i32>;      // replica weights (R x)
@group(0) @binding(4) var<storage, read_write> rngs_all: array<vec2<u32>>; // clause rngs then class rngs
@group(0) @binding(5) var<storage, read> valid: array<u32>;
@group(0) @binding(6) var<storage, read> prob_table: array<vec2<u32>>;
@group(0) @binding(7) var<storage, read> batch_lits: array<u32>;
@group(0) @binding(8) var<storage, read> lit_active: array<u32>;
@group(0) @binding(9) var<storage, read_write> scratch: array<u32>;      // R x (4 + 4*w32)
@group(0) @binding(10) var<storage, read_write> dst_ta: array<u32>;      // merge target
@group(0) @binding(11) var<storage, read_write> dst_include: array<u32>;
@group(0) @binding(12) var<storage, read_write> dst_weights: array<i32>;
@group(1) @binding(0) var<uniform> dp: DpParams;

// One entry per replica for the current super-step. `sample_idx == 0xFFFFFFFF`
// means this replica's shard is exhausted (skip). MAX_R must match the host.
struct SampleDP {
    sample_idx: u32,
    y: u32,
    neg: u32,
    la_row: u32,
}
struct DpParams {
    s: array<SampleDP, 64>,
}

const WG: u32 = 128u;
var<workgroup> partial: array<i32, WG>;
var<workgroup> wg_action: u32;
var<workgroup> wg_fired_under: u32;

fn n_clauses_total() -> u32 { return cfg.n_classes * cfg.cps; }
fn class_rng_base() -> u32 { return cfg.n_replicas * n_clauses_total(); }
fn scratch_stride() -> u32 { return 4u + 4u * cfg.w32; }

fn active_word(k: u32, la_base: u32) -> u32 {
    if (cfg.has_lit_active == 0u) { return 0xFFFFFFFFu; }
    return lit_active[la_base + k];
}

fn clause_fire_train(cj: u32, lit_base: u32, la_base: u32) -> bool {
    let inc_base = cj * cfg.w32;
    var v: u32 = 0u;
    for (var k: u32 = 0u; k < cfg.w32; k = k + 1u) {
        v = v | (include[inc_base + k] & valid[k] & active_word(k, la_base) & ~batch_lits[lit_base + k]);
    }
    return v == 0u;
}

// ---- train_prep_dp: workgroups (2 slots, R replicas) -------------------------

@compute @workgroup_size(WG)
fn train_prep_dp(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let slot = wid.x;      // 0 or 1
    let replica = wid.y;   // 0..R
    let sp = dp.s[replica];
    if (sp.sample_idx == 0xFFFFFFFFu) { return; }

    var cls: u32;
    var tgt: u32;
    if (slot == 0u) { cls = sp.y; tgt = 1u; } else { cls = sp.neg; tgt = 0u; }

    let lit_base = sp.sample_idx * cfg.w32;
    let la_base = sp.la_row * cfg.w32;
    let clause0 = replica * n_clauses_total() + cls * cfg.cps;

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
        let sbase = replica * scratch_stride();
        let t = i32(cfg.threshold);
        let v = clamp(partial[0], -t, t);
        let span = 2u * cfg.threshold + 1u;
        let idx = ((tgt * cfg.n_classes + cls) * span) + u32(v + t);
        let thr = prob_table[idx];
        scratch[sbase + 2u * slot] = thr.x;
        scratch[sbase + 2u * slot + 1u] = thr.y;

        let words = cfg.w32 / 2u;
        let crng_idx = class_rng_base() + replica * cfg.n_classes + cls;
        var st = rngs_all[crng_idx];
        let ib = sbase + 4u + (2u * slot) * cfg.w32;
        for (var k: u32 = 0u; k < words; k = k + 1u) {
            let m = bmask_word(st, cfg.dig_inv);
            st = m.state;
            scratch[ib + 2u * k] = m.word.x;
            scratch[ib + 2u * k + 1u] = m.word.y;
        }
        let kb = sbase + 4u + (2u * slot + 1u) * cfg.w32;
        for (var k: u32 = 0u; k < words; k = k + 1u) {
            let m = bmask_word(st, cfg.dig_keep);
            st = m.state;
            scratch[kb + 2u * k] = m.word.x;
            scratch[kb + 2u * k + 1u] = m.word.y;
        }
        rngs_all[crng_idx] = st;
    }
}

// ---- clause_update_dp: workgroups (cps, 2 slots, R replicas) ------------------

@compute @workgroup_size(WG)
fn clause_update_dp(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let j = wid.x;
    let slot = wid.y;
    let replica = wid.z;
    if (j >= cfg.cps) { return; }
    let sp = dp.s[replica];
    if (sp.sample_idx == 0xFFFFFFFFu) { return; }

    var cls: u32;
    var tgt: u32;
    if (slot == 0u) { cls = sp.y; tgt = 1u; } else { cls = sp.neg; tgt = 0u; }
    let cj = replica * n_clauses_total() + cls * cfg.cps + j;

    let lit_base = sp.sample_idx * cfg.w32;
    let la_base = sp.la_row * cfg.w32;
    let ta_base = cj * cfg.n_literals;
    let inc_base = cj * cfg.w32;
    let sbase = replica * scratch_stride();

    if (lid.x == 0u) {
        let step = splitmix_next(rngs_all[cj]);
        rngs_all[cj] = step.state;
        let m = u64_shr(step.value, 11u);
        let thr = vec2<u32>(scratch[sbase + 2u * slot], scratch[sbase + 2u * slot + 1u]);
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
            let inv = (scratch[sbase + 4u + (2u * slot) * cfg.w32 + word] >> bit) & 1u;
            let keep = (scratch[sbase + 4u + (2u * slot + 1u) * cfg.w32 + word] >> bit) & 1u;
            if (fired_under) {
                let inc = present & (boost | keep) & la;
                let not_at_max = select(0u, 1u, t < ms);
                let dec = (1u - present) & inv & not_at_max & la;
                let vv = min(t + inc, ms);
                nt = select(vv - dec, 0u, vv < dec);
            } else {
                let not_at_max = select(0u, 1u, t < ms);
                let dec = inv & not_at_max & la;
                nt = select(t - dec, 0u, t < dec);
            }
        } else {
            let absent = 1u - present;
            let excluded = select(0u, 1u, t < half);
            let not_zero = select(0u, 1u, t > 0u);
            let inc = absent & excluded & not_zero & la;
            nt = min(t + inc, ms);
        }
        ta[ta_base + l] = nt;
    }
    workgroupBarrier();

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

// ---- merge kernels -----------------------------------------------------------
// Average the R replicas into the canonical (dst_*) buffers, matching CPU
// fit_epoch_data_parallel: rounded mean of TA counters and weights, then rebuild.

// avg_ta: grid-stride over canonical TA counters (rounded mean of R replicas).
@compute @workgroup_size(WG)
fn avg_ta(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let total = n_clauses_total() * cfg.n_literals;
    let r = cfg.n_replicas;
    let stride = ng.x * WG;
    var i = gid.x;
    loop {
        if (i >= total) { break; }
        var sum: u32 = 0u;
        for (var rr: u32 = 0u; rr < r; rr = rr + 1u) {
            sum = sum + ta[rr * total + i];
        }
        dst_ta[i] = (sum + r / 2u) / r;
        i = i + stride;
    }
}

// avg_weights: grid-stride; rounded mean clamped to [1, T].
@compute @workgroup_size(WG)
fn avg_weights(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let total = n_clauses_total();
    let r = cfg.n_replicas;
    let stride = ng.x * WG;
    var i = gid.x;
    loop {
        if (i >= total) { break; }
        var sum: i32 = 0;
        for (var rr: u32 = 0u; rr < r; rr = rr + 1u) {
            sum = sum + weights[rr * total + i];
        }
        let avg = (sum + i32(r) / 2) / i32(r);
        dst_weights[i] = clamp(avg, 1, i32(cfg.threshold));
        i = i + stride;
    }
}

// merge_rebuild: one workgroup per clause (grid-stride over clauses); rebuild
// the canonical include bitset from dst_ta.
@compute @workgroup_size(WG)
fn merge_rebuild(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    let total = n_clauses_total();
    let half = cfg.half;
    var cj = wid.x;
    loop {
        if (cj >= total) { break; }
        let ta_base = cj * cfg.n_literals;
        let inc_base = cj * cfg.w32;
        for (var k: u32 = lid.x; k < cfg.w32; k = k + WG) {
            let base = k * 32u;
            var limit: u32 = 32u;
            if (cfg.n_literals < base + 32u) {
                if (cfg.n_literals > base) { limit = cfg.n_literals - base; } else { limit = 0u; }
            }
            var w: u32 = 0u;
            for (var bit: u32 = 0u; bit < limit; bit = bit + 1u) {
                if (dst_ta[ta_base + base + bit] >= half) {
                    w = w | (1u << bit);
                }
            }
            dst_include[inc_base + k] = w & valid[k];
        }
        cj = cj + ng.x;
    }
}
