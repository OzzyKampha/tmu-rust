// GPU inference kernels for the vanilla TsetlinMachine.
//
// Mirrors vanilla_classifier.rs::predict_lit / scores:
//   score[c] = clamp( sum_j (+w if clause j fires and j even, -w if fires and j odd),
//                     -T, T )
//   pred      = argmax_c score[c]  (first strictly-greater in ascending class order)
// Fire test is fire_predict (dense.rs): empty clause does NOT fire.

@group(0) @binding(0) var<uniform> cfg: Config;
@group(0) @binding(1) var<storage, read> include: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<i32>;
@group(0) @binding(3) var<storage, read> valid: array<u32>;
@group(0) @binding(4) var<storage, read> batch_lits: array<u32>;
@group(0) @binding(5) var<storage, read_write> score_buf: array<i32>;
@group(0) @binding(6) var<storage, read_write> preds: array<u32>;
@group(1) @binding(0) var<uniform> infer: InferParams;

struct InferParams {
    n_samples: u32,
    sample_offset: u32,
}

const WG: u32 = 128u;

var<workgroup> partial: array<i32, WG>;

// fire_predict for clause `cj` on sample literal row `lit_base` (u32 word offsets).
fn fire_predict(cj: u32, lit_base: u32) -> bool {
    let inc_base = cj * cfg.w32;
    var violation: u32 = 0u;
    var included: u32 = 0u;
    for (var k: u32 = 0u; k < cfg.w32; k = k + 1u) {
        let inc_k = include[inc_base + k] & valid[k];
        violation = violation | (inc_k & ~batch_lits[lit_base + k]);
        included = included | inc_k;
    }
    return violation == 0u && included != 0u;
}

// One workgroup per (sample, class): reduce the weighted firing-clause sum.
@compute @workgroup_size(WG)
fn scores(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let sample = infer.sample_offset + wid.x;
    let cls = wid.y;
    if (sample >= infer.n_samples || cls >= cfg.n_classes) {
        return;
    }
    let lit_base = sample * cfg.w32;
    let clause0 = cls * cfg.cps;

    var local: i32 = 0;
    for (var j: u32 = lid.x; j < cfg.cps; j = j + WG) {
        let cj = clause0 + j;
        if (fire_predict(cj, lit_base)) {
            let w = weights[cj];
            if ((j & 1u) == 0u) {
                local = local + w;
            } else {
                local = local - w;
            }
        }
    }
    partial[lid.x] = local;
    workgroupBarrier();

    // Tree reduction (order-independent: integer addition is associative).
    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) {
            partial[lid.x] = partial[lid.x] + partial[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lid.x == 0u) {
        let t = i32(cfg.threshold);
        score_buf[sample * cfg.n_classes + cls] = clamp(partial[0], -t, t);
    }
}

// One thread per sample: argmax with first-strictly-greater tie-break.
@compute @workgroup_size(WG)
fn argmax(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sample = infer.sample_offset + gid.x;
    if (sample >= infer.n_samples) {
        return;
    }
    let base = sample * cfg.n_classes;
    var best: i32 = -2147483648;
    var best_c: u32 = 0u;
    for (var c: u32 = 0u; c < cfg.n_classes; c = c + 1u) {
        let v = score_buf[base + c];
        if (v > best) {
            best = v;
            best_c = c;
        }
    }
    preds[sample] = best_c;
}
