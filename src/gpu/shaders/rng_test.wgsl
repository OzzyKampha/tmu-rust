// Test-only kernels: emit SplitMix64 / bmask_word streams so a unit test can
// assert bitwise parity with the host `Rng`. Compiled only in tests.

struct RngParams {
    n_seeds: u32,
    n_outputs: u32,
    digits: u32,   // for mask_stream
}

@group(0) @binding(0) var<uniform> p: RngParams;
@group(0) @binding(1) var<storage, read> seeds: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> outputs: array<vec2<u32>>;

@compute @workgroup_size(64)
fn rng_stream(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if (s >= p.n_seeds) { return; }
    var st = seeds[s];
    for (var i: u32 = 0u; i < p.n_outputs; i = i + 1u) {
        let step = splitmix_next(st);
        st = step.state;
        outputs[s * p.n_outputs + i] = step.value;
    }
}

@compute @workgroup_size(64)
fn mask_stream(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if (s >= p.n_seeds) { return; }
    var st = seeds[s];
    for (var i: u32 = 0u; i < p.n_outputs; i = i + 1u) {
        let m = bmask_word(st, p.digits);
        st = m.state;
        outputs[s * p.n_outputs + i] = m.word;
    }
}
