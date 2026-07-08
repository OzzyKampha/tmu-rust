//! Device buffers for the GPU backend and the host<->device conversions.
//!
//! Bitset arrays (`include`, `valid`, packed sample literals) are `Vec<u64>` on
//! the host and are reinterpreted as `u32` pairs on the device (little-endian, so
//! u64 word `k` becomes u32 words `2k`,`2k+1` — bit `l` stays at u32 word `l/32`,
//! bit `l%32`). TA counters are stored one `u32` per counter (values 0..=255) for
//! simple, branch-free per-literal update kernels. RNG state (`u64`) is stored as
//! `vec2<u32>` matching the host `Rng` internal state.

use wgpu::util::DeviceExt;

use crate::TsetlinMachine;

use super::context::GpuContext;

/// Static per-model dimensions (do not change after `to_gpu`).
#[derive(Clone, Copy)]
pub(crate) struct Dims {
    pub n_classes: usize,
    pub cps: usize,
    pub n_literals: usize,
    pub w32: usize,       // u32 words per clause bitset = 2 * words(u64)
    pub n_clauses: usize, // n_classes * cps
    pub threshold: i32,
}

impl Dims {
    pub fn from(host: &TsetlinMachine) -> Self {
        Dims {
            n_classes: host.n_classes,
            cps: host.clauses_per_class,
            n_literals: host.n_literals,
            w32: host.words * 2,
            n_clauses: host.n_classes * host.clauses_per_class,
            threshold: host.threshold,
        }
    }
}

/// The uniform `Config` mirrored in common.wgsl (12 x u32 = 48 bytes).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuConfig {
    pub n_classes: u32,
    pub cps: u32,
    pub n_literals: u32,
    pub w32: u32,
    pub threshold: u32,
    pub half: u32,
    pub max_state: u32,
    pub boost: u32,
    pub max_inc: u32,
    pub dig_inv: u32,
    pub dig_keep: u32,
    pub has_lit_active: u32,
    pub n_replicas: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// Pack up to 32 Bernoulli digit bytes (`0`/`1`) into a u32 (bit `i` = `digits[i]`).
fn pack_digits(digits: &[u8]) -> u32 {
    let mut m = 0u32;
    for (i, &d) in digits.iter().enumerate() {
        if d != 0 {
            m |= 1u32 << i;
        }
    }
    m
}

impl GpuConfig {
    pub fn build(
        host: &TsetlinMachine,
        dims: &Dims,
        has_lit_active: bool,
        n_replicas: u32,
    ) -> Self {
        let max_inc = if host.max_included_literals == usize::MAX {
            0xFFFF_FFFFu32
        } else {
            host.max_included_literals as u32
        };
        GpuConfig {
            n_classes: dims.n_classes as u32,
            cps: dims.cps as u32,
            n_literals: dims.n_literals as u32,
            w32: dims.w32 as u32,
            threshold: dims.threshold as u32,
            half: host.half as u32,
            max_state: host.max_state as u32,
            boost: host.boost_true_positive as u32,
            max_inc,
            dig_inv: pack_digits(&host.dig_inv),
            dig_keep: pack_digits(&host.dig_keep),
            has_lit_active: has_lit_active as u32,
            n_replicas,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

/// Build the feedback-probability threshold table: for `target` in {0,1}, `class`
/// in `0..C`, and clamped sum `v` in `-T..=T`, store `floor(p * 2^53)` where `p`
/// matches `update_class` exactly. On the GPU, `skip iff (rng>>11) > table[...]`,
/// which is bitwise-equivalent to CPU's `next_f64() > p`.
pub(crate) fn build_prob_table(threshold: i32, class_weights: &[f64]) -> Vec<[u32; 2]> {
    let t = threshold as f64;
    let span = (2 * threshold + 1) as usize;
    let c = class_weights.len();
    let mut table = Vec::with_capacity(2 * c * span);
    const SCALE: f64 = 9_007_199_254_740_992.0; // 2^53
    for target in 0..2u8 {
        for &cw in class_weights.iter() {
            for iv in 0..span {
                let v = iv as f64 - t;
                let p = if target == 1 {
                    ((t - v) / (2.0 * t) * cw).min(1.0)
                } else {
                    ((t + v) / (2.0 * t) * cw).min(1.0)
                };
                let scaled = (p * SCALE) as u64;
                table.push([scaled as u32, (scaled >> 32) as u32]);
            }
        }
    }
    table
}

/// Convert host u8 TA counters to one-u32-per-counter for the device.
fn ta_to_u32(ta: &[u8]) -> Vec<u32> {
    ta.iter().map(|&b| b as u32).collect()
}

/// Convert host RNG streams to `[lo, hi]` u32 pairs.
fn rngs_to_u32(rngs: &[crate::Rng]) -> Vec<u32> {
    let mut v = Vec::with_capacity(rngs.len() * 2);
    for r in rngs {
        let s = r.raw();
        v.push(s as u32);
        v.push((s >> 32) as u32);
    }
    v
}

/// Persistent device buffers mirroring the model state, plus reusable config.
pub(crate) struct DeviceState {
    pub dims: Dims,
    pub config: wgpu::Buffer,
    pub ta: wgpu::Buffer,
    pub include: wgpu::Buffer,
    pub weights: wgpu::Buffer,
    pub rngs: wgpu::Buffer,
    pub class_rngs: wgpu::Buffer,
    pub valid: wgpu::Buffer,
    pub prob_table: wgpu::Buffer,
    /// All-ones literal-active row used when literal dropout is disabled.
    pub ones_lit_active: wgpu::Buffer,
}

const STORAGE_RW: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);
const STORAGE_R: wgpu::BufferUsages =
    wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST);

impl DeviceState {
    pub fn new(ctx: &GpuContext, host: &TsetlinMachine, has_lit_active: bool) -> Self {
        let dims = Dims::from(host);
        let dev = &ctx.device;

        let config_data = GpuConfig::build(host, &dims, has_lit_active, 1);
        let config = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tmu config"),
            contents: bytemuck::bytes_of(&config_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let ta = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tmu ta"),
            contents: bytemuck::cast_slice(&ta_to_u32(&host.ta)),
            usage: STORAGE_RW,
        });
        let include = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tmu include"),
            contents: bytemuck::cast_slice(&host.include),
            usage: STORAGE_RW,
        });
        let weights = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tmu weights"),
            contents: bytemuck::cast_slice(&host.weights),
            usage: STORAGE_RW,
        });
        let rngs = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tmu rngs"),
            contents: bytemuck::cast_slice(&rngs_to_u32(&host.rngs)),
            usage: STORAGE_RW,
        });
        let class_rngs = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tmu class_rngs"),
            contents: bytemuck::cast_slice(&rngs_to_u32(&host.class_rngs)),
            usage: STORAGE_RW,
        });
        let valid = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tmu valid"),
            contents: bytemuck::cast_slice(&host.valid),
            usage: STORAGE_R,
        });
        let table = build_prob_table(dims.threshold, &host.class_weights);
        let prob_table = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tmu prob_table"),
            contents: bytemuck::cast_slice(&table),
            usage: STORAGE_R,
        });
        let ones = vec![0xFFFF_FFFFu32; dims.w32];
        let ones_lit_active = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tmu ones_lit_active"),
            contents: bytemuck::cast_slice(&ones),
            usage: STORAGE_R,
        });

        DeviceState {
            dims,
            config,
            ta,
            include,
            weights,
            rngs,
            class_rngs,
            valid,
            prob_table,
            ones_lit_active,
        }
    }

    /// Rewrite the config uniform (e.g. when toggling literal dropout per epoch,
    /// or setting `n_replicas` for a data-parallel epoch).
    pub fn write_config(
        &self,
        ctx: &GpuContext,
        host: &TsetlinMachine,
        has_lit_active: bool,
        n_replicas: u32,
    ) {
        let cfg = GpuConfig::build(host, &self.dims, has_lit_active, n_replicas);
        ctx.queue
            .write_buffer(&self.config, 0, bytemuck::bytes_of(&cfg));
    }

    /// Download device state back into the host model (ta, include, weights,
    /// per-clause and per-class RNG streams). Blocking.
    pub fn download_into(&self, ctx: &GpuContext, host: &mut TsetlinMachine) {
        let ta_u32 = read_u32(ctx, &self.ta, host.ta.len());
        for (dst, &src) in host.ta.iter_mut().zip(ta_u32.iter()) {
            *dst = src as u8;
        }

        let inc_u32 = read_u32(ctx, &self.include, host.include.len() * 2);
        bytemuck::cast_slice_mut::<u64, u32>(&mut host.include).copy_from_slice(&inc_u32);

        let w_u32 = read_u32(ctx, &self.weights, host.weights.len());
        host.weights
            .copy_from_slice(bytemuck::cast_slice::<u32, i32>(&w_u32));

        let rng_u32 = read_u32(ctx, &self.rngs, host.rngs.len() * 2);
        for (i, r) in host.rngs.iter_mut().enumerate() {
            let lo = rng_u32[2 * i] as u64;
            let hi = rng_u32[2 * i + 1] as u64;
            *r = crate::Rng::from_raw(lo | (hi << 32));
        }
        let crng_u32 = read_u32(ctx, &self.class_rngs, host.class_rngs.len() * 2);
        for (i, r) in host.class_rngs.iter_mut().enumerate() {
            let lo = crng_u32[2 * i] as u64;
            let hi = crng_u32[2 * i + 1] as u64;
            *r = crate::Rng::from_raw(lo | (hi << 32));
        }
    }
}

/// Copy a device storage buffer of `len` u32s to the host (blocking map-read).
pub(crate) fn read_u32(ctx: &GpuContext, src: &wgpu::Buffer, len: usize) -> Vec<u32> {
    let bytes = (len * 4) as u64;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tmu readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(src, 0, &staging, 0, bytes);
    ctx.queue.submit(Some(enc.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let data = slice.get_mapped_range().expect("map buffer for readback");
    let out = bytemuck::cast_slice::<u8, u32>(&data).to_vec();
    drop(data);
    staging.unmap();
    out
}
