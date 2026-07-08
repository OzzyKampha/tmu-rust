//! Internal GPU unit tests that need device access (RNG bit-exactness).
//!
//! All tests skip cleanly (return) when no adapter is available, so they pass in
//! environments without a GPU or software Vulkan driver.

use super::context::GpuContext;
use crate::clause_bank::dense::{bmask_word, digits_of, MASK_BITS};
use crate::Rng;

fn ctx() -> Option<GpuContext> {
    match GpuContext::new() {
        Ok(c) => {
            eprintln!("gpu test adapter: {}", c.adapter_info().name);
            Some(c)
        }
        Err(super::GpuError::NoAdapter) => {
            eprintln!("no GPU adapter; skipping GPU test");
            None
        }
        Err(e) => panic!("adapter present but device creation failed: {e}"),
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RngParams {
    n_seeds: u32,
    n_outputs: u32,
    digits: u32,
    _pad: u32,
}

/// Run one entry point of rng_test.wgsl and return `n_seeds * n_outputs` u64s.
fn run_stream(ctx: &GpuContext, entry: &str, raw_states: &[u64], n_outputs: usize, digits: u32) -> Vec<u64> {
    use wgpu::util::DeviceExt;
    let dev = &ctx.device;

    let common = include_str!("shaders/common.wgsl");
    let src = format!("{common}\n{}", include_str!("shaders/rng_test.wgsl"));
    let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rng_test"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    });

    let seeds_u32: Vec<u32> = raw_states
        .iter()
        .flat_map(|&s| [s as u32, (s >> 32) as u32])
        .collect();
    let params = RngParams {
        n_seeds: raw_states.len() as u32,
        n_outputs: n_outputs as u32,
        digits,
        _pad: 0,
    };
    let out_len = raw_states.len() * n_outputs;

    let p_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let seed_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&seeds_u32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buf = dev.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (out_len * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: p_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: seed_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
        ],
    });

    let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(raw_states.len().div_ceil(64) as u32, 1, 1);
    }
    ctx.queue.submit(Some(enc.finish()));

    let raw = super::buffers::read_u32(ctx, &out_buf, out_len * 2);
    (0..out_len)
        .map(|i| raw[2 * i] as u64 | ((raw[2 * i + 1] as u64) << 32))
        .collect()
}

#[test]
fn splitmix_stream_bit_exact() {
    let Some(ctx) = ctx() else { return };
    let seeds: Vec<u64> = vec![
        0, 1, 2, 42, 12345, u64::MAX, u64::MAX - 1, 0x9E3779B97F4A7C15,
        0xDEADBEEF, 0xFFFF_0000_FFFF_0000, 7, 99999999,
    ];
    let n_out = 4096;

    // Host: mirror what the GPU does — start from the raw state of Rng::new(seed).
    let raw: Vec<u64> = seeds.iter().map(|&s| Rng::new(s).raw()).collect();
    let gpu = run_stream(&ctx, "rng_stream", &raw, n_out, 0);

    for (si, &seed) in seeds.iter().enumerate() {
        let mut r = Rng::new(seed);
        for i in 0..n_out {
            let cpu = r.next_u64();
            let g = gpu[si * n_out + i];
            assert_eq!(cpu, g, "seed {seed} output {i}: cpu {cpu:#x} != gpu {g:#x}");
        }
    }
}

#[test]
fn bmask_stream_bit_exact() {
    let Some(ctx) = ctx() else { return };
    let seeds: Vec<u64> = vec![0, 1, 42, u64::MAX, 0xABCDEF, 555];
    let n_out = 512;

    for &s_param in &[3.9_f64, 10.0, 2.0, 100.0] {
        let digits_vec = digits_of(1.0 / s_param, MASK_BITS);
        let mut digmask = 0u32;
        for (i, &d) in digits_vec.iter().enumerate() {
            if d != 0 {
                digmask |= 1u32 << i;
            }
        }
        let raw: Vec<u64> = seeds.iter().map(|&s| Rng::new(s).raw()).collect();
        let gpu = run_stream(&ctx, "mask_stream", &raw, n_out, digmask);

        for (si, &seed) in seeds.iter().enumerate() {
            let mut r = Rng::new(seed);
            for i in 0..n_out {
                let cpu = bmask_word(&mut r, &digits_vec);
                let g = gpu[si * n_out + i];
                assert_eq!(cpu, g, "s={s_param} seed={seed} i={i}: {cpu:#x} != {g:#x}");
            }
        }
    }
}
