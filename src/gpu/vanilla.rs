//! `GpuTsetlinMachine`: GPU-resident training/inference for the vanilla model.

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::TsetlinMachine;
use crate::encoder::EncodedBatch;
use crate::models::classification::dp_seed;

use super::buffers::{DeviceState, read_u32};
use super::context::{GpuContext, GpuError};

/// Host-RNG decisions for one epoch, precomputed on the CPU so `self.rng` /
/// `self.literal_rng` advance identically to CPU training. Produced by
/// `TsetlinMachine::gpu_epoch_plan`.
pub(crate) struct GpuEpochPlan {
    /// Shuffled sample index per training step.
    pub order: Vec<usize>,
    /// Sampled negative class per training step.
    pub negs: Vec<usize>,
    /// Step-major literal-active masks (`n * words` u64), empty if dropout is off.
    pub lit_active: Vec<u64>,
}

/// Host-precomputed plan for one data-parallel epoch (`R` replicas). Produced by
/// `TsetlinMachine::dp_epoch_plan`; consumed by `fit_epoch_dp`.
pub(crate) struct DpPlan {
    /// Shuffled sample indices.
    pub order: Vec<usize>,
    /// Per-replica master seed (also seeds the on-device clause/class RNGs).
    pub seeds: Vec<u64>,
    /// Samples per replica shard (`ceil(n / R)`).
    pub shard_len: usize,
    /// Negative class per `[replica * shard_len + step]` (valid where active).
    pub negs: Vec<usize>,
    /// Per-`[replica*shard_len+step]` literal-active masks (`* words`); empty if
    /// literal dropout is off.
    pub lit_active: Vec<u64>,
}

/// Per-sample kernel parameters (mirrors `SampleParams` in train.wgsl).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SampleParams {
    sample_idx: u32,
    step_idx: u32,
    y: u32,
    neg: u32,
}

/// Inference parameters (mirrors `InferParams` in infer.wgsl).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InferParams {
    n_samples: u32,
    sample_offset: u32,
}

/// Per-replica sample params for the data-parallel path (mirrors `SampleDP` in
/// train_dp.wgsl). `sample_idx == u32::MAX` marks an exhausted replica shard.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SampleDP {
    sample_idx: u32,
    y: u32,
    neg: u32,
    la_row: u32,
}

/// Uniform dynamic-offset stride (must satisfy `min_uniform_buffer_offset_alignment`).
const PARAM_STRIDE: u64 = 256;
/// Max workgroups per dispatch dimension we rely on (WebGPU baseline).
const MAX_DISPATCH: usize = 65535;
/// Max data-parallel replicas (matches the `array<SampleDP, 64>` in train_dp.wgsl).
const MAX_REPLICAS: usize = 64;
/// One 64-entry `SampleDP` block per super-step: 64 * 16 bytes = 1024 (256-aligned).
const DP_PARAM_STRIDE: u64 = (MAX_REPLICAS * std::mem::size_of::<SampleDP>()) as u64;

/// A GPU-resident vanilla [`TsetlinMachine`].
///
/// Created with [`TsetlinMachine::to_gpu`]. Owns the host model as the source of
/// truth at boundaries; the device holds a mirrored copy synced on [`sync`] /
/// [`into_cpu`].
///
/// [`sync`]: GpuTsetlinMachine::sync
/// [`into_cpu`]: GpuTsetlinMachine::into_cpu
pub struct GpuTsetlinMachine {
    host: TsetlinMachine,
    ctx: Arc<GpuContext>,
    /// Device-resident model state, or `None` when the model is too large for GPU
    /// memory — in that case training/inference transparently run on the CPU.
    dev: Option<DeviceState>,
    /// Device buffers hold updates not yet downloaded into `host`.
    device_dirty: bool,
    /// Whether literal dropout is active (affects lit_active binding + config).
    has_lit_active: bool,
    /// Reused per-epoch training scratch (fixed size for this model); allocated
    /// once and shared across `fit_epoch` calls.
    scratch: Option<wgpu::Buffer>,
    /// Number of data-parallel replicas for the approximate fast path
    /// (`data_parallel(true)`). `None` = choose automatically from VRAM.
    replicas: Option<usize>,
}

impl TsetlinMachine {
    /// Move a copy of this model onto the GPU for training and/or inference.
    ///
    /// The original CPU model is left untouched (the host state is cloned).
    /// Returns [`GpuError::Unsupported`] if the model uses an option the GPU
    /// backend does not implement yet (`type_iii_feedback`, `clause_drop_p > 0`).
    #[cfg(feature = "gpu")]
    pub fn to_gpu(&self, ctx: &Arc<GpuContext>) -> Result<GpuTsetlinMachine, GpuError> {
        if self.type_iii {
            return Err(GpuError::Unsupported("type_iii_feedback"));
        }
        if self.clause_drop_p > 0.0 {
            return Err(GpuError::Unsupported("clause_drop_p"));
        }
        GpuTsetlinMachine::new(self.clone(), Arc::clone(ctx))
    }
}

impl GpuTsetlinMachine {
    fn new(host: TsetlinMachine, ctx: Arc<GpuContext>) -> Result<Self, GpuError> {
        let has_lit_active = host.literal_drop_p > 0.0;

        // If the model fits in GPU memory, keep it device-resident; otherwise run
        // on the CPU (never fail on size). The largest buffers are `ta` (u32 per
        // counter) and `include`; check both against the per-binding and
        // per-allocation caps.
        let dims = super::buffers::Dims::from(&host);
        let max_bind = ctx.limits.max_storage_buffer_binding_size;
        let max_buf = ctx.limits.max_buffer_size;
        let ta_bytes = (dims.n_clauses * dims.n_literals * 4) as u64;
        let inc_bytes = (dims.n_clauses * dims.w32 * 4) as u64;
        let cap = max_bind.min(max_buf);
        let fits = ta_bytes <= cap && inc_bytes <= cap;

        let dev = if fits {
            Some(DeviceState::new(&ctx, &host, has_lit_active))
        } else {
            eprintln!(
                "tmu-rs: model ({} clauses × {} literals) exceeds GPU buffer limit \
                 ({} MiB) — training/inference will run on the CPU.",
                dims.n_clauses,
                dims.n_literals,
                cap / (1024 * 1024)
            );
            None
        };
        Ok(Self {
            host,
            ctx,
            dev,
            device_dirty: false,
            has_lit_active,
            scratch: None,
            replicas: None,
        })
    }

    /// Device-resident state (only call on GPU-resident paths).
    #[inline]
    fn dev(&self) -> &DeviceState {
        self.dev.as_ref().expect("GPU-resident model")
    }

    /// Whether the model is resident on the GPU. When `false`, training and
    /// inference transparently run on the CPU (the model was too large for VRAM).
    pub fn is_gpu_resident(&self) -> bool {
        self.dev.is_some()
    }

    /// Set the number of data-parallel replicas used by the approximate fast
    /// path ([`data_parallel(true)`](crate::TsetlinMachine::data_parallel)).
    /// `None` (the default) chooses automatically from available memory. Only
    /// affects training when the model was built with `.data_parallel(true)`.
    pub fn set_replicas(&mut self, replicas: Option<usize>) {
        self.replicas = replicas;
    }

    /// Train for one epoch on the GPU.
    ///
    /// Exact (bitwise-identical to CPU) by default. If the model was built with
    /// [`data_parallel(true)`](crate::TsetlinMachine::data_parallel) and enough
    /// replicas fit in memory, uses the faster **approximate** data-parallel path
    /// (see [`set_replicas`](Self::set_replicas)).
    pub fn fit_epoch(&mut self, batch: &EncodedBatch, ys: &[usize]) {
        let n = batch.n;
        assert_eq!(n, ys.len());
        if n == 0 {
            return;
        }

        // Model too large for GPU memory → train on the CPU (host is the model).
        if self.dev.is_none() {
            self.host.fit_epoch(batch, ys);
            return;
        }

        // Batch too large to upload as one storage binding → train this epoch on
        // the CPU and re-sync the device mirror, so later GPU epochs stay correct.
        // (Inference chunks on-device instead; see `predict_batch`.)
        let w32 = self.dev().dims.w32;
        let batch_bytes = (n as u64) * (w32 as u64) * 4;
        if batch_bytes > self.ctx.limits.max_storage_buffer_binding_size {
            self.sync();
            self.host.fit_epoch(batch, ys);
            let has_la = self.host.literal_drop_p > 0.0;
            self.has_lit_active = has_la;
            self.dev = Some(DeviceState::new(&self.ctx, &self.host, has_la));
            self.device_dirty = false;
            return;
        }

        if self.host.data_parallel
            && let Some(r) = self.choose_replicas(n)
        {
            self.fit_epoch_dp(batch, ys, r);
            return;
        }

        let d = self.dev().dims;
        let w32 = d.w32;

        // Host-RNG epoch plan (advances host rng / literal_rng exactly as CPU).
        let plan = self.host.gpu_epoch_plan(n, ys);

        // Mirror CPU's transient `literals` scratch (the last processed sample),
        // so a GPU-trained model serializes byte-identically to a CPU-trained one.
        if let Some(&last) = plan.order.last() {
            let words = self.host.words;
            self.host
                .literals
                .copy_from_slice(&batch.data[last * words..(last + 1) * words]);
        }

        // Config may need refreshing if literal dropout was toggled.
        let has_la = self.host.literal_drop_p > 0.0;
        if has_la != self.has_lit_active {
            self.has_lit_active = has_la;
            self.dev().write_config(&self.ctx, &self.host, has_la, 1);
        }

        // Training scratch is a fixed size for this model — allocate once, reuse
        // across epochs (done before borrowing `self.ctx.device` as `dev`).
        let scratch_len = 4 + 4 * w32;
        if self.scratch.is_none() {
            self.scratch = Some(self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tmu scratch"),
                size: (scratch_len * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
        }
        let scratch = self.scratch.clone().unwrap();

        let dev = &self.ctx.device;

        // --- per-call buffers ---
        let batch_lits = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tmu batch_lits"),
            contents: bytemuck::cast_slice(&batch.data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // When dropout is off, bind the persistent all-ones row (never read by the
        // kernel since has_lit_active == 0).
        let lit_active_owned = if has_la {
            Some(dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tmu lit_active"),
                contents: bytemuck::cast_slice(&plan.lit_active),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }))
        } else {
            None
        };
        let lit_active = lit_active_owned
            .as_ref()
            .unwrap_or(&self.dev().ones_lit_active);

        // Per-sample params, one 256-byte slot per step.
        let mut params = vec![0u8; n * PARAM_STRIDE as usize];
        for (k, &sample_idx) in plan.order.iter().enumerate() {
            let sp = SampleParams {
                sample_idx: sample_idx as u32,
                step_idx: k as u32,
                y: ys[sample_idx] as u32,
                neg: plan.negs[k] as u32,
            };
            let off = k * PARAM_STRIDE as usize;
            params[off..off + std::mem::size_of::<SampleParams>()]
                .copy_from_slice(bytemuck::bytes_of(&sp));
        }
        let sample_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tmu sample_params"),
            contents: &params,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // --- bind groups (buffers stable across the epoch) ---
        let l = &self.ctx.layouts;
        let bg_prep = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_prep"),
            layout: &l.train_prep0,
            entries: &[
                be(0, &self.dev().config),
                be(2, &self.dev().include),
                be(3, &self.dev().weights),
                be(5, &self.dev().class_rngs),
                be(6, &self.dev().valid),
                be(7, &self.dev().prob_table),
                be(8, &batch_lits),
                be(9, lit_active),
                be(10, &scratch),
            ],
        });
        let bg_clause = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_clause"),
            layout: &l.clause_update0,
            entries: &[
                be(0, &self.dev().config),
                be(1, &self.dev().ta),
                be(2, &self.dev().include),
                be(3, &self.dev().weights),
                be(4, &self.dev().rngs),
                be(6, &self.dev().valid),
                be(8, &batch_lits),
                be(9, lit_active),
                be(10, &scratch),
            ],
        });
        let bg_sample = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_sample"),
            layout: &l.sample1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &sample_buf,
                    offset: 0,
                    size: std::num::NonZeroU64::new(std::mem::size_of::<SampleParams>() as u64),
                }),
            }],
        });

        // --- encode: 2 dispatches per sample, chunked into submissions ---
        const CHUNK: usize = 256;
        let cps = d.cps as u32;
        let mut k = 0usize;
        while k < n {
            let end = (k + CHUNK).min(n);
            let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tmu train enc"),
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("tmu train pass"),
                    timestamp_writes: None,
                });
                for step in k..end {
                    let off = (step as u64) * PARAM_STRIDE;
                    pass.set_pipeline(&self.ctx.pipelines.train_prep);
                    pass.set_bind_group(0, &bg_prep, &[]);
                    pass.set_bind_group(1, &bg_sample, &[off as u32]);
                    pass.dispatch_workgroups(2, 1, 1);

                    pass.set_pipeline(&self.ctx.pipelines.clause_update);
                    pass.set_bind_group(0, &bg_clause, &[]);
                    pass.set_bind_group(1, &bg_sample, &[off as u32]);
                    pass.dispatch_workgroups(cps, 2, 1);
                }
            }
            self.ctx.queue.submit(Some(enc.finish()));
            k = end;
        }
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .ok();
        self.device_dirty = true;
    }

    /// Choose the data-parallel replica count for this model, or `None` to fall
    /// back to the exact path (fewer than 2 replicas fit / are requested).
    ///
    /// An explicit [`set_replicas`](Self::set_replicas) value is honored (still
    /// capped by memory and `n`). Auto mode also keeps each shard reasonably
    /// large (>= `MIN_SHARD` samples), since averaging too many tiny-shard
    /// replicas hurts accuracy.
    fn choose_replicas(&self, n: usize) -> Option<usize> {
        /// Minimum samples per replica shard for the auto heuristic.
        const MIN_SHARD: usize = 64;
        let d = self.dev().dims;
        let ta_per = (d.n_clauses * d.n_literals * 4) as u64;
        let inc_per = (d.n_clauses * d.w32 * 4) as u64;
        let max_bind = self.ctx.limits.max_storage_buffer_binding_size;
        // R replicas share one buffer per component, so R*per must fit the limit.
        let cap = |per: u64| {
            max_bind
                .checked_div(per)
                .map_or(MAX_REPLICAS, |v| (v as usize).max(1))
        };
        let requested = self
            .replicas
            .unwrap_or_else(|| (n / MIN_SHARD).clamp(1, MAX_REPLICAS));
        let r = requested
            .min(MAX_REPLICAS)
            .min(cap(ta_per))
            .min(cap(inc_per))
            .min(n);
        if r >= 2 { Some(r) } else { None }
    }

    /// Data-parallel (approximate) epoch: train `r` replicas in lockstep on
    /// sample shards, then merge by averaging TA counters + weights and rebuilding
    /// include bitsets. Mirrors CPU `fit_epoch_data_parallel`.
    fn fit_epoch_dp(&mut self, batch: &EncodedBatch, ys: &[usize], r: usize) {
        let n = batch.n;
        let d = self.dev().dims;
        let w32 = d.w32;
        let n_clauses = d.n_clauses;
        let n_literals = d.n_literals;
        let n_classes = d.n_classes;
        let cps = d.cps as u32;

        // Host plan (advances host.rng exactly like CPU data-parallel).
        let plan = self.host.dp_epoch_plan(n, ys, r);
        let shard_len = plan.shard_len;
        let has_la = self.host.literal_drop_p > 0.0;
        self.has_lit_active = has_la;
        self.dev()
            .write_config(&self.ctx, &self.host, has_la, r as u32);

        // rngs_all: per-replica clause RNGs first, then per-replica class RNGs.
        let mut rngs_all: Vec<u32> = Vec::with_capacity((r * n_clauses + r * n_classes) * 2);
        for &sd in &plan.seeds {
            for i in 0..n_clauses {
                let s = dp_seed::clause_rng(sd, i).raw();
                rngs_all.push(s as u32);
                rngs_all.push((s >> 32) as u32);
            }
        }
        for &sd in &plan.seeds {
            for c in 0..n_classes {
                let s = dp_seed::class_rng(sd, c, n_clauses).raw();
                rngs_all.push(s as u32);
                rngs_all.push((s >> 32) as u32);
            }
        }

        let dev = &self.ctx.device;
        let storage = wgpu::BufferUsages::STORAGE;
        let storage_dst = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

        // Replica state buffers are seeded from the CANONICAL device model (which
        // holds the previous epoch's merged result) via GPU copies — not from the
        // host, which is only synced on demand. All replicas start each epoch from
        // the current model, matching CPU `fit_epoch_data_parallel`.
        let ta_bytes = (n_clauses * n_literals * 4) as u64;
        let inc_bytes = (n_clauses * w32 * 4) as u64;
        let w_bytes = (n_clauses * 4) as u64;
        let replica_ta = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dp ta"),
            size: ta_bytes * r as u64,
            usage: storage_dst,
            mapped_at_creation: false,
        });
        let replica_inc = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dp include"),
            size: inc_bytes * r as u64,
            usage: storage_dst,
            mapped_at_creation: false,
        });
        let replica_w = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dp weights"),
            size: w_bytes * r as u64,
            usage: storage_dst,
            mapped_at_creation: false,
        });
        let rngs_all_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dp rngs_all"),
            contents: bytemuck::cast_slice(&rngs_all),
            usage: storage,
        });
        // Fill each replica block with a copy of the canonical model.
        let mut seed_enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dp seed replicas"),
        });
        for ri in 0..r as u64 {
            seed_enc.copy_buffer_to_buffer(&self.dev().ta, 0, &replica_ta, ri * ta_bytes, ta_bytes);
            seed_enc.copy_buffer_to_buffer(
                &self.dev().include,
                0,
                &replica_inc,
                ri * inc_bytes,
                inc_bytes,
            );
            seed_enc.copy_buffer_to_buffer(
                &self.dev().weights,
                0,
                &replica_w,
                ri * w_bytes,
                w_bytes,
            );
        }
        self.ctx.queue.submit(Some(seed_enc.finish()));
        let batch_lits = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dp batch_lits"),
            contents: bytemuck::cast_slice(&batch.data),
            usage: storage_dst,
        });
        let lit_active_owned = if has_la {
            Some(dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dp lit_active"),
                contents: bytemuck::cast_slice(&plan.lit_active),
                usage: storage_dst,
            }))
        } else {
            None
        };
        let lit_active = lit_active_owned
            .as_ref()
            .unwrap_or(&self.dev().ones_lit_active);
        let dp_scratch = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dp scratch"),
            size: ((r * (4 + 4 * w32)) * 4) as u64,
            usage: storage,
            mapped_at_creation: false,
        });

        // --- per-super-step params (64-entry SampleDP blocks) ---
        let mut params = vec![0xFFu8; shard_len * DP_PARAM_STRIDE as usize];
        let spdp = std::mem::size_of::<SampleDP>();
        for s in 0..shard_len {
            let block = s * DP_PARAM_STRIDE as usize;
            for ri in 0..r {
                let gk = ri * shard_len + s;
                let end = ((ri + 1) * shard_len).min(n);
                if gk < end {
                    let i = plan.order[gk];
                    let sp = SampleDP {
                        sample_idx: i as u32,
                        y: ys[i] as u32,
                        neg: plan.negs[ri * shard_len + s] as u32,
                        la_row: (ri * shard_len + s) as u32,
                    };
                    let off = block + ri * spdp;
                    params[off..off + spdp].copy_from_slice(bytemuck::bytes_of(&sp));
                }
                // else: leave 0xFF bytes -> sample_idx == u32::MAX (inactive).
            }
        }
        let param_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dp params"),
            contents: &params,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // --- bind groups ---
        let l = &self.ctx.layouts;
        let bg_prep = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dp bg_prep"),
            layout: &l.dp_prep0,
            entries: &[
                be(0, &self.dev().config),
                be(2, &replica_inc),
                be(3, &replica_w),
                be(4, &rngs_all_buf),
                be(5, &self.dev().valid),
                be(6, &self.dev().prob_table),
                be(7, &batch_lits),
                be(8, lit_active),
                be(9, &dp_scratch),
            ],
        });
        let bg_clause = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dp bg_clause"),
            layout: &l.dp_clause0,
            entries: &[
                be(0, &self.dev().config),
                be(1, &replica_ta),
                be(2, &replica_inc),
                be(3, &replica_w),
                be(4, &rngs_all_buf),
                be(5, &self.dev().valid),
                be(7, &batch_lits),
                be(8, lit_active),
                be(9, &dp_scratch),
            ],
        });
        let bg_sample = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dp bg_sample"),
            layout: &l.dp_sample1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &param_buf,
                    offset: 0,
                    size: std::num::NonZeroU64::new(DP_PARAM_STRIDE),
                }),
            }],
        });

        // --- lockstep training ---
        const CHUNK: usize = 128;
        let mut s0 = 0usize;
        while s0 < shard_len {
            let end = (s0 + CHUNK).min(shard_len);
            let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dp train enc"),
            });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("dp train pass"),
                    timestamp_writes: None,
                });
                for s in s0..end {
                    let off = (s as u64) * DP_PARAM_STRIDE;
                    pass.set_pipeline(&self.ctx.pipelines.train_prep_dp);
                    pass.set_bind_group(0, &bg_prep, &[]);
                    pass.set_bind_group(1, &bg_sample, &[off as u32]);
                    pass.dispatch_workgroups(2, r as u32, 1);

                    pass.set_pipeline(&self.ctx.pipelines.clause_update_dp);
                    pass.set_bind_group(0, &bg_clause, &[]);
                    pass.set_bind_group(1, &bg_sample, &[off as u32]);
                    pass.dispatch_workgroups(cps, 2, r as u32);
                }
            }
            self.ctx.queue.submit(Some(enc.finish()));
            s0 = end;
        }

        // --- merge replicas into the canonical model buffers ---
        let bg_avg_ta = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dp bg_avg_ta"),
            layout: &l.dp_avg_ta0,
            entries: &[
                be(0, &self.dev().config),
                be(1, &replica_ta),
                be(10, &self.dev().ta),
            ],
        });
        let bg_avg_w = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dp bg_avg_w"),
            layout: &l.dp_avg_w0,
            entries: &[
                be(0, &self.dev().config),
                be(3, &replica_w),
                be(12, &self.dev().weights),
            ],
        });
        let bg_rebuild = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dp bg_rebuild"),
            layout: &l.dp_rebuild0,
            entries: &[
                be(0, &self.dev().config),
                be(5, &self.dev().valid),
                be(10, &self.dev().ta),
                be(11, &self.dev().include),
            ],
        });
        let groups = |work: usize| ((work.div_ceil(128)).min(MAX_DISPATCH)) as u32;
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dp merge enc"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dp merge pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.ctx.pipelines.avg_ta);
            pass.set_bind_group(0, &bg_avg_ta, &[]);
            pass.dispatch_workgroups(groups(n_clauses * n_literals), 1, 1);
            pass.set_pipeline(&self.ctx.pipelines.avg_weights);
            pass.set_bind_group(0, &bg_avg_w, &[]);
            pass.dispatch_workgroups(groups(n_clauses), 1, 1);
            pass.set_pipeline(&self.ctx.pipelines.merge_rebuild);
            pass.set_bind_group(0, &bg_rebuild, &[]);
            pass.dispatch_workgroups((n_clauses.min(MAX_DISPATCH)) as u32, 1, 1);
        }
        self.ctx.queue.submit(Some(enc.finish()));
        self.ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .ok();
        self.device_dirty = true;
    }

    /// Predict a class for every sample in `batch` (GPU, or CPU if the model is
    /// too large to be GPU-resident).
    pub fn predict_batch(&mut self, batch: &EncodedBatch) -> Vec<usize> {
        let n = batch.n;
        if n == 0 {
            return Vec::new();
        }
        if self.dev.is_none() {
            return self.host.predict_batch(batch);
        }
        let d = self.dev().dims;
        let words_u64 = d.w32 / 2;
        // Chunk over samples so each chunk's buffers fit the dispatch and
        // storage-binding limits; per-chunk sample indices are local.
        let block = self.max_rows_per_chunk(d.w32).clamp(1, MAX_DISPATCH);

        let mut out = Vec::with_capacity(n);
        let mut off = 0usize;
        while off < n {
            let rows = (n - off).min(block);
            out.extend(
                self.predict_chunk(&batch.data[off * words_u64..(off + rows) * words_u64], rows),
            );
            off += rows;
        }
        out
    }

    /// Max sample rows whose packed literals fit one storage-buffer binding.
    fn max_rows_per_chunk(&self, w32: usize) -> usize {
        let per_row = (w32 * 4) as u64;
        self.ctx
            .limits
            .max_storage_buffer_binding_size
            .checked_div(per_row)
            .map_or(usize::MAX, |v| (v as usize).max(1))
    }

    /// Run scores+argmax for one contiguous chunk (`rows` samples, local indices).
    fn predict_chunk(&self, chunk_data: &[u64], rows: usize) -> Vec<usize> {
        let d = self.dev().dims;
        let dev = &self.ctx.device;
        let batch_lits = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tmu infer batch_lits"),
            contents: bytemuck::cast_slice(chunk_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let scores = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tmu scores"),
            size: (rows * d.n_classes * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let preds = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tmu preds"),
            size: (rows * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let l = &self.ctx.layouts;
        let bg_scores = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_scores"),
            layout: &l.scores0,
            entries: &[
                be(0, &self.dev().config),
                be(1, &self.dev().include),
                be(2, &self.dev().weights),
                be(3, &self.dev().valid),
                be(4, &batch_lits),
                be(5, &scores),
            ],
        });
        let bg_argmax = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_argmax"),
            layout: &l.argmax0,
            entries: &[be(0, &self.dev().config), be(5, &scores), be(6, &preds)],
        });
        let ip = InferParams {
            n_samples: rows as u32,
            sample_offset: 0,
        };
        let ip_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tmu infer_params"),
            contents: bytemuck::bytes_of(&ip),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg_infer = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_infer1"),
            layout: &l.infer1,
            entries: &[be(0, &ip_buf)],
        });

        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tmu infer enc"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tmu infer pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.ctx.pipelines.scores);
            pass.set_bind_group(0, &bg_scores, &[]);
            pass.set_bind_group(1, &bg_infer, &[]);
            pass.dispatch_workgroups(rows as u32, d.n_classes as u32, 1);

            pass.set_pipeline(&self.ctx.pipelines.argmax);
            pass.set_bind_group(0, &bg_argmax, &[]);
            pass.set_bind_group(1, &bg_infer, &[]);
            pass.dispatch_workgroups(rows.div_ceil(128) as u32, 1, 1);
        }
        self.ctx.queue.submit(Some(enc.finish()));
        read_u32(&self.ctx, &preds, rows)
            .into_iter()
            .map(|v| v as usize)
            .collect()
    }

    /// Accuracy over an encoded batch (GPU inference).
    pub fn accuracy(&mut self, batch: &EncodedBatch, ys: &[usize]) -> f64 {
        let preds = self.predict_batch(batch);
        let correct = preds.iter().zip(ys).filter(|(p, y)| p == y).count();
        correct as f64 / ys.len() as f64
    }

    /// Download device state into the host model and return a reference to it.
    ///
    /// After this the host [`TsetlinMachine`] is a full, exact copy of the
    /// GPU-trained model — ready for CPU inference, introspection, or
    /// [`save`](crate::SaveLoad::save).
    pub fn sync(&mut self) -> &TsetlinMachine {
        if self.device_dirty {
            // `device_dirty` is only set on GPU-resident paths, so `dev` is Some.
            let dev = self.dev.as_ref().expect("GPU-resident model");
            dev.download_into(&self.ctx, &mut self.host);
            self.device_dirty = false;
        }
        &self.host
    }

    /// Consume the GPU model, returning the synced CPU [`TsetlinMachine`].
    pub fn into_cpu(mut self) -> TsetlinMachine {
        self.sync();
        self.host
    }

    /// Adapter information for the underlying context.
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        self.ctx.adapter_info()
    }
}

/// Shorthand for a whole-buffer bind group entry.
fn be(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
