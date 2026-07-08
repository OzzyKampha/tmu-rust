//! `GpuTsetlinMachine`: GPU-resident training/inference for the vanilla model.

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::encoder::EncodedBatch;
use crate::TsetlinMachine;

use super::buffers::{read_u32, DeviceState};
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

/// Uniform dynamic-offset stride (must satisfy `min_uniform_buffer_offset_alignment`).
const PARAM_STRIDE: u64 = 256;
/// Max workgroups per dispatch dimension we rely on (WebGPU baseline).
const MAX_DISPATCH: usize = 65535;

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
    dev: DeviceState,
    /// Device buffers hold updates not yet downloaded into `host`.
    device_dirty: bool,
    /// Whether literal dropout is active (affects lit_active binding + config).
    has_lit_active: bool,
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

        // Guard the largest buffers against the adapter's binding-size limit.
        let dims = super::buffers::Dims::from(&host);
        let max_bind = ctx.limits.max_storage_buffer_binding_size as u64;
        let ta_bytes = (dims.n_clauses * dims.n_literals * 4) as u64;
        let inc_bytes = (dims.n_clauses * dims.w32 * 4) as u64;
        for (what, needed) in [("ta", ta_bytes), ("include", inc_bytes)] {
            if needed > max_bind {
                return Err(GpuError::LimitExceeded {
                    what,
                    needed,
                    max: max_bind,
                });
            }
        }

        let dev = DeviceState::new(&ctx, &host, has_lit_active);
        Ok(Self {
            host,
            ctx,
            dev,
            device_dirty: false,
            has_lit_active,
        })
    }

    /// Train for one epoch on the GPU, bitwise-identical to CPU `fit_epoch`.
    pub fn fit_epoch(&mut self, batch: &EncodedBatch, ys: &[usize]) {
        let n = batch.n;
        assert_eq!(n, ys.len());
        if n == 0 {
            return;
        }
        let d = self.dev.dims;
        let w32 = d.w32;

        // Host-RNG epoch plan (advances host rng / literal_rng exactly as CPU).
        let plan = self.host.gpu_epoch_plan(n, ys);

        // Config may need refreshing if literal dropout was toggled.
        let has_la = self.host.literal_drop_p > 0.0;
        if has_la != self.has_lit_active {
            self.has_lit_active = has_la;
            self.dev.write_config(&self.ctx, &self.host, has_la);
        }

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
            .unwrap_or(&self.dev.ones_lit_active);

        let scratch_len = 4 + 4 * w32;
        let scratch = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tmu scratch"),
            size: (scratch_len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

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
                be(0, &self.dev.config),
                be(2, &self.dev.include),
                be(3, &self.dev.weights),
                be(5, &self.dev.class_rngs),
                be(6, &self.dev.valid),
                be(7, &self.dev.prob_table),
                be(8, &batch_lits),
                be(9, lit_active),
                be(10, &scratch),
            ],
        });
        let bg_clause = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_clause"),
            layout: &l.clause_update0,
            entries: &[
                be(0, &self.dev.config),
                be(1, &self.dev.ta),
                be(2, &self.dev.include),
                be(3, &self.dev.weights),
                be(4, &self.dev.rngs),
                be(6, &self.dev.valid),
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
        self.ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        self.device_dirty = true;
    }

    /// Predict a class for every sample in `batch`, on the GPU.
    pub fn predict_batch(&mut self, batch: &EncodedBatch) -> Vec<usize> {
        let n = batch.n;
        if n == 0 {
            return Vec::new();
        }
        let d = self.dev.dims;
        let dev = &self.ctx.device;

        let batch_lits = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tmu infer batch_lits"),
            contents: bytemuck::cast_slice(&batch.data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let scores = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tmu scores"),
            size: (n * d.n_classes * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let preds = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tmu preds"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let l = &self.ctx.layouts;
        let bg_scores = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_scores"),
            layout: &l.scores0,
            entries: &[
                be(0, &self.dev.config),
                be(1, &self.dev.include),
                be(2, &self.dev.weights),
                be(3, &self.dev.valid),
                be(4, &batch_lits),
                be(5, &scores),
            ],
        });
        let bg_argmax = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_argmax"),
            layout: &l.argmax0,
            entries: &[be(0, &self.dev.config), be(5, &scores), be(6, &preds)],
        });

        let mut off = 0usize;
        while off < n {
            let block = (n - off).min(MAX_DISPATCH);
            let ip = InferParams {
                n_samples: n as u32,
                sample_offset: off as u32,
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
                pass.dispatch_workgroups(block as u32, d.n_classes as u32, 1);

                pass.set_pipeline(&self.ctx.pipelines.argmax);
                pass.set_bind_group(0, &bg_argmax, &[]);
                pass.set_bind_group(1, &bg_infer, &[]);
                pass.dispatch_workgroups(block.div_ceil(128) as u32, 1, 1);
            }
            self.ctx.queue.submit(Some(enc.finish()));
            off += block;
        }

        let raw = read_u32(&self.ctx, &preds, n);
        raw.into_iter().map(|v| v as usize).collect()
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
            self.dev.download_into(&self.ctx, &mut self.host);
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
