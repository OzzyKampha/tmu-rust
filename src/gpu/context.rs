//! GPU device/queue acquisition and compiled compute pipelines.

use std::fmt;

/// Errors from the GPU backend.
#[derive(Debug)]
pub enum GpuError {
    /// No compatible GPU adapter was found (no Vulkan/Metal/DX12 device, and no
    /// software fallback). GPU tests/examples should skip gracefully on this.
    NoAdapter,
    /// An adapter was found but the logical device could not be created.
    DeviceRequest(wgpu::RequestDeviceError),
    /// The model uses an option the GPU backend does not support yet
    /// (e.g. `"type_iii_feedback"`, `"clause_drop_p"`). The original CPU model
    /// is left untouched.
    Unsupported(&'static str),
    /// A required buffer would exceed the adapter's limits.
    LimitExceeded {
        /// Which buffer/limit was exceeded.
        what: &'static str,
        /// Bytes required.
        needed: u64,
        /// Bytes the adapter allows.
        max: u64,
    },
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::NoAdapter => write!(f, "no compatible GPU adapter available"),
            GpuError::DeviceRequest(e) => write!(f, "failed to create GPU device: {e}"),
            GpuError::Unsupported(what) => {
                write!(f, "GPU backend does not support `{what}` yet")
            }
            GpuError::LimitExceeded { what, needed, max } => write!(
                f,
                "GPU limit exceeded for `{what}`: need {needed} bytes, adapter allows {max}"
            ),
        }
    }
}

impl std::error::Error for GpuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GpuError::DeviceRequest(e) => Some(e),
            _ => None,
        }
    }
}

/// Shared GPU device, queue, and compiled pipelines.
///
/// Create one with [`GpuContext::new`] and reuse it across many
/// [`GpuTsetlinMachine`](crate::GpuTsetlinMachine)s (wrap it in an
/// [`Arc`](std::sync::Arc)). Acquiring an adapter is relatively expensive, so a
/// single context per process is recommended.
pub struct GpuContext {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) adapter_info: wgpu::AdapterInfo,
    pub(crate) limits: wgpu::Limits,
    pub(crate) pipelines: Pipelines,
    pub(crate) layouts: Layouts,
}

/// Explicit bind-group layouts (kept per-kernel so each stays within the
/// 8-storage-buffer baseline and so `sample_params` can use a dynamic offset).
pub(crate) struct Layouts {
    pub train_prep0: wgpu::BindGroupLayout,
    pub clause_update0: wgpu::BindGroupLayout,
    pub sample1: wgpu::BindGroupLayout,
    pub scores0: wgpu::BindGroupLayout,
    pub argmax0: wgpu::BindGroupLayout,
    pub infer1: wgpu::BindGroupLayout,
}

/// Compiled compute pipelines, one per kernel entry point.
pub(crate) struct Pipelines {
    pub train_prep: wgpu::ComputePipeline,
    pub clause_update: wgpu::ComputePipeline,
    pub scores: wgpu::ComputePipeline,
    pub argmax: wgpu::ComputePipeline,
}

/// Kinds of buffer binding used by the kernels.
#[derive(Clone, Copy)]
enum Bind {
    Uniform,
    UniformDyn,
    StorageR,
    StorageRw,
}

fn bgl(device: &wgpu::Device, label: &str, entries: &[(u32, Bind)]) -> wgpu::BindGroupLayout {
    let e: Vec<wgpu::BindGroupLayoutEntry> = entries
        .iter()
        .map(|&(binding, kind)| {
            let ty = match kind {
                Bind::Uniform => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                Bind::UniformDyn => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                Bind::StorageR => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                Bind::StorageRw => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            };
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty,
                count: None,
            }
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &e,
    })
}

impl GpuContext {
    /// Acquire a GPU adapter and device (blocking).
    ///
    /// Returns [`GpuError::NoAdapter`] if no compatible adapter (including a
    /// software fallback such as mesa llvmpipe) is available — callers can treat
    /// this as "GPU not available" and fall back to the CPU.
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|_| GpuError::NoAdapter)?;

        let adapter_info = adapter.get_info();
        let limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("tmu-rs gpu device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .map_err(GpuError::DeviceRequest)?;

        let layouts = Layouts::build(&device);
        let pipelines = Pipelines::build(&device, &layouts);

        Ok(Self {
            device,
            queue,
            adapter_info,
            limits,
            pipelines,
            layouts,
        })
    }

    /// Information about the selected adapter (name, backend, device type).
    ///
    /// Useful for logging and for tests to report whether they ran on real
    /// hardware or a software fallback.
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }
}

impl Layouts {
    fn build(device: &wgpu::Device) -> Self {
        use Bind::*;
        // Binding numbers must match src/gpu/shaders/train.wgsl and infer.wgsl.
        let train_prep0 = bgl(
            device,
            "train_prep0",
            &[
                (0, Uniform),    // config
                (2, StorageRw),  // include
                (3, StorageRw),  // weights
                (5, StorageRw),  // class_rngs
                (6, StorageR),   // valid
                (7, StorageR),   // prob_table
                (8, StorageR),   // batch_lits
                (9, StorageR),   // lit_active
                (10, StorageRw), // scratch
            ],
        );
        let clause_update0 = bgl(
            device,
            "clause_update0",
            &[
                (0, Uniform),    // config
                (1, StorageRw),  // ta
                (2, StorageRw),  // include
                (3, StorageRw),  // weights
                (4, StorageRw),  // rngs
                (6, StorageR),   // valid
                (8, StorageR),   // batch_lits
                (9, StorageR),   // lit_active
                (10, StorageRw), // scratch
            ],
        );
        let sample1 = bgl(device, "sample1", &[(0, UniformDyn)]);
        let scores0 = bgl(
            device,
            "scores0",
            &[
                (0, Uniform),   // config
                (1, StorageR),  // include
                (2, StorageR),  // weights
                (3, StorageR),  // valid
                (4, StorageR),  // batch_lits
                (5, StorageRw), // scores
            ],
        );
        let argmax0 = bgl(
            device,
            "argmax0",
            &[
                (0, Uniform),   // config
                (5, StorageRw), // scores
                (6, StorageRw), // preds
            ],
        );
        let infer1 = bgl(device, "infer1", &[(0, Uniform)]);

        Layouts {
            train_prep0,
            clause_update0,
            sample1,
            scores0,
            argmax0,
            infer1,
        }
    }
}

impl Pipelines {
    fn build(device: &wgpu::Device, layouts: &Layouts) -> Self {
        // WGSL is assembled from a shared helper module plus the training and
        // inference kernels, so `include_str!`ed fragments can share code.
        let common = include_str!("shaders/common.wgsl");
        let train_src = format!("{common}\n{}", include_str!("shaders/train.wgsl"));
        let infer_src = format!("{common}\n{}", include_str!("shaders/infer.wgsl"));

        let train = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tmu-rs train.wgsl"),
            source: wgpu::ShaderSource::Wgsl(train_src.into()),
        });
        let infer = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tmu-rs infer.wgsl"),
            source: wgpu::ShaderSource::Wgsl(infer_src.into()),
        });

        let pl = |label: &str, groups: &[Option<&wgpu::BindGroupLayout>]| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: groups,
                immediate_size: 0,
            })
        };
        let train_prep_pl = pl(
            "train_prep_pl",
            &[Some(&layouts.train_prep0), Some(&layouts.sample1)],
        );
        let clause_update_pl = pl(
            "clause_update_pl",
            &[Some(&layouts.clause_update0), Some(&layouts.sample1)],
        );
        let scores_pl = pl(
            "scores_pl",
            &[Some(&layouts.scores0), Some(&layouts.infer1)],
        );
        let argmax_pl = pl(
            "argmax_pl",
            &[Some(&layouts.argmax0), Some(&layouts.infer1)],
        );

        let mk = |module: &wgpu::ShaderModule, layout: &wgpu::PipelineLayout, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Pipelines {
            train_prep: mk(&train, &train_prep_pl, "train_prep"),
            clause_update: mk(&train, &clause_update_pl, "clause_update"),
            scores: mk(&infer, &scores_pl, "scores"),
            argmax: mk(&infer, &argmax_pl, "argmax"),
        }
    }
}
