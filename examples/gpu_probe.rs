//! Probe the GPU: print the selected adapter and the limits that matter, so you
//! can confirm your GPU is actually being used before running anything heavy.
//!
//!   cargo run --release --features gpu --example gpu_probe
//!
//! You want to see your graphics card and a non-`Cpu` device type, e.g.
//! `NVIDIA GeForce RTX 2050 (DiscreteGpu, Vulkan)`. If it reports an `llvmpipe`
//! / `Cpu` adapter, Vulkan cannot see your GPU driver and it fell back to
//! software — install/update your GPU driver (on Linux also `vulkan-tools` to
//! verify with `vulkaninfo`).

use tmu_rs::GpuContext;

fn main() {
    match GpuContext::new() {
        Ok(ctx) => {
            println!("GPU is available:\n");
            println!("{}", ctx.describe());

            // The TA-counter buffer (4 bytes per literal per clause) is the
            // largest single binding and is what gates model size: it must fit
            // within the "max storage buf" limit shown above.
            let max_ta = ctx.limits_max_storage_binding() / 4;
            println!(
                "\nModel-size gate: total TA counters (classes x clauses/class x literals)\n\
                 must be <= max storage buf / 4 = {max_ta} counters.\n\
                 to_gpu returns GpuError::LimitExceeded (no crash) if a model is too big."
            );

            if ctx.is_software() {
                println!(
                    "\n>> This is a SOFTWARE (CPU) Vulkan driver, not your GPU. <<\n\
                     Install your NVIDIA/AMD/Intel driver (Windows: the vendor driver includes\n\
                     Vulkan; Linux: the proprietary driver + `vulkan-tools`), then re-run."
                );
            } else {
                println!("\nLooks good — train with `to_gpu(&ctx)` and `fit_epoch`.");
            }
        }
        Err(e) => {
            eprintln!("No usable GPU adapter: {e}\n");
            eprintln!(
                "To enable the GPU backend:\n\
                 - Windows: install/update the GeForce (NVIDIA) driver — it includes Vulkan.\n\
                 - Linux:   install the proprietary NVIDIA driver plus `vulkan-tools`,\n\
                            then check `vulkaninfo --summary` lists your GPU.\n\
                 - No CUDA toolkit is required; the backend uses Vulkan.\n\
                 Hybrid laptops: force the discrete GPU (NVIDIA Control Panel on Windows, or\n\
                 `__NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only` on Linux)."
            );
            std::process::exit(1);
        }
    }
}
