//! Build script.
//!
//! Only does work with the `cuda-emu` feature: compiles the CUDA kernels in
//! `src/cuda/kernels.cu` as plain C++ (with the shim in `src/cuda/emu.h`) so
//! `CudaDevice::emulator()` can execute the exact same kernel source on the CPU.
//! The real `cuda` feature needs no build step — kernels are JIT-compiled with
//! NVRTC at runtime.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(feature = "cuda-emu")]
    {
        println!("cargo:rerun-if-changed=src/cuda/kernels.cu");
        println!("cargo:rerun-if-changed=src/cuda/emu.h");
        cc::Build::new()
            .cpp(true)
            .file("src/cuda/kernels.cu")
            .flag("-x")
            .flag("c++")
            .std("c++20")
            .define("TMU_EMU", None)
            // Match the device build (`--fmad=false`): no FMA contraction, so f64
            // feedback probabilities round exactly like the Rust CPU path.
            .flag_if_supported("-ffp-contract=off")
            .flag_if_supported("-Wno-unused-parameter")
            .opt_level(2)
            .compile("tmu_cuda_emu");
    }
}
