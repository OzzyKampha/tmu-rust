//! Minimal runtime-loaded bindings to the CUDA driver API and NVRTC.
//!
//! Both libraries are opened with `dlopen` (via `libloading`) the first time a
//! [`CudaDevice`](super::CudaDevice) is created, so building with `--features cuda`
//! needs neither the CUDA toolkit nor a GPU — only *running* does.
//!
//! Only the handful of entry points the crate needs are bound. Kernels are
//! compiled from `kernels.cu` with NVRTC for the device's exact architecture
//! (CUBIN when the NVRTC build supports it, PTX otherwise).

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;

use super::CudaError;

pub(crate) type CuResult = c_int;
pub(crate) type CuDevice = c_int;
pub(crate) type CuContext = *mut c_void;
pub(crate) type CuModule = *mut c_void;
pub(crate) type CuFunction = *mut c_void;
pub(crate) type CuDevicePtr = u64;
type NvrtcProgram = *mut c_void;
type NvrtcResult = c_int;

const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

macro_rules! sym {
    ($lib:expr, $name:literal) => {
        // SAFETY: the signature on the receiving field matches the documented
        // CUDA driver / NVRTC C prototype for this symbol.
        unsafe {
            *$lib
                .get(concat!($name, "\0").as_bytes())
                .map_err(|e| CudaError::Load(format!("missing symbol {}: {e}", $name)))?
        }
    };
}

/// Function table for the CUDA driver API (`libcuda`).
#[allow(non_snake_case)]
pub(crate) struct Driver {
    _lib: libloading::Library,
    pub cuInit: unsafe extern "C" fn(c_uint) -> CuResult,
    pub cuDeviceGetCount: unsafe extern "C" fn(*mut c_int) -> CuResult,
    pub cuDeviceGet: unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult,
    pub cuDeviceGetName: unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult,
    pub cuDeviceGetAttribute: unsafe extern "C" fn(*mut c_int, c_int, CuDevice) -> CuResult,
    pub cuDevicePrimaryCtxRetain: unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult,
    pub cuDevicePrimaryCtxRelease: unsafe extern "C" fn(CuDevice) -> CuResult,
    pub cuCtxSetCurrent: unsafe extern "C" fn(CuContext) -> CuResult,
    pub cuCtxSynchronize: unsafe extern "C" fn() -> CuResult,
    pub cuModuleLoadData: unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult,
    pub cuModuleUnload: unsafe extern "C" fn(CuModule) -> CuResult,
    pub cuModuleGetFunction:
        unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult,
    pub cuMemAlloc: unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult,
    pub cuMemFree: unsafe extern "C" fn(CuDevicePtr) -> CuResult,
    pub cuMemcpyHtoD: unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> CuResult,
    pub cuMemcpyDtoH: unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> CuResult,
    pub cuMemsetD8: unsafe extern "C" fn(CuDevicePtr, u8, usize) -> CuResult,
    #[allow(clippy::type_complexity)]
    pub cuLaunchKernel: unsafe extern "C" fn(
        CuFunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CuResult,
    pub cuGetErrorString: unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult,
}

/// Function table for NVRTC (`libnvrtc`).
#[allow(non_snake_case)]
struct Nvrtc {
    _lib: libloading::Library,
    nvrtcCreateProgram: unsafe extern "C" fn(
        *mut NvrtcProgram,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> NvrtcResult,
    nvrtcDestroyProgram: unsafe extern "C" fn(*mut NvrtcProgram) -> NvrtcResult,
    nvrtcCompileProgram:
        unsafe extern "C" fn(NvrtcProgram, c_int, *const *const c_char) -> NvrtcResult,
    nvrtcGetPTXSize: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult,
    nvrtcGetPTX: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult,
    nvrtcGetCUBINSize: Option<unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult>,
    nvrtcGetCUBIN: Option<unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult>,
    nvrtcGetProgramLogSize: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> NvrtcResult,
    nvrtcGetProgramLog: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> NvrtcResult,
    nvrtcGetErrorString: unsafe extern "C" fn(NvrtcResult) -> *const c_char,
}

/// Open the first library in `names` that loads. `env` (if set) is tried first.
fn open_lib(env: &str, names: &[&str]) -> Result<libloading::Library, CudaError> {
    let mut tried = Vec::new();
    let from_env = std::env::var(env).ok();
    for name in from_env
        .iter()
        .map(String::as_str)
        .chain(names.iter().copied())
    {
        // SAFETY: loading a system CUDA library runs its initialisers, which is
        // the documented way to use these libraries.
        match unsafe { libloading::Library::new(name) } {
            Ok(lib) => return Ok(lib),
            Err(e) => tried.push(format!("{name}: {e}")),
        }
    }
    Err(CudaError::Load(format!(
        "could not load any of [{}] (set {env} to the library path)",
        tried.join("; ")
    )))
}

#[cfg(windows)]
const CUDA_LIBS: &[&str] = &["nvcuda.dll"];
#[cfg(not(windows))]
const CUDA_LIBS: &[&str] = &["libcuda.so.1", "libcuda.so"];

#[cfg(windows)]
const NVRTC_LIBS: &[&str] = &[
    "nvrtc64_130_0.dll",
    "nvrtc64_120_0.dll",
    "nvrtc64_112_0.dll",
    "nvrtc64_111_0.dll",
];
#[cfg(not(windows))]
const NVRTC_LIBS: &[&str] = &[
    "libnvrtc.so",
    "libnvrtc.so.13",
    "libnvrtc.so.12",
    "libnvrtc.so.11.2",
    "libnvrtc.so.11.1",
];

impl Driver {
    fn load() -> Result<Self, CudaError> {
        let lib = open_lib("TMU_CUDA_DRIVER_PATH", CUDA_LIBS)?;
        let d = Driver {
            cuInit: sym!(lib, "cuInit"),
            cuDeviceGetCount: sym!(lib, "cuDeviceGetCount"),
            cuDeviceGet: sym!(lib, "cuDeviceGet"),
            cuDeviceGetName: sym!(lib, "cuDeviceGetName"),
            cuDeviceGetAttribute: sym!(lib, "cuDeviceGetAttribute"),
            cuDevicePrimaryCtxRetain: sym!(lib, "cuDevicePrimaryCtxRetain"),
            cuDevicePrimaryCtxRelease: sym!(lib, "cuDevicePrimaryCtxRelease_v2"),
            cuCtxSetCurrent: sym!(lib, "cuCtxSetCurrent"),
            cuCtxSynchronize: sym!(lib, "cuCtxSynchronize"),
            cuModuleLoadData: sym!(lib, "cuModuleLoadData"),
            cuModuleUnload: sym!(lib, "cuModuleUnload"),
            cuModuleGetFunction: sym!(lib, "cuModuleGetFunction"),
            cuMemAlloc: sym!(lib, "cuMemAlloc_v2"),
            cuMemFree: sym!(lib, "cuMemFree_v2"),
            cuMemcpyHtoD: sym!(lib, "cuMemcpyHtoD_v2"),
            cuMemcpyDtoH: sym!(lib, "cuMemcpyDtoH_v2"),
            cuMemsetD8: sym!(lib, "cuMemsetD8_v2"),
            cuLaunchKernel: sym!(lib, "cuLaunchKernel"),
            cuGetErrorString: sym!(lib, "cuGetErrorString"),
            _lib: lib,
        };
        // SAFETY: plain driver call with a valid flags argument.
        d.check(unsafe { (d.cuInit)(0) }, "cuInit")?;
        Ok(d)
    }

    /// Process-wide driver handle (loaded and initialised once).
    pub(crate) fn get() -> Result<&'static Driver, CudaError> {
        static DRIVER: OnceLock<Result<Driver, CudaError>> = OnceLock::new();
        DRIVER
            .get_or_init(Driver::load)
            .as_ref()
            .map_err(Clone::clone)
    }

    /// Map a `CUresult` to `Ok(())` or a descriptive [`CudaError::Driver`].
    pub(crate) fn check(&self, r: CuResult, what: &str) -> Result<(), CudaError> {
        if r == 0 {
            return Ok(());
        }
        let mut s: *const c_char = std::ptr::null();
        // SAFETY: cuGetErrorString writes a pointer to a static string (or leaves null).
        unsafe { (self.cuGetErrorString)(r, &mut s) };
        let msg = if s.is_null() {
            "unknown error".to_string()
        } else {
            // SAFETY: non-null result is a NUL-terminated static string.
            unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
        };
        Err(CudaError::Driver {
            call: what.to_string(),
            code: r,
            msg,
        })
    }

    pub(crate) fn device_count(&self) -> Result<usize, CudaError> {
        let mut n: c_int = 0;
        // SAFETY: valid out-pointer.
        self.check(
            unsafe { (self.cuDeviceGetCount)(&mut n) },
            "cuDeviceGetCount",
        )?;
        Ok(n.max(0) as usize)
    }

    pub(crate) fn device_name(&self, dev: CuDevice) -> Result<String, CudaError> {
        let mut buf = [0 as c_char; 256];
        // SAFETY: buffer length passed matches the buffer.
        self.check(
            unsafe { (self.cuDeviceGetName)(buf.as_mut_ptr(), buf.len() as c_int, dev) },
            "cuDeviceGetName",
        )?;
        // SAFETY: the driver NUL-terminates within the given length.
        Ok(unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }

    pub(crate) fn compute_capability(&self, dev: CuDevice) -> Result<(i32, i32), CudaError> {
        let (mut major, mut minor) = (0, 0);
        // SAFETY: valid out-pointers and attribute ids.
        unsafe {
            self.check(
                (self.cuDeviceGetAttribute)(
                    &mut major,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    dev,
                ),
                "cuDeviceGetAttribute",
            )?;
            self.check(
                (self.cuDeviceGetAttribute)(
                    &mut minor,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    dev,
                ),
                "cuDeviceGetAttribute",
            )?;
        }
        Ok((major, minor))
    }
}

impl Nvrtc {
    fn load() -> Result<Self, CudaError> {
        let lib = open_lib("TMU_NVRTC_PATH", NVRTC_LIBS)?;
        // SAFETY (optional symbols): same prototype contract as `sym!`.
        let cubin_size = unsafe { lib.get(b"nvrtcGetCUBINSize\0").ok().map(|s| *s) };
        let cubin = unsafe { lib.get(b"nvrtcGetCUBIN\0").ok().map(|s| *s) };
        Ok(Nvrtc {
            nvrtcCreateProgram: sym!(lib, "nvrtcCreateProgram"),
            nvrtcDestroyProgram: sym!(lib, "nvrtcDestroyProgram"),
            nvrtcCompileProgram: sym!(lib, "nvrtcCompileProgram"),
            nvrtcGetPTXSize: sym!(lib, "nvrtcGetPTXSize"),
            nvrtcGetPTX: sym!(lib, "nvrtcGetPTX"),
            nvrtcGetCUBINSize: cubin_size,
            nvrtcGetCUBIN: cubin,
            nvrtcGetProgramLogSize: sym!(lib, "nvrtcGetProgramLogSize"),
            nvrtcGetProgramLog: sym!(lib, "nvrtcGetProgramLog"),
            nvrtcGetErrorString: sym!(lib, "nvrtcGetErrorString"),
            _lib: lib,
        })
    }

    fn get() -> Result<&'static Nvrtc, CudaError> {
        static NVRTC: OnceLock<Result<Nvrtc, CudaError>> = OnceLock::new();
        NVRTC
            .get_or_init(Nvrtc::load)
            .as_ref()
            .map_err(Clone::clone)
    }

    fn err(&self, r: NvrtcResult, what: &str) -> CudaError {
        // SAFETY: returns a pointer to a static NUL-terminated string.
        let s = unsafe { CStr::from_ptr((self.nvrtcGetErrorString)(r)) };
        CudaError::Compile(format!("{what}: {}", s.to_string_lossy()))
    }
}

/// Compiled kernel image, ready for `cuModuleLoadData`.
pub(crate) enum Image {
    Cubin(Vec<u8>),
    /// NUL-terminated PTX text.
    Ptx(Vec<u8>),
}

impl Image {
    pub(crate) fn as_ptr(&self) -> *const c_void {
        match self {
            Image::Cubin(b) | Image::Ptx(b) => b.as_ptr() as *const c_void,
        }
    }
}

/// Compile CUDA C++ `src` with NVRTC for compute capability `cc`.
///
/// `--fmad=false` keeps every f64 expression rounding exactly like the Rust CPU
/// path (no fused multiply-add contraction), which the bit-identical training
/// guarantee relies on.
pub(crate) fn compile(src: &str, cc: (i32, i32)) -> Result<Image, CudaError> {
    let nv = Nvrtc::get()?;
    // Prefer a CUBIN for the exact SM (no PTX-JIT / PTX-ISA-version coupling with
    // the driver); fall back to PTX for the virtual arch when unavailable.
    if nv.nvrtcGetCUBIN.is_some()
        && let Ok(img) = compile_with(nv, src, &format!("sm_{}{}", cc.0, cc.1), true)
    {
        return Ok(img);
    }
    compile_with(nv, src, &format!("compute_{}{}", cc.0, cc.1), false)
}

fn compile_with(nv: &Nvrtc, src: &str, arch: &str, cubin: bool) -> Result<Image, CudaError> {
    let csrc = CString::new(src).expect("kernel source has no NUL");
    let name = c"tmu_kernels.cu";
    let mut prog: NvrtcProgram = std::ptr::null_mut();
    // SAFETY: valid NUL-terminated strings; no headers.
    let r = unsafe {
        (nv.nvrtcCreateProgram)(
            &mut prog,
            csrc.as_ptr(),
            name.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        )
    };
    if r != 0 {
        return Err(nv.err(r, "nvrtcCreateProgram"));
    }
    let opts: Vec<CString> = [
        format!("--gpu-architecture={arch}"),
        "--fmad=false".to_string(),
        "-default-device".to_string(),
    ]
    .into_iter()
    .map(|o| CString::new(o).unwrap())
    .collect();
    let opt_ptrs: Vec<*const c_char> = opts.iter().map(|o| o.as_ptr()).collect();

    let result = (|| {
        // SAFETY: `prog` is a live program; option pointers outlive the call.
        let r =
            unsafe { (nv.nvrtcCompileProgram)(prog, opt_ptrs.len() as c_int, opt_ptrs.as_ptr()) };
        if r != 0 {
            let mut n = 0usize;
            // SAFETY: valid program and out-pointer; buffer sized from the query.
            unsafe { (nv.nvrtcGetProgramLogSize)(prog, &mut n) };
            let mut log = vec![0u8; n.max(1)];
            unsafe { (nv.nvrtcGetProgramLog)(prog, log.as_mut_ptr() as *mut c_char) };
            let log = String::from_utf8_lossy(&log)
                .trim_end_matches('\0')
                .to_string();
            return Err(CudaError::Compile(format!(
                "{} ({arch}):\n{log}",
                // SAFETY: static string.
                unsafe { CStr::from_ptr((nv.nvrtcGetErrorString)(r)) }.to_string_lossy()
            )));
        }
        let mut n = 0usize;
        if cubin {
            let (size_fn, get_fn) = (nv.nvrtcGetCUBINSize.unwrap(), nv.nvrtcGetCUBIN.unwrap());
            // SAFETY: valid program; buffer sized from the size query.
            let r = unsafe { size_fn(prog, &mut n) };
            if r != 0 {
                return Err(nv.err(r, "nvrtcGetCUBINSize"));
            }
            let mut buf = vec![0u8; n];
            let r = unsafe { get_fn(prog, buf.as_mut_ptr() as *mut c_char) };
            if r != 0 {
                return Err(nv.err(r, "nvrtcGetCUBIN"));
            }
            Ok(Image::Cubin(buf))
        } else {
            // SAFETY: as above.
            let r = unsafe { (nv.nvrtcGetPTXSize)(prog, &mut n) };
            if r != 0 {
                return Err(nv.err(r, "nvrtcGetPTXSize"));
            }
            let mut buf = vec![0u8; n + 1];
            let r = unsafe { (nv.nvrtcGetPTX)(prog, buf.as_mut_ptr() as *mut c_char) };
            if r != 0 {
                return Err(nv.err(r, "nvrtcGetPTX"));
            }
            Ok(Image::Ptx(buf))
        }
    })();
    // SAFETY: destroying the program we created.
    unsafe { (nv.nvrtcDestroyProgram)(&mut prog) };
    result
}

/// Compile the kernel source for a virtual architecture, without a GPU or driver.
/// Used by tests to validate that `kernels.cu` is accepted by NVRTC.
#[allow(dead_code)]
pub(crate) fn compile_ptx_only(src: &str, arch: &str) -> Result<usize, CudaError> {
    let nv = Nvrtc::get()?;
    match compile_with(nv, src, arch, false)? {
        Image::Ptx(p) | Image::Cubin(p) => Ok(p.len()),
    }
}
