//! Device handle, typed device buffers, and kernel launch.
//!
//! [`CudaDevice`] wraps one of two backends behind the same buffer/launch API:
//!
//! * **CUDA** — a real GPU through the driver API, kernels compiled with NVRTC;
//! * **emulator** (`--features cuda-emu`) — the *same* `kernels.cu`, compiled as
//!   C++ by `build.rs` and executed block-by-block on the CPU. Lets the full
//!   GPU code path (host orchestration + kernels) be tested without a GPU.
//!
//! Every kernel takes a single `#[repr(C)]` parameter struct by value whose
//! layout mirrors a struct of the same name in `kernels.cu`.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda-emu")]
use std::ffi::CString;
use std::ffi::c_void;

use super::CudaError;
use super::driver::{CuContext, CuDevice, CuFunction, CuModule, Driver};

/// CUDA source for every kernel in the crate.
pub(crate) const KERNEL_SRC: &str = include_str!("kernels.cu");

#[cfg(feature = "cuda-emu")]
unsafe extern "C" {
    fn tmu_emu_launch(
        name: *const std::ffi::c_char,
        grid: u32,
        block: u32,
        smem: u32,
        params: *const c_void,
        param_size: usize,
    ) -> i32;
}

struct CudaCtx {
    drv: &'static Driver,
    dev: CuDevice,
    ctx: CuContext,
    module: CuModule,
    funcs: Mutex<HashMap<&'static str, usize>>,
}

// SAFETY: the context/module handles are process-global driver objects; every
// use first makes the context current on the calling thread.
unsafe impl Send for CudaCtx {}
unsafe impl Sync for CudaCtx {}

impl Drop for CudaCtx {
    fn drop(&mut self) {
        // SAFETY: tearing down handles we created; errors are ignored on drop.
        unsafe {
            (self.drv.cuCtxSetCurrent)(self.ctx);
            (self.drv.cuModuleUnload)(self.module);
            (self.drv.cuDevicePrimaryCtxRelease)(self.dev);
        }
    }
}

enum Backend {
    Cuda(CudaCtx),
    #[cfg(feature = "cuda-emu")]
    Emu,
}

pub(crate) struct DeviceInner {
    backend: Backend,
    name: String,
    /// Threads per block for clause-cooperative kernels.
    block: u32,
}

/// A handle to a CUDA GPU (or the CPU kernel emulator), shared by every model
/// placed on it. Cheap to clone.
///
/// ```rust,no_run
/// use tmu_rs::cuda::CudaDevice;
/// let dev = CudaDevice::new(0).expect("no CUDA GPU");
/// println!("training on {}", dev.name());
/// ```
#[derive(Clone)]
pub struct CudaDevice {
    pub(crate) inner: Arc<DeviceInner>,
}

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaDevice")
            .field("name", &self.inner.name)
            .finish()
    }
}

impl CudaDevice {
    /// Open GPU `ordinal` (0 = first GPU) and JIT-compile the kernels for it.
    ///
    /// Requires an NVIDIA driver (`libcuda`) and NVRTC (`libnvrtc`, part of the
    /// CUDA toolkit or the `nvidia-cuda-nvrtc` pip wheel). Set
    /// `TMU_CUDA_DRIVER_PATH` / `TMU_NVRTC_PATH` to point at non-standard
    /// locations.
    pub fn new(ordinal: usize) -> Result<Self, CudaError> {
        let drv = Driver::get()?;
        let count = drv.device_count()?;
        if ordinal >= count {
            return Err(CudaError::NoDevice(format!(
                "ordinal {ordinal} requested but {count} CUDA device(s) present"
            )));
        }
        let mut dev: CuDevice = 0;
        // SAFETY: valid out-pointer; ordinal checked above.
        drv.check(
            unsafe { (drv.cuDeviceGet)(&mut dev, ordinal as i32) },
            "cuDeviceGet",
        )?;
        let name = drv.device_name(dev)?;
        let cc = drv.compute_capability(dev)?;
        let mut ctx: CuContext = std::ptr::null_mut();
        // SAFETY: valid out-pointer and device.
        drv.check(
            unsafe { (drv.cuDevicePrimaryCtxRetain)(&mut ctx, dev) },
            "cuDevicePrimaryCtxRetain",
        )?;
        // SAFETY: ctx was just retained.
        drv.check(unsafe { (drv.cuCtxSetCurrent)(ctx) }, "cuCtxSetCurrent")?;
        let image = super::driver::compile(KERNEL_SRC, cc)?;
        let mut module: CuModule = std::ptr::null_mut();
        // SAFETY: image is a complete CUBIN or NUL-terminated PTX.
        let r = unsafe { (drv.cuModuleLoadData)(&mut module, image.as_ptr()) };
        if let Err(e) = drv.check(r, "cuModuleLoadData") {
            // SAFETY: release the context we retained above.
            unsafe { (drv.cuDevicePrimaryCtxRelease)(dev) };
            return Err(e);
        }
        Ok(CudaDevice {
            inner: Arc::new(DeviceInner {
                backend: Backend::Cuda(CudaCtx {
                    drv,
                    dev,
                    ctx,
                    module,
                    funcs: Mutex::new(HashMap::new()),
                }),
                name: format!("{name} (sm_{}{})", cc.0, cc.1),
                block: 128,
            }),
        })
    }

    /// Number of CUDA devices visible to the driver (0 if the driver is missing).
    pub fn count() -> usize {
        Driver::get().and_then(|d| d.device_count()).unwrap_or(0)
    }

    /// The CPU emulator backend: runs the exact CUDA kernels on the host, one
    /// block at a time, with 4 cooperating threads per block. Results are
    /// bit-identical to a real GPU. Intended for testing — it is slow.
    #[cfg(feature = "cuda-emu")]
    pub fn emulator() -> Self {
        Self::emulator_with_block(4)
    }

    /// Emulator with `block` cooperating threads per clause block (1 = serial,
    /// fastest; >1 exercises the inter-thread synchronisation paths).
    #[cfg(feature = "cuda-emu")]
    pub fn emulator_with_block(block: u32) -> Self {
        assert!(
            (1..=256).contains(&block),
            "emulator block size must be 1..=256"
        );
        CudaDevice {
            inner: Arc::new(DeviceInner {
                backend: Backend::Emu,
                name: format!("CPU emulator ({block} threads/block)"),
                block,
            }),
        }
    }

    /// Human-readable device name.
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// Block until all queued GPU work has finished.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.inner.synchronize()
    }
}

impl DeviceInner {
    fn cuda(&self) -> Option<&CudaCtx> {
        match &self.backend {
            Backend::Cuda(c) => Some(c),
            #[cfg(feature = "cuda-emu")]
            Backend::Emu => None,
        }
    }

    fn make_current(c: &CudaCtx) -> Result<(), CudaError> {
        // SAFETY: ctx is a live retained primary context.
        c.drv
            .check(unsafe { (c.drv.cuCtxSetCurrent)(c.ctx) }, "cuCtxSetCurrent")
    }

    pub(crate) fn synchronize(&self) -> Result<(), CudaError> {
        if let Some(c) = self.cuda() {
            Self::make_current(c)?;
            // SAFETY: plain synchronisation call on the current context.
            c.drv
                .check(unsafe { (c.drv.cuCtxSynchronize)() }, "cuCtxSynchronize")?;
        }
        Ok(())
    }

    /// Threads per block for clause-cooperative kernels operating on clauses of
    /// `words` 64-bit words: enough threads to cover the words, capped by the
    /// device preference (and at least one warp on a real GPU).
    pub(crate) fn clause_block(&self, words: usize) -> u32 {
        match &self.backend {
            Backend::Cuda(_) => (words.next_power_of_two() as u32).clamp(32, self.block),
            #[cfg(feature = "cuda-emu")]
            Backend::Emu => self.block.min(words.next_power_of_two() as u32).max(1),
        }
    }

    /// Threads per block for one-thread-per-item kernels.
    pub(crate) fn item_block(&self) -> u32 {
        match &self.backend {
            Backend::Cuda(_) => 128,
            #[cfg(feature = "cuda-emu")]
            Backend::Emu => self.block,
        }
    }

    fn alloc_bytes(&self, bytes: usize) -> Result<u64, CudaError> {
        let bytes = bytes.max(16);
        match &self.backend {
            Backend::Cuda(c) => {
                Self::make_current(c)?;
                let mut p = 0u64;
                // SAFETY: valid out-pointer.
                c.drv
                    .check(unsafe { (c.drv.cuMemAlloc)(&mut p, bytes) }, "cuMemAlloc")?;
                // SAFETY: freshly allocated range of `bytes`.
                c.drv
                    .check(unsafe { (c.drv.cuMemsetD8)(p, 0, bytes) }, "cuMemsetD8")?;
                Ok(p)
            }
            #[cfg(feature = "cuda-emu")]
            Backend::Emu => {
                let layout = std::alloc::Layout::from_size_align(bytes, 16).unwrap();
                // SAFETY: non-zero size layout.
                let p = unsafe { std::alloc::alloc_zeroed(layout) };
                if p.is_null() {
                    std::alloc::handle_alloc_error(layout);
                }
                Ok(p as u64)
            }
        }
    }

    fn free_bytes(&self, ptr: u64, bytes: usize) {
        match &self.backend {
            Backend::Cuda(c) => {
                let _ = bytes;
                if Self::make_current(c).is_ok() {
                    // SAFETY: ptr came from cuMemAlloc in this context.
                    unsafe { (c.drv.cuMemFree)(ptr) };
                }
            }
            #[cfg(feature = "cuda-emu")]
            Backend::Emu => {
                let layout = std::alloc::Layout::from_size_align(bytes.max(16), 16).unwrap();
                // SAFETY: allocated by alloc_bytes with this exact layout.
                unsafe { std::alloc::dealloc(ptr as *mut u8, layout) };
            }
        }
    }

    fn htod(&self, dst: u64, src: *const u8, bytes: usize) -> Result<(), CudaError> {
        if bytes == 0 {
            return Ok(());
        }
        match &self.backend {
            Backend::Cuda(c) => {
                Self::make_current(c)?;
                // SAFETY: caller guarantees both ranges are valid for `bytes`.
                c.drv.check(
                    unsafe { (c.drv.cuMemcpyHtoD)(dst, src as *const c_void, bytes) },
                    "cuMemcpyHtoD",
                )
            }
            #[cfg(feature = "cuda-emu")]
            Backend::Emu => {
                // SAFETY: as above; emulator "device" memory is host memory.
                unsafe { std::ptr::copy_nonoverlapping(src, dst as *mut u8, bytes) };
                Ok(())
            }
        }
    }

    fn dtoh(&self, dst: *mut u8, src: u64, bytes: usize) -> Result<(), CudaError> {
        if bytes == 0 {
            return Ok(());
        }
        match &self.backend {
            Backend::Cuda(c) => {
                Self::make_current(c)?;
                // SAFETY: caller guarantees both ranges are valid for `bytes`.
                c.drv.check(
                    unsafe { (c.drv.cuMemcpyDtoH)(dst as *mut c_void, src, bytes) },
                    "cuMemcpyDtoH",
                )
            }
            #[cfg(feature = "cuda-emu")]
            Backend::Emu => {
                // SAFETY: as above.
                unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst, bytes) };
                Ok(())
            }
        }
    }

    fn memset0(&self, dst: u64, bytes: usize) -> Result<(), CudaError> {
        if bytes == 0 {
            return Ok(());
        }
        match &self.backend {
            Backend::Cuda(c) => {
                Self::make_current(c)?;
                // SAFETY: caller guarantees the range is a live allocation.
                c.drv
                    .check(unsafe { (c.drv.cuMemsetD8)(dst, 0, bytes) }, "cuMemsetD8")
            }
            #[cfg(feature = "cuda-emu")]
            Backend::Emu => {
                // SAFETY: as above.
                unsafe { std::ptr::write_bytes(dst as *mut u8, 0, bytes) };
                Ok(())
            }
        }
    }

    /// Launch kernel `name` on a 1-D grid with `params` passed by value.
    ///
    /// `P` must be `#[repr(C)]` and mirror the kernel's parameter struct.
    pub(crate) fn launch<P: Copy>(
        &self,
        name: &'static str,
        grid: u32,
        block: u32,
        smem: u32,
        params: &P,
    ) -> Result<(), CudaError> {
        if grid == 0 {
            return Ok(());
        }
        match &self.backend {
            Backend::Cuda(c) => {
                Self::make_current(c)?;
                let f = {
                    let mut funcs = c.funcs.lock().unwrap();
                    match funcs.get(name) {
                        Some(&f) => f as CuFunction,
                        None => {
                            let cname = std::ffi::CString::new(name).unwrap();
                            let mut f: CuFunction = std::ptr::null_mut();
                            // SAFETY: module is loaded; out-pointer valid.
                            c.drv.check(
                                unsafe {
                                    (c.drv.cuModuleGetFunction)(&mut f, c.module, cname.as_ptr())
                                },
                                "cuModuleGetFunction",
                            )?;
                            funcs.insert(name, f as usize);
                            f
                        }
                    }
                };
                let mut arg = params as *const P as *mut c_void;
                // SAFETY: one kernel argument (the param struct), matching the
                // kernel prototype; `params` outlives the (copying) launch call.
                c.drv.check(
                    unsafe {
                        (c.drv.cuLaunchKernel)(
                            f,
                            grid,
                            1,
                            1,
                            block,
                            1,
                            1,
                            smem,
                            std::ptr::null_mut(),
                            &mut arg,
                            std::ptr::null_mut(),
                        )
                    },
                    name,
                )
            }
            #[cfg(feature = "cuda-emu")]
            Backend::Emu => {
                let cname = CString::new(name).unwrap();
                // SAFETY: emulator executes the kernel synchronously on the host
                // with the same parameter struct layout.
                let r = unsafe {
                    tmu_emu_launch(
                        cname.as_ptr(),
                        grid,
                        block,
                        smem,
                        params as *const P as *const c_void,
                        std::mem::size_of::<P>(),
                    )
                };
                match r {
                    0 => Ok(()),
                    1 => Err(CudaError::Driver {
                        call: name.to_string(),
                        code: 1,
                        msg: "emulator: unknown kernel".into(),
                    }),
                    _ => Err(CudaError::Driver {
                        call: name.to_string(),
                        code: r,
                        msg: "emulator: parameter struct size mismatch".into(),
                    }),
                }
            }
        }
    }
}

/// A typed allocation in device memory, freed on drop.
pub(crate) struct DevBuf<T: Copy> {
    ptr: u64,
    len: usize,
    dev: Arc<DeviceInner>,
    _t: PhantomData<T>,
}

impl<T: Copy> DevBuf<T> {
    /// Zero-initialised buffer of `len` elements.
    pub(crate) fn zeros(dev: &CudaDevice, len: usize) -> Result<Self, CudaError> {
        let ptr = dev.inner.alloc_bytes(len * size_of::<T>())?;
        Ok(DevBuf {
            ptr,
            len,
            dev: dev.inner.clone(),
            _t: PhantomData,
        })
    }

    /// Buffer initialised from host data.
    pub(crate) fn from_slice(dev: &CudaDevice, data: &[T]) -> Result<Self, CudaError> {
        let b = Self::zeros(dev, data.len())?;
        b.upload(data)?;
        Ok(b)
    }

    /// Device address of element 0.
    pub(crate) fn ptr(&self) -> u64 {
        self.ptr
    }

    /// Device address of element `i`.
    pub(crate) fn at(&self, i: usize) -> u64 {
        debug_assert!(i <= self.len);
        self.ptr + (i * size_of::<T>()) as u64
    }

    /// Copy `data` into the start of the buffer.
    pub(crate) fn upload(&self, data: &[T]) -> Result<(), CudaError> {
        assert!(data.len() <= self.len, "upload larger than device buffer");
        self.dev.htod(
            self.ptr,
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data),
        )
    }

    /// Copy the whole buffer back to the host.
    pub(crate) fn download(&self) -> Result<Vec<T>, CudaError> {
        let mut out: Vec<T> = Vec::with_capacity(self.len);
        self.dev.dtoh(
            out.as_mut_ptr() as *mut u8,
            self.ptr,
            self.len * size_of::<T>(),
        )?;
        // SAFETY: dtoh fully initialised `len` elements of a plain-old-data T.
        unsafe { out.set_len(self.len) };
        Ok(out)
    }

    /// Copy the first `out.len()` elements back into `out`.
    pub(crate) fn download_into(&self, out: &mut [T]) -> Result<(), CudaError> {
        assert!(out.len() <= self.len);
        self.dev.dtoh(
            out.as_mut_ptr() as *mut u8,
            self.ptr,
            std::mem::size_of_val(out),
        )
    }

    /// Zero the first `n` elements.
    pub(crate) fn zero_prefix(&self, n: usize) -> Result<(), CudaError> {
        assert!(n <= self.len);
        self.dev.memset0(self.ptr, n * size_of::<T>())
    }
}

impl<T: Copy> Drop for DevBuf<T> {
    fn drop(&mut self) {
        self.dev.free_bytes(self.ptr, self.len * size_of::<T>());
    }
}
