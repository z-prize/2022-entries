use crate::api::{
    cuLaunchKernel, cuStreamCreate, cuStreamDestroy_v2, cuStreamSynchronize,
    CUstream,
};
use crate::context::CudaContext;
use crate::error::CudaResult;
use crate::flags::StreamFlags;
use crate::kernel::CudaKernel;

use log::error;

#[derive(Clone)]
pub struct CudaStream {
    inner: CUstream,
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    pub fn new() -> CudaResult<Self> {
        let flags = StreamFlags::NON_BLOCKING;

        let mut stream = 0 as CUstream;

        let result = unsafe { cuStreamCreate(&mut stream as *mut CUstream, flags.bits()) };

        if result == 0 {
            Ok(CudaStream { inner: stream })
        } else {
            Err(result.into())
        }
    }

    pub fn new_with_context(ctx: &CudaContext) -> CudaResult<Self> {
        ctx.set()?;

        let flags = StreamFlags::NON_BLOCKING;

        let mut stream = 0 as CUstream;

        let result = unsafe { cuStreamCreate(&mut stream as *mut CUstream, flags.bits()) };

        if result == 0 {
            Ok(CudaStream { inner: stream })
        } else {
            Err(result.into())
        }
    }

    pub fn get_inner(&self) -> CUstream {
        self.inner
    }

    pub fn destroy(&self) -> CudaResult<()> {
        let result = unsafe { cuStreamDestroy_v2(self.inner) };

        if result == 0 {
            Ok(())
        } else {
            Err(result.into())
        }
    }

    pub fn sync(&self) -> CudaResult<()> {
        let result = unsafe { cuStreamSynchronize(self.inner) };

        if result == 0 {
            Ok(())
        } else {
            Err(result.into())
        }
    }

    pub fn launch(&self, kern: &CudaKernel) -> CudaResult<()> {
        let result = unsafe {
            cuLaunchKernel(
                kern.kernel.func,
                kern.kernel.gridDimX,
                kern.kernel.gridDimY,
                kern.kernel.gridDimZ,
                kern.kernel.blockDimX,
                kern.kernel.blockDimY,
                kern.kernel.blockDimZ,
                kern.kernel.sharedMemBytes,
                self.inner,
                kern.kernel.kernelParams,
                kern.kernel.extra,
            )
        };

        if result == 0 {
            Ok(())
        } else {
            Err(result.into())
        }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        let r = self.destroy();
        if let Err(e) = r {
            error!("failed when dropping stream, cuda error: {}", e);
        }
    }
}
