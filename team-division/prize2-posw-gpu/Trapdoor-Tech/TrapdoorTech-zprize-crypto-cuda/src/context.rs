/// This module manages cuda context
use crate::api::{
    cuCtxCreate_v2, cuCtxDestroy_v2, cuCtxEnablePeerAccess, cuCtxSetCurrent, cuCtxSynchronize,
    CUcontext,
};
use crate::device::CudaDevice;
use crate::error::CudaResult;
use crate::flags::ContextFlags;

use log::error;

#[derive(Clone)]
pub struct CudaContext {
    inner: CUcontext,
}

/// USE THIS WITH CAUTION: literally context can be shared between threads, but can only be dropped once
/// better to use an `Arc` reference counter for sharing context
unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaContext {
    pub fn new(dev: CudaDevice) -> CudaResult<Self> {
        let mut ctx = 0 as CUcontext;
        let flags = ContextFlags::SCHED_AUTO | ContextFlags::MAP_HOST;

        let result =
            unsafe { cuCtxCreate_v2(&mut ctx as *mut CUcontext, flags.bits(), dev.into()) };

        if result == 0 {
            Ok(CudaContext { inner: ctx })
        } else {
            error!("cuda context not created!");
            Err(result.into())
        }
    }

    pub fn destroy(&self) -> CudaResult<()> {
        let result = unsafe { cuCtxDestroy_v2(self.inner) };

        if result == 0 {
            Ok(())
        } else {
            Err(result.into())
        }
    }

    pub fn set(&self) -> CudaResult<()> {
        let result = unsafe { cuCtxSetCurrent(self.inner) };

        if result == 0 {
            Ok(())
        } else {
            Err(result.into())
        }
    }

    pub fn sync(&self) -> CudaResult<()> {
        self.set()?;

        let result = unsafe { cuCtxSynchronize() };

        if result == 0 {
            Ok(())
        } else {
            Err(result.into())
        }
    }

    /// According to cuda docs, the flags must be set to 0
    /// [enable_peer_access](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PEER__ACCESS.html#group__CUDA__PEER__ACCESS_1g0889ec6728e61c05ed359551d67b3f5a)
    pub fn enable_peer(&self, other: &CudaContext) -> CudaResult<()> {
        self.set()?;

        let flags = 0 as u32;
        let result = unsafe { cuCtxEnablePeerAccess(other.get_inner(), flags) };

        if result == 0 {
            Ok(())
        } else {
            Err(result.into())
        }
    }

    pub fn get_inner(&self) -> CUcontext {
        self.inner
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        let r = self.destroy();
        if let Err(e) = r {
            error!("failed when dropping context, cuda error: {}", e);
        }
    }
}
