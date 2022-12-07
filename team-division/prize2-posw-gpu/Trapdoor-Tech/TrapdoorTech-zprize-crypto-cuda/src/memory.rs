use std::marker::PhantomData;

use crate::api::{
    cuMemAlloc_v2, cuMemFree_v2, cuMemcpy, cuMemcpyAsync, cuMemcpyDtoDAsync_v2, cuMemcpyDtoD_v2,
    cuMemcpyDtoH_v2, cuMemcpyHtoD_v2, cuMemcpyPeer, CUdeviceptr, cuMemsetD32Async,
};
use crate::context::CudaContext;
use crate::error::CudaResult;
use crate::stream::CudaStream;

use crate::host_memory::HostMemory;

use log::error;

#[derive(Clone)]
pub struct DeviceMemory<T> {
    inner: CUdeviceptr,
    size: usize,
    _marker: PhantomData<T>,
}

impl<T> DeviceMemory<T> {
    pub fn new(ctx: &CudaContext, size: usize) -> CudaResult<Self> {
        ctx.set()?;

        let mut mem_ptr = 0 as CUdeviceptr;

        let byte_size = (size as u64) * (std::mem::size_of::<T>() as u64);

        let result = unsafe { cuMemAlloc_v2(&mut mem_ptr as *mut CUdeviceptr, byte_size) };

        if result == 0 {
            Ok(DeviceMemory {
                inner: mem_ptr,
                size,
                _marker: PhantomData,
            })
        } else {
            Err(result.into())
        }
    }

    fn free(&mut self) -> CudaResult<()> {
        let result = unsafe { cuMemFree_v2(self.inner) };

        if result == 0 {
            Ok(())
        } else {
            Err(result.into())
        }
    }

    pub fn get_inner(&self) -> CUdeviceptr {
        self.inner
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn read_from_alter(&self, src: &[T], size: usize) -> CudaResult<()> {
        let n = std::cmp::min(self.size, size);

        let size_in_bytes = std::mem::size_of::<T>() * n;

        let result = unsafe {
            cuMemcpyHtoD_v2(
                self.inner,
                src.as_ptr() as *const std::os::raw::c_void,
                size_in_bytes as u64,
            )
        };

        if result == 0 {
            Ok(())
        } else {
            Err(result.into())
        }
    }

    pub fn write_to_alter(&self, dst: &mut [T], size: usize) -> CudaResult<()> {
        let n = std::cmp::min(self.size, size);

        let size_in_bytes = std::mem::size_of::<T>() * n;

        let result = unsafe {
            cuMemcpyDtoH_v2(
                dst.as_mut_ptr() as *mut std::os::raw::c_void,
                self.inner,
                size_in_bytes as u64,
            )
        };

        if result == 0 {
            Ok(())
        } else {
            Err(result.into())
        }
    }

    pub fn read_from(&self, src: &[T], size: usize) -> CudaResult<()> {
        let n = std::cmp::min(self.size, size);

        let size_in_bytes = std::mem::size_of::<T>() * n;

        let result = unsafe {
            cuMemcpy(
                self.inner,
                src.as_ptr() as CUdeviceptr,
                size_in_bytes as u64,
            )
        };

        if result == 0 {
            Ok(())
        } else {
            Err(result.into())
        }
    }

    pub fn read_from_async(&self, src: &[T], size: usize, stream: &CudaStream) -> CudaResult<()> {
        let n = std::cmp::min(self.size, size);

        let size_in_bytes = std::mem::size_of::<T>() * n;

        let result = unsafe {
            cuMemcpyAsync(
                self.inner,
                src.as_ptr() as CUdeviceptr,
                size_in_bytes as u64,
                stream.get_inner(),
            )
        };

        if result == 0 {
            Ok(())
        } else {
            Err(result.into())
        }
    }

    pub fn write_to(&self, dst: &mut [T], size: usize) -> CudaResult<()> {
        let n = std::cmp::min(self.size, size);

        let size_in_bytes = std::mem::size_of::<T>() * n;

        let result = unsafe {
            cuMemcpy(
                dst.as_mut_ptr() as CUdeviceptr,
                self.inner,
                size_in_bytes as u64,
            )
        };

        if result == 0 {
            Ok(())
        } else {
            Err(result.into())
        }
    }

    pub fn write_to_async(
        &self,
        dst: &mut [T],
        size: usize,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let n = std::cmp::min(self.size, size);

        let size_in_bytes = std::mem::size_of::<T>() * n;

        let result = unsafe {
            cuMemcpyAsync(
                dst.as_mut_ptr() as CUdeviceptr,
                self.inner,
                size_in_bytes as u64,
                stream.get_inner(),
            )
        };

        if result == 0 {
            Ok(())
        } else {
            Err(result.into())
        }
    }

    pub fn read_from_host(&self, src: &HostMemory<T>, size: usize) -> CudaResult<()> {
        let result = unsafe {
            cuMemcpyHtoD_v2(
                self.inner,
                src.as_ptr() as *const std::os::raw::c_void,
                size as u64,
            )
        };

        if result == 0 {
            Ok(())
        } else {
            Err(result.into())
        }
    }

    pub fn write_to_host(&self, dst: &mut HostMemory<T>, size: usize) -> CudaResult<()> {
        let result = unsafe {
            cuMemcpyDtoH_v2(
                dst.as_mut_ptr() as *mut std::os::raw::c_void,
                self.inner,
                size as u64,
            )
        };

        if result == 0 {
            Ok(())
        } else {
            Err(result.into())
        }
    }

    pub fn memcpy_from_raw(&self, src: CUdeviceptr, size: usize) -> CudaResult<()> {
        let n = std::cmp::min(self.size, size);

        let size_in_bytes = std::mem::size_of::<T>() * n;

        let result = unsafe { cuMemcpy(self.inner, src, size_in_bytes as u64) };

        if result == 0 {
            Ok(())
        } else {
            Err(result.into())
        }
    }

    pub fn memset_async(
        &self,
        value: u32,
        size: usize,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let n = std::cmp::min(self.size, size);

        let size_in_u32 = std::mem::size_of::<T>() * n / 4;

        let result = unsafe { cuMemsetD32Async(self.inner, value, size_in_u32 as u64, stream.get_inner()) };

        if result == 0 {
            Ok(())
        } else {
            Err(result.into())
        }

    }

    pub fn memcpy_from_to(
        src: &DeviceMemory<T>,
        dst: &DeviceMemory<T>,
        size: Option<usize>,
    ) -> CudaResult<()> {
        let n = if let Some(x) = size {
            x
        } else {
            std::cmp::min(src.size(), dst.size())
        };

        let n = std::cmp::min(n, src.size());
        let n = std::cmp::min(n, dst.size());

        let size_in_bytes: u64 = n as u64 * std::mem::size_of::<T>() as u64;

        let result = unsafe { cuMemcpyDtoD_v2(dst.get_inner(), src.get_inner(), size_in_bytes) };

        if result == 0 {
            Ok(())
        } else {
            Err(result.into())
        }
    }

    pub fn memcpy_from_to_async(
        src: &DeviceMemory<T>,
        dst: &DeviceMemory<T>,
        size: Option<usize>,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let n = if let Some(x) = size {
            x
        } else {
            std::cmp::min(src.size(), dst.size())
        };

        let n = std::cmp::min(n, src.size());
        let n = std::cmp::min(n, dst.size());

        let size_in_bytes: u64 = n as u64 * std::mem::size_of::<T>() as u64;

        let result = unsafe {
            cuMemcpyDtoDAsync_v2(
                dst.get_inner(),
                src.get_inner(),
                size_in_bytes,
                stream.get_inner(),
            )
        };

        if result == 0 {
            Ok(())
        } else {
            Err(result.into())
        }
    }
}

/// This is supposed to be asynchronous due to cuda docs.
/// [CUDA synchronization behavior](https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html#api-sync-behavior)
pub fn cuda_link_transfer<T>(
    src: &DeviceMemory<T>,
    src_context: &CudaContext,
    dst: &DeviceMemory<T>,
    dst_context: &CudaContext,
    size: usize,
) -> CudaResult<()> {
    let bytes: u64 = size as u64 * std::mem::size_of::<T>() as u64;
    let result = unsafe {
        cuMemcpyPeer(
            dst.get_inner(),
            dst_context.get_inner(),
            src.get_inner(),
            src_context.get_inner(),
            bytes,
        )
    };

    if result == 0 {
        Ok(())
    } else {
        Err(result.into())
    }
}

impl<T> Drop for DeviceMemory<T> {
    fn drop(&mut self) {
        let r = self.free();
        if let Err(e) = r {
            error!("failed when dropping device memory, cuda error: {}", e);
        }
    }
}
