use std::ptr;

use crate::api::{cuMemAllocHost_v2, cuMemFreeHost};
use crate::error::CudaResult;

use log::error;

/// HostMemory can maximize the transfer speed between host and device, but need to prepare the data first
/// not very convenient to use
pub struct HostMemory<T> {
    buf: *mut T,
    length: usize,
}

impl<T> HostMemory<T> {
    pub fn new(size: usize) -> CudaResult<Self> {
        let mut mem_ptr: *mut std::os::raw::c_void = ptr::null_mut();

        let byte_size = (size as u64) * (std::mem::size_of::<T>() as u64);

        let result =
            unsafe { cuMemAllocHost_v2(&mut mem_ptr as *mut *mut std::os::raw::c_void, byte_size) };

        if result == 0 {
            Ok(HostMemory {
                buf: mem_ptr as *mut T,
                length: size,
            })
        } else {
            Err(result.into())
        }
    }

    fn free(&mut self) -> CudaResult<()> {
        let result = unsafe { cuMemFreeHost(self.buf as *mut std::os::raw::c_void) };

        if result == 0 {
            Ok(())
        } else {
            Err(result.into())
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }

    pub fn as_slice(&self) -> &[T] {
        self
    }
}

impl<T> AsRef<[T]> for HostMemory<T> {
    fn as_ref(&self) -> &[T] {
        self
    }
}

impl<T> AsMut<[T]> for HostMemory<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self
    }
}

impl<T> std::ops::Deref for HostMemory<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe {
            let p = self.buf;
            std::slice::from_raw_parts(p, self.length)
        }
    }
}
impl<T> std::ops::DerefMut for HostMemory<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe {
            let ptr = self.buf;
            std::slice::from_raw_parts_mut(ptr, self.length)
        }
    }
}

impl<T> Drop for HostMemory<T> {
    fn drop(&mut self) {
        let r = self.free();
        if let Err(e) = r {
            error!("failed when dropping host memory, cuda error: {}", e);
        }
    }
}
