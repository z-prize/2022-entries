// stream management
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html

use crate::event::CudaEvent;
use crate::result::{CudaResult, CudaResultWrap};
use bellman_cuda_cudart_sys::*;
use bitflags::bitflags;
use std::mem::{self, MaybeUninit};
use std::ptr::null_mut;

#[repr(transparent)]
#[derive(Debug)]
pub struct CudaStream {
    handle: cudaStream_t,
}

bitflags! {
    pub struct CudaStreamCreateFlags: u32 {
        const DEFAULT = bellman_cuda_cudart_sys::cudaStreamDefault;
        const NON_BLOCKING = bellman_cuda_cudart_sys::cudaStreamNonBlocking;
    }
}

impl Default for CudaStreamCreateFlags {
    fn default() -> Self {
        Self::DEFAULT
    }
}

bitflags! {
    pub struct CudaStreamWaitEventFlags: u32 {
        const DEFAULT = bellman_cuda_cudart_sys::cudaEventWaitDefault;
        const WAIT_EXTERNAL = bellman_cuda_cudart_sys::cudaEventWaitExternal;
    }
}

impl Default for CudaStreamWaitEventFlags {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl CudaStream {
    fn from_handle(handle: cudaStream_t) -> Self {
        Self { handle }
    }

    pub fn create() -> CudaResult<Self> {
        let mut handle = MaybeUninit::<cudaStream_t>::uninit();
        unsafe {
            cudaStreamCreate(handle.as_mut_ptr())
                .wrap_maybe_uninit(handle)
                .map(CudaStream::from_handle)
        }
    }

    pub fn create_with_flags(flags: CudaStreamCreateFlags) -> CudaResult<Self> {
        let mut handle = MaybeUninit::<cudaStream_t>::uninit();
        unsafe {
            cudaStreamCreateWithFlags(handle.as_mut_ptr(), flags.bits)
                .wrap_maybe_uninit(handle)
                .map(CudaStream::from_handle)
        }
    }

    pub fn destroy(self) -> CudaResult<()> {
        let handle = self.handle;
        mem::forget(self);
        if handle.is_null() {
            Ok(())
        } else {
            unsafe { cudaStreamDestroy(handle).wrap() }
        }
    }

    pub fn query(&self) -> CudaResult<bool> {
        let error = unsafe { cudaStreamQuery(self.handle) };
        match error {
            CudaError::Success => Ok(true),
            CudaError::ErrorNotReady => Ok(false),
            _ => Err(error),
        }
    }

    pub fn synchronize(&self) -> CudaResult<()> {
        unsafe { cudaStreamSynchronize(self.handle).wrap() }
    }

    pub fn wait_event(&self, event: &CudaEvent, flags: CudaStreamWaitEventFlags) -> CudaResult<()> {
        unsafe { cudaStreamWaitEvent(self.handle, event.into(), flags.bits).wrap() }
    }
}

impl Default for CudaStream {
    fn default() -> Self {
        Self { handle: null_mut() }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        let handle = self.handle;
        if handle.is_null() {
            return;
        }
        let _ = unsafe { cudaStreamDestroy(handle) };
    }
}

impl From<&CudaStream> for cudaStream_t {
    fn from(stream: &CudaStream) -> Self {
        stream.handle
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{launch_host_fn, HostFn};
    use serial_test::serial;
    use std::thread;
    use std::time::Duration;

    #[test]
    #[serial]
    fn create_is_ok() {
        let result = CudaStream::create();
        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn create_handle_is_not_null() {
        let stream = CudaStream::create().unwrap();
        assert_ne!(stream.handle, null_mut());
    }

    #[test]
    #[serial]
    fn create_with_flags_is_ok() {
        let result = CudaStream::create_with_flags(CudaStreamCreateFlags::NON_BLOCKING);
        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn create_with_flags_handle_is_not_null() {
        let stream = CudaStream::create_with_flags(CudaStreamCreateFlags::NON_BLOCKING).unwrap();
        assert_ne!(stream.handle, null_mut());
    }

    #[test]
    #[serial]
    fn destroy_is_ok() {
        let stream = CudaStream::create().unwrap();
        let result = stream.destroy();
        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn query_is_true() {
        let stream = CudaStream::create().unwrap();
        let result = stream.query();
        assert_eq!(result, Ok(true));
    }

    #[test]
    #[serial]
    fn query_is_false() {
        let stream = CudaStream::create().unwrap();
        let func = HostFn::new(|| thread::sleep(Duration::from_millis(100)));
        launch_host_fn(&stream, &func).unwrap();
        let result = stream.query();
        assert_eq!(result, Ok(false));
    }

    #[test]
    #[serial]
    fn synchronize_is_ok() {
        let stream = CudaStream::create().unwrap();
        let result = stream.synchronize();
        assert_eq!(result, Ok(()));
    }

    #[test]
    #[serial]
    fn wait_event_is_ok() {
        let stream = CudaStream::create().unwrap();
        let event = CudaEvent::create().unwrap();
        event.record(&stream).unwrap();
        let result = stream.wait_event(&event, CudaStreamWaitEventFlags::DEFAULT);
        assert_eq!(result, Ok(()));
    }
}
