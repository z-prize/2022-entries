// event management
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html

use crate::result::{CudaResult, CudaResultWrap};
use crate::stream::CudaStream;
use bellman_cuda_cudart_sys::*;
use bitflags::bitflags;
use std::mem::{self, MaybeUninit};
use std::ptr::NonNull;

#[repr(transparent)]
#[derive(Debug)]
pub struct CudaEvent {
    handle: NonNull<CUevent_st>,
}

bitflags! {
    pub struct CudaEventCreateFlags: u32 {
        const DEFAULT = bellman_cuda_cudart_sys::cudaEventDefault;
        const BLOCKING_SYNC = bellman_cuda_cudart_sys::cudaEventBlockingSync;
        const DISABLE_TIMING = bellman_cuda_cudart_sys::cudaEventDisableTiming;
        const INTERPROCESS = bellman_cuda_cudart_sys::cudaEventInterprocess;
    }
}

impl Default for CudaEventCreateFlags {
    fn default() -> Self {
        Self::DEFAULT
    }
}

bitflags! {
    pub struct CudaEventRecordFlags: u32 {
        const DEFAULT = bellman_cuda_cudart_sys::cudaEventRecordDefault;
        const EXTERNAL = bellman_cuda_cudart_sys::cudaEventRecordExternal;
    }
}

impl Default for CudaEventRecordFlags {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl CudaEvent {
    fn from_handle(handle: cudaEvent_t) -> Self {
        Self {
            handle: NonNull::new(handle).unwrap(),
        }
    }

    pub fn create() -> CudaResult<Self> {
        let mut handle = MaybeUninit::<cudaEvent_t>::uninit();
        unsafe {
            cudaEventCreate(handle.as_mut_ptr())
                .wrap_maybe_uninit(handle)
                .map(Self::from_handle)
        }
    }

    pub fn create_with_flags(flags: CudaEventCreateFlags) -> CudaResult<Self> {
        let mut handle = MaybeUninit::<cudaEvent_t>::uninit();
        unsafe {
            cudaEventCreateWithFlags(handle.as_mut_ptr(), flags.bits)
                .wrap_maybe_uninit(handle)
                .map(Self::from_handle)
        }
    }

    pub fn destroy(self) -> CudaResult<()> {
        let handle = self.handle;
        mem::forget(self);
        unsafe { cudaEventDestroy(handle.as_ptr()).wrap() }
    }

    pub fn query(&self) -> CudaResult<bool> {
        let error = unsafe { cudaEventQuery(self.handle.as_ptr()) };
        match error {
            CudaError::Success => Ok(true),
            CudaError::ErrorNotReady => Ok(false),
            _ => Err(error),
        }
    }

    pub fn record(&self, stream: &CudaStream) -> CudaResult<()> {
        unsafe { cudaEventRecord(self.handle.as_ptr(), stream.into()).wrap() }
    }

    pub fn record_with_flags(
        &self,
        stream: &CudaStream,
        flags: CudaEventRecordFlags,
    ) -> CudaResult<()> {
        unsafe { cudaEventRecordWithFlags(self.handle.as_ptr(), stream.into(), flags.bits).wrap() }
    }

    pub fn synchronize(&self) -> CudaResult<()> {
        unsafe { cudaEventSynchronize(self.handle.as_ptr()).wrap() }
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        let _ = unsafe { cudaEventDestroy(self.handle.as_ptr()) };
    }
}

impl From<&CudaEvent> for cudaEvent_t {
    fn from(event: &CudaEvent) -> Self {
        event.handle.as_ptr()
    }
}

pub fn elapsed_time(start: &CudaEvent, end: &CudaEvent) -> CudaResult<f32> {
    let mut ms = MaybeUninit::<f32>::uninit();
    unsafe {
        cudaEventElapsedTime(ms.as_mut_ptr(), start.handle.as_ptr(), end.handle.as_ptr())
            .wrap_maybe_uninit(ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{self, HostFn};
    use serial_test::serial;
    use std::ptr::null_mut;
    use std::thread;
    use std::time::Duration;

    #[test]
    #[serial]
    fn create_is_ok() {
        let result = CudaEvent::create();
        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn create_handle_is_not_null() {
        let event = CudaEvent::create().unwrap();
        assert_ne!(event.handle.as_ptr(), null_mut());
    }

    #[test]
    #[serial]
    fn create_with_flags_is_ok() {
        let result = CudaEvent::create_with_flags(
            CudaEventCreateFlags::DISABLE_TIMING | CudaEventCreateFlags::BLOCKING_SYNC,
        );
        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn create_with_flags_handle_is_not_null() {
        let event = CudaEvent::create_with_flags(CudaEventCreateFlags::DISABLE_TIMING).unwrap();
        assert_ne!(event.handle.as_ptr(), null_mut());
    }

    #[test]
    #[serial]
    fn destroy_is_ok() {
        let event = CudaEvent::create().unwrap();
        let result = event.destroy();
        assert_eq!(result, Ok(()));
    }

    #[test]
    #[serial]
    fn query_is_true() {
        let stream = CudaStream::create().unwrap();
        let event = CudaEvent::create().unwrap();
        event.record(&stream).unwrap();
        stream.synchronize().unwrap();
        let result = event.query();
        assert_eq!(result, Ok(true));
    }

    #[test]
    #[serial]
    fn query_is_false() {
        let stream = CudaStream::create().unwrap();
        let event = CudaEvent::create().unwrap();
        let func = HostFn::new(|| thread::sleep(Duration::from_millis(100)));
        execution::launch_host_fn(&stream, &func).unwrap();
        event.record(&stream).unwrap();
        let result = event.query();
        stream.synchronize().unwrap();
        assert_eq!(result, Ok(false));
    }

    #[test]
    #[serial]
    fn record_is_ok() {
        let stream = CudaStream::create().unwrap();
        let event = CudaEvent::create().unwrap();
        let result = event.record(&stream);
        stream.synchronize().unwrap();
        assert_eq!(result, Ok(()));
    }

    #[test]
    #[serial]
    fn synchronize_is_ok() {
        let stream = CudaStream::create().unwrap();
        let event = CudaEvent::create().unwrap();
        event.record(&stream).unwrap();
        let result = event.synchronize();
        assert_eq!(result, Ok(()));
    }

    #[test]
    #[serial]
    fn elapsed_time_in_range() {
        let stream = CudaStream::create().unwrap();
        let start = CudaEvent::create().unwrap();
        let end = CudaEvent::create().unwrap();
        let func = HostFn::new(|| thread::sleep(Duration::from_millis(10)));
        start.record(&stream).unwrap();
        execution::launch_host_fn(&stream, &func).unwrap();
        end.record(&stream).unwrap();
        stream.synchronize().unwrap();
        let elapsed = elapsed_time(&start, &end).unwrap();
        assert!(elapsed > 10.0 && elapsed < 100.0);
    }
}
