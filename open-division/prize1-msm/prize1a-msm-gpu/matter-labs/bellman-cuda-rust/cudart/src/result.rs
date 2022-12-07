use bellman_cuda_cudart_sys::CudaError;
use std::mem::MaybeUninit;

pub type CudaResult<T> = Result<T, CudaError>;

pub trait CudaResultWrap {
    fn wrap(self) -> CudaResult<()>;
    fn wrap_value<T>(self, value: T) -> CudaResult<T>;
    fn wrap_maybe_uninit<T>(self, value: MaybeUninit<T>) -> CudaResult<T>;
}

impl CudaResultWrap for CudaError {
    fn wrap(self) -> CudaResult<()> {
        self.wrap_value(())
    }

    fn wrap_value<T>(self, value: T) -> CudaResult<T> {
        if self == CudaError::Success {
            Ok(value)
        } else {
            Err(self)
        }
    }

    fn wrap_maybe_uninit<T>(self, value: MaybeUninit<T>) -> CudaResult<T> {
        self.wrap_value(value).map(|x| unsafe { x.assume_init() })
    }
}
