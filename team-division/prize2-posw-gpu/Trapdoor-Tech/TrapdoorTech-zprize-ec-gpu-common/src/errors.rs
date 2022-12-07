#[derive(thiserror::Error, Debug)]
pub enum GPUError {
    #[error("GPUError: {0}")]
    Simple(String),
    #[error("Cuda Wrapped Error: {0}")]
    CudaWrappedError(crypto_cuda::error::CudaError),
}

pub type GPUResult<T> = std::result::Result<T, GPUError>;

impl From<crypto_cuda::error::CudaError> for GPUError {
    fn from(e: crypto_cuda::error::CudaError) -> GPUError {
        GPUError::CudaWrappedError(e)
    }
}
