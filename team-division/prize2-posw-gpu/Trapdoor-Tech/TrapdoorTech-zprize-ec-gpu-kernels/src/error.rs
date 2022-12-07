#[derive(thiserror::Error, Debug)]
pub enum GPUError {
    #[error("GPUError: {0}")]
    Simple(&'static str),
    #[error("Ocl Error: {0}")]
    Ocl(ocl::Error),
    #[error("Cuda Wrapped Error: {0}")]
    CudaWrappedError(crypto_cuda::error::CudaError),
}

pub type GPUResult<T> = std::result::Result<T, GPUError>;

impl From<ocl::Error> for GPUError {
    fn from(error: ocl::Error) -> Self {
        GPUError::Ocl(error)
    }
}

impl From<crypto_cuda::error::CudaError> for GPUError {
    fn from(e: crypto_cuda::error::CudaError) -> GPUError {
        GPUError::CudaWrappedError(e)
    }
}
