use crate::api::CUfunction;

#[derive(Clone)]
pub struct CudaFunction {
    pub inner: CUfunction,
}

unsafe impl Send for CudaFunction {}
unsafe impl Sync for CudaFunction {}
