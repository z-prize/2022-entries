use crate::api::{CUfunction, CUDA_KERNEL_NODE_PARAMS};
use crate::error::CudaResult;

pub struct CudaKernel {
    pub kernel: CUDA_KERNEL_NODE_PARAMS,
}

#[derive(Clone, Debug, Copy)]
pub struct CudaGrid {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[derive(Clone, Debug, Copy)]
pub struct CudaBlock {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl CudaKernel {
    pub fn new(
        func: CUfunction,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
        shm: u32,
        params: &[*mut std::ffi::c_void],
        extra: &[*mut std::ffi::c_void],
    ) -> CudaResult<Self> {
        let kernel = CUDA_KERNEL_NODE_PARAMS {
            func,

            gridDimX: grid_x,
            gridDimY: grid_y,
            gridDimZ: grid_z,

            blockDimX: block_x,
            blockDimY: block_y,
            blockDimZ: block_z,

            sharedMemBytes: shm,

            kernelParams: params as *const _ as *mut *mut std::os::raw::c_void,
            extra: extra as *const _ as *mut *mut std::os::raw::c_void,
        };

        Ok(Self { kernel })
    }
}
