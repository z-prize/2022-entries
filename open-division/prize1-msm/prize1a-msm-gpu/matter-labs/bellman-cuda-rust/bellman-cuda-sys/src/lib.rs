#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use bellman_cuda_cudart_sys::*;
use core::ffi::c_void;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct msm_configuration {
    pub mem_pool: cudaMemPool_t,
    pub stream: cudaStream_t,
    pub bases: *const c_void,
    pub scalars: *const c_void,
    pub results: *mut c_void,
    pub log_scalars_count: u32,
    pub h2d_copy_finished: cudaEvent_t,
    pub h2d_copy_finished_callback: cudaHostFn_t,
    pub h2d_copy_finished_callback_data: *const c_void,
    pub d2h_copy_finished: cudaEvent_t,
    pub d2h_copy_finished_callback: cudaHostFn_t,
    pub d2h_copy_finished_callback_data: *const c_void,
    pub force_min_chunk_size: bool,
    pub log_min_chunk_size: u32,
    pub force_max_chunk_size: bool,
    pub log_max_chunk_size: u32,
    pub window_bits_count: u32,
    pub precomputed_windows_stride: u32,
    pub precomputed_bases_stride: u32,
    pub scalars_not_montgomery: bool,
}

extern "C" {
    pub fn msm_set_up() -> cudaError_t;

    pub fn msm_left_shift(
        values: *mut c_void,
        shift: u32,
        count: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn msm_execute_async(configuration: msm_configuration) -> cudaError_t;

    pub fn msm_tear_down() -> cudaError_t;
}
