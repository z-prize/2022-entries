/// Define the parameters that a kernel needs

/// For polynomial kernel
pub const MAX_POLY_LEN_LOG: usize = 17; // 2^17 poly
pub const MAX_POLY_LEN: usize = 1 << MAX_POLY_LEN_LOG; // 2^17 poly

pub const MAX_POLY_REDUCTION_BATCH: usize = 1;
pub const MAX_BUCKET_SIZE: usize = 6;
pub const MAX_BUCKET_NUM: usize = 1 << MAX_BUCKET_SIZE;

pub const LOG2_MAX_ELEMENTS: usize = 32; // At most 2^32 elements is supported.
pub const MAX_RADIX_DEGREE: u32 = 8; // Radix256
pub const MAX_LOCAL_WORK_SIZE_DEGREE: u32 = 7; // 128

/// For Cuda Performance
pub const CUDA_BEST_BATCH: usize = 2;

/// For multiexp
pub const MAX_WINDOW_SIZE: usize = 10; // it's enough for snarkVM
pub const MIN_WINDOW_SIZE: usize = 4;
pub const MSM_MAX_BUCKET_SIZE: usize = 4;
pub const LOCAL_WORK_SIZE: usize = 32;

/// for cuda
pub const CUDA_BLOCKS: u32 = 256;
pub const CUDA_LWS: u32 = 32;
pub const CORE_N: usize = 6;

/// only for SnarkVM
pub const MAX_SRS_G_LEN: usize = (1 << 16) + 5;
