// device management
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html

use crate::memory_pools::CudaMemPool;
use crate::result::{CudaResult, CudaResultWrap};
use bellman_cuda_cudart_sys::*;
use std::mem::MaybeUninit;

pub fn device_get_attribute(attr: CudaDeviceAttr, device_id: i32) -> CudaResult<i32> {
    let mut value = MaybeUninit::<i32>::uninit();
    unsafe { cudaDeviceGetAttribute(value.as_mut_ptr(), attr, device_id).wrap_maybe_uninit(value) }
}

pub fn device_reset() -> CudaResult<()> {
    unsafe { cudaDeviceReset().wrap() }
}

pub fn device_synchronize() -> CudaResult<()> {
    unsafe { cudaDeviceSynchronize().wrap() }
}

pub fn get_device_count() -> CudaResult<i32> {
    let mut count = MaybeUninit::<i32>::uninit();
    unsafe { cudaGetDeviceCount(count.as_mut_ptr()).wrap_maybe_uninit(count) }
}

pub fn get_device() -> CudaResult<i32> {
    let mut device_id = MaybeUninit::<i32>::uninit();
    unsafe { cudaGetDevice(device_id.as_mut_ptr()).wrap_maybe_uninit(device_id) }
}

pub fn get_device_properties(device_id: i32) -> CudaResult<CudaDeviceProperties> {
    let mut props = MaybeUninit::<CudaDeviceProperties>::uninit();
    unsafe { cudaGetDeviceProperties(props.as_mut_ptr(), device_id).wrap_maybe_uninit(props) }
}

pub fn set_device(device_id: i32) -> CudaResult<()> {
    unsafe { cudaSetDevice(device_id).wrap() }
}

pub fn device_get_default_mem_pool(device_id: i32) -> CudaResult<CudaMemPool> {
    let mut handle = MaybeUninit::<cudaMemPool_t>::uninit();
    unsafe {
        cudaDeviceGetDefaultMemPool(handle.as_mut_ptr(), device_id)
            .wrap_maybe_uninit(handle)
            .map(CudaMemPool::from_handle)
    }
}

pub fn device_get_mem_pool(device_id: i32) -> CudaResult<CudaMemPool> {
    let mut handle = MaybeUninit::<cudaMemPool_t>::uninit();
    unsafe {
        cudaDeviceGetMemPool(handle.as_mut_ptr(), device_id)
            .wrap_maybe_uninit(handle)
            .map(CudaMemPool::from_handle)
    }
}

pub fn device_set_mem_pool(device_id: i32, pool: &CudaMemPool) -> CudaResult<()> {
    unsafe { cudaDeviceSetMemPool(device_id, pool.into()).wrap() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::ffi::CStr;

    #[test]
    #[serial]
    fn device_get_attribute_is_ok() {
        let result = device_get_attribute(CudaDeviceAttr::MaxBlocksPerMultiprocessor, 0);
        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn device_get_attribute_max_blocks_per_multiprocessor_is_gt_zero() {
        let result = device_get_attribute(CudaDeviceAttr::MaxBlocksPerMultiprocessor, 0).unwrap();
        assert!(result > 0);
    }

    #[test]
    #[serial]
    fn device_reset_is_ok() {
        let result = device_reset();
        assert_eq!(result, Ok(()));
    }

    #[test]
    #[serial]
    fn device_synchronize_is_ok() {
        let result = device_synchronize();
        assert_eq!(result, Ok(()));
    }

    #[test]
    #[serial]
    fn get_device_count_is_not_zero() {
        let count = get_device_count().unwrap();
        assert_ne!(count, 0);
    }

    #[test]
    #[serial]
    fn device_id_is_smaller_than_device_count() {
        let device_id = get_device().unwrap();
        let count = get_device_count().unwrap();
        assert!(device_id < count);
    }

    #[test]
    #[serial]
    fn device_properties_name_is_not_empty_for_all_devices() {
        let count = get_device_count().unwrap();
        for i in 0..count {
            let props = get_device_properties(i).unwrap();
            let name = unsafe { CStr::from_ptr(props.name.as_ptr()) }
                .to_str()
                .unwrap();
            assert!(!name.is_empty());
        }
    }

    #[test]
    #[serial]
    fn set_device_works_for_all_devices() {
        let count = get_device_count().unwrap();
        let original_device_id = get_device().unwrap();
        for i in 0..count {
            set_device(i).unwrap();
            let current_device_id = get_device().unwrap();
            assert_eq!(i, current_device_id);
        }
        set_device(original_device_id).unwrap();
    }

    #[test]
    #[serial]
    fn device_get_default_mem_pool_is_ok_for_all_devices() {
        let count = get_device_count().unwrap();
        for i in 0..count {
            let result = device_get_default_mem_pool(i);
            assert!(result.is_ok());
        }
    }

    #[test]
    #[serial]
    fn device_get_mem_pool_is_ok_for_all_devices() {
        let count = get_device_count().unwrap();
        for i in 0..count {
            let result = device_get_mem_pool(i);
            assert!(result.is_ok());
        }
    }

    #[test]
    #[serial]
    fn device_set_mem_pool_is_ok_for_all_devices() {
        let count = get_device_count().unwrap();
        for i in 0..count {
            let pool = device_get_mem_pool(i).unwrap();
            let result = device_set_mem_pool(i, &pool);
            assert!(result.is_ok());
        }
    }
}
