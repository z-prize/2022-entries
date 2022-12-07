use std::ffi::CStr;
use std::os::raw::c_char;

use crate::api::{cuDeviceGet, cuDeviceGetName, cuDeviceTotalMem_v2, cuDeviceGetAttribute, CUdevice, CUresult, size_t};
use crate::error::CudaResult;
use crate::flags::DeviceAttribute;

use std::convert::TryInto;

pub const DEVICE_NAME_LENGTH: usize = 32;

#[derive(Debug, Clone, Copy)]
pub struct CudaDevice {
    inner: CUdevice,
}

impl From<i32> for CudaDevice {
    fn from(dev: i32) -> Self {
        Self {
            inner: dev as CUdevice,
        }
    }
}

impl Into<i32> for CudaDevice {
    fn into(self) -> i32 {
        self.inner as i32
    }
}

impl CudaDevice {
    pub fn new(dev: usize) -> CudaResult<Self> {
        let mut device = 0 as CUdevice;
        let result: CUresult = unsafe { cuDeviceGet(&mut device, dev as i32) };

        if result == 0 {
            Ok(CudaDevice { inner: device })
        } else {
            Err(result.into())
        }
    }

    pub fn get_attr(&self, attr: DeviceAttribute) -> CudaResult<i32> {
        let mut value = 0i32;

        let result = unsafe {
            cuDeviceGetAttribute(
                &mut value as *mut i32,
                attr as u32,
                self.inner,
            )
        };

        if result == 0 {
            Ok(value)
        } else {
            Err(result.into())
        }
    }

 
    pub fn get_cores(&self) -> CudaResult<usize> {
        let mut sms = 0i32;

        let result = unsafe {
            cuDeviceGetAttribute(
                &mut sms as *mut i32,
                DeviceAttribute::MultiprocessorCount as u32,
                self.inner,
            )
        };

        // TODO: should find a way to calculate cuda cores
        let cores: usize = (sms * 64).try_into().unwrap();

        if result == 0 {
            Ok(cores)
        } else {
            Err(result.into())
        }
    }

    pub fn get_memory(&self) -> CudaResult<u64> {
        let mut total_mem = 0u64;

        let result = unsafe {
            cuDeviceTotalMem_v2(
                &mut total_mem as *mut size_t,
                self.inner,
            )
        };

        if result == 0 {
            Ok(total_mem)
        } else {
            Err(result.into())
        }
    }

    pub fn get_name(&self) -> CudaResult<String> {
        let mut cstr_holder = [0i8; DEVICE_NAME_LENGTH];
        let result = unsafe {
            cuDeviceGetName(
                cstr_holder.as_mut_ptr() as *mut c_char,
                DEVICE_NAME_LENGTH as i32,
                0,
            )
        };

        if result == 0 {
            let cstr = unsafe {
                CStr::from_ptr(cstr_holder.as_ptr())
                    .to_str()
                    .unwrap()
                    .to_string()
            };
            Ok(cstr)
        } else {
            Err(result.into())
        }
    }

    pub fn get_inner(&self) -> CUdevice {
        self.inner
    }
}
