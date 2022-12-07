use std::ffi::CString;
use std::ptr;

use crate::api::{cuModuleGetFunction, cuModuleLoad, CUfunction, CUmodule};
use crate::error::CudaResult;

#[derive(Clone)]
pub struct CudaModule {
    inner: CUmodule,
}

unsafe impl Send for CudaModule {}
unsafe impl Sync for CudaModule {}

impl CudaModule {
    pub fn new(path: &str) -> CudaResult<Self> {
        let module_cstring =
            CString::new(path).expect(format!("error loading file {}", path).as_str());

        let mut module = 0 as CUmodule;

        let result = unsafe {
            cuModuleLoad(
                &mut module as *mut CUmodule,
                module_cstring.as_c_str().as_ptr(),
            )
        };

        if result == 0 {
            Ok(CudaModule { inner: module })
        } else {
            Err(result.into())
        }
    }

    pub fn get_func(&self, func: &str) -> CudaResult<CUfunction> {
        let mut func_ptr = ptr::null_mut();

        let func_cstring =
            CString::new(func).expect(format!("error finding func {} in module", func).as_str());

        let result = unsafe {
            cuModuleGetFunction(
                &mut func_ptr as *mut CUfunction,
                self.inner,
                func_cstring.as_c_str().as_ptr(),
            )
        };

        if result == 0 {
            Ok(func_ptr)
        } else {
            Err(result.into())
        }
    }
}
