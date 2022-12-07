#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::error::Error;
use std::ffi::CStr;
use std::fmt::{Debug, Display, Formatter};

include!("bindings.rs");

impl Display for CudaError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let name = unsafe { CStr::from_ptr(cudaGetErrorName(*self)) };
        name.fmt(f)
    }
}

impl Error for CudaError {}
