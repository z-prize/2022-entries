// execution control
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html

use crate::result::{CudaResult, CudaResultWrap};
use crate::stream::CudaStream;
use bellman_cuda_cudart_sys::*;
use core::ffi::c_void;
use std::sync::{Arc, Weak};

pub struct HostFn<'a> {
    arc: Arc<Box<dyn Fn() + Send + 'a>>,
}

impl<'a> HostFn<'a> {
    pub fn new(func: impl Fn() + Send + 'a) -> Self {
        Self {
            arc: Arc::new(Box::new(func) as Box<dyn Fn() + Send>),
        }
    }
}

unsafe extern "C" fn launch_host_fn_callback(data: *mut c_void) {
    let raw = data as *const Box<dyn Fn() + Send>;
    let weak = Weak::from_raw(raw);
    if let Some(func) = weak.upgrade() {
        func();
    }
}

pub fn get_raw_fn_and_data(host_fn: &HostFn) -> (cudaHostFn_t, *mut c_void) {
    let weak = Arc::downgrade(&host_fn.arc);
    let raw = weak.into_raw();
    let data = raw as *mut c_void;
    (Some(launch_host_fn_callback), data)
}

pub fn launch_host_fn(stream: &CudaStream, host_fn: &HostFn) -> CudaResult<()> {
    let (func, data) = get_raw_fn_and_data(host_fn);
    unsafe { cudaLaunchHostFunc(stream.into(), func, data).wrap() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::sync::Mutex;
    use std::thread;
    use std::time::Duration;

    #[test]
    #[serial]
    fn host_fn_add_executes_one_time() {
        let stream = CudaStream::create().unwrap();
        let mut a = 0;
        let add = || {
            a += 1;
            thread::sleep(Duration::from_millis(10));
        };
        let add_mutex = Mutex::new(add);
        let add_fn = HostFn::new(move || add_mutex.lock().unwrap()());
        let sleep_fn = HostFn::new(|| thread::sleep(Duration::from_millis(10)));
        launch_host_fn(&stream, &add_fn).unwrap();
        stream.synchronize().unwrap();
        launch_host_fn(&stream, &sleep_fn).unwrap();
        launch_host_fn(&stream, &add_fn).unwrap();
        drop(add_fn);
        stream.synchronize().unwrap();
        assert_eq!(a, 1);
    }
}
