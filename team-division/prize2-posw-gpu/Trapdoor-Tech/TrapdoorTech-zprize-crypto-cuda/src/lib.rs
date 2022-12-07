#[macro_use]
extern crate bitflags;

pub mod api;
pub mod context;
pub mod device;
pub mod error;
pub mod flags;
pub mod host_memory;
pub mod kernel;
pub mod macros;
pub mod memory;
pub mod module;
pub mod stream;
pub mod function;

use api::{cuDeviceCanAccessPeer, cuDeviceGetCount};

pub use context::CudaContext;
pub use memory::{DeviceMemory};
pub use host_memory::HostMemory;
pub use module::CudaModule;
pub use stream::CudaStream;
pub use device::CudaDevice;
pub use error::{CudaError, CudaResult};
pub use api::CUdeviceptr;
pub use function::CudaFunction;

pub type CudaParams = [*mut std::ffi::c_void];

pub fn cuda_init() -> CudaResult<()> {
    let result = unsafe {
        // as stated in cuda docs, the flags used in cuInit() must be 0
        crate::api::cuInit(0)
    };

    if result == 0 {
        Ok(())
    } else {
        Err(result.into())
    }
}

pub fn cuda_get_device_count() -> CudaResult<usize> {
    let mut count = 0 as std::os::raw::c_int;

    let result = unsafe { cuDeviceGetCount(&mut count as *mut std::os::raw::c_int) };

    if result == 0 {
        Ok(count as usize)
    } else {
        Err(result.into())
    }
}

pub fn cuda_can_access_peer(src: &CudaDevice, dst: &CudaDevice) -> CudaResult<bool> {
    let mut can_access = 1;

    let result = unsafe {
        cuDeviceCanAccessPeer(
            &mut can_access as *mut std::os::raw::c_int,
            src.get_inner(),
            dst.get_inner(),
        )
    };

    if result == 0 {
        match can_access {
            1 => Ok(true),
            _ => Ok(false),
        }
    } else {
        Err(result.into())
    }
}

#[cfg(test)]
mod tests {
    use crate::context::CudaContext;
    use crate::device::CudaDevice;
    use crate::error::CudaResult;
    use crate::host_memory::HostMemory;
    use crate::memory::{cuda_link_transfer, DeviceMemory};
    use crate::module::CudaModule;
    use crate::stream::CudaStream;
    use crate::macros::*;

    use crate::create_kernel;
    use crate::make_params;
    use crate::create_kernel_with_params;

    const MAX_LENGTH: usize = 1 << 26;

    use crate::{cuda_get_device_count, cuda_init};

    use log::info;

    #[test]
    fn test_kernel() -> CudaResult<()> {
        let _ = env_logger::try_init();

        cuda_init()?;

        let n = cuda_get_device_count()?;

        info!("device total = {}", n);

        let mut devs = Vec::new();
        let mut ctxs = Vec::new();

        for i in 0..n {
            let dev = CudaDevice::new(i)?;
            info!("device name = {}", dev.get_name()?);
            devs.push(dev);
            ctxs.push(CudaContext::new(dev)?);
        }

        let buffer_a = DeviceMemory::<f32>::new(&ctxs[0], MAX_LENGTH)?;
        let buffer_b = DeviceMemory::<f32>::new(&ctxs[0], MAX_LENGTH)?;
        let buffer_out = DeviceMemory::<f32>::new(&ctxs[0], MAX_LENGTH)?;

        let mut buffer_host_a = HostMemory::<f32>::new(MAX_LENGTH)?;
        let mut buffer_host_b = HostMemory::<f32>::new(MAX_LENGTH)?;
        let mut buffer_host_out = HostMemory::<f32>::new(MAX_LENGTH)?;

        buffer_host_a
            .as_mut_slice()
            .iter_mut()
            .for_each(|r| *r = 5.6f32);

        buffer_host_b
            .as_mut_slice()
            .iter_mut()
            .for_each(|r| *r = 9.2f32);

        buffer_a.read_from(&buffer_host_a, MAX_LENGTH)?;
        buffer_b.read_from(&buffer_host_b, MAX_LENGTH)?;

        let module = CudaModule::new("add.ptx")?;

        let f = module.get_func("sum")?;

        /* add.cu

        extern "C" __constant__ int my_constant = 314;

        extern "C" __global__ void sum(const float* x, const float* y, float* out, int count) {
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
                out[i] = x[i] + y[i];
            }
        }

        */

        let stream = CudaStream::new_with_context(&ctxs[0])?;

        let params = make_params!(buffer_a.get_inner(), buffer_b.get_inner(), buffer_out.get_inner(), MAX_LENGTH as u32);
        let kernel = create_kernel_with_params!(f, <<<10, 10, 0>>>(params));

        stream.launch(&kernel)?;
        stream.sync()?;

        buffer_out.write_to(&mut buffer_host_out, MAX_LENGTH)?;

        info!(
            "after kernel launch, out[10000] = {}",
            buffer_host_out.as_ref()[10000]
        );

        Ok(())
    }
}
