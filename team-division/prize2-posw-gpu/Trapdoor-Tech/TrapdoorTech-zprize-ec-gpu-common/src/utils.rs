use crate::params::CUDA_LWS;
use crate::{Device, GPUResult, GPUSourceCore};
use crypto_cuda::{cuda_get_device_count, cuda_init, flags::*};
use log::info;

pub fn get_cuda_devices() -> GPUResult<Vec<Device>> {
    cuda_init()?;

    let device_count = cuda_get_device_count()?;

    info!("total cuda device = {}", device_count);

    let mut devs = Vec::new();

    for i in 0..device_count {
        let dev = Device::new(i)?;
        devs.push(dev);
    }

    Ok(devs)
}

#[inline]
pub fn calc_cuda_wg_threads(n: usize) -> (u32, u32) {
    let lws = CUDA_LWS as u32;
    let gws = n as u32 / lws + (n as u32 % lws != 0) as u32;
    (gws, lws)
}

pub fn get_cuda_cores() -> GPUResult<Vec<GPUSourceCore>> {
    get_cuda_devices()?;

    let device_count = cuda_get_device_count()?;

    let mut cores = Vec::new();

    for i in 0..device_count {
        let core = GPUSourceCore::create_cuda(i)?;
        cores.push(core);
    }

    Ok(cores)
}

pub fn get_device_attribute(idx: usize, attr: DeviceAttribute) -> GPUResult<i32> {
    let devs = get_cuda_devices()?;

    Ok(devs[idx].get_attr(attr)?)
}

pub fn get_kernel_name() -> GPUResult<String> {
    let kernel_name = match std::env::var("KERNEL_NAME") {
        Ok(name) => name,
        Err(_) => "gpu-kernel.bin".to_string()
    };
    Ok(format!("./{}", kernel_name))
}
