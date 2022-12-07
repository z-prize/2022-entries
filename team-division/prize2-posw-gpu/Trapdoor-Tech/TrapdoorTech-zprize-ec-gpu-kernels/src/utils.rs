use error::{GPUError, GPUResult};
use crypto_cuda::{cuda_get_device_count, cuda_init, flags::*};

use crate::*;

use log::info;

pub const GPU_NVIDIA_PLATFORM_NAME: &str = "NVIDIA CUDA";

pub fn get_opencl_devices(platform_name: &str) -> GPUResult<Vec<OpenclDevice>> {
    let platform = OpenclPlatform::list().into_iter().find(|&p| match p.name() {
        Ok(p) => p == platform_name,
        Err(_) => false,
    });

    let all_gpu_devices = match platform {
        Some(p) => OpenclDevice::list_all(p)?,
        None => {
            return Err(GPUError::Simple("GPU platform not found!"));
        },
    };

    let real_gpu_num = match std::env::var("GPU_NUM") {
        Ok(gpu_num_str) => {
            let gpu_num = gpu_num_str.parse::<usize>().unwrap();
            std::cmp::min(gpu_num, all_gpu_devices.len())
        },
        Err(_) => {
            all_gpu_devices.len()
        }
    };

    Ok(all_gpu_devices[0..real_gpu_num].to_vec())
}

pub fn get_cuda_devices() -> GPUResult<Vec<CudaDevice>> {
    cuda_init()?;

    let device_count = cuda_get_device_count()?;

    info!("total cuda device = {}", device_count);

    let mut devs = Vec::new();

    for i in 0..device_count {
        let dev = CudaDevice::new(i)?;
        devs.push(dev);
    }

    Ok(devs)
}
