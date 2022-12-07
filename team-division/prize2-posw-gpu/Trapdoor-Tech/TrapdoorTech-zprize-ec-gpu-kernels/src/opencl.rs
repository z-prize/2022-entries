use super::structs::*;
use super::utils::*;
use super::sources::*;
use super::*;

use ocl::{builders::ProgramBuilder, Context, Device, Platform, ProQue, Program};

lazy_static::lazy_static! {
    pub static ref GPU_NVIDIA_OPENCL_DEVICES: Vec<Device> = get_opencl_devices(GPU_NVIDIA_PLATFORM_NAME).unwrap_or_default();
}
// call this function first to get a prebuilt kernel
pub fn write_kernel<F1: PrimeField, F2: Field>(kernel_name: &str) -> GPUResult<()> {
    use log::info;

    let devices = &GPU_NVIDIA_OPENCL_DEVICES;
    if devices.is_empty() {
        return Err(GPUError::Simple("No working GPUs found!"));
    }

    // just use the first nvidia device to build kernel
    let d = devices[0];

    use sources::Limb64 as Limb;
    let src = gen_all_source::<F1, F2, Limb>();

    let pq = ProQue::builder().device(d).src(src).dims(1).build()?;

    info!("building opencl kernel for {}", kernel_name);
    println!("writing kernel");
    let bins = match pq
        .program()
        .info(ocl::enums::ProgramInfo::Binaries)
        .unwrap()
    {
        ocl::enums::ProgramInfoResult::Binaries(bin_vec) => bin_vec[0].clone(),
        _ => panic!("proque doesn't contain a binary!"),
    };
    std::fs::write(kernel_name, bins).unwrap();

    Ok(())
}
