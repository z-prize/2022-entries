use super::structs::*;
use super::utils::*;
use super::sources::*;
use super::*;

use crypto_cuda::{CudaContext, CudaDevice, CudaModule, CudaParams};

lazy_static::lazy_static! {
    pub static ref GPU_NVIDIA_CUDA_DEVICES: Vec<CudaDevice> = get_cuda_devices().unwrap_or_default();
}

// call this function first to get a prebuilt kernel
pub fn write_kernel<F1: PrimeField, F2: Field>(kernel_name: &str) -> GPUResult<()> {
    use log::info;
    use std::ffi::CString;
    use std::fs;
    use std::process::Command;

    // cuda source can only use 32bit limbs
    use sources::Limb32 as Limb;

    let tmpdir = tempfile::tempdir().expect("Cannot create temporary directory.");
    let source_path = tmpdir.path().join("kernel.cu");
    let src = gen_all_source::<F1, F2, Limb>();

    fs::write(&source_path, src.as_bytes()).expect("Cannot write kernel source file.");

    let mut cmd = Command::new("nvcc");
    cmd.arg("--optimize=6")
       .arg("--default-stream=per-thread")
       .arg("--fatbin")
       .arg("-arch")
       .arg("compute_86")
       .arg("-code")
       .arg("sm_86")
       .arg("--output-file")
       .arg(kernel_name)
       .arg(&source_path);
    #[cfg(feature = "cuda")]
    cmd.arg("-DCUDA");
    #[cfg(feature = "sn_cuda")]
    cmd.arg("-DSN_CUDA");

    println!("compile cmd is {:?}", cmd);

    let nvcc = cmd.status().expect("Cannot run nvcc");

    println!("writing kernel");

    if !nvcc.success() {
        panic!(
            "nvcc failed. See the kernel source at {}.",
            source_path.display()
        );
    }

    Ok(())
}
