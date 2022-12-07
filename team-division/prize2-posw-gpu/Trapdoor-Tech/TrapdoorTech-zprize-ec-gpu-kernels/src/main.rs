mod opencl;
use opencl::*;

mod cuda;
use cuda::*;

mod error;
use error::*;

mod sources;
use sources::*;

mod structs;
use structs::*;

mod utils;
use utils::*;

mod tools;
use tools::*;

use clap::{Arg, App};

use crypto_cuda::{CudaDevice, CudaContext, CudaModule, CudaParams};
pub use ocl::{Device as OpenclDevice, Platform as OpenclPlatform};

fn generate_opencl_kernel() {
    println!("generating OpenCL kernel...");
    opencl::write_kernel::<Fr, Fq>("gpu-kernel.bin").unwrap();
}

fn generate_cuda_kernel() {
    println!("generating Cuda kernel...");
    cuda::write_kernel::<Fr, Fq>("gpu-kernel.bin").unwrap();
}

fn main() {
    let _ = env_logger::try_init();

    let m = App::new("Elliptic Curve GPU kernel generator")
        .arg(Arg::with_name("opencl").long("opencl"))
        .arg(Arg::with_name("cuda").long("cuda"))
        .get_matches();

    let kernel_type_opencl = m.is_present("opencl");
    let kernel_type_cuda = m.is_present("cuda");

    if kernel_type_opencl {
        generate_opencl_kernel();
    }

    if kernel_type_cuda {
        generate_cuda_kernel();
    }

    if !kernel_type_opencl && !kernel_type_cuda {
        println!("Error: must specify kernel type!");
    }
}
