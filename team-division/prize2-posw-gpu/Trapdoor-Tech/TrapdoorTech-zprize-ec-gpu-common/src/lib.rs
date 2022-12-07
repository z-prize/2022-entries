#[macro_use]
extern crate crypto_cuda;

#[cfg(test)]
#[macro_use]
extern crate serial_test;

mod kernel;
pub use kernel::*;

mod utils;
pub use utils::*;

mod structs;
pub use structs::*;

mod errors;
pub use errors::*;

mod params;
pub use params::*;

mod cuda;
pub use cuda::*;

use crypto_cuda::{CudaContext, CudaDevice, CudaModule, CudaParams, CudaFunction};

pub type Device = CudaDevice;
pub type Context = CudaContext;
pub type Module = CudaModule;
pub type Stream = CudaStream;
pub type Params = CudaParams;
pub type Function = CudaFunction;

lazy_static::lazy_static! {
    pub static ref GPU_CUDA_DEVICES: Vec<Device> = get_cuda_devices().unwrap();
    pub static ref GPU_CUDA_CORES: Vec<GPUSourceCore> = get_cuda_cores().unwrap();
}

#[inline(always)]
pub fn log2_floor(num: usize) -> u32 {
    assert!(num > 0);

    let mut pow = 0;

    while (1 << (pow + 1)) <= num {
        pow += 1;
    }

    pow
}

#[inline(always)]
pub fn bitreverse(n: usize, l: usize) -> usize {
    let mut r = n.reverse_bits();
    // now we need to only use the bits that originally were "last" l, so shift

    r >>= (std::mem::size_of::<usize>() * 8) - l;

    r
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn it_works() {
        let _ = env_logger::try_init();

        GPUSourceCore::create_cuda(0).unwrap();
    }
}
