// Peer Device Memory Access
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html

use crate::result::{CudaResult, CudaResultWrap};
use bellman_cuda_cudart_sys::*;
use std::mem::MaybeUninit;

pub fn device_can_access_peer(device_id: i32, device_peer_id: i32) -> CudaResult<bool> {
    let mut can_access_peer = MaybeUninit::<i32>::uninit();
    unsafe {
        cudaDeviceCanAccessPeer(can_access_peer.as_mut_ptr(), device_id, device_peer_id)
            .wrap_maybe_uninit(can_access_peer)
            .map(|value| value != 0)
    }
}

pub fn device_disable_peer_access(device_peer_id: i32) -> CudaResult<()> {
    unsafe { cudaDeviceDisablePeerAccess(device_peer_id).wrap() }
}

pub fn device_enable_peer_access(device_peer_id: i32) -> CudaResult<()> {
    unsafe { cudaDeviceEnablePeerAccess(device_peer_id, 0).wrap() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::device_reset;
    use serial_test::serial;

    #[test]
    #[serial]
    #[ignore = "needs multiple peer-enabled GPUs"]
    fn device_can_access_peer_is_true() {
        let result = device_can_access_peer(0, 1);
        assert_eq!(result, Ok(true));
    }

    #[test]
    #[serial]
    #[ignore = "needs multiple peer-enabled GPUs"]
    fn device_disable_peer_access_is_ok() {
        device_reset().unwrap();
        device_enable_peer_access(1).unwrap();
        let result = device_disable_peer_access(1);
        assert_eq!(result, Ok(()));
    }

    #[test]
    #[serial]
    #[ignore = "needs multiple peer-enabled GPUs"]
    fn device_enable_peer_access_is_ok() {
        device_reset().unwrap();
        let result = device_enable_peer_access(1);
        assert_eq!(result, Ok(()));
        device_disable_peer_access(1).unwrap();
    }
}
