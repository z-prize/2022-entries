#[macro_export]
macro_rules! create_kernel {
    ($func: tt, <<<$grid: expr, $block: expr, $shm: tt>>>($($params: expr),*)) => {
        crate::kernel::CudaKernel::new($func, $grid, 1, 1, $block, 1, 1, $shm, &[$(&$params as *const _ as *mut std::ffi::c_void,)*], &[std::ptr::null_mut()])?
    };
}

#[macro_export]
macro_rules! make_params {
    ($($params: expr),*) => {
        &[$(&$params as *const _ as *mut std::ffi::c_void,)*]
    };
}

#[macro_export]
macro_rules! create_kernel_with_params {
    ($func: tt, <<<$grid: expr, $block: expr, $shm: tt>>>($params: tt)) => {
        crate::kernel::CudaKernel::new(
            $func,
            $grid,
            1,
            1,
            $block,
            1,
            1,
            $shm,
            $params,
            &[std::ptr::null_mut()],
        )?
    };
}

#[macro_export]
macro_rules! xgrid {
    ($x:tt) => {
        crate::kernel::CudaGrid {
            x: $x,
            y: 1u32,
            z: 1u32,
        }
    };
}

#[macro_export]
macro_rules! xblock {
    ($x:tt) => {
        crate::kernel::CudaBlock {
            x: $x,
            y: 1u32,
            z: 1u32,
        }
    };
}
