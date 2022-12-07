// Copyright (C) 2019-2022 Aleo Systems Inc.
// This file is part of the snarkVM library.

// The snarkVM library is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// The snarkVM library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with the snarkVM library. If not, see <https://www.gnu.org/licenses/>.

use core::str::FromStr;
use std::{sync::{Arc, Mutex}, any::TypeId, env};
use std::sync::atomic::{Ordering, AtomicUsize};
use std::cmp::min;
use std::os::raw::c_void;

use rust_gpu_tools::GPUError;
use snarkvm_curves::{AffineCurve, bls12_377::{Fr, G1Affine}};
use snarkvm_fields::{Zero, PrimeField, FftField};

use blst::*;

#[allow(unused_imports)]
use chrono::Local;
sppark::cuda_error!();

const BELLMAN_WORKING_GPUS: &str = "BELLMAN_WORKING_GPUS";

#[repr(C)]
pub struct MultiScalarMultContext {
    context: *mut c_void
}

unsafe impl Send for MultiScalarMultContext {}

unsafe impl Sync for MultiScalarMultContext {}

impl Clone for MultiScalarMultContext {
    fn clone(&self) -> Self {
        Self { context: self.context.clone()}
    }
}

#[cfg_attr(feature = "quiet", allow(improper_ctypes))]
extern "C" {
    fn mult_pippenger_init(
        context: *mut MultiScalarMultContext,
        npoints: usize,
        device_index: usize,
        ffi_affine_sz: usize, 
        acc_level: i32
    ) -> cuda::Error;

    fn mult_pippenger_inf(
        context: *const MultiScalarMultContext,
        out: *mut u64,
        points_with_infinity: *const G1Affine,
        npoints: usize,
        batch_size: usize,
        scalars: *const Fr,
        ffi_affine_sz: usize,
    ) -> cuda::Error;

    fn mult_io_helper(
        context: *const MultiScalarMultContext,
        inout: *mut Fr,
        roots: *const Fr,
        npoints: usize
    ) -> cuda::Error;

    fn mult_oi_helper(
        context: *const MultiScalarMultContext,
        inout: *mut Fr,
        roots: *const Fr,
        npoints: usize
    ) -> cuda::Error;
}

pub fn multi_scalar_mult_init_withindex(
    npoints: usize,
    device_index: usize,
) -> MultiScalarMultContext {
    let a0 = Local::now().timestamp_millis();
    let mut ret = MultiScalarMultContext {
        context: std::ptr::null_mut()
    };

    let acc_level = *MAX_ACC_LEVEL;

    if std::env::var(BELLMAN_WORKING_GPUS).is_ok() {
        let err = unsafe {
            mult_pippenger_init(
                &mut ret,
                npoints,
                device_index,
                std::mem::size_of::<G1Affine>(),
                acc_level
            )
        };
        if err.code != 0 {
            eprintln!("{}", String::from(err));
        }
    }
    if std::env::var("ENABLE_INIT_LOG").is_ok() {
        eprintln!(
            "{}  blst-msm init: {}, {} use {} acc_level {}",
            Local::now().to_rfc3339(), npoints, device_index, Local::now().timestamp_millis() - a0, acc_level);
    }
    ret
}

pub fn multi_scalar_mult<G: AffineCurve, F: PrimeField>(
    context: &MultiScalarMultContext,
    bases: &[G],
    scalars: &[F],
) -> Vec<G::Projective> {
    let a0 = Local::now().timestamp_millis();

    let npoints = min(bases.len(),scalars.len());
    let batch_size = 1;//scalars.len() / npoints;

    let mut ret = vec![G::Projective::zero(); batch_size];

    let err = unsafe {
        let result_ptr =
            &mut *(&mut ret as *mut Vec<G::Projective>
                as *mut Vec<u64>);
            mult_pippenger_inf(
                context,
                result_ptr.as_mut_ptr(),
                bases as *const _ as *const G1Affine,
                npoints,
                batch_size,
                scalars as *const _ as *const Fr,
                std::mem::size_of::<G1Affine>())

    };

    if std::env::var("ENABLE_MSM_LOG").is_ok() {
        eprintln!(
            "{}  blst-msm: {}, {} use  {}",
            Local::now().to_rfc3339(), npoints, batch_size, Local::now().timestamp_millis() - a0);
    }

    if err.code != 0 {
        panic!("{}", String::from(err));
    }

    ret
}

use lazy_static::lazy_static;

lazy_static! {
    // The owned CUDA contexts are stored globally. Each devives contains an unowned reference,
    // so that devices can be cloned.
    static ref INSTANCE: Vec<Arc<Mutex<MultiScalarMultContext>>> = {
        let size = max_available_context();
        let devices_list = match std::env::var(BELLMAN_WORKING_GPUS) {
            Ok(v) => {
                let indexes: Vec<_> = v.split(',').map(|x| x.parse::<usize>().unwrap()).collect();
                indexes
            },
            Err(_e) => {
                (0..1).collect()
            },
        };

        let mut pool = vec![];
        for _ in 0..size {
            for j in &devices_list {
                pool.push(Arc::new(Mutex::new(getmsm_context(65536, *j))));
            }
        }
        pool
    };
    static ref CONTEXT_INDEX: Vec<Arc<AtomicUsize>> = {
        let size = max_available_context();
        let devices_list = match std::env::var(BELLMAN_WORKING_GPUS) {
            Ok(v) => {
                let indexes: Vec<_> = v.split(',').map(|x| x.parse::<usize>().unwrap()).collect();
                indexes
            },
            Err(_e) => {
                (0..1).collect()
            },
        };
        (0..devices_list.len()*size).map(|_| Arc::new(AtomicUsize::new(1))).collect::<Vec<_>>()
    };

     static ref MAX_ACC_LEVEL: i32 = {
          let mut acc_level = 3;
          if let Ok(t) = std::env::var("CLEVEL") {
              let m = t.parse::<i32>().unwrap();
              acc_level = if m <= 8 { m } else { 8 };
          }

        //   println!("LEVEL {}", acc_level);
          acc_level
    };
}

pub fn max_available_context() -> usize {
    match env::var("MAX_CUDA_CONTEXT")
        .ok()
        .and_then(|s| usize::from_str(&s).ok()) {
        Some(x) if x > 0 =>  x,
        _ => {8}
    }
}

fn fetch_gpu() -> usize {
    let mut min_i = 0usize;
    let len = INSTANCE.len();
    let mut min_q = 100000usize;

    for i in (0..len).rev() {
        let callers = CONTEXT_INDEX[i].load(Ordering::SeqCst);
        if  callers <= min_q {
            min_q = callers;
            min_i = i;
        }
    }
    CONTEXT_INDEX[min_i].fetch_add(1, Ordering::SeqCst);
    min_i
}

fn release_gpu(index: usize) {
    CONTEXT_INDEX[index].fetch_sub(1, Ordering::SeqCst);
}

pub fn getmsm_context(
    npoints: usize,
    device: usize,
) -> MultiScalarMultContext {
    let context = multi_scalar_mult_init_withindex(npoints, device);
    return context;
}

#[allow(clippy::transmute_undefined_repr)]
pub(super) fn msm_cuda<G: AffineCurve, F: PrimeField>(
    bases: &[G],
    scalars: &[F],
) -> Result<G::Projective, GPUError> {
    if TypeId::of::<G>() != TypeId::of::<G1Affine>() {
        unimplemented!("trying to use cuda for unsupported curve");
    }

    let index = fetch_gpu();
    let context = INSTANCE[index].lock().unwrap();
    let result = multi_scalar_mult(&*context, &bases, &scalars);
    release_gpu(index);
    Ok(result[0])
}

#[allow(clippy::transmute_undefined_repr)]
pub(super) fn msm_io_helper<F: FftField, T: crate::fft::DomainCoeff<F>>(bases: &mut [T], root: F) {
    let index = fetch_gpu();
    let context = INSTANCE[index].lock().unwrap();
    unsafe { mult_io_helper(&*context, 
        bases.as_mut_ptr() as *mut Fr, 
        [root].as_ptr() as *const Fr, bases.len())};
    release_gpu(index);
}

#[allow(clippy::transmute_undefined_repr)]
pub(super) fn msm_oi_helper<F: FftField, T: crate::fft::DomainCoeff<F>>(bases: &mut [T], root: F) {
    let index = fetch_gpu();
    let context = INSTANCE[index].lock().unwrap();
    unsafe { mult_oi_helper(&*context, 
        bases.as_mut_ptr() as *mut Fr, 
        [root].as_ptr() as *const Fr, bases.len())};
    release_gpu(index);
}