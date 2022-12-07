// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use std::os::raw::c_void;
use ark_bls12_377::{Fr, G1Affine};
use ark_ec::AffineCurve;
use ark_ff::PrimeField;
use ark_std::Zero;

#[allow(unused_imports)]
use blst::*;

sppark::cuda_error!();

pub mod util;

#[repr(C)]
pub struct MultiScalarMultContext {
    context: *mut c_void,
}

#[cfg_attr(feature = "quiet", allow(improper_ctypes))]
extern "C" {
    fn mult_pippenger_init(
        context: *mut MultiScalarMultContext,
        points_with_infinity: *const G1Affine,
        npoints: usize,
        ffi_affine_sz: usize,
    ) -> cuda::Error;
    
    fn mult_pippenger_inf(
        context: *mut MultiScalarMultContext,
        out: *mut u64,
        points_with_infinity: *const G1Affine,
        npoints: usize,
        batch_size: usize,
        scalars: *const Fr,
        ffi_affine_sz: usize,
    ) -> cuda::Error;

    fn msm(out: *mut u64,
           scalars: *const Fr,
           scalar_num: usize,
           bits_window_length: *const i32,
           window_num: i32) -> cuda::Error;

    fn init(points_with_infinity: *const G1Affine,
            npoints: usize,
            max_window_size_in_bits: i32,
            ffi_affine_sz: usize) -> cuda::Error;
}

pub fn multi_scalar_mult_init<G: AffineCurve>(
    points: &[G],
) -> MultiScalarMultContext {
    let mut ret = MultiScalarMultContext {
        context: std::ptr::null_mut(),
    };
        
    let err = unsafe {
        mult_pippenger_init(
            &mut ret,
            points as *const _ as *const G1Affine,
            points.len(),
            std::mem::size_of::<G1Affine>(),
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }

    ret
}
    
pub fn multi_scalar_mult<G: AffineCurve>(
    context: &mut MultiScalarMultContext,
    points: &[G],
    scalars: &[<G::ScalarField as PrimeField>::BigInt],
) -> Vec<G::Projective> {
    let npoints = points.len();
    if scalars.len() % npoints != 0 {
        panic!("length mismatch")
    }

    //let mut context = multi_scalar_mult_init(points);

    let batch_size = scalars.len() / npoints;
    let mut ret = vec![G::Projective::zero(); batch_size];
    let err = unsafe {
        let result_ptr = 
            &mut *(&mut ret as *mut Vec<G::Projective>
                   as *mut Vec<u64>);

        mult_pippenger_inf(
            context,
            result_ptr.as_mut_ptr(),
            points as *const _ as *const G1Affine,
            npoints, batch_size,
            scalars as *const _ as *const Fr,
            std::mem::size_of::<G1Affine>(),
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }

    ret
}

pub fn msm_wrapper<G: AffineCurve>(
    scalars: &[<G::ScalarField as PrimeField>::BigInt],
    window_len: &[i32],
    npoints:usize
) -> Vec<G::Projective>{
    let nscalars = scalars.len();
    let batch_size = scalars.len() / npoints;
    let mut ret = vec![G::Projective::zero(); batch_size];
    let err = unsafe {
        let result_ptr =
            &mut *(&mut ret as *mut Vec<G::Projective>
                as *mut Vec<u64>);
        msm(
            result_ptr.as_mut_ptr(),
            scalars as *const _ as *const Fr,
            nscalars,
            window_len as *const _ as *const i32,
            window_len.len() as i32
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }

    ret
}

pub fn init_wrapper<G: AffineCurve>(
    points: &[G],
    max_window_size_in_bits: i32,
) -> () {
    let npoints = points.len();
    let err = unsafe {
        init(
            points as *const _ as *const G1Affine,
            npoints,
            max_window_size_in_bits,
            std::mem::size_of::<G1Affine>(),
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}