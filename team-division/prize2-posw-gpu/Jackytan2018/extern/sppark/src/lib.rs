// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use std::os::raw::c_void;
use ark_bls12_377::{Fr, G1Affine};
use ark_ec::AffineCurve;
use ark_ff::PrimeField;
use ark_std::Zero;
use std::time::Instant;

use snarkvm_curves::AffineCurve as ArsAffineCurve;
use snarkvm_fields::PrimeField as ArsPrimeField;
use snarkvm_fields::Zero as ArsZero;
use snarkvm_curves::bls12_377::G1Projective as ArsG1Projective;

#[allow(unused_imports)]
use blst::*;

sppark::cuda_error!();

#[repr(C)]
#[derive(Copy, Clone)]
pub struct MultiScalarMultContext {
    pub device_id: usize,
    pub gpu_id: usize,
    pub obj_id: usize,
    pub bit_len: usize,
    pub is_used: bool,
    context: *mut c_void,
}
unsafe impl Send for MultiScalarMultContext {}

#[cfg_attr(feature = "quiet", allow(improper_ctypes))]
extern "C" {
    fn mult_pippenger_init(
        context: *mut MultiScalarMultContext,
        dev_id: usize,
        points_with_infinity: *const G1Affine,
        npoints: usize,
        ffi_affine_sz: usize,
    ) -> cuda::Error;
    
    fn mult_pippenger_inf(
        context: *mut MultiScalarMultContext,
        out: *mut u64,
        npoints: usize,
        batch_size: usize,
        scalars: *const Fr,
        ffi_affine_sz: usize,
    ) -> cuda::Error;
    
    fn mult_pippenger_update(
        context: *mut MultiScalarMultContext,
        points_with_infinity: *const G1Affine,
    ) -> cuda::Error;
}
/*
pub fn multi_scalar_mult_init<G: AffineCurve>(
    dev_id: usize,
    points: &[G],
) -> MultiScalarMultContext {
    let mut ret = MultiScalarMultContext {
        device_id: 0,
        gpu_id : dev_id,
        obj_id : 0,
        bit_len : 0,
        is_used : false,
        context: std::ptr::null_mut(),
    };

    let err = unsafe {
        mult_pippenger_init(
            &mut ret,
            dev_id,
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

    let batch_size = scalars.len() / npoints;
    let mut ret = vec![G::Projective::zero(); batch_size];
    let err = unsafe {
        let result_ptr = 
            &mut *(&mut ret as *mut Vec<G::Projective>
                   as *mut Vec<u64>);

        mult_pippenger_inf(
            context,
            result_ptr.as_mut_ptr(),
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

pub fn multi_scalar_update<G: ArsAffineCurve>(
    context: &mut MultiScalarMultContext,
    points: &[G],
) {
    let err = unsafe {
        mult_pippenger_update(
            context,
            points as *const _ as *const G1Affine,
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}*/

pub fn ars_multi_scalar_mult_init<G: ArsAffineCurve>(
    dev_id: usize,
    points: &[G],
) -> MultiScalarMultContext {
    let mut ret = MultiScalarMultContext {
        device_id: 0,
        gpu_id : dev_id,
        obj_id : 0,
        bit_len : 0,
        is_used : false,
        context: std::ptr::null_mut(),
    };

    let mut bit_len = 15u32;
    if points.len() > (1 << bit_len){
        bit_len = 16;
    }

    let err = unsafe {
        mult_pippenger_init(
            &mut ret,
            dev_id,
            points as *const _ as *const G1Affine,
            1 << bit_len,
            std::mem::size_of::<G1Affine>(),
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }

    ret
}

pub fn ars_multi_scalar_update<G: ArsAffineCurve>(
    context: &mut MultiScalarMultContext,
    points: &[G],
) {
    let err = unsafe {
        mult_pippenger_update(
            context,
            points as *const _ as *const G1Affine,
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}

pub fn ars_multi_scalar_mult<G: ArsAffineCurve>(
    context: &mut MultiScalarMultContext,
    points: &[G],
    scalars: &[<G::ScalarField as ArsPrimeField>::BigInteger],
) -> Vec<ArsG1Projective> {
    let mut bit_len = 15u32;
    if scalars.len() > (1 << bit_len){
        bit_len = 16;
    }
    let npoints = 1 << bit_len;

    let mut batch_size = scalars.len() / npoints;

    let mut ret = vec![G::Projective::zero(); batch_size];
    let err = unsafe {
        let result_ptr =
            &mut *(&mut ret as *mut Vec<G::Projective>
                   as *mut Vec<u64>);

        mult_pippenger_inf(
            context,
            result_ptr.as_mut_ptr(),
            npoints, batch_size,
            scalars as *const _ as *const Fr,
            std::mem::size_of::<G1Affine>(),
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }
    unsafe { std::mem::transmute::<&[_], &[ArsG1Projective]>(&ret).to_vec() }
}