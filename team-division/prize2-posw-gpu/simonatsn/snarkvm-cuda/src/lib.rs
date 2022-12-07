// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![feature(nonnull_slice_from_raw_parts)]

#[allow(unused_imports)]
use blst::*;

use core::ffi::c_void;
sppark::cuda_error!();
pub mod util;

#[repr(C)]
pub enum NTTInputOutputOrder {
    NN = 0,
    NR = 1,
    RN = 2,
    RR = 3,
}

#[repr(C)]
enum NTTDirection {
    Forward = 0,
    Inverse = 1,
}

#[repr(C)]
enum NTTType {
    Standard = 0,
}

#[repr(C)]
pub enum ArithVar {
      ArithA = 0,
      ArithB,
      ArithC
}

#[cfg_attr(feature = "quiet", allow(improper_ctypes))]
extern "C" {
    fn snarkvm_init_gpu();

    fn snarkvm_cleanup_gpu();

    fn snarkvm_ntt_batch(
        inout: *mut core::ffi::c_void,
        N: usize,
        lg_domain_size: u32,
        ntt_order: NTTInputOutputOrder,
        ntt_direction: NTTDirection,
        ntt_type: NTTType,
    ) -> cuda::Error;

    fn snarkvm_polymul(
        out: *mut core::ffi::c_void,
        in0: *const core::ffi::c_void,
        in1: *const core::ffi::c_void,
        lg_domain_size: u32,
    ) -> cuda::Error;

    fn snarkvm_calculate_summed_z_m(
        out: *mut core::ffi::c_void,
        z_a: *const core::ffi::c_void,
        z_b: *const core::ffi::c_void,
        eta_c: *const core::ffi::c_void,
        eta_b_over_eta_c: *const core::ffi::c_void,
        lg_domain_size: u32,
    ) -> cuda::Error;

    fn snarkvm_cache_poly_t_inputs(
        a_len: usize,
        a_r: *const core::ffi::c_void,
        a_c: *const core::ffi::c_void,
        a_coeff: *const core::ffi::c_void,
        b_len: usize,
        b_r: *const core::ffi::c_void,
        b_c: *const core::ffi::c_void,
        b_coeff: *const core::ffi::c_void,
        c_len: usize,
        c_r: *const core::ffi::c_void,
        c_c: *const core::ffi::c_void,
        c_coeff: *const core::ffi::c_void,

        a_arith_row_on_k: *const core::ffi::c_void,
        a_arith_col_on_k: *const core::ffi::c_void,
        a_arith_row_col_on_k: *const core::ffi::c_void,
        a_arith_val: *const core::ffi::c_void,
        a_arith_evals_on_k: *const core::ffi::c_void,
        
        b_arith_row_on_k: *const core::ffi::c_void,
        b_arith_col_on_k: *const core::ffi::c_void,
        b_arith_row_col_on_k: *const core::ffi::c_void,
        b_arith_val: *const core::ffi::c_void,
        b_arith_evals_on_k: *const core::ffi::c_void,
        
        c_arith_row_on_k: *const core::ffi::c_void,
        c_arith_col_on_k: *const core::ffi::c_void,
        c_arith_row_col_on_k: *const core::ffi::c_void,
        c_arith_val: *const core::ffi::c_void,
        c_arith_evals_on_k: *const core::ffi::c_void,
    );

    fn snarkvm_compute_poly_t(
        out:    *mut core::ffi::c_void,
        eta_b:  *const core::ffi::c_void,
        eta_c:  *const core::ffi::c_void,
        r_alpha_x_evals:  *const core::ffi::c_void,
        lg_constraint_domain_size: u32,
        lg_input_domain_size: u32,
    ) -> cuda::Error;
    
    fn snarkvm_calculate_lhs(
        out: *mut core::ffi::c_void,
        vanish_x:  *const core::ffi::c_void,
        denoms: *const core::ffi::c_void, denoms_len: usize,
        b: *const core::ffi::c_void, b_len: usize,
        c: *const core::ffi::c_void, c_len: usize,
        d: *const core::ffi::c_void, d_len: usize,
        lg_domain_size: u32, lg_ext_domain_size: u32,
    ) -> cuda::Error;
    
    fn snarkvm_matrix_sumcheck(
        lg_domain_size : u32,
        cache_var: ArithVar,
        
        //f_coeff0:           *mut core::ffi::c_void,
        h_poly:             *mut core::ffi::c_void,
        g_poly:             *mut core::ffi::c_void,
        
        alpha:              *const core::ffi::c_void,
        beta:               *const core::ffi::c_void,
        v_H_alpha_v_H_beta: *const core::ffi::c_void,
        inverses:           *const core::ffi::c_void,
    ) -> cuda::Error;

    fn snarkvm_msm(
        out: *mut c_void,
        points_with_infinity: *const c_void,
        npoints: usize,
        bases_len: usize,
        scalars: *const c_void,
        ffi_affine_sz: usize,
    ) -> cuda::Error;

    fn snarkvm_msm_cache(
        points_with_infinity: *const c_void,
        bases_len: usize,
        ffi_affine_sz: usize,
    ) -> cuda::Error;
}

///////////////////////////////////////////////////////////////////////////////
// Rust functions
///////////////////////////////////////////////////////////////////////////////

pub fn init_gpu() {
    println!("Init GPU");
    unsafe {
        snarkvm_init_gpu();
    }
}

pub fn cleanup_gpu() {
    unsafe {
        snarkvm_cleanup_gpu();
    }
}

/// Compute an in-place NTT on the input data.
#[allow(non_snake_case)]
pub fn NTT_batch<T>(domain_size: usize, batch_size: usize,
                    inout: &mut [T], order: NTTInputOutputOrder) {
    if (domain_size & (domain_size - 1)) != 0 {
        panic!("domain_size is not power of 2");
    }
    let lg_domain_size = domain_size.trailing_zeros();

    let err = unsafe {
        snarkvm_ntt_batch(
            inout.as_mut_ptr() as *mut core::ffi::c_void,
            batch_size,
            lg_domain_size,
            order,
            NTTDirection::Forward,
            NTTType::Standard,
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}

/// Compute an in-place iNTT on the input data.
#[allow(non_snake_case)]
pub fn iNTT_batch<T>(domain_size: usize, batch_size: usize,
                     inout: &mut [T], order: NTTInputOutputOrder) {
    if (domain_size & (domain_size - 1)) != 0 {
        panic!("domain_size is not power of 2");
    }
    let lg_domain_size = domain_size.trailing_zeros();
    let err = unsafe {
        snarkvm_ntt_batch(
            inout.as_mut_ptr() as *mut core::ffi::c_void,
            batch_size,
            lg_domain_size,
            order,
            NTTDirection::Inverse,
            NTTType::Standard,
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}

/// Compute a polynomial multiply
#[allow(non_snake_case)]
pub fn polymul<T: std::clone::Clone>(in0: &[T], in1: &[T], zero: &T) -> Vec<T> {
    let initial_domain_size = in0.len();
    if (initial_domain_size & (initial_domain_size - 1)) != 0 {
        panic!("domain_size is not power of 2");
    }
    assert_eq!(in0.len(), in1.len());

    let lg_domain_size = initial_domain_size.trailing_zeros();
    let ext_domain_size = initial_domain_size * 2;
    
    let mut out = Vec::new();
    out.resize(ext_domain_size, zero.clone());
    let err = unsafe {
        snarkvm_polymul(
            out.as_mut_ptr() as *mut core::ffi::c_void,
            in0.as_ptr() as *const core::ffi::c_void,
            in1.as_ptr() as *const core::ffi::c_void,
            lg_domain_size,
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
    out
}

/// Compute summed_z_m
// Inputs: z_a(32768), z_b(32768), eta_c(1), eta_b_over_eta_c:(1)
// Outputs: summed_z_m(65536)
// z_b  = z_b:P * eta_c:F
// z_b += 1
// summed_z_m = z_a:P * eta_c_z_b_plus_one:P - Increases domain by one
// z_b -= 1
// summed_z_m += eta_b_over_eta_c:F * z_b:P
#[allow(non_snake_case)]
pub fn calculate_summed_z_m<T: std::clone::Clone>
    (z_a: &[T], z_b: &[T], eta_c: &T, eta_b_over_eta_c: &T,
     domain_size: usize, zero: &T) -> Vec<T>
{
    if (domain_size & (domain_size - 1)) != 0 {
        panic!("domain_size is not power of 2");
    }
    assert_eq!(domain_size, z_a.len());
    assert_eq!(domain_size, z_b.len());
    
    let lg_domain_size = domain_size.trailing_zeros();
    let ext_domain_size = domain_size * 2;

    let eta_c_slice = &[ eta_c.clone() ];
    let eta_b_over_eta_c_slice = &[ eta_b_over_eta_c.clone() ];
    
    let mut out = Vec::new();
    out.resize(ext_domain_size, zero.clone());
    let err = unsafe {
        snarkvm_calculate_summed_z_m(
            out.as_mut_ptr() as *mut core::ffi::c_void,
            z_a.as_ptr() as *const core::ffi::c_void,
            z_b.as_ptr() as *const core::ffi::c_void,
            eta_c_slice.as_ptr() as *const core::ffi::c_void,
            eta_b_over_eta_c_slice.as_ptr() as *const core::ffi::c_void,
            lg_domain_size,
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
    out
}

pub fn cache_poly_t_inputs<T: std::clone::Clone>
    (a_matrix_len: usize, a_flat_r: &[u32], a_flat_c: &[u32], a_flat_coeff: &[T],
     b_matrix_len: usize, b_flat_r: &[u32], b_flat_c: &[u32], b_flat_coeff: &[T],
     c_matrix_len: usize, c_flat_r: &[u32], c_flat_c: &[u32], c_flat_coeff: &[T],

     a_arith_row_on_k: &[T],
     a_arith_col_on_k: &[T],
     a_arith_row_col_on_k: &[T],
     a_arith_val: &[T],
     a_arith_evals_on_k: &[T],

     b_arith_row_on_k: &[T],
     b_arith_col_on_k: &[T],
     b_arith_row_col_on_k: &[T],
     b_arith_val: &[T],
     b_arith_evals_on_k: &[T],

     c_arith_row_on_k: &[T],
     c_arith_col_on_k: &[T],
     c_arith_row_col_on_k: &[T],
     c_arith_val: &[T],
     c_arith_evals_on_k: &[T],
    )
{
    unsafe {
        snarkvm_cache_poly_t_inputs
            (a_matrix_len,
             a_flat_r.as_ptr() as *const core::ffi::c_void,
             a_flat_c.as_ptr() as *const core::ffi::c_void,
             a_flat_coeff.as_ptr() as *const core::ffi::c_void,
             b_matrix_len,
             b_flat_r.as_ptr() as *const core::ffi::c_void,
             b_flat_c.as_ptr() as *const core::ffi::c_void,
             b_flat_coeff.as_ptr() as *const core::ffi::c_void,
             c_matrix_len,
             c_flat_r.as_ptr() as *const core::ffi::c_void,
             c_flat_c.as_ptr() as *const core::ffi::c_void,
             c_flat_coeff.as_ptr() as *const core::ffi::c_void,
             
             a_arith_row_on_k.as_ptr() as *const core::ffi::c_void,
             a_arith_col_on_k.as_ptr() as *const core::ffi::c_void,
             a_arith_row_col_on_k.as_ptr() as *const core::ffi::c_void,
             a_arith_val.as_ptr() as *const core::ffi::c_void,
             a_arith_evals_on_k.as_ptr() as *const core::ffi::c_void,
             
             b_arith_row_on_k.as_ptr() as *const core::ffi::c_void,
             b_arith_col_on_k.as_ptr() as *const core::ffi::c_void,
             b_arith_row_col_on_k.as_ptr() as *const core::ffi::c_void,
             b_arith_val.as_ptr() as *const core::ffi::c_void,
             b_arith_evals_on_k.as_ptr() as *const core::ffi::c_void,
             
             c_arith_row_on_k.as_ptr() as *const core::ffi::c_void,
             c_arith_col_on_k.as_ptr() as *const core::ffi::c_void,
             c_arith_row_col_on_k.as_ptr() as *const core::ffi::c_void,
             c_arith_val.as_ptr() as *const core::ffi::c_void,
             c_arith_evals_on_k.as_ptr() as *const core::ffi::c_void,
        );
    }
}

pub fn compute_poly_t<T: std::clone::Clone>
    (eta_b: &T, eta_c: &T, r_alpha_x_evals: &[T],
     constraint_domain_size: u32,
     input_domain_size: u32) -> Vec<T>
{
    assert!(constraint_domain_size == 32768);
    assert!(r_alpha_x_evals.len() == 32768);
    if (constraint_domain_size & (constraint_domain_size - 1)) != 0 {
        panic!("constraint_domain_size is not power of 2");
    }
    let lg_constraint_domain_size = constraint_domain_size.trailing_zeros();
    if (input_domain_size & (input_domain_size - 1)) != 0 {
        panic!("input_domain_size is not power of 2");
    }
    let lg_input_domain_size = input_domain_size.trailing_zeros();
    
    let eta_b_slice = &[ eta_b.clone() ];
    let eta_c_slice = &[ eta_c.clone() ];

    let mut out = Vec::new();
    out.resize(constraint_domain_size as usize, eta_b.clone());
    let err = unsafe {
        snarkvm_compute_poly_t(
            out.as_mut_ptr() as *mut core::ffi::c_void,
            eta_b_slice.as_ptr() as *const core::ffi::c_void,
            eta_c_slice.as_ptr() as *const core::ffi::c_void,
            r_alpha_x_evals.as_ptr() as *const core::ffi::c_void,
            lg_constraint_domain_size,
            lg_input_domain_size)
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }
    out
}

/// Compute lhs
#[allow(non_snake_case)]
pub fn calculate_lhs<T: std::clone::Clone>(vanish_x: &T, denoms: &[T],
                                           b: &[T], c: &[T], d: &[T],
                                           constraint_domain_size: usize,
                                           mul_domain_size: usize,
                                           zero: &T) -> Vec<T> {
    if (constraint_domain_size & (constraint_domain_size - 1)) != 0 {
        panic!("constraint_domain_size is not power of 2");
    }
    if (mul_domain_size & (mul_domain_size - 1)) != 0 {
        panic!("mul_domain_size is not power of 2");
    }
    // A will be in evaluation domain already
    let lg_domain_size = constraint_domain_size.trailing_zeros();
    let lg_ext_domain_size = mul_domain_size.trailing_zeros();
    let ext_domain_size: usize = 1 << lg_ext_domain_size;

    let vanish_x_slice = &[ vanish_x.clone() ];
    
    let mut out = Vec::new();
    out.resize(ext_domain_size, zero.clone());
    let err = unsafe {
        snarkvm_calculate_lhs(
            out.as_mut_ptr() as *mut core::ffi::c_void,
            vanish_x_slice.as_ptr() as *const core::ffi::c_void,
            denoms.as_ptr() as *const core::ffi::c_void, denoms.len(), 
            b.as_ptr() as *const core::ffi::c_void, b.len(),
            c.as_ptr() as *const core::ffi::c_void, c.len(),
            d.as_ptr() as *const core::ffi::c_void, d.len(),
            lg_domain_size, lg_ext_domain_size,
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
    out
}

#[allow(non_snake_case)]
pub fn matrix_sumcheck<T: std::clone::Clone>
    (non_zero_domain_size: usize,
     cache_var: ArithVar,
     alpha: &T,
     beta: &T,
     v_H_alpha_v_H_beta: &T,
     inverses: &[T],
     //zero: &T) -> (T, Vec<T>, Vec<T>)
     zero: &T) -> (Vec<T>, Vec<T>)
{
    assert!(non_zero_domain_size == 65536);
    assert!(inverses.len()       == non_zero_domain_size);

    if (non_zero_domain_size & (non_zero_domain_size - 1)) != 0 {
        panic!("non_zero_domain_size is not power of 2");
    }
    // A will be in evaluation domain already
    let lg_domain_size = non_zero_domain_size.trailing_zeros();
    
    let alpha_slice = &[ alpha.clone() ];
    let beta_slice = &[ beta.clone() ];
    let v_H_alpha_v_H_beta_slice = &[ v_H_alpha_v_H_beta.clone() ];
    
    let mut h_poly = Vec::new();
    h_poly.resize(non_zero_domain_size * 2, zero.clone());
    let mut g_poly = Vec::new();
    g_poly.resize(non_zero_domain_size, zero.clone());
    
    let err = unsafe {
        snarkvm_matrix_sumcheck(
            lg_domain_size,
            cache_var,
            
            h_poly.as_mut_ptr()               as *mut core::ffi::c_void,
            g_poly.as_mut_ptr()               as *mut core::ffi::c_void,

            alpha_slice.as_ptr()              as *const core::ffi::c_void,
            beta_slice.as_ptr()               as *const core::ffi::c_void,
            v_H_alpha_v_H_beta_slice.as_ptr() as *const core::ffi::c_void,
            inverses.as_ptr()                 as *const core::ffi::c_void,
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
    (h_poly, g_poly)
}

pub fn msm<Affine, Projective, Scalar>(
    points: &[Affine],
    scalars: &[Scalar],
) -> Projective {
    let npoints = scalars.len();
    if npoints > points.len() {
        panic!("length mismatch {} points < {} scalars",
               npoints, scalars.len())
    }
    let mut ret: Projective = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
    let err = unsafe {
        snarkvm_msm(
            &mut ret as *mut _ as *mut c_void,
            points as *const _ as *const c_void,
            npoints,
            points.len(),
            scalars as *const _ as *const c_void,
            std::mem::size_of::<Affine>(),
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }
    ret
}

pub fn msm_cache<Affine>(
    points: &[Affine],
) {
    let err = unsafe {
        snarkvm_msm_cache(
            points as *const _ as *const c_void,
            points.len(),
            std::mem::size_of::<Affine>(),
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}

