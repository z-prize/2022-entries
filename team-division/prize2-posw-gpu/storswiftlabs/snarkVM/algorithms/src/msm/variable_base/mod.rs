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

mod batched;
mod standard;

#[cfg(target_arch = "x86_64")]
pub mod prefetch;

use core::str::FromStr;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use snarkvm_curves::traits::{AffineCurve};
use snarkvm_fields::{PrimeField};
use snarkvm_utilities::{cfg_iter};

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "opencl")]
mod opencl;

mod test;

use snarkvm_fields::{FftField, Field};

#[cfg(any(feature = "cuda", feature = "opencl"))]
use core::sync::atomic::{AtomicBool, Ordering};

use std::env;


#[cfg(any(feature = "cuda", feature = "opencl"))]
static HAS_GPU_FAILED: AtomicBool = AtomicBool::new(false);

const BELLMAN_WORKING_GPUS: &str = "BELLMAN_WORKING_GPUS";

pub struct VariableBase;

use lazy_static::lazy_static;
use rand::Rng;

lazy_static! {
    // The owned CUDA contexts are stored globally. Each devives contains an unowned reference,
    // so that devices can be cloned.
    static ref MAX_SCALARS: usize = {
          let max = get_max_scalars_len();
        //   println!("MAX_SCALARS {}", max);
          max
    };

     static ref MAX_FFT: usize = {
          let max = get_max_fft_len();
        //   println!("MAX_FFT {}", max);
          max
    };

    static ref FFT_LEVEL: u32 = {
          let level = get_fft_level();
        //   println!("FFT_LEVEL {}", level);
          level
    };
}

pub fn get_max_scalars_len() -> usize {
    if !env::var(BELLMAN_WORKING_GPUS).is_ok() {
        return 99999999;
    }

    match env::var("MAX_SCALARS_LEN")
        .ok()
        .and_then(|s| usize::from_str(&s).ok())
    {
        Some(x) if x > 0 => x,
        _=> 4,
    }
}

pub fn get_fft_level() -> u32 {
    use aleo_std::Cpu;
    match env::var("FFT_LEVEL")
        .ok()
        .and_then(|s| u32::from_str(&s).ok())
    {
        Some(x) if x > 0 => x,
        _ => {
            match aleo_std::get_cpu() {
                Cpu::Intel => 0,
                Cpu::AMD | Cpu::Unknown => 5,
            }
        },
    }
}

pub fn get_max_fft_len() -> usize {
    if !env::var(BELLMAN_WORKING_GPUS).is_ok() {
        return 99999999;
    }

    match env::var("MAX_FFT_LEN")
        .ok()
        .and_then(|s| usize::from_str(&s).ok())
    {
        Some(x) if x > 0 => x,
        _ => 131072,
    }
}

impl VariableBase {
    pub fn msm2<G: AffineCurve, F: PrimeField>(
        bases: &[G],
        scalars: &[F],
    )
        -> G::Projective {
        // For BLS12-377, we perform variable base MSM using a batched addition technique.
        if scalars.len() >= *MAX_SCALARS {
            #[cfg(feature = "opencl")]
            match opencl::msm_opencl(bases, scalars) {
                Ok(x) => return x,
                Err(e) => {
                    HAS_GPU_FAILED.store(true, Ordering::SeqCst);
                    eprintln!("GPU OPENCL failed, moving to the CPU MSM method {:?}", e);
                }
            }

            #[cfg(feature = "cuda")]
            match cuda::msm_cuda(bases, scalars) {
                Ok(x) => {
                    //let coeffs = cfg_iter!(scalars).map(|s| s.to_repr()).collect::<Vec<_>>();
                    //let cpu = batched::msm(bases, coeffs.as_slice());
                    //eprintln!("msm2   len() {} / {}", bases.len(), scalars.len());
                    //eprintln!("msm2    cpu {}", cpu.to_affine());
                    //eprintln!("msm2   cuda {}", x.to_affine());
                    //assert_eq!(cpu.to_affine(), x.to_affine());
                    return x;
                }
                Err(e) => {
                    HAS_GPU_FAILED.store(true, Ordering::SeqCst);
                    eprintln!("GPU CUDA failed, moving to the CPU MSM method {:?}", e);
                }
            }
        }

        let coeffs = cfg_iter!(scalars).map(|s| s.to_repr()).collect::<Vec<_>>();
        batched::msm(bases, coeffs.as_slice())
    }

    #[allow(unused)]
    pub fn msm<G: AffineCurve>(bases: &[G], scalars: &[<G::ScalarField as PrimeField>::BigInteger]) -> G::Projective {
        // For BLS12-377, we perform variable base MSM using a batched addition technique.
        /*        if TypeId::of::<G>() == TypeId::of::<G1Affine>() {
                   #[cfg(feature = "cuda")]
                    if !HAS_CUDA_FAILED.load(Ordering::SeqCst) {
                        match cuda::msm_cuda(bases, scalars) {
                            Ok(x) => return x,
                            Err(e) => {
                                HAS_CUDA_FAILED.store(true, Ordering::SeqCst);
                                eprintln!("GPU CUDA failed, moving to the CPU MSM method {:?}", e);
                            }
                        }
                    }
                    batched::msm(bases, scalars)
                }
        */
        batched::msm(bases, scalars)
    }

    // cuda :false  cpu:false, opencl :intel true, amd false
    pub fn working_gpu(datasize: usize) -> bool {
        if !env::var(BELLMAN_WORKING_GPUS).is_ok() {
            return false;
        }

        if (datasize * 2) > (*MAX_FFT * 3) {
            return true;
        }

        let mut rng = rand::thread_rng();

        let n: u32 = rng.gen_range(0..10);

        return (datasize >= *MAX_FFT) && (n >= *FFT_LEVEL)
    }

    pub fn msm4_mul_assign<F: FftField, T: crate::fft::DomainCoeff<F>>(_bases: &mut [T], _inv: F) {
        #[cfg(feature = "opencl")]
        if _bases.len() > 32 {
            opencl::msm4_mul_assign(_bases, _inv).unwrap();
        } else {
            _bases.iter_mut().for_each(|val| *val *= _inv);
        }
    }

    pub fn msm4_mul2_assign<F: FftField>(_bases: &mut [F], _others: &[F]) {
        #[cfg(feature = "opencl")]
        if _bases.len() > 32 {
            opencl::msm4_mul2_assign(_bases, _others).unwrap();
        } else {
            _bases.iter_mut().zip(_others).for_each(|(a, b)| *a *= b);
        }
    }


    pub fn msm4_io_helper<F: FftField, T: crate::fft::DomainCoeff<F>>(_bases: &mut [T], _root: F) {
        #[cfg(any(feature = "cuda", feature = "opencl"))]
        {
            if _bases.len() < 5 {
                let powers = vec![F::one(), _root];
                let mut gap = _bases.len() / 2;
                while gap > 0 {
                    let chunk_size = 2 * gap;
                    let num_chunks = _bases.len() / chunk_size;
                    crate::fft::EvaluationDomain::apply_butterfly(
                        crate::fft::EvaluationDomain::butterfly_fn_io,
                        _bases,
                        &powers,
                        num_chunks,
                        chunk_size,
                        num_chunks,
                        1,
                        gap,
                    );
                    gap /= 2;
                }
            } else {
                #[cfg(feature = "opencl")]
                opencl::msm4_io_helper(_bases, _root).unwrap();

                #[cfg(feature = "cuda")]
                cuda::msm_io_helper(_bases, _root);
            }
        }
    }


    pub fn msm4_oi_helper<F: FftField, T: crate::fft::DomainCoeff<F>>(_bases: &mut [T], _root: F) {
        #[cfg(any(feature = "cuda", feature = "opencl"))]
        {
            if _bases.len() < 5 {
                let powers = vec![F::one(), _root];
                let mut gap = 1;
                while gap < _bases.len() {
                    let chunk_size = 2 * gap;
                    let num_chunks = _bases.len() / chunk_size;
                    crate::fft::EvaluationDomain::apply_butterfly(
                        crate::fft::EvaluationDomain::butterfly_fn_oi,
                        _bases,
                        &powers,
                        num_chunks,
                        chunk_size,
                        num_chunks,
                        1,
                        gap,
                    );
                    gap *= 2;
                }
            } else {
                #[cfg(feature = "opencl")]
                opencl::msm4_oi_helper(_bases, _root).unwrap();

                #[cfg(feature = "cuda")]
                cuda::msm_oi_helper(_bases, _root);
            }
        }
    }


    pub fn msm4_powers_serial<F: Field>(_root: F, _rsize: usize) -> Vec<F> {
        #[cfg(feature = "opencl")]
        {
            let result = opencl::msm4_powers_serial(_root, _rsize).unwrap();
            unsafe { std::mem::transmute(result) }
        }
        #[cfg(not(feature = "opencl"))]
        vec![]
    }


    pub fn msm4_evaluate<F: PrimeField>(_coeffs: &[F], _roots: &[F]) -> Vec<F> {
        #[cfg(feature = "opencl")]
        {
            let result = opencl::msm4_evaluate(_coeffs, _roots).unwrap();
            unsafe { std::mem::transmute(result) }
        }
        #[cfg(not(feature = "opencl"))]
        vec![F::zero()]
    }

    #[cfg(test)]
    fn msm_naive<G: AffineCurve>(bases: &[G], scalars: &[<G::ScalarField as PrimeField>::BigInteger]) -> G::Projective {
        use itertools::Itertools;
        use snarkvm_utilities::BitIteratorBE;

        bases.iter().zip_eq(scalars).map(|(base, scalar)| base.mul_bits(BitIteratorBE::new(*scalar))).sum()
    }

    #[cfg(test)]
    fn msm_naive_parallel<G: AffineCurve>(
        bases: &[G],
        scalars: &[<G::ScalarField as PrimeField>::BigInteger],
    ) -> G::Projective {
        use rayon::prelude::*;
        use snarkvm_utilities::BitIteratorBE;

        bases.par_iter().zip_eq(scalars).map(|(base, scalar)| base.mul_bits(BitIteratorBE::new(*scalar))).sum()
    }
}


