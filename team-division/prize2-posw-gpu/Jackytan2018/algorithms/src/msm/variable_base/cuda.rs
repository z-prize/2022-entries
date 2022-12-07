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

use snarkvm_curves::{
    bls12_377::{Fq, Fr, G1Affine, G1Projective},
    traits::{AffineCurve, ProjectiveCurve},
};
use snarkvm_fields::{PrimeField, Zero};
use snarkvm_utilities::BitIteratorBE;

use rust_gpu_tools::GPUError;
use crate::ars_gpu::*;

use std::{any::TypeId, path::Path, process::Command};

use msm_sppark::*;

#[allow(clippy::transmute_undefined_repr)]
pub fn msm_cuda<G: AffineCurve>(
    mut bases: &[G],
    mut scalars: &[<G::ScalarField as PrimeField>::BigInteger],
    round: usize,
) -> Result<G::Projective, GPUError> {
    if TypeId::of::<G>() != TypeId::of::<G1Affine>() {
        unimplemented!("trying to use cuda for unsupported curve");
    }

    if scalars.len() < 4 {
        let mut acc = G::Projective::zero();

        for (base, scalar) in bases.iter().zip(scalars.iter()) {
            acc += &base.mul_bits(BitIteratorBE::new(*scalar))
        }
        return Ok(acc);
    }

    let mut bit_len = 15usize;
    if scalars.len() > (1 << bit_len){
        bit_len = 16;
    }

    let mut msm_key = 1usize;
    if bit_len == 15 {
        if round == 2 {
            msm_key = 2;
        }
    }else{
        if round == 3 {
            msm_key = 2;
        }
    }

    if bases.len() < (1 << bit_len) {
        match ars_fetch_msm_context(&bases, bit_len, msm_key){
            Ok(program) => {
                let mut context = program;
                let msm_results = ars_multi_scalar_mult(&mut context, bases, scalars);
                ars_recycle_msm_context(context, bit_len, msm_key);
                Ok(unsafe { std::mem::transmute_copy(&msm_results[0]) })
            }
            Err(_) => {
                gpu_msg_dispatch_notify("GPUError: Device Not Found".to_string());
                Err(GPUError::DeviceNotFound)
            }
        }
    }else{
        match ars_fetch_msm_context(&bases[..(1 << bit_len)], bit_len, msm_key){
            Ok(program) => {
                let mut context = program;
                let msm_results = ars_multi_scalar_mult(&mut context, &bases[..(1 << bit_len)], scalars);
                ars_recycle_msm_context(context, bit_len, msm_key);
                Ok(unsafe { std::mem::transmute_copy(&msm_results[0]) })
            }
            Err(_) => {
                gpu_msg_dispatch_notify("GPUError: Device Not Found".to_string());
                Err(GPUError::DeviceNotFound)
            }
        }
    }
}