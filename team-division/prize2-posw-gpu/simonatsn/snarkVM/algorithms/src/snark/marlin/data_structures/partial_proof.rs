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

use crate::{
    fft::{DensePolynomial, EvaluationDomain},
    polycommit::sonic_pc::{Commitment, LabeledCommitment, LabeledPolynomial},
    snark::marlin::{
        fiat_shamir::traits::FiatShamirRng,
        MarlinMode,
        prover,
    },
};

use snarkvm_curves::PairingEngine;

#[derive(Clone)]
pub struct PartialProof<E: PairingEngine, FS: FiatShamirRng<E::Fr, E::Fq>> {
    /// The LDE of `w`.
    pub w_poly: LabeledPolynomial<E::Fr>,
    /// The LDE of `Az`.
    pub z_a_poly: LabeledPolynomial<E::Fr>,
    /// The LDE of `Bz`.
    pub z_b_poly: LabeledPolynomial<E::Fr>,

    /// padded public inputs for the entire batch.
    pub public_inputs: Vec<Vec<E::Fr>>,

    pub sponge: FS,

    pub first_commitments: Vec<LabeledCommitment<Commitment<E>>>,

    /// A list of polynomials corresponding to the interpolation of the public input.
    /// The length of this list must be equal to the batch size.
    pub x_poly: DensePolynomial<E::Fr>,

    /// A domain that is sized for the public input.
    pub input_domain: EvaluationDomain<E::Fr>,

    /// A domain that is sized for the number of constraints.
    pub constraint_domain: EvaluationDomain<E::Fr>,

    /// A domain that is sized for the number of non-zero elements in A.
    pub non_zero_a_domain: EvaluationDomain<E::Fr>,
    /// A domain that is sized for the number of non-zero elements in B.
    pub non_zero_b_domain: EvaluationDomain<E::Fr>,
    /// A domain that is sized for the number of non-zero elements in C.
    pub non_zero_c_domain: EvaluationDomain<E::Fr>,

    pub v_h_div_v_x: DensePolynomial<E::Fr>,

    pub w_comm_inc: E::G1Affine,

    pub z_c: DensePolynomial<E::Fr>,
}

impl<E: PairingEngine, FS: FiatShamirRng<E::Fr, E::Fq>> PartialProof<E, FS> {
    pub fn new<'a, MM: MarlinMode>(
        prover_state: &prover::State<E::Fr, MM>,
        sponge: FS,
        first_commitments: Vec<LabeledCommitment<Commitment<E>>>,
        v_h_div_v_x: DensePolynomial<E::Fr>,
        w_comm_inc: E::G1Affine,
        z_c: DensePolynomial<E::Fr>,
    ) -> Self {

        Self {
            w_poly: prover_state.first_round_oracles.as_ref().unwrap().batches[0].w_poly.clone(),
            z_a_poly: prover_state.first_round_oracles.as_ref().unwrap().batches[0].z_a_poly.clone(),
            z_b_poly: prover_state.first_round_oracles.as_ref().unwrap().batches[0].z_b_poly.clone(),
            public_inputs: prover_state.public_inputs(),
            sponge: sponge,
            first_commitments: first_commitments,
            x_poly: prover_state.x_poly[0].clone(),
            input_domain: prover_state.input_domain,
            constraint_domain: prover_state.constraint_domain,
            non_zero_a_domain: prover_state.non_zero_a_domain,
            non_zero_b_domain: prover_state.non_zero_b_domain,
            non_zero_c_domain: prover_state.non_zero_c_domain,
            v_h_div_v_x,
            w_comm_inc,
            z_c,
        }
    }
}
