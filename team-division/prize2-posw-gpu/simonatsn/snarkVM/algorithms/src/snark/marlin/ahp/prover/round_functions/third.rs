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

use core::convert::TryInto;
use std::collections::BTreeMap;

use crate::{
    fft::{
        domain::{FFTPrecomputation, IFFTPrecomputation},
        //polynomial::PolyMultiplier,
        DensePolynomial,
        EvaluationDomain,
        //Evaluations as EvaluationsOnDomain,
    },
    polycommit::sonic_pc::{LabeledPolynomial, PolynomialInfo, PolynomialLabel},
    snark::marlin::{
        ahp::{indexer::CircuitInfo, verifier, AHPError, AHPForR1CS},
        matrices::MatrixArithmetization,
        prover,
        MarlinMode,
    },
};
use snarkvm_fields::{batch_inversion_and_mul, PrimeField};
use snarkvm_utilities::{cfg_iter, cfg_iter_mut, ExecutionPool};

use rand_core::RngCore;

#[cfg(not(feature = "parallel"))]
use itertools::Itertools;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

impl<F: PrimeField, MM: MarlinMode> AHPForR1CS<F, MM> {
    /// Output the number of oracles sent by the prover in the third round.
    pub fn num_third_round_oracles() -> usize {
        3
    }

    /// Output the degree bounds of oracles in the first round.
    pub fn third_round_polynomial_info(info: &CircuitInfo<F>) -> BTreeMap<PolynomialLabel, PolynomialInfo> {
        let non_zero_a_size = EvaluationDomain::<F>::compute_size_of_domain(info.num_non_zero_a).unwrap();
        let non_zero_b_size = EvaluationDomain::<F>::compute_size_of_domain(info.num_non_zero_b).unwrap();
        let non_zero_c_size = EvaluationDomain::<F>::compute_size_of_domain(info.num_non_zero_c).unwrap();

        [
            PolynomialInfo::new("g_a".into(), Some(non_zero_a_size - 2), None),
            PolynomialInfo::new("g_b".into(), Some(non_zero_b_size - 2), None),
            PolynomialInfo::new("g_c".into(), Some(non_zero_c_size - 2), None),
        ]
        .into_iter()
        .map(|info| (info.label().into(), info))
        .collect()
    }

    /// Output the third round message and the next state.
    pub fn prover_third_round<'a, R: RngCore>(
        verifier_message: &verifier::SecondMessage<F>,
        mut state: prover::State<'a, F, MM>,
        _r: &mut R,
    ) -> Result<(prover::ThirdMessage<F>, prover::ThirdOracles<F>, prover::State<'a, F, MM>), AHPError> {
        let round_time = start_timer!(|| "AHP::Prover::ThirdRound");

        let verifier::FirstMessage { alpha, .. } = state
            .verifier_first_message
            .as_ref()
            .expect("prover::State should include verifier_first_msg when prover_third_round is called");

        let beta = verifier_message.beta;

        let v_H_at_alpha = state.constraint_domain.evaluate_vanishing_polynomial(*alpha);
        let v_H_at_beta = state.constraint_domain.evaluate_vanishing_polynomial(beta);

        let v_H_alpha_v_H_beta = v_H_at_alpha * v_H_at_beta;

        let largest_non_zero_domain_size = Self::max_non_zero_domain(&state.index.index_info).size_as_field_element;
        let mut pool = ExecutionPool::with_capacity(3);
        pool.add_job(|| {
            Self::matrix_sumcheck_helper(
                "a",
                state.non_zero_a_domain,
                &state.index.a_arith,
                *alpha,
                beta,
                v_H_alpha_v_H_beta,
                largest_non_zero_domain_size,
                state.fft_precomputation(),
                state.ifft_precomputation(),
            )
        });

        pool.add_job(|| {
            Self::matrix_sumcheck_helper(
                "b",
                state.non_zero_b_domain,
                &state.index.b_arith,
                *alpha,
                beta,
                v_H_alpha_v_H_beta,
                largest_non_zero_domain_size,
                state.fft_precomputation(),
                state.ifft_precomputation(),
            )
        });

        pool.add_job(|| {
            Self::matrix_sumcheck_helper(
                "c",
                state.non_zero_c_domain,
                &state.index.c_arith,
                *alpha,
                beta,
                v_H_alpha_v_H_beta,
                largest_non_zero_domain_size,
                state.fft_precomputation(),
                state.ifft_precomputation(),
            )
        });

        let [(sum_a, lhs_a, g_a), (sum_b, lhs_b, g_b), (sum_c, lhs_c, g_c)]: [_; 3] =
            pool.execute_all().try_into().unwrap();

        let msg = prover::ThirdMessage { sum_a, sum_b, sum_c };
        let oracles = prover::ThirdOracles { g_a, g_b, g_c };
        state.lhs_polynomials = Some([lhs_a, lhs_b, lhs_c]);
        state.sums = Some([sum_a, sum_b, sum_c]);
        assert!(oracles.matches_info(&Self::third_round_polynomial_info(&state.index.index_info)));

        end_timer!(round_time);

        Ok((msg, oracles, state))
    }

    #[allow(clippy::too_many_arguments)]
    fn matrix_sumcheck_helper(
        label: &str,
        non_zero_domain: EvaluationDomain<F>,
        arithmetization: &MatrixArithmetization<F>,
        alpha: F,
        beta: F,
        v_H_alpha_v_H_beta: F,
        largest_non_zero_domain_size: F,
        _fft_precomputation: &FFTPrecomputation<F>,
        _ifft_precomputation: &IFFTPrecomputation<F>,
    ) -> (F, DensePolynomial<F>, LabeledPolynomial<F>) {
        let (row_on_K, col_on_K, _row_col_on_K) =
            (&arithmetization.evals_on_K.row, &arithmetization.evals_on_K.col, &arithmetization.evals_on_K.row_col);

        let f_evals_time = start_timer!(|| "Computing f evals on K");
        let mut inverses: Vec<_> = cfg_iter!(row_on_K.evaluations)
            .zip_eq(&col_on_K.evaluations)
            .map(|(r, c)| (beta - r) * (alpha - c))
            .collect();
        batch_inversion_and_mul(&mut inverses, &v_H_alpha_v_H_beta);
        end_timer!(f_evals_time);

        let sumcheck_gpu = start_timer!(|| "Computing sumcheck GPU");
        let cache_var = if label == "a" {
            snarkvm_cuda::ArithVar::ArithA
        } else if label == "b" {
            snarkvm_cuda::ArithVar::ArithB
        } else {
            snarkvm_cuda::ArithVar::ArithC
        };
        
        let (h_poly_vec, g_poly_vec) = snarkvm_cuda::matrix_sumcheck
            (non_zero_domain.size(), cache_var,
             &alpha, &beta, &v_H_alpha_v_H_beta,
             &inverses, &F::zero());
        end_timer!(sumcheck_gpu);
        
        let f_coeff0 = g_poly_vec[0];
        let h = DensePolynomial::from_coefficients_slice(&h_poly_vec);
        let g = DensePolynomial::from_coefficients_slice(&g_poly_vec[1..]);

        let (mut h, remainder) = h.divide_by_vanishing_poly(non_zero_domain).unwrap();
        assert!(remainder.is_zero());
        let multiplier = non_zero_domain.size_as_field_element / largest_non_zero_domain_size;
        cfg_iter_mut!(h.coeffs).for_each(|c| *c *= multiplier);

        let g = LabeledPolynomial::new("g_".to_string() + label, g, Some(non_zero_domain.size() - 2), None);

        assert!(h.degree() <= non_zero_domain.size() - 2);
        assert!(g.degree() <= non_zero_domain.size() - 2);
        (f_coeff0, h, g)
    }
}
