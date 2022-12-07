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
        polynomial::PolyMultiplier,
        DensePolynomial, EvaluationDomain, Evaluations as EvaluationsOnDomain,
    },
    polycommit::sonic_pc::{LabeledPolynomial, PolynomialInfo, PolynomialLabel},
    snark::marlin::{
        ahp::{indexer::CircuitInfo, verifier, AHPError, AHPForR1CS},
        matrices::MatrixArithmetization,
        prover, MarlinMode,
    },
};
use snarkvm_fields::{batch_inversion_and_mul, PrimeField};
use snarkvm_utilities::{cfg_iter, cfg_iter_mut, ExecutionPool};

use rand_core::RngCore;

use crate::fft::cuda_fft::*;

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


    #[allow(clippy::too_many_arguments)]
    fn matrix_sumcheck_helper(
        label: &str,
        non_zero_domain: EvaluationDomain<F>,
        arithmetization: &MatrixArithmetization<F>,
        alpha: F,
        beta: F,
        v_H_alpha_v_H_beta: F,
        largest_non_zero_domain_size: F,
        fft_precomputation: &FFTPrecomputation<F>,
        ifft_precomputation: &IFFTPrecomputation<F>,
    ) -> (F, DensePolynomial<F>, LabeledPolynomial<F>) {
        let mut job_pool = snarkvm_utilities::ExecutionPool::with_capacity(2);
        job_pool.add_job(|| {            
            let a_poly = {
                let coeffs = cfg_iter!(arithmetization.val.as_dense().unwrap().coeffs())
                    .map(|a| v_H_alpha_v_H_beta * a)
                    .collect();
                DensePolynomial::from_coefficients_vec(coeffs)
            };            
            a_poly
        });

        let (row_on_K, col_on_K, row_col_on_K) =
            (&arithmetization.evals_on_K.row, &arithmetization.evals_on_K.col, &arithmetization.evals_on_K.row_col);

        job_pool.add_job(|| {            
            let alpha_beta = alpha * beta;
            let b_poly = {
                let evals: Vec<F> = cfg_iter!(row_on_K.evaluations)
                    .zip_eq(&col_on_K.evaluations)
                    .zip_eq(&row_col_on_K.evaluations)
                    .map(|((r, c), r_c)| alpha_beta - alpha * r - beta * c + r_c)
                    .collect();
                EvaluationsOnDomain::from_vec_and_domain(evals, non_zero_domain)
                    .interpolate_with_pc(ifft_precomputation)
            };            
            b_poly
        });
        let [a_poly, b_poly]: [_; 2] = job_pool.execute_all().try_into().unwrap();
        
        let mut inverses: Vec<_> = cfg_iter!(row_on_K.evaluations)
            .zip_eq(&col_on_K.evaluations)
            .map(|(r, c)| (beta - r) * (alpha - c))
            .collect();
        batch_inversion_and_mul(&mut inverses, &v_H_alpha_v_H_beta);

        cfg_iter_mut!(inverses).zip_eq(&arithmetization.evals_on_K.val.evaluations).for_each(|(inv, a)| *inv *= a);
        let f_evals_on_K = inverses;
        
        let f = EvaluationsOnDomain::from_vec_and_domain(f_evals_on_K, non_zero_domain)
            .interpolate_with_pc(ifft_precomputation);        
        let g = DensePolynomial::from_coefficients_slice(&f.coeffs[1..]);
        let h = &a_poly
            - &{
                let mut multiplier = PolyMultiplier::new();
                multiplier.add_polynomial_ref(&b_poly, "b");
                multiplier.add_polynomial_ref(&f, "f");
                multiplier.add_precomputation(fft_precomputation, ifft_precomputation);
                multiplier.multiply().unwrap()
            };
        // Let K_max = largest_non_zero_domain;
        // Let K = non_zero_domain;
        // Let s := K_max.selector_polynomial(K) = (v_K_max / v_K) * (K.size() / K_max.size());
        // Let v_K_max := K_max.vanishing_polynomial();
        // Let v_K := K.vanishing_polynomial();

        // Later on, we multiply `h` by s, and divide by v_K_max.
        // Substituting in s, we get that h * s / v_K_max = h / v_K * (K.size() / K_max.size());
        // That's what we're computing here.
        let (mut h, remainder) = h.divide_by_vanishing_poly(non_zero_domain).unwrap();
        assert!(remainder.is_zero());
        let multiplier = non_zero_domain.size_as_field_element / largest_non_zero_domain_size;
        cfg_iter_mut!(h.coeffs).for_each(|c| *c *= multiplier);

        let g = LabeledPolynomial::new("g_".to_string() + label, g, Some(non_zero_domain.size() - 2), None);

        assert!(h.degree() <= non_zero_domain.size() - 2);
        assert!(g.degree() <= non_zero_domain.size() - 2);
        (f.coeffs[0], h, g)
    }

    /// Consolidate multiple ffts into one to reduce the interaction between CPU memory and GPU memory
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

        let (row_on_K_a, col_on_K_a, row_col_on_K_a) = (
            &state.index.a_arith.evals_on_K.row,
            &state.index.a_arith.evals_on_K.col,
            &state.index.a_arith.evals_on_K.row_col,
        );
        let (row_on_K_b, col_on_K_b, row_col_on_K_b) = (
            &state.index.b_arith.evals_on_K.row,
            &state.index.b_arith.evals_on_K.col,
            &state.index.b_arith.evals_on_K.row_col,
        );
        let (row_on_K_c, col_on_K_c, row_col_on_K_c) = (
            &state.index.c_arith.evals_on_K.row,
            &state.index.c_arith.evals_on_K.col,
            &state.index.c_arith.evals_on_K.row_col,
        );

        let mut pool = ExecutionPool::with_capacity(3);
        pool.add_job(|| {
            let coeffs = cfg_iter!(state.index.a_arith.val.as_dense().unwrap().coeffs())
                .map(|a| v_H_alpha_v_H_beta * a)
                .collect();
            coeffs
        });
        pool.add_job(|| {
            let coeffs = cfg_iter!(state.index.b_arith.val.as_dense().unwrap().coeffs())
                .map(|a| v_H_alpha_v_H_beta * a)
                .collect();
            coeffs
        });
        pool.add_job(|| {
            let coeffs = cfg_iter!(state.index.c_arith.val.as_dense().unwrap().coeffs())
                .map(|a| v_H_alpha_v_H_beta * a)
                .collect();
            coeffs
        });
        let [mut a2_a, mut a2_b, mut a2_c]: [_; 3] = pool.execute_all().try_into().unwrap();

        let mut pool = ExecutionPool::with_capacity(3);
        pool.add_job(|| {
            let mut inverses_a: Vec<_> = cfg_iter!(row_on_K_a.evaluations)
                .zip_eq(&col_on_K_a.evaluations)
                .map(|(r, c)| (beta - r) * (*alpha - c))
                .collect();
            batch_inversion_and_mul(&mut inverses_a, &v_H_alpha_v_H_beta);
            cfg_iter_mut!(inverses_a)
                .zip_eq(&state.index.a_arith.evals_on_K.val.evaluations)
                .for_each(|(inv, a)| *inv *= a);
            inverses_a
        });
        pool.add_job(|| {
            let mut inverses_b: Vec<_> = cfg_iter!(row_on_K_b.evaluations)
                .zip_eq(&col_on_K_b.evaluations)
                .map(|(r, c)| (beta - r) * (*alpha - c))
                .collect();
            batch_inversion_and_mul(&mut inverses_b, &v_H_alpha_v_H_beta);
            cfg_iter_mut!(inverses_b)
                .zip_eq(&state.index.b_arith.evals_on_K.val.evaluations)
                .for_each(|(inv, a)| *inv *= a);
            inverses_b
        });
        pool.add_job(|| {
            let mut inverses_c: Vec<_> = cfg_iter!(row_on_K_c.evaluations)
                .zip_eq(&col_on_K_c.evaluations)
                .map(|(r, c)| (beta - r) * (*alpha - c))
                .collect();
            batch_inversion_and_mul(&mut inverses_c, &v_H_alpha_v_H_beta);
            cfg_iter_mut!(inverses_c)
                .zip_eq(&state.index.c_arith.evals_on_K.val.evaluations)
                .for_each(|(inv, a)| *inv *= a);
            inverses_c
        });
        let [mut f_a_evals_on_K, mut f_b_evals_on_K, mut f_c_evals_on_K]: [_; 3] =
            pool.execute_all().try_into().unwrap();

        let alpha_beta = *alpha * beta;
        let mut pool = ExecutionPool::with_capacity(3);
        pool.add_job(|| {
            let evals_a: Vec<F> = cfg_iter!(row_on_K_a.evaluations)
                .zip_eq(&col_on_K_a.evaluations)
                .zip_eq(&row_col_on_K_a.evaluations)
                .map(|((r, c), r_c)| alpha_beta - *alpha * r - beta * c + r_c)
                .collect();
            evals_a
        });
        pool.add_job(|| {
            let evals_b: Vec<F> = cfg_iter!(row_on_K_b.evaluations)
                .zip_eq(&col_on_K_b.evaluations)
                .zip_eq(&row_col_on_K_b.evaluations)
                .map(|((r, c), r_c)| alpha_beta - *alpha * r - beta * c + r_c)
                .collect();
            evals_b
        });
        pool.add_job(|| {
            let evals_c: Vec<F> = cfg_iter!(row_on_K_c.evaluations)
                .zip_eq(&col_on_K_c.evaluations)
                .zip_eq(&row_col_on_K_c.evaluations)
                .map(|((r, c), r_c)| alpha_beta - *alpha * r - beta * c + r_c)
                .collect();
            evals_c
        });
        let [mut evals_a, mut evals_b, mut evals_c]: [_; 3] = pool.execute_all().try_into().unwrap();

        evals_a.resize(state.non_zero_a_domain.size as usize, F::zero());
        evals_b.resize(state.non_zero_b_domain.size as usize, F::zero());
        evals_c.resize(state.non_zero_c_domain.size as usize, F::zero());
        f_a_evals_on_K.resize(state.non_zero_a_domain.size as usize, F::zero());
        f_b_evals_on_K.resize(state.non_zero_b_domain.size as usize, F::zero());
        f_c_evals_on_K.resize(state.non_zero_c_domain.size as usize, F::zero());

        let f_i = state.non_zero_a_domain.evaluate_vanishing_polynomial(F::multiplicative_generator()).inverse().unwrap();
        third_compute_b_f_h_poly(&mut evals_a, &mut f_a_evals_on_K, &mut a2_a, &mut evals_b, &mut f_b_evals_on_K, &mut a2_b, &mut evals_c, &mut f_c_evals_on_K, &mut a2_c,
            &state.non_zero_c_domain.group_gen, &state.non_zero_a_domain.group_gen_inv, state.non_zero_c_domain.log_size_of_group, &state.non_zero_c_domain.size_inv, F::multiplicative_generator(), state.non_zero_a_domain.generator_inv, f_i, 
            &state.ifft_precomputation().inverse_roots, state.ifft_precomputation().domain.size, &state.fft_precomputation().roots, state.fft_precomputation().domain.size).unwrap();
        
        let (bf_a, bf_b, bf_c) = (evals_a, evals_b, evals_c);

        let f_a = DensePolynomial::from_coefficients_vec((*f_a_evals_on_K).to_vec());
        let f_b = DensePolynomial::from_coefficients_vec((*f_b_evals_on_K).to_vec());
        let f_c = DensePolynomial::from_coefficients_vec((*f_c_evals_on_K).to_vec());

        let g_1 = DensePolynomial::from_coefficients_slice(&f_a.coeffs[1..]);
        let g_2 = DensePolynomial::from_coefficients_slice(&f_b.coeffs[1..]);
        let g_3 = DensePolynomial::from_coefficients_slice(&f_c.coeffs[1..]);

        let h_a = DensePolynomial::from_coefficients_vec(bf_a);
        let h_b = DensePolynomial::from_coefficients_vec(bf_b);
        let h_c = DensePolynomial::from_coefficients_vec(bf_c);

        let g_a = LabeledPolynomial::new("g_".to_string() + "a", g_1, Some(state.non_zero_a_domain.size() - 2), None);
        let g_b = LabeledPolynomial::new("g_".to_string() + "b", g_2, Some(state.non_zero_b_domain.size() - 2), None);
        let g_c = LabeledPolynomial::new("g_".to_string() + "c", g_3, Some(state.non_zero_c_domain.size() - 2), None);
        assert!(h_a.degree() <= state.non_zero_a_domain.size() - 2);
        assert!(g_a.degree() <= state.non_zero_a_domain.size() - 2);
        assert!(h_b.degree() <= state.non_zero_b_domain.size() - 2);
        assert!(g_b.degree() <= state.non_zero_b_domain.size() - 2);
        assert!(h_c.degree() <= state.non_zero_c_domain.size() - 2);
        assert!(g_c.degree() <= state.non_zero_c_domain.size() - 2);
        let sum_a = f_a.coeffs[0];
        let sum_b = f_b.coeffs[0];
        let sum_c = f_c.coeffs[0];

        let msg = prover::ThirdMessage { sum_a, sum_b, sum_c };
        let oracles = prover::ThirdOracles { g_a, g_b, g_c };
        state.lhs_polynomials = Some([h_a, h_b, h_c]);
        state.sums = Some([sum_a, sum_b, sum_c]);
        assert!(oracles.matches_info(&Self::third_round_polynomial_info(&state.index.index_info)));
        end_timer!(round_time);

        Ok((msg, oracles, state))
    }
}