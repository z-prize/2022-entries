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

//use core::convert::TryInto;
use std::collections::BTreeMap;

use crate::{
    //fft,
    fft::{
        //domain::IFFTPrecomputation,
        //polynomial::PolyMultiplier,
        DensePolynomial,
        EvaluationDomain,
        //SparsePolynomial,
    },
    polycommit::sonic_pc::{LabeledPolynomial, PolynomialInfo, PolynomialLabel},
    snark::marlin::{
        ahp::{
            //indexer::{CircuitInfo, Matrix},
            indexer::{CircuitInfo},
            verifier,
            AHPForR1CS,
            UnnormalizedBivariateLagrangePoly,
        },
        prover,
        MarlinMode,
    },
};
use snarkvm_fields::PrimeField;
//use snarkvm_utilities::{cfg_iter, cfg_iter_mut, ExecutionPool};
use snarkvm_utilities::{cfg_iter, cfg_iter_mut};

//use itertools::Itertools;
use rand_core::RngCore;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

impl<F: PrimeField, MM: MarlinMode> AHPForR1CS<F, MM> {
    /// Output the number of oracles sent by the prover in the second round.
    pub fn num_second_round_oracles() -> usize {
        2
    }

    /// Output the degree bounds of oracles in the first round.
    pub fn second_round_polynomial_info(info: &CircuitInfo<F>) -> BTreeMap<PolynomialLabel, PolynomialInfo> {
        let constraint_domain_size = EvaluationDomain::<F>::compute_size_of_domain(info.num_constraints).unwrap();
        [
            PolynomialInfo::new("g_1".into(), Some(constraint_domain_size - 2), Self::zk_bound()),
            PolynomialInfo::new("h_1".into(), None, None),
        ]
        .into_iter()
        .map(|info| (info.label().into(), info))
        .collect()
    }

    /// Output the second round message and the next state.
    pub fn prover_second_round<'a, R: RngCore>(
        verifier_message: &verifier::FirstMessage<F>,
        mut state: prover::State<'a, F, MM>,
        _r: &mut R,
        z: DensePolynomial<F>,
        z_c: DensePolynomial<F>,
    ) -> (prover::SecondOracles<F>, prover::State<'a, F, MM>) {
        let round_time = start_timer!(|| "AHP::Prover::SecondRound");

        let constraint_domain = state.constraint_domain;
        let zk_bound = Self::zk_bound();

        let verifier::FirstMessage { alpha, eta_b, eta_c, .. } = verifier_message;

        //let (summed_z_m, t) = Self::calculate_summed_z_m_and_t(&state, *alpha, *eta_b, *eta_c, batch_combiners, z_c);
        ////////////////////////////////////////////////// calculate_summed_z_m_and_t
        let (summed_z_m, t) = {
            ////////////////////////////////////////////// summed_z_m
            // summed_z_m = z_a + z_b * eta_b + z_c * eta_c
            let summed_z_m_poly_time = start_timer!(|| "Compute z_m poly");
            let mut summed_z_m = z_c.clone(); // z_c constant
            let first_msg = state.first_round_oracles.as_ref().unwrap();
            let z_a_m = first_msg.batches[0].z_a_poly.polynomial().as_dense().unwrap(); // constant
            let mut z_b_m = first_msg.batches[0].z_b_poly.polynomial().as_dense().unwrap().clone(); // constant
            cfg_iter_mut!(z_b_m.coeffs).for_each(|b| *b *= eta_b);
            cfg_iter_mut!(summed_z_m.coeffs).for_each(|c| *c *= eta_c);
            cfg_iter_mut!(summed_z_m.coeffs).zip(&z_b_m.coeffs).zip(&z_a_m.coeffs).for_each(|((z, b), c)| *z += *b + *c);
            end_timer!(summed_z_m_poly_time);
            ////////////////////////////////////////////// summed_z_m

            ////////////////////////////////////////////// t
            let t_poly_time = start_timer!(|| "Compute t poly");
            let t_poly_time_inversion = start_timer!(|| "Compute t poly inversion");
            let r_alpha_x_evals =
                constraint_domain.batch_eval_unnormalized_bivariate_lagrange_poly_with_diff_inputs(*alpha);
            end_timer!(t_poly_time_inversion);
            
            let b_eta = *eta_b;
            let c_eta = *eta_c;
            let input_domain = state.input_domain;
            let constraint_domain = state.constraint_domain;

            // Compute t evals and perform iNTT
            let t_evals_on_h_3_vec = snarkvm_cuda::compute_poly_t(&b_eta, &c_eta, &r_alpha_x_evals,
                                                                  constraint_domain.size() as u32,
                                                                  input_domain.size() as u32);
            let t_evals_on_h_gpu = DensePolynomial::from_coefficients_vec(t_evals_on_h_3_vec);
            let t = t_evals_on_h_gpu;
            
            end_timer!(t_poly_time);
            /////////////////////////////////////////////// t

            (summed_z_m, t)
        };
        ////////////////////////////////////////////////// calculate_summed_z_m_and_t

        let sumcheck_lhs = Self::calculate_lhs(&state, t, summed_z_m, z, *alpha);

        debug_assert!(
            sumcheck_lhs.evaluate_over_domain_by_ref(constraint_domain).evaluations.into_iter().sum::<F>().is_zero()
        );

        let sumcheck_time = start_timer!(|| "Compute sumcheck h and g polys");
        let (h_1, x_g_1) = sumcheck_lhs.divide_by_vanishing_poly(constraint_domain).unwrap();
        let g_1 = DensePolynomial::from_coefficients_slice(&x_g_1.coeffs[1..]);
        drop(x_g_1);
        end_timer!(sumcheck_time);

        assert!(g_1.degree() <= constraint_domain.size() - 2);
        assert!(h_1.degree() <= 2 * constraint_domain.size() + 2 * zk_bound.unwrap_or(0) - 2);

        let oracles = prover::SecondOracles {
            g_1: LabeledPolynomial::new("g_1".into(), g_1, Some(constraint_domain.size() - 2), zk_bound),
            h_1: LabeledPolynomial::new("h_1".into(), h_1, None, None),
        };
        assert!(oracles.matches_info(&Self::second_round_polynomial_info(&state.index.index_info)));

        state.verifier_first_message = Some(verifier_message.clone());
        end_timer!(round_time);

        (oracles, state)
    }

    fn calculate_lhs(
        state: &prover::State<F, MM>,
        t: DensePolynomial<F>,
        summed_z_m: DensePolynomial<F>,
        z: DensePolynomial<F>,
        alpha: F,
    ) -> DensePolynomial<F> {
        let constraint_domain = state.constraint_domain;
        let q_1_time = start_timer!(|| "Compute LHS of sumcheck");

        let mul_domain_size = (constraint_domain.size() + summed_z_m.coeffs.len()).max(t.coeffs.len() + z.len());
        let mul_domain =
            EvaluationDomain::new(mul_domain_size).expect("field is not smooth enough to construct domain");

        let q_1_time_inversion = start_timer!(|| "Compute LHS of sumcheck inversion");
        //   elements is just roots of unity
        //   batch_inversion blocks move to GPU
        let vanish_x = constraint_domain.evaluate_vanishing_polynomial(alpha);
        let elements = mul_domain.elements().collect::<Vec<_>>();
        
        let mut denoms = cfg_iter!(elements).map(|e| alpha - e).collect::<Vec<_>>();
        assert!(mul_domain.size() > constraint_domain.size());
        snarkvm_fields::batch_inversion(&mut denoms);
        end_timer!(q_1_time_inversion);

        let q_1_time_lhs = start_timer!(|| "Compute LHS of sumcheck GPU lhs");
        let gpu_result_vec = snarkvm_cuda::calculate_lhs(&vanish_x, &denoms,
                                                         &summed_z_m, &z, &t,
                                                         constraint_domain.size(),
                                                         constraint_domain.size() * 4, &F::zero());
        let lhs_gpu = DensePolynomial::from_coefficients_vec(gpu_result_vec);
        let lhs = lhs_gpu;
        end_timer!(q_1_time_lhs);

        end_timer!(q_1_time);

        lhs
    }
}
