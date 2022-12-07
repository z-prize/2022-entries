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
    fft::{DensePolynomial, Polynomial},
    msm::VariableBase,
    polycommit::PCError,
};
use snarkvm_curves::traits::{AffineCurve, PairingEngine, ProjectiveCurve};
use snarkvm_fields::{One, PrimeField};
use snarkvm_utilities::BitIteratorBE;

use core::sync::atomic::{AtomicBool, Ordering};
use rand_core::RngCore;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::polycommit::kzg10::*;

use crate::fft::EvaluationDomain;
use ec_gpu_common::{log2_floor, GpuPolyContainer, PolyKernel, PrimeField as GpuPrimeField, GPU_OP};

impl<E: PairingEngine> KZG10<E> {
    /// Outputs a commitment to `polynomial`.
    pub fn commit_on_gpu<Fg: GpuPrimeField>(
        kern: &PolyKernel<Fg>,
        gpu_container: &mut GpuPolyContainer<Fg>,
        powers: &Powers<E>,
        polynomial: &Polynomial<'_, E::Fr>,
        name: &str,
        hiding_bound: Option<usize>,
        terminator: &AtomicBool,
        rng: Option<&mut dyn RngCore>,
        start: usize,
        is_g: bool,
    ) -> Result<(Commitment<E>, Randomness<E>), PCError> {
        let real_name = match name {
            "g_a" => "g_a_evals".to_owned(),
            "g_b" => "g_b_evals".to_owned(),
            "g_c" => "g_c_evals".to_owned(),
            _ => name.to_owned(),
        };

        let gpu_poly = gpu_container.find(kern, &real_name)?;

        Self::check_degree_is_too_large(polynomial.degree(), powers.size())?;

        let commit_time = start_timer!(|| format!(
            "Committing to polynomial of degree {} with hiding_bound: {:?}",
            polynomial.degree(),
            hiding_bound,
        ));

        let mut commitment = match polynomial {
            Polynomial::Dense(_polynomial) => {
                let msm_time = start_timer!(|| "MSM to compute commitment to plaintext poly");
                let commitment = if real_name == "g_a_evals" || real_name == "g_b_evals" || real_name == "g_c_evals" {
                    VariableBase::msm_gpu_ptr_reuse_base(
                        &powers.powers_of_beta_g[..],
                        gpu_poly.get_memory(),
                        start,
                        GPU_OP::REUSE_SHIFTED_LAGRANGE_G,
                        true,
                    )
                } else if is_g {
                    VariableBase::msm_gpu_ptr_reuse_base(
                        &powers.powers_of_beta_g[..],
                        gpu_poly.get_memory(),
                        start,
                        GPU_OP::REUSE_G,
                        true,
                    )
                } else {
                    VariableBase::msm_gpu_ptr_reuse_base(
                        &powers.powers_of_beta_g[..],
                        gpu_poly.get_memory(),
                        start,
                        GPU_OP::REUSE_SHIFTED_G,
                        true,
                    )
                };

                end_timer!(msm_time);

                if terminator.load(Ordering::Relaxed) {
                    return Err(PCError::Terminated);
                }
                commitment
            }
            Polynomial::Sparse(polynomial) => polynomial
                .coeffs()
                .map(|(i, coeff)| {
                    powers.powers_of_beta_g[*i].mul_bits(BitIteratorBE::new_without_leading_zeros(coeff.to_repr()))
                })
                .sum(),
        };

        gpu_container.save(&real_name, gpu_poly)?;

        let mut randomness = Randomness::empty();
        if let Some(hiding_degree) = hiding_bound {
            let mut rng = rng.ok_or(PCError::MissingRng)?;
            let sample_random_poly_time =
                start_timer!(|| format!("Sampling a random polynomial of degree {}", hiding_degree));

            randomness = Randomness::rand(hiding_degree, false, &mut rng);
            Self::check_hiding_bound(
                randomness.blinding_polynomial.degree(),
                powers.powers_of_beta_times_gamma_g.len(),
            )?;
            end_timer!(sample_random_poly_time);
        }

        let random_ints = convert_to_bigints(&randomness.blinding_polynomial.coeffs);
        let msm_time = start_timer!(|| "MSM to compute commitment to random poly");

        let random_commitment =
            VariableBase::msm(&powers.powers_of_beta_times_gamma_g, random_ints.as_slice()).to_affine();
        end_timer!(msm_time);

        if terminator.load(Ordering::Relaxed) {
            return Err(PCError::Terminated);
        }

        commitment.add_assign_mixed(&random_commitment);

        end_timer!(commit_time);
        Ok((Commitment(commitment.into()), randomness))
    }

    /// Outputs a commitment to `polynomial`.
    pub fn commit_lagrange_on_gpu<Fg: GpuPrimeField>(
        _kern: &PolyKernel<Fg>,
        _gpu_container: &mut GpuPolyContainer<Fg>,
        lagrange_basis: &LagrangeBasis<E>,
        evaluations: &[E::Fr],
        _name: &str,
        hiding_bound: Option<usize>,
        terminator: &AtomicBool,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<(Commitment<E>, Randomness<E>), PCError> {
        Self::check_degree_is_too_large(evaluations.len() - 1, lagrange_basis.size())?;
        assert_eq!(evaluations.len().next_power_of_two(), lagrange_basis.size());

        let commit_time = start_timer!(|| format!(
            "Committing to polynomial of degree {} with hiding_bound: {:?}",
            evaluations.len() - 1,
            hiding_bound,
        ));

        let msm_time = start_timer!(|| "MSM to compute commitment to plaintext poly");
        let mut commitment = VariableBase::msm_gpu_reuse_base(
            &lagrange_basis.lagrange_basis_at_beta_g,
            evaluations,
            0,
            GPU_OP::REUSE_LAGRANGE_G,
            true,
        );
        end_timer!(msm_time);

        if terminator.load(Ordering::Relaxed) {
            return Err(PCError::Terminated);
        }

        let mut randomness = Randomness::empty();
        if let Some(hiding_degree) = hiding_bound {
            let mut rng = rng.ok_or(PCError::MissingRng)?;
            let sample_random_poly_time =
                start_timer!(|| format!("Sampling a random polynomial of degree {}", hiding_degree));

            randomness = Randomness::rand(hiding_degree, false, &mut rng);
            Self::check_hiding_bound(
                randomness.blinding_polynomial.degree(),
                lagrange_basis.powers_of_beta_times_gamma_g.len(),
            )?;
            end_timer!(sample_random_poly_time);
        }

        let random_ints = convert_to_bigints(&randomness.blinding_polynomial.coeffs);
        let msm_time = start_timer!(|| "MSM to compute commitment to random poly");
        let random_commitment =
            VariableBase::msm(&lagrange_basis.powers_of_beta_times_gamma_g, random_ints.as_slice()).to_affine();
        end_timer!(msm_time);

        if terminator.load(Ordering::Relaxed) {
            return Err(PCError::Terminated);
        }

        commitment.add_assign_mixed(&random_commitment);

        end_timer!(commit_time);
        Ok((Commitment(commitment.into()), randomness))
    }

    pub(crate) fn open_with_witness_polynomial_on_gpu<Fg: GpuPrimeField>(
        kern: &PolyKernel<Fg>,
        gpu_container: &mut GpuPolyContainer<Fg>,
        powers: &Powers<E>,
        point: E::Fr,
        randomness: &Randomness<E>,
        witness_polynomial_name: &str,
        hiding_witness_polynomial: Option<&DensePolynomial<E::Fr>>,
    ) -> Result<Proof<E>, PCError> {
        let witness_comm_time = start_timer!(|| "Computing commitment to witness polynomial");

        let gpu_poly = gpu_container.find(&kern, &witness_polynomial_name)?;

        Self::check_degree_is_too_large(gpu_poly.size(), powers.size())?;

        let mut w = VariableBase::msm_gpu_ptr_reuse_base(
            &powers.powers_of_beta_g,
            gpu_poly.get_memory(),
            0,
            GPU_OP::REUSE_G,
            true,
        );

        end_timer!(witness_comm_time);

        gpu_container.recycle(gpu_poly)?;

        let random_v = if let Some(hiding_witness_polynomial) = hiding_witness_polynomial {
            let blinding_p = &randomness.blinding_polynomial;
            let blinding_eval_time = start_timer!(|| "Evaluating random polynomial");
            let blinding_evaluation = blinding_p.evaluate(point);
            end_timer!(blinding_eval_time);

            let random_witness_coeffs = convert_to_bigints(&hiding_witness_polynomial.coeffs);
            let witness_comm_time = start_timer!(|| "Computing commitment to random witness polynomial");
            w += &VariableBase::msm(&powers.powers_of_beta_times_gamma_g, &random_witness_coeffs);
            end_timer!(witness_comm_time);
            Some(blinding_evaluation)
        } else {
            None
        };

        Ok(Proof { w: w.to_affine(), random_v })
    }

    /// On input a polynomial `p` and a point `point`, outputs a proof for the same.
    pub(crate) fn open_on_gpu<Fg: GpuPrimeField>(
        kern: &PolyKernel<Fg>,
        gpu_container: &mut GpuPolyContainer<Fg>,
        powers: &Powers<E>,
        point: E::Fr,
        rand: &Randomness<E>,
        name: &str,
    ) -> Result<Proof<E>, PCError> {
        // Self::check_degree_is_too_large(polynomial.degree(), powers.size())?;
        let open_time = start_timer!(|| format!("Opening polynomial"));

        let witness_time = start_timer!(|| "Computing witness polynomials");
        let (witness_poly_name, hiding_witness_poly) =
            Self::compute_witness_polynomial_on_gpu(kern, gpu_container, point, rand, name)?;
        end_timer!(witness_time);

        let proof = Self::open_with_witness_polynomial_on_gpu(
            kern,
            gpu_container,
            powers,
            point,
            rand,
            &witness_poly_name,
            hiding_witness_poly.as_ref(),
        );

        end_timer!(open_time);
        proof
    }

    #[allow(clippy::type_complexity)]
    pub fn compute_witness_polynomial_on_gpu<Fg: GpuPrimeField>(
        kern: &PolyKernel<Fg>,
        gpu_container: &mut GpuPolyContainer<Fg>,
        point: E::Fr,
        randomness: &Randomness<E>,
        name: &str,
    ) -> Result<(String, Option<DensePolynomial<E::Fr>>), PCError> {
        let divisor = DensePolynomial::from_coefficients_vec(vec![-point, E::Fr::one()]);

        let witness_polynomial_name = String::from("witness_polynomial");

        let witness_time = start_timer!(|| "Computing witness polynomial");

        let named_gpu_poly = gpu_container.find(&kern, name)?;

        let domain_n = EvaluationDomain::<E::Fr>::new(named_gpu_poly.size()).unwrap();
        let n = domain_n.size();
        let lg_n = log2_floor(n);

        let mut gpu_poly = gpu_container.ask_for(kern, n)?;
        let mut gpu_poly_2 = gpu_container.ask_for(kern, n)?;
        let mut gpu_powers = gpu_container.ask_for_powers(kern)?;
        let mut gpu_result = gpu_container.ask_for_results(kern)?;
        let mut gpu_buckets = gpu_container.ask_for_buckets(kern)?;

        // for FFT/iFFT
        let mut gpu_tmp_buffer = gpu_container.ask_for(kern, n)?;
        let gpu_pq = gpu_container.find(kern, &format!("domain_{n}_pq"))?;
        let gpu_omegas = gpu_container.find(kern, &format!("domain_{n}_omegas"))?;
        let gpu_pq_ifft = gpu_container.find(kern, &format!("domain_{n}_pq_ifft"))?;
        let gpu_omegas_ifft = gpu_container.find(kern, &format!("domain_{n}_omegas_ifft"))?;

        gpu_poly.fill_with_zero()?;
        gpu_poly.copy_from_gpu(&named_gpu_poly)?;

        let point_eval = gpu_poly.evaluate_at(&mut gpu_powers, &mut gpu_result, &point)?;

        gpu_poly.fft(&mut gpu_tmp_buffer, &gpu_pq, &gpu_omegas, lg_n)?;

        gpu_poly.sub_constant(&point_eval)?;

        let gpu_saved_powers = gpu_container.find(kern, &format!("domain_{n}_powers_group_gen"))?;
        gpu_poly_2.copy_from_gpu(&gpu_saved_powers)?;
        gpu_poly_2.sub_constant(&point)?;

        gpu_poly_2.batch_inversion(&mut gpu_result, &mut gpu_buckets)?;

        gpu_poly.mul_assign(&gpu_poly_2)?;

        gpu_poly.ifft(&mut gpu_tmp_buffer, &gpu_pq_ifft, &gpu_omegas_ifft, &domain_n.size_inv, lg_n)?;
        kern.sync()?;

        // don't forget to do cleaning!
        gpu_container.save(&witness_polynomial_name, gpu_poly)?;
        gpu_container.save(&format!("domain_{n}_powers_group_gen"), gpu_saved_powers)?;
        gpu_container.save(&format!("domain_{n}_pq"), gpu_pq)?;
        gpu_container.save(&format!("domain_{n}_omegas"), gpu_omegas)?;
        gpu_container.save(&format!("domain_{n}_pq_ifft"), gpu_pq_ifft)?;
        gpu_container.save(&format!("domain_{n}_omegas_ifft"), gpu_omegas_ifft)?;
        gpu_container.save(name, named_gpu_poly)?;

        gpu_container.recycle(gpu_poly_2)?;
        gpu_container.recycle(gpu_powers)?;
        gpu_container.recycle(gpu_result)?;
        gpu_container.recycle(gpu_buckets)?;
        gpu_container.recycle(gpu_tmp_buffer)?;

        end_timer!(witness_time);

        let random_witness_polynomial = if randomness.is_hiding() {
            let random_p = &randomness.blinding_polynomial;

            let witness_time = start_timer!(|| "Computing random witness polynomial");
            let random_witness_polynomial = random_p / &divisor;
            end_timer!(witness_time);
            Some(random_witness_polynomial)
        } else {
            None
        };

        Ok((witness_polynomial_name, random_witness_polynomial))
    }
}
