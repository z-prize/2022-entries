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
    fft::{
        domain::{FFTPrecomputation, IFFTPrecomputation},
        DensePolynomial, EvaluationDomain, Evaluations as EvaluationsOnDomain, SparsePolynomial,
    },
    polycommit::sonic_pc::{LabeledPolynomial, LabeledPolynomialWithBasis, PolynomialInfo, PolynomialWithBasis},
    snark::marlin::{
        ahp::{verifier, AHPError, AHPForR1CS},
        matrices::MatrixArithmetization,
        prover, witness_label, MarlinMode,
    },
};
use snarkvm_fields::PrimeField;
use snarkvm_utilities::{cfg_into_iter, test_rng};

use rand_core::RngCore;

use rayon::prelude::*;

use crate::snark::marlin::prover::round_functions::first::PoolResult;
use snarkvm_curves::AffineCurve;
use snarkvm_curves::PairingEngine;
use snarkvm_curves::ProjectiveCurve;
use std::str::FromStr;

use log::info;

use crate::snark::marlin::prover::FirstOracles;
use ec_gpu_common::{log2_floor, GPUResult, GpuPolyContainer, PolyKernel, PrimeField as GpuPrimeField, GPU_OP};

impl<F: PrimeField, MM: MarlinMode> AHPForR1CS<F, MM> {
    /// used for FFT domain preparation
    pub fn prepare_domain<Fg: GpuPrimeField>(
        kern: &PolyKernel<Fg>,
        gpu_container: &mut GpuPolyContainer<Fg>,
        domain_size: usize,
    ) -> GPUResult<()> {
        let domain = EvaluationDomain::<F>::new(domain_size).unwrap();
        let n = domain.size();
        let lg_n = log2_floor(n);

        let mut gpu_tmp_buffer = gpu_container.ask_for(&kern, n)?;

        let mut gpu_pq = gpu_container.ask_for_pq(&kern)?;
        let mut gpu_omegas = gpu_container.ask_for_omegas_with_size(&kern, n)?;
        let mut gpu_pq_ifft = gpu_container.ask_for_pq(&kern)?;
        let mut gpu_omegas_ifft = gpu_container.ask_for_omegas_with_size(&kern, n)?;

        let mut gpu_powers = gpu_container.ask_for_powers(&kern)?;
        let mut gpu_powers_inv = gpu_container.ask_for_powers(&kern)?;
        let mut gpu_domain_powers = gpu_container.ask_for(&kern, n)?;
        let mut gpu_domain_powers_group_gen = gpu_container.ask_for(&kern, n)?;
        let mut gpu_domain_powers_group_gen_inv = gpu_container.ask_for(&kern, n)?;
        let mut gpu_domain_powers_inv = gpu_container.ask_for(&kern, n)?;
        let mut gpu_domain_powers_neg = gpu_container.ask_for(&kern, n)?;

        let max_deg = std::cmp::min(ec_gpu_common::MAX_RADIX_DEGREE, lg_n);

        gpu_pq.setup_pq(&domain.group_gen, n, max_deg)?;
        gpu_omegas.setup_omegas_with_size(&domain.group_gen, n)?;

        gpu_pq_ifft.setup_pq(&domain.group_gen_inv, n, max_deg)?;
        gpu_omegas_ifft.setup_omegas_with_size(&domain.group_gen_inv, n)?;

        let mut gpu_result = gpu_container.ask_for_results(&kern)?;
        let mut gpu_buckets = gpu_container.ask_for_buckets(&kern)?;
        gpu_domain_powers.fill_with_fe(&F::one())?;
        gpu_domain_powers.distribute_powers(&mut gpu_powers, &F::multiplicative_generator(), 0)?;
        gpu_domain_powers_group_gen.generate_powers(&mut gpu_powers, &domain.group_gen)?;
        gpu_domain_powers_neg.generate_powers(&mut gpu_powers, &domain.group_gen)?;
        gpu_domain_powers_neg.negate()?;
        gpu_domain_powers_group_gen_inv.copy_from_gpu(&gpu_domain_powers_group_gen)?;
        gpu_domain_powers_group_gen_inv.batch_inversion(&mut gpu_result, &mut gpu_buckets)?;
        gpu_domain_powers_inv.copy_from_gpu(&gpu_domain_powers)?;
        gpu_domain_powers_inv.batch_inversion(&mut gpu_result, &mut gpu_buckets)?;

        let half_n = n / 2;
        let mut gpu_vanishing_on_coset_inv = gpu_container.ask_for(kern, n)?;
        gpu_vanishing_on_coset_inv.fill_with_zero()?;
        gpu_vanishing_on_coset_inv.add_at_offset(&F::one(), half_n)?;
        gpu_vanishing_on_coset_inv.add_at_offset(&-F::one(), 0)?;
        gpu_vanishing_on_coset_inv.mul_assign(&gpu_domain_powers)?;
        gpu_vanishing_on_coset_inv.fft(&mut gpu_tmp_buffer, &gpu_pq, &gpu_omegas, lg_n)?;
        gpu_vanishing_on_coset_inv.batch_inversion(&mut gpu_result, &mut gpu_buckets)?;

        gpu_container.save(&format!("domain_{n}_pq"), gpu_pq)?;
        gpu_container.save(&format!("domain_{n}_omegas"), gpu_omegas)?;
        gpu_container.save(&format!("domain_{n}_pq_ifft"), gpu_pq_ifft)?;
        gpu_container.save(&format!("domain_{n}_omegas_ifft"), gpu_omegas_ifft)?;
        gpu_container.save(&format!("domain_{n}_powers"), gpu_domain_powers)?;
        gpu_container.save(&format!("domain_{n}_powers_group_gen"), gpu_domain_powers_group_gen)?;
        gpu_container.save(&format!("domain_{n}_powers_group_gen_inv"), gpu_domain_powers_group_gen_inv)?;
        gpu_container.save(&format!("domain_{n}_powers_inv"), gpu_domain_powers_inv)?;
        gpu_container.save(&format!("domain_{n}_powers_neg"), gpu_domain_powers_neg)?;
        gpu_container.save(&format!("domain_{n}_vanishing_on_coset_inv"), gpu_vanishing_on_coset_inv)?;

        gpu_powers.setup_powers(&domain.group_gen)?;
        gpu_powers_inv.setup_powers(&domain.group_gen_inv)?;
        gpu_container.save(&format!("domain_{n}_omega_powers"), gpu_powers)?;
        gpu_container.save(&format!("domain_{n}_omega_inv_powers"), gpu_powers_inv)?;

        gpu_container.recycle(gpu_tmp_buffer)?;
        gpu_container.recycle(gpu_result)?;
        gpu_container.recycle(gpu_buckets)?;

        Ok(())
    }

    /// used for FFT domain preparation
    pub fn preload_arith_matrix<Fg: GpuPrimeField>(
        kern: &PolyKernel<Fg>,
        gpu_container: &mut GpuPolyContainer<Fg>,
        arith: &MatrixArithmetization<F>,
        label: &str,
    ) -> GPUResult<()> {
        let domain_k_size = arith.evals_on_K.row.domain().size();

        let mut gpu_save = gpu_container.ask_for(&kern, arith.row.degree() + 1)?;
        gpu_save.read_from(arith.row.as_dense().unwrap().coeffs())?;
        gpu_container.save(&("row_".to_owned() + label), gpu_save)?;

        let mut gpu_save = gpu_container.ask_for(&kern, arith.col.degree() + 1)?;
        gpu_save.read_from(arith.col.as_dense().unwrap().coeffs())?;
        gpu_container.save(&("col_".to_owned() + label), gpu_save)?;

        let mut gpu_save = gpu_container.ask_for(&kern, arith.row_col.degree() + 1)?;
        gpu_save.read_from(arith.row_col.as_dense().unwrap().coeffs())?;
        gpu_container.save(&("row_col_".to_owned() + label), gpu_save)?;

        let mut gpu_save = gpu_container.ask_for(&kern, arith.val.degree() + 1)?;
        gpu_save.read_from(arith.val.as_dense().unwrap().coeffs())?;
        gpu_container.save(&("val_".to_owned() + label), gpu_save)?;

        let mut gpu_save = gpu_container.ask_for(&kern, domain_k_size)?;
        gpu_save.fill_with_zero()?;
        gpu_save.read_from(arith.evals_on_K.row.evaluations.as_slice())?;
        gpu_container.save(&("evals_row_".to_owned() + label), gpu_save)?;

        let mut gpu_save = gpu_container.ask_for(&kern, domain_k_size)?;
        gpu_save.fill_with_zero()?;
        gpu_save.read_from(arith.evals_on_K.col.evaluations.as_slice())?;
        gpu_container.save(&("evals_col_".to_owned() + label), gpu_save)?;

        let mut gpu_save = gpu_container.ask_for(&kern, domain_k_size)?;
        gpu_save.fill_with_zero()?;
        gpu_save.read_from(arith.evals_on_K.row_col.evaluations.as_slice())?;
        gpu_container.save(&("evals_row_col_".to_owned() + label), gpu_save)?;

        let mut gpu_save = gpu_container.ask_for(&kern, domain_k_size)?;
        gpu_save.fill_with_zero()?;
        gpu_save.read_from(arith.evals_on_K.val.evaluations.as_slice())?;
        gpu_container.save(&("evals_val_".to_owned() + label), gpu_save)?;

        // precalc val on coset
        let domain_k = EvaluationDomain::<F>::new(arith.val.as_dense().unwrap().coeffs().len()).unwrap();
        let domain_k_size = domain_k.size();
        let domain_2k_size = domain_k_size * 2;
        let domain_2k_omega = F::get_root_of_unity(domain_2k_size).unwrap();

        let lg_k = log2_floor(domain_k_size);
        let mut gpu_save = gpu_container.ask_for(&kern, domain_k_size)?;
        let mut gpu_tmp_powers = gpu_container.ask_for_powers(kern)?;

        let mut gpu_k_tmp_buffer = gpu_container.ask_for(kern, domain_k_size)?;
        let gpu_k_pq = gpu_container.find(kern, &format!("domain_{domain_k_size}_pq"))?;
        let gpu_k_omegas = gpu_container.find(kern, &format!("domain_{domain_k_size}_omegas"))?;
        let gpu_k_pq_ifft = gpu_container.find(kern, &format!("domain_{domain_k_size}_pq_ifft"))?;
        let gpu_k_omegas_ifft = gpu_container.find(kern, &format!("domain_{domain_k_size}_omegas_ifft"))?;

        gpu_save.fill_with_zero()?;
        gpu_save.read_from(arith.val.as_dense().unwrap().coeffs())?;
        gpu_save.distribute_powers(&mut gpu_tmp_powers, &domain_2k_omega, 0)?;
        gpu_save.fft(&mut gpu_k_tmp_buffer, &gpu_k_pq, &gpu_k_omegas, lg_k)?;

        gpu_container.save(&format!("val_{label}_coset"), gpu_save)?;

        gpu_container.save(&format!("domain_{domain_k_size}_pq"), gpu_k_pq)?;
        gpu_container.save(&format!("domain_{domain_k_size}_omegas"), gpu_k_omegas)?;
        gpu_container.save(&format!("domain_{domain_k_size}_pq_ifft"), gpu_k_pq_ifft)?;
        gpu_container.save(&format!("domain_{domain_k_size}_omegas_ifft"), gpu_k_omegas_ifft)?;
        gpu_container.recycle(gpu_k_tmp_buffer)?;

        Ok(())
    }

    // omega_subs[i][j] = \product k=0..j (i!=k) (omega^i - omega^k)
    pub fn calc_all_omega_subs(domain_size: usize, omegas: &[F]) -> Vec<Vec<F>> {
        let res = (0..domain_size)
            .into_par_iter()
            .map(|i| -> Vec<F> {
                let mut omega_subs = vec![F::zero(); domain_size];

                let mut omega_acc = F::one();

                for j in 0..domain_size {
                    // set `omega^i - omega^i` to `one` so that we don't need explicitly pick out for mul acc
                    if i == j {
                        omega_subs[j] = omega_acc;
                    } else {
                        omega_acc = omega_acc * (omegas[i] - omegas[j]);
                        omega_subs[j] = omega_acc;
                    }
                }

                omega_subs
            })
            .collect::<Vec<Vec<_>>>();

        res
    }

    // assume that we have only up to `n` evaluations, `n` is less than a fft domain size
    // we need to compute all Lk(x), in order to do further commitment
    // to quickly compute Lk(x), we get the evaluations of Lk(x) on all omegas, then do an ifft to obtain Lk(x)
    // Lk(x) has the form "(x - omega^0)(x - omega^1).../(omega^k - omega^0)(omega^k - omega^1)..."
    // 2022.07.18: due to a stack-overflow issue, we put ifft part outside of this function
    pub fn calc_lk_poly_dp(k: usize, n: usize, domain_size: usize, omegas: &[F], omega_subs: &Vec<Vec<F>>) -> Vec<F> {
        let omegas_acc = omega_subs[k][n - 1];

        let divisor = omegas_acc.inverse().unwrap();

        let mut l_k_coeffs = vec![F::zero(); domain_size];

        for i in 0..domain_size {
            if i >= n || i == k {
                let inner_omega_acc = omega_subs[i][n - 1];

                let pick_out = if i != k {
                    let inv = omegas[i] - omegas[k];
                    let inv = inv.inverse().unwrap();
                    inv
                } else {
                    F::one()
                };

                l_k_coeffs[i] = inner_omega_acc * divisor * pick_out;
            }
        }

        // let domain: EvaluationDomain<F> = EvaluationDomain::new(domain_size).unwrap();
        // domain.ifft_in_place(&mut l_k_coeffs);

        l_k_coeffs
    }

    pub fn prepare_lagrange_bases<Fg: GpuPrimeField, E: PairingEngine>(
        kern: &PolyKernel<Fg>,
        gpu_container: &mut GpuPolyContainer<Fg>,
        powers: &[E::G1Affine],
        n: usize,
        domain_size: usize,
    ) -> GPUResult<()> {
        use crate::msm::VariableBase;
        use snarkvm_utilities::fs::File;
        use snarkvm_utilities::io::BufWriter;
        use snarkvm_utilities::path::PathBuf;
        use snarkvm_utilities::ToBytes;
        use snarkvm_utilities::Write;

        assert!(n <= domain_size);

        info!("lagrange n = {}, domain_size = {}, powers = {}", n, domain_size, powers.len());

        let domain = EvaluationDomain::<F>::new(domain_size).expect("cannot instantialize such domain");

        let mut omegas = vec![F::zero(); domain_size];

        for i in 0..domain_size {
            omegas[i] = domain.group_gen.pow(&[i as u64]);
        }

        let omega_subs = Self::calc_all_omega_subs(domain_size, &omegas);

        info!("all omega subs are calculated");

        let mut file = BufWriter::new(File::create(PathBuf::from("./lagrange_g_calced.data")).unwrap());

        file.write(&n.to_le_bytes()).unwrap();

        // REMINDER: these polys are in eval form, iffts are needed
        let lk_polys: Vec<_> = (0..n)
            .into_par_iter()
            .map(|i| -> Vec<F> { Self::calc_lk_poly_dp(i, n, domain_size, &omegas, &omega_subs) })
            .collect::<Vec<_>>();

        let mut gpu_poly = gpu_container.ask_for(kern, domain_size)?;
        let mut gpu_tmp_buffer = gpu_container.ask_for(kern, domain_size)?;
        let gpu_pq_ifft = gpu_container.find(kern, &format!("domain_{domain_size}_pq_ifft"))?;
        let gpu_omegas_ifft = gpu_container.find(kern, &format!("domain_{domain_size}_omegas_ifft"))?;

        let v: Vec<E::Fr> = Vec::new();
        let bases: Vec<E::G1Affine> = Vec::new();
        VariableBase::msm_gpu_reuse_base(&bases, v.as_slice(), 0, GPU_OP::SETUP_SHIFTED_G, false);

        let lg_domain = log2_floor(domain_size);
        let mut commitments = Vec::new();

        for i in 0..n {
            if i % 512 == 0 {
                info!("commiting for {}/{} poly", i, n);
            }

            gpu_poly.fill_with_zero()?;
            gpu_poly.read_from(lk_polys[i].as_slice())?;
            gpu_poly.ifft(&mut gpu_tmp_buffer, &gpu_pq_ifft, &gpu_omegas_ifft, &domain.size_inv, lg_domain)?;
            kern.sync()?;

            let commitment =
                VariableBase::msm_gpu_ptr_reuse_base(&powers, gpu_poly.get_memory(), 0, GPU_OP::REUSE_SHIFTED_G, true);
            let commitment = commitment.to_affine();
            commitment.write_le(&mut file).unwrap();
            commitments.push(commitment);
        }

        // use the above lagrange bases to do a small test
        let mut gpu_coeffs = gpu_container.ask_for(kern, n)?;
        gpu_coeffs.fill_with_zero()?;

        {
            let mut rng = test_rng();

            // we have a poly that has `n` coeffs
            let coeffs = (0..n).into_iter().map(|_| F::rand(&mut rng)).collect::<Vec<_>>();

            gpu_coeffs.read_from(coeffs.as_slice())?;

            let coeffs_commitment = VariableBase::msm_gpu_ptr_reuse_base(
                &powers,
                gpu_coeffs.get_memory(),
                0,
                GPU_OP::REUSE_SHIFTED_G,
                true,
            );

            let mut gpu_powers = gpu_container.ask_for_powers(kern)?;
            let mut gpu_results = gpu_container.ask_for_results(kern)?;

            let mut evals = Vec::new();
            for i in 0..n {
                if i % 512 == 0 {
                    info!("evaluating for {}/{} poly", i, n);
                }

                let eval = gpu_coeffs.evaluate_at(&mut gpu_powers, &mut gpu_results, &omegas[i])?;
                evals.push(eval);
            }

            let evals_commitment =
                VariableBase::msm_gpu_pippenger_scalar(commitments.as_slice(), evals.as_slice(), true);

            gpu_container.recycle(gpu_powers)?;
            gpu_container.recycle(gpu_results)?;

            assert_eq!(evals_commitment, coeffs_commitment);
        }

        // don't forget to do cleaning
        gpu_container.recycle(gpu_tmp_buffer)?;
        gpu_container.recycle(gpu_poly)?;
        gpu_container.recycle(gpu_coeffs)?;
        gpu_container.save(&format!("domain_{domain_size}_pq_ifft"), gpu_pq_ifft)?;
        gpu_container.save(&format!("domain_{domain_size}_omegas_ifft"), gpu_omegas_ifft)?;

        Ok(())
    }

    pub fn commit_lagrange_tail<Fg: GpuPrimeField, E: PairingEngine>(
        _kern: &PolyKernel<Fg>,
        _gpu_container: &mut GpuPolyContainer<Fg>,
        start: usize,
        _domain_size: usize,
    ) -> GPUResult<E::G1Projective> {
        // let mut gpu_test_head = gpu_container.ask_for(&kern, start)?;
        // let mut gpu_test_tail = gpu_container.ask_for(&kern, domain_size - 1)?;
        // let gpu_inv = gpu_container.find(&kern, &format!("domain_{domain_size}_powers_group_gen_inv"))?;

        // gpu_test_head.fill_with_fe(&F::zero());
        // gpu_test_tail.fill_with_fe(&-F::one());
        // gpu_test_tail.copy_from_gpu(&gpu_test_head);
        // gpu_test_tail.mul_assign(&gpu_inv);

        // let commit_g_a = VariableBase::msm_gpu_ptr_reuse_base::<E::G1Affine, Fg>(
        //     &[],
        //     gpu_test_tail.get_memory(),
        //     0,
        //     GPU_OP::REUSE_SHIFTED_LAGRANGE_G,
        //     true,
        // );

        // println!("start = {start}, commit_g_a = {}", commit_g_a);

        // gpu_container.recycle(gpu_test_head);
        // gpu_container.recycle(gpu_test_tail);
        // gpu_container.save(&format!("domain_{domain_size}_powers_group_gen_inv"), gpu_inv);

        let (coordinates, greatest) = match start {
            37908 => (E::Fq::from_str("36483310698954439716662709669692332880776825219837852409909653369267225406350598525325470920135050727479567693976").unwrap(), true),
            38031 => (E::Fq::from_str("211642613627364241482069853627786082983910635208271756081882538664665531260336314406567420630912450984192516331671").unwrap(), true),
            48434 => (E::Fq::from_str("93248758203246509815658837719327991621580805733012498790677405630803908650159556378357366907933902852321217308914").unwrap(), true),
            _ => panic!("invalid tail"),
        };

        let commit_g_a =
            <E::G1Affine as AffineCurve>::from_x_coordinate(coordinates, greatest).unwrap().to_projective();

        Ok(commit_g_a)
    }
}

/// GPU proving first round
impl<F: PrimeField, MM: MarlinMode> AHPForR1CS<F, MM> {
    /// Output the first round message and the next state.
    #[allow(clippy::type_complexity)]
    pub fn prover_first_round_on_gpu<'a, 'b, R: RngCore, Fg: GpuPrimeField>(
        kern: &PolyKernel<Fg>,
        gpu_container: &mut GpuPolyContainer<Fg>,
        state: &mut prover::State<'a, F, MM>,
        rng: &mut R,
    ) -> Result<FirstOracles<'b, F>, AHPError> {
        let round_time = start_timer!(|| "AHP::Prover::FirstRound");
        let constraint_domain = state.constraint_domain;
        let batch_size = state.batch_size;

        let z_a = state.z_a.take().unwrap();
        let z_b = state.z_b.take().unwrap();
        let private_variables = core::mem::take(&mut state.private_variables);
        assert_eq!(z_a.len(), batch_size);
        assert_eq!(z_b.len(), batch_size);
        assert_eq!(private_variables.len(), batch_size);
        let mut r_b_s = Vec::with_capacity(batch_size);

        let mut batches = Vec::new();

        let state_ref = &state;
        for (i, (z_a, z_b, private_variables, x_poly)) in
            itertools::izip!(z_a, z_b, private_variables, &state.x_poly).enumerate()
        {
            let w_inner = Self::calculate_w_on_gpu(
                kern,
                gpu_container,
                witness_label("w", i),
                private_variables,
                x_poly,
                state_ref,
            )?;
            let z_a_inner =
                Self::calculate_z_m_on_gpu(kern, gpu_container, witness_label("z_a", i), z_a, false, state_ref, None)?;
            let r_b = F::rand(rng);
            let z_b_inner = Self::calculate_z_m_on_gpu(
                kern,
                gpu_container,
                witness_label("z_b", i),
                z_b,
                true,
                state_ref,
                Some(r_b),
            )?;
            if MM::ZK {
                r_b_s.push(r_b);
            }

            let w_poly = w_inner.witness().unwrap();
            let (_, z_a_poly, z_a_evals) = z_a_inner.z_m().unwrap();
            let (_, z_b_poly, z_b_evals) = z_b_inner.z_m().unwrap();

            let mut gpu_w = gpu_container.ask_for(kern, w_poly.polynomial().degree() + 1)?;
            gpu_w.read_from(w_poly.polynomial().as_dense().unwrap())?;
            gpu_container.save(w_poly.label(), gpu_w)?;

            let mut gpu_za = gpu_container.ask_for(kern, z_a_poly.polynomial().degree() + 1)?;
            gpu_za.read_from(z_a_poly.polynomial().as_dense().unwrap())?;
            gpu_container.save(z_a_poly.label(), gpu_za)?;

            let mut gpu_zb = gpu_container.ask_for(kern, z_b_poly.polynomial().degree() + 1)?;
            gpu_zb.read_from(z_b_poly.polynomial().as_dense().unwrap())?;
            gpu_container.save(z_b_poly.label(), gpu_zb)?;

            batches.push(prover::SingleEntry { z_a: z_a_evals, z_b: z_b_evals, w_poly, z_a_poly, z_b_poly });
        }
        kern.sync()?;

        let mask_poly = Self::calculate_mask_poly(constraint_domain, rng);

        let oracles = prover::FirstOracles { batches, mask_poly };

        assert!(oracles.matches_info(&Self::first_round_polynomial_info(batch_size)));

        state.mz_poly_randomizer = MM::ZK.then(|| r_b_s);
        end_timer!(round_time);

        Ok(oracles)
    }

    #[allow(clippy::type_complexity)]
    pub fn prover_first_round_on_gpu_saved<'a, 'b, R: RngCore, Fg: GpuPrimeField>(
        kern: &PolyKernel<Fg>,
        gpu_container: &mut GpuPolyContainer<Fg>,
        state: &mut prover::State<'a, F, MM>,
        rng: &mut R,
        first_time: bool,
        saved_rand: &mut F,
        saved_first_oracles: &mut Vec<prover::MyFirstOracles<F>>,
    ) -> Result<(FirstOracles<'b, F>, Vec<prover::MyFirstOracles<F>>), AHPError> {
        let round_time = start_timer!(|| "AHP::Prover::FirstRound");
        let constraint_domain = state.constraint_domain;
        let batch_size = state.batch_size;

        let rand = F::rand(rng);
        // *saved_rand = rand.clone();
        *saved_rand = *saved_rand + rand.clone();

        if !first_time {
            let hiding_bound = if MM::ZK { Some(1) } else { None };
            //TODO we already know that only one batch is used for POSW
            saved_first_oracles[0].z_a_poly[0] -= rand;
            saved_first_oracles[0].z_a_poly[constraint_domain.size()] += rand;

            let w_poly = &saved_first_oracles[0].w_poly;
            let z_a_poly = &saved_first_oracles[0].z_a_poly;
            let z_b_poly = &saved_first_oracles[0].z_b_poly;

            let gpu_za_name = witness_label("z_a", 0);
            let mut gpu_za = gpu_container.find(kern, &gpu_za_name)?;

            gpu_za.add_at_offset(&(rand), constraint_domain.size())?;
            gpu_za.add_at_offset(&(-rand), 0)?;

            gpu_container.save(&gpu_za_name, gpu_za)?;

            let mask_poly = Self::calculate_mask_poly(constraint_domain, rng);

            let mut batches = Vec::new();

            batches.push(prover::SingleEntry {
                z_a: LabeledPolynomialWithBasis {
                    polynomial: vec![],
                    info: PolynomialInfo::new(witness_label("z_a", 0), None, hiding_bound),
                },
                z_b: LabeledPolynomialWithBasis {
                    polynomial: vec![],
                    info: PolynomialInfo::new(witness_label("z_b", 0), None, hiding_bound),
                },
                w_poly: w_poly.clone(),
                z_a_poly: LabeledPolynomial::new(witness_label("z_a", 0), z_a_poly.clone(), None, hiding_bound),
                z_b_poly: LabeledPolynomial::new(witness_label("z_b", 0), z_b_poly.clone(), None, hiding_bound),
            });

            let oracles = prover::FirstOracles { batches, mask_poly };

            end_timer!(round_time);
            return Ok((oracles, Vec::new()));
        }

        let z_a = state.z_a.take().unwrap();
        let z_b = state.z_b.take().unwrap();
        let private_variables = core::mem::take(&mut state.private_variables);
        assert_eq!(z_a.len(), batch_size);
        assert_eq!(z_b.len(), batch_size);
        assert_eq!(private_variables.len(), batch_size);
        let mut r_b_s = Vec::with_capacity(batch_size);

        let mut batches = Vec::new();
        let mut saved_batches = Vec::new();

        let state_ref = &state;
        for (i, (z_a, z_b, private_variables, x_poly)) in
            itertools::izip!(z_a, z_b, private_variables, &state.x_poly).enumerate()
        {
            let w_inner = Self::calculate_w_on_gpu(
                kern,
                gpu_container,
                witness_label("w", i),
                private_variables,
                x_poly,
                state_ref,
            )?;
            let z_a_inner = Self::calculate_z_m_on_gpu(
                kern,
                gpu_container,
                witness_label("z_a", i),
                z_a.clone(),
                false,
                state_ref,
                Some(rand),
            )?;
            let r_b = F::rand(rng);
            let z_b_inner = Self::calculate_z_m_on_gpu(
                kern,
                gpu_container,
                witness_label("z_b", i),
                z_b.clone(),
                true,
                state_ref,
                Some(r_b),
            )?;
            if MM::ZK {
                r_b_s.push(r_b);
            }

            let w_poly = w_inner.witness().unwrap();
            let (z_a_d, z_a_poly, z_a_evals) = z_a_inner.z_m().unwrap();
            let (z_b_d, z_b_poly, z_b_evals) = z_b_inner.z_m().unwrap();

            let mut gpu_w = gpu_container.ask_for(kern, w_poly.polynomial().degree() + 1)?;
            gpu_w.read_from(w_poly.polynomial().as_dense().unwrap())?;
            gpu_container.save(w_poly.label(), gpu_w)?;

            let mut gpu_za = gpu_container.ask_for(kern, z_a_poly.polynomial().degree() + 1)?;
            gpu_za.read_from(z_a_poly.polynomial().as_dense().unwrap())?;
            gpu_container.save(z_a_poly.label(), gpu_za)?;

            let mut gpu_zb = gpu_container.ask_for(kern, z_b_poly.polynomial().degree() + 1)?;
            gpu_zb.read_from(z_b_poly.polynomial().as_dense().unwrap())?;
            gpu_container.save(z_b_poly.label(), gpu_zb)?;

            saved_batches.push(prover::MyFirstOracles {
                w_poly: w_poly.clone(),
                z_a_poly: z_a_d.clone(),
                z_a_evals: EvaluationsOnDomain::from_vec_and_domain(z_a.clone(), constraint_domain),
                z_b_poly: z_b_d.clone(),
                z_b_evals: EvaluationsOnDomain::from_vec_and_domain(z_b.clone(), constraint_domain),
            });

            batches.push(prover::SingleEntry { z_a: z_a_evals, z_b: z_b_evals, w_poly, z_a_poly, z_b_poly });
        }
        kern.sync()?;

        let mask_poly = Self::calculate_mask_poly(constraint_domain, rng);

        let oracles = prover::FirstOracles { batches, mask_poly };
        assert!(oracles.matches_info(&Self::first_round_polynomial_info(batch_size)));

        state.mz_poly_randomizer = MM::ZK.then(|| r_b_s);
        end_timer!(round_time);

        Ok((oracles, saved_batches))
    }

    fn calculate_w_on_gpu<'a, 'b, Fg: GpuPrimeField>(
        kern: &PolyKernel<Fg>,
        gpu_container: &mut GpuPolyContainer<Fg>,
        label: String,
        private_variables: Vec<F>,
        x_poly: &DensePolynomial<F>,
        state: &prover::State<'a, F, MM>,
    ) -> Result<PoolResult<'b, F>, AHPError> {
        let constraint_domain = state.constraint_domain;
        let input_domain = state.input_domain;

        let mut w_extended = private_variables;
        let ratio = constraint_domain.size() / input_domain.size();
        w_extended.resize(constraint_domain.size() - input_domain.size(), F::zero());

        let n = constraint_domain.size();
        let lgn = log2_floor(n);

        let mut gpu_x_evals = gpu_container.ask_for(&kern, n)?;
        let mut tmp_buf = gpu_container.ask_for(&kern, n)?;
        let pq_buf = gpu_container.find(kern, &format!("domain_{n}_pq"))?;
        let omega_buf = gpu_container.find(kern, &format!("domain_{n}_omegas"))?;
        let ipq_buf = gpu_container.find(kern, &format!("domain_{n}_pq_ifft"))?;
        let iomega_buf = gpu_container.find(kern, &format!("domain_{n}_omegas_ifft"))?;
        gpu_x_evals.fill_with_zero()?;
        gpu_x_evals.read_from(x_poly.coeffs())?;
        gpu_x_evals.fft(&mut tmp_buf, &pq_buf, &omega_buf, lgn)?;

        let mut x_evals = vec![F::zero(); n];
        gpu_x_evals.write_to(x_evals.as_mut_slice())?;

        let w_poly_time = start_timer!(|| "Computing w polynomial");
        let w_poly_evals: Vec<F> = cfg_into_iter!(0..constraint_domain.size())
            .map(|k| match k % ratio {
                0 => F::zero(),
                _ => w_extended[k - (k / ratio) - 1] - x_evals[k],
            })
            .collect();

        let mut gpu_w_poly = gpu_container.ask_for(&kern, n)?;
        let mut w_poly = vec![F::zero(); n];
        gpu_w_poly.read_from(w_poly_evals.as_slice())?;
        gpu_w_poly.ifft(&mut tmp_buf, &ipq_buf, &iomega_buf, &constraint_domain.size_inv, lgn)?;
        gpu_w_poly.write_to(w_poly.as_mut_slice())?;

        let w_poly = DensePolynomial::from_coefficients_vec(w_poly);

        let (w_poly, remainder) = w_poly.divide_by_vanishing_poly(input_domain).unwrap();
        assert!(remainder.is_zero());

        // don't forget to do cleaning!
        gpu_container.recycle(gpu_w_poly)?;
        gpu_container.recycle(gpu_x_evals)?;
        gpu_container.recycle(tmp_buf)?;
        gpu_container.save(&format!("domain_{n}_pq"), pq_buf)?;
        gpu_container.save(&format!("domain_{n}_pq_ifft"), ipq_buf)?;
        gpu_container.save(&format!("domain_{n}_omegas"), omega_buf)?;
        gpu_container.save(&format!("domain_{n}_omegas_ifft"), iomega_buf)?;

        assert!(w_poly.degree() < constraint_domain.size() - input_domain.size());
        end_timer!(w_poly_time);

        Ok(PoolResult::Witness(LabeledPolynomial::new(label, w_poly, None, Self::zk_bound())))
    }

    fn calculate_z_m_on_gpu<'a, 'b, Fg: GpuPrimeField>(
        kern: &PolyKernel<Fg>,
        gpu_container: &mut GpuPolyContainer<Fg>,
        label: impl ToString,
        evaluations: Vec<F>,
        will_be_evaluated: bool,
        state: &prover::State<'a, F, MM>,
        r: Option<F>,
    ) -> Result<PoolResult<'b, F>, AHPError> {
        let constraint_domain = state.constraint_domain;
        let v_H = constraint_domain.vanishing_polynomial();
        let should_randomize = MM::ZK && will_be_evaluated;
        let label = label.to_string();
        let poly_time = start_timer!(|| format!("Computing {label}"));

        let n = constraint_domain.size();
        let lgn = log2_floor(n);

        let evals = EvaluationsOnDomain::from_vec_and_domain(evaluations, constraint_domain);

        let mut gpu_poly = gpu_container.ask_for(&kern, n)?;
        let mut tmp_buf = gpu_container.ask_for(&kern, n)?;
        let ipq_buf = gpu_container.find(kern, &format!("domain_{n}_pq_ifft"))?;
        let iomega_buf = gpu_container.find(kern, &format!("domain_{n}_omegas_ifft"))?;
        gpu_poly.fill_with_zero()?;
        gpu_poly.read_from(evals.evaluations.as_slice())?;
        gpu_poly.ifft(&mut tmp_buf, &ipq_buf, &iomega_buf, &constraint_domain.size_inv, lgn)?;

        if should_randomize {
            // poly += &(&v_H * r.unwrap());
            let mut gpu_vanishing = gpu_container.ask_for(&kern, n + 1)?;
            gpu_vanishing.fill_with_zero()?;
            gpu_vanishing.add_at_offset(&-r.unwrap(), 0)?;
            gpu_vanishing.add_at_offset(&r.unwrap(), n + 1)?;
            gpu_vanishing.add_assign(&gpu_poly)?;

            std::mem::swap(&mut gpu_poly, &mut gpu_vanishing);
            gpu_container.recycle(gpu_vanishing)?;
        }

        let mut n_v = n;
        if !will_be_evaluated {
            //z_a
            n_v = n + 1;
        }

        let mut poly = vec![F::zero(); n_v];
        gpu_poly.write_to(poly.as_mut_slice())?;

        if !will_be_evaluated {
            //z_a
            poly[0] -= r.unwrap();
            poly[constraint_domain.size()] += r.unwrap();
        }

        let poly = DensePolynomial::from_coefficients_vec(poly);

        let poly_for_opening = LabeledPolynomial::new(label.to_string(), poly.clone(), None, Self::zk_bound());

        // // Our polynomials are stored in GPU buffer
        // if should_randomize {
        //     assert!(poly_for_opening.degree() < constraint_domain.size() + Self::zk_bound().unwrap());
        // } else {
        //     assert!(poly_for_opening.degree() < constraint_domain.size());
        // }

        let poly_for_committing = if should_randomize {
            let poly_terms = vec![
                (F::one(), PolynomialWithBasis::new_lagrange_basis(evals)),
                (F::one(), PolynomialWithBasis::new_sparse_monomial_basis(&v_H * r.unwrap(), None)),
            ];
            LabeledPolynomialWithBasis::new_linear_combination(label, poly_terms, Self::zk_bound())
        } else {
            if !will_be_evaluated {
                let poly_terms = vec![
                    (F::one(), PolynomialWithBasis::new_lagrange_basis(evals)),
                    (F::one(), PolynomialWithBasis::new_sparse_monomial_basis(&v_H * r.unwrap(), None)),
                ];
                LabeledPolynomialWithBasis::new_linear_combination(label, poly_terms, Self::zk_bound())
            } else {
                LabeledPolynomialWithBasis::new_lagrange_basis(label, evals, Self::zk_bound())
            }
        };

        end_timer!(poly_time);

        // don't forget to do cleaning!
        gpu_container.recycle(gpu_poly)?;
        gpu_container.recycle(tmp_buf)?;
        gpu_container.save(&format!("domain_{n}_pq_ifft"), ipq_buf)?;
        gpu_container.save(&format!("domain_{n}_omegas_ifft"), iomega_buf)?;

        Ok(PoolResult::MatrixPoly(poly.clone(), poly_for_opening, poly_for_committing))
    }
}

/// GPU proving second round
impl<F: PrimeField, MM: MarlinMode> AHPForR1CS<F, MM> {
    /// Output the second round message and the next state.
    pub fn prover_second_round_on_gpu<'a, 'b, R: RngCore, Fg: GpuPrimeField>(
        kern: &PolyKernel<Fg>,
        gpu_container: &mut GpuPolyContainer<Fg>,
        first_round_oracles: &FirstOracles<'b, F>,
        verifier_message: &verifier::FirstMessage<F>,
        mut state: prover::State<'a, F, MM>,
        _r: &mut R,
        first_time: bool,
        saved_rand: &F,
    ) -> Result<(prover::SecondOracles<F>, prover::State<'a, F, MM>), AHPError> {
        let round_time = start_timer!(|| "AHP::Prover::SecondRound");

        let constraint_domain = state.constraint_domain;
        let zk_bound = Self::zk_bound();

        let verifier::FirstMessage { alpha, eta_b, eta_c, batch_combiners } = verifier_message;

        let (_, t) = Self::calculate_summed_z_m_and_t_on_gpu(
            kern,
            gpu_container,
            &state,
            first_round_oracles,
            *alpha,
            *eta_b,
            *eta_c,
            batch_combiners,
            first_time,
            saved_rand,
        )?;

        let n = constraint_domain.size();

        let z_time = start_timer!(|| "Compute z poly");

        if first_time {
            let mut gpu_z = gpu_container.ask_for(&kern, n)?;
            let mut gpu_x_poly = gpu_container.ask_for(&kern, n)?;
            let mut gpu_z_summed = gpu_container.ask_for(&kern, n + state.input_domain.size())?;
            let mut gpu_z_mul_vanishing = gpu_container.ask_for(&kern, n + state.input_domain.size())?;

            gpu_z.fill_with_zero()?;
            gpu_x_poly.fill_with_zero()?;
            gpu_z_summed.fill_with_zero()?;
            gpu_z_mul_vanishing.fill_with_zero()?;

            for (((i, _b), &coeff), x_poly) in
                first_round_oracles.batches.iter().enumerate().zip(batch_combiners.iter()).zip(state.x_poly.iter())
            {
                let gpu_w = gpu_container.find(kern, &witness_label("w", i))?;
                gpu_z.copy_from_gpu(&gpu_w)?;
                gpu_container.save(&witness_label("w", i), gpu_w)?;

                gpu_z_mul_vanishing.assign_mul_vanishing(&gpu_z, state.input_domain.size())?;

                gpu_x_poly.read_from(x_poly.coeffs())?;
                gpu_z_mul_vanishing.add_assign(&gpu_x_poly)?;
                gpu_z_mul_vanishing.scale(&coeff)?;

                gpu_z_summed.add_assign(&gpu_z_mul_vanishing)?;
                kern.sync()?;
            }

            // don't forget to do cleaning!
            gpu_container.recycle(gpu_z)?;
            gpu_container.recycle(gpu_x_poly)?;
            gpu_container.recycle(gpu_z_mul_vanishing)?;
            gpu_container.save("gpu_z_summed", gpu_z_summed)?;
        }

        end_timer!(z_time);

        let sumcheck_lhs = Self::calculate_lhs_on_gpu(kern, gpu_container, &state, first_round_oracles, t, first_time)?;

        debug_assert!(sumcheck_lhs
            .evaluate_over_domain_by_ref(constraint_domain)
            .evaluations
            .into_iter()
            .sum::<F>()
            .is_zero());

        let sumcheck_time = start_timer!(|| "Compute sumcheck h and g polys");

        let (h_1, x_g_1) = sumcheck_lhs.divide_by_vanishing_poly(constraint_domain).unwrap();
        let g_1 = DensePolynomial::from_coefficients_slice(&x_g_1.coeffs[1..]);
        drop(x_g_1);
        end_timer!(sumcheck_time);

        assert!(g_1.degree() <= constraint_domain.size() - 2);
        assert!(h_1.degree() <= 2 * constraint_domain.size() + 2 * zk_bound.unwrap_or(0) - 2);

        let mut gpu_g_1 = gpu_container.ask_for(kern, g_1.coeffs().len())?;
        gpu_g_1.read_from(g_1.coeffs())?;
        gpu_container.save("g_1", gpu_g_1)?;

        let mut gpu_h_1 = gpu_container.ask_for(kern, h_1.coeffs().len())?;
        gpu_h_1.read_from(h_1.coeffs())?;
        gpu_container.save("h_1", gpu_h_1)?;
        kern.sync()?;

        let oracles = prover::SecondOracles {
            g_1: LabeledPolynomial::new("g_1".into(), g_1, Some(constraint_domain.size() - 2), zk_bound),
            h_1: LabeledPolynomial::new("h_1".into(), h_1, None, None),
        };
        assert!(oracles.matches_info(&Self::second_round_polynomial_info(&state.index.index_info)));
        state.verifier_first_message = Some(verifier_message.clone());
        end_timer!(round_time);

        Ok((oracles, state))
    }

    fn calculate_lhs_on_gpu<'b, Fg: GpuPrimeField>(
        kern: &PolyKernel<Fg>,
        gpu_container: &mut GpuPolyContainer<Fg>,
        state: &prover::State<F, MM>,
        first_round_oracles: &FirstOracles<'b, F>,
        t: DensePolynomial<F>,
        first_time: bool,
    ) -> Result<DensePolynomial<F>, AHPError> {
        let constraint_domain = state.constraint_domain;
        let q_1_time = start_timer!(|| "Compute LHS of sumcheck");

        let mask_poly = first_round_oracles.mask_poly.as_ref();
        assert_eq!(MM::ZK, mask_poly.is_some());

        let gpu_summed_z_m = gpu_container.find(&kern, "gpu_summed_z_m")?;
        let gpu_z_summed = gpu_container.find(&kern, "gpu_z_summed")?;

        let mul_domain_size =
            (constraint_domain.size() + gpu_summed_z_m.size()).max(t.coeffs.len() + gpu_z_summed.size());

        let mul_domain =
            EvaluationDomain::<F>::new(mul_domain_size).expect("field is not smooth enough to construct domain");

        let n = mul_domain.size();
        let lg_n = log2_floor(n);

        let mut gpu_tmp_buffer = gpu_container.ask_for(&kern, n)?;
        let gpu_pq = gpu_container.find(&kern, &format!("domain_{n}_pq"))?;
        let gpu_omegas = gpu_container.find(&kern, &format!("domain_{n}_omegas"))?;
        let gpu_pq_ifft = gpu_container.find(&kern, &format!("domain_{n}_pq_ifft"))?;
        let gpu_omegas_ifft = gpu_container.find(&kern, &format!("domain_{n}_omegas_ifft"))?;

        let mut gpu_summed_z_m_mul_domain = gpu_container.ask_for(&kern, n)?;
        gpu_summed_z_m_mul_domain.fill_with_zero()?;
        gpu_summed_z_m_mul_domain.copy_from_gpu(&gpu_summed_z_m)?;
        gpu_summed_z_m_mul_domain.fft(&mut gpu_tmp_buffer, &gpu_pq, &gpu_omegas, lg_n)?;

        let gpu_z_mul_domain = if first_time {
            let mut gpu_z_mul_domain = gpu_container.ask_for(&kern, n)?;
            gpu_z_mul_domain.fill_with_zero()?;
            gpu_z_mul_domain.copy_from_gpu(&gpu_z_summed)?;
            gpu_z_mul_domain.fft(&mut gpu_tmp_buffer, &gpu_pq, &gpu_omegas, lg_n)?;
            gpu_z_mul_domain
        } else {
            gpu_container.find(&kern, "gpu_z_mul_domain")?
        };

        let mut gpu_t_mul_domain = gpu_container.ask_for(&kern, n)?;
        let gpu_t = gpu_container.find(&kern, "gpu_t")?;
        gpu_t_mul_domain.fill_with_zero()?;
        gpu_t_mul_domain.copy_from_gpu(&gpu_t)?;
        gpu_t_mul_domain.fft(&mut gpu_tmp_buffer, &gpu_pq, &gpu_omegas, lg_n)?;

        let mut gpu_r_alpha_x_mul_domain = gpu_container.ask_for(kern, n)?;
        let gpu_r_alpha_x_coeffs = gpu_container.find(kern, "gpu_r_alpha_x_coeffs_domain_h")?;

        gpu_r_alpha_x_mul_domain.fill_with_zero()?;
        gpu_r_alpha_x_mul_domain.copy_from_gpu(&gpu_r_alpha_x_coeffs)?;

        gpu_r_alpha_x_mul_domain.fft(&mut gpu_tmp_buffer, &gpu_pq, &gpu_omegas, lg_n)?;

        gpu_r_alpha_x_mul_domain.mul_assign(&gpu_summed_z_m_mul_domain)?;
        gpu_tmp_buffer.copy_from_gpu(&gpu_z_mul_domain)?;
        gpu_tmp_buffer.mul_assign(&gpu_t_mul_domain)?;
        gpu_r_alpha_x_mul_domain.sub_assign(&gpu_tmp_buffer)?;
        gpu_r_alpha_x_mul_domain.ifft(
            &mut gpu_tmp_buffer,
            &gpu_pq_ifft,
            &gpu_omegas_ifft,
            &mul_domain.size_inv,
            lg_n,
        )?;

        let mut lhs = vec![F::zero(); n];
        gpu_r_alpha_x_mul_domain.write_to(lhs.as_mut_slice())?;
        let mut lhs = DensePolynomial::from_coefficients_vec(lhs);
        lhs += &mask_poly.map_or(SparsePolynomial::zero(), |p| p.polynomial().as_sparse().unwrap().clone());

        // don't forget to do cleaning!
        gpu_container.save(&format!("domain_{n}_pq"), gpu_pq)?;
        gpu_container.save(&format!("domain_{n}_omegas"), gpu_omegas)?;
        gpu_container.save(&format!("domain_{n}_pq_ifft"), gpu_pq_ifft)?;
        gpu_container.save(&format!("domain_{n}_omegas_ifft"), gpu_omegas_ifft)?;

        gpu_container.recycle(gpu_tmp_buffer)?;
        gpu_container.recycle(gpu_summed_z_m)?;
        gpu_container.recycle(gpu_summed_z_m_mul_domain)?;
        //gpu_container.recycle(gpu_z_mul_domain)?;
        // gpu_container.recycle(gpu_z_summed)?;

        gpu_container.save("gpu_z_summed", gpu_z_summed)?;
        gpu_container.save("gpu_z_mul_domain", gpu_z_mul_domain)?;
        gpu_container.recycle(gpu_t)?;
        gpu_container.recycle(gpu_t_mul_domain)?;
        gpu_container.recycle(gpu_r_alpha_x_mul_domain)?;
        gpu_container.recycle(gpu_r_alpha_x_coeffs)?;

        end_timer!(q_1_time);
        Ok(lhs)
    }

    fn calculate_summed_z_m_and_t_on_gpu<'b, Fg: GpuPrimeField>(
        kern: &PolyKernel<Fg>,
        gpu_container: &mut GpuPolyContainer<Fg>,
        state: &prover::State<F, MM>,
        first_round_oracles: &FirstOracles<'b, F>,
        alpha: F,
        eta_b: F,
        eta_c: F,
        batch_combiners: &[F],
        first_time: bool,
        saved_rand: &F,
    ) -> Result<(DensePolynomial<F>, DensePolynomial<F>), AHPError> {
        let constraint_domain = state.constraint_domain;
        let summed_z_m_poly_time = start_timer!(|| "Compute z_m poly");

        let first_msg = first_round_oracles;

        let n = constraint_domain.size();
        let n_2 = n * 2;
        let domain_2k = EvaluationDomain::<F>::new(n_2).unwrap();
        let lg_2k = log2_floor(n_2);

        let mut gpu_2k_tmp_buffer = gpu_container.ask_for(&kern, n_2)?;
        let gpu_2k_pq = gpu_container.find(kern, &format!("domain_{n_2}_pq"))?;
        let gpu_2k_omegas = gpu_container.find(kern, &format!("domain_{n_2}_omegas"))?;
        let gpu_2k_pq_ifft = gpu_container.find(kern, &format!("domain_{n_2}_pq_ifft"))?;
        let gpu_2k_omegas_ifft = gpu_container.find(kern, &format!("domain_{n_2}_omegas_ifft"))?;

        let mut gpu_summed_z_m = gpu_container.ask_for(&kern, n_2)?;
        let mut gpu_local_summed_z_m = gpu_container.ask_for(&kern, n_2)?;
        let mut gpu_z_b_tmp = gpu_container.ask_for(&kern, n_2)?;

        gpu_summed_z_m.fill_with_zero()?;

        for ((i, _entry), combiner) in first_msg.batches.iter().enumerate().zip(batch_combiners.iter()) {
            // // `z_a` is added by a scaled vanishing polynomial
            // assert!(z_a.degree() < constraint_domain.size());
            // if MM::ZK {
            //     assert_eq!(z_b.degree(), constraint_domain.size());
            // } else {
            //     assert!(z_b.degree() < constraint_domain.size());
            // }

            // now z_a' = z_a + saved_rand * v_H
            let gpu_z_a = gpu_container.find(kern, &witness_label("z_a", i))?;
            let gpu_z_b = gpu_container.find(kern, &witness_label("z_b", i))?;

            gpu_local_summed_z_m.fill_with_zero()?;
            gpu_local_summed_z_m.copy_from_gpu(&gpu_z_a)?;

            if first_time {
                {
                    let mut gpu_z_a_tmp = gpu_container.ask_for(kern, n_2)?;

                    // calc `za_origin * zb`
                    gpu_z_a_tmp.fill_with_zero()?;
                    gpu_z_a_tmp.copy_from_gpu(&gpu_z_a)?;
                    gpu_z_a_tmp.add_at_offset(&-*saved_rand, constraint_domain.size())?;
                    gpu_z_a_tmp.add_at_offset(saved_rand, 0)?;

                    gpu_z_b_tmp.fill_with_zero()?;
                    gpu_z_b_tmp.copy_from_gpu(&gpu_z_b)?;

                    gpu_z_a_tmp.fft(&mut gpu_2k_tmp_buffer, &gpu_2k_pq, &gpu_2k_omegas, lg_2k)?;
                    gpu_z_b_tmp.fft(&mut gpu_2k_tmp_buffer, &gpu_2k_pq, &gpu_2k_omegas, lg_2k)?;
                    gpu_z_a_tmp.mul_assign(&gpu_z_b_tmp)?;
                    gpu_z_a_tmp.ifft(
                        &mut gpu_2k_tmp_buffer,
                        &gpu_2k_pq_ifft,
                        &gpu_2k_omegas_ifft,
                        &domain_2k.size_inv,
                        lg_2k,
                    )?;

                    // calc `zb * v_H`
                    let mut gpu_zb_vh = gpu_container.ask_for(kern, gpu_z_b.size() + constraint_domain.size())?;
                    gpu_zb_vh.fill_with_zero()?;
                    gpu_zb_vh.assign_mul_vanishing(&gpu_z_b, constraint_domain.size())?;

                    let za_zb_label = witness_label("gpu_za_zb", i);
                    gpu_container.save(&za_zb_label, gpu_z_a_tmp)?;

                    let zb_vh_label = witness_label("gpu_zb_vh", i);
                    gpu_container.save(&zb_vh_label, gpu_zb_vh)?;
                }

                gpu_z_b_tmp.fill_with_zero()?;
                gpu_z_b_tmp.copy_from_gpu(&gpu_z_b)?;

                gpu_z_b_tmp.scale(&eta_c)?;
                gpu_z_b_tmp.add_at_offset(&F::one(), 0)?;

                gpu_local_summed_z_m.fft(&mut gpu_2k_tmp_buffer, &gpu_2k_pq, &gpu_2k_omegas, lg_2k)?;
                gpu_z_b_tmp.fft(&mut gpu_2k_tmp_buffer, &gpu_2k_pq, &gpu_2k_omegas, lg_2k)?;

                gpu_local_summed_z_m.mul_assign(&gpu_z_b_tmp)?;
                gpu_local_summed_z_m.ifft(
                    &mut gpu_2k_tmp_buffer,
                    &gpu_2k_pq_ifft,
                    &gpu_2k_omegas_ifft,
                    &domain_2k.size_inv,
                    lg_2k,
                )?;
            } else {
                // we don't need FFT/iFFT here
                let za_zb_label = witness_label("gpu_za_zb", i);
                let zb_vh_label = witness_label("gpu_zb_vh", i);

                let gpu_za_zb = gpu_container.find(kern, &za_zb_label)?;
                let gpu_zb_vh = gpu_container.find(kern, &zb_vh_label)?;

                gpu_local_summed_z_m.add_assign_scale(&gpu_za_zb, &eta_c)?;
                gpu_local_summed_z_m.add_assign_scale(&gpu_zb_vh, &(eta_c * saved_rand))?;

                gpu_container.save(&za_zb_label, gpu_za_zb)?;
                gpu_container.save(&zb_vh_label, gpu_zb_vh)?;
            }
            gpu_local_summed_z_m.add_assign_scale(&gpu_z_b, &eta_b)?;

            // Can do better
            gpu_summed_z_m.add_assign_scale(&gpu_local_summed_z_m, combiner)?;

            gpu_container.save(&witness_label("z_a", i), gpu_z_a)?;
            gpu_container.save(&witness_label("z_b", i), gpu_z_b)?;
            kern.sync()?;
        }

        // don't forget to do cleaning!
        gpu_container.recycle(gpu_local_summed_z_m)?;
        gpu_container.recycle(gpu_z_b_tmp)?;

        // get result
        gpu_container.save("gpu_summed_z_m", gpu_summed_z_m)?;
        let dummy_summed_z_m = DensePolynomial::zero();

        gpu_container.save(&format!("domain_{n_2}_pq"), gpu_2k_pq)?;
        gpu_container.save(&format!("domain_{n_2}_omegas"), gpu_2k_omegas)?;
        gpu_container.save(&format!("domain_{n_2}_pq_ifft"), gpu_2k_pq_ifft)?;
        gpu_container.save(&format!("domain_{n_2}_omegas_ifft"), gpu_2k_omegas_ifft)?;
        gpu_container.recycle(gpu_2k_tmp_buffer)?;

        end_timer!(summed_z_m_poly_time);

        let t_poly_time = start_timer!(|| "Compute t poly");

        // let r_alpha_x_evals = constraint_domain.batch_eval_unnormalized_bivariate_lagrange_poly_with_diff_inputs(alpha);
        let mut r_alpha_x_evals = vec![F::zero(); constraint_domain.size()];

        {
            let mut gpu_results = gpu_container.ask_for_results(kern)?;
            let mut gpu_buckets = gpu_container.ask_for_buckets(kern)?;

            let h = constraint_domain.size();
            let lg_h = log2_floor(h);
            let vanish_x = constraint_domain.evaluate_vanishing_polynomial(alpha);

            let mut gpu_h_tmp_buffer = gpu_container.ask_for(&kern, h)?;
            let gpu_h_pq_ifft = gpu_container.find(&kern, &format!("domain_{h}_pq_ifft"))?;
            let gpu_h_omegas_ifft = gpu_container.find(&kern, &format!("domain_{h}_omegas_ifft"))?;

            let mut poly_1 = gpu_container.ask_for(kern, h)?;
            let mut poly_2 = gpu_container.ask_for(kern, h)?;

            let gpu_g = gpu_container.find(kern, &format!("domain_{h}_powers_neg"))?;
            poly_1.copy_from_gpu(&gpu_g)?;
            poly_1.add_constant(&alpha)?;
            poly_1.batch_inversion(&mut gpu_results, &mut gpu_buckets)?;
            poly_1.scale(&vanish_x)?;
            poly_1.write_to(&mut r_alpha_x_evals)?;

            poly_2.copy_from_gpu(&poly_1)?;
            poly_2.ifft(
                &mut gpu_h_tmp_buffer,
                &gpu_h_pq_ifft,
                &gpu_h_omegas_ifft,
                &constraint_domain.size_inv,
                lg_h,
            )?;
            kern.sync()?;

            // gpu_container.save("gpu_r_alpha_x_evals_domain_h", poly_1)?;
            gpu_container.save("gpu_r_alpha_x_coeffs_domain_h", poly_2)?;
            gpu_container.save(&format!("domain_{h}_pq_ifft"), gpu_h_pq_ifft)?;
            gpu_container.save(&format!("domain_{h}_omegas_ifft"), gpu_h_omegas_ifft)?;
            gpu_container.save(&format!("domain_{h}_powers_neg"), gpu_g)?;

            gpu_container.recycle(poly_1)?;
            gpu_container.recycle(gpu_h_tmp_buffer)?;
            gpu_container.recycle(gpu_results)?;
            gpu_container.recycle(gpu_buckets)?;
        }

        let dummy_t = Self::calculate_t_on_gpu(
            kern,
            gpu_container,
            //&[&state.index.a, &state.index.b, &state.index.c],
            &state.reindex_ABC_matrix,
            &state.coeff_one_vec,
            &state.col_zero_vec,
            [F::one(), eta_b, eta_c, eta_b + F::one(), eta_c + F::one(), eta_b - F::one()],
            state.constraint_domain,
            &r_alpha_x_evals,
        )?;

        end_timer!(t_poly_time);

        Ok((dummy_summed_z_m, dummy_t))
    }

    /// t is stored in gpu, returning a dummy polynomial
    fn calculate_t_on_gpu<'a, Fg: GpuPrimeField>(
        kern: &PolyKernel<Fg>,
        gpu_container: &mut GpuPolyContainer<Fg>,
        //matrices: &[&'a Matrix<F>],
        matrices: &Vec<Vec<(usize, usize, usize, F)>>,
        coeff_one_vec: &Vec<(usize, usize, usize, F)>,
        col_zero_vec: &Vec<(usize, usize, usize, F)>,
        matrix_randomizers: [F; 6],
        constraint_domain: EvaluationDomain<F>,
        r_alpha_x_on_h: &[F],
    ) -> Result<DensePolynomial<F>, AHPError> {
        let mut t_evals_on_h = vec![F::zero(); constraint_domain.size()];

        // for (matrix, eta) in matrices.iter().zip_eq(matrix_randomizers) {
        //     for (r, row) in matrix.iter().enumerate() {
        //         let mut x = r_alpha_x_on_h[r];
        //         if eta != F::one() {
        //             x = eta * x;
        //         }
        //         for (coeff, c) in row.iter() {
        //             /* there are so many one in the coeffs */
        //             if *coeff == F::zero() {
        //                 continue
        //             }
        //             let index = constraint_domain.reindex_by_subdomain(input_domain, *c);
        //             if *coeff == F::one() {
        //                 t_evals_on_h[index] += &(x);
        //             } else {
        //                 t_evals_on_h[index] += &(*coeff * x);
        //             }
        //         }
        //     }
        // }

        for (_, rows) in matrices.iter().enumerate() {
            let (index, r, _, _) = rows[0];
            let mut r_alpha_x = r_alpha_x_on_h[r];
            let row_offset = index >> 24;
            if row_offset > 0 {
                r_alpha_x += r_alpha_x_on_h[r + row_offset];
            }
            let mut eta_rs = [r_alpha_x; 6];
            for i in 1..6 {
                if (index & (1 << (16 + i))) > 0 {
                    eta_rs[i] = matrix_randomizers[i] * r_alpha_x;
                }
            }

            for (index, _, c, coeff) in rows.iter() {
                let eta_r = eta_rs[*index & 0xff];
                t_evals_on_h[*c] += &(eta_r * coeff);
            }
        }

        let mut last_col = 0;
        let mut col_sum = F::zero();
        for (index, r, c, _) in coeff_one_vec.iter() {
            let mut eta_r = r_alpha_x_on_h[*r];
            if *index & 0xff != 0 {
                eta_r *= matrix_randomizers[*index & 0xff];
            }
            if *c == last_col {
                col_sum += eta_r;
            } else {
                t_evals_on_h[last_col] += col_sum;

                last_col = *c;
                col_sum = eta_r;
            }
        }

        if col_sum != F::zero() {
            t_evals_on_h[last_col] += col_sum;
        }

        let mut sum = F::zero();
        for (index, r, _, _) in col_zero_vec.iter() {
            let eta_r = matrix_randomizers[*index & 0xff] * r_alpha_x_on_h[*r];
            sum += eta_r;
        }
        t_evals_on_h[0] += sum;

        let n = constraint_domain.size();
        let lgn = log2_floor(n);

        let mut gpu_t_evals_on_h = gpu_container.ask_for(&kern, n)?;
        let mut gpu_h_tmp_buffer = gpu_container.ask_for(kern, n)?;
        let gpu_h_pq_ifft = gpu_container.find(kern, &format!("domain_{n}_pq_ifft"))?;
        let gpu_h_omegas_ifft = gpu_container.find(kern, &format!("domain_{n}_omegas_ifft"))?;

        gpu_t_evals_on_h.read_from(t_evals_on_h.as_slice())?;
        gpu_t_evals_on_h.ifft(
            &mut gpu_h_tmp_buffer,
            &gpu_h_pq_ifft,
            &gpu_h_omegas_ifft,
            &constraint_domain.size_inv,
            lgn,
        )?;
        kern.sync()?;

        // don't forget to do cleaning!
        gpu_container.save("gpu_t", gpu_t_evals_on_h)?;
        gpu_container.recycle(gpu_h_tmp_buffer)?;
        gpu_container.save(&format!("domain_{n}_pq_ifft"), gpu_h_pq_ifft)?;
        gpu_container.save(&format!("domain_{n}_omegas_ifft"), gpu_h_omegas_ifft)?;

        Ok(DensePolynomial::zero())
    }
}

/// GPU proving third round
impl<F: PrimeField, MM: MarlinMode> AHPForR1CS<F, MM> {
    /// Output the third round message and the next state.
    pub fn prover_third_round_on_gpu<'a, R: RngCore, Fg: GpuPrimeField>(
        kern: &PolyKernel<Fg>,
        gpu_container: &mut GpuPolyContainer<Fg>,
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

        let (sum_a, lhs_a, g_a) = Self::matrix_sumcheck_helper_on_gpu(
            kern,
            gpu_container,
            "a",
            state.non_zero_a_domain,
            &state.index.a_arith,
            *alpha,
            beta,
            v_H_alpha_v_H_beta,
            largest_non_zero_domain_size,
            state.fft_precomputation(),
            state.ifft_precomputation(),
        )?;

        let (sum_b, lhs_b, g_b) = Self::matrix_sumcheck_helper_on_gpu(
            kern,
            gpu_container,
            "b",
            state.non_zero_b_domain,
            &state.index.b_arith,
            *alpha,
            beta,
            v_H_alpha_v_H_beta,
            largest_non_zero_domain_size,
            state.fft_precomputation(),
            state.ifft_precomputation(),
        )?;

        let (sum_c, lhs_c, g_c) = Self::matrix_sumcheck_helper_on_gpu(
            kern,
            gpu_container,
            "c",
            state.non_zero_c_domain,
            &state.index.c_arith,
            *alpha,
            beta,
            v_H_alpha_v_H_beta,
            largest_non_zero_domain_size,
            state.fft_precomputation(),
            state.ifft_precomputation(),
        )?;

        let msg = prover::ThirdMessage { sum_a, sum_b, sum_c };
        let oracles = prover::ThirdOracles { g_a, g_b, g_c };
        state.lhs_polynomials = Some([lhs_a, lhs_b, lhs_c]);
        state.sums = Some([sum_a, sum_b, sum_c]);
        assert!(oracles.matches_info(&Self::third_round_polynomial_info(&state.index.index_info)));

        end_timer!(round_time);

        Ok((msg, oracles, state))
    }

    #[allow(clippy::too_many_arguments)]
    fn matrix_sumcheck_helper_on_gpu<Fg: GpuPrimeField>(
        kern: &PolyKernel<Fg>,
        gpu_container: &mut GpuPolyContainer<Fg>,
        label: &str,
        non_zero_domain: EvaluationDomain<F>,
        _arithmetization: &MatrixArithmetization<F>,
        alpha: F,
        beta: F,
        v_H_alpha_v_H_beta: F,
        largest_non_zero_domain_size: F,
        _fft_precomputation: &FFTPrecomputation<F>,
        _ifft_precomputation: &IFFTPrecomputation<F>,
    ) -> Result<(F, DensePolynomial<F>, LabeledPolynomial<F>), AHPError> {
        let f_prepare_time = start_timer!(|| "Computing gpu_b");
        let domain_k_size = non_zero_domain.size();

        let domain_2k_size = domain_k_size * 2;
        let lg_k = log2_floor(domain_k_size);

        let alpha_beta = alpha * beta;

        let mut gpu_k_tmp_buffer = gpu_container.ask_for(kern, domain_k_size)?;
        let gpu_k_pq = gpu_container.find(kern, &format!("domain_{domain_k_size}_pq"))?;
        let gpu_k_omegas = gpu_container.find(kern, &format!("domain_{domain_k_size}_omegas"))?;
        let gpu_k_pq_ifft = gpu_container.find(kern, &format!("domain_{domain_k_size}_pq_ifft"))?;
        let gpu_k_omegas_ifft = gpu_container.find(kern, &format!("domain_{domain_k_size}_omegas_ifft"))?;

        // let (row_on_K, col_on_K, row_col_on_K) =
        //     (&arithmetization.evals_on_K.row, &arithmetization.evals_on_K.col, &arithmetization.evals_on_K.row_col);

        let gpu_row_on_k = gpu_container.find(kern, &("evals_row_".to_owned() + label))?;
        let gpu_col_on_k = gpu_container.find(kern, &("evals_col_".to_owned() + label))?;
        let gpu_row_col_on_k = gpu_container.find(kern, &("evals_row_col_".to_owned() + label))?;
        let gpu_evals_on_k = gpu_container.find(kern, &("evals_val_".to_owned() + label))?;

        let mut gpu_b_poly = gpu_container.ask_for(kern, domain_k_size)?;
        gpu_b_poly.fill_with_fe(&alpha_beta)?;

        gpu_b_poly.sub_assign_scale(&gpu_row_on_k, &alpha)?;
        gpu_b_poly.sub_assign_scale(&gpu_col_on_k, &beta)?;
        gpu_b_poly.add_assign(&gpu_row_col_on_k)?;
        let mut gpu_f = gpu_container.ask_for(kern, domain_k_size)?;
        gpu_f.copy_from_gpu(&gpu_b_poly)?;

        gpu_b_poly.ifft(&mut gpu_k_tmp_buffer, &gpu_k_pq_ifft, &gpu_k_omegas_ifft, &non_zero_domain.size_inv, lg_k)?;
        end_timer!(f_prepare_time);

        let f_evals_time = start_timer!(|| "Computing g");

        let mut gpu_rev_res_buf = gpu_container.ask_for_results(kern)?;
        let mut gpu_rev_buck_buf = gpu_container.ask_for_buckets(kern)?;

        // Can do better
        gpu_f.batch_inversion(&mut gpu_rev_res_buf, &mut gpu_rev_buck_buf)?;
        gpu_f.mul_assign(&gpu_evals_on_k)?;
        gpu_f.scale(&v_H_alpha_v_H_beta)?;

        let start = match label {
            "a" => 37908,
            "b" => 38031,
            "c" => 48434,
            _ => panic!("invalid label"),
        };

        // save g in evals form
        let mut gpu_g_evals = gpu_container.ask_for(kern, start)?;
        gpu_g_evals.copy_from_gpu(&gpu_f)?;

        gpu_f.ifft(&mut gpu_k_tmp_buffer, &gpu_k_pq_ifft, &gpu_k_omegas_ifft, &non_zero_domain.size_inv, lg_k)?;

        let mut gpu_g = gpu_container.ask_for(kern, gpu_f.size() - 1)?;
        gpu_g.copy_from_gpu_offset(&gpu_f, 1)?;

        let mut f_0 = vec![F::zero(); 1];
        gpu_f.write_to(&mut f_0)?;

        let gpu_inv = gpu_container.find(kern, &format!("domain_{domain_k_size}_powers_group_gen_inv"))?;
        gpu_g_evals.sub_constant(&f_0[0])?;
        gpu_g_evals.mul_assign(&gpu_inv)?;

        gpu_container.save(&format!("domain_{domain_k_size}_powers_group_gen_inv"), gpu_inv)?;
        gpu_container.save(&format!("g_{}_evals", label), gpu_g_evals)?;

        gpu_container.save(&("g_".to_string() + label), gpu_g)?;

        end_timer!(f_evals_time);

        let f_evals_time = start_timer!(|| "Computing h");

        let multiplier = non_zero_domain.size_as_field_element / largest_non_zero_domain_size;

        let domain_2k_omega = F::get_root_of_unity(domain_2k_size).unwrap();
        let domain_2k_omega_inv = domain_2k_omega.inverse().unwrap();

        let gpu_val_coset = gpu_container.find(kern, &format!("val_{label}_coset"))?;

        let gpu_2k_omega_powers = gpu_container.find(kern, &format!("domain_{domain_2k_size}_omega_powers"))?;
        let gpu_2k_omega_inv_powers = gpu_container.find(kern, &format!("domain_{domain_2k_size}_omega_inv_powers"))?;

        gpu_f.distribute_powers_naive(&gpu_2k_omega_powers, &domain_2k_omega, 0)?;
        gpu_b_poly.distribute_powers_naive(&gpu_2k_omega_powers, &domain_2k_omega, 0)?;

        gpu_f.fft(&mut gpu_k_tmp_buffer, &gpu_k_pq, &gpu_k_omegas, lg_k)?;
        gpu_b_poly.fft(&mut gpu_k_tmp_buffer, &gpu_k_pq, &gpu_k_omegas, lg_k)?;

        gpu_f.mul_assign(&gpu_b_poly)?;
        gpu_f.sub_assign_scale(&gpu_val_coset, &v_H_alpha_v_H_beta)?;

        let factor = (F::one() / (F::one() + F::one())) * multiplier;
        gpu_f.scale(&factor)?;

        gpu_f.ifft(&mut gpu_k_tmp_buffer, &gpu_k_pq_ifft, &gpu_k_omegas_ifft, &non_zero_domain.size_inv, lg_k)?;
        gpu_f.distribute_powers_naive(&gpu_2k_omega_inv_powers, &domain_2k_omega_inv, 0)?;

        let dummy_h = DensePolynomial::<F>::zero();
        let dummy_g = DensePolynomial::<F>::zero();

        let mut gpu_h = gpu_container.ask_for(kern, domain_k_size - 1)?;
        gpu_h.copy_from_gpu(&gpu_f)?;

        gpu_container.save(&("lhs_".to_owned() + label), gpu_h)?;
        kern.sync()?;

        let dummy_g = LabeledPolynomial::new("g_".to_string() + label, dummy_g, Some(non_zero_domain.size() - 2), None);

        // assert!(h.degree() <= non_zero_domain.size() - 2);
        // assert!(g.degree() <= non_zero_domain.size() - 2);

        // don't forget to do cleaning
        gpu_container.recycle(gpu_rev_res_buf)?;
        gpu_container.recycle(gpu_rev_buck_buf)?;

        gpu_container.recycle(gpu_f)?;
        gpu_container.recycle(gpu_b_poly)?;
        gpu_container.save(&format!("val_{label}_coset"), gpu_val_coset)?;

        gpu_container.save(&("evals_row_".to_owned() + label), gpu_row_on_k)?;
        gpu_container.save(&("evals_col_".to_owned() + label), gpu_col_on_k)?;
        gpu_container.save(&("evals_row_col_".to_owned() + label), gpu_row_col_on_k)?;
        gpu_container.save(&("evals_val_".to_owned() + label), gpu_evals_on_k)?;

        gpu_container.save(&format!("domain_{domain_2k_size}_omega_powers"), gpu_2k_omega_powers)?;
        gpu_container.save(&format!("domain_{domain_2k_size}_omega_inv_powers"), gpu_2k_omega_inv_powers)?;
        gpu_container.save(&format!("domain_{domain_k_size}_pq"), gpu_k_pq)?;
        gpu_container.save(&format!("domain_{domain_k_size}_omegas"), gpu_k_omegas)?;
        gpu_container.save(&format!("domain_{domain_k_size}_pq_ifft"), gpu_k_pq_ifft)?;
        gpu_container.save(&format!("domain_{domain_k_size}_omegas_ifft"), gpu_k_omegas_ifft)?;
        gpu_container.recycle(gpu_k_tmp_buffer)?;

        end_timer!(f_evals_time);

        // Ok((f.coeffs[0], h, g))
        Ok((f_0[0], dummy_h, dummy_g))
    }
}

/// GPU proving fourth round
impl<F: PrimeField, MM: MarlinMode> AHPForR1CS<F, MM> {
    pub fn prover_fourth_round_on_gpu<'a, R: RngCore, Fg: GpuPrimeField>(
        kern: &PolyKernel<Fg>,
        gpu_container: &mut GpuPolyContainer<Fg>,
        verifier_message: &verifier::ThirdMessage<F>,
        _state: prover::State<'a, F, MM>,
        _r: &mut R,
    ) -> Result<prover::FourthOracles<F>, AHPError> {
        let verifier::ThirdMessage { r_b, r_c, .. } = verifier_message;
        // let [mut lhs_a, mut lhs_b, mut lhs_c] = state.lhs_polynomials.unwrap();

        let gpu_lhs_a = gpu_container.find(kern, "lhs_a")?;
        let gpu_lhs_b = gpu_container.find(kern, "lhs_b")?;
        let gpu_lhs_c = gpu_container.find(kern, "lhs_c")?;

        let n = gpu_lhs_a.size();

        let mut gpu_h_2 = gpu_container.ask_for(kern, n)?;

        gpu_h_2.copy_from_gpu(&gpu_lhs_a)?;

        gpu_h_2.add_assign_scale(&gpu_lhs_b, r_b)?;
        gpu_h_2.add_assign_scale(&gpu_lhs_c, r_c)?;
        kern.sync()?;

        let dummy_h_2 = DensePolynomial::<F>::zero();

        gpu_container.save("h_2", gpu_h_2)?;
        gpu_container.save("lhs_a", gpu_lhs_a)?;
        gpu_container.save("lhs_b", gpu_lhs_b)?;
        gpu_container.save("lhs_c", gpu_lhs_c)?;

        let h_2 = LabeledPolynomial::new("h_2".into(), dummy_h_2, None, None);

        let oracles = prover::FourthOracles { h_2 };
        assert!(oracles.matches_info(&Self::fourth_round_polynomial_info()));
        Ok(oracles)
    }
}
