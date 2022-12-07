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

use crate::fft::Polynomial;
use std::ops::AddAssign;

use crate::{
    fft::EvaluationDomain,
    fft::polynomial::PolyMultiplier,
    polycommit::sonic_pc::{
        Commitment,
        Evaluations,
        LabeledCommitment,
        LabeledPolynomial,
        LabeledPolynomialWithBasis,
        Randomness,
        SonicKZG10
    },
    snark::marlin::{
        ahp::{AHPError, AHPForR1CS, EvaluationsProvider},
        fiat_shamir::traits::FiatShamirRng,
        params::OptimizationType,
        proof,
        prover,
        witness_label,
        CircuitProvingKey,
        CircuitVerifyingKey,
        MarlinError,
        MarlinMode,
        PartialProof,
        Proof,
        UniversalSRS,
    },
    Prepare,
    SNARKError,
    SNARK,
    SRS,
};
use itertools::Itertools;
use rand::{CryptoRng, Rng};
use rand_core::RngCore;
use snarkvm_curves::PairingEngine;
use snarkvm_curves::AffineCurve;
use snarkvm_curves::ProjectiveCurve;
use snarkvm_fields::{One, PrimeField, ToConstraintField, Zero};
use snarkvm_r1cs::ConstraintSynthesizer;
use snarkvm_utilities::{to_bytes_le, ToBytes, Uniform};
use std::thread;
use rand::thread_rng;

use std::{borrow::Borrow, sync::Arc};
use std::str::FromStr;
use std::fmt::Debug;

#[cfg(not(feature = "std"))]
use snarkvm_utilities::println;

use core::{
    marker::PhantomData,
    sync::atomic::{AtomicBool, Ordering},
};

use std::sync::{Condvar, Mutex};
use std::collections::VecDeque;

fn get_config<T:FromStr>(key: &str, default: T) -> T where <T as FromStr>::Err: Debug {
    let val = match std::env::var(key) {
        Ok(val) => val.parse::<T>().unwrap(),
        Err(_e) => default
    };
    val
}


/// The Marlin proof system.
#[derive(Clone, Debug)]
pub struct MarlinSNARK<
    E: PairingEngine,
    FS: FiatShamirRng<E::Fr, E::Fq> + Send + Sync,
    MM: MarlinMode,
    Input: ToConstraintField<E::Fr> + ?Sized,
>(#[doc(hidden)] PhantomData<(E, FS, MM, Input)>);

impl<E: PairingEngine, FS: FiatShamirRng<E::Fr, E::Fq> + Send + Sync + 'static, MM: MarlinMode, Input: ToConstraintField<E::Fr> + ?Sized>
    MarlinSNARK<E, FS, MM, Input>
{
    /// The personalization string for this protocol.
    /// Used to personalize the Fiat-Shamir RNG.
    pub const PROTOCOL_NAME: &'static [u8] = b"MARLIN-2019";

    /// Generate the index-specific (i.e., circuit-specific) prover and verifier
    /// keys. This is a trusted setup.
    ///
    /// # Warning
    ///
    /// This method should be used *only* for testing purposes, and not in production.
    /// In production, one should instead perform a universal setup via [`Self::universal_setup`],
    /// and then deterministically specialize the resulting universal SRS via [`Self::circuit_setup`].
    #[allow(clippy::type_complexity)]
    pub fn circuit_specific_setup<C: ConstraintSynthesizer<E::Fr>, R: RngCore + CryptoRng>(
        c: &C,
        rng: &mut R,
    ) -> Result<(CircuitProvingKey<E, MM>, CircuitVerifyingKey<E, MM>), SNARKError> {
        let circuit = AHPForR1CS::<_, MM>::index(c)?;
        let srs = Self::universal_setup(&circuit.max_degree(), rng)?;
        Self::circuit_setup(&srs, c)
    }

    /// Generates the circuit proving and verifying keys.
    /// This is a deterministic algorithm that anyone can rerun.
    #[allow(clippy::type_complexity)]
    pub fn circuit_setup<C: ConstraintSynthesizer<E::Fr>>(
        universal_srs: &UniversalSRS<E>,
        circuit: &C,
    ) -> Result<(CircuitProvingKey<E, MM>, CircuitVerifyingKey<E, MM>), SNARKError> {
        let index_time = start_timer!(|| "Marlin::CircuitSetup");

        // TODO: Add check that c is in the correct mode.
        // Increase the universal SRS size to support the circuit size.
        let index = AHPForR1CS::<_, MM>::index(circuit)?;
        if universal_srs.max_degree() < index.max_degree() {
            universal_srs
                .increase_degree(index.max_degree())
                .map_err(|_| MarlinError::IndexTooLarge(universal_srs.max_degree(), index.max_degree()))?;
        }

        let coefficient_support = AHPForR1CS::<_, MM>::get_degree_bounds(&index.index_info);

        // Marlin only needs degree 2 random polynomials.
        let supported_hiding_bound = 1;
        let (committer_key, verifier_key) = SonicKZG10::<E, FS>::trim(
            universal_srs,
            index.max_degree(),
            [index.constraint_domain_size()],
            supported_hiding_bound,
            Some(&coefficient_support),
        )?;

        let commit_time = start_timer!(|| "Commit to index polynomials");
        let (mut circuit_commitments, circuit_commitment_randomness): (_, _) =
            SonicKZG10::<E, FS>::commit(&committer_key, index.iter().map(Into::into), None)?;
        end_timer!(commit_time);

        circuit_commitments.sort_by(|c1, c2| c1.label().cmp(c2.label()));
        let circuit_commitments = circuit_commitments.into_iter().map(|c| *c.commitment()).collect();
        let circuit_verifying_key = CircuitVerifyingKey {
            circuit_info: index.index_info,
            circuit_commitments,
            verifier_key,
            mode: PhantomData,
        };

        let _vk_prep = circuit_verifying_key.prepare();

        let circuit_proving_key = CircuitProvingKey {
            circuit: index,
            circuit_commitment_randomness,
            circuit_verifying_key: circuit_verifying_key.clone(),
            committer_key,
        };

        end_timer!(index_time);

        Ok((circuit_proving_key, circuit_verifying_key))
    }

    fn terminate(terminator: &AtomicBool) -> Result<(), MarlinError> {
        if terminator.load(Ordering::Relaxed) { Err(MarlinError::Terminated) } else { Ok(()) }
    }

    fn init_sponge(
        batch_size: usize,
        circuit_commitments: &[crate::polycommit::sonic_pc::Commitment<E>],
        inputs: &[Vec<E::Fr>],
    ) -> FS {
        let mut sponge = FS::new();
        sponge.absorb_bytes(&to_bytes_le![&Self::PROTOCOL_NAME].unwrap());
        sponge.absorb_bytes(&batch_size.to_le_bytes());
        sponge.absorb_native_field_elements(circuit_commitments);
        for input in inputs {
            sponge.absorb_nonnative_field_elements(input.iter().copied(), OptimizationType::Weight);
        }
        sponge
    }

    fn absorb_labeled_with_msg(
        comms: &[LabeledCommitment<Commitment<E>>],
        message: &prover::ThirdMessage<E::Fr>,
        sponge: &mut FS,
    ) {
        let commitments: Vec<_> = comms.iter().map(|c| *c.commitment()).collect();
        Self::absorb_with_msg(&commitments, message, sponge)
    }

    fn absorb_labeled(comms: &[LabeledCommitment<Commitment<E>>], sponge: &mut FS) {
        let commitments: Vec<_> = comms.iter().map(|c| *c.commitment()).collect();
        Self::absorb(&commitments, sponge)
    }

    fn absorb(commitments: &[Commitment<E>], sponge: &mut FS) {
        sponge.absorb_native_field_elements(commitments);
    }

    fn absorb_with_msg(commitments: &[Commitment<E>], msg: &prover::ThirdMessage<E::Fr>, sponge: &mut FS) {
        Self::absorb(commitments, sponge);
        sponge.absorb_nonnative_field_elements([msg.sum_a, msg.sum_b, msg.sum_c], OptimizationType::Weight);
    }
}

impl<E: PairingEngine, FS, MM, Input> SNARK for MarlinSNARK<E, FS, MM, Input>
where
    E::Fr: PrimeField,
    E::Fq: PrimeField,
    FS: FiatShamirRng<E::Fr, E::Fq> + Send + Sync + 'static,
    MM: MarlinMode,
    Input: ToConstraintField<E::Fr> + ?Sized,
{
    type BaseField = E::Fq;
    type PartialProof = PartialProof<E, FS>;
    type Proof = Proof<E>;
    type ProvingKey = CircuitProvingKey<E, MM>;
    type ScalarField = E::Fr;
    type UniversalSetupConfig = usize;
    type UniversalSetupParameters = UniversalSRS<E>;
    type VerifierInput = Input;
    type VerifyingKey = CircuitVerifyingKey<E, MM>;

    fn universal_setup<R: Rng + CryptoRng>(
        max_degree: &Self::UniversalSetupConfig,
        rng: &mut R,
    ) -> Result<Self::UniversalSetupParameters, SNARKError> {
        let setup_time = start_timer!(|| { format!("Marlin::UniversalSetup with max_degree {}", max_degree,) });

        let srs = SonicKZG10::<E, FS>::setup(*max_degree, rng).map_err(Into::into);
        end_timer!(setup_time);
        srs
    }

    fn setup<C: ConstraintSynthesizer<E::Fr>, R: Rng + CryptoRng>(
        circuit: &C,
        srs: &mut SRS<R, Self::UniversalSetupParameters>,
    ) -> Result<(Self::ProvingKey, Self::VerifyingKey), SNARKError> {
        match srs {
            SRS::CircuitSpecific(rng) => Self::circuit_specific_setup(circuit, rng),
            SRS::Universal(srs) => Self::circuit_setup(srs, circuit),
        }
        .map_err(SNARKError::from)
    }

    #[allow(clippy::only_used_in_recursion)]
    fn prove_batch_with_terminator<C: ConstraintSynthesizer<E::Fr>, R: Rng + CryptoRng>(
        _circuit_proving_key: &CircuitProvingKey<E, MM>,
        _circuits: &[C],
        _terminator: &AtomicBool,
        _zk_rng: &mut R,
    ) -> Result<Self::Proof, SNARKError> {
        unimplemented!()
    }

//**************************************************************************************************
// Test first/second half
//**************************************************************************************************
    fn prove_batch_with_terminator_test<C: ConstraintSynthesizer<E::Fr>, R: Rng + CryptoRng>(
        _circuit_proving_key: &CircuitProvingKey<E, MM>,
        _circuits: &[C],
        _terminator: &AtomicBool,
        _zk_rng: &mut R,
    ) -> Result<Self::Proof, SNARKError> {
        unimplemented!()
        //let pp = Self::prove_first_half(circuit_proving_key, circuits, terminator, zk_rng)?;
        //let proof = Self::prove_second_half(circuit_proving_key, terminator, zk_rng, &pp)?;
        //let proof = Self::prove_second_half_thread(circuit_proving_key, terminator, &pp)?;
        //Ok(proof)
    }
//**************************************************************************************************
//**************************************************************************************************
// First Half
//**************************************************************************************************
//**************************************************************************************************

    fn cache_committer_key(
        circuit_proving_key: &CircuitProvingKey<E, MM>,
    ) {
        let bases = circuit_proving_key.committer_key.shifted_powers_of_beta_g.as_ref().unwrap();
        snarkvm_cuda::msm_cache(bases);
        snarkvm_cuda::msm_cache(&bases[32768..]);
        
    }

    fn prove_first_half<C: ConstraintSynthesizer<Self::ScalarField>, R: Rng + CryptoRng>(
        circuit_proving_key: &CircuitProvingKey<E, MM>,
        circuits: &[C],
        terminator: &AtomicBool,
        zk_rng: &mut R,
    ) -> Result<Self::PartialProof, SNARKError> {
        let prover_time = start_timer!(|| "Marlin::Prover first half");
        let batch_size = circuits.len();
        if batch_size == 0 {
            return Err(SNARKError::EmptyBatch);
        }

        Self::terminate(terminator)?;

        let prover_state = AHPForR1CS::<_, MM>::init_prover(&circuit_proving_key.circuit, circuits)?;
        let padded_public_input = prover_state.padded_public_inputs();
        assert_eq!(prover_state.batch_size, batch_size);

        let sponge = Self::init_sponge(
            batch_size,
            &circuit_proving_key.circuit_verifying_key.circuit_commitments,
            &padded_public_input,
        );

        // --------------------------------------------------------------------
        // First round

        Self::terminate(terminator)?;
        let mut prover_state = AHPForR1CS::<_, MM>::prover_first_round(prover_state, zk_rng)?;
        Self::terminate(terminator)?;

        let first_round_comm_time = start_timer!(|| "Committing to first round polys");
        let (first_commitments, _first_commitment_randomnesses) = {
            let first_round_oracles = Arc::get_mut(prover_state.first_round_oracles.as_mut().unwrap()).unwrap();
            SonicKZG10::<E, FS>::commit(
                &circuit_proving_key.committer_key,
                first_round_oracles.iter_for_commit(),
                Some(zk_rng),
            )?
        };
        end_timer!(first_round_comm_time);

        let (v_h_div_v_x, rem) = Polynomial::from(prover_state.constraint_domain.vanishing_polynomial()).divide_with_q_and_r(&Polynomial::from(prover_state.input_domain.vanishing_polynomial())).unwrap();
        assert!(rem.is_zero());
        let v_h_div_v_x_lp = LabeledPolynomial::new("v_h_div_v_x".to_string(), v_h_div_v_x.clone(), None, None);
        let (w_comm, _) = SonicKZG10::<E, FS>::commit(
            &circuit_proving_key.committer_key,
            [v_h_div_v_x_lp.clone().into()].into_iter(),
            Some(zk_rng),
        )?;
        let w_comm_aff = w_comm[0].commitment().0;

        let z_a_m = prover_state.first_round_oracles.as_ref().unwrap().batches[0].z_a_poly.polynomial().as_dense().unwrap();
        let z_b_m = prover_state.first_round_oracles.as_ref().unwrap().batches[0].z_b_poly.polynomial().as_dense().unwrap();
        let mut multiplier = PolyMultiplier::new();
        multiplier.add_polynomial_ref(z_a_m, "z_a");
        multiplier.add_polynomial_ref(z_b_m, "z_b");
        let z_c = multiplier.multiply().unwrap();

        // Prepare cache for computing t polynomial
        {
            // Prepare a vector
            let a_matrix_len = crate::snark::marlin::ahp::indexer::num_non_zero(&prover_state.index.a);
            let mut a_flat_r: Vec<u32> = vec![0; a_matrix_len];
            let mut a_flat_c: Vec<u32> = vec![0; a_matrix_len];
            let mut a_flat_coeff = vec![E::Fr::zero(); a_matrix_len];
            let mut idx_flat = 0;
            for (r, row) in prover_state.index.a.iter().enumerate() {
                for (coeff, c) in row.iter() {
                    a_flat_r[idx_flat] = r as u32;
                    a_flat_c[idx_flat] = *c as u32;
                    a_flat_coeff[idx_flat] = *coeff;
                    idx_flat += 1;
                }
            }
            // Prepare b vector
            let b_matrix_len = crate::snark::marlin::ahp::indexer::num_non_zero(&prover_state.index.b);
            let mut b_flat_r: Vec<u32> = vec![0; b_matrix_len];
            let mut b_flat_c: Vec<u32> = vec![0; b_matrix_len];
            let mut b_flat_coeff = vec![E::Fr::zero(); b_matrix_len];
            let mut idx_flat = 0;
            for (r, row) in prover_state.index.b.iter().enumerate() {
                for (coeff, c) in row.iter() {
                    b_flat_r[idx_flat] = r as u32;
                    b_flat_c[idx_flat] = *c as u32;
                    b_flat_coeff[idx_flat] = *coeff;
                    idx_flat += 1;
                }
            }
            
            // Prepare c vector
            let c_matrix_len = crate::snark::marlin::ahp::indexer::num_non_zero(&prover_state.index.c);
            let mut c_flat_r: Vec<u32> = vec![0; c_matrix_len];
            let mut c_flat_c: Vec<u32> = vec![0; c_matrix_len];
            let mut c_flat_coeff = vec![E::Fr::zero(); c_matrix_len];
            let mut idx_flat = 0;
            for (r, row) in prover_state.index.c.iter().enumerate() {
                for (coeff, c) in row.iter() {
                    c_flat_r[idx_flat] = r as u32;
                    c_flat_c[idx_flat] = *c as u32;
                    c_flat_coeff[idx_flat] = *coeff;
                    idx_flat += 1;
                }
            }
            
            snarkvm_cuda::cache_poly_t_inputs
                (a_matrix_len, &a_flat_r, &a_flat_c, &a_flat_coeff,
                 b_matrix_len, &b_flat_r, &b_flat_c, &b_flat_coeff,
                 c_matrix_len, &c_flat_r, &c_flat_c, &c_flat_coeff,

                 &prover_state.index.a_arith.evals_on_K.row.evaluations,
                 &prover_state.index.a_arith.evals_on_K.col.evaluations,
                 &prover_state.index.a_arith.evals_on_K.row_col.evaluations,
                 prover_state.index.a_arith.val.as_dense().unwrap().coeffs(),
                 &prover_state.index.a_arith.evals_on_K.val.evaluations,

                 &prover_state.index.b_arith.evals_on_K.row.evaluations,
                 &prover_state.index.b_arith.evals_on_K.col.evaluations,
                 &prover_state.index.b_arith.evals_on_K.row_col.evaluations,
                 prover_state.index.b_arith.val.as_dense().unwrap().coeffs(),
                 &prover_state.index.b_arith.evals_on_K.val.evaluations,

                 &prover_state.index.c_arith.evals_on_K.row.evaluations,
                 &prover_state.index.c_arith.evals_on_K.col.evaluations,
                 &prover_state.index.c_arith.evals_on_K.row_col.evaluations,
                 prover_state.index.c_arith.val.as_dense().unwrap().coeffs(),
                 &prover_state.index.c_arith.evals_on_K.val.evaluations,
                );
        }
        
        let proof = PartialProof::new(
            &prover_state,
            sponge,
            first_commitments,
            v_h_div_v_x,
            w_comm_aff,
            z_c,
        );

        end_timer!(prover_time);
        Ok(proof)
    }

//**************************************************************************************************
//**************************************************************************************************
// Second Half
//**************************************************************************************************
//**************************************************************************************************
    fn prove_second_half(
        circuit_proving_key: &CircuitProvingKey<E, MM>,
        terminator: &AtomicBool,
        partial_proof: &Self::PartialProof,
        buffer: &Arc<(Mutex<VecDeque<Result<Self::Proof, SNARKError>>>, Condvar)>,
    ) -> Result<Self::Proof, SNARKError> {
        lazy_static::lazy_static! {
            static ref PROVE_STARTED: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
            static ref T_COPY: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
        }

        if !PROVE_STARTED.load(Ordering::Relaxed) {
            PROVE_STARTED.store(true, Ordering::Relaxed);
            //println!("prove_second_half prove_started");

            let mut threads_to_run = get_config::<usize>("POSW_THREADS", 15);
            let core_ids = core_affinity::get_core_ids().unwrap();
            threads_to_run = std::cmp::min(threads_to_run, core_ids.len());
            
            //println!("Launch {} threads", threads_to_run);

            for thrid in 0..threads_to_run {

                let c = circuit_proving_key.clone();
                let t = Arc::clone(&T_COPY);
                let p = partial_proof.clone();
                let b = Arc::clone(buffer);
                let cores = core_ids.clone();
                thread::spawn(move|| {
                    core_affinity::set_for_current(cores[thrid]);
                    let proof = Self::prove_second_half_thread(&c, &t, &p, &b);
                    let lock = &b.0;
                    let cvar = &b.1;
                    let mut guard = lock.lock().unwrap();
                    guard.push_back(proof);
                    cvar.notify_one();
                });
            }
        }

        if terminator.load(Ordering::Relaxed) {
            PROVE_STARTED.store(false, Ordering::Relaxed);
            T_COPY.store(true, Ordering::Relaxed);
        }

        let lock = &buffer.0;
        let cvar = &buffer.1;
        let mut guard = cvar.wait_while(lock.lock().unwrap(), |v| v.is_empty()).unwrap();
        guard.pop_front().unwrap()
    }

    fn prove_second_half_thread(
        circuit_proving_key: &CircuitProvingKey<E, MM>,
        terminator: &AtomicBool,
        partial_proof: &Self::PartialProof,
        buffer: &Arc<(Mutex<VecDeque<Result<Self::Proof, SNARKError>>>, Condvar)>,
    ) -> Result<Self::Proof, SNARKError> {
        let public_input = partial_proof.public_inputs.clone();
        let mut w_poly = partial_proof.w_poly.clone();
        let first_commitment_randomnesses = vec![
            Randomness::empty(),
            Randomness::empty(),
            Randomness::empty(),
        ];
        let batch_size = 1;
        let mut first_commitments = partial_proof.first_commitments.clone();
        let mut zk_rng = thread_rng();

        // Large randomize to distinguish threads
        let r_w = E::Fr::rand(&mut zk_rng);
        let w_poly_tweak = w_poly.polynomial_mut().as_dense_mut().unwrap();
        w_poly_tweak.add_assign(&(&partial_proof.v_h_div_v_x * r_w));
        let (w_comm, _) = SonicKZG10::<E, FS>::commit(
            &circuit_proving_key.committer_key,
            [w_poly.clone().into()].into_iter(),
            Some(&mut zk_rng),
        )?;
        first_commitments[0] = w_comm[0].clone();

        let mut z_poly = w_poly.polynomial().as_dense().unwrap().mul_by_vanishing_poly(partial_proof.input_domain);
        let x_poly = &partial_proof.x_poly;
        z_poly.coeffs.iter_mut().zip(&x_poly.coeffs).for_each(|(z, x)| *z += x);

        let v_h = partial_proof.constraint_domain.vanishing_polynomial();
        let z_c = &partial_proof.z_c;

        loop {
            let mut sponge = partial_proof.sponge.clone();

            let mut c = first_commitments[0].commitment().0.to_projective();
            c.add_assign_mixed(&partial_proof.w_comm_inc);
            first_commitments[0].commitment_mut().0 = c.to_affine();
            let w_poly_tweak = w_poly.polynomial_mut().as_dense_mut().unwrap();
            w_poly_tweak.add_assign(&partial_proof.v_h_div_v_x);

            z_poly.add_assign(&v_h);

            let batches = vec![prover::SingleEntry {
                z_a: LabeledPolynomialWithBasis::from(partial_proof.z_a_poly.clone()),
                z_b: LabeledPolynomialWithBasis::from(partial_proof.z_b_poly.clone()),
                w_poly: w_poly.clone(),
                z_a_poly: partial_proof.z_a_poly.clone(),
                z_b_poly: partial_proof.z_b_poly.clone(),
            }];
            let first_round_oracles = prover::FirstOracles { batches, mask_poly: None};

            let prover_state = prover::State {
                index: &circuit_proving_key.circuit,
                input_domain: partial_proof.input_domain,
                constraint_domain: partial_proof.constraint_domain,
                non_zero_a_domain: partial_proof.non_zero_a_domain,
                non_zero_b_domain: partial_proof.non_zero_b_domain,
                non_zero_c_domain: partial_proof.non_zero_c_domain,
                batch_size,
                padded_public_variables: vec![], // not needed
                x_poly: vec![partial_proof.x_poly.clone()],
                private_variables: vec![],       // not needed
                z_a: None,
                z_b: None,
                first_round_oracles: Some(Arc::new(first_round_oracles)),
                mz_poly_randomizer: None,        // no randomization
                verifier_first_message: None,    // second round
                lhs_polynomials: None,           // third round
                sums: None,                      // third round
            };

            // --------------------------------------------------------------------
            // Finish First round
            Self::absorb_labeled(&first_commitments, &mut sponge);
            //Self::terminate(terminator)?;

            let (verifier_first_message, verifier_state) = AHPForR1CS::<_, MM>::verifier_first_round(
                circuit_proving_key.circuit_verifying_key.circuit_info,
                batch_size,
                &mut sponge,
            )?;

            // --------------------------------------------------------------------

            // --------------------------------------------------------------------
            // Second round

            let (second_oracles, prover_state) =
                //AHPForR1CS::<_, MM>::prover_second_round(&verifier_first_message, prover_state, zk_rng);
                AHPForR1CS::<_, MM>::prover_second_round(&verifier_first_message, prover_state, &mut zk_rng, z_poly.clone(), z_c.clone());

            let (second_commitments, second_commitment_randomnesses) = SonicKZG10::<E, FS>::commit_with_terminator(
                &circuit_proving_key.committer_key,
                second_oracles.iter().map(Into::into),
                terminator,
                //Some(zk_rng),
                Some(&mut zk_rng),
            )?;

            Self::absorb_labeled(&second_commitments, &mut sponge);

            let (verifier_second_msg, verifier_state) =
                AHPForR1CS::<_, MM>::verifier_second_round(verifier_state, &mut sponge)?;
            // --------------------------------------------------------------------

            // --------------------------------------------------------------------
            // Third round
            let (prover_third_message, third_oracles, prover_state) =
                //AHPForR1CS::<_, MM>::prover_third_round(&verifier_second_msg, prover_state, zk_rng)?;
                AHPForR1CS::<_, MM>::prover_third_round(&verifier_second_msg, prover_state, &mut zk_rng)?;

            let (third_commitments, third_commitment_randomnesses) = SonicKZG10::<E, FS>::commit_with_terminator(
                &circuit_proving_key.committer_key,
                third_oracles.iter().map(Into::into),
                terminator,
                //Some(zk_rng),
                Some(&mut zk_rng),
            )?;

            Self::absorb_labeled_with_msg(&third_commitments, &prover_third_message, &mut sponge);

            let (verifier_third_msg, verifier_state) =
                AHPForR1CS::<_, MM>::verifier_third_round(verifier_state, &mut sponge)?;
            // --------------------------------------------------------------------

            // --------------------------------------------------------------------
            // Fourth round
            let first_round_oracles = Arc::clone(prover_state.first_round_oracles.as_ref().unwrap());
            //let fourth_oracles = AHPForR1CS::<_, MM>::prover_fourth_round(&verifier_third_msg, prover_state, zk_rng)?;
            let fourth_oracles = AHPForR1CS::<_, MM>::prover_fourth_round(&verifier_third_msg, prover_state, &mut zk_rng)?;

            let (fourth_commitments, fourth_commitment_randomnesses) = SonicKZG10::<E, FS>::commit_with_terminator(
                &circuit_proving_key.committer_key,
                fourth_oracles.iter().map(Into::into),
                terminator,
                //Some(zk_rng),
                Some(&mut zk_rng),
            )?;

            Self::absorb_labeled(&fourth_commitments, &mut sponge);

            let verifier_state = AHPForR1CS::<_, MM>::verifier_fourth_round(verifier_state, &mut sponge)?;
            // --------------------------------------------------------------------
            // Gather prover polynomials in one vector.
            let polynomials: Vec<_> = circuit_proving_key
                .circuit
                .iter() // 12 items
                .chain(first_round_oracles.iter_for_open()) // 3 * batch_size + (MM::ZK as usize) items
                .chain(second_oracles.iter())// 2 items
                .chain(third_oracles.iter())// 3 items
                .chain(fourth_oracles.iter())// 1 item
                .collect();

            // Gather commitments in one vector.
            let witness_commitments = first_commitments.chunks_exact(3);
            let mask_poly = MM::ZK.then(|| *witness_commitments.remainder()[0].commitment());
            let witness_commitments = witness_commitments
                .map(|c| proof::WitnessCommitments {
                    w: *c[0].commitment(),
                    z_a: *c[1].commitment(),
                    z_b: *c[2].commitment(),
                })
                .collect();
            #[rustfmt::skip]
            let commitments = proof::Commitments {
                witness_commitments,
                mask_poly,

                g_1: *second_commitments[0].commitment(),
                h_1: *second_commitments[1].commitment(),


                g_a: *third_commitments[0].commitment(),
                g_b: *third_commitments[1].commitment(),
                g_c: *third_commitments[2].commitment(),

                h_2: *fourth_commitments[0].commitment(),
            };

            let labeled_commitments: Vec<_> = circuit_proving_key
                .circuit_verifying_key
                .iter()
                .cloned()
                .zip_eq(AHPForR1CS::<E::Fr, MM>::index_polynomial_info().values())
                .map(|(c, info)| LabeledCommitment::new_with_info(info, c))
                .chain(first_commitments.clone().into_iter())
                .chain(second_commitments.into_iter())
                .chain(third_commitments.into_iter())
                .chain(fourth_commitments.into_iter())
                .collect();

            // Gather commitment randomness together.
            let commitment_randomnesses: Vec<Randomness<E>> = circuit_proving_key
                .circuit_commitment_randomness
                .clone()
                .into_iter()
                .chain(first_commitment_randomnesses.clone())
                .chain(second_commitment_randomnesses)
                .chain(third_commitment_randomnesses)
                .chain(fourth_commitment_randomnesses)
                .collect();

            if !MM::ZK {
                let empty_randomness = Randomness::<E>::empty();
                assert!(commitment_randomnesses.iter().all(|r| r == &empty_randomness));
            }

            // Compute the AHP verifier's query set.
            let (query_set, verifier_state) = AHPForR1CS::<_, MM>::verifier_query_set(verifier_state);
            let lc_s = AHPForR1CS::<_, MM>::construct_linear_combinations(
                &public_input,
                &polynomials,
                &prover_third_message,
                &verifier_state,
            )?;

            let mut evaluations = std::collections::BTreeMap::new();
            for (label, (_, point)) in query_set.to_set() {
                if !AHPForR1CS::<E::Fr, MM>::LC_WITH_ZERO_EVAL.contains(&label.as_str()) {
                    let lc = lc_s.get(&label).ok_or_else(|| AHPError::MissingEval(label.to_string()))?;
                    let evaluation = polynomials.get_lc_eval(lc, point)?;
                    evaluations.insert(label, evaluation);
                }
            }

            let evaluations = proof::Evaluations::from_map(&evaluations, batch_size);

            sponge.absorb_nonnative_field_elements(evaluations.to_field_elements(), OptimizationType::Weight);

            let pc_proof = SonicKZG10::<E, FS>::open_combinations(
                &circuit_proving_key.committer_key,
                lc_s.values(),
                polynomials,
                &labeled_commitments,
                &query_set.to_set(),
                &commitment_randomnesses,
                &mut sponge,
            )?;

            Self::terminate(terminator)?;

            let proof = Proof::<E>::new(batch_size, commitments, evaluations, prover_third_message, pc_proof);

            let lock = &buffer.0;
            let cvar = &buffer.1;
            let mut guard = lock.lock().unwrap();
            guard.push_back(Ok(proof));
            cvar.notify_one();
        }
    }

    fn verify_batch_prepared<B: Borrow<Self::VerifierInput>>(
        prepared_verifying_key: &<Self::VerifyingKey as Prepare>::Prepared,
        public_inputs: &[B],
        proof: &Self::Proof,
    ) -> Result<bool, SNARKError> {
        let circuit_verifying_key = &prepared_verifying_key.orig_vk;
        if public_inputs.is_empty() {
            return Err(SNARKError::EmptyBatch);
        }
        let verifier_time = start_timer!(|| "Marlin::Verify");

        let comms = &proof.commitments;
        let proof_has_correct_zk_mode = if MM::ZK {
            proof.pc_proof.is_hiding() & comms.mask_poly.is_some()
        } else {
            !proof.pc_proof.is_hiding() & comms.mask_poly.is_none()
        };
        if !proof_has_correct_zk_mode {
            eprintln!(
                "Found `mask_poly` in the first round when not expected, or proof has incorrect hiding mode ({})",
                proof.pc_proof.is_hiding()
            );
            return Ok(false);
        }

        let batch_size = public_inputs.len();

        let first_round_info = AHPForR1CS::<E::Fr, MM>::first_round_polynomial_info(batch_size);
        let mut first_commitments = comms
            .witness_commitments
            .iter()
            .enumerate()
            .flat_map(|(i, c)| {
                [
                    LabeledCommitment::new_with_info(&first_round_info[&witness_label("w", i)], c.w),
                    LabeledCommitment::new_with_info(&first_round_info[&witness_label("z_a", i)], c.z_a),
                    LabeledCommitment::new_with_info(&first_round_info[&witness_label("z_b", i)], c.z_b),
                ]
            })
            .collect::<Vec<_>>();
        if MM::ZK {
            first_commitments.push(LabeledCommitment::new_with_info(
                first_round_info.get("mask_poly").unwrap(),
                comms.mask_poly.unwrap(),
            ));
        }

        let second_round_info =
            AHPForR1CS::<E::Fr, MM>::second_round_polynomial_info(&circuit_verifying_key.circuit_info);
        let second_commitments = [
            LabeledCommitment::new_with_info(&second_round_info["g_1"], comms.g_1),
            LabeledCommitment::new_with_info(&second_round_info["h_1"], comms.h_1),
        ];
        let third_round_info =
            AHPForR1CS::<E::Fr, MM>::third_round_polynomial_info(&circuit_verifying_key.circuit_info);
        let third_commitments = [
            LabeledCommitment::new_with_info(&third_round_info["g_a"], comms.g_a),
            LabeledCommitment::new_with_info(&third_round_info["g_b"], comms.g_b),
            LabeledCommitment::new_with_info(&third_round_info["g_c"], comms.g_c),
        ];
        let fourth_round_info = AHPForR1CS::<E::Fr, MM>::fourth_round_polynomial_info();
        let fourth_commitments = [LabeledCommitment::new_with_info(&fourth_round_info["h_2"], comms.h_2)];

        let input_domain =
            EvaluationDomain::<E::Fr>::new(circuit_verifying_key.circuit_info.num_public_inputs).unwrap();

        let (padded_public_inputs, public_inputs): (Vec<_>, Vec<_>) = {
            public_inputs
                .iter()
                .map(|input| {
                    let input = input.borrow().to_field_elements().unwrap();
                    let mut new_input = vec![E::Fr::one()];
                    new_input.extend_from_slice(&input);
                    new_input.resize(input.len().max(input_domain.size()), E::Fr::zero());
                    if cfg!(debug_assertions) {
                        println!("Number of padded public variables: {}", new_input.len());
                    }
                    let unformatted = prover::ConstraintSystem::unformat_public_input(&new_input);
                    (new_input, unformatted)
                })
                .unzip()
        };

        let mut sponge =
            Self::init_sponge(batch_size, &circuit_verifying_key.circuit_commitments, &padded_public_inputs);

        // --------------------------------------------------------------------
        // First round
        Self::absorb_labeled(&first_commitments, &mut sponge);
        let (_, verifier_state) =
            AHPForR1CS::<_, MM>::verifier_first_round(circuit_verifying_key.circuit_info, batch_size, &mut sponge)?;
        // --------------------------------------------------------------------

        // --------------------------------------------------------------------
        // Second round
        Self::absorb_labeled(&second_commitments, &mut sponge);
        let (_, verifier_state) = AHPForR1CS::<_, MM>::verifier_second_round(verifier_state, &mut sponge)?;
        // --------------------------------------------------------------------

        // --------------------------------------------------------------------
        // Third round
        Self::absorb_labeled_with_msg(&third_commitments, &proof.msg, &mut sponge);
        let (_, verifier_state) = AHPForR1CS::<_, MM>::verifier_third_round(verifier_state, &mut sponge)?;
        // --------------------------------------------------------------------

        // --------------------------------------------------------------------
        // Fourth round
        Self::absorb_labeled(&fourth_commitments, &mut sponge);
        let verifier_state = AHPForR1CS::<_, MM>::verifier_fourth_round(verifier_state, &mut sponge)?;
        // --------------------------------------------------------------------

        // Collect degree bounds for commitments. Indexed polynomials have *no*
        // degree bounds because we know the committed index polynomial has the
        // correct degree.

        // Gather commitments in one vector.
        let commitments: Vec<_> = circuit_verifying_key
            .iter()
            .cloned()
            .zip_eq(AHPForR1CS::<E::Fr, MM>::index_polynomial_info().values())
            .map(|(c, info)| LabeledCommitment::new_with_info(info, c))
            .chain(first_commitments)
            .chain(second_commitments)
            .chain(third_commitments)
            .chain(fourth_commitments)
            .collect();

        let (query_set, verifier_state) = AHPForR1CS::<_, MM>::verifier_query_set(verifier_state);

        sponge.absorb_nonnative_field_elements(proof.evaluations.to_field_elements(), OptimizationType::Weight);

        let mut evaluations = Evaluations::new();

        for (label, (_point_name, q)) in query_set.to_set() {
            if AHPForR1CS::<E::Fr, MM>::LC_WITH_ZERO_EVAL.contains(&label.as_ref()) {
                evaluations.insert((label, q), E::Fr::zero());
            } else {
                let eval = proof.evaluations.get(&label).ok_or_else(|| AHPError::MissingEval(label.clone()))?;
                evaluations.insert((label, q), eval);
            }
        }

        let lc_s = AHPForR1CS::<_, MM>::construct_linear_combinations(
            &public_inputs,
            &evaluations,
            &proof.msg,
            &verifier_state,
        )?;

        let evaluations_are_correct = SonicKZG10::<E, FS>::check_combinations(
            &circuit_verifying_key.verifier_key,
            lc_s.values(),
            &commitments,
            &query_set.to_set(),
            &evaluations,
            &proof.pc_proof,
            &mut sponge,
        )?;

        if !evaluations_are_correct {
            #[cfg(debug_assertions)]
            eprintln!("SonicKZG10::<E, FS>::Check failed");
        }
        end_timer!(verifier_time, || format!(
            " SonicKZG10::<E, FS>::Check for AHP Verifier linear equations: {}",
            evaluations_are_correct & proof_has_correct_zk_mode
        ));
        Ok(evaluations_are_correct & proof_has_correct_zk_mode)
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::{
        crypto_hash::PoseidonSponge,
        snark::marlin::{fiat_shamir::FiatShamirAlgebraicSpongeRng, MarlinHidingMode, MarlinSNARK},
        SRS,
    };
    use snarkvm_curves::bls12_377::{Bls12_377, Fq, Fr};
    use snarkvm_fields::Field;
    use snarkvm_r1cs::{ConstraintSystem, SynthesisError};
    use snarkvm_utilities::{test_crypto_rng, Uniform};

    use core::ops::MulAssign;

    const ITERATIONS: usize = 10;

    #[derive(Copy, Clone)]
    pub struct Circuit<F: Field> {
        pub a: Option<F>,
        pub b: Option<F>,
        pub num_constraints: usize,
        pub num_variables: usize,
    }

    impl<ConstraintF: Field> ConstraintSynthesizer<ConstraintF> for Circuit<ConstraintF> {
        fn generate_constraints<CS: ConstraintSystem<ConstraintF>>(&self, cs: &mut CS) -> Result<(), SynthesisError> {
            let a = cs.alloc(|| "a", || self.a.ok_or(SynthesisError::AssignmentMissing))?;
            let b = cs.alloc(|| "b", || self.b.ok_or(SynthesisError::AssignmentMissing))?;
            let c = cs.alloc_input(
                || "c",
                || {
                    let mut a = self.a.ok_or(SynthesisError::AssignmentMissing)?;
                    let b = self.b.ok_or(SynthesisError::AssignmentMissing)?;

                    a.mul_assign(&b);
                    Ok(a)
                },
            )?;

            for i in 0..(self.num_variables - 3) {
                let _ = cs.alloc(|| format!("var {}", i), || self.a.ok_or(SynthesisError::AssignmentMissing))?;
            }

            for i in 0..(self.num_constraints - 1) {
                cs.enforce(|| format!("constraint {}", i), |lc| lc + a, |lc| lc + b, |lc| lc + c);
            }

            Ok(())
        }
    }

    type FS = FiatShamirAlgebraicSpongeRng<Fr, Fq, PoseidonSponge<Fq, 6, 1>>;
    type TestSNARK = MarlinSNARK<Bls12_377, FS, MarlinHidingMode, Vec<Fr>>;

    #[test]
    fn marlin_snark_test() {
        let mut rng = test_crypto_rng();

        for _ in 0..ITERATIONS {
            // Construct the circuit.

            let a = Fr::rand(&mut rng);
            let b = Fr::rand(&mut rng);
            let mut c = a;
            c.mul_assign(&b);

            let circ = Circuit { a: Some(a), b: Some(b), num_constraints: 100, num_variables: 25 };

            // Generate the circuit parameters.

            let (pk, vk) = TestSNARK::setup(&circ, &mut SRS::CircuitSpecific(&mut rng)).unwrap();

            // Test native proof and verification.

            let proof = TestSNARK::prove(&pk, &circ, &mut rng).unwrap();

            assert!(TestSNARK::verify(&vk.clone(), &vec![c], &proof).unwrap(), "The native verification check fails.");
        }
    }
}
