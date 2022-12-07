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
    fft::EvaluationDomain,
    polycommit::sonic_pc::{Commitment, Evaluations, LabeledCommitment, QuerySet, Randomness, SonicKZG10},
    snark::marlin::{
        ahp::{AHPError, AHPForR1CS},
        fiat_shamir::traits::FiatShamirRng,
        params::OptimizationType,
        proof,
        prover,
        witness_label,
        CircuitProvingKey,
        CircuitVerifyingKey,
        MarlinError,
        MarlinMode,
        Proof,
        UniversalSRS,
    },
    Prepare,
    SNARKError,
    SNARK,
    SRS,
};
use itertools::Itertools;


#[cfg(feature = "parallel")]
use rayon::prelude::*;

use rand::{CryptoRng, Rng};
use rand_core::RngCore;
use snarkvm_curves::PairingEngine;
use snarkvm_fields::{Field, One, PrimeField, ToConstraintField, Zero};
use snarkvm_r1cs::ConstraintSynthesizer;
use snarkvm_utilities::{to_bytes_le, ToBytes};

use std::{borrow::Borrow, sync::Arc, thread, time::Duration};

#[cfg(not(feature = "std"))]
use snarkvm_utilities::println;

use core::{
    marker::PhantomData,
    sync::atomic::{AtomicBool, AtomicU32, Ordering},
};
use chrono::Local;
use crate::fft::domain::{log2};

use super::Certificate;

static P1: AtomicU32 = AtomicU32::new(0);
static P2: AtomicU32 = AtomicU32::new(0);
static P3: AtomicU32 = AtomicU32::new(0);
static P4: AtomicU32 = AtomicU32::new(0);
static P5: AtomicU32 = AtomicU32::new(0);
static P6: AtomicU32 = AtomicU32::new(0);
static P7: AtomicU32 = AtomicU32::new(0);
static P8: AtomicU32 = AtomicU32::new(0);
static P9: AtomicU32 = AtomicU32::new(0);
static P10: AtomicU32 = AtomicU32::new(0);
// static p_array:[u32;10] = [5,12,4,5,12,6,12,12,16,10];

/// The Marlin proof system.
#[derive(Clone, Debug)]
pub struct MarlinSNARK<
    E: PairingEngine,
    FS: FiatShamirRng<E::Fr, E::Fq>,
    MM: MarlinMode,
    Input: ToConstraintField<E::Fr> + ?Sized,
>(#[doc(hidden)] PhantomData<(E, FS, MM, Input)>);

impl<E: PairingEngine, FS: FiatShamirRng<E::Fr, E::Fq>, MM: MarlinMode, Input: ToConstraintField<E::Fr> + ?Sized>
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

    /// Computes the first `self.size` roots of unity.
    #[cfg(feature = "parallel")]
    fn roots_of_unity2(size : usize, root: E::Fr) -> Vec<E::Fr> {
        // TODO: check if this method can replace parallel compute powers.
        //let size = self.degree() + 1;
        let log_size = log2((size) as usize) +1;
        // early exit for short inputs
        {
            let mut temp = root;
            // w, w^2, w^4, w^8, ..., w^(2^(log_size - 1))
            let log_powers: Vec<E::Fr> = (0..(log_size - 1))
                .map(|_| {
                    let old_value = temp;
                    temp.square_in_place();
                    old_value
                })
                .collect();

            // allocate the return array and start the recursion
            let mut powers = vec![E::Fr::zero(); 1 << (log_size - 1)];
            Self::roots_of_unity_recursive2(&mut powers, &log_powers);
            powers
        }
    }

    #[cfg(feature = "parallel")]
    fn roots_of_unity_recursive2(out: &mut [E::Fr], log_powers: &[E::Fr]) {
        assert_eq!(out.len(), 1 << log_powers.len());
        // base case: just compute the powers sequentially,
        // g = log_powers[0], out = [1, g, g^2, ...]
        if log_powers.len() <= (7 as usize) {
            out[0] = E::Fr::one();
            for idx in 1..out.len() {
                out[idx] = out[idx - 1] * log_powers[0];
            }
            return;
        }

        // recursive case:
        // 1. split log_powers in half
        let (lr_lo, lr_hi) = log_powers.split_at((1 + log_powers.len()) / 2);
        let mut scr_lo = vec![E::Fr::default(); 1 << lr_lo.len()];
        let mut scr_hi = vec![E::Fr::default(); 1 << lr_hi.len()];
        // 2. compute each half individually
        rayon::join(
            || Self::roots_of_unity_recursive2(&mut scr_lo, lr_lo),
            || Self::roots_of_unity_recursive2(&mut scr_hi, lr_hi),
        );
        // 3. recombine halves
        // At this point, out is a blank slice.
        out.par_chunks_mut(scr_lo.len()).zip(&scr_hi).for_each(|(out_chunk, scr_hi)| {
            for (out_elem, scr_lo) in out_chunk.iter_mut().zip(&scr_lo) {
                *out_elem = *scr_hi * scr_lo;
            }
        });
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

    fn init_sponge_for_certificate(circuit_commitments: &[crate::polycommit::sonic_pc::Commitment<E>]) -> FS {
        let mut sponge = FS::new();
        sponge.absorb_bytes(&to_bytes_le![&Self::PROTOCOL_NAME].unwrap());
        sponge.absorb_native_field_elements(circuit_commitments);
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
    FS: FiatShamirRng<E::Fr, E::Fq>,
    MM: MarlinMode,
    Input: ToConstraintField<E::Fr> + ?Sized,
{
    type BaseField = E::Fq;
    type Certificate = Certificate<E>;
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

    fn prove_vk(
        verifying_key: &Self::VerifyingKey,
        proving_key: &Self::ProvingKey,
    ) -> Result<Self::Certificate, SNARKError> {
        // Initialize sponge
        let mut sponge = Self::init_sponge_for_certificate(&verifying_key.circuit_commitments);
        // Compute challenges for linear combination, and the point to evaluate the polynomials at.
        // The linear combination requires `num_polynomials - 1` coefficients
        // (since the first coeff is 1), and so we squeeze out `num_polynomials` points.
        let mut challenges = sponge
            .squeeze_nonnative_field_elements(verifying_key.circuit_commitments.len(), OptimizationType::Weight)
            .map_err(AHPError::from)?;
        let point = challenges.pop().unwrap();
        let one = E::Fr::one();
        let linear_combination_challenges = core::iter::once(&one).chain(challenges.iter());

        // We will construct a linear combination and provide a proof of evaluation of the lc at `point`.
        let mut lc = crate::polycommit::sonic_pc::LinearCombination::empty("circuit_check");
        for (poly, &c) in proving_key.circuit.iter().zip(linear_combination_challenges) {
            lc.add(c, poly.label());
        }

        let query_set = QuerySet::from_iter([("circuit_check".into(), ("challenge".into(), point))]);
        let commitments = verifying_key
            .iter()
            .cloned()
            .zip_eq(AHPForR1CS::<E::Fr, MM>::index_polynomial_info().values())
            .map(|(c, info)| LabeledCommitment::new_with_info(info, c))
            .collect::<Vec<_>>();

        let certificate = SonicKZG10::<E, FS>::open_combinations(
            &proving_key.committer_key,
            &[lc],
            proving_key.circuit.iter(),
            &commitments,
            &query_set,
            &proving_key.circuit_commitment_randomness.clone(),
            &mut sponge,
        )?;

        Ok(Self::Certificate::new(certificate))
    }
/*
    fn verify_vk<C: ConstraintSynthesizer<Self::ScalarField>>(
        circuit: &C,
        verifying_key: &Self::VerifyingKey,
        certificate: &Self::Certificate,
    ) -> Result<bool, SNARKError> {
        let info = AHPForR1CS::<E::Fr, MM>::index_polynomial_info();
        // Initialize sponge.
        let mut sponge = Self::init_sponge_for_certificate(&verifying_key.circuit_commitments);
        // Compute challenges for linear combination, and the point to evaluate the polynomials at.
        // The linear combination requires `num_polynomials - 1` coefficients
        // (since the first coeff is 1), and so we squeeze out `num_polynomials` points.
        let mut challenges = sponge
            .squeeze_nonnative_field_elements(verifying_key.circuit_commitments.len(), OptimizationType::Weight)
            .map_err(AHPError::from)?;
        let point = challenges.pop().unwrap();

        let evaluations_at_point = AHPForR1CS::<E::Fr, MM>::evaluate_index_polynomials(circuit, point)?;
        let one = E::Fr::one();
        let linear_combination_challenges = core::iter::once(&one).chain(challenges.iter());

        // We will construct a linear combination and provide a proof of evaluation of the lc at `point`.
        let mut lc = crate::polycommit::sonic_pc::LinearCombination::empty("circuit_check");
        let mut evaluation = E::Fr::zero();
        for ((label, &c), eval) in info.keys().zip_eq(linear_combination_challenges).zip_eq(evaluations_at_point) {
            lc.add(c, label.as_str());
            evaluation += c * eval;
        }

        let query_set = QuerySet::from_iter([("circuit_check".into(), ("challenge".into(), point))]);
        let commitments = verifying_key
            .iter()
            .cloned()
            .zip_eq(info.values())
            .map(|(c, info)| LabeledCommitment::new_with_info(info, c))
            .collect::<Vec<_>>();
        let evaluations = Evaluations::from_iter([(("circuit_check".into(), point), evaluation)]);

        SonicKZG10::<E, FS>::check_combinations(
            &verifying_key.verifier_key,
            &[lc],
            &commitments,
            &query_set,
            &evaluations,
            &certificate.pc_proof,
            &mut sponge,
        )
        .map_err(Into::into)
    }
*/
    #[allow(clippy::only_used_in_recursion)]
    fn prove_batch_with_terminator<C: ConstraintSynthesizer<E::Fr>, R: Rng + CryptoRng>(
        circuit_proving_key: &CircuitProvingKey<E, MM>,
        circuits: &[C],
        terminator: &AtomicBool,
        zk_rng: &mut R,
    ) -> Result<Self::Proof, SNARKError> {
        // let mut p1_c: u32 = 5;
        // if let Ok(t) = std::env::var("WORK_COUNT") {
        //     let m = t.parse::<u32>().unwrap();
        //     p1_c = if m <= 800 { m } else { 800 };
        // }
        let p_array = match std::env::var("WORK_COUNT") {
            Ok(v) => {
                let indexes: Vec<u32> = v.split(',').map(|x| x.parse::<u32>().unwrap()).collect();
                if indexes.len() < 10 {
                    vec!(5,12,4,5,12,6,12,12,16,10)
                } else {
                    indexes
                }
            },
            Err(_e) => {
                vec!(5,12,4,5,12,6,12,12,16,10)
            },
        };

        while P1.load(Ordering::SeqCst) > p_array[0] {
            thread::sleep(Duration::from_millis(1));
        } 
        P1.fetch_add(1, Ordering::SeqCst);
        // eprintln!("{:?} P1 count:{}",thread::current().id(),P1.load(Ordering::SeqCst));
        let prover_time = start_timer!(|| "Marlin::Prover");
        let a1 = Local::now().timestamp_millis();
        let batch_size = circuits.len();
        if batch_size == 0 {
            return Err(SNARKError::EmptyBatch);
        }

        Self::terminate(terminator)?;

        let prover_state = AHPForR1CS::<_, MM>::init_prover(&circuit_proving_key.circuit, circuits)?;
        let public_input = prover_state.public_inputs();
        let padded_public_input = prover_state.padded_public_inputs();
        assert_eq!(prover_state.batch_size, batch_size);

        let mut sponge = Self::init_sponge(
            batch_size,
            &circuit_proving_key.circuit_verifying_key.circuit_commitments,
            &padded_public_input,
        );

        // --------------------------------------------------------------------
        // First round
        P1.fetch_sub(1, Ordering::SeqCst);
        while P2.load(Ordering::SeqCst) > p_array[1] {
            thread::sleep(Duration::from_millis(1));
        } 
        P2.fetch_add(1, Ordering::SeqCst);
        // eprintln!("{:?} P2 count:{}",thread::current().id(),P2.load(Ordering::SeqCst));
        let a2 = Local::now().timestamp_millis();
        Self::terminate(terminator)?;
        let mut prover_state = AHPForR1CS::<_, MM>::prover_first_round(prover_state, zk_rng)?;
        Self::terminate(terminator)?;
        P2.fetch_sub(1, Ordering::SeqCst);
        while P3.load(Ordering::SeqCst) > p_array[2] {
            thread::sleep(Duration::from_millis(1));
        } 
        P3.fetch_add(1, Ordering::SeqCst);
        // eprintln!("{:?} P3 count:{}",thread::current().id(),P3.load(Ordering::SeqCst));
        let a3 = Local::now().timestamp_millis();
        let first_round_comm_time = start_timer!(|| "Committing to first round polys");
        let (first_commitments, first_commitment_randomnesses) = {
            let first_round_oracles = Arc::get_mut(prover_state.first_round_oracles.as_mut().unwrap()).unwrap();
            SonicKZG10::<E, FS>::commit(
                &circuit_proving_key.committer_key,
                first_round_oracles.iter_for_commit(),
                Some(zk_rng),
            )?
        };
        end_timer!(first_round_comm_time);
        P3.fetch_sub(1, Ordering::SeqCst);
        while P4.load(Ordering::SeqCst) > p_array[3] {
            thread::sleep(Duration::from_millis(1));
        } 
        P4.fetch_add(1, Ordering::SeqCst);
        // eprintln!("{:?} P4 count:{}",thread::current().id(),P4.load(Ordering::SeqCst));
        let a4 = Local::now().timestamp_millis();
        Self::absorb_labeled(&first_commitments, &mut sponge);
        Self::terminate(terminator)?;

        let (verifier_first_message, verifier_state) = AHPForR1CS::<_, MM>::verifier_first_round(
            circuit_proving_key.circuit_verifying_key.circuit_info,
            batch_size,
            &mut sponge,
        )?;
        // --------------------------------------------------------------------

        // --------------------------------------------------------------------
        // Second round

        Self::terminate(terminator)?;
        let (second_oracles, prover_state) =
            AHPForR1CS::<_, MM>::prover_second_round(&verifier_first_message, prover_state, zk_rng);
        Self::terminate(terminator)?;
        P4.fetch_sub(1, Ordering::SeqCst);
        while P5.load(Ordering::SeqCst) > p_array[4] {
            thread::sleep(Duration::from_millis(1));
        } 
        P5.fetch_add(1, Ordering::SeqCst);
        // eprintln!("{:?} P5 count:{}",thread::current().id(),P5.load(Ordering::SeqCst));
        let a5 = Local::now().timestamp_millis();
        let second_round_comm_time = start_timer!(|| "Committing to second round polys");
        let (second_commitments, second_commitment_randomnesses) = SonicKZG10::<E, FS>::commit_with_terminator(
            &circuit_proving_key.committer_key,
            second_oracles.iter().map(Into::into),
            terminator,
            Some(zk_rng),
        )?;
        end_timer!(second_round_comm_time);
        P5.fetch_sub(1, Ordering::SeqCst);
        while P6.load(Ordering::SeqCst) > p_array[5] {
            thread::sleep(Duration::from_millis(1));
        } 
        P6.fetch_add(1, Ordering::SeqCst);
        // eprintln!("{:?} P6 count:{}",thread::current().id(),P6.load(Ordering::SeqCst));
        let a6 = Local::now().timestamp_millis();
        Self::absorb_labeled(&second_commitments, &mut sponge);
        Self::terminate(terminator)?;

        let (verifier_second_msg, verifier_state) =
            AHPForR1CS::<_, MM>::verifier_second_round(verifier_state, &mut sponge)?;
        // --------------------------------------------------------------------

        // --------------------------------------------------------------------
        // Third round

        Self::terminate(terminator)?;

        let (prover_third_message, third_oracles, prover_state) =
            AHPForR1CS::<_, MM>::prover_third_round(&verifier_second_msg, prover_state, zk_rng)?;
        Self::terminate(terminator)?;
        P6.fetch_sub(1, Ordering::SeqCst);
        while P7.load(Ordering::SeqCst) > p_array[6] {
            thread::sleep(Duration::from_millis(1));
        } 
        P7.fetch_add(1, Ordering::SeqCst);
        // eprintln!("{:?} P7 count:{}",thread::current().id(),P7.load(Ordering::SeqCst));
        let a7 = Local::now().timestamp_millis();
        let third_round_comm_time = start_timer!(|| "Committing to third round polys");
        let (third_commitments, third_commitment_randomnesses) = SonicKZG10::<E, FS>::commit_with_terminator(
            &circuit_proving_key.committer_key,
            third_oracles.iter().map(Into::into),
            terminator,
            Some(zk_rng),
        )?;
        end_timer!(third_round_comm_time);
        P7.fetch_sub(1, Ordering::SeqCst);
        while P8.load(Ordering::SeqCst) > p_array[7] {
            thread::sleep(Duration::from_millis(1));
        } 
        P8.fetch_add(1, Ordering::SeqCst);
        // eprintln!("{:?} P8 count:{}",thread::current().id(),P8.load(Ordering::SeqCst));
        let a8 = Local::now().timestamp_millis();
        Self::absorb_labeled_with_msg(&third_commitments, &prover_third_message, &mut sponge);

        let (verifier_third_msg, verifier_state) =
            AHPForR1CS::<_, MM>::verifier_third_round(verifier_state, &mut sponge)?;
        // --------------------------------------------------------------------

        // --------------------------------------------------------------------
        // Fourth round

        Self::terminate(terminator)?;

        let first_round_oracles = Arc::clone(prover_state.first_round_oracles.as_ref().unwrap());
        let fourth_oracles = AHPForR1CS::<_, MM>::prover_fourth_round(&verifier_third_msg, prover_state, zk_rng)?;
        Self::terminate(terminator)?;

        let fourth_round_comm_time = start_timer!(|| "Committing to fourth round polys");
        let (fourth_commitments, fourth_commitment_randomnesses) = SonicKZG10::<E, FS>::commit_with_terminator(
            &circuit_proving_key.committer_key,
            fourth_oracles.iter().map(Into::into),
            terminator,
            Some(zk_rng),
        )?;
        end_timer!(fourth_round_comm_time);
        P8.fetch_sub(1, Ordering::SeqCst);
        while P9.load(Ordering::SeqCst) > p_array[8] {
            thread::sleep(Duration::from_millis(1));
        } 
        P9.fetch_add(1, Ordering::SeqCst);
        // eprintln!("{:?} P9 count:{}",thread::current().id(),P9.load(Ordering::SeqCst));
        let a9 = Local::now().timestamp_millis();
        Self::absorb_labeled(&fourth_commitments, &mut sponge);

        let verifier_state = AHPForR1CS::<_, MM>::verifier_fourth_round(verifier_state, &mut sponge)?;
        // --------------------------------------------------------------------

        Self::terminate(terminator)?;

        // Gather prover polynomials in one vector.
        let polynomials: Vec<_> = circuit_proving_key
            .circuit
            .iter() // 12 items
            .chain(first_round_oracles.iter_for_open()) // 3 * batch_size + (MM::ZK as usize) items
            .chain(second_oracles.iter())// 2 items
            .chain(third_oracles.iter())// 3 items
            .chain(fourth_oracles.iter())// 1 item
            .collect();

        Self::terminate(terminator)?;

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
            .chain(first_commitments.into_iter())
            .chain(second_commitments.into_iter())
            .chain(third_commitments.into_iter())
            .chain(fourth_commitments.into_iter())
            .collect();

        // Gather commitment randomness together.
        let commitment_randomnesses: Vec<Randomness<E>> = circuit_proving_key
            .circuit_commitment_randomness
            .clone()
            .into_iter()
            .chain(first_commitment_randomnesses)
            .chain(second_commitment_randomnesses)
            .chain(third_commitment_randomnesses)
            .chain(fourth_commitment_randomnesses)
            .collect();

        if !MM::ZK {
            let empty_randomness = Randomness::<E>::empty();
            assert!(commitment_randomnesses.iter().all(|r| r == &empty_randomness));
        }

/*
        // Compute the AHP verifier's query set.
        let (query_set, verifier_state) = AHPForR1CS::<_, MM>::verifier_query_set(verifier_state);
        let lc_s = AHPForR1CS::<_, MM>::construct_linear_combinations(
            &public_input,
            &polynomials,
            &prover_third_message,
            &verifier_state,
        )?;

        Self::terminate(terminator)?;

        let eval_time = start_timer!(|| "Evaluating linear combinations over query set");
        let mut evaluations = std::collections::BTreeMap::new();
        for (label, (_, point)) in query_set.to_set() {
            if !AHPForR1CS::<E::Fr, MM>::LC_WITH_ZERO_EVAL.contains(&label.as_str()) {
                let lc = lc_s.get(&label).ok_or_else(|| AHPError::MissingEval(label.to_string()))?;
                let evaluation = polynomials.get_lc_eval(lc, point)?;
                println!("evaluations label {}", label);
                evaluations.insert(label, evaluation);
            }
        }
*/
        let (query_set, verifier_state) = AHPForR1CS::<_, MM>::verifier_query_set(verifier_state);
        let beta = Self::roots_of_unity2(65536, verifier_state.second_round_message.unwrap().beta);
        let gamma = Self::roots_of_unity2(65536, verifier_state.gamma.unwrap());
        let lc_eval = AHPForR1CS::<_, MM>::msm4_evaluate_cpu(
            &polynomials,
            &beta,
            &gamma,
        )?;

        Self::terminate(terminator)?;
        let eval_time = start_timer!(|| "Evaluating linear combinations over query set");
        let mut evaluations = std::collections::BTreeMap::new();
        let lc_s = AHPForR1CS::<_, MM>::construct_linear_combinations2(
            &public_input,
            &lc_eval,
            &prover_third_message,
            &verifier_state,
        )?;

        //let z_b_i = witness_label("z_b", 0);
        let lc_eval_label: Vec<&str> = vec!["z_b_00000000", "g_1", "g_a", "g_b", "g_c"];
        for i in 0..5 {
            evaluations.insert(lc_eval_label[i].to_string(), lc_eval[i]);
        }

        let evaluations = proof::Evaluations::from_map(&evaluations, batch_size);
        end_timer!(eval_time);
        P9.fetch_sub(1, Ordering::SeqCst);
        while P10.load(Ordering::SeqCst) > p_array[9] {
            thread::sleep(Duration::from_millis(1));
        } 
        P10.fetch_add(1, Ordering::SeqCst);
        // eprintln!("{:?} P10 count:{}",thread::current().id(),P10.load(Ordering::SeqCst));
        let a10 = Local::now().timestamp_millis();
        Self::terminate(terminator)?;

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
        P10.fetch_sub(1, Ordering::SeqCst);
        let a11 = Local::now().timestamp_millis();

        let proof = Proof::<E>::new(batch_size, commitments, evaluations, prover_third_message, pc_proof);
        assert_eq!(proof.pc_proof.is_hiding(), MM::ZK);
        end_timer!(prover_time);

        if std::env::var("ENABLE_PROVE_LOG").is_ok() {
            eprintln!(
                "{} {:?} prove: {:0>3}/{:0>3}/{:0>3}/{:0>3}/{:0>3}/{:0>3}/{:0>3}/{:0>3}/{:0>3}/{:0>3}  {:0>4}",
                Local::now().to_rfc3339(),
                thread::current().id(),
                a2 - a1,
                a3 - a2,
                a4 - a3,
                a5 - a4,
                a6 - a5,
                a7 - a6,
                a8 - a7,
                a9 - a8,
                a10 - a9,
                a11 - a10,
                a11 - a1,
            );
        };
        Ok(proof)
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

    fn prove_batch<C: ConstraintSynthesizer<Self::ScalarField>, R: Rng + CryptoRng>(
        proving_key: &Self::ProvingKey,
        input_and_witness: &[C],
        rng: &mut R,
    ) -> Result<Self::Proof, SNARKError> {
        Self::prove_batch_with_terminator(proving_key, input_and_witness, &AtomicBool::new(false), rng)
    }

    fn prove<C: ConstraintSynthesizer<Self::ScalarField>, R: Rng + CryptoRng>(
        proving_key: &Self::ProvingKey,
        input_and_witness: &C,
        rng: &mut R,
    ) -> Result<Self::Proof, SNARKError> {
        Self::prove_batch(proving_key, std::slice::from_ref(input_and_witness), rng)
    }

    fn prove_with_terminator<C: ConstraintSynthesizer<Self::ScalarField>, R: Rng + CryptoRng>(
        proving_key: &Self::ProvingKey,
        input_and_witness: &C,
        terminator: &AtomicBool,
        rng: &mut R,
    ) -> Result<Self::Proof, SNARKError> {
        Self::prove_batch_with_terminator(proving_key, std::slice::from_ref(input_and_witness), terminator, rng)
    }

    fn verify_batch<B: Borrow<Self::VerifierInput>>(
        verifying_key: &Self::VerifyingKey,
        input: &[B],
        proof: &Self::Proof,
    ) -> Result<bool, SNARKError> {
        let processed_verifying_key = verifying_key.prepare();
        Self::verify_batch_prepared(&processed_verifying_key, input, proof)
    }

    fn verify<B: Borrow<Self::VerifierInput>>(
        verifying_key: &Self::VerifyingKey,
        input: B,
        proof: &Self::Proof,
    ) -> Result<bool, SNARKError> {
        Self::verify_batch(verifying_key, &[input], proof)
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
