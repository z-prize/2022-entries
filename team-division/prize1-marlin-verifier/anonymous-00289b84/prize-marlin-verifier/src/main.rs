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

use std::time::{Duration, Instant};

use snarkvm_algorithms::{
    crypto_hash::PoseidonSponge,
    snark::marlin::{
        ahp::AHPForR1CS, CircuitVerifyingKey, FiatShamirAlgebraicSpongeRng, MarlinHidingMode,
        MarlinSNARK,
    },
    SNARK,
};
use snarkvm_curves::bls12_377::{Bls12_377, Fq, Fr};
use snarkvm_fields::Field;
use snarkvm_r1cs::{errors::SynthesisError, ConstraintSynthesizer, ConstraintSystem};
use snarkvm_utilities::Uniform;

use rand::{self, thread_rng, Rng};
use rstat::{normal::UvNormal, Distribution};

//********************************************************************
// TODO - The following function may be modified by contestants
//********************************************************************
fn verify_group(
    group_proofs: &Vec<((<MarlinInst as SNARK>::Proof, Vec<Fr>), bool)>,
    circuit_vk: &CircuitVerifyingKey<Bls12_377, MarlinHidingMode>,
) -> Vec<bool> {
    use std::sync::Arc;
    use snarkvm_algorithms::{Prepare, snark::marlin::FiatShamirRng};
    use snarkvm_fields::{PoseidonParameters, Fp384};
    use snarkvm_curves::bls12_377::FqParameters;
    use lazy_static::lazy_static;

    lazy_static!{
        pub static ref FS_PARAMS: Arc<PoseidonParameters<Fp384<FqParameters>, 6, 1>> = {
            <FiatShamirAlgebraicSpongeRng<Fr, Fq, PoseidonSponge<Fq, 6, 1>> as FiatShamirRng<_, _>>::parameters()
        };
    }
    let fs_parameters = &FS_PARAMS;
    let prepared_vk = circuit_vk.prepare();
    let mut results = vec![];
    group_proofs.iter().for_each(|((proof, inputs), _faulty)| {
        let verified = MarlinInst::verify_batch_with_prepared_vk(fs_parameters, &prepared_vk, &inputs, &proof).unwrap();
        results.push(verified);
    });
    results
}

//********************************************************************
// All code below here may not be modified
//********************************************************************

type MarlinInst = MarlinSNARK<Bls12_377, FS, MarlinHidingMode, Fr>;
type FS = FiatShamirAlgebraicSpongeRng<Fr, Fq, PoseidonSponge<Fq, 6, 1>>;

const CHALLENGE_DURATION: u64 = 10;

#[derive(Copy, Clone)]
pub struct Benchmark<F: Field> {
    pub a: Option<F>,
    pub b: Option<F>,
    pub num_constraints: usize,
    pub num_variables: usize,
}

impl<ConstraintF: Field> ConstraintSynthesizer<ConstraintF> for Benchmark<ConstraintF> {
    fn generate_constraints<CS: ConstraintSystem<ConstraintF>>(
        &self,
        cs: &mut CS,
    ) -> Result<(), SynthesisError> {
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
            let _ = cs.alloc(
                || format!("var {}", i),
                || self.a.ok_or(SynthesisError::AssignmentMissing),
            )?;
        }

        for i in 0..(self.num_constraints - 1) {
            cs.enforce(
                || format!("constraint {}", i),
                |lc| lc + a,
                |lc| lc + b,
                |lc| lc + c,
            );
        }

        Ok(())
    }
}

fn snark_prove(
    num_batch: usize,
    pk: &<MarlinInst as SNARK>::ProvingKey,
) -> (<MarlinInst as SNARK>::Proof, Vec<Fr>) {
    let num_constraints = 100;
    let num_variables = 100;
    let rng = &mut thread_rng();

    let mut circuits = vec![];
    let mut inputs = vec![];
    for _ in 0..num_batch {
        let x = Fr::rand(rng);
        let y = Fr::rand(rng);

        let circuit = Benchmark::<Fr> {
            a: Some(x),
            b: Some(y),
            num_constraints,
            num_variables,
        };

        circuits.push(circuit);
        inputs.push(x * y);
    }

    (MarlinInst::prove_batch(pk, &circuits, rng).unwrap(), inputs)
}

fn check_group_results(
    group_proofs: &Vec<((<MarlinInst as SNARK>::Proof, Vec<Fr>), bool)>,
    results: &Vec<bool>,
) {
    for (((_p, _i), faulty), result) in group_proofs.iter().zip(results) {
        if *faulty == *result {
            panic!("Verifier algorithm failed!");
        }
    }
}

fn main() {
    println!("Running initial setup...");
    let max_degree = AHPForR1CS::<Fr, MarlinHidingMode>::max_degree(1000, 1000, 1000).unwrap();
    let universal_srs = MarlinInst::universal_setup(&max_degree, &mut thread_rng()).unwrap();
    let num_constraints = 100;
    let num_variables = 100;
    let x = Fr::rand(&mut thread_rng());
    let y = Fr::rand(&mut thread_rng());
    let circuit = Benchmark::<Fr> {
        a: Some(x),
        b: Some(y),
        num_constraints,
        num_variables,
    };
    let params = MarlinInst::circuit_setup(&universal_srs, &circuit).unwrap();

    println!("Generating round 1 proofs...");
    let round_1_proofs = {
        let mut proofs = vec![];
        for _ in 0..10 {
            let mut faulty = false;
            let mut proof = snark_prove(100, &params.0);
            // Randomly introduce faulty proofs with a 1/5 probability.
            if thread_rng().gen_range::<u8, _>(0..5) == 0 {
                // Set one of the commitments to a wrong value.
                proof.0.commitments.g_1.0 = -proof.0.commitments.g_1.0;
                faulty = true;
            }

            proofs.push((proof, faulty));
        }

        proofs
    };

    println!("Generating round 2 proofs...");
    let round_2_proofs = {
        let mut proofs = vec![];
        for _ in 0..100 {
            let mut faulty = false;
            let mut proof = snark_prove(10, &params.0);
            // Randomly introduce faulty proofs with a 1/50 probability.
            if thread_rng().gen_range::<u8, _>(0..50) == 0 {
                // Set one of the commitments to a wrong value.
                proof.0.commitments.g_1.0 = -proof.0.commitments.g_1.0;
                faulty = true;
            }

            proofs.push((proof, faulty));
        }

        proofs
    };

    println!("Generating round 3 proofs...");
    let round_3_proofs = {
        let mut proofs = vec![];
        let normal = UvNormal::new(50.0, 25.0).unwrap();
        let mut rng = old_rand::thread_rng();
        for _ in 0..20 {
            let sample = normal.sample(&mut rng);
            let mut faulty = false;
            let mut proof = snark_prove(sample as usize, &params.0);
            // Randomly introduce faulty proofs with a 1/10 probability.
            if thread_rng().gen_range::<u8, _>(0..10) == 0 {
                // Set one of the commitments to a wrong value.
                proof.0.commitments.g_1.0 = -proof.0.commitments.g_1.0;
                faulty = true;
            }

            proofs.push((proof, faulty));
        }

        proofs
    };

    println!("Done! Running verifying challenge...");
    let mut round_counter = 0;
    let now = Instant::now();
    loop {
        let results = verify_group(&round_1_proofs, &params.1);
        check_group_results(&round_1_proofs, &results);
        round_counter += 1;
        if now.elapsed() > Duration::from_secs(CHALLENGE_DURATION) {
            break;
        }

        let results = verify_group(&round_2_proofs, &params.1);
        check_group_results(&round_2_proofs, &results);
        round_counter += 1;
        if now.elapsed() > Duration::from_secs(CHALLENGE_DURATION) {
            break;
        }

        let results = verify_group(&round_3_proofs, &params.1);
        check_group_results(&round_3_proofs, &results);
        round_counter += 1;
        if now.elapsed() > Duration::from_secs(CHALLENGE_DURATION) {
            break;
        }
    }

    println!("Finished!");
    println!("{round_counter} rounds finished");
}
