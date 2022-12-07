// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use ark_bls12_377::G1Affine;
use fpga_msm::{multi_scalar_mult, multi_scalar_mult_init};
use rand::{prelude::StdRng, RngCore, SeedableRng};

use std::str::FromStr;

#[test]
fn msm_correctness() {
    let test_npow = std::env::var("TEST_NPOW").unwrap_or("15".to_string());
    let npoints_npow = i32::from_str(&test_npow).unwrap();
    let rng = match std::env::var("MSM_SEED") {
        Ok(seed) => rand::rngs::StdRng::seed_from_u64(seed.parse().unwrap()),
        Err(_) => {
            let seed = StdRng::from_entropy().next_u64();
            eprintln!("using seed: {}", seed);
            StdRng::seed_from_u64(seed)
        }
    };

    let batches = 4;
    eprintln!("Generating test data");
    let (points, scalars, results) =
        fpga_msm::gen::generate::<G1Affine>(rng, 1 << npoints_npow, batches);

    eprintln!("Initializing context");
    let mut context = multi_scalar_mult_init(points.as_slice());
    eprintln!("Running test");
    let msm_results = multi_scalar_mult(&mut context, points.as_slice(), scalars.as_slice());

    for i in 0..batches {
        assert_eq!(&msm_results[i], &results[i]);
    }
}
