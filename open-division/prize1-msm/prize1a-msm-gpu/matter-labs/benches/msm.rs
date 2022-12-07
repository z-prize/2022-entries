// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use criterion::{criterion_group, criterion_main, Criterion};

use ark_bls12_377::G1Affine;
use fpga_msm::{multi_scalar_mult, multi_scalar_mult_init};
use rand::{prelude::StdRng, RngCore, SeedableRng};

use std::str::FromStr;

fn criterion_benchmark(c: &mut Criterion) {
    let bench_npow = std::env::var("BENCH_NPOW").unwrap_or("26".to_string());
    let npoints_npow = i32::from_str(&bench_npow).unwrap();
    let rng = match std::env::var("MSM_SEED") {
        Ok(seed) => rand::rngs::StdRng::seed_from_u64(seed.parse().unwrap()),
        Err(_) => {
            let seed = StdRng::from_entropy().next_u64();
            eprintln!("using seed: {}", seed);
            StdRng::seed_from_u64(seed)
        }
    };

    let batches = 4;
    let (points, scalars, _results) =
        fpga_msm::gen::generate::<G1Affine>(rng, 1 << npoints_npow, batches);

    let mut context = multi_scalar_mult_init(points.as_slice());

    let mut group = c.benchmark_group("FPGA-MSM");
    group.sample_size(10);

    let name = format!("2**{}x{}", npoints_npow, batches);
    group.bench_function(name, |b| {
        b.iter(|| {
            let _ = multi_scalar_mult(&mut context, &points.as_slice(),
                &scalars);
        })
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
