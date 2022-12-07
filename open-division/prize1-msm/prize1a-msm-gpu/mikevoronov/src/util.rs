// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use ark_ec::{AffineCurve, ProjectiveCurve};
//use ark_ff::{BigInteger256, ToBytes};
use ark_std::UniformRand;

/*use rayon::prelude::*;

pub fn generate_points_scalars<G: AffineCurve>(len: usize, batch_size: usize) -> (Vec<G>, Vec<G::ScalarField>) {
    let rand_gen: usize = len;

    let num_threads = rayon::current_num_threads();
    let step = (len + num_threads - 1) / num_threads;
    println!(
        "start points generation in {} threads, each will generate {} points",
        num_threads, step
    );
    let projective_points = (0..rand_gen)
        .into_par_iter()
        .step_by(step)
        .map(|_| {
            let mut rng = ChaCha20Rng::from_entropy();
            (0..step)
                .map(|_| G::Projective::rand(&mut rng))
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect::<Vec<_>>();

    let mut points =
        <G::Projective as ProjectiveCurve>::batch_normalization_into_affine(&projective_points);

    // Sprinkle in some infinity points
    //points[3] = G::zero();

    println!("start scalars generation");
    let mut rng = ChaCha20Rng::from_entropy();
    let scalars = (0..rand_gen * batch_size)
        .map(|_| G::ScalarField::rand(&mut rng))
        .collect::<Vec<_>>();

    (points, scalars)
}*/

use rustacuda::memory::*;

pub fn generate_points_scalars<G: AffineCurve>(
    len: usize,
    batch_size: usize,
) -> (Vec<G>, Vec<G::ScalarField>) {
    let rand_gen: usize = 1 << 11;
    let mut rng = ChaCha20Rng::from_entropy();

    let mut points =
        <G::Projective as ProjectiveCurve>::batch_normalization_into_affine(
            &(0..rand_gen)
                .map(|_| G::Projective::rand(&mut rng))
                .collect::<Vec<_>>(),
        );
    // Sprinkle in some infinity points
    //points[3] = G::zero();
    while points.len() < len {
        points.append(&mut points.clone());
    }

    let scalars_count = len * batch_size;
    let scalars = (0..scalars_count)
        .map(|_| G::ScalarField::rand(&mut rng))
        .collect::<Vec<_>>();

    (points, scalars)
}
