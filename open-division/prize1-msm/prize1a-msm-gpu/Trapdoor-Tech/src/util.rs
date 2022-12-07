// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use ark_ec::{AffineCurve, ProjectiveCurve};
use ark_std::UniformRand;

pub fn generate_points_scalars<G: AffineCurve>(
    len: usize,
    batch_size: usize,
) -> (Vec<G>, Vec<G::ScalarField>) {
    let rand_gen: usize = std::cmp::min(len, 1 << 16);

    let mut rng = ChaCha20Rng::from_entropy();

    let random_points =
        <G::Projective as ProjectiveCurve>::batch_normalization_into_affine(
            &(0..rand_gen)
                .map(|_| G::Projective::rand(&mut rng))
                .collect::<Vec<_>>(),
        );

    let mut points = random_points.clone();

    // Sprinkle in some infinity points
    points[0] = G::zero();
    while points.len() < len {
        points.append(&mut random_points.clone());
        println!("appending {} points", random_points.len());
    }

    // let scalars = (0..len * batch_size)
    //     .map(|_| G::ScalarField::rand(&mut rng))
    //     .collect::<Vec<_>>();

    let random_scalars = (0..rand_gen)
        .map(|_| G::ScalarField::rand(&mut rng))
        .collect::<Vec<_>>();

    let mut scalars = random_scalars.clone();

    while scalars.len() < len * batch_size {
        scalars.append(&mut random_scalars.clone());
        println!("appending {} scalars", random_scalars.len());
    }

    (points, scalars)
}
