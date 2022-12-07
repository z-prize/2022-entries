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
    fft::DensePolynomial,
    polycommit::{kzg10, optional_rng::OptionalRng, PCError},
    snark::marlin::FiatShamirRng,
};
use hashbrown::HashMap;
use itertools::Itertools;
use snarkvm_curves::traits::{PairingEngine, ProjectiveCurve};
use snarkvm_fields::{One, Zero};

use core::{
    convert::TryInto,
    sync::atomic::{AtomicBool, Ordering},
};
use rand_core::{RngCore, SeedableRng};
use std::collections::{BTreeMap, BTreeSet};

use ec_gpu_common::{GpuPolyContainer, PolyKernel, PrimeField as GpuPrimeField};

use crate::polycommit::sonic_pc::*;

impl<E: PairingEngine, S: FiatShamirRng<E::Fr, E::Fq>> SonicKZG10<E, S> {
    #[allow(clippy::type_complexity)]
    pub fn commit_on_gpu<'b, Fg: GpuPrimeField>(
        kern: &PolyKernel<Fg>,
        gpu_container: &mut GpuPolyContainer<Fg>,
        ck: &CommitterKey<E>,
        polynomials: impl IntoIterator<Item = LabeledPolynomialWithBasis<'b, E::Fr>>,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<(Vec<LabeledCommitment<Commitment<E>>>, Vec<Randomness<E>>), PCError> {
        Self::commit_with_terminator_on_gpu(kern, gpu_container, ck, polynomials, &AtomicBool::new(false), rng)
    }

    pub fn open_combinations_on_gpu<'a, Fg: GpuPrimeField>(
        kern: &PolyKernel<Fg>,
        gpu_container: &mut GpuPolyContainer<Fg>,
        ck: &CommitterKey<E>,
        linear_combinations: impl IntoIterator<Item = &'a LinearCombination<E::Fr>>,
        polynomials: impl IntoIterator<Item = &'a LabeledPolynomial<E::Fr>>,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Commitment<E>>>,
        query_set: &QuerySet<E::Fr>,
        rands: impl IntoIterator<Item = &'a Randomness<E>>,
        fs_rng: &mut S,
    ) -> Result<BatchLCProof<E>, PCError>
    where
        Randomness<E>: 'a,
        Commitment<E>: 'a,
    {
        let label_map = polynomials
            .into_iter()
            .zip_eq(rands)
            .zip_eq(commitments)
            .map(|((p, r), c)| (p.label(), (p, r, c)))
            .collect::<BTreeMap<_, _>>();

        let mut lc_polynomials = Vec::new();
        let mut lc_randomness = Vec::new();
        let mut lc_commitments = Vec::new();
        let mut lc_info = Vec::new();

        let mut gpu_combined_poly = gpu_container.ask_for_max(kern)?;

        for lc in linear_combinations {
            let lc_label = lc.label().to_string();

            gpu_combined_poly.fill_with_zero()?;

            let mut degree_bound = None;
            let mut hiding_bound = None;

            let mut randomness = Randomness::empty();
            let mut coeffs_and_comms = Vec::new();

            let mut max_degree = 0;

            let num_polys = lc.len();
            for (coeff, label) in lc.iter().filter(|(_, l)| !l.is_one()) {
                let label: &String = label.try_into().expect("cannot be one!");

                // we use the labeled polynomial here
                let gpu_poly = gpu_container.find(&kern, label)?;

                let &(cur_poly, cur_rand, cur_comm) =
                    label_map.get(label as &str).ok_or(PCError::MissingPolynomial { label: label.to_string() })?;

                max_degree = std::cmp::max(max_degree, gpu_poly.size());

                if num_polys == 1 && cur_poly.degree_bound().is_some() {
                    assert!(coeff.is_one(), "Coefficient must be one for degree-bounded equations");
                    degree_bound = cur_poly.degree_bound();
                } else if cur_poly.degree_bound().is_some() {
                    return Err(PCError::EquationHasDegreeBounds(lc_label));
                }
                // Some(_) > None, always.
                hiding_bound = core::cmp::max(hiding_bound, cur_poly.hiding_bound());

                gpu_combined_poly.add_assign_scale(&gpu_poly, coeff)?;

                gpu_container.save(label, gpu_poly)?;

                randomness += (*coeff, cur_rand);
                coeffs_and_comms.push((*coeff, cur_comm.commitment()));
            }
            let summed_poly = DensePolynomial::zero();

            let mut gpu_lc_poly = gpu_container.ask_for(kern, max_degree)?;
            gpu_lc_poly.copy_from_gpu(&gpu_combined_poly)?;
            gpu_container.save(&lc_label, gpu_lc_poly)?;

            let lc_poly = LabeledPolynomial::new(lc_label.clone(), summed_poly, degree_bound, hiding_bound);

            lc_polynomials.push(lc_poly);
            lc_randomness.push(randomness);
            lc_commitments.push(Self::combine_commitments(coeffs_and_comms));
            lc_info.push((lc_label, degree_bound));
        }

        gpu_container.recycle(gpu_combined_poly)?;

        let comms = Self::normalize_commitments(lc_commitments);
        let lc_commitments = lc_info
            .into_iter()
            .zip_eq(comms)
            .map(|((label, d), c)| LabeledCommitment::new(label, c, d))
            .collect::<Vec<_>>();

        let proof = Self::batch_open_on_gpu(
            kern,
            gpu_container,
            ck,
            lc_polynomials.iter(),
            lc_commitments.iter(),
            query_set,
            lc_randomness.iter(),
            fs_rng,
        )?;
        kern.sync()?;

        Ok(BatchLCProof { proof, evaluations: None })
    }

    /// Outputs a commitment to `polynomial`.
    #[allow(clippy::type_complexity)]
    #[allow(clippy::format_push_string)]
    pub fn commit_with_terminator_on_gpu<'a, Fg: GpuPrimeField>(
        kern: &PolyKernel<Fg>,
        gpu_container: &mut GpuPolyContainer<Fg>,
        ck: &CommitterKey<E>,
        polynomials: impl IntoIterator<Item = LabeledPolynomialWithBasis<'a, E::Fr>>,
        terminator: &AtomicBool,
        rng: Option<&mut dyn RngCore>,
    ) -> Result<(Vec<LabeledCommitment<Commitment<E>>>, Vec<Randomness<E>>), PCError> {
        let rng = &mut OptionalRng(rng);
        let commit_time = start_timer!(|| "Committing to polynomials");
        let mut labeled_comms: Vec<LabeledCommitment<Commitment<E>>> = Vec::new();
        let mut randomness: Vec<Randomness<E>> = Vec::new();
        let mut results: Vec<Result<_, PCError>> = Vec::new();

        for p in polynomials {
            if terminator.load(Ordering::Relaxed) {
                return Err(PCError::Terminated);
            }
            let seed = rng.0.as_mut().map(|r| {
                let mut seed = [0u8; 32];
                r.fill_bytes(&mut seed);
                seed
            });

            kzg10::KZG10::<E>::check_degrees_and_bounds(
                ck.supported_degree(),
                ck.max_degree,
                ck.enforced_degree_bounds.as_deref(),
                p.clone(),
            )?;
            let degree_bound = p.degree_bound();
            let hiding_bound = p.hiding_bound();
            let label = p.label().to_string();

            let mut rng = seed.map(rand::rngs::StdRng::from_seed);
            add_to_trace!(|| "PC::Commit", || format!(
                "Polynomial {} of degree {}, degree bound {:?}, and hiding bound {:?}",
                label,
                p.degree(),
                degree_bound,
                hiding_bound,
            ));

            let label = p.label();

            let p_iter = p.sum();
            let mut comms_and_rands = Vec::new();

            for inner_p in p_iter {
                let rng_ref = rng.as_mut().map(|s| s as _);
                let res = match inner_p {
                    PolynomialWithBasis::Lagrange { evaluations } => {
                        let domain = crate::fft::EvaluationDomain::new(evaluations.evaluations.len()).unwrap();
                        let lagrange_basis =
                            ck.lagrange_basis(domain).ok_or(PCError::UnsupportedLagrangeBasisSize(domain.size()))?;
                        assert!(domain.size().is_power_of_two());
                        assert!(lagrange_basis.size().is_power_of_two());

                        let res = kzg10::KZG10::commit_lagrange_on_gpu(
                            kern,
                            gpu_container,
                            &lagrange_basis,
                            &evaluations.evaluations,
                            label,
                            hiding_bound,
                            terminator,
                            rng_ref,
                        );

                        res
                    }
                    PolynomialWithBasis::Monomial { polynomial, degree_bound } => {
                        let powers = if let Some(degree_bound) = degree_bound {
                            ck.shifted_powers_of_beta_g(degree_bound).unwrap()
                        } else {
                            ck.powers()
                        };

                        let mut is_g = false;

                        let start = if let Some(degree_bound) = degree_bound {
                            let max_bound = ck.enforced_degree_bounds.as_ref().unwrap().last().unwrap();
                            max_bound - degree_bound
                        } else {
                            is_g = true;
                            0
                        };

                        let res = kzg10::KZG10::commit_on_gpu(
                            kern,
                            gpu_container,
                            &powers,
                            &polynomial,
                            label,
                            hiding_bound,
                            terminator,
                            None,
                            start,
                            is_g,
                        );
                        res
                    }
                };
                comms_and_rands.push(res?);
            }

            let (comm, rand) =
                comms_and_rands.into_iter().fold((E::G1Projective::zero(), Randomness::empty()), |mut a, b| {
                    a.0.add_assign_mixed(&b.0 .0);
                    a.1 += (E::Fr::one(), &b.1);
                    a
                });

            let comm = kzg10::Commitment(comm.to_affine());

            results.push(Ok((LabeledCommitment::new(label.to_string(), comm, degree_bound), rand)));
        }

        for result in results {
            let (comm, rand) = result?;
            labeled_comms.push(comm);
            randomness.push(rand);
        }

        end_timer!(commit_time);
        Ok((labeled_comms, randomness))
    }

    pub fn batch_open_on_gpu<'a, Fg: GpuPrimeField>(
        kern: &PolyKernel<Fg>,
        gpu_container: &mut GpuPolyContainer<Fg>,
        ck: &CommitterKey<E>,
        labeled_polynomials: impl IntoIterator<Item = &'a LabeledPolynomial<E::Fr>>,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Commitment<E>>>,
        query_set: &QuerySet<E::Fr>,
        rands: impl IntoIterator<Item = &'a Randomness<E>>,
        fs_rng: &mut S,
    ) -> Result<BatchProof<E>, PCError>
    where
        Randomness<E>: 'a,
        Commitment<E>: 'a,
    {
        let poly_rand_comm: HashMap<_, _> = labeled_polynomials
            .into_iter()
            .zip_eq(rands)
            .zip_eq(commitments.into_iter())
            .map(|((poly, r), comm)| (poly.label(), (poly, r, comm)))
            .collect();

        let open_time = start_timer!(|| format!(
            "Opening {} polynomials at query set of size {}",
            poly_rand_comm.len(),
            query_set.len(),
        ));

        let mut query_to_labels_map = BTreeMap::new();

        for (label, (point_name, point)) in query_set.iter() {
            let labels = query_to_labels_map.entry(point_name).or_insert((point, BTreeSet::new()));
            labels.1.insert(label);
        }

        let mut batch_proof = BatchProof { 0: Vec::new() };

        for (_point_name, (&query, labels)) in query_to_labels_map.into_iter() {
            let mut query_polys = Vec::with_capacity(labels.len());
            let mut query_rands = Vec::with_capacity(labels.len());
            let mut query_comms = Vec::with_capacity(labels.len());

            for label in labels {
                let (polynomial, rand, comm) =
                    poly_rand_comm.get(label as &str).ok_or(PCError::MissingPolynomial { label: label.to_string() })?;

                query_polys.push(*polynomial);
                query_rands.push(*rand);
                query_comms.push(*comm);
            }

            let (polynomial_name, rand) =
                Self::combine_for_open_on_gpu(kern, gpu_container, ck, query_polys, query_rands, fs_rng)?;

            let proof_time = start_timer!(|| "Creating proof");
            let proof = kzg10::KZG10::open_on_gpu(kern, gpu_container, &ck.powers(), query, &rand, &polynomial_name)?;

            end_timer!(proof_time);
            batch_proof.0.push(proof);
        }

        end_timer!(open_time);

        Ok(batch_proof)
    }

    pub fn combine_for_open_on_gpu<'a, Fg: GpuPrimeField>(
        kern: &PolyKernel<Fg>,
        gpu_container: &mut GpuPolyContainer<Fg>,
        ck: &CommitterKey<E>,
        labeled_polynomials: impl IntoIterator<Item = &'a LabeledPolynomial<E::Fr>>,
        rands: impl IntoIterator<Item = &'a Randomness<E>>,
        fs_rng: &mut S,
    ) -> Result<(String, Randomness<E>), PCError>
    where
        Randomness<E>: 'a,
        Commitment<E>: 'a,
    {
        let mut gpu_combined_poly = gpu_container.ask_for_max(&kern)?;
        gpu_combined_poly.fill_with_zero()?;

        let mut combined_rand = Randomness::empty();

        let mut max_degree = 0;
        let mut combined_name = String::new();

        labeled_polynomials.into_iter().zip_eq(rands).for_each(|(p, r)| {
            let enforced_degree_bounds: Option<&[usize]> = ck.enforced_degree_bounds.as_deref();

            kzg10::KZG10::<E>::check_degrees_and_bounds(
                ck.supported_degree(),
                ck.max_degree,
                enforced_degree_bounds,
                p,
            )
            .unwrap();
            let challenge = fs_rng.squeeze_short_nonnative_field_element().unwrap();

            let gpu_poly = gpu_container.find(&kern, p.label()).unwrap();
            gpu_combined_poly.add_assign_scale(&gpu_poly, &challenge).unwrap();
            max_degree = std::cmp::max(max_degree, gpu_poly.size());

            combined_rand += (challenge, r);
            combined_name += p.label();
            combined_name += "+";

            gpu_container.save(p.label(), gpu_poly).unwrap();
        });

        let mut gpu_saved_poly = gpu_container.ask_for(&kern, max_degree)?;
        gpu_saved_poly.copy_from_gpu(&gpu_combined_poly)?;

        // don't forget to do cleaning!
        gpu_container.save(&combined_name, gpu_saved_poly)?;
        gpu_container.recycle(gpu_combined_poly)?;

        Ok((combined_name, combined_rand))
    }
}
