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

use std::{borrow::Borrow, collections::BTreeMap};

use crate::fft::domain::{FFTPrecomputation, IFFTPrecomputation};
use crate::fft::cuda_fft::{ars_radix_fft, ars_radix_ifft};

/// A struct that helps multiply a batch of polynomials
use super::*;
use snarkvm_utilities::{cfg_iter, cfg_iter_mut, ExecutionPool};

#[derive(Default)]
pub struct PolyMultiplier<'a, F: PrimeField> {
    polynomials: Vec<(String, Cow<'a, DensePolynomial<F>>)>,
    evaluations: Vec<(String, Cow<'a, crate::fft::Evaluations<F>>)>,
    fft_precomputation: Option<Cow<'a, FFTPrecomputation<F>>>,
    ifft_precomputation: Option<Cow<'a, IFFTPrecomputation<F>>>,
}

impl<'a, F: PrimeField> PolyMultiplier<'a, F> {
    #[inline]
    pub fn new() -> Self {
        Self { polynomials: Vec::new(), evaluations: Vec::new(), fft_precomputation: None, ifft_precomputation: None }
    }

    #[inline]
    pub fn add_precomputation(&mut self, fft_pc: &'a FFTPrecomputation<F>, ifft_pc: &'a IFFTPrecomputation<F>) {
        self.fft_precomputation = Some(Cow::Borrowed(fft_pc));
        self.ifft_precomputation = Some(Cow::Borrowed(ifft_pc));
    }

    #[inline]
    pub fn add_polynomial(&mut self, poly: DensePolynomial<F>, label: impl ToString) {
        self.polynomials.push((label.to_string(), Cow::Owned(poly)))
    }

    #[inline]
    pub fn add_evaluation(&mut self, evals: Evaluations<F>, label: impl ToString) {
        self.evaluations.push((label.to_string(), Cow::Owned(evals)))
    }

    #[inline]
    pub fn add_polynomial_ref(&mut self, poly: &'a DensePolynomial<F>, label: impl ToString) {
        self.polynomials.push((label.to_string(), Cow::Borrowed(poly)))
    }

    #[inline]
    pub fn add_evaluation_ref(&mut self, evals: &'a Evaluations<F>, label: impl ToString) {
        self.evaluations.push((label.to_string(), Cow::Borrowed(evals)))
    }    

    pub fn multiply(self) -> Option<DensePolynomial<F>> {
        if self.polynomials.is_empty() && self.evaluations.is_empty() {
            Some(DensePolynomial::zero())
        } else {
            let degree = self.polynomials.iter().map(|(_, p)| p.degree() + 1).sum::<usize>();
            let domain = EvaluationDomain::new(degree)?;
            if self.evaluations.iter().any(|(_, e)| e.domain() != domain) {
                None
            } else {
                let fft_pc = &self.fft_precomputation.unwrap();
                let ifft_pc = &self.ifft_precomputation.unwrap();
                let mut pool = ExecutionPool::new();
                for (_, p) in self.polynomials {
                    pool.add_job(move || {
                        let mut p = p.to_owned().into_owned().coeffs;
                        p.resize(domain.size(), F::zero());
                        ars_radix_fft(&mut p, &domain.group_gen, domain.log_size_of_group, &fft_pc.roots, fft_pc.domain.size).unwrap();
                        p
                    })
                }
                for (_, e) in self.evaluations {
                    pool.add_job(move || {
                        let mut e = e.to_owned().into_owned().evaluations;
                        e.resize(domain.size(), F::zero());
                        e
                    })
                }
                let results = pool.execute_all();
                #[cfg(feature = "parallel")]
                let mut result = results
                    .into_par_iter()
                    .reduce_with(|mut a, b| {
                        cfg_iter_mut!(a).zip(b).for_each(|(a, b)| *a *= b);
                        a
                    })
                    .unwrap();
                #[cfg(not(feature = "parallel"))]
                let mut result = results
                    .into_iter()
                    .reduce(|mut a, b| {
                        cfg_iter_mut!(a).zip(b).for_each(|(a, b)| *a *= b);
                        a
                    })
                    .unwrap();                
                ars_radix_ifft(&mut result, &domain.group_gen_inv, domain.log_size_of_group, &domain.size_inv, &ifft_pc.inverse_roots, ifft_pc.domain.size).unwrap();
                Some(DensePolynomial::from_coefficients_vec(result))
            }
        }
    }

    pub fn element_wise_arithmetic_4_over_domain<T: Borrow<str>>(
        self,
        domain: EvaluationDomain<F>,
        labels: [T; 4],
        f: impl Fn(F, F, F, F) -> F + Sync,
    ) -> Option<DensePolynomial<F>> {
        let fft_pc = &self.fft_precomputation.unwrap();
        let ifft_pc = &self.ifft_precomputation.unwrap();
        let mut pool = ExecutionPool::new();
        for (l, p) in self.polynomials {
            pool.add_job(move || {
                let mut p = p.to_owned().into_owned().coeffs;
                p.resize(domain.size(), F::zero());                
                ars_radix_fft(&mut p, &domain.group_gen, domain.log_size_of_group, &fft_pc.roots, fft_pc.domain.size).unwrap();
                (l, p)
            })
        }
        for (l, e) in self.evaluations {
            pool.add_job(move || {
                let mut e = e.to_owned().into_owned().evaluations;
                e.resize(domain.size(), F::zero());
                (l, e)
            })
        }
        let p = pool.execute_all().into_iter().collect::<BTreeMap<_, _>>();
        assert_eq!(p.len(), 4);
        let mut result = cfg_iter!(p[labels[0].borrow()])
            .zip(&p[labels[1].borrow()])
            .zip(&p[labels[2].borrow()])
            .zip(&p[labels[3].borrow()])
            .map(|(((a, b), c), d)| f(*a, *b, *c, *d))
            .collect::<Vec<_>>();
        drop(p);        
        ars_radix_ifft(&mut result, &domain.group_gen_inv, domain.log_size_of_group, &domain.size_inv, &ifft_pc.inverse_roots, ifft_pc.domain.size).unwrap();
        Some(DensePolynomial::from_coefficients_vec(result))
    }    
}
