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

use std::sync::Arc;

use crate::{
    fft::{
        domain::{FFTPrecomputation, IFFTPrecomputation},
        DensePolynomial, EvaluationDomain, Evaluations as EvaluationsOnDomain,
    },
    snark::marlin::{
        ahp::{indexer::Circuit, verifier},
        AHPError, MarlinMode,
    },
};
use snarkvm_fields::PrimeField;
use snarkvm_r1cs::SynthesisError;

use crate::fft::Evaluations;
use crate::polycommit::sonic_pc::LabeledPolynomial;

/// State for the AHP prover.
#[derive(Clone)]
pub struct State<'a, F: PrimeField, MM: MarlinMode> {
    pub(super) index: &'a Circuit<F, MM>,

    /*
       reindex_ABC_matrix tries to record the coeff matrix by the column index.
       A/B/C index, column index, A/B/C coeff
    */
    pub(super) reindex_ABC_matrix: Vec<Vec<(usize, usize, usize, F)>>,
    pub(super) coeff_one_vec: Vec<(usize, usize, usize, F)>,
    pub(super) col_zero_vec: Vec<(usize, usize, usize, F)>,

    /// A domain that is sized for the public input.
    pub(super) input_domain: EvaluationDomain<F>,

    /// A domain that is sized for the number of constraints.
    pub(crate) constraint_domain: EvaluationDomain<F>,

    /// A domain that is sized for the number of non-zero elements in A.
    pub(in crate::snark) non_zero_a_domain: EvaluationDomain<F>,
    /// A domain that is sized for the number of non-zero elements in B.
    pub(in crate::snark) non_zero_b_domain: EvaluationDomain<F>,
    /// A domain that is sized for the number of non-zero elements in C.
    pub(in crate::snark) non_zero_c_domain: EvaluationDomain<F>,

    /// The number of instances being proved in this batch.
    pub(in crate::snark) batch_size: usize,

    /// The list of public inputs for each instance in the batch.
    /// The length of this list must be equal to the batch size.
    pub(super) padded_public_variables: Vec<Vec<F>>,

    /// The list of private variables for each instance in the batch.
    /// The length of this list must be equal to the batch size.
    pub(super) private_variables: Vec<Vec<F>>,

    /// The list of Az vectors for each instance in the batch.
    /// The length of this list must be equal to the batch size.
    pub(super) z_a: Option<Vec<Vec<F>>>,

    /// The list of Bz vectors for each instance in the batch.
    /// The length of this list must be equal to the batch size.
    pub(super) z_b: Option<Vec<Vec<F>>>,

    /// A list of polynomials corresponding to the interpolation of the public input.
    /// The length of this list must be equal to the batch size.
    pub(super) x_poly: Vec<DensePolynomial<F>>,

    /// The first round oracles sent by the prover.
    /// The length of this list must be equal to the batch size.
    pub(in crate::snark) first_round_oracles: Option<Arc<super::FirstOracles<'a, F>>>,

    /// Randomizers for z_b.
    /// The length of this list must be equal to the batch size.
    pub(super) mz_poly_randomizer: Option<Vec<F>>,

    /// The challenges sent by the verifier in the first round
    pub(super) verifier_first_message: Option<verifier::FirstMessage<F>>,

    /// Polynomials involved in the holographic sumcheck.
    pub(super) lhs_polynomials: Option<[DensePolynomial<F>; 3]>,
    /// Polynomials involved in the holographic sumcheck.
    pub(super) sums: Option<[F; 3]>,
}

#[derive(Clone)]
pub struct MyFirstOracles<F: PrimeField> {
    pub(super) w_poly: LabeledPolynomial<F>,
    pub(super) z_a_poly: DensePolynomial<F>,
    pub(super) z_a_evals: Evaluations<F>,
    pub(super) z_b_poly: DensePolynomial<F>,
    pub(super) z_b_evals: Evaluations<F>,
}
impl<'a, F: PrimeField, MM: MarlinMode> State<'a, F, MM> {
    // pub fn initialize(
    //     padded_public_input: Vec<Vec<F>>,
    //     private_variables: Vec<Vec<F>>,
    //     index: &'a Circuit<F, MM>,
    // ) -> Result<Self, AHPError> {
    //     let index_info = &index.index_info;
    //     let constraint_domain =
    //         EvaluationDomain::new(index_info.num_constraints).ok_or(SynthesisError::PolynomialDegreeTooLarge)?;

    //     let non_zero_a_domain =
    //         EvaluationDomain::new(index_info.num_non_zero_a).ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
    //     let non_zero_b_domain =
    //         EvaluationDomain::new(index_info.num_non_zero_b).ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
    //     let non_zero_c_domain =
    //         EvaluationDomain::new(index_info.num_non_zero_c).ok_or(SynthesisError::PolynomialDegreeTooLarge)?;

    //     let input_domain =
    //         EvaluationDomain::new(padded_public_input[0].len()).ok_or(SynthesisError::PolynomialDegreeTooLarge)?;

    //     let x_poly = padded_public_input
    //         .iter()
    //         .map(|padded_public_input| {
    //             EvaluationsOnDomain::from_vec_and_domain(padded_public_input.clone(), input_domain).interpolate()
    //         })
    //         .collect();
    //     let batch_size = private_variables.len();
    //     assert_eq!(padded_public_input.len(), batch_size);

    //     Ok(Self {
    //         index,
    //         input_domain,
    //         constraint_domain,
    //         non_zero_a_domain,
    //         non_zero_b_domain,
    //         non_zero_c_domain,
    //         batch_size,
    //         padded_public_variables: padded_public_input,
    //         x_poly,
    //         private_variables,
    //         z_a: None,
    //         z_b: None,
    //         first_round_oracles: None,
    //         mz_poly_randomizer: None,
    //         verifier_first_message: None,
    //         lhs_polynomials: None,
    //         sums: None,
    //     })
    // }

    pub fn initialize(
        padded_public_input: Vec<Vec<F>>,
        private_variables: Vec<Vec<F>>,
        index: &'a Circuit<F, MM>,
    ) -> Result<Self, AHPError> {
        let index_info = &index.index_info;
        let constraint_domain =
            EvaluationDomain::new(index_info.num_constraints).ok_or(SynthesisError::PolynomialDegreeTooLarge)?;

        let non_zero_a_domain =
            EvaluationDomain::new(index_info.num_non_zero_a).ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
        let non_zero_b_domain =
            EvaluationDomain::new(index_info.num_non_zero_b).ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
        let non_zero_c_domain =
            EvaluationDomain::new(index_info.num_non_zero_c).ok_or(SynthesisError::PolynomialDegreeTooLarge)?;

        let input_domain =
            EvaluationDomain::new(padded_public_input[0].len()).ok_or(SynthesisError::PolynomialDegreeTooLarge)?;

        let x_poly = padded_public_input
            .iter()
            .map(|padded_public_input| {
                EvaluationsOnDomain::from_vec_and_domain(padded_public_input.clone(), input_domain).interpolate()
            })
            .collect();
        let batch_size = private_variables.len();
        assert_eq!(padded_public_input.len(), batch_size);

        let mut reindex_ABC_matrix_pre = Vec::new();
        let mut reindex_ABC_matrix = Vec::new();
        let mut coeff_one_vec = Vec::new();
        let mut col_zero_vec = Vec::new();

        let mut last_row_info = Vec::<(usize, usize, usize, F)>::new();
        let mut row_offsets = [0; 32768];
        for i in 0..constraint_domain.size as usize {
            let mut row = Vec::new();

            if i < index.a.len() {
                let row_a = &index.a.as_slice()[i];
                for (coeff, c) in row_a.iter() {
                    let index = constraint_domain.reindex_by_subdomain(input_domain, *c);
                    if *coeff != F::zero() {
                        row.push((0, index, *coeff));
                    }
                }
            }

            if i < index.b.len() {
                let row_b = &index.b.as_slice()[i];
                for (coeff, c) in row_b.iter() {
                    let index = constraint_domain.reindex_by_subdomain(input_domain, *c);
                    if *coeff != F::zero() {
                        row.push((1, index, *coeff));
                    }
                }
            }

            if i < index.c.len() {
                let row_c = &index.c.as_slice()[i];
                for (coeff, c) in row_c.iter() {
                    let index = constraint_domain.reindex_by_subdomain(input_domain, *c);
                    if *coeff != F::zero() {
                        row.push((2, index, *coeff));
                    }
                }
            }

            //sort by col
            row.sort_by(|a, b| a.1.cmp(&b.1));

            /* the index is extended to cover more cases
             *
             * bit0-bit7  - index
             * bit8       - minus (not used)
             * bit16-bit23 - all index in one row (not more than 16)
             * bit24-bit31 - offset
             *
             * if minus, try to sub, not add.
             *
             * */

            /* de-duplicate
             *
             * 1/ If co-eff and col is same, the entry can be merged.
             *
             * 0 - eta_a
             * 1 - eta_b
             * 2 - eta_c
             * 3 - eta_a + eta_b
             * 4 - eta_a + eta_c
             * 5 - eta_b - eta_a
             * */

            let mut de_row = Vec::new();
            let mut last = None;
            for (index, c, coeff) in row.iter() {
                if last == None {
                    last = Some((index, c, coeff));
                    continue;
                }

                let last_val = last.unwrap();
                if *c == *last_val.1 && *coeff == *last_val.2 {
                    let mut new_index = index;
                    match (*index, *last_val.0) {
                        (0, 1) | (1, 0) => {
                            new_index = &3;
                        }
                        (0, 2) | (2, 0) => {
                            new_index = &4;
                        }
                        (_, _) => {}
                    };

                    //println!("found - {:?}, {:?}, -> {:?}", *index, *last_val.0, *new_index);
                    de_row.push((*new_index, *c, *coeff));
                    last = None;
                    continue;
                } else if *c == *last_val.1 && *coeff == F::one() && *last_val.2 == (F::zero() - F::one()) {
                    let mut new_index = index;
                    match (*index, *last_val.0) {
                        (1, 0) => {
                            new_index = &5;
                        }
                        (_, _) => {}
                    };

                    //println!("found - {:?}, {:?}, -> {:?}", *index, *last_val.0, *new_index);

                    de_row.push((*new_index, *c, *coeff));
                    last = None;
                    continue;
                } else {
                    de_row.push((*last_val.0, *last_val.1, *last_val.2));
                    last = Some((index, c, coeff));
                }
            }

            if last != None {
                let last_val = last.unwrap();
                de_row.push((*last_val.0, *last_val.1, *last_val.2));
            }

            let mut row_wo_coeff_one = Vec::new();
            /*
             * Filtering - two facts are found: 1/ many coeff is 1 2/ many on col 0.
             *
             * Currently there are two classifications:
             * 1/ coeff is 1
             * 2/ one case 1, col is 0.
             *
             */
            for (index, c, coeff) in de_row.iter() {
                let index = *index;
                let c = *c;
                let coeff = *coeff;
                if coeff == F::zero() - F::one() {
                    //coeff = F::one();
                    //index += 1<<8;
                }

                if coeff == F::one() {
                    if c == 0 {
                        col_zero_vec.push((index, i, c, coeff));
                    } else {
                        coeff_one_vec.push((index, i, c, coeff));
                    }
                } else {
                    row_wo_coeff_one.push((index, c, coeff));
                }
            }

            let mut final_with_indexinfo = Vec::new();
            let mut index_summary = 0;
            for (index, _, _) in row_wo_coeff_one.iter() {
                let index_tmp = 1 << (16 + (*index & 0xff));
                if index_tmp & index_summary == 0 {
                    index_summary += index_tmp;
                }
            }
            for (index, c, coeff) in row_wo_coeff_one.iter() {
                final_with_indexinfo.push((*index + index_summary, i, *c, *coeff));
            }

            if final_with_indexinfo.len() > 0 {
                //double-check whether the current row is the offset of the last row
                let mut is_row_offseted = true;
                let mut row_offset = 0;
                let mut last_row = 0;
                if last_row_info.len() == final_with_indexinfo.len() {
                    let last_row0 = last_row_info[0];
                    let curr_row0 = final_with_indexinfo[0];
                    last_row = last_row0.1;
                    row_offset = curr_row0.1 - last_row0.1;

                    for i in 0..last_row_info.len() {
                        let lrow = last_row_info[i];
                        let crow = final_with_indexinfo[i];

                        if lrow.0 != crow.0 || (crow.1 - lrow.1) != row_offset || lrow.2 != crow.2 || lrow.3 != crow.3 {
                            is_row_offseted = false;
                            break;
                        }
                    }
                } else {
                    is_row_offseted = false;
                }

                if is_row_offseted {
                    row_offsets[last_row] = row_offset;
                } else {
                    reindex_ABC_matrix_pre.push(final_with_indexinfo.clone());
                }
                last_row_info = final_with_indexinfo.clone(); //udpate last row
            }
        }

        /*
        let mut offset_count = 0;
        for i in 0..row_offsets.len() {
            if row_offsets[i] != 0 {
                offset_count += 1;
                println!(" row: {:?}, offset: {:?}", i, row_offsets[i]);
            }
        }
        println!("rows offset: {:?}", offset_count);
        */

        for (_, rows) in reindex_ABC_matrix_pre.iter().enumerate() {
            if rows.len() == 0 {
                continue;
            }
            let mut new_rows = Vec::new();
            for (index, r, c, coeff) in rows.iter() {
                if row_offsets[*r] >= 0 {
                    new_rows.push(((*index) + (row_offsets[*r] << 24), *r, *c, *coeff));
                }
            }

            if new_rows.len() > 0 {
                reindex_ABC_matrix.push(new_rows);
            }
        }

        //sort by col
        coeff_one_vec.sort_by(|a, b| a.2.cmp(&b.2));
        col_zero_vec.sort_by(|a, b| a.1.cmp(&b.1));

        /*
        println!("====> ABC ");
        let mut total = 0;
        for (_, rows) in reindex_ABC_matrix.iter().enumerate() {
                if rows.len() == 0 {
                    continue;
                }
                let (index, r, c, coeff) = rows[0];
                println!("----> row: {:?}, len: {:?}", r, rows.len());
                for (index, r, c, coeff) in rows.iter() {
                    println!("index: {:?}-{:?}-{:?}, minus: {:?}, row: {:?}, col: {:?}, coeff: {:?}", *index>>24, (*index>>16)&0xff, *index&0xff, *index&0x100, r, *c, *coeff);
                    total += 1;
                }
        }
        println!("====> ABC {:?}", total);
        println!("====> coeff 1: {:?}", coeff_one_vec.len());
        for i in 0..coeff_one_vec.len() {
            println!("index: {:?}-{:?}, minus: {:?}, row: {:?}, col: {:?}, coeff: {:?}", coeff_one_vec[i].0>>16, coeff_one_vec[i].0&0xff, coeff_one_vec[i].0&0x100, coeff_one_vec[i].1, coeff_one_vec[i].2, coeff_one_vec[i].3);
        }
        println!("====> col zero: {:?}", col_zero_vec.len());
        for i in 0..col_zero_vec.len() {
            println!("index: {:?}-{:?}, minus: {:?}, row: {:?}, col: {:?}, coeff: {:?}", col_zero_vec[i].0>>16, col_zero_vec[i].0&0xff, col_zero_vec[i].0&0x100, col_zero_vec[i].1, col_zero_vec[i].2, col_zero_vec[i].3);
        }
        */

        Ok(Self {
            index,
            reindex_ABC_matrix,
            coeff_one_vec,
            col_zero_vec,
            input_domain,
            constraint_domain,
            non_zero_a_domain,
            non_zero_b_domain,
            non_zero_c_domain,
            batch_size,
            padded_public_variables: padded_public_input,
            x_poly,
            private_variables,
            z_a: None,
            z_b: None,
            first_round_oracles: None,
            mz_poly_randomizer: None,
            verifier_first_message: None,
            lhs_polynomials: None,
            sums: None,
        })
    }

    /// Get the batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get the public inputs for the entire batch.
    pub fn public_inputs(&self) -> Vec<Vec<F>> {
        self.padded_public_variables.iter().map(|v| super::ConstraintSystem::unformat_public_input(v)).collect()
    }

    /// Get the padded public inputs for the entire batch.
    pub fn padded_public_inputs(&self) -> Vec<Vec<F>> {
        self.padded_public_variables.clone()
    }

    pub fn fft_precomputation(&self) -> &FFTPrecomputation<F> {
        &self.index.fft_precomputation
    }

    pub fn ifft_precomputation(&self) -> &IFFTPrecomputation<F> {
        &self.index.ifft_precomputation
    }
}
