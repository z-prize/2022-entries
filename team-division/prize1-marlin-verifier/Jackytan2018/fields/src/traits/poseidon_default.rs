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

use std::ops::Range;
use crate::{PoseidonGrainLFSR, PrimeField};

use anyhow::{bail, Result};

/// Add by ars
pub type Matrix<T> = Vec<Vec<T>>;

pub fn rows<T>(matrix: &Matrix<T>) -> usize {
    matrix.len()
}

/// Parameters and RNG used
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoseidonParameters<F: PrimeField, const RATE: usize, const CAPACITY: usize> {
    /// number of rounds in a full-round operation
    pub full_rounds: usize,
    /// number of rounds in a partial-round operation
    pub partial_rounds: usize,
    /// Exponent used in S-boxes
    pub alpha: u64,
    /// Additive Round keys. These are added before each MDS matrix application to make it an affine shift.
    /// They are indexed by `ark[round_num][state_element_index]`
    pub ark: Vec<Vec<F>>,
    /// Maximally Distance Separating Matrix.
    pub mds: Vec<Vec<F>>,

    /// Add by ars
    /// optimized round constants matrix
    pub optimized_round_constants: Vec<Vec<F>>,
    /// Add by ars
    /// prepare sparse matrix
    pub pre_sparse_matrix: Vec<Vec<F>>,
    /// Add by ars
    /// optimized mds matrixes
    pub optimized_mds_matrixes: Vec<Vec<Vec<F>>>,
}

/// A field with Poseidon parameters associated
pub trait PoseidonDefaultField {
    /// Obtain the default Poseidon parameters for this rate and for this prime field,
    /// with a specific optimization goal.
    fn default_poseidon_parameters<const RATE: usize>() -> Result<PoseidonParameters<Self, RATE, 1>>
        where
            Self: PrimeField,
    {
        // Modified by ars
        // Since optimized_round_constants pre_sparse_matrix optimized_mds_matrixes matriceshave been added to the structure,
        // the new function is used
        return PoseidonDefaultField::ars_default_poseidon_parameters::<RATE>();

        // /// Internal function that computes the ark and mds from the Poseidon Grain LFSR.
        // #[allow(clippy::type_complexity)]
        // fn find_poseidon_ark_and_mds<F: PrimeField, const RATE: usize>(
        //     full_rounds: u64,
        //     partial_rounds: u64,
        //     skip_matrices: u64,
        // ) -> Result<(Vec<Vec<F>>, Vec<Vec<F>>)> {
        //     // println!("f 类型={}  full_rounds = {:#?}; partial_rounds = {:#?}; skip_matrices = {:#?};RATE = {:?}", any::type_name::<F>(), full_rounds, partial_rounds, skip_matrices, RATE);
        //     let mut lfsr =
        //         PoseidonGrainLFSR::new(false, F::size_in_bits() as u64, (RATE + 1) as u64, full_rounds, partial_rounds);
        //
        //     let mut ark = Vec::<Vec<F>>::new();
        //     for _ in 0..(full_rounds + partial_rounds) {
        //         ark.push(lfsr.get_field_elements_rejection_sampling(RATE + 1)?);
        //     }
        //
        //     let mut mds = vec![vec![F::zero(); RATE + 1]; RATE + 1];
        //     for _ in 0..skip_matrices {
        //         let _ = lfsr.get_field_elements_mod_p::<F>(2 * (RATE + 1))?;
        //     }
        //
        //     // A qualifying matrix must satisfy the following requirements:
        //     // - There is no duplication among the elements in x or y.
        //     // - There is no i and j such that x[i] + y[j] = p.
        //     // - There resultant MDS passes all three tests.
        //
        //     let xs = lfsr.get_field_elements_mod_p::<F>(RATE + 1)?;
        //     let ys = lfsr.get_field_elements_mod_p::<F>(RATE + 1)?;
        //
        //     for (i, x) in xs.iter().enumerate().take(RATE + 1) {
        //         for (j, y) in ys.iter().enumerate().take(RATE + 1) {
        //             mds[i][j] = (*x + y).inverse().unwrap();
        //         }
        //     }
        //
        //     Ok((ark, mds))
        // }
        //
        // match Self::Parameters::PARAMS_OPT_FOR_CONSTRAINTS.iter().find(|entry| entry.rate == RATE) {
        //     Some(entry) => {
        //         let (ark, mds) = find_poseidon_ark_and_mds::<Self, RATE>(
        //             entry.full_rounds as u64,
        //             entry.partial_rounds as u64,
        //             entry.skip_matrices as u64,
        //         )?;
        //
        //         Ok(PoseidonParameters {
        //             full_rounds: entry.full_rounds,
        //             partial_rounds: entry.partial_rounds,
        //             alpha: entry.alpha as u64,
        //             ark,
        //             mds,
        //             optimized_round_constants: vec![],
        //             pre_sparse_matrix: vec![],
        //             optimized_mds_matrixes: vec![],
        //         })
        //     }
        //     None => bail!("No Poseidon parameters were found for this rate"),
        // }
    }


    /// Add by ars
    /// Generate poseidon parameters
    fn ars_default_poseidon_parameters<const RATE: usize>() -> Result<PoseidonParameters<Self, RATE, 1>>
        where
            Self: PrimeField,
    {
        /// Add by ars
        /// Generate round concatenants and MDS matrices
        #[allow(clippy::type_complexity)]
        fn find_poseidon_ark_and_mds<F: PrimeField, const RATE: usize>(
            full_rounds: u64,
            partial_rounds: u64,
            skip_matrices: u64,
        ) -> Result<(Vec<Vec<F>>, Vec<Vec<F>>)> {
            let mut lfsr = PoseidonGrainLFSR::new(false, F::size_in_bits() as u64, (RATE + 1) as u64, full_rounds, partial_rounds);

            let mut ark = Vec::<Vec<F>>::new();
            for _ in 0..(full_rounds + partial_rounds) {
                ark.push(lfsr.get_field_elements_rejection_sampling(RATE + 1)?);
            }

            let mut mds = vec![vec![F::zero(); RATE + 1]; RATE + 1];
            for _ in 0..skip_matrices {
                let _ = lfsr.get_field_elements_mod_p::<F>(2 * (RATE + 1))?;
            }

            // A qualifying matrix must satisfy the following requirements:
            // - There is no duplication among the elements in x or y.
            // - There is no i and j such that x[i] + y[j] = p.
            // - There resultant MDS passes all three tests.

            let xs = lfsr.get_field_elements_mod_p::<F>(RATE + 1)?;
            let ys = lfsr.get_field_elements_mod_p::<F>(RATE + 1)?;

            for (i, x) in xs.iter().enumerate().take(RATE + 1) {
                for (j, y) in ys.iter().enumerate().take(RATE + 1) {
                    mds[i][j] = (*x + y).inverse().unwrap();
                }
            }

            Ok((ark, mds))
        }

        /// Add by ars
        /// Multiplying Matrix and Vector
        fn mmul_assign<F: PrimeField>(matrix: &Matrix<F>, vector: &mut Vec<F>) {
            let mut new_vector = vector.clone();
            new_vector.iter_mut().zip(matrix).for_each(|(new_elem, matrix_row)| {
                *new_elem = vector.iter().zip(matrix_row).map(|(vector_elem, matrix_elem)| *vector_elem * *matrix_elem).sum::<F>();
            });
            *vector = new_vector;
        }

        /// Add by ars
        /// Take submatrix from large matrix
        fn sub_matrix<F: PrimeField>(
            matrix: &Matrix<F>,
            row_range: std::ops::Range<usize>,
            col_range: std::ops::Range<usize>,
        ) -> Matrix<F> {
            // ) -> [[F; SUBDIM]; SUBDIM] {
            // we need following decompositions for optimized matrixes
            //          row     col
            // M' => 1..DIM   1..DIM
            // w  => 1..DIM   0..1
            // v  => 0..1     1..DIM
            let size = matrix.len() - 1;
            assert!(
                (row_range.len() == size || row_range.len() == 1)
                    && (col_range.len() == size || col_range.len() == 1),
                "row/col length should be in range"
            );
            // let mut sub_matrix = [[F::zero(); size]; size];
            let mut sub_matrix = vec![vec![F::zero(); size]; size];

            for (row_id, row) in matrix[row_range].iter().enumerate() {
                for (col_id, col) in row[col_range.clone()].iter().enumerate() {
                    sub_matrix[row_id][col_id] = *col;
                }
            }

            sub_matrix
        }


        /// Add by ars
        /// Set Submatrix
        fn set_sub_matrix<F: PrimeField>(
            // matrix: &mut [[F; DIM]; DIM],
            matrix: &mut Matrix<F>,
            row_range: Range<usize>,
            col_range: Range<usize>,
            sub_matrix: &Matrix<F>,
        ) {
            for (row_a, row_b) in matrix[row_range].iter_mut().zip(sub_matrix.iter()) {
                for (col_a, col_b) in row_a[col_range.clone()].iter_mut().zip(row_b.iter()) {
                    *col_a = col_b.clone();
                }
            }
        }


        /// Add by ars
        /// Matrix transpose
        fn transpose<F: PrimeField>(
            matrix: &Matrix<F>,
        ) -> Matrix<F> {
            let size = rows(matrix);
            let mut new = Vec::with_capacity(size);
            for j in 0..size {
                let mut row = Vec::with_capacity(size);
                for i in 0..size {
                    row.push(matrix[i][j])
                }
                new.push(row);
            }
            new
        }


        /// Add by ars
        /// Identity matrix
        fn identity<F: PrimeField>(size: usize) -> Matrix<F> {
            let mut identity = vec![vec![F::zero(); size]; size];
            for i in 0..size {
                for j in 0..size {
                    let el = if i == j { F::one() } else { F::zero() };
                    identity[i][j] = el;
                }
            }
            identity
        }


        /// Add by ars
        /// Vector inner product
        fn scalar_product<F: PrimeField>(a: &Vec<F>, b: &Vec<F>) -> F {
            let mut acc = F::zero();
            for (a, b) in a.iter().zip(b.iter()) {
                let mut tmp = a.clone();
                tmp.mul_assign(b);
                acc.add_assign(tmp);
            }
            acc
        }

        /// Add by ars
        /// Matrix multiplication
        fn multiply<F: PrimeField>(
            m1: &Matrix<F>,
            m2: &Matrix<F>,
        ) -> Vec<Vec<F>> {
            let size = rows(m1);
            let transposed_m2 = transpose::<F>(m2);
            let mut result = vec![vec![F::zero(); size]; size];
            for (i, rv) in m1.iter().enumerate() {
                for (j, cv) in transposed_m2.iter().enumerate() {
                    result[i][j] = scalar_product::<F>(rv, cv);
                }
            }
            result
        }


        /// Add by ars
        /// Constant multiplication vector
        fn scalar_vec_mul<F: PrimeField>(scalar: F, vec: &[F]) -> Vec<F> {
            vec.iter().map(|val| {
                let mut prod = scalar;
                prod.mul_assign(val);
                prod
            }).collect::<Vec<_>>()
        }

        /// Add by ars
        /// Vector subtraction
        fn vec_sub<F: PrimeField>(a: &[F], b: &[F]) -> Vec<F> {
            a.iter().zip(b).map(|(a, b)| {
                let mut res = *a;
                res.sub_assign(b);
                res
            }).collect::<Vec<_>>()
        }

        /// Add by ars
        /// Gaussian elimination
        fn eliminate<F: PrimeField>(
            matrix: &Matrix<F>,
            column: usize,
            shadow: &mut Matrix<F>,
        ) -> Option<Matrix<F>> {
            let zero = F::zero();
            let pivot_index = (0..rows(matrix)).find(|&i| matrix[i][column] != zero && (0..column).all(|j| matrix[i][j] == zero))?;

            let pivot = &matrix[pivot_index];
            let pivot_val = pivot[column];

            // This should never fail since we have a non-zero `pivot_val` if we got here.
            let inv_pivot = Option::from(pivot_val.inverse())?;
            let mut result = Vec::with_capacity(matrix.len());
            result.push(pivot.clone());

            for (i, row) in matrix.iter().enumerate() {
                if i == pivot_index {
                    continue;
                };
                let val = row[column];
                if val == zero {
                    // Value is already eliminated.
                    result.push(row.to_vec());
                } else {
                    let mut factor = val;
                    factor.mul_assign(&inv_pivot);

                    let scaled_pivot = scalar_vec_mul(factor, pivot);
                    let eliminated = vec_sub(row, &scaled_pivot);
                    result.push(eliminated);

                    let shadow_pivot = &shadow[pivot_index];
                    let scaled_shadow_pivot = scalar_vec_mul(factor, shadow_pivot);
                    let shadow_row = &shadow[i];
                    shadow[i] = vec_sub(shadow_row, &scaled_shadow_pivot);
                }
            }

            let pivot_row = shadow.remove(pivot_index);
            shadow.insert(0, pivot_row);

            Some(result)
        }

        /// Add by ars
        /// `matrix` must be square.
        /// Generate upper triangular matrix
        fn upper_triangular<F: PrimeField>(
            matrix: &Matrix<F>,
            shadow: &mut Matrix<F>,
        ) -> Option<Matrix<F>> {
            //assert!(is_square(matrix));
            let mut result = Vec::with_capacity(matrix.len());
            let mut shadow_result = Vec::with_capacity(matrix.len());
            let mut curr = matrix.clone();
            let mut column = 0;

            while curr.len() > 1 {
                let initial_rows = curr.len();
                curr = eliminate(&curr, column, shadow)?;
                result.push(curr[0].clone());
                shadow_result.push(shadow[0].clone());
                column += 1;
                curr = curr[1..].to_vec();
                *shadow = shadow[1..].to_vec();
                assert_eq!(curr.len(), initial_rows - 1);
            }

            result.push(curr[0].clone());
            shadow_result.push(shadow[0].clone());
            *shadow = shadow_result;
            Some(result)
        }

        /// Add by ars
        /// Reduction to identity matrix
        fn reduce_to_identity<F: PrimeField>(
            matrix: &Matrix<F>,
            shadow: &mut Matrix<F>,
        ) -> Option<Matrix<F>> {
            let size = rows(matrix);
            let mut result: Matrix<F> = Vec::new();
            let mut shadow_result: Matrix<F> = Vec::new();

            for i in 0..size {
                let idx = size - i - 1;
                let row = &matrix[idx];
                let shadow_row = &shadow[idx];
                let val = row[idx];
                let inv = {
                    let inv = val.inverse();
                    // If `val` is zero, then there is no inverse, and we cannot compute a result.
                    if inv.is_none().into() {
                        return None;
                    }
                    inv.unwrap()
                };
                let mut normalized = scalar_vec_mul(inv, row);
                let mut shadow_normalized = scalar_vec_mul(inv, shadow_row);

                for j in 0..i {
                    let idx = size - j - 1;
                    let val = normalized[idx];
                    let subtracted = scalar_vec_mul(val, &result[j]);
                    let result_subtracted = scalar_vec_mul(val, &shadow_result[j]);

                    normalized = vec_sub(&normalized, &subtracted);
                    shadow_normalized = vec_sub(&shadow_normalized, &result_subtracted);
                }
                result.push(normalized);
                shadow_result.push(shadow_normalized);
            }

            result.reverse();
            shadow_result.reverse();
            *shadow = shadow_result;
            Some(result)
        }

        /// Add by ars
        /// Matrix inversion
        fn invert<F: PrimeField>(matrix: &Vec<Vec<F>>) -> Option<Vec<Vec<F>>> {
            let len: usize = matrix.len();
            let mut shadow = identity::<F>(len);
            let ut = upper_triangular(matrix, &mut shadow);
            ut.and_then(|x| {
                reduce_to_identity(&x, &mut shadow)
            }).and(Some(shadow))
        }


        /// Add by ars
        /// Calculate the optimization round constant of Poseidon hash
        fn compute_optimized_round_constants<F: PrimeField>(
            constants: &Matrix<F>,
            original_mds: &Matrix<F>,
            number_of_partial_rounds: usize,
            number_of_full_rounds: usize,
            rate: usize,
        ) -> Matrix<F> {
            assert_eq!(
                constants.len(),
                number_of_full_rounds + number_of_partial_rounds,
                "non-optimized constants length does not match with total number of rounds"
            );
	    // Matrix inversion
            let mds_inverse = invert(original_mds).expect("has inverse");
            let number_of_half_rounds = number_of_full_rounds / 2;
            let start = number_of_half_rounds;
            let end = start + number_of_partial_rounds - 1;
            let mut acc = constants[end].to_vec();
            let mut optimized_constants: Matrix<F> = vec![];
            for round in (start..end).rev() {
                let mut inv = acc;
		// Matrix multiplication
                mmul_assign(&mds_inverse, &mut inv);
                // make it two parts
                let mut second = vec![F::zero(); rate + 1];
                second[0] = inv[0];
                optimized_constants.push(second);

                let mut first = inv;
                first[0] = F::zero();

                // vector addition
                acc = vec![F::zero(); rate + 1];
                constants[round].iter().enumerate().zip(first.iter()).for_each(|((idx, a), b)| {
                    let mut tmp = a.clone();
                    tmp.add_assign(b);
                    acc[idx] = tmp;
                });
            }
            optimized_constants.push(acc);
            optimized_constants.reverse();

            let mut final_constants = constants.to_vec();
            final_constants[start..end + 1].iter_mut().zip(optimized_constants).for_each(|(a, b)| {
                *a = b;
            });

            final_constants
        }


        /// Add by ars
        /// Calculate the optimization matrix of Poseidon Hash
        fn compute_optimized_matrixes<F: PrimeField>(
            number_of_rounds: usize,
            original_mds: &Matrix<F>,
            rate: usize,
        ) -> (Matrix<F>, Vec<Matrix<F>>) {
            let original_mds = transpose::<F>(original_mds);
            let mut matrix = original_mds.clone();
            let mut m_prime = identity::<F>(rate + 1);
            let mut sparse_matrixes = vec![vec![vec![F::zero(); rate + 1]; rate + 1]; number_of_rounds];

            for round in 0..number_of_rounds {
                // M'
                let mut m_hat = sub_matrix::<F>(&matrix, 1..(rate + 1), 1..(rate + 1));
                m_prime = identity::<F>(rate + 1);
                set_sub_matrix::<F>(&mut m_prime, 1..(rate + 1), 1..(rate + 1), &mut m_hat);

                // M"
                let w = sub_matrix::<F>(&matrix, 1..(rate + 1), 0..1);
                let mut v = sub_matrix::<F>(&matrix, 0..1, 1..(rate + 1));
                let m_hat_inv = invert(&m_hat).expect("inverse");
                let mut w_hat = multiply::<F>(&m_hat_inv, &w);

                let mut sparse_matrix = identity::<F>(rate + 1);
                sparse_matrix[0][0] = matrix[0][0];
                set_sub_matrix::<F>(&mut sparse_matrix, 0..1, 1..(rate + 1), &mut v);
                set_sub_matrix::<F>(&mut sparse_matrix, 1..(rate + 1), 0..1, &mut w_hat);
                {
                    // sanity check
                    let actual = multiply::<F>(&m_prime, &sparse_matrix);
                    assert_eq!(matrix, actual);
                }
                sparse_matrixes[round] = transpose::<F>(&sparse_matrix);
                matrix = multiply::<F>(&original_mds, &m_prime);
            }

            sparse_matrixes.reverse();
            return (transpose::<F>(&m_prime), sparse_matrixes);
        }

        match Self::Parameters::PARAMS_OPT_FOR_CONSTRAINTS.iter().find(|entry| entry.rate == RATE) {
            Some(entry) => {
                let (ark, mds) = find_poseidon_ark_and_mds::<Self, RATE>(
                    entry.full_rounds as u64,
                    entry.partial_rounds as u64,
                    entry.skip_matrices as u64,
                )?;

                let num_of_rounds = entry.partial_rounds;
                let optimized_round_constants = compute_optimized_round_constants::<Self>(
                    &ark,
                    &mds,
                    entry.partial_rounds,
                    entry.full_rounds,
                    RATE,
                );
                let (pre_sparse_matrix, optimized_mds_matrixes) = compute_optimized_matrixes::<Self>(num_of_rounds, &mds, RATE);

                Ok(PoseidonParameters {
                    full_rounds: entry.full_rounds,
                    partial_rounds: entry.partial_rounds,
                    alpha: entry.alpha as u64,
                    ark,
                    mds,
                    optimized_round_constants,
                    pre_sparse_matrix,
                    optimized_mds_matrixes,
                })
            }
            None => bail!("No Poseidon parameters were found for this rate"),
        }
    }
}

/// A trait for default Poseidon parameters associated with a prime field
pub trait PoseidonDefaultParameters {
    /// An array of the parameters optimized for constraints
    /// (rate, alpha, full_rounds, partial_rounds, skip_matrices)
    /// for rate = 2, 3, 4, 5, 6, 7, 8
    ///
    /// Here, `skip_matrices` denote how many matrices to skip before
    /// finding one that satisfy all the requirements.
    const PARAMS_OPT_FOR_CONSTRAINTS: [PoseidonDefaultParametersEntry; 7];
}

/// An entry in the default Poseidon parameters
pub struct PoseidonDefaultParametersEntry {
    /// The rate (in terms of number of field elements).
    pub rate: usize,
    /// Exponent used in S-boxes.
    pub alpha: usize,
    /// Number of rounds in a full-round operation.
    pub full_rounds: usize,
    /// Number of rounds in a partial-round operation.
    pub partial_rounds: usize,
    /// Number of matrices to skip when generating parameters using the Grain LFSR.
    ///
    /// The matrices being skipped are those that do not satisfy all the desired properties.
    /// See the [reference implementation](https://extgit.iaik.tugraz.at/krypto/hadeshash/-/blob/master/code/generate_parameters_grain.sage) for more detail.
    pub skip_matrices: usize,
}

impl PoseidonDefaultParametersEntry {
    /// Create an entry in PoseidonDefaultParameters.
    pub const fn new(
        rate: usize,
        alpha: usize,
        full_rounds: usize,
        partial_rounds: usize,
        skip_matrices: usize,
    ) -> Self {
        Self { rate, alpha, full_rounds, partial_rounds, skip_matrices }
    }
}
