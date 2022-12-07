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
    impl_add_sub_from_field_ref,
    impl_mul_div_from_field_ref,
    FftField,
    Field,
    FieldError,
    FieldParameters,
    LegendreSymbol,
    One,
    PoseidonDefaultField,
    PoseidonDefaultParameters,
    PrimeField,
    SquareRootField,
    Zero,
};
use snarkvm_utilities::{
    biginteger::{arithmetic as fa, BigInteger as _BigInteger, BigInteger256 as BigInteger},
    serialize::CanonicalDeserialize,
    FromBytes,
    ToBits,
    ToBytes,
};

use std::{
    cmp::{Ord, Ordering, PartialOrd},
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    io::{Read, Result as IoResult, Write},
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    str::FromStr,
};

pub type LimbT = ::std::os::raw::c_ulonglong;

#[link(name = "blst", kind = "static")]
extern "C" {
    pub fn add_mod_256(
        ret: *mut LimbT,
        a: *const LimbT,
        b: *const LimbT,
        p: *const LimbT,
    );

    pub fn sub_mod_256(
        ret: *mut LimbT,
        a: *const LimbT,
        b: *const LimbT,
        p: *const LimbT,
    );

    pub fn mulx_mont_sparse_256(
        ret: *mut LimbT,
        a: *const LimbT,
        b: *const LimbT,
        p: *const LimbT,
        n0: u64,
    );

    pub fn sqrx_mont_sparse_256(
        ret: *mut LimbT,
        a: *const LimbT,
        p: *const LimbT,
        n0: u64,
    );

    pub fn blst_scalar_from_fr(
        ret: *mut LimbT,
        a: *const LimbT,
    );

    //pub fn blst_fr_eucl_inverse(
    //    ret: *mut LimbT,
    //    a: *const LimbT,
    //);
}

pub trait Fp256Parameters: FieldParameters<BigInteger = BigInteger> {}

#[derive(Derivative)]
#[derivative(
    Default(bound = ""),
    Hash(bound = ""),
    Clone(bound = ""),
    Copy(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
pub struct Fp256<P>(
    pub BigInteger,
    #[derivative(Debug = "ignore")]
    #[doc(hidden)]
    pub PhantomData<P>,
);

impl<P: Fp256Parameters> Fp256<P> {
    #[inline]
    pub fn new(element: BigInteger) -> Self {
        Fp256::<P>(element, PhantomData)
    }

    #[inline]
    fn is_valid(&self) -> bool {
        self.0 < P::MODULUS
    }

    #[inline]
    fn reduce(&mut self) {
        if !self.is_valid() {
            self.0.sub_noborrow(&P::MODULUS);
        }
    }
}

impl<P: Fp256Parameters> Zero for Fp256<P> {
    #[inline]
    fn zero() -> Self {
        Fp256::<P>(BigInteger::from(0), PhantomData)
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl<P: Fp256Parameters> One for Fp256<P> {
    #[inline]
    fn one() -> Self {
        Fp256::<P>(P::R, PhantomData)
    }

    #[inline]
    fn is_one(&self) -> bool {
        self == &Self::one()
    }
}

impl<P: Fp256Parameters> Field for Fp256<P> {
    type BasePrimeField = Self;

    // 256/64 = 4 limbs.
    impl_field_from_random_bytes_with_flags!(4);

    fn from_base_prime_field(other: Self::BasePrimeField) -> Self {
        other
    }

    /// Returns the constant 2^{-1}.
    fn half() -> Self {
        // Compute 1/2 `(p+1)/2` as `1/2`.
        // This is cheaper than `Self::one().double().inverse()`
        let mut two_inv = P::MODULUS;
        two_inv.add_nocarry(&1u64.into());
        two_inv.div2();
        Self::from_repr(two_inv).unwrap() // Guaranteed to be valid.
    }

    #[inline]
    fn double(&self) -> Self {
        let mut temp = *self;
        temp.double_in_place();
        temp
    }

    #[inline]
    fn double_in_place(&mut self) {
        unsafe {
            add_mod_256(((self.0).0).as_mut_ptr(), ((self.0).0).as_ptr(), ((self.0).0).as_ptr(), (P::MODULUS.0).as_ptr());
        }
    }

    #[inline]
    fn characteristic<'a>() -> &'a [u64] {
        P::MODULUS.as_ref()
    }

    #[inline]
    fn square(&self) -> Self {
        let mut temp = *self;
        temp.square_in_place();
        temp
    }

    #[inline]
    fn square_in_place(&mut self) -> &mut Self {
        unsafe {
            sqrx_mont_sparse_256(((self.0).0).as_mut_ptr(), ((self.0).0).as_ptr(), (P::MODULUS.0).as_ptr(), P::INV);
        }
        self
    }

    #[inline]
    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            // Guajardo Kumar Paar Pelzl
            // Efficient Software-Implementation of Finite Fields with Applications to
            // Cryptography
            // Algorithm 16 (BEA for Inversion in Fp)

            let one = BigInteger::from(1);

            let mut u = self.0;
            let mut v = P::MODULUS;
            let mut b = Fp256::<P>(P::R2, PhantomData); // Avoids unnecessary reduction step.
            let mut c = Self::zero();

            while u != one && v != one {
                while u.is_even() {
                    u.div2();

                    if b.0.is_even() {
                        b.0.div2();
                    } else {
                        b.0.add_nocarry(&P::MODULUS);
                        b.0.div2();
                    }
                }

                while v.is_even() {
                    v.div2();

                    if c.0.is_even() {
                        c.0.div2();
                    } else {
                        c.0.add_nocarry(&P::MODULUS);
                        c.0.div2();
                    }
                }

                if v < u {
                    u.sub_noborrow(&v);
                    b.sub_assign(&c);
                } else {
                    v.sub_noborrow(&u);
                    c.sub_assign(&b);
                }
            }

            if u == one { Some(b) } else { Some(c) }
        }
    }

    fn inverse_in_place(&mut self) -> Option<&mut Self> {
        if let Some(inverse) = self.inverse() {
            *self = inverse;
            Some(self)
        } else {
            None
        }
    }

    #[inline]
    fn frobenius_map(&mut self, _: usize) {
        // No-op: No effect in a prime field.
    }
}

impl<P: Fp256Parameters> PrimeField for Fp256<P> {
    type BigInteger = BigInteger;
    type Parameters = P;

    #[inline]
    fn from_repr(r: BigInteger) -> Option<Self> {
        let mut r = Fp256(r, PhantomData);
        if r.is_zero() {
            Some(r)
        } else if r.is_valid() {
            r *= &Fp256(P::R2, PhantomData);
            Some(r)
        } else {
            None
        }
    }

    #[inline]
    fn to_repr(&self) -> BigInteger {
        let mut tmp = self.0;
        let mut r = tmp.0;
        // Montgomery Reduction
        let k = r[0].wrapping_mul(P::INV);
        let mut carry = 0;
        fa::mac_with_carry(r[0], k, P::MODULUS.0[0], &mut carry);
        r[1] = fa::mac_with_carry(r[1], k, P::MODULUS.0[1], &mut carry);
        r[2] = fa::mac_with_carry(r[2], k, P::MODULUS.0[2], &mut carry);
        r[3] = fa::mac_with_carry(r[3], k, P::MODULUS.0[3], &mut carry);
        r[0] = carry;

        let k = r[1].wrapping_mul(P::INV);
        let mut carry = 0;
        fa::mac_with_carry(r[1], k, P::MODULUS.0[0], &mut carry);
        r[2] = fa::mac_with_carry(r[2], k, P::MODULUS.0[1], &mut carry);
        r[3] = fa::mac_with_carry(r[3], k, P::MODULUS.0[2], &mut carry);
        r[0] = fa::mac_with_carry(r[0], k, P::MODULUS.0[3], &mut carry);
        r[1] = carry;

        let k = r[2].wrapping_mul(P::INV);
        let mut carry = 0;
        fa::mac_with_carry(r[2], k, P::MODULUS.0[0], &mut carry);
        r[3] = fa::mac_with_carry(r[3], k, P::MODULUS.0[1], &mut carry);
        r[0] = fa::mac_with_carry(r[0], k, P::MODULUS.0[2], &mut carry);
        r[1] = fa::mac_with_carry(r[1], k, P::MODULUS.0[3], &mut carry);
        r[2] = carry;

        let k = r[3].wrapping_mul(P::INV);
        let mut carry = 0;
        fa::mac_with_carry(r[3], k, P::MODULUS.0[0], &mut carry);
        r[0] = fa::mac_with_carry(r[0], k, P::MODULUS.0[1], &mut carry);
        r[1] = fa::mac_with_carry(r[1], k, P::MODULUS.0[2], &mut carry);
        r[2] = fa::mac_with_carry(r[2], k, P::MODULUS.0[3], &mut carry);
        r[3] = carry;

        tmp.0 = r;
        tmp
    }

    #[inline]
    fn to_repr_unchecked(&self) -> BigInteger {
        let r = *self;
        r.0
    }
}

impl<P: Fp256Parameters> FftField for Fp256<P> {
    type FftParameters = P;

    #[inline]
    fn two_adic_root_of_unity() -> Self {
        Self(P::TWO_ADIC_ROOT_OF_UNITY, PhantomData)
    }

    #[inline]
    fn large_subgroup_root_of_unity() -> Option<Self> {
        Some(Self(P::LARGE_SUBGROUP_ROOT_OF_UNITY?, PhantomData))
    }

    #[inline]
    fn multiplicative_generator() -> Self {
        Self(P::GENERATOR, PhantomData)
    }
}

impl<P: Fp256Parameters> SquareRootField for Fp256<P> {
    #[inline]
    fn legendre(&self) -> LegendreSymbol {
        use crate::LegendreSymbol::*;

        // s = self^((MODULUS - 1) // 2)
        let mut s = self.pow(P::MODULUS_MINUS_ONE_DIV_TWO);
        s.reduce();

        if s.is_zero() {
            Zero
        } else if s.is_one() {
            QuadraticResidue
        } else {
            QuadraticNonResidue
        }
    }

    // Only works for p = 1 (mod 16).
    #[inline]
    fn sqrt(&self) -> Option<Self> {
        sqrt_impl!(Self, P, self)
    }

    fn sqrt_in_place(&mut self) -> Option<&mut Self> {
        if let Some(sqrt) = self.sqrt() {
            *self = sqrt;
            Some(self)
        } else {
            None
        }
    }
}

impl<P: Fp256Parameters + PoseidonDefaultParameters> PoseidonDefaultField for Fp256<P> {}

impl_primefield_from_int!(Fp256, u128, Fp256Parameters);
impl_primefield_from_int!(Fp256, u64, Fp256Parameters);
impl_primefield_from_int!(Fp256, u32, Fp256Parameters);
impl_primefield_from_int!(Fp256, u16, Fp256Parameters);
impl_primefield_from_int!(Fp256, u8, Fp256Parameters);

impl_primefield_standard_sample!(Fp256, Fp256Parameters);

impl_add_sub_from_field_ref!(Fp256, Fp256Parameters);
impl_mul_div_from_field_ref!(Fp256, Fp256Parameters);

impl<P: Fp256Parameters> ToBits for Fp256<P> {
    fn to_bits_le(&self) -> Vec<bool> {
        let mut bits_vec = self.to_repr().to_bits_le();
        bits_vec.truncate(P::MODULUS_BITS as usize);
        bits_vec
    }

    fn to_bits_be(&self) -> Vec<bool> {
        let mut bits_vec = self.to_bits_le();
        bits_vec.reverse();
        bits_vec
    }
}

impl<P: Fp256Parameters> ToBytes for Fp256<P> {
    #[inline]
    fn write_le<W: Write>(&self, writer: W) -> IoResult<()> {
        self.to_repr().write_le(writer)
    }
}

impl<P: Fp256Parameters> FromBytes for Fp256<P> {
    #[inline]
    fn read_le<R: Read>(reader: R) -> IoResult<Self> {
        BigInteger::read_le(reader).and_then(|b| match Self::from_repr(b) {
            Some(f) => Ok(f),
            None => Err(FieldError::InvalidFieldElement.into()),
        })
    }
}

/// `Fp` elements are ordered lexicographically.
impl<P: Fp256Parameters> Ord for Fp256<P> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.to_repr().cmp(&other.to_repr())
    }
}

impl<P: Fp256Parameters> PartialOrd for Fp256<P> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<P: Fp256Parameters> FromStr for Fp256<P> {
    type Err = FieldError;

    /// Interpret a string of numbers as a (congruent) prime field element.
    /// Does not accept unnecessary leading zeroes or a blank string.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            return Err(FieldError::ParsingEmptyString);
        }

        if s == "0" {
            return Ok(Self::zero());
        }

        let mut res = Self::zero();

        let ten = Self::from_repr(<Self as PrimeField>::BigInteger::from(10)).ok_or(FieldError::InvalidFieldElement)?;

        let mut first_digit = true;

        for c in s.chars() {
            match c.to_digit(10) {
                Some(c) => {
                    if first_digit {
                        if c == 0 {
                            return Err(FieldError::InvalidString);
                        }

                        first_digit = false;
                    }

                    res.mul_assign(&ten);
                    res.add_assign(
                        &Self::from_repr(<Self as PrimeField>::BigInteger::from(u64::from(c)))
                            .ok_or(FieldError::InvalidFieldElement)?,
                    );
                }
                None => {
                    return Err(FieldError::ParsingNonDigitCharacter);
                }
            }
        }

        if !res.is_valid() { Err(FieldError::InvalidFieldElement) } else { Ok(res) }
    }
}

impl<P: Fp256Parameters> Debug for Fp256<P> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.to_repr())
    }
}

impl<P: Fp256Parameters> Display for Fp256<P> {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.to_repr())
    }
}

impl<P: Fp256Parameters> Neg for Fp256<P> {
    type Output = Self;

    #[inline]
    #[must_use]
    fn neg(self) -> Self {
        if !self.is_zero() {
            let mut tmp = P::MODULUS;
            tmp.sub_noborrow(&self.0);
            Fp256::<P>(tmp, PhantomData)
        } else {
            self
        }
    }
}

impl<'a, P: Fp256Parameters> Add<&'a Fp256<P>> for Fp256<P> {
    type Output = Self;

    #[inline]
    fn add(self, other: &Self) -> Self {
        let mut result = self;
        result.add_assign(other);
        result
    }
}

impl<'a, P: Fp256Parameters> Sub<&'a Fp256<P>> for Fp256<P> {
    type Output = Self;

    #[inline]
    fn sub(self, other: &Self) -> Self {
        let mut result = self;
        result.sub_assign(other);
        result
    }
}

impl<'a, P: Fp256Parameters> Mul<&'a Fp256<P>> for Fp256<P> {
    type Output = Self;

    #[inline]
    fn mul(self, other: &Self) -> Self {
        let mut result = self;
        result.mul_assign(other);
        result
    }
}

impl<'a, P: Fp256Parameters> Div<&'a Fp256<P>> for Fp256<P> {
    type Output = Self;

    #[inline]
    fn div(self, other: &Self) -> Self {
        let mut result = self;
        result.mul_assign(&other.inverse().unwrap());
        result
    }
}

impl<'a, P: Fp256Parameters> AddAssign<&'a Self> for Fp256<P> {
    #[inline]
    fn add_assign(&mut self, other: &Self) {
        unsafe {
            add_mod_256(((self.0).0).as_mut_ptr(), ((self.0).0).as_ptr(), ((other.0).0).as_ptr(), (P::MODULUS.0).as_ptr());
        }
    }
}

impl<'a, P: Fp256Parameters> SubAssign<&'a Self> for Fp256<P> {
    #[inline]
    fn sub_assign(&mut self, other: &Self) {
        unsafe {
            sub_mod_256(((self.0).0).as_mut_ptr(), ((self.0).0).as_ptr(), ((other.0).0).as_ptr(), (P::MODULUS.0).as_ptr());
         }
    }
}

impl<'a, P: Fp256Parameters> MulAssign<&'a Self> for Fp256<P> {
    #[inline]
    fn mul_assign(&mut self, other: &Self) {
        unsafe {
            mulx_mont_sparse_256(((self.0).0).as_mut_ptr(), ((self.0).0).as_ptr(), ((other.0).0).as_ptr(), (P::MODULUS.0).as_ptr(), P::INV);
        }
    }
}

impl<'a, P: Fp256Parameters> DivAssign<&'a Self> for Fp256<P> {
    #[inline]
    fn div_assign(&mut self, other: &Self) {
        self.mul_assign(&other.inverse().unwrap());
    }
}