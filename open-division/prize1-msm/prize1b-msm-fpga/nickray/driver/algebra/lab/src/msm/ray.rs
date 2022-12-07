use ark_ec::msm::Msm;
use ark_ff::{BigInteger, PrimeField, Zero};
use core::{
    cmp::Ordering,
    ops::{Add, AddAssign, Shl, ShlAssign, SubAssign},
};
use std::ops::Neg;

fn ceil(numerator: usize, denominator: usize) -> usize {
    (numerator + denominator - 1) / denominator
}

pub struct Digits {
    // stores digits as "digit-major", i.e.
    // starting with the first digit of all scalars,
    // then the second digit, and so one.
    data: Vec<i32>,
    scalar_count: usize,
}

impl Digits {
    pub fn new<I: BigInteger>(b: usize, c: usize, scalars: &[I]) -> Self {
        // ceil(b/c)
        let digit_count = ceil(b, c);
        let scalar_count = scalars.len();

        // allocate upfront
        let mut data = vec![0; scalar_count * digit_count];

        for (i, scalar) in scalars.iter().enumerate() {
            let digit_iterator = data
                .iter_mut()
                .skip(i)
                .step_by(scalar_count)
                .take(digit_count);
            signed_digits(scalar, c, digit_iterator);
        }

        Self { data, scalar_count }
    }
}

impl core::ops::Index<usize> for Digits {
    type Output = [i32];

    /// returns i'th digit of all the scalars
    fn index(&self, i: usize) -> &[i32] {
        &self.data[i * self.scalar_count..][..self.scalar_count]
    }
}

#[inline]
fn unsigned_digit(i: usize, c: usize, limbs: &[u64]) -> u32 {
    debug_assert!(c < 32);
    // let (limb, offset) = div_rem(i * c, 64);
    let first_bit = i * c;
    let limb = first_bit / 64;
    let offset = first_bit % 64;
    let mask = (1 << c) - 1;

    let value = if offset < 64 - c || limb == limbs.len() - 1 {
        // This window's bits are contained in a single u64,
        // or it's the last u64 anyway.
        limbs[limb] >> offset
    } else {
        // Combine the current u64's bits with the bits from the next u64
        (limbs[limb] >> offset) | (limbs[limb + 1] << (64 - offset))
    };

    (value as u32) & mask
}

/// a is stored little-endian in a slice of u64
/// our goal is to extract the representation in a slice w-bit signed integers
/// (stored as i32).
fn signed_digits<'l>(a: &impl BigInteger, c: usize, digits: impl Iterator<Item = &'l mut i32>) {
    let scalar = a.as_ref();
    let max_signed: u32 = 1 << (c - 1);
    let mut carry = 0u32;

    for (i, signed_digit) in digits.enumerate() {
        let unsigned_digit = unsigned_digit(i, c, scalar) + carry;

        // direct algorithm (SIGNED-DIGITS on slide 43 of
        // https://www.slideshare.net/GusGutoski/multiscalar-multiplication-state-of-the-art-and-new-ideas)
        //
        // if >= 2^{w - 1}, then
        //   * a_i <- a_i - 2^w
        //   * a_{i + 1} <- a_{i + 1} + 1
        // else;
        //   * keep a_i

        // *signed_digit = if unsigned_digit > max_signed {
        //     carry = 1;
        //     unsigned_digit as i32 - radix
        // } else {
        //     carry = 0;
        //     unsigned_digit as i32
        // };

        // by inspection, in both cases:
        carry = (unsigned_digit + max_signed) >> c;
        *signed_digit = (unsigned_digit as i32) - ((carry as i32) << c);
    }
}

#[repr(i8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Op {
    Add = 1,
    Sub = -1,
    Skip = 0,
}

impl Default for Op {
    fn default() -> Self {
        Self::Skip
    }
}

impl From<Ordering> for Op {
    fn from(ordering: Ordering) -> Op {
        match ordering {
            Ordering::Greater => Op::Add,
            Ordering::Less => Op::Sub,
            Ordering::Equal => Op::Skip,
        }
    }
}

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Bucket {
    pub op: Op,
    pub bucket: u32,
}

impl Bucket {
    pub fn new(op: Op, bucket: u32) -> Self {
        Self { op, bucket }
    }
}

pub struct Buckets {
    // stores digits as "digit-major", i.e.
    // starting with the first digit of all scalars,
    // then the second digit, and so one.
    data: Vec<Bucket>,
    scalar_count: usize,
}

impl Buckets {
    pub fn new<I: BigInteger>(b: usize, c: usize, scalars: &[I]) -> Self {
        // ceil(b/c)
        let bucket_count = ceil(b, c);
        let scalar_count = scalars.len();

        // allocate upfront
        let mut data = vec![Default::default(); scalar_count * bucket_count];

        for (i, scalar) in scalars.iter().enumerate() {
            let bucket_iterator = data
                .iter_mut()
                .skip(i)
                .step_by(scalar_count)
                .take(bucket_count);
            op_buckets(scalar, c, bucket_iterator);
        }

        Self { data, scalar_count }
    }
}

impl core::ops::Index<usize> for Buckets {
    type Output = [Bucket];

    /// returns i'th digit of all the scalars
    fn index(&self, i: usize) -> &Self::Output {
        &self.data[i * self.scalar_count..][..self.scalar_count]
    }
}

/// a is stored little-endian in a slice of u64
/// our goal is to extract the representation in a slice w-bit signed integers
/// (stored as i32).
fn op_buckets<'l>(a: &impl BigInteger, c: usize, buckets: impl Iterator<Item = &'l mut Bucket>) {
    let scalar = a.as_ref();
    let max_signed: u32 = 1 << (c - 1);
    let mut carry = 0u32;

    for (i, op_bucket) in buckets.enumerate() {
        let unsigned_digit = unsigned_digit(i, c, scalar) + carry;

        // direct algorithm (SIGNED-DIGITS on slide 43 of
        // https://www.slideshare.net/GusGutoski/multiscalar-multiplication-state-of-the-art-and-new-ideas)
        //
        // if >= 2^{w - 1}, then
        //   * a_i <- a_i - 2^w
        //   * a_{i + 1} <- a_{i + 1} + 1
        // else;
        //   * keep a_i

        // *signed_digit = if unsigned_digit > max_signed {
        //     carry = 1;
        //     unsigned_digit as i32 - radix
        // } else {
        //     carry = 0;
        //     unsigned_digit as i32
        // };

        // by inspection, in both cases:
        carry = (unsigned_digit + max_signed) >> c;
        let signed_digit = (unsigned_digit as i32) - ((carry as i32) << c);
        *op_bucket = Bucket::new(signed_digit.cmp(&0).into(), signed_digit.abs() as u32)
    }
}

/// Optimized implementation of multi-scalar multiplication.
pub fn msm_bigint<Input: Copy + Neg, T: Msm<Input> + 'static>(
    c: usize,
    points: &[Input],
    scalars: &[<T::Scalar as PrimeField>::BigInt],
) -> T {
    debug_assert_eq!(points.len(), scalars.len());
    // need some wiggle room to avoid overflows
    debug_assert!(c < 32);
    let b = T::Scalar::MODULUS_BIT_SIZE as usize;
    let window_count = ceil(b, c);
    let bucket_count = 1 << (c - 1);

    let digits = Digits::new(b, c, &scalars);
    let windows: Vec<_> = ark_std::cfg_into_iter!(0..window_count)
        .map(|window| {
            // for instance, +42 and -42 go into bucket 41
            let mut buckets = vec![T::zero(); bucket_count];

            for (digit, point) in digits[window].iter().zip(points.iter()) {
                let bucket = digit.abs() as usize; // - 1;
                match digit.cmp(&0) {
                    Ordering::Greater => buckets[bucket - 1] += point,
                    Ordering::Less => buckets[bucket - 1] -= point,
                    Ordering::Equal => (),
                }
            }
            aggregate_buckets(buckets.iter().rev())
        })
        .collect();

    aggregate_windows(c, windows.iter().rev())
}

pub fn full_msm<Input: Copy + Neg, T: Msm<Input> + 'static>(
    c: usize,
    points: &[Input],
    scalars: &[<T::Scalar as PrimeField>::BigInt],
) -> T {
    debug_assert_eq!(points.len(), scalars.len());
    // need some wiggle room to avoid overflows
    debug_assert!(c < 32);
    let b = T::Scalar::MODULUS_BIT_SIZE as usize;
    let window_count = ceil(b, c);
    let bucket_count = 1 << (c - 1);

    let op_buckets = Buckets::new(b, c, &scalars);
    let windows: Vec<_> = ark_std::cfg_into_iter!(0..window_count)
        .map(|window| window_msm(bucket_count, &op_buckets[window], points))
        .collect();

    aggregate_windows(c, windows.iter().rev())
}

pub fn window_msm<I, T>(bucket_count: usize, buckets: &[Bucket], points: &[I]) -> T
where
    I: Copy + Neg,
    T: Msm<I> + 'static,
{
    // for instance, +42 and -42 go into bucket 41
    let mut acc = vec![T::zero(); bucket_count];

    // TODO: insert scheduler
    let inputs = buckets.iter().zip(points.iter());

    for (Bucket { op, bucket }, point) in inputs {
        match op {
            Op::Add => acc[*bucket as usize - 1] += point,
            Op::Sub => acc[*bucket as usize - 1] -= point,
            _ => (),
        }
    }
    aggregate_buckets(acc.iter().rev())
}

pub fn aggregate_buckets<'a, T>(iter: impl Iterator<Item = &'a T>) -> T
where
    T: for<'l> AddAssign<&'l T> + ShlAssign<usize> + Zero + 'static,
{
    // Compute sum_{i in 0..num_buckets} (sum_{j in i..num_buckets} bucket[j])
    // This is computed below for b buckets, using 2b curve additions.
    //
    // We could first normalize `buckets` and then use mixed-addition
    // here, but that's slower for the kinds of groups we care about
    // (Short Weierstrass curves and Twisted Edwards curves).
    // In the case of Short Weierstrass curves,
    // mixed addition saves ~4 field multiplications per addition.
    // However normalization (with the inversion batched) takes ~6
    // field multiplications per element,
    // hence batch normalization is a slowdown.

    // `running_sum` = sum_{j in i..num_buckets} bucket[j],
    // where we iterate backward from i = num_buckets to 0.
    let mut sum = T::zero();
    let mut running_sum = T::zero();
    iter.for_each(|bucket| {
        running_sum += bucket;
        sum += &running_sum;
    });
    sum
}

/// Shifts window sums into correct position and sums up
pub fn aggregate_windows<'a, T>(c: usize, mut iter: impl Iterator<Item = &'a T>) -> T
where
    T: Add<&'a T, Output = T> + Copy + ShlAssign<usize> + Zero + 'static,
{
    // We're traversing windows from high to low.
    let highest = *iter.next().unwrap();
    iter
        // could use .reduce here, but it's not less gymnastics
        .fold(highest, |mut sum, window| {
            sum <<= c;
            sum + window
        })
}
