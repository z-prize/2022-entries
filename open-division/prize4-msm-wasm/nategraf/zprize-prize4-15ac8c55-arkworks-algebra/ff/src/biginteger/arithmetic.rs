use ark_std::{vec, vec::Vec};

macro_rules! adc {
    ($a:expr, $b:expr, &mut $carry:expr$(,)?) => {{
        let tmp = ($a as u128) + ($b as u128) + ($carry as u128);
        $carry = (tmp >> 64) as u64;
        tmp as u64
    }};
}

/// adc with the option to accept a variable number of args.
/// carry is given as the first arg, followed by any number of inputs.
// NOTE(victor) Need to look at the assembly output for this since it was likely written
// specifically to ensure the compiler implements it with a particular instruction and I may have
// borked that.
#[cfg(feature = "square-no-carry")]
macro_rules! adc_var {
    (&mut $carry:expr, $($x:expr),*) => {{
        let tmp = $(($x as u128) + )* ($carry as u128);
        $carry = (tmp >> 64) as u64;
        tmp as u64
    }};
}

/// Calculate a + b + carry, returning the sum and modifying the
/// carry value.
#[inline(always)]
pub(crate) fn adc(a: u64, b: u64, carry: &mut u64) -> u64 {
    let tmp = a as u128 + b as u128 + *carry as u128;
    *carry = (tmp >> 64) as u64;
    tmp as u64
}

/// Calculate a + b + carry, returning the sum
#[inline(always)]
pub(crate) fn adc_no_carry(a: u64, b: u64, carry: &mut u64) -> u64 {
    let tmp = a as u128 + b as u128 + *carry as u128;
    tmp as u64
}

/// Calculate a + b returning the 64-bit word result and a boolean carry.
/// Alternative implementation of add with the intention that it be more friendly to WASM.
/// In particular, this function does not use any u128 values.
#[inline(always)]
#[cfg(feature = "no-u128")]
pub(crate) fn add(a: u64, b: u64) -> (u64, u64) {
    let tmp = (a as u128) + (b as u128);
    ((tmp >> 64) as u64, tmp as u64)
}

#[macro_export]
macro_rules! sbb {
    ($a:expr, $b:expr, &mut $borrow:expr$(,)?) => {{
        let tmp = (1u128 << 64) + ($a as u128) - ($b as u128) - ($borrow as u128);
        $borrow = if tmp >> 64 == 0 { 1 } else { 0 };
        tmp as u64
    }};
}

/// Calculate a - b - borrow, returning the result and modifying
/// the borrow value.
#[inline(always)]
pub(crate) fn sbb(a: u64, b: u64, borrow: &mut u64) -> u64 {
    sbb!(a, b, &mut *borrow)
}

/// Calculate a + b * c, returning the lower 64 bits of the result and setting
/// `carry` to the upper 64 bits.
#[inline(always)]
#[cfg(not(feature = "no-u128"))]
pub(crate) fn mac(a: u64, b: u64, c: u64, carry: &mut u64) -> u64 {
    let tmp = (a as u128) + (b as u128 * c as u128);
    *carry = (tmp >> 64) as u64;
    tmp as u64
}

/// Calculate a + b * c, discarding the lower 64 bits of the result and setting
/// `carry` to the upper 64 bits.
#[inline(always)]
#[cfg(not(feature = "no-u128"))]
pub(crate) fn mac_discard(a: u64, b: u64, c: u64, carry: &mut u64) {
    let tmp = (a as u128) + (b as u128 * c as u128);
    *carry = (tmp >> 64) as u64;
}

macro_rules! mac_with_carry {
    ($a:expr, $b:expr, $c:expr, &mut $carry:expr$(,)?) => {{
        let tmp = ($a as u128) + ($b as u128 * $c as u128) + ($carry as u128);
        $carry = (tmp >> 64) as u64;
        tmp as u64
    }};
}

macro_rules! mac {
    ($a:expr, $b:expr, $c:expr, &mut $carry:expr$(,)?) => {{
        let tmp = ($a as u128) + ($b as u128 * $c as u128);
        $carry = (tmp >> 64) as u64;
        tmp as u64
    }};
}

/// Calculate a + (b * c) + carry, returning the least significant digit
/// and setting carry to the most significant digit.
#[inline(always)]
#[cfg(not(feature = "no-u128"))]
pub(crate) fn mac_with_carry(a: u64, b: u64, c: u64, carry: &mut u64) -> u64 {
    let tmp = (a as u128) + (b as u128 * c as u128) + (*carry as u128);
    *carry = (tmp >> 64) as u64;
    tmp as u64
}

/// Calculate a + b * c returning the two 64-bit word result as (upper, lower).
/// Alternative implementation of mac with the intention that it be more friendly to WASM.
/// In particular, this function does not use any u128 values.
#[inline(always)]
#[cfg(feature = "no-u128")]
pub(crate) fn mac(a: u64, b: u64, c: u64, carry: &mut u64) -> u64 {
    // Split the input product values into two 32-bit values.
    let (a1, a0) = (a >> 32, a & 0xFFFFFFFF);
    let (b1, b0) = (b >> 32, b & 0xFFFFFFFF);
    let (c1, c0) = (c >> 32, c & 0xFFFFFFFF);

    // Compute the 4 32-bit multiplications with 64-bit results.
    let b0c0 = b0 * c0;
    let b1c0 = b1 * c0;
    let b0c1 = b0 * c1;
    let b1c1 = b1 * c1;

    // Sum the multiplication results into their respective 64-bit result words. Simultanously add
    // in the input value a, which can be done without trigger extra carries.
    // Results in (mac1, mac0), the 128-bit result of a + b*c.
    let (mid_carry, mid) = add(b1c0 + a1, b0c1);
    let (mac0_carry, mac0) = add((mid << 32) | a0, b0c0);
    let mac1 = b1c1 + (((mid_carry as u64) << 32) | (mid >> 32)) + (mac0_carry as u64);

    *carry = mac1;
    mac0
}

/// Calculate a + b * c + carry returning the two 64-bit word result as (upper, lower).
/// Alternative implementation of mac_with_carry with the intention that it be more friendly to
/// WASM.  In particular, this function does not use any u128 values.
#[inline(always)]
#[cfg(feature = "no-u128")]
pub(crate) fn mac_with_carry(a: u64, b: u64, c: u64, carry: &mut u64) -> u64 {
    // Split the input product values into two 32-bit values.
    let (a1, a0) = (a >> 32, a & 0xFFFFFFFF);
    let (b1, b0) = (b >> 32, b & 0xFFFFFFFF);
    let (c1, c0) = (c >> 32, c & 0xFFFFFFFF);
    let (d1, d0) = (*carry >> 32, *carry & 0xFFFFFFFF);

    // Compute the 4 32-bit multiplications with 64-bit results.
    let b0c0 = b0 * c0;
    let b1c0 = b1 * c0;
    let b0c1 = b0 * c1;
    let b1c1 = b1 * c1;

    // Sum the multiplication results into their respective 64-bit result words. Simultanously add
    // in the input values a and carry, which can be done without trigger extra carries.
    // Results in (mac1, mac0), the 128-bit result of a + b*c + carry.
    let (mid_carry, mid) = add(b1c0 + a1, b0c1 + d1);
    let (mac0_carry, mac0) = add((mid << 32) | a0, b0c0 + d0);
    let mac1 = b1c1 + (((mid_carry as u64) << 32) | (mid >> 32)) + (mac0_carry as u64);

    *carry = mac1;
    mac0
}

/// Calculate a + b * c returning the upper 64-bit word result.
/// Alternative implementation of mac_discard with the intention that it be more friendly to WASM.
/// In particular, this function does not use any u128 values.
#[inline(always)]
#[cfg(feature = "no-u128")]
pub(crate) fn mac_discard(a: u64, b: u64, c: u64, carry: &mut u64) {
    let _ = mac(a, b, c, carry);
}

/// Compute the NAF (non-adjacent form) of num
pub fn find_naf(num: &[u64]) -> Vec<i8> {
    let is_zero = |num: &[u64]| num.iter().all(|x| *x == 0u64);
    let is_odd = |num: &[u64]| num[0] & 1 == 1;
    let sub_noborrow = |num: &mut [u64], z: u64| {
        let mut other = vec![0u64; num.len()];
        other[0] = z;
        let mut borrow = 0;

        for (a, b) in num.iter_mut().zip(other) {
            *a = sbb(*a, b, &mut borrow);
        }
    };
    let add_nocarry = |num: &mut [u64], z: u64| {
        let mut other = vec![0u64; num.len()];
        other[0] = z;
        let mut carry = 0;

        for (a, b) in num.iter_mut().zip(other) {
            *a = adc(*a, b, &mut carry);
        }
    };
    let div2 = |num: &mut [u64]| {
        let mut t = 0;
        for i in num.iter_mut().rev() {
            let t2 = *i << 63;
            *i >>= 1;
            *i |= t;
            t = t2;
        }
    };

    let mut num = num.to_vec();
    let mut res = vec![];

    while !is_zero(&num) {
        let z: i8;
        if is_odd(&num) {
            z = 2 - (num[0] % 4) as i8;
            if z >= 0 {
                sub_noborrow(&mut num, z as u64)
            } else {
                add_nocarry(&mut num, (-z) as u64)
            }
        } else {
            z = 0;
        }
        res.push(z);
        div2(&mut num);
    }

    res
}

/// We define relaxed NAF as a variant of NAF with a very small tweak.
///
/// Note that the cost of scalar multiplication grows with the length of the sequence (for doubling)
/// plus the Hamming weight of the sequence (for addition, or subtraction).
///
/// NAF is optimizing for the Hamming weight only and therefore can be suboptimal.
/// For example, NAF may generate a sequence (in little-endian) of the form ...0 -1 0 1.
///
/// This can be rewritten as ...0 1 1 to avoid one doubling, at the cost that we are making an
/// exception of non-adjacence for the most significant bit.
///
/// Since this representation is no longer a strict NAF, we call it "relaxed NAF".
pub fn find_relaxed_naf(num: &[u64]) -> Vec<i8> {
    let mut res = find_naf(num);

    let len = res.len();
    if res[len - 2] == 0 && res[len - 3] == -1 {
        res[len - 3] = 1;
        res[len - 2] = 1;
        res.resize(len - 1, 0);
    }

    res
}

#[test]
fn test_mac_does_not_overflow() {
    let mut carry = u64::MAX;
    let _ = mac(u64::MAX, u64::MAX, u64::MAX, &mut carry);
}

#[test]
fn test_mac_with_carry_does_not_overflow() {
    let mut carry = u64::MAX;
    let _ = mac_with_carry(u64::MAX, u64::MAX, u64::MAX, &mut carry);
}

#[test]
fn test_find_relaxed_naf_usefulness() {
    let vec = find_naf(&[12u64]);
    assert_eq!(vec.len(), 5);

    let vec = find_relaxed_naf(&[12u64]);
    assert_eq!(vec.len(), 4);
}

#[test]
fn test_find_relaxed_naf_correctness() {
    use ark_std::{One, UniformRand, Zero};
    use num_bigint::BigInt;

    let mut rng = ark_std::test_rng();

    for _ in 0..10 {
        let num = [
            u64::rand(&mut rng),
            u64::rand(&mut rng),
            u64::rand(&mut rng),
            u64::rand(&mut rng),
        ];
        let relaxed_naf = find_relaxed_naf(&num);

        let test = {
            let mut sum = BigInt::zero();
            let mut cur = BigInt::one();
            for v in relaxed_naf {
                sum += cur.clone() * v;
                cur *= 2;
            }
            sum
        };

        let test_expected = {
            let mut sum = BigInt::zero();
            let mut cur = BigInt::one();
            for v in num.iter() {
                sum += cur.clone() * v;
                cur <<= 64;
            }
            sum
        };

        assert_eq!(test, test_expected);
    }
}
