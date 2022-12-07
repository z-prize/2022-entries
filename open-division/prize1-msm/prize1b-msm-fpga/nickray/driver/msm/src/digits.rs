use seq_macro::seq;

pub type Digit = i16;
pub type Limb = u64;
pub type Scalar = [Limb; 4];

#[inline(always)]
pub fn single_digit_carry(carried: &Scalar, i: usize, j: u8) -> i16 {
    const HI: u16 = 1 << 15; //0x8000;

    let mut carry: bool;
    let mut s = carried[i];
    for _ in 0..j {
        carry = ((s as u16) & HI) != 0;
        s >>= 16;
        if carry {
            s += 1;
        }
    }
    s as i16
}

#[inline]
pub fn limb_carries(scalars: &[Scalar], carried_limbs: &mut [Scalar]) {
    const HI: u16 = 1 << 15;
    debug_assert_eq!(scalars.len(), carried_limbs.len());

    for (carried_limb, scalar) in carried_limbs.iter_mut().zip(scalars.iter()) {
        let [mut s, mut u1, mut u2, mut u3] = scalar;
        let mut carry: bool;

        // maybe put this back if it's easier in later use
        // carried_limb[0] = s;

        // each step is:
        // - carry calculation (leaving data untouched with 64B chunks)
        // - carry propagation (between 64B chunks)

        // 0
        // -> remains untouched

        // 1
        for _ in 0..3 {
            carry = ((s as u16) & HI) != 0;
            s >>= 16;
            if carry {
                s += 1;
            }
        }
        carry = s >= HI as _;
        u1 = u1.wrapping_add(carry as u64);
        carried_limb[1] = u1;

        if carry && u1 == 0 {
            u2 = u2.wrapping_add(1);
            if u2 == 0 {
                u3 = u3.wrapping_add(1);
            }
        }

        // 2
        s = carried_limb[1];
        for _ in 0..3 {
            carry = ((s as u16) & HI) != 0;
            s >>= 16;
            if carry {
                s += 1;
            }
        }
        carry = s >= HI as _;
        u2 = u2.wrapping_add(carry as u64);
        carried_limb[2] = u2;

        if carry && u2 == 0 {
            u3 = u3.wrapping_add(1);
        }

        // 3
        s = carried_limb[2];
        for _ in 0..3 {
            carry = ((s as u16) & HI) != 0;
            s >>= 16;
            if carry {
                s += 1;
            }
        }
        carry = s >= HI as _;
        // don't need wrapping add here, the scalar is assumed to be "small enough"
        u3 += carry as u64;
        carried_limb[3] = u3;
    }
}

pub fn display_scalars(scalars: &[Scalar], len: usize) {
    println!("len = {}", scalars.len());
    for scalar in scalars.iter().take(len) {
        println!(
            "{:016X} {:016X} {:016X} {:016X}",
            scalar[0], scalar[1], scalar[2], scalar[3],
        );
    }
}

// direct algorithm SIGNED-DIGITS on slide 43 of
// https://www.slideshare.net/GusGutoski/multiscalar-multiplication-state-of-the-art-and-new-ideas
// - iterate the unsigned digits ascending
// - if one is > MAX_SIGNED, then
//   - subtract MAX
//   - compensate by carrying 1 into next digit
//
// slightly faster than the previous "unsigned" method,
// we have 620ms instead of 660ms for 1x2^18
#[inline(always)]
pub fn signed_digit_16(scalar: &Scalar, i: usize) -> i32 {
    const MAX_SIGNED: u32 = 1 << 15;
    let mut carry = 0u32;
    let mut signed_digit: i32 = 0;
    for j in 0..=i {
        let unsigned_digit = unsigned_digit_16(scalar, j) + carry;
        // SIGNED-DIGITS simplifies to this
        carry = (unsigned_digit + MAX_SIGNED) >> 16;
        signed_digit = (unsigned_digit as i32) - ((carry as i32) << 16);
    }
    signed_digit
}

#[inline(always)]
pub fn unsigned_digit_16(scalar: &Scalar, i: usize) -> u32 {
    const DIGITS_PER_LIMB: usize = (u64::BITS / u16::BITS) as usize;
    const BITS_PER_DIGIT: usize = u16::BITS as usize;

    let limb = i / DIGITS_PER_LIMB;
    let offset = i % DIGITS_PER_LIMB;
    let bit_offset = offset * BITS_PER_DIGIT;
    (scalar.as_ref()[limb] >> bit_offset) as u16 as u32
}

#[inline(always)]
// this *actually* is faster, on 1x2^18 we have 560ms vs 620ms
pub fn unrolled_signed_digit_16(scalar: &Scalar, i: usize) -> i32 {
    debug_assert!(i < 16);
    const MAX_SIGNED: u32 = 1 << 15;
    let mut carry = 0u32;
    let mut signed_digit: i32;
    seq!(j in 0..16 {
        let unsigned_digit = unsigned_digit_16(scalar, j) + carry;
        carry = (unsigned_digit + MAX_SIGNED) >> 16;
        debug_assert!(carry <= 1);
        signed_digit = (unsigned_digit as i32) - ((carry as i32) << 16);
        if j == i {
            return signed_digit;
        }
    });
    unreachable!();
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn fast_digits() {
        let size = 16;
        let mut scalars = crate::random_scalars(size);

        // add some edge cases
        scalars[0] = [u64::MAX, scalars[0][1], scalars[0][2], scalars[0][3]];
        scalars[1] = [scalars[1][0], u64::MAX, scalars[1][2], scalars[1][3]];
        scalars[2] = [scalars[2][0], scalars[2][1], u64::MAX, scalars[2][3]];
        scalars[3] = [u64::MAX, u64::MAX, scalars[3][2], scalars[3][3]];
        scalars[4] = [u64::MAX, u64::MAX, u64::MAX, scalars[4][3]];

        // perform the initial limb-level carries
        let mut carried = vec![Scalar::default(); scalars.len()];
        limb_carries(&scalars, carried.as_mut_slice());

        // the actual test: all digit-level carries
        for (point, (scalar, carried)) in scalars.iter().zip(carried.iter()).enumerate() {
            for i in 0..4 {
                for j in 0..4 {
                    let k = 4 * i + j;
                    println!("point = {point}, i = {i}, j = {j}, k = {k}");

                    // first limb does not need the limb carry,
                    // and in fact we don't write it in the output.
                    if i == 0 {
                        assert_eq!(
                            single_digit_carry(scalar, i, j as u8),
                            unrolled_signed_digit_16(scalar, k) as i16,
                        );
                    } else {
                        assert_eq!(
                            single_digit_carry(carried, i, j as u8),
                            unrolled_signed_digit_16(scalar, k) as i16,
                        );
                    }
                }
            }
        }
    }
}
