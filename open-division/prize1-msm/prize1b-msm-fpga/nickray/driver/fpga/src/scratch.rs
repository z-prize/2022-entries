
// pub fn udigit16(scalar: &Scalar, column: usize) -> u32 {
//     const DIGITS_PER_LIMB: usize = (u64::BITS / u16::BITS) as usize;
//     const BITS_PER_DIGIT: usize = u16::BITS as usize;
//
//     let limb = column / DIGITS_PER_LIMB;
//     let offset = column % DIGITS_PER_LIMB;
//     let bit_offset = offset * BITS_PER_DIGIT;
//     (scalar[limb] >> bit_offset) as u16 as u32
// }
//
// pub fn sdigit16(scalar: &Scalar, column: u8) -> i32 {
//     const MAX_SIGNED: u32 = 1 << 15;
//     let mut carry = 0u32;
//     let mut signed_digit: i32 = 0;
//     for j in 0..=column {
//         let unsigned_digit = udigit16(scalar, j as usize) + carry;
//         carry = (unsigned_digit + MAX_SIGNED) >> 16;
//         signed_digit = (unsigned_digit as i32) - ((carry as i32) << 16);
//     }
//     signed_digit
// }
//
// pub fn sdigits16(scalar: &Scalar) -> [SDigit; COLUMNS] {
//     const MAX_SIGNED: u32 = 1 << 15;
//     let mut carry = 0u32;
//     let mut signed_digit: i32;
//     let mut digits = [0; COLUMNS];
//     for j in 0..COLUMNS {
//         let unsigned_digit = udigit16(scalar, j) + carry;
//         carry = (unsigned_digit + MAX_SIGNED) >> 16;
//         signed_digit = (unsigned_digit as i32) - ((carry as i32) << 16);
//         digits[j] = signed_digit;
//     }
//     digits
// }
//
// pub fn columnize(scalars: &[Scalar]) -> Vec<Column> {
//     let mut all_digits = Vec::with_capacity(COLUMNS);
//     for _ in 0..COLUMNS {
//         all_digits.push(vec![0; scalars.len()]);
//     }
//     for (i, scalar) in scalars.iter().enumerate() {
//         let sdigits = sdigits16(scalar);
//         for column in 0..COLUMNS {
//             all_digits[column][i] = sdigits[column];
//         }
//     }
//     all_digits
// }
//
// pub fn ucolumnize(scalars: &[Scalar], ucolumns: &mut Vec<Vec<u32>>) {
//     for (i, scalar) in scalars.iter().enumerate() {
//         for column in 0..COLUMNS {
//             ucolumns[column][i] = udigit16(scalar, column);
//         }
//     }
// }

