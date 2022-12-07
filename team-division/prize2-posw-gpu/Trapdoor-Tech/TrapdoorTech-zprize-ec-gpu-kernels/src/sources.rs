// adapt Filecoin code for most of the field ops code generation
// https://github.com/filecoin-project/ec-gpu/blob/master/ec-gpu-gen/src/source.rs

use super::structs::{Field, FpParameters, PrimeField};
use log::{debug, info};
use std::any::TypeId;
use std::fmt::Write;

static COMMON_SRC: &str = include_str!("./kernels/common.cl");
static FIELD_SRC: &str = include_str!("./kernels/field.cl");
static FIELD2_SRC: &str = include_str!("./kernels/field2.cl");
static EC_SRC: &str = include_str!("./kernels/ec.cl");
static FFT_SRC: &str = include_str!("./kernels/fft.cl");
static MULTIEXP_SRC: &str = include_str!("./kernels/multiexp.cl");
static POLYNOMIAL_SRC: &str = include_str!("./kernels/polynomial.cl");

/// Divide anything into 64bit chunks
fn limbs_of_u32<T>(value: &T) -> &[u32] {
    unsafe {
        std::slice::from_raw_parts(
            value as *const T as *const u32,
            std::mem::size_of::<T>() / 4,
        )
    }
}

fn limbs_of_u64<T>(value: &T) -> &[u64] {
    unsafe {
        std::slice::from_raw_parts(
            value as *const T as *const u64,
            std::mem::size_of::<T>() / 8,
        )
    }
}

fn limbs_of_p_u32<T>(value: &[T]) -> &[u32] {
    unsafe {
        std::slice::from_raw_parts(
            value.as_ptr() as *const T as *const u32,
            std::mem::size_of::<T>() * value.len() / 4,
        )
    }
}

fn limbs_of_p_u64<T>(value: &[T]) -> &[u64] {
    unsafe {
        std::slice::from_raw_parts(
            value.as_ptr() as *const T as *const u64,
            std::mem::size_of::<T>() * value.len() / 8,
        )
    }
}

/// Generates the source for FFT and Multiexp operations.
pub fn gen_all_source<F1: Field, F2: Field, L: Limb>() -> String {
    vec![
        common(),
        gen_ec_source::<F1, F2, L>(),
        poly("Fr"),
        fft("Fr"),
        multiexp("G1", "Fr"),
    ]
    .join("\n\n")
}

/// Generates the source for the elliptic curve and group operations, as defined by `E`.
///
/// The code from the [`common()`] call needs to be included before this on is used.
pub fn gen_ec_source<F1: Field, F2: Field, L: Limb>() -> String {
    vec![
        field::<F1, L>("Fr"),
        field::<F2, L>("Fq"),
        gen_ed_a_d::<L>("Fq"),
        ec("Fq", "G1"),
    ]
    .join("\n\n")
}

/// Only Fq_ED_A is useful
pub fn gen_ed_a_d<L: Limb>(field: &str) -> String {
    use crate::structs::Fq;
    use ark_ff::One;
    use std::ops::{Div, Mul};
    use std::str::FromStr;

    let alpha = Fq::from_str("80949648264912719408558363140637477264845294720710499478137287262712535938301461879813459410946").unwrap();
    let beta = Fq::from_str("207913745465435703873309001080708636764682407053260289242004673792544811711776497012639468972230205966814119707502").unwrap();

    let three = Fq::from(3u64);

    let B = Fq::one() / beta;
    let A = three * B * alpha;

    let a = (A + Fq::from(2u64)) / B;
    let d = (A - Fq::from(2u64)) / B;

    let k = Fq::from_str("14127858815617975033680055377622475342429738925160381380815955502653114777569241314884161457101288630590399651847").unwrap();

    let a_def = if TypeId::of::<L>() == TypeId::of::<Limb32>() {
        let a_vec = limbs_of_u32(&a).to_vec();
        let limbs_of_a = unsafe { std::mem::transmute::<_, Vec<Limb32>>(a_vec) };
        const_field("FIELD_ED_A", limbs_of_a)
    } else {
        let a_vec = limbs_of_u64(&a).to_vec();
        let limbs_of_a = unsafe { std::mem::transmute::<_, Vec<Limb64>>(a_vec) };
        const_field("FIELD_ED_A", limbs_of_a)
    };

    let d_def = if TypeId::of::<L>() == TypeId::of::<Limb32>() {
        let d_vec = limbs_of_u32(&d).to_vec();
        let limbs_of_d = unsafe { std::mem::transmute::<_, Vec<Limb32>>(d_vec) };
        const_field("FIELD_ED_D", limbs_of_d)
    } else {
        let d_vec = limbs_of_u64(&d).to_vec();
        let limbs_of_d = unsafe { std::mem::transmute::<_, Vec<Limb64>>(d_vec) };
        const_field("FIELD_ED_D", limbs_of_d)
    };

    let k_def = if TypeId::of::<L>() == TypeId::of::<Limb32>() {
        let k_vec = limbs_of_u32(&k).to_vec();
        let limbs_of_k = unsafe { std::mem::transmute::<_, Vec<Limb32>>(k_vec) };
        const_field("FIELD_ED_K", limbs_of_k)
    } else {
        let k_vec = limbs_of_u64(&k).to_vec();
        let limbs_of_k = unsafe { std::mem::transmute::<_, Vec<Limb64>>(k_vec) };
        const_field("FIELD_ED_K", limbs_of_k)
    };

    let mut defs = [a_def, d_def, k_def].join("\n");

    String::from(defs).replace("FIELD", field)
}

fn ec(field: &str, point: &str) -> String {
    String::from(EC_SRC)
        .replace("FIELD", field)
        .replace("POINT", point)
}

fn field2(field2: &str, field: &str) -> String {
    String::from(FIELD2_SRC)
        .replace("FIELD2", field2)
        .replace("FIELD", field)
}

fn fft(field: &str) -> String {
    String::from(FFT_SRC).replace("FIELD", field)
}

fn poly(field: &str) -> String {
    return String::from(POLYNOMIAL_SRC).replace("FIELD", field);
}

fn multiexp(point: &str, exp: &str) -> String {
    String::from(MULTIEXP_SRC)
        .replace("POINT", point)
        .replace("EXPONENT", exp)
}

/// Trait to implement limbs of different underlying bit sizes.
pub trait Limb: Sized + Clone + Copy + 'static {
    /// The underlying size of the limb, e.g. `u32`
    type LimbType: Clone + std::fmt::Display;
    /// Returns the value representing zero.
    fn zero() -> Self;
    /// Returns a new limb.
    fn new(val: Self::LimbType) -> Self;
    /// Returns the raw value of the limb.
    fn value(&self) -> Self::LimbType;
    /// Returns the bit size of the limb.
    fn bits() -> usize {
        std::mem::size_of::<Self::LimbType>() * 8
    }
    /// Returns a tuple with the strings that PTX is using to describe the type and the register.
    fn ptx_info() -> (&'static str, &'static str);
    /// Returns the type that OpenCL is using to represent the limb.
    fn opencl_type() -> &'static str;
    /// Returns the field modulus in non-Montgomery form as a vector of `Self::LimbType` (least
    /// Returns the limbs that represent the multiplicative identity of the given field.
    fn one_limbs<F: Field>() -> Vec<Self>;
    /// Returns the field modulus in non-Montgomery form as a vector of `Self::LimbType` (least
    /// significant limb first).
    fn modulus_limbs<F: Field>() -> Vec<Self>;
    /// Calculate the `INV` parameter of Montgomery reduction algorithm for 32/64bit limbs
    /// * `a` - Is the first limb of modulus.
    fn calc_inv(a: Self) -> Self;
    /// Returns the limbs that represent `R ^ 2 mod P`.
    fn calculate_r2<F: Field>() -> Vec<Self>;
}

/// A 32-bit limb.
#[derive(Debug, Clone, Copy)]
pub struct Limb32(u32);
impl Limb for Limb32 {
    type LimbType = u32;
    fn zero() -> Self {
        Self(0)
    }
    fn new(val: Self::LimbType) -> Self {
        Self(val)
    }
    fn value(&self) -> Self::LimbType {
        self.0
    }
    fn ptx_info() -> (&'static str, &'static str) {
        ("u32", "r")
    }
    fn opencl_type() -> &'static str {
        "uint"
    }
    fn one_limbs<F: Field>() -> Vec<Self> {
        let one = F::one();

        let limbs_of_one = limbs_of_u32(&one).to_vec();
        let limbs_of_one = unsafe { std::mem::transmute::<_, Vec<Self>>(limbs_of_one) };

        debug!("u32 one = {:x?}", limbs_of_one);
        limbs_of_one
    }
    fn modulus_limbs<F: Field>() -> Vec<Self> {
        // TOOD: is this correct?
        let modulus =
            <<<F as ark_ff::Field>::BasePrimeField as PrimeField>::Params>::MODULUS.clone();

        let limbs_of_modulus = limbs_of_u32(&modulus).to_vec();
        let limbs_of_modulus = unsafe { std::mem::transmute::<_, Vec<Self>>(limbs_of_modulus) };

        debug!("u32 modulus = {:x?}", limbs_of_modulus);
        limbs_of_modulus
    }
    fn calc_inv(a: Self) -> Self {
        let mut inv = 1u32;
        for _ in 0..31 {
            inv = inv.wrapping_mul(inv);
            inv = inv.wrapping_mul(a.value());
        }
        Self(inv.wrapping_neg())
    }

    fn calculate_r2<F: Field>() -> Vec<Self> {
        let r2 = <<<F as ark_ff::Field>::BasePrimeField as PrimeField>::Params>::R2.clone();

        let limbs_of_r2 = limbs_of_u32(&r2).to_vec();
        let limbs_of_r2 = unsafe { std::mem::transmute::<_, Vec<Self>>(limbs_of_r2) };

        debug!("u32 r2 = {:x?}", limbs_of_r2);
        limbs_of_r2
    }
}

/// A 64-bit limb.
#[derive(Debug, Clone, Copy)]
pub struct Limb64(u64);
impl Limb for Limb64 {
    type LimbType = u64;
    fn zero() -> Self {
        Self(0)
    }
    fn new(val: Self::LimbType) -> Self {
        Self(val)
    }
    fn value(&self) -> Self::LimbType {
        self.0
    }
    fn ptx_info() -> (&'static str, &'static str) {
        ("u64", "l")
    }
    fn opencl_type() -> &'static str {
        "ulong"
    }
    fn one_limbs<F: Field>() -> Vec<Self> {
        let one = F::one();

        let limbs_of_one = limbs_of_u64(&one).to_vec();
        let limbs_of_one = unsafe { std::mem::transmute::<_, Vec<Self>>(limbs_of_one) };

        debug!("u64 one = {:x?}", limbs_of_one);
        limbs_of_one
    }

    fn modulus_limbs<F: Field>() -> Vec<Self> {
        // TOOD: is this correct?
        let modulus =
            <<<F as ark_ff::Field>::BasePrimeField as PrimeField>::Params>::MODULUS.clone();

        let limbs_of_modulus = limbs_of_u64(&modulus).to_vec();
        let limbs_of_modulus = unsafe { std::mem::transmute::<_, Vec<Self>>(limbs_of_modulus) };

        debug!("u64 modulus = {:x?}", limbs_of_modulus);
        limbs_of_modulus
    }

    fn calc_inv(a: Self) -> Self {
        let mut inv = 1u64;
        for _ in 0..63 {
            inv = inv.wrapping_mul(inv);
            inv = inv.wrapping_mul(a.value());
        }
        Self(inv.wrapping_neg())
    }
    fn calculate_r2<F: Field>() -> Vec<Self> {
        let r2 = <<<F as ark_ff::Field>::BasePrimeField as PrimeField>::Params>::R2.clone();

        let limbs_of_r2 = limbs_of_u64(&r2).to_vec();
        let limbs_of_r2 = unsafe { std::mem::transmute::<_, Vec<Self>>(limbs_of_r2) };

        debug!("u64 r2 = {:x?}", limbs_of_r2);
        limbs_of_r2
    }
}

fn const_field<L: Limb>(name: &str, limbs: Vec<L>) -> String {
    format!(
        "CONSTANT FIELD {} = {{ {{ {} }} }};",
        name,
        limbs
            .iter()
            .map(|l| l.value().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Generates CUDA/OpenCL constants and type definitions of prime-field `F`
fn params<F, L: Limb>() -> String
where
    F: Field,
{
    let one = L::one_limbs::<F>(); // Get Montgomery form of F::one()
    let p = L::modulus_limbs::<F>(); // Get field modulus in non-Montgomery form
    let r2 = L::calculate_r2::<F>();
    let limbs = one.len(); // Number of limbs
    let inv = L::calc_inv(p[0]);
    let limb_def = format!("#define FIELD_limb {}", L::opencl_type());
    let limbs_def = format!("#define FIELD_LIMBS {}", limbs);
    let limb_bits_def = format!("#define FIELD_LIMB_BITS {}", L::bits());
    let p_def = const_field("FIELD_P", p);
    let r2_def = const_field("FIELD_R2", r2);
    let one_def = const_field("FIELD_ONE", one);
    let zero_def = const_field("FIELD_ZERO", vec![L::zero(); limbs]);
    let inv_def = format!("#define FIELD_INV {}", inv.value());
    let typedef = "typedef struct { FIELD_limb val[FIELD_LIMBS]; } FIELD;".to_string();
    [
        limb_def,
        limbs_def,
        limb_bits_def,
        inv_def,
        typedef,
        one_def,
        p_def,
        r2_def,
        zero_def,
    ]
    .join("\n")
}

/// Generates PTX-Assembly implementation of FIELD_add_/FIELD_sub_
fn field_add_sub_nvidia<F, L: Limb>() -> Result<String, std::fmt::Error>
where
    F: Field,
{
    let mut result = String::new();
    let (ptx_type, ptx_reg) = L::ptx_info();

    writeln!(result, "#if defined(OPENCL_NVIDIA) || defined(CUDA) || defined(SN_CUDA)\n")?;
    for op in &["sub", "add"] {
        let len = L::one_limbs::<F>().len();

        writeln!(
            result,
            "DEVICE FIELD FIELD_{}_nvidia(FIELD a, FIELD b) {{",
            op
        )?;
        if len > 1 {
            write!(result, "asm(")?;
            writeln!(result, "\"{}.cc.{} %0, %0, %{};\\r\\n\"", op, ptx_type, len)?;

            for i in 1..len - 1 {
                writeln!(
                    result,
                    "\"{}c.cc.{} %{}, %{}, %{};\\r\\n\"",
                    op,
                    ptx_type,
                    i,
                    i,
                    len + i
                )?;
            }
            writeln!(
                result,
                "\"{}c.{} %{}, %{}, %{};\\r\\n\"",
                op,
                ptx_type,
                len - 1,
                len - 1,
                2 * len - 1
            )?;

            write!(result, ":")?;
            for n in 0..len {
                write!(result, "\"+{}\"(a.val[{}])", ptx_reg, n)?;
                if n != len - 1 {
                    write!(result, ", ")?;
                }
            }

            write!(result, "\n:")?;
            for n in 0..len {
                write!(result, "\"{}\"(b.val[{}])", ptx_reg, n)?;
                if n != len - 1 {
                    write!(result, ", ")?;
                }
            }
            writeln!(result, ");")?;
        }
        writeln!(result, "return a;\n}}")?;
    }
    writeln!(result, "#endif")?;

    Ok(result)
}

/// Returns CUDA/OpenCL source-code of a ff::PrimeField with name `name`
/// Find details in README.md
///
/// The code from the [`common()`] call needs to be included before this on is used.
pub fn field<F, L: Limb>(name: &str) -> String
where
    F: Field,
{
    [
        params::<F, L>(),
        field_add_sub_nvidia::<F, L>().expect("preallocated"),
        String::from(FIELD_SRC),
    ]
    .join("\n")
    .replace("FIELD", name)
}

/// Returns CUDA/OpenCL source-code that contains definitions/functions that are shared across
/// fields.
///
/// It needs to be called before any other function like [`field`] or [`gen_ec_source`] is called,
/// as it contains deinitions, used in those.
pub fn common() -> String {
    COMMON_SRC.to_string()
}
