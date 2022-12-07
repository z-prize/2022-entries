/// There are a few kinds of rust ECC crypto libraries such as `zexe` and `bellman`
/// To independently define ECC related structs, regardless of backend, we need to define a common trait
/// The only constrain we make is that all field elements can be converted into/from slice `[u64]`
pub use ark_bls12_377::{Fq, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
pub use ark_ec::{
    models::short_weierstrass_jacobian::GroupAffine,
    models::short_weierstrass_jacobian::GroupProjective, AffineCurve, PairingEngine,
    ProjectiveCurve,
};
pub use ark_ff::{fields::Fp256, fields::Fp384, Field, FromBytes, One, PrimeField, ToBytes, Zero};
pub use ark_poly::{domain::EvaluationDomain, domain::GeneralEvaluationDomain};

use ark_std::io::{Read, Write};

/// TODO: figure out a way to define suitable GPU structs for different curves
/// A possible solution is to only convert from/into slice `[u8]`
pub const FR_LIMBS: usize = 4;
pub const FQ_LIMBS: usize = 6;

pub type GpuFr = [u64; FR_LIMBS];
pub type GpuFq = [u64; FQ_LIMBS];

pub const GPU_FR_ZERO: GpuFr = [0u64; FR_LIMBS];
pub const GPU_FQ_ZERO: GpuFq = [0u64; FQ_LIMBS];

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct GpuAffine {
    pub x: Fq,
    pub y: Fq,
}

impl Default for GpuAffine {
    fn default() -> Self {
        let x = Fq::zero();
        let y = Fq::zero();
        Self { x, y }
    }
}

impl ToBytes for GpuAffine {
    #[inline]
    fn write<W: Write>(&self, mut writer: W) -> ark_std::io::Result<()> {
        self.x.write(&mut writer)?;
        self.y.write(&mut writer)
    }
}

impl FromBytes for GpuAffine {
    #[inline]
    fn read<R: Read>(mut reader: R) -> ark_std::io::Result<Self> {
        let x = Fq::read(&mut reader)?;
        let y = Fq::read(&mut reader)?;

        Ok(Self { x, y })
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct GpuProjective {
    pub x: Fq,
    pub y: Fq,
    pub z: Fq,
}

impl Default for GpuProjective {
    fn default() -> Self {
        let x = Fq::zero();
        let y = Fq::one();
        let z = Fq::zero();
        Self { x, y, z }
    }
}

/// for Edwards curves
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct GpuEdAffine {
    pub x: Fq,
    pub y: Fq,
    pub t: Fq,
}

impl Default for GpuEdAffine {
    fn default() -> Self {
        let x = Fq::zero();
        let y = Fq::one();
        let t = Fq::zero();
        Self { x, y, t }
    }
}

impl ToBytes for GpuEdAffine {
    #[inline]
    fn write<W: Write>(&self, mut writer: W) -> ark_std::io::Result<()> {
        self.x.write(&mut writer)?;
        self.y.write(&mut writer)?;
        self.t.write(&mut writer)
    }
}

impl FromBytes for GpuEdAffine {
    #[inline]
    fn read<R: Read>(mut reader: R) -> ark_std::io::Result<Self> {
        let x = Fq::read(&mut reader)?;
        let y = Fq::read(&mut reader)?;
        let t = Fq::read(&mut reader)?;

        Ok(Self { x, y, t })
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct GpuEdProjective {
    pub x: Fq,
    pub y: Fq,
    pub t: Fq,
    pub z: Fq,
}

impl Default for GpuEdProjective {
    fn default() -> Self {
        let x = Fq::zero();
        let y = Fq::one();
        let t = Fq::zero();
        let z = Fq::one();

        Self { x, y, t, z }
    }
}

/// Define all available polynomial operations
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum PolyArith {
    AddConstant,
    SubConstant,
    Scaling,
    AddAssign,
    SubAssign,
    MulAssign,
    AddAssignScaled,
    SubAssignScaled,
    GeneratePowers,
    DistributePowers,
    BitreverseEnumeration,
    BatchInversion,
    EvaluateAt,
    CalculateGP,
    CalculateGS,
    CalculateShiftedGP,
    SetupL0,
    SetFE,
    CopyBuffer,
    CopyBufferFromOffset,
    CopyBufferToOffset,
    AddAtOffset,
    Negate,
    Square,
    PowersLong,
    Shift,
    SetupVanishing,
    ShrinkDomain,
}

use PolyArith::*;

impl std::fmt::Display for PolyArith {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            AddConstant => f.write_str("add_constant"),
            SubConstant => f.write_str("sub_constant"),
            Scaling => f.write_str("scale"),
            AddAssign => f.write_str("add_assign"),
            SubAssign => f.write_str("sub_assign"),
            MulAssign => f.write_str("mul_assign"),
            AddAssignScaled => f.write_str("add_assign_scaled"),
            SubAssignScaled => f.write_str("sub_assign_scaled"),
            DistributePowers => f.write_str("distribute_powers"),
            GeneratePowers => f.write_str("generate_powers"),
            BitreverseEnumeration => f.write_str("reverse_bits"),
            BatchInversion => f.write_str("batch_inversion"),
            EvaluateAt => f.write_str("evaluate_at"),
            CalculateGP => f.write_str("grand_product"),
            CalculateGS => f.write_str("grand_sum"),
            CalculateShiftedGP => f.write_str("shifted_grand_product"),
            SetupL0 => f.write_str("setup_l0"),
            SetFE => f.write_str("set_fe"),
            CopyBuffer => f.write_str("copy_from_to"),
            CopyBufferFromOffset => f.write_str("copy_from_offset_to"),
            CopyBufferToOffset => f.write_str("copy_from_to_offset"),
            AddAtOffset => f.write_str("add_at_offset"),
            Negate => f.write_str("negate"),
            Square => f.write_str("square"),
            PowersLong => f.write_str("pow_long"),
            Shift => f.write_str("shift"),
            SetupVanishing => f.write_str("setup_vanishing"),
            ShrinkDomain => f.write_str("shrink_domain"),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum GPU_OP {
    LOAD_BASE,
    REUSE_G,
    REUSE_SHIFTED_G,
    REUSE_LAGRANGE_G,
    REUSE_SHIFTED_LAGRANGE_G,
    SETUP_G,
    SETUP_SHIFTED_G,
    SETUP_LAGRANGE_G,
    SETUP_SHIFTED_LAGRANGE_G,
    RESET,
}
