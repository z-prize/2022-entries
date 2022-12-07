use ark_std::ops::{AddAssign, MulAssign, SubAssign};
use lab::*;

use ark_bls12_377::{fq::Fq, fr::Fr, G1Affine, G1Projective as G1};
use ark_ec::ProjectiveCurve;
use ark_ff::{
    biginteger::{BigInteger256 as FrRepr, BigInteger384 as FqRepr},
    BigInteger, Field, PrimeField, SquareRootField, UniformRand,
};

mod g1 {
    use super::*;
    ec_bench!(G1, G1Affine);
}

f_bench!(Fq, Fq, FqRepr, FqRepr, fq);
f_bench!(Fr, Fr, FrRepr, FrRepr, fr);

bencher::benchmark_main!(fq, fr, g1::group_ops);
