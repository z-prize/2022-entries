use ark_bls12_377::{
    // fr::Fr, G1Affine, G1PTEAffine, G1Projective as G1, G1TEAffine, G1TEProjective,
    fr::Fr,
    G1Affine,
    G1PTEAffine,
    G1TEAffine,
    G1TEProjective,
};
// use ark_ec::AffineCurve as _; // mul
use ark_ec::ProjectiveCurve as _; // into_affine
use ark_ff::fields::PrimeField as _; // into_bigint
use ark_std::UniformRand as _; // ::rand

use lab::msm::{ema, ray};
use lab::zprize_fpga_msm;

// #[test]
// pub fn correctness_random() {
// const N: usize = 8;
// const POINTS: usize = 1 << N;
//
// let mut rng = ark_std::test_rng();
//
// let points: Vec<_> = (0..POINTS)
// .map(|_| G1::rand(&mut rng).into_affine())
// .collect();
//
// let scalars: Vec<_> = (0..POINTS)
// .map(|_| Fr::rand(&mut rng).into_bigint())
// .collect();
//
// let msm =
// <G1 as ark_ec::msm::VariableBaseMSM<G1Affine>>::msm_bigint(points.as_slice(),
// &scalars); let naive: G1Affine = points
// .iter()
// .zip(scalars.iter())
// .map(|(point, scalar)| point.mul(*scalar).into_affine())
// .sum();
//
// let ema = ema::msm_bigint_ema::<G1Affine, G1>(points.as_slice(), &scalars);
// let ray = ray::msm_bigint::<G1Affine, G1>(16, points.as_slice(), &scalars);
// let full = ray::full_msm::<G1Affine, G1>(16, points.as_slice(), &scalars);
//
// assert_eq!(msm, naive);
// assert_eq!(ema, msm);
// assert_eq!(ray, msm);
// assert_eq!(full, msm);
// }
//
// #[test]
// pub fn correctness() {
// const N: usize = 20;
// const POINTS: usize = 1 << N;
//
// let mut rng = ark_std::test_rng();
//
// let points: Vec<_> = (0..POINTS)
//     .map(|_| G1::rand(&mut rng).into_affine())
//     .collect();
//
// let point = G1::rand(&mut rng).into_affine();
//
// just uses same point
// let points: Vec<_> = (0..POINTS).map(|_| point).collect();
//
// scalars are random though
// let scalars: Vec<_> = (0..POINTS)
// .map(|_| Fr::rand(&mut rng).into_bigint())
// .collect();
//
// let msm =
// <G1 as ark_ec::msm::VariableBaseMSM<G1Affine>>::msm_bigint(points.as_slice(),
// &scalars); let naive: G1Affine = points
//     .iter()
//     .zip(scalars.iter())
//     .map(|(point, scalar)| point.mul(*scalar).into_affine())
//     .sum();
//
// let ema = ema::msm_bigint_ema::<G1Affine, G1>(points.as_slice(), &scalars);
// let ray = ray::msm_bigint::<G1Affine, G1>(16, points.as_slice(), &scalars);
// let full = ray::full_msm::<G1Affine, G1>(16, points.as_slice(), &scalars);
//
// assert_eq!(msm, naive);
// assert_eq!(ema, msm);
// assert_eq!(ray, naive);
// assert_eq!(full, naive);
// }

#[test]
pub fn pete_correctness_random() {
    const N: usize = 8;
    const POINTS: usize = 1 << N;

    let mut rng = ark_std::test_rng();

    let points: Vec<G1TEAffine> = (0..POINTS)
        .map(|_| G1TEProjective::rand(&mut rng).into_affine())
        .collect();

    let ppoints: Vec<G1PTEAffine> = points.iter().map(|p| p.into()).collect();

    // let scalars: Vec<_> = (0..POINTS)
    //     .map(|_| Fr::rand(&mut rng).into_bigint())
    //     .collect();
    let mut scalars: Vec<_> = (0..POINTS)
        // .map(|_| Fr::rand(&mut rng).into_bigint())
        // .map(|_| Fr::from(u8::rand(&mut rng)).into_bigint())
        // .map(|_| Fr::from(u16::rand(&mut rng)).into_bigint())
        .map(|i| Fr::from(0 as u16).into_bigint())
        .collect();
    scalars[4] = Fr::from(1).into_bigint();
    scalars[200] = Fr::from(2).into_bigint();
    // scalars[4] = Fr::from(3).into_bigint();

    let msm = <G1TEProjective as ark_ec::msm::VariableBaseMSM<G1TEAffine>>::msm_bigint(
        points.as_slice(),
        &scalars,
    );
    // let naive: G1TEAffine = points
    //     .iter()
    //     .zip(scalars.iter())
    //     .map(|(point, scalar)| point.mul(*scalar).into_affine())
    //     .sum();

    let ema = ema::msm_bigint_ema::<G1PTEAffine, G1TEProjective>(ppoints.as_slice(), &scalars);
    let ray = ray::msm_bigint::<G1PTEAffine, G1TEProjective>(17, ppoints.as_slice(), &scalars);

    let hardcoded = points[4] + points[200] + points[200];
    // let mut context = ema::multi_scalar_mult_init(points.as_slice());
    // let msm_results = ema::multi_scalar_mult(&mut context, points.as_slice(), scalars.as_slice());

    // assert_eq!(ema, naive);
    assert_eq!(ema, msm);
    assert_eq!(ray, msm);
    assert_eq!(hardcoded, msm);
    // assert_eq!(msm_results[0], msm);
    
    // println!{"res = {}", msm}
    // println!{"ema = {}", ema}
}
