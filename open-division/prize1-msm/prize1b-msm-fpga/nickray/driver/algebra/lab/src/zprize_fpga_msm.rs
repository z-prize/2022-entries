// use ark_ff::PrimeField;
// use ark_ec::AffineCurve;
// use ark_bls12_377::{
//     // fr::Fr, G1Affine, G1PTEAffine, G1Projective as G1, G1TEAffine, G1TEProjective,
//     fr::Fr,
//     G1Affine,
//     G1Projective,
//     G1PTEAffine,
//     G1TEAffine,
//     G1TEProjective,
// };
// use super::msm::ema::msm_bigint_ema;

// pub struct MultiScalarMultContext {
//     input: Vec<G1Affine>,
//     bases: Vec<G1PTEAffine>,
// }    

// pub fn multi_scalar_mult_init<G: AffineCurve>(points: &[G]) -> MultiScalarMultContext {
//     let ppoints: Vec<G1PTEAffine> = points.iter().map(|p| p.into()).collect();
//     let context = MultiScalarMultContext {
//         input: points.to_vec(),
//         bases: ppoints,
//     };
//     context
// }

// pub fn multi_scalar_mult<G: AffineCurve>(
//     context: &mut MultiScalarMultContext,
//     points: &[G],
//     scalars: &[<G::ScalarField as PrimeField>::BigInt],
// ) -> G::Projective {
//     let res = msm_bigint_ema::<G1PTEAffine, G1TEProjective>(context.bases, scalars);
//     res.into()
// }
