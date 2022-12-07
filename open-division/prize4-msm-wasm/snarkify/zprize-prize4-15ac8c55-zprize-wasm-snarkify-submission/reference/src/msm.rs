use ark_bls12_381::G1Affine;
use ark_ec::{msm, AffineCurve, ProjectiveCurve};
use ark_ff::{PrimeField, UniformRand};

pub fn generate_msm_inputs<A>(
    size: usize,
) -> (
    Vec<<A::Projective as ProjectiveCurve>::Affine>,
    Vec<<A::ScalarField as PrimeField>::BigInt>,
)
where
    A: AffineCurve,
{
    let mut rng = ark_std::test_rng();
    let scalar_vec = (0..size)
        .map(|_| A::ScalarField::rand(&mut rng).into_repr())
        .collect();
    let point_vec = (0..size)
        .map(|_| A::Projective::rand(&mut rng))
        .collect::<Vec<_>>();
    (
        <A::Projective as ProjectiveCurve>::batch_normalization_into_affine(&point_vec),
        scalar_vec,
    )
}

pub fn compute_msm<A>(
    point_vec: &Vec<<A::Projective as ProjectiveCurve>::Affine>,
    scalar_vec: &Vec<<A::ScalarField as PrimeField>::BigInt>,
) -> A::Projective
where
    A: AffineCurve,
{
    msm::VariableBaseMSM::multi_scalar_mul(point_vec.as_slice(), scalar_vec.as_slice())
}

#[test]
fn test() {
    let size = 1 << 14;
    let (point_vec, scalar_vec) = generate_msm_inputs::<G1Affine>(size);
    let _ = compute_msm::<G1Affine>(&point_vec, &scalar_vec);
}
