use ark_std::Zero;
use our_bls12_377::{
    G1Affine, G1PTEAffine, G1Projective, G1TEAffine, G1TEProjective, FQ_ONE, FQ_ZERO,
};
use our_bls12_377::{FQ_S, FQ_SQRT_MIN_A, FQ_S_INV};

/// convert affine Weierstrass to affine extended Twisted Edwards
pub fn into_preprocessed(p: &G1Affine) -> G1PTEAffine {
    if p.is_zero() {
        return G1PTEAffine::new(FQ_ZERO, FQ_ONE);
    }

    let xpo = p.x + FQ_ONE;
    let sxpo = xpo * FQ_S;
    let axpo = xpo * FQ_SQRT_MIN_A;
    let syxpo = sxpo * p.y;

    let x = (sxpo + FQ_ONE) * axpo;
    let y = syxpo - p.y;
    let z = syxpo + p.y;
    let z_inv = FQ_ONE / z;

    G1PTEAffine::new(x * z_inv, y * z_inv)
}

/// convert affine Weierstrass to affine extended Twisted Edwards in batch
pub fn into_preprocessed_batched(a: &[G1Affine], b: &mut [G1PTEAffine]) {
    debug_assert!(a.len() == b.len());

    let mut x = vec![FQ_ZERO; a.len()];
    let mut y = vec![FQ_ZERO; a.len()];
    let mut z = vec![FQ_ZERO; a.len()];

    for (i, p) in a.iter().enumerate() {
        if p.is_zero() {
            x[i] = FQ_ZERO;
            y[i] = FQ_ONE;
            z[i] = FQ_ONE;
            continue;
        }

        let xpo = p.x + FQ_ONE;
        let sxpo = xpo * FQ_S;
        let axpo = xpo * FQ_SQRT_MIN_A;
        let syxpo = sxpo * p.y;

        x[i] = (sxpo + FQ_ONE) * axpo;
        y[i] = syxpo - p.y;
        z[i] = syxpo + p.y;
    }
    our_ff::batch_inversion(&mut z);

    for i in 0..a.len() {
        b[i] = G1PTEAffine::new(x[i] * z[i], y[i] * z[i]);
    }
}

pub fn into_twisted(p: &G1Affine) -> G1TEAffine {
    if p.is_zero() {
        return G1TEAffine::new(FQ_ZERO, FQ_ONE);
    }

    let xpo = p.x + FQ_ONE;
    let sxpo = xpo * FQ_S;
    let axpo = xpo * FQ_SQRT_MIN_A;
    let syxpo = sxpo * p.y;

    let x = (sxpo + FQ_ONE) * axpo;
    let y = syxpo - p.y;
    let z = syxpo + p.y;
    let z_inv = FQ_ONE / z;

    G1TEAffine::new(x * z_inv, y * z_inv)
}

/// convert projective extended Twisted Edwards to projective Weierstrass
pub fn into_weierstrass(point: &G1TEProjective) -> G1Projective {
    if point.is_zero() {
        return G1Projective::zero();
    }

    let z_inv = FQ_ONE / point.z;
    // let check = p.x / point.z;
    let aff_x = point.x * z_inv;
    let aff_y = point.y * z_inv;

    let p = FQ_ONE + aff_y;
    let m = FQ_ONE - aff_y;
    let u = p / m;
    let v = u / aff_x;

    let x = (u * FQ_S_INV) - FQ_ONE;
    let y = v * FQ_S_INV * FQ_SQRT_MIN_A;

    G1Projective::new(x, y, FQ_ONE)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn conversions() {
        let size = 3;
        let points = crate::random_points(size);
        for point in points.iter() {
            let projective: G1Projective = (*point).into();
            let affine: G1Affine = projective.into();
            assert_eq!(point, &affine);
        }
    }
}
