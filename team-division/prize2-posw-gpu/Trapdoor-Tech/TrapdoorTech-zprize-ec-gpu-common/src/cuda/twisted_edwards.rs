use std::str::FromStr;
use ark_ff::Field;
use ark_std::{One, Zero};

use crate::Fq;
use crate::{GpuEdAffine, GpuEdProjective, GpuProjective, GpuAffine};

lazy_static::lazy_static! {
    pub static ref MONT_ALPHA: Fq = Fq::from_str("80949648264912719408558363140637477264845294720710499478137287262712535938301461879813459410946").unwrap();
    pub static ref MONT_BETA: Fq = Fq::from_str("207913745465435703873309001080708636764682407053260289242004673792544811711776497012639468972230205966814119707502").unwrap();

    // these two parameters define a twisted edwards curve: a * X^2 + Y^2 = 1 + d * X^2 * Y^2
    pub static ref ED_COEFF_A: Fq = Fq::from_str("157163064917902313978814213261261898218646390773518349738660969080500653509624033038447657619791437448628296189665").unwrap();
    pub static ref ED_COEFF_D: Fq = Fq::from_str("101501361095066780517536410023107951769097300825221174390295061910482811707540513312796446149590693954692781734188").unwrap();

    // to make calculations even faster, we are actually manipulating on the curve: -X^2 + Y^2 = 1 + (-d / a) * X^2 * Y^2
    // thus, renaming (-d / a) to `dd`, we get another curve: -X^2 + Y^2 = 1 + dd * X^2 * Y^2
    // we need `k = 2 * dd` in unified addition, so we save it here:
    pub static ref ED_COEFF_DD: Fq = Fq::from_str("136396142414293534522166394536258004439411625840037520960350109084686791562955032044926524798337324377515360555012").unwrap();
    pub static ref ED_COEFF_K: Fq = Fq::from_str("14127858815617975033680055377622475342429738925160381380815955502653114777569241314884161457101288630590399651847").unwrap();

    // in order to do coordinates transform, we need `sqrt(-a)`
    pub static ref ED_COEFF_SQRT_NEG_A: Fq = Fq::from_str("237258690121739794091542072758217926613126300728951001700615245829450947395696022962309165363059235018940120114447").unwrap();
    pub static ref ED_COEFF_SQRT_NEG_A_INV: Fq = Fq::from_str("85493388116597753391764605746615521878764370024930535315959456146985744891605502660739892967955718798310698221510").unwrap();

    pub static ref FQ_TWO: Fq = Fq::from(2u64);
}


#[inline]
fn get_alpha_beta() -> (Fq, Fq) {
    (*MONT_ALPHA, *MONT_BETA)
}

#[inline]
fn get_a_d() -> (Fq, Fq) {
    (*ED_COEFF_A, *ED_COEFF_D)
}

#[inline]
fn get_dd_k() -> (Fq, Fq) {
    (*ED_COEFF_DD, *ED_COEFF_K)
}

#[inline]
fn get_sqrt_neg_a() -> (Fq, Fq) {
    (*ED_COEFF_SQRT_NEG_A, *ED_COEFF_SQRT_NEG_A_INV)
}

/// we don't introduce new Rust struct here, instead we reuse `GpuEdAffine` to represent the new coordinates under `-x^2 + y^2 = 1 + dd * x^2 * y^2`
#[allow(unused)]
pub(crate) fn edwards_to_neg_one_a(ed: GpuEdAffine) -> GpuEdAffine {
    let (divisor, _) = get_sqrt_neg_a();

    let t = ed.x * divisor * ed.y;

    GpuEdAffine {
        x: ed.x * divisor,
        y: ed.y,
        t,
    }
}

/// we don't introduce new Rust struct here, instead we reuse `GpuEdAffine` to represent the new coordinates under `-x^2 + y^2 = 1 + dd * x^2 * y^2`
#[allow(unused)]
pub(crate) fn edwards_from_neg_one_a(ed: GpuEdAffine) -> GpuEdAffine {
    let (_, multiplier) = get_sqrt_neg_a();

    let t = ed.x * multiplier * ed.y;

    GpuEdAffine {
        x: ed.x * multiplier,
        y: ed.y,
        t,
    }
}

#[allow(unused)]
pub(crate) fn sw_to_edwards(g: GpuAffine) -> GpuEdAffine {
    let (alpha, beta) = get_alpha_beta();

    // first convert sw to montgomery form
    let mont_x = (g.x - alpha) / beta;
    let mont_y = g.y / beta;

    // then from mont to edwards form
    let one = Fq::one();

    // map sw curve infinity point to te curve inf point
    if mont_y.is_zero() || (mont_x + one).is_zero() {
        return GpuEdAffine::default();
    }
    
    let ed_x = mont_x / mont_y;
    let ed_y = (mont_x - one) / (mont_x + one);
    let ed_t = ed_x * ed_y;

    GpuEdAffine {
        x: ed_x,
        y: ed_y,
        t: ed_t,
    }
}

#[allow(unused)]
pub(crate) fn edwards_to_sw(ed: GpuEdAffine) -> GpuAffine {
    let (alpha, beta) = get_alpha_beta();

    // first convert ed form to mont form
    let one = Fq::one();

    if (one - ed.y).is_zero() || ed.x.is_zero() {
        return GpuAffine::default();
    }
 
    let mont_x = (one + ed.y) / (one - ed.y);
    let mont_y = (one + ed.y) / (ed.x - ed.x * ed.y);

    // then from mont form to sw form
    let g_x = mont_x * beta + alpha;
    let g_y = mont_y * beta;

    GpuAffine { x: g_x, y: g_y }
}

#[allow(unused)]
pub(crate) fn edwards_to_sw_proj(ed: GpuEdAffine) -> GpuProjective {
    let (alpha, beta) = get_alpha_beta();

    // first convert ed form to mont form
    let one = Fq::one();

    if (one - ed.y).is_zero() || ed.x.is_zero() {
        return GpuProjective::default();
    }
 
    let mont_x = (one + ed.y) / (one - ed.y);
    let mont_y = (one + ed.y) / (ed.x - ed.x * ed.y);

    // then from mont form to sw form
    let g_x = mont_x * beta + alpha;
    let g_y = mont_y * beta;

    GpuProjective {
        x: g_x,
        y: g_y,
        z: Fq::one(),
    }
}

#[allow(unused)]
pub(crate) fn edwards_affine_to_proj(ed: GpuEdAffine) -> GpuEdProjective {
    GpuEdProjective {
        x: ed.x,
        y: ed.y,
        t: ed.x * ed.y,
        z: Fq::one(),
    }
}

#[allow(unused)]
pub(crate) fn edwards_proj_to_affine(ed: GpuEdProjective) -> GpuEdAffine {
    if ed.z.is_zero() {
        return GpuEdAffine::default();
    }

    let x = ed.x / ed.z;
    let y = ed.y / ed.z;

    GpuEdAffine { x, y, t: x * y }
}

// for a =/= -1
// http://www.hyperelliptic.org/EFD/g1p/auto-twisted-extended.html#addition-add-2008-hwcd
#[allow(non_snake_case, unused)]
pub(crate) fn edwards_add(ed1: GpuEdProjective, ed2: GpuEdProjective) -> GpuEdProjective {
    let (a, d) = get_a_d();

    // doing add arithmetic
    let A = ed1.x * ed2.x;
    let B = ed1.y * ed2.y;
    let C = d * ed1.t * ed2.t;
    let D = ed1.z * ed2.z;

    let E = (ed1.x + ed1.y) * (ed2.x + ed2.y) - A - B;
    let F = D - C;
    let G = D + C;
    let H = B - a * A;

    GpuEdProjective {
        x: E * F,
        y: G * H,
        t: E * H,
        z: F * G,
    }
}

// for a =/= -1
// http://www.hyperelliptic.org/EFD/g1p/auto-twisted-extended.html#doubling-dbl-2008-hwcd
#[allow(non_snake_case, unused)]
pub(crate) fn edwards_double(ed: GpuEdProjective) -> GpuEdProjective {
    let (a, _) = get_a_d();

    let A = ed.x.square();
    let B = ed.y.square();
    let C = (*FQ_TWO) * ed.z * ed.z;
    let D = a * A;
    let E = (ed.x + ed.y) * (ed.x + ed.y) - A - B;
    let G = D + B;
    let F = G - C;
    let H = D - B;

    GpuEdProjective {
        x: E * F,
        y: G * H,
        t: E * H,
        z: F * G,
    }
}

// for a = -1
// http://www.hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html#addition-add-2008-hwcd-3
#[allow(non_snake_case, unused)]
pub(crate) fn edwards_add_with_neg_one_a(ed1: GpuEdProjective, ed2: GpuEdProjective) -> GpuEdProjective {
    let (_, k) = get_dd_k();

    // doing add arithmetic
    let A = (ed1.y - ed1.x) * (ed2.y - ed2.x);
    let B = (ed1.y + ed1.x) * (ed2.y + ed2.x);
    let C = k * ed1.t * ed2.t;
    let D = (*FQ_TWO) * ed1.z * ed2.z;

    let E = B - A;
    let F = D - C;
    let G = D + C;
    let H = B + A;

    GpuEdProjective {
        x: E * F,
        y: G * H,
        t: E * H,
        z: F * G,
    }
}

// for a = -1
// http://www.hyperelliptic.org/EFD/g1p/auto-twisted-extended.html#doubling-dbl-2008-hwcd
#[allow(non_snake_case, unused)]
pub(crate) fn edwards_double_with_neg_one_a(ed: GpuEdProjective) -> GpuEdProjective {
    // doing add arithmetic
    let A = ed.x.square();
    let B = ed.y.square();
    let C = (*FQ_TWO) * ed.z.square();
    let D = -A;
    let E = (ed.x + ed.y).square() - A - B;
    let G = D + B;
    let F = G - C;
    let H = D - B;

    GpuEdProjective {
        x: E * F,
        y: G * H,
        t: E * H,
        z: F * G,
    }
}

