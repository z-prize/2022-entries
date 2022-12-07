use core::mem::transmute;

use ark_ec::AffineCurve;
use ark_ff::PrimeField;
use std::os::raw::c_void;

use fpga::F1;
use msm_fpga::{timed, App, Scalar};

#[repr(C)]
pub struct MultiScalarMultContext {
    context: *mut c_void,
}

struct Context {
    app: App,
}

pub fn multi_scalar_mult_init<G: AffineCurve>(points: &[G]) -> MultiScalarMultContext {
    let size = ark_std::log2(points.len());
    assert_eq!(points.len(), 1 << size);
    assert!(size <= 26);

    let f1 = F1::new(0, 0x500).unwrap();
    let mut app = App::new(f1, size as u8);

    // get rid of the generic parameter, this is a specific implementation
    let points: &[our_bls12_377::G1Affine] = unsafe { transmute(points) };
    let points = msm_fpga::preprocess_points(points);
    app.set_preprocessed_points(&points);

    let context: &'static mut _ = Box::leak(Box::new(Context { app }));

    // return wrapper to Context
    MultiScalarMultContext {
        context: context as *mut Context as *mut std::ffi::c_void,
    }
}

pub fn multi_scalar_mult<G: AffineCurve>(
    context: &mut MultiScalarMultContext,
    points: &[G],
    scalars: &[<G::ScalarField as PrimeField>::BigInt],
) -> Vec<G::Projective> {
    let context: &mut Context = unsafe { &mut *(context.context as *mut Context) };
    let scalars: &[Scalar] = unsafe { transmute(scalars) };
    let len = context.app.len();

    debug_assert_eq!(len, points.len());
    debug_assert_eq!(scalars.len() % len, 0);

    let results: Vec<_> = scalars
        .chunks(len)
        .map(|scalars| timed("MSM", || context.app.msm(scalars)))
        .collect();

    unsafe { transmute(results) }
}
