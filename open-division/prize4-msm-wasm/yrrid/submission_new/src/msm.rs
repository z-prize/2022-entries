use std::future::Future;
use std::panic;
use ark_bls12_381::G1Affine;
use ark_ec::{AffineCurve, ProjectiveCurve};
use ark_ff::{BigInteger, PrimeField, UniformRand};
use ark_serialize::CanonicalSerialize;
use futures::FutureExt;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::{ JsFuture};
use js_sys::{Array, Error, Function, Object, Reflect, Uint8Array, WebAssembly::{
    instantiate_buffer, Instance, Memory
}};
use js_sys::WebAssembly::{instantiate_module, Module};

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



const WASM: &[u8] = include_bytes!("./submission.wasm");

use web_sys::console;

pub(crate) async fn compute_msm<A>(
    point_vec: &Vec<<A::Projective as ProjectiveCurve>::Affine>,
    scalar_vec: &Vec<<A::ScalarField as PrimeField>::BigInt>,
) -> Result<Vec<u8>, Error>
    where
        A:  AffineCurve,
{

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let wasm_instance: Instance = {
        let promise = instantiate_buffer(WASM, &Object::new());
        let a: JsFuture = JsFuture::from(promise);
        let msg = format!("inside compute_msm after JsFuture {:?}", a);

        let b = a.await;
        Reflect::get(&b.expect("REASON"), &"instance".into()).unwrap().dyn_into().unwrap()

    };

    let c = wasm_instance.exports();

    let size = scalar_vec.len();
    let window_bits = if size > 128*1024 {
        16
    } else if size > 96*1024 {
        15
    } else {
        13
    };

    let msmInitialize = Reflect::get(c.as_ref(), &"msmInitialize".into())?
        .dyn_into::<Function>()
        .expect("msmInitialize export wasn't a function");
    let msmScalarsOffset = Reflect::get(c.as_ref(), &"msmScalarsOffset".into())?
        .dyn_into::<Function>()
        .expect("msmScalarsOffset export wasn't a function");
    let msmPointsOffset = Reflect::get(c.as_ref(), &"msmPointsOffset".into())?
        .dyn_into::<Function>()
        .expect("msmPointsOffset export wasn't a function");
    let msm_run = Reflect::get(c.as_ref(), &"msmRun".into())?
        .dyn_into::<Function>()
        .expect("msmRun export wasn't a function");

    let args = Array::new_with_length(4);
    args.set(0, size.into());
    args.set(1, window_bits.into());
    args.set(2, 1024.into());
    args.set(3, 128.into());
    msmInitialize.apply(&JsValue::undefined(), &args)?;

    let mut mem: Memory = Reflect::get(c.as_ref(), &"memory".into())?
        .dyn_into::<Memory>()
        .expect("memory export wasn't a `WebAssembly.Memory`");

    let buffer = &mem.buffer();

    let scalar_offset: JsValue = msmScalarsOffset.call0(&JsValue::undefined())?;
    let scalar_mem: Uint8Array = Uint8Array::new_with_byte_offset_and_length(
        &buffer,
        scalar_offset.as_f64().unwrap() as u32,
        size as u32 * 32
    );

    let mut ptr: u32 = 0;
    for scalar in (scalar_vec).into_iter() {
        for s in scalar.to_bytes_le() {
            // let y = format!("{:?}", s.clone());
            // console::log_1(&y.into());
            Uint8Array::set_index(&scalar_mem, ptr, s);
            ptr += 1;
        }
    }

    let point_offset:JsValue = msmPointsOffset.call0(&JsValue::undefined())?;
    let point_mem: Uint8Array = Uint8Array::new_with_byte_offset_and_length(
        &buffer,
        point_offset.as_f64().unwrap() as u32,
        size as u32 * 96,
    );

    let mut ptr:u32 = 0;
    for point in point_vec.into_iter() {
        let mut point_buffer = Vec::new();
        let _ = point.serialize_uncompressed(&mut point_buffer);
        for s in point_buffer {

            Uint8Array::set_index(&point_mem, ptr, s);
            ptr += 1;
        }
    }

    let result = msm_run.call0(&JsValue::undefined());

    let result_offset: JsValue = result.unwrap();
    let buffer_new = &mem.buffer();
    let result_mem: Uint8Array = Uint8Array::new_with_byte_offset_and_length(
        &buffer_new,
        result_offset.as_f64().unwrap() as u32,
        96
    );

    Ok(result_mem.to_vec())

}

#[test]
fn test() {
    let size = 1 << 14;
    let (point_vec, scalar_vec) = generate_msm_inputs::<G1Affine>(size);
    let _ = compute_msm::<G1Affine>(&point_vec, &scalar_vec);
}
