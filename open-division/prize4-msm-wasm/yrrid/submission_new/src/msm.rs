use std::future::Future;
use std::io::Cursor;
use std::panic;
use ark_bls12_381::{G1Affine, fr::Fr, G1Projective, Fq};
use ark_ec::AffineRepr;
use ark_ff::{BigInt, BigInteger, BigInteger256, BigInteger384, Fp384, FpConfig, MontBackend, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use futures::FutureExt;
use wasm_bindgen::{JsCast, JsError, JsValue};
use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen_futures::{ JsFuture};
use js_sys::{Array, Error, Function, Object, Reflect, Uint8Array, WebAssembly::{
    instantiate_buffer, Instance, Memory
}};
use js_sys::WebAssembly::{instantiate_module, Module};

use wasm_bindgen_test::console_log;


const WASM: &[u8] = include_bytes!("./submission.wasm");
static mut WASM_INSTANCE: Option<Instance> = None;

#[wasm_bindgen]
pub async fn init_fast_msm_wasm() {
    unsafe {
        let a: JsValue = JsFuture::from(instantiate_buffer(WASM, &Object::new())).await.unwrap();
        WASM_INSTANCE = Some(Reflect::get(&a, &"instance".into()).unwrap().dyn_into().unwrap());
    }
}

use web_sys::console;

pub(crate) unsafe fn compute_msm(
    point_vec: Vec<G1Projective>,
    scalar_vec: Vec<Fr>,
) -> Result<G1Projective, Error>
{

    if WASM_INSTANCE.is_none() {
        panic!("wasm not initialized");
    }

    let c = WASM_INSTANCE.clone().unwrap().exports();

    let size = scalar_vec.len();
    let window_bits = if size > 128*1024 {
        16
    } else if size > 96*1024 {
        15
    } else {
        13
    };

    macro_rules! load_wasm_func{
                ($a:expr, $b:ty)=>{
                    {
                        Reflect::get(c.as_ref(), &$a.into())
                        .unwrap()
                        .dyn_into::<$b>()
                        .expect("$a export wasn't a function")
                    }
                }
            }

    let msm_initialize = load_wasm_func!("msmInitialize", Function);
    let msm_scalars_offset = load_wasm_func!("msmScalarsOffset", Function);
    let msm_points_offset = load_wasm_func!("msmPointsOffset", Function);
    let msm_run = load_wasm_func!("msmRun", Function);

    let args = Array::new_with_length(4);
    args.set(0, size.into());
    args.set(1, window_bits.into());
    args.set(2, 1024.into());
    args.set(3, 128.into());
    msm_initialize.apply(&JsValue::undefined(), &args)?;

    let mem: Memory = load_wasm_func!("memory", Memory);
    let buffer = &mem.buffer();

    let scalar_offset: JsValue = msm_scalars_offset.call0(&JsValue::undefined())?;
    let scalar_mem: Uint8Array = Uint8Array::new_with_byte_offset_and_length(
        &buffer,
        scalar_offset.as_f64().unwrap() as u32,
        size as u32 * 32
    );

    let mut ptr: u32 = 0;
    for scalar in scalar_vec.into_iter() {
        for s in scalar.into_bigint().to_bytes_le() {
            Uint8Array::set_index(&scalar_mem, ptr, s);
            ptr += 1;
        }
    }

    let point_offset:JsValue = msm_points_offset.call0(&JsValue::undefined())?;
    let point_mem: Uint8Array = Uint8Array::new_with_byte_offset_and_length(
        &buffer,
        point_offset.as_f64().unwrap() as u32,
        size as u32 * 96,
    );

    let mut ptr: u32 = 0;
    for point in point_vec.into_iter() {
        let affine = G1Affine::from(point);
        for s in affine.x.into_bigint().to_bytes_le() {
            Uint8Array::set_index(&point_mem, ptr, s);
            ptr += 1;
        }
        for s in affine.y.into_bigint().to_bytes_le() {
            Uint8Array::set_index(&point_mem, ptr, s);
            ptr += 1;
        }
    }

    let result = msm_run.call0(&JsValue::undefined());

    let result_offset: JsValue = result.unwrap();
    let result_mem: Uint8Array = Uint8Array::new_with_byte_offset_and_length(
        &buffer,
        result_offset.as_f64().unwrap() as u32,
        96
    );
    let a1 = result_mem.to_vec()[0..48].to_vec();
    let a2 = result_mem.to_vec()[48..96].to_vec();

    let affine = G1Affine::new(fq_from_bytes(a1), fq_from_bytes(a2));
    Ok(G1Projective::from(affine))
}


fn fq_from_bytes(bytes: Vec<u8>) -> Fq {
    let buffer = Cursor::new(bytes.clone());
    let b = BigInteger384::deserialize_uncompressed(buffer).unwrap();
    let fq = MontBackend::from_bigint(b).unwrap();
    fq
}

#[test]
fn test() {
    let size = 1 << 14;
    let (point_vec, scalar_vec) = generate_msm_inputs::<G1Affine>(size);
    let _ = compute_msm::<G1Affine>(&point_vec, &scalar_vec);
}
