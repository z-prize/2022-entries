use std::io::Cursor;
use ark_bls12_381::{Fq, FqConfig, Fr, G1Affine, G1Projective};
use ark_ff::{BigInt, BigInteger, BigInteger256, BigInteger384, Fp384, FpConfig, MontBackend, PrimeField};
use js_sys::{Array, Promise, Uint8Array};
use wasm_bindgen::prelude::wasm_bindgen;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use futures::{AsyncReadExt, StreamExt};
use num_bigint::BigUint;
use wasm_bindgen::JsValue;
use wasm_bindgen_test::console_log;

pub mod msm;
//
// #[wasm_bindgen]
// pub struct PointVectorInput {
//     point_vec: Vec<<<G1Affine as AffineCurve>::Projective as ProjectiveCurve>::Affine>,
// }
//
// #[wasm_bindgen]
// impl PointVectorInput {
//     #[wasm_bindgen(constructor)]
//     pub fn new(size: usize) -> Self {
//         Self {
//             point_vec: msm::generate_msm_inputs::<G1Affine>(size).0,
//         }
//     }
//
//     #[wasm_bindgen(js_name = "toJsArray")]
//     pub fn to_js_array(&self) -> Array {
//         let arr = Array::new_with_length(self.point_vec.len() as u32);
//         for (i, point) in (&self.point_vec).into_iter().enumerate() {
//             let x = point.x;
//             let y = point.y;
//             let is_infinity = point.infinity;
//
//             let mut x_bytes: Vec<u8> = Vec::with_capacity(48);
//             x.write(&mut x_bytes).unwrap();
//             let mut y_bytes: Vec<u8> = Vec::with_capacity(48);
//             y.write(&mut y_bytes).unwrap();
//
//             let point = Array::new_with_length(3);
//             point.set(0, Uint8Array::from(x_bytes.as_slice()).into());
//             point.set(1, Uint8Array::from(y_bytes.as_slice()).into());
//             point.set(2, is_infinity.into());
//             arr.set(i as u32, point.into());
//         }
//         arr
//     }
// }
//
// #[wasm_bindgen]
// pub struct ScalarVectorInput {
//     scalar_vec: Vec<<<G1Affine as AffineCurve>::ScalarField as PrimeField>::BigInt>,
// }
//
// #[wasm_bindgen]
// impl ScalarVectorInput {
//     #[wasm_bindgen(constructor)]
//     pub fn new(size: usize) -> Self {
//         Self {
//             scalar_vec: msm::generate_msm_inputs::<G1Affine>(size).1,
//         }
//     }
//
//     #[wasm_bindgen(js_name = "toJsArray")]
//     pub fn to_js_array(&self) -> Array {
//         let arr = Array::new_with_length(self.scalar_vec.len() as u32);
//         for (i, scalar) in (&self.scalar_vec).into_iter().enumerate() {
//             let mut bytes: Vec<u8> = Vec::with_capacity(32);
//             scalar.write(&mut bytes).unwrap();
//             arr.set(i as u32, Uint8Array::from(bytes.as_slice()).into());
//         }
//         arr
//     }
//
// }

#[wasm_bindgen]
pub struct PointOutput {
    point: G1Projective,
}

#[wasm_bindgen]
impl PointOutput {
    #[wasm_bindgen(js_name = "toJsArray")]
    pub fn to_js_array(&self) -> Array {

        let mut x_bytes: Vec<u8> = vec![0u8; 48];
        let mut y_bytes: Vec<u8> = vec![0u8; 48];

        let affine = G1Affine::from(self.point);
        for (i, s) in affine.x.into_bigint().to_bytes_le().into_iter().enumerate() {
            x_bytes[i] = s;
        }
        for (i, s) in affine.y.into_bigint().to_bytes_le().into_iter().enumerate() {
            y_bytes[i] = s;
        }

        let point = Array::new_with_length(3);
        point.set(0, Uint8Array::from(x_bytes.as_slice()).into());
        point.set(1, Uint8Array::from(y_bytes.as_slice()).into());
        point.set(2, false.into());
        point
    }
}

#[wasm_bindgen]
pub unsafe fn compute_msm(
    point_vec: Array,
    scalar_vec: Array,
) -> PointOutput {

    let mut scalar: Vec<Fr> = Default::default();
    let mut x = |e, _, _| {
        let array = Uint8Array::new(&e);
        let bytes = array.to_vec();
        let c = Cursor::new(bytes);

        //console_log!("{:?}", c.clone().get_ref());
        let bigint: BigInt<4> = BigInt::deserialize_uncompressed(c).unwrap();
        let fr = ark_ff::Fp256::from_bigint(bigint).unwrap();
        scalar.push(fr);
    };
    scalar_vec.for_each(&mut x);

    let mut point: Vec<G1Projective> = Vec::new();
    let mut a = |e, _, _| {
        let p = Array::from(&e);
        let a1 = Uint8Array::new(&p.at(0));
        let a2 = Uint8Array::new(&p.at(1));

        let affine = G1Affine::new(fq_from_bytes(a1.to_vec()), fq_from_bytes(a2.to_vec()));

        let projective = G1Projective::from(affine);
        point.push(projective);
    };
    point_vec.for_each(&mut a);

    use web_sys::console;

    let result =  msm::compute_msm(
        point,
        scalar
    ).unwrap();

    PointOutput{
        point: result
    }
}

fn fq_from_bytes(bytes: Vec<u8>) -> Fq {
    let buffer = Cursor::new(bytes.clone());
    let b = BigInteger384::deserialize_uncompressed(buffer).unwrap();
    let fq = MontBackend::from_bigint(b).unwrap();
    fq
}