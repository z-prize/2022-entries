use std::io::Cursor;
use ark_bls12_381::G1Affine;
use ark_ec::{AffineCurve, ProjectiveCurve};
use ark_ff::{BigInteger256, FromBytes, PrimeField, ToBytes};
use js_sys::{Array, Promise, Uint8Array};
use wasm_bindgen::prelude::wasm_bindgen;
use ark_serialize::CanonicalDeserialize;
use wasm_bindgen::JsValue;

pub mod msm;

#[wasm_bindgen]
pub struct PointVectorInput {
    point_vec: Vec<<<G1Affine as AffineCurve>::Projective as ProjectiveCurve>::Affine>,
}

#[wasm_bindgen]
impl PointVectorInput {
    #[wasm_bindgen(constructor)]
    pub fn new(size: usize) -> Self {
        Self {
            point_vec: msm::generate_msm_inputs::<G1Affine>(size).0,
        }
    }

    #[wasm_bindgen(js_name = "toJsArray")]
    pub fn to_js_array(&self) -> Array {
        let arr = Array::new_with_length(self.point_vec.len() as u32);
        for (i, point) in (&self.point_vec).into_iter().enumerate() {
            let x = point.x;
            let y = point.y;
            let is_infinity = point.infinity;

            let mut x_bytes: Vec<u8> = Vec::with_capacity(48);
            x.write(&mut x_bytes).unwrap();
            let mut y_bytes: Vec<u8> = Vec::with_capacity(48);
            y.write(&mut y_bytes).unwrap();

            let point = Array::new_with_length(3);
            point.set(0, Uint8Array::from(x_bytes.as_slice()).into());
            point.set(1, Uint8Array::from(y_bytes.as_slice()).into());
            point.set(2, is_infinity.into());
            arr.set(i as u32, point.into());
        }
        arr
    }
}

#[wasm_bindgen]
pub struct ScalarVectorInput {
    scalar_vec: Vec<<<G1Affine as AffineCurve>::ScalarField as PrimeField>::BigInt>,
}

#[wasm_bindgen]
impl ScalarVectorInput {
    #[wasm_bindgen(constructor)]
    pub fn new(size: usize) -> Self {
        Self {
            scalar_vec: msm::generate_msm_inputs::<G1Affine>(size).1,
        }
    }

    #[wasm_bindgen(js_name = "toJsArray")]
    pub fn to_js_array(&self) -> Array {
        let arr = Array::new_with_length(self.scalar_vec.len() as u32);
        for (i, scalar) in (&self.scalar_vec).into_iter().enumerate() {
            let mut bytes: Vec<u8> = Vec::with_capacity(32);
            scalar.write(&mut bytes).unwrap();
            arr.set(i as u32, Uint8Array::from(bytes.as_slice()).into());
        }
        arr
    }

}

#[wasm_bindgen]
pub struct PointOutput {
    point: <<G1Affine as AffineCurve>::Projective as ProjectiveCurve>::Affine,
}

#[wasm_bindgen]
impl PointOutput {
    #[wasm_bindgen(js_name = "toJsArray")]
    pub fn to_js_array(&self) -> Array {
        let x = self.point.x;
        let y = self.point.y;
        let is_infinity = self.point.infinity;

        let mut x_bytes: Vec<u8> = Vec::with_capacity(48);
        x.write(&mut x_bytes).unwrap();
        let mut y_bytes: Vec<u8> = Vec::with_capacity(48);
        y.write(&mut y_bytes).unwrap();

        let point = Array::new_with_length(3);
        point.set(0, Uint8Array::from(x_bytes.as_slice()).into());
        point.set(1, Uint8Array::from(y_bytes.as_slice()).into());
        point.set(2, is_infinity.into());
        point
    }
}

#[wasm_bindgen]
pub unsafe fn compute_msm(
    point_vec: Array,
    scalar_vec: Array,
) -> PointOutput {

    let mut scalar: Vec<<<G1Affine as AffineCurve>::ScalarField as PrimeField>::BigInt> = Vec::new();
    let mut x = |e, _, _| {
        let array = Uint8Array::new(&e);
        let bytes: Vec<u8> = array.to_vec();
        let c = Cursor::new(bytes);
        let bigint = BigInteger256::read(c).unwrap();
        scalar.push(bigint);
    };
    scalar_vec.for_each(&mut x);

    let mut point: Vec<<<G1Affine as AffineCurve>::Projective as ProjectiveCurve>::Affine> = Vec::new();
    let mut a = |e, _, _| {
        let p = Array::from(&e);
        let a1 = Uint8Array::new(&p.at(0));
        let a2 = Uint8Array::new(&p.at(1));
        let a3 = p.at(2).as_bool().unwrap();
        let mut v: Vec<u8> = a1.to_vec();
        v.append(&mut a2.to_vec());
        v.append(&mut vec![a3 as u8]);
        let cursor = Cursor::new(v);
        point.push(G1Affine::read(cursor).unwrap());
    };
    point_vec.for_each(&mut a);

    use web_sys::console;

    let result =  msm::compute_msm::<G1Affine>(
        &point,
        &scalar
    ).unwrap();

    let m = format!("size {}", &result.len());

    let cursor = Cursor::new(result);

    let m2 = format!("error {:?}", G1Affine::deserialize_uncompressed(cursor.clone()).err());

    PointOutput{
        point: G1Affine::deserialize_uncompressed(cursor).unwrap()
    }
}