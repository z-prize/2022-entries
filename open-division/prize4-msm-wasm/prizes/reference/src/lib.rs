use ark_bls12_381::G1Affine;
use ark_ec::{AffineCurve, ProjectiveCurve};
use ark_ff::{PrimeField, ToBytes};
use js_sys::{Array, Uint8Array};
use wasm_bindgen::prelude::wasm_bindgen;

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
pub fn compute_msm(
    point_vec: &PointVectorInput,
    scalar_vec: &ScalarVectorInput,
) -> PointOutput {
    PointOutput {
        point: msm::compute_msm::<G1Affine>(&point_vec.point_vec, &scalar_vec.scalar_vec)
            .into_affine(),
    }
}
