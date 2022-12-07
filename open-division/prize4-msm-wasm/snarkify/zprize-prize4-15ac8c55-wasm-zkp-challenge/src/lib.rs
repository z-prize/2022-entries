use ark_ec::{AffineCurve, ProjectiveCurve};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use js_sys::{Array, Uint8Array};
use wasm_bindgen::prelude::*;

#[cfg(feature = "debug")]
use console_error_panic_hook;

// If the console.err panic hook is included, initialize it exactly once.
// init_panic_hook is called at the top of every public function.
fn init_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// A println! style macro to allow output to the JS console.
/// ```ignore
/// crate::console_log!("hello from {}", "rust!");
/// ```
/// Will only have an effect in builds with the `debug` feature enabled.
#[macro_export]
macro_rules! console_log {
    ($($t:tt)*) => {
        #[cfg(feature = "debug")]
        web_sys::console::log_1(&format_args!($($t)*).to_string().into());
    }
}

pub mod msm;

#[wasm_bindgen]
pub struct PointVectorInput {
    point_vec: Vec<msm::G1Affine>,
}

#[wasm_bindgen]
impl PointVectorInput {
    #[wasm_bindgen(constructor)]
    pub fn new(size: usize) -> Self {
        init_panic_hook();
        let (point_vec, _) = msm::generate_msm_inputs(size);

        Self { point_vec }
    }

    #[wasm_bindgen(js_name = "toJsArray")]
    pub fn to_js_array(&self) -> Array {
        let arr = Array::new_with_length(self.point_vec.len() as u32);
        for (i, point) in (&self.point_vec).into_iter().enumerate() {
            let x = point.x;
            let y = point.y;
            let is_infinity = point.infinity;

            let mut x_bytes: Vec<u8> = Vec::with_capacity(48);
            x.serialize(&mut x_bytes).unwrap();
            let mut y_bytes: Vec<u8> = Vec::with_capacity(48);
            y.serialize(&mut y_bytes).unwrap();

            let point = Array::new_with_length(3);
            point.set(0, Uint8Array::from(x_bytes.as_slice()).into());
            point.set(1, Uint8Array::from(y_bytes.as_slice()).into());
            point.set(2, is_infinity.into());
            arr.set(i as u32, point.into());
        }
        arr
    }

    #[wasm_bindgen(js_name = "fromJsArray")]
    pub fn from_js_array(arr: &Array) -> Self {
        let mut point_vec = Vec::<msm::G1Affine>::with_capacity(arr.length() as usize);
        for i in 0..arr.length() {
            let tuple = Array::from(&arr.get(i));

            // Check whether the given encoded point is the point at infinity.
            let is_infinity = tuple.get(2).as_bool().unwrap();
            if is_infinity {
                point_vec.push(msm::G1Affine::identity());
                continue;
            }

            let x_bytes: Vec<u8> = Uint8Array::from(tuple.get(0)).to_vec();
            let x =
                <msm::G1Affine as AffineCurve>::BaseField::deserialize(x_bytes.as_slice()).unwrap();

            let y_bytes: Vec<u8> = Uint8Array::from(tuple.get(1)).to_vec();
            let y =
                <msm::G1Affine as AffineCurve>::BaseField::deserialize(y_bytes.as_slice()).unwrap();

            point_vec.push(msm::G1Affine::new_unchecked(x, y));
        }

        Self { point_vec }
    }
}

#[wasm_bindgen]
pub struct ScalarVectorInput {
    scalar_vec: Vec<msm::BigInt>,
}

#[wasm_bindgen]
impl ScalarVectorInput {
    #[wasm_bindgen(constructor)]
    pub fn new(size: usize) -> Self {
        init_panic_hook();
        let (_, scalar_vec) = msm::generate_msm_inputs(size);

        Self { scalar_vec }
    }

    #[wasm_bindgen(js_name = "toJsArray")]
    pub fn to_js_array(&self) -> Array {
        let arr = Array::new_with_length(self.scalar_vec.len() as u32);
        for (i, scalar) in (&self.scalar_vec).into_iter().enumerate() {
            let mut bytes: Vec<u8> = Vec::with_capacity(32);
            scalar.serialize(&mut bytes).unwrap();
            arr.set(i as u32, Uint8Array::from(bytes.as_slice()).into());
        }
        arr
    }

    #[wasm_bindgen(js_name = "fromJsArray")]
    pub fn from_js_array(arr: &Array) -> Self {
        let mut scalar_vec = Vec::<msm::BigInt>::with_capacity(arr.length() as usize);
        for i in 0..arr.length() {
            let bytes: Vec<u8> = Uint8Array::from(arr.get(i)).to_vec();
            scalar_vec.push(msm::BigInt::deserialize(bytes.as_slice()).unwrap());
        }

        Self { scalar_vec }
    }
}

#[wasm_bindgen]
pub struct InstanceObject {
    points: Vec<msm::G1Affine>,
    scalars: Vec<msm::BigInt>,
}

#[wasm_bindgen]
impl InstanceObject {
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.points.len()
    }

    #[wasm_bindgen]
    pub fn points(&self) -> PointVectorInput {
        PointVectorInput {
            point_vec: self.points.clone(),
        }
    }

    #[wasm_bindgen]
    pub fn scalars(&self) -> ScalarVectorInput {
        ScalarVectorInput {
            scalar_vec: self.scalars.clone(),
        }
    }
}

#[wasm_bindgen]
pub struct InstanceObjectVector {
    instances: Vec<InstanceObject>,
}

#[wasm_bindgen]
impl InstanceObjectVector {
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.instances.len()
    }

    // Copy the instance to hand off the the JS VM.
    // Note that this copies the full undderlying data, which may be quite large.
    pub fn at(&self, i: usize) -> InstanceObject {
        InstanceObject {
            points: self.instances[i].points.clone(),
            scalars: self.instances[i].scalars.clone(),
        }
    }
}

#[wasm_bindgen]
pub struct PointOutput {
    point: msm::G1Affine,
}

#[wasm_bindgen]
impl PointOutput {
    #[wasm_bindgen(js_name = "toJsArray")]
    pub fn to_js_array(&self) -> Array {
        let x = self.point.x;
        let y = self.point.y;
        let is_infinity = self.point.infinity;

        let mut x_bytes: Vec<u8> = Vec::with_capacity(48);
        x.serialize(&mut x_bytes).unwrap();
        let mut y_bytes: Vec<u8> = Vec::with_capacity(48);
        y.serialize(&mut y_bytes).unwrap();

        let point = Array::new_with_length(3);
        point.set(0, Uint8Array::from(x_bytes.as_slice()).into());
        point.set(1, Uint8Array::from(y_bytes.as_slice()).into());
        point.set(2, is_infinity.into());
        point
    }
}

#[wasm_bindgen]
pub fn deserialize_msm_inputs(data: &[u8]) -> InstanceObjectVector {
    init_panic_hook();
    let instances = Vec::<msm::Instance>::deserialize_unchecked(data).unwrap();
    InstanceObjectVector {
        instances: instances
            .into_iter()
            .map(|i| InstanceObject {
                points: i.points,
                scalars: i.scalars,
            })
            .collect(),
    }
}

#[wasm_bindgen]
pub fn generate_msm_inputs(size: usize) -> InstanceObject {
    init_panic_hook();
    let (points, scalars) = msm::generate_msm_inputs(size);
    InstanceObject { points, scalars }
}

#[wasm_bindgen]
pub fn compute_msm_baseline(
    point_vec: &PointVectorInput,
    scalar_vec: &ScalarVectorInput,
) -> PointOutput {
    init_panic_hook();
    PointOutput {
        point: msm::compute_msm_baseline(&point_vec.point_vec, &scalar_vec.scalar_vec)
            .into_affine(),
    }
}

#[wasm_bindgen]
pub fn compute_msm(point_vec: &PointVectorInput, scalar_vec: &ScalarVectorInput) -> PointOutput {
    init_panic_hook();
    PointOutput {
        point: msm::compute_msm::<true, true>(&point_vec.point_vec, &scalar_vec.scalar_vec, None)
            .into_affine(),
    }
}

#[wasm_bindgen]
pub fn compute_msm_with_c(
    point_vec: &PointVectorInput,
    scalar_vec: &ScalarVectorInput,
    c: usize,
) -> PointOutput {
    init_panic_hook();
    PointOutput {
        point: msm::compute_msm::<true, true>(
            &point_vec.point_vec,
            &scalar_vec.scalar_vec,
            Some(c),
        )
        .into_affine(),
    }
}

#[cfg(feature = "coverage")]
#[wasm_bindgen]
pub fn minicov_capture_coverage() -> Vec<u8> {
    minicov::capture_coverage()
}
