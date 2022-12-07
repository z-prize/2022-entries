/*
 * Copyright (c) 2022 EulerSmile ( see https://github.com/EulerSmile).
 *
 * Dual-licensed under both the MIT and Apache-2.0 licenses;
 * you may not use this file except in compliance with the License.
 */

use js_sys::{Array, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;

#[link(name = "mcore")]
extern "C" {
    fn ECP_muln_rust(
        p: *mut u8,
        n: i32,
        x: *const u8,
        ix: *mut i32,
        e: *const u8,
        ie: *mut i32,
        ib: *mut i32,
    );
}

#[wasm_bindgen]
pub fn compute_msm(points: Array, scalars: Array) -> Array {
    let size = std::cmp::min(points.length(), scalars.length());
    let mut x = Vec::new();
    let mut e = Vec::new();

    points.iter().into_iter().for_each(|pt: JsValue| {
        let pt = Array::from(&pt);
        // Arkworks use Little-Endian, so we need reverse x and y.
        let mut xs = Uint8Array::new(&pt.get(0)).to_vec();
        xs.reverse();
        let mut ys = Uint8Array::new(&pt.get(1)).to_vec();
        ys.reverse();

        x.push(0x04);
        x.extend_from_slice(&xs);
        x.extend_from_slice(&ys);
    });

    scalars.iter().into_iter().for_each(|sc: JsValue| {
        let mut sc: Vec<u8> = Uint8Array::new(&sc).to_vec();
        sc.reverse();
        e.extend_from_slice(&sc);
    });

    let mut out = vec![0; 97];
    // Allocating memory space for C code
    // cx will be the points space
    let mut cx: Vec<i32> = vec![0; (size * 45) as usize];
    // ce will be the scalars space
    let mut ce: Vec<i32> = vec![0; (size * 14) as usize];
    // cb will be the window space
    let mut cb: Vec<i32> = vec![0; 2 * 8192 * 45];

    unsafe {
        ECP_muln_rust(
            out.as_mut_ptr(),
            size as i32,
            x.as_ptr(),
            cx.as_mut_ptr(),
            e.as_ptr(),
            ce.as_mut_ptr(),
            cb.as_mut_ptr(),
        );
        out.set_len(97);
    };

    // Again, we need re-serialize Miracl's bytes.
    // remove the first 0x04.
    out.remove(0);
    out[0..48].reverse();
    out[48..96].reverse();

    let point = Array::new_with_length(3);
    point.set(0, Uint8Array::from(&out[0..48]).into());
    point.set(1, Uint8Array::from(&out[48..96]).into());
    point.set(2, false.into());

    point
}
