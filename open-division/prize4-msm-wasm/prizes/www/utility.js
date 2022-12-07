// Utilities
// Feel free to modify anything in this file.

import * as submission from "./submission.wasm"

// Loads scalar array `scalar_arr` and point array `point_arr` of length `size` from JS memory
// to wasm memory (`dst`) starting at index `ptr`.
// 
// This function assumes that wasm requires the following memory layout: 
// ptr -- ptr+32*size : scalar
// ptr+32*size - ptr+32*size+48*2*size : point
// 
// You may choose any other memory layout inside wasm. Feel free to modify this function.
export function load_to_wasm(dst, ptr, scalar_arr, piont_arr, size) {
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < 32; j++) {
            dst[ptr] = scalar_arr[i][j];
            ptr = ptr + 1;
        }
    }
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < 48; j++) {
            dst[ptr] = piont_arr[i][0][j];
            ptr = ptr + 1;
        }
        for (let j = 0; j < 48; j++) {
            dst[ptr] = piont_arr[i][1][j];
            ptr = ptr + 1;
        }
    }
}

export function to_mont(num_element, data_ptr) {
    let offset = num_element * 32; // point vec start
    //affine
    for (let i = 0; i < num_element; i++) {
        let point_ptr = data_ptr + offset + i * 48 * 2;
        submission.f1m_toMontgomery(point_ptr, point_ptr)
        submission.f1m_toMontgomery(point_ptr + 48, point_ptr + 48)
    }
}

