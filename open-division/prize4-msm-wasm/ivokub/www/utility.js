import * as submission from "./submission.wasm";

export function allocate_scalars_mem(size) {
    return submission.getScalarsMem();
}

export function allocate_points_mem(size) {
    return submission.getPointsMem();
}

export function load_scalars_to_wasm(dst, scalar_ptr, scalar_arr, size) {
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < 32; j++) {
            dst[scalar_ptr + 32 * i + j] = scalar_arr[i][j]; // reverse order
        }
    }
}

export function load_points_to_wasm(dst, point_ptr, point_arr, size) {
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < 48; j++) {
            dst[point_ptr + 96 * i + j] = point_arr[i][0][j];
            dst[point_ptr + 96 * i + 48 + j] = point_arr[i][1][j];
        }
    }
    submission.toMontgomery(point_ptr, 2*size);
}

export function compute_msm(point_vec, scalar_vec, size) {
    const res = submission.compute_msm(point_vec, scalar_vec, size);
    const mem = memory();
    const x = new Uint8Array(48);
    const y = new Uint8Array(48);
    for (let i = 0; i < 48; i++) {
        x[i] = mem[res+47-i];
        y[i] = mem[res+48+47-i];
    }
    return [x, y, false];
}

export function memory() {
    return new Uint8Array(submission.memory.buffer);
}