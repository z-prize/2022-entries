import * as wasm from './wasm_zkp_challenge_bg.wasm';

const lTextDecoder = typeof TextDecoder === 'undefined' ? (0, module.require)('util').TextDecoder : TextDecoder;

let cachedTextDecoder = new lTextDecoder('utf-8', { ignoreBOM: true, fatal: true });

cachedTextDecoder.decode();

let cachegetUint8Memory0 = null;
function getUint8Memory0() {
    if (cachegetUint8Memory0 === null || cachegetUint8Memory0.buffer !== wasm.memory.buffer) {
        cachegetUint8Memory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachegetUint8Memory0;
}

function getStringFromWasm0(ptr, len) {
    return cachedTextDecoder.decode(getUint8Memory0().subarray(ptr, ptr + len));
}

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
    return instance.ptr;
}
/**
* @param {PointVectorInput} point_vec
* @param {ScalarVectorInput} scalar_vec
*/
export function compute_msm(point_vec, scalar_vec) {
    _assertClass(point_vec, PointVectorInput);
    var ptr0 = point_vec.ptr;
    point_vec.ptr = 0;
    _assertClass(scalar_vec, ScalarVectorInput);
    var ptr1 = scalar_vec.ptr;
    scalar_vec.ptr = 0;
    wasm.compute_msm(ptr0, ptr1);
}

/**
*/
export class PointVectorInput {

    static __wrap(ptr) {
        const obj = Object.create(PointVectorInput.prototype);
        obj.ptr = ptr;

        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.ptr;
        this.ptr = 0;

        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_pointvectorinput_free(ptr);
    }
    /**
    * @param {number} size
    */
    constructor(size) {
        const ret = wasm.pointvectorinput_new(size);
        return PointVectorInput.__wrap(ret);
    }
}
/**
*/
export class ScalarVectorInput {

    static __wrap(ptr) {
        const obj = Object.create(ScalarVectorInput.prototype);
        obj.ptr = ptr;

        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.ptr;
        this.ptr = 0;

        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_scalarvectorinput_free(ptr);
    }
    /**
    * @param {number} size
    */
    constructor(size) {
        const ret = wasm.scalarvectorinput_new(size);
        return ScalarVectorInput.__wrap(ret);
    }
}

export function __wbindgen_throw(arg0, arg1) {
    throw new Error(getStringFromWasm0(arg0, arg1));
};

