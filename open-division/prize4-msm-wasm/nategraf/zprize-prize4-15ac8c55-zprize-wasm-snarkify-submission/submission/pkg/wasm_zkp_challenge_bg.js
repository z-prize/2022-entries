import * as wasm from './wasm_zkp_challenge_bg.wasm';

const heap = new Array(32).fill(undefined);

heap.push(undefined, null, true, false);

function getObject(idx) { return heap[idx]; }

let heap_next = heap.length;

function dropObject(idx) {
    if (idx < 36) return;
    heap[idx] = heap_next;
    heap_next = idx;
}

function takeObject(idx) {
    const ret = getObject(idx);
    dropObject(idx);
    return ret;
}

const lTextDecoder = typeof TextDecoder === 'undefined' ? (0, module.require)('util').TextDecoder : TextDecoder;

let cachedTextDecoder = new lTextDecoder('utf-8', { ignoreBOM: true, fatal: true });

cachedTextDecoder.decode();

let cachedUint8Memory0 = new Uint8Array();

function getUint8Memory0() {
    if (cachedUint8Memory0.byteLength === 0) {
        cachedUint8Memory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8Memory0;
}

function getStringFromWasm0(ptr, len) {
    return cachedTextDecoder.decode(getUint8Memory0().subarray(ptr, ptr + len));
}

function addHeapObject(obj) {
    if (heap_next === heap.length) heap.push(heap.length + 1);
    const idx = heap_next;
    heap_next = heap[idx];

    heap[idx] = obj;
    return idx;
}

let stack_pointer = 32;

function addBorrowedObject(obj) {
    if (stack_pointer == 1) throw new Error('out of js stack');
    heap[--stack_pointer] = obj;
    return stack_pointer;
}

let WASM_VECTOR_LEN = 0;

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1);
    getUint8Memory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}
/**
* @param {Uint8Array} data
* @returns {InstanceObjectVector}
*/
export function deserialize_msm_inputs(data) {
    const ptr0 = passArray8ToWasm0(data, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.deserialize_msm_inputs(ptr0, len0);
    return InstanceObjectVector.__wrap(ret);
}

/**
* @param {number} size
* @returns {InstanceObject}
*/
export function generate_msm_inputs(size) {
    const ret = wasm.generate_msm_inputs(size);
    return InstanceObject.__wrap(ret);
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
* @returns {PointOutput}
*/
export function compute_msm_baseline(point_vec, scalar_vec) {
    _assertClass(point_vec, PointVectorInput);
    _assertClass(scalar_vec, ScalarVectorInput);
    const ret = wasm.compute_msm_baseline(point_vec.ptr, scalar_vec.ptr);
    return PointOutput.__wrap(ret);
}

/**
* @param {PointVectorInput} point_vec
* @param {ScalarVectorInput} scalar_vec
* @returns {PointOutput}
*/
export function compute_msm(point_vec, scalar_vec) {
    _assertClass(point_vec, PointVectorInput);
    _assertClass(scalar_vec, ScalarVectorInput);
    const ret = wasm.compute_msm(point_vec.ptr, scalar_vec.ptr);
    return PointOutput.__wrap(ret);
}

/**
* @param {PointVectorInput} point_vec
* @param {ScalarVectorInput} scalar_vec
* @param {number} c
* @returns {PointOutput}
*/
export function compute_msm_with_c(point_vec, scalar_vec, c) {
    _assertClass(point_vec, PointVectorInput);
    _assertClass(scalar_vec, ScalarVectorInput);
    const ret = wasm.compute_msm_with_c(point_vec.ptr, scalar_vec.ptr, c);
    return PointOutput.__wrap(ret);
}

/**
*/
export class InstanceObject {

    static __wrap(ptr) {
        const obj = Object.create(InstanceObject.prototype);
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
        wasm.__wbg_instanceobject_free(ptr);
    }
    /**
    * @returns {number}
    */
    get length() {
        const ret = wasm.instanceobject_length(this.ptr);
        return ret >>> 0;
    }
    /**
    * @returns {PointVectorInput}
    */
    points() {
        const ret = wasm.instanceobject_points(this.ptr);
        return PointVectorInput.__wrap(ret);
    }
    /**
    * @returns {ScalarVectorInput}
    */
    scalars() {
        const ret = wasm.instanceobject_scalars(this.ptr);
        return ScalarVectorInput.__wrap(ret);
    }
}
/**
*/
export class InstanceObjectVector {

    static __wrap(ptr) {
        const obj = Object.create(InstanceObjectVector.prototype);
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
        wasm.__wbg_instanceobjectvector_free(ptr);
    }
    /**
    * @returns {number}
    */
    get length() {
        const ret = wasm.instanceobject_length(this.ptr);
        return ret >>> 0;
    }
    /**
    * @param {number} i
    * @returns {InstanceObject}
    */
    at(i) {
        const ret = wasm.instanceobjectvector_at(this.ptr, i);
        return InstanceObject.__wrap(ret);
    }
}
/**
*/
export class PointOutput {

    static __wrap(ptr) {
        const obj = Object.create(PointOutput.prototype);
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
        wasm.__wbg_pointoutput_free(ptr);
    }
    /**
    * @returns {Array<any>}
    */
    toJsArray() {
        const ret = wasm.pointoutput_toJsArray(this.ptr);
        return takeObject(ret);
    }
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
    /**
    * @returns {Array<any>}
    */
    toJsArray() {
        const ret = wasm.pointvectorinput_toJsArray(this.ptr);
        return takeObject(ret);
    }
    /**
    * @param {Array<any>} arr
    * @returns {PointVectorInput}
    */
    static fromJsArray(arr) {
        try {
            const ret = wasm.pointvectorinput_fromJsArray(addBorrowedObject(arr));
            return PointVectorInput.__wrap(ret);
        } finally {
            heap[stack_pointer++] = undefined;
        }
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
    /**
    * @returns {Array<any>}
    */
    toJsArray() {
        const ret = wasm.scalarvectorinput_toJsArray(this.ptr);
        return takeObject(ret);
    }
    /**
    * @param {Array<any>} arr
    * @returns {ScalarVectorInput}
    */
    static fromJsArray(arr) {
        try {
            const ret = wasm.scalarvectorinput_fromJsArray(addBorrowedObject(arr));
            return ScalarVectorInput.__wrap(ret);
        } finally {
            heap[stack_pointer++] = undefined;
        }
    }
}

export function __wbindgen_object_drop_ref(arg0) {
    takeObject(arg0);
};

export function __wbindgen_boolean_get(arg0) {
    const v = getObject(arg0);
    const ret = typeof(v) === 'boolean' ? (v ? 1 : 0) : 2;
    return ret;
};

export function __wbg_get_57245cc7d7c7619d(arg0, arg1) {
    const ret = getObject(arg0)[arg1 >>> 0];
    return addHeapObject(ret);
};

export function __wbg_length_6e3bbe7c8bd4dbd8(arg0) {
    const ret = getObject(arg0).length;
    return ret;
};

export function __wbg_newwithlength_7c42f7e738a9d5d3(arg0) {
    const ret = new Array(arg0 >>> 0);
    return addHeapObject(ret);
};

export function __wbg_set_a68214f35c417fa9(arg0, arg1, arg2) {
    getObject(arg0)[arg1 >>> 0] = takeObject(arg2);
};

export function __wbg_from_7ce3cb27cb258569(arg0) {
    const ret = Array.from(getObject(arg0));
    return addHeapObject(ret);
};

export function __wbg_buffer_3f3d764d4747d564(arg0) {
    const ret = getObject(arg0).buffer;
    return addHeapObject(ret);
};

export function __wbg_newwithbyteoffsetandlength_d9aa266703cb98be(arg0, arg1, arg2) {
    const ret = new Uint8Array(getObject(arg0), arg1 >>> 0, arg2 >>> 0);
    return addHeapObject(ret);
};

export function __wbg_new_8c3f0052272a457a(arg0) {
    const ret = new Uint8Array(getObject(arg0));
    return addHeapObject(ret);
};

export function __wbg_set_83db9690f9353e79(arg0, arg1, arg2) {
    getObject(arg0).set(getObject(arg1), arg2 >>> 0);
};

export function __wbg_length_9e1ae1900cb0fbd5(arg0) {
    const ret = getObject(arg0).length;
    return ret;
};

export function __wbindgen_throw(arg0, arg1) {
    throw new Error(getStringFromWasm0(arg0, arg1));
};

export function __wbindgen_memory() {
    const ret = wasm.memory;
    return addHeapObject(ret);
};

