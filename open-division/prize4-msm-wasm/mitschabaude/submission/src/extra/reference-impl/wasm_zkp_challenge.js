let imports = {};
imports['__wbindgen_placeholder__'] = module.exports;
let wasm;
const { TextDecoder } = require(`util`);

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

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });

cachedTextDecoder.decode();

let cachedUint8Memory0;
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

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
    return instance.ptr;
}
/**
* @param {PointVectorInput} point_vec
* @param {ScalarVectorInput} scalar_vec
* @returns {Array<any>}
*/
module.exports.compute_msm = function(point_vec, scalar_vec) {
    _assertClass(point_vec, PointVectorInput);
    var ptr0 = point_vec.ptr;
    point_vec.ptr = 0;
    _assertClass(scalar_vec, ScalarVectorInput);
    var ptr1 = scalar_vec.ptr;
    scalar_vec.ptr = 0;
    const ret = wasm.compute_msm(ptr0, ptr1);
    return takeObject(ret);
};

/**
*/
class PointVectorInput {

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
        const ret = wasm.pointvectorinput_fromJsArray(addHeapObject(arr));
        return PointVectorInput.__wrap(ret);
    }
}
module.exports.PointVectorInput = PointVectorInput;
/**
*/
class ScalarVectorInput {

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
        const ret = wasm.scalarvectorinput_fromJsArray(addHeapObject(arr));
        return ScalarVectorInput.__wrap(ret);
    }
}
module.exports.ScalarVectorInput = ScalarVectorInput;

module.exports.__wbindgen_object_drop_ref = function(arg0) {
    takeObject(arg0);
};

module.exports.__wbindgen_boolean_get = function(arg0) {
    const v = getObject(arg0);
    const ret = typeof(v) === 'boolean' ? (v ? 1 : 0) : 2;
    return ret;
};

module.exports.__wbg_get_f0f4f1608ebf633e = function(arg0, arg1) {
    const ret = getObject(arg0)[arg1 >>> 0];
    return addHeapObject(ret);
};

module.exports.__wbg_length_93debb0e2e184ab6 = function(arg0) {
    const ret = getObject(arg0).length;
    return ret;
};

module.exports.__wbg_newwithlength_51bd08aed34ec6a3 = function(arg0) {
    const ret = new Array(arg0 >>> 0);
    return addHeapObject(ret);
};

module.exports.__wbg_set_c1d04f8b45a036e7 = function(arg0, arg1, arg2) {
    getObject(arg0)[arg1 >>> 0] = takeObject(arg2);
};

module.exports.__wbg_buffer_de1150f91b23aa89 = function(arg0) {
    const ret = getObject(arg0).buffer;
    return addHeapObject(ret);
};

module.exports.__wbg_newwithbyteoffsetandlength_9ca61320599a2c84 = function(arg0, arg1, arg2) {
    const ret = new Uint8Array(getObject(arg0), arg1 >>> 0, arg2 >>> 0);
    return addHeapObject(ret);
};

module.exports.__wbg_new_97cf52648830a70d = function(arg0) {
    const ret = new Uint8Array(getObject(arg0));
    return addHeapObject(ret);
};

module.exports.__wbg_set_a0172b213e2469e9 = function(arg0, arg1, arg2) {
    getObject(arg0).set(getObject(arg1), arg2 >>> 0);
};

module.exports.__wbg_length_e09c0b925ab8de5d = function(arg0) {
    const ret = getObject(arg0).length;
    return ret;
};

module.exports.__wbindgen_throw = function(arg0, arg1) {
    throw new Error(getStringFromWasm0(arg0, arg1));
};

module.exports.__wbindgen_memory = function() {
    const ret = wasm.memory;
    return addHeapObject(ret);
};

const path = require('path').join(__dirname, 'wasm_zkp_challenge_bg.wasm');
const bytes = require('fs').readFileSync(path);

const wasmModule = new WebAssembly.Module(bytes);
const wasmInstance = new WebAssembly.Instance(wasmModule, imports);
wasm = wasmInstance.exports;
module.exports.__wasm = wasm;

cachedUint8Memory0 = new Uint8Array(wasm.memory.buffer);

