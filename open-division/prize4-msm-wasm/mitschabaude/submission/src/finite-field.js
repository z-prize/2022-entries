import { createFiniteField } from "./finite-field-generate.js";
import * as wasm from "./wasm/finite-field.wasm.js";

export {
  mod,
  randomBaseField,
  randomBaseFieldx2,
  randomScalars,
  scalar,
} from "./finite-field-js.js";

export { p, toMontgomery, fromMontgomery };

let p =
  0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaabn;
let w = 30;

export {
  constants,
  multiply,
  reduce,
  inverse,
  add,
  subtract,
  sqrt,
  square,
  isEqual,
  isZero,
  addAffine,
  writeBigint,
  readBigInt,
  copy,
  getPointer,
  getPointers,
  getPointersInMemory,
  getEmptyPointersInMemory,
  resetPointers,
  readBytes,
  writeBytes,
  benchMultiply,
  benchInverse,
  n,
  w,
  packedSizeBytes,
  memory,
  memoryBytes,
  getZeroPointer,
  getZeroPointers,
  getOffset,
  getAndResetOpCounts,
};
let {
  // finite field
  constants,
  multiply,
  reduce,
  inverse,
  add,
  subtract,
  sqrt,
  square,
  isEqual,
  isZero,
  // elliptic curve
  addAffine,
  // helper
  writeBigint,
  readBigInt,
  copy,
  getPointer,
  getPointers,
  getPointersInMemory,
  getEmptyPointersInMemory,
  resetPointers,
  readBytes,
  writeBytes,
  // benchmarks
  benchMultiply,
  benchInverse,
  n,
  packedSizeBytes,
  memory,
  getZeroPointer,
  getZeroPointers,
  getOffset,
  getAndResetOpCounts,
} = createFiniteField(p, w, wasm);
let memoryBytes = new Uint8Array(memory.buffer);

/**
 *
 * @param {number} x
 */
function fromMontgomery(x) {
  multiply(x, x, constants.one);
  reduce(x);
}
/**
 *
 * @param {number} x
 */
function toMontgomery(x) {
  multiply(x, x, constants.R2);
}
