import {
  add,
  multiply,
  reduce,
  subtract,
  makeOdd,
  copy,
  isEqual,
  isEqualNegative,
  leftShift,
  square,
  subtractPositive,
  inverse,
  multiplySchoolbook,
  batchInverse,
} from "./src/wasm/finite-field.wasm.js";
import {
  p,
  constants,
  mod,
  randomBaseFieldx2,
  writeBigint,
  readBigInt,
  toMontgomery,
  getPointers,
  w,
  n,
} from "./src/finite-field.js";
import { webcrypto } from "node:crypto";
import { extractBitSlice } from "./src/util.js";
import { modInverse } from "./src/finite-field-js.js";
import { testDecomposeRandomScalar } from "./src/scalar-glv.js";
// web crypto compat
if (Number(process.version.slice(1, 3)) < 19) globalThis.crypto = webcrypto;

function toWasm(x0, x) {
  writeBigint(x, x0);
  toMontgomery(x);
}
function ofWasm([tmp], x) {
  multiply(tmp, x, constants.one);
  reduce(tmp);
  return mod(readBigInt(tmp), p);
}

let [x, y, z, z_hi, ...scratch] = getPointers(10);

let R = mod(1n << BigInt(w * n), p);
let Rinv = modInverse(R, p);

function test() {
  let x0 = randomBaseFieldx2();
  let y0 = randomBaseFieldx2();
  toWasm(x0, x);
  toWasm(y0, y);

  // multiply
  let z0 = mod(x0 * y0, p);
  multiply(z, x, y);
  let z1 = ofWasm(scratch, z);
  if (z0 !== z1) throw Error("multiply");
  z0 = 0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffffn; // test overflow resistance
  toWasm(z0, z);
  z0 = mod(z0 * z0, p);
  multiply(z, z, z);
  z1 = ofWasm(scratch, z);
  if (z0 !== z1) throw Error("multiply");

  // square
  z0 = mod(x0 * x0, p);
  square(z, x);
  z1 = ofWasm(scratch, z);
  if (z0 !== z1) throw Error("square");

  // leftShift
  let k = 97;
  z0 = 1n << BigInt(k);
  // computes R^2 * 2^k / R = 2^k R, which is 2^k in montgomery form
  leftShift(z, constants.R2, k);
  z1 = ofWasm(scratch, z);
  if (z1 !== z0) throw Error("leftShift");

  // barrett multiplication
  writeBigint(x, x0);
  writeBigint(y, y0);
  let xy0 = x0 * y0;
  z0 = mod(x0 * y0, p);
  multiplySchoolbook(z, x, y);
  z1 = readBigInt(z);
  let l = readBigInt(z_hi);
  let lTrue = (xy0 - z0) / p;
  let xHi = xy0 >> 380n;
  let m = 2n ** (380n + 390n) / p;
  let lApprox = (xHi * m) >> 390n;
  console.assert(lTrue * p + z0 === xy0, "barrett: test correctness");
  console.assert(l === lApprox, "barrett: l");
  console.assert([0n, 1n].includes(lTrue - l), "barrett: error is 0 or 1");
  if (mod(z0 - z1, p) !== 0n) throw Error("barrett multiply");
  toWasm(x0, x);
  toWasm(y0, y);

  // add
  z0 = mod(x0 + y0, p);
  add(z, x, y);
  z1 = ofWasm(scratch, z);
  if (z0 !== z1) throw Error("add");

  // subtract
  z0 = mod(x0 - y0, p);
  subtract(z, x, y);
  z1 = ofWasm(scratch, z);
  if (z0 !== z1) throw Error("subtract");

  // subtract plus 2p
  z0 = mod(x0 - y0, p);
  subtractPositive(z, x, y);
  z1 = ofWasm(scratch, z);
  if (z0 !== z1) throw Error("subtract");

  // reduceInPlace
  z0 = x0 >= p ? x0 - p : x0;
  copy(z, x);
  reduce(z);
  z1 = ofWasm(scratch, z);
  if (z0 !== z1) throw Error("reduceInPlace");

  // isEqual
  if (isEqual(x, x) !== 1) throw Error("isEqual");
  if (isEqual(x, y) !== 0) throw Error("isEqual");
  subtract(y, constants.p, x);
  if (isEqualNegative(x, y) !== 1) throw Error("isEqualNegative");

  // inverse
  inverse(scratch[0], z, x);
  multiply(z, z, x);
  z1 = ofWasm(scratch, z);
  if (z1 !== 1n) throw Error("inverse");

  // makeOdd
  writeBigint(x, 5n << 120n);
  writeBigint(z, 3n);
  makeOdd(x, z);
  x0 = readBigInt(x);
  z0 = readBigInt(z);
  if (!(x0 === 5n && z0 === 3n << 120n)) throw Error("makeOdd");

  // extractBitSlice
  let arr = new Uint8Array([0b0010_0110, 0b1101_0101, 0b1111_1111]);
  let e = Error("extractBitSlice");
  if (extractBitSlice(arr, 2, 4) !== 0b10_01) throw e;
  if (extractBitSlice(arr, 0, 2) !== 0b10) throw e;
  if (extractBitSlice(arr, 0, 8) !== 0b0010_0110) throw e;
  if (extractBitSlice(arr, 3, 9) !== 0b0101_0010_0) throw e;
  if (extractBitSlice(arr, 8, 8) !== 0b1101_0101) throw e;
  if (extractBitSlice(arr, 5, 3 + 8 + 2) !== 0b11_1101_0101_001) throw e;
  if (extractBitSlice(arr, 16, 10) !== 0b1111_1111) throw e;
}

function testBatchMontgomery() {
  let n = 1000;
  let X = getPointers(n);
  let invX = getPointers(n);
  let scratch = getPointers(10);
  for (let i = 0; i < n; i++) {
    let x0 = randomBaseFieldx2();
    writeBigint(X[i], x0);
    // compute inverses normally
    inverse(scratch[0], invX[i], X[i]);
  }
  // compute inverses as batch
  let invX1 = getPointers(n);
  batchInverse(scratch[0], invX1[0], X[0], n);

  // check that all inverses are equal
  for (let i = 0; i < n; i++) {
    if (mod(readBigInt(invX1[i]) - readBigInt(invX[i]), p) !== 0n)
      throw Error("batch inverse");
    if (!isEqual(reduce(invX1[i]), reduce(invX[i])))
      console.warn("WARNING: batch inverse not exactly equal after reducing");
  }
}

for (let i = 0; i < 20; i++) {
  test();
}
for (let i = 0; i < 100; i++) {
  let ok = testDecomposeRandomScalar();
  if (!ok) throw Error("scalar decomposition");
}

testBatchMontgomery();
