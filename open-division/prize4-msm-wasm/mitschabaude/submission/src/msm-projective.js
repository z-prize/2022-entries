/**
 * Simple MSM implementation based on projective arithmetic and vanilla bucket method.
 *
 * ~2.5x faster than the reference, even though the high-level algorithm is less optimized
 * (doesn't use mixed additions) -- thanks to the improvements in low-level field arithmetic
 *
 * This is useful to keep around as a reference that's simple to reason about.
 *
 * TODO: currently, this uses a slightly different point layout and therefor duplicates implementations
 * used for the affine MSM; should be consolidated.
 */
import {
  constants,
  writeBigint,
  inverse,
  sqrt,
  square,
  randomBaseField,
  scalar,
  fromMontgomery,
  toMontgomery,
  getPointers,
  resetPointers,
  readBytes,
  writeBytes,
  readBigInt,
} from "./finite-field.js";
import {
  multiply,
  add,
  subtract,
  copy,
  multiplyCount,
  resetMultiplyCount,
  isEqual,
} from "./wasm/finite-field.wasm.js";
import { extractBitSlice, log2 } from "./util.js";

export {
  msmProjective,
  randomCurvePoints,
  doubleInPlaceProjective,
  addAssignProjective,
};

/**
 * @typedef {{x: number; y: number; z: number, isZero?: boolean}} Point
 * @typedef {{x: number; y: number}} AffinePoint
 * @typedef {[xArray: Uint8Array, yArray: Uint8Array, isInfinity: boolean]} CompatiblePoint
 * @typedef {Uint8Array} CompatibleScalar
 */

const curve = {
  zero: {
    x: constants.mg1,
    y: constants.mg1,
    z: constants.zero,
    isZero: true,
  },
};

let numberOfAdds = 0;
let numberOfDoubles = 0;

/**
 *
 * @param {CompatibleScalar[]} scalars
 * @param {CompatiblePoint[]} inputPoints
 */
function msmProjective(scalars, inputPoints) {
  // initialize buckets
  let n = scalars.length;
  let c = log2(n) - 3; // TODO: determine c from n and hand-optimized lookup table
  let minC = 1;
  if (c < minC) c = minC;

  let K = Math.ceil(256 / c); // number of partitions
  let L = 2 ** c - 1; // number of buckets per partition (skipping the 0 bucket)
  let points = getPointers(n * 3); // initialize points
  let buckets = getPointers(L * K * 3); // initialize buckets
  let bucketsZero = Array(L * K).fill(true);
  let bucketSums = getPointers(K * 3);
  let bucketSumsZero = Array(K).fill(true);
  let partialSums = getPointers(K * 3);
  let partialSumsZero = Array(K).fill(true);
  let finalSumXyz = getPointers(3);
  let finalSum = {
    x: finalSumXyz[0],
    y: finalSumXyz[1],
    z: finalSumXyz[2],
    isZero: true,
  };
  let scratchSpace = getPointers(20);
  let affinePoint = takeAffinePoint(scratchSpace);

  resetMultiplyCount();
  numberOfAdds = 0;
  numberOfDoubles = 0;

  // first loop -- compute buckets
  for (let i = 0; i < n; i++) {
    let scalar = scalars[i];
    let inputPoint = inputPoints[i];
    // convert point to projective
    writeBytes(scratchSpace, affinePoint.x, inputPoint[0]);
    writeBytes(scratchSpace, affinePoint.y, inputPoint[1]);
    toMontgomery(affinePoint.x);
    toMontgomery(affinePoint.y);
    // TODO: make points have contiguous memory representation
    let x = points[i * 2];
    let y = points[i * 2 + 1];
    let z = points[i * 2 + 2];
    let point = { x, y, z, isZero: inputPoint[2] };
    fromAffine(point, affinePoint);
    // partition 32-byte scalar into c-bit chunks
    for (let k = 0; k < K; k++) {
      // compute k-th digit from scalar
      let l = extractBitSlice(scalar, k * c, c);
      if (l === 0) continue;
      // get bucket for digit and add point to it
      let idx = k * L + (l - 1);
      let x = buckets[idx * 3];
      let y = buckets[idx * 3 + 1];
      let z = buckets[idx * 3 + 2];
      let isZero = bucketsZero[idx];
      let bucket = { x, y, z, isZero };
      addAssignProjective(scratchSpace, bucket, point);
      bucketsZero[idx] = bucket.isZero;
    }
  }

  let nMul1 = multiplyCount.valueOf();
  resetMultiplyCount();

  // second loop -- sum up buckets to partial sums
  for (let l = L; l > 0; l--) {
    for (let k = 0; k < K; k++) {
      let idx = k * L + (l - 1);
      let bucket = {
        x: buckets[idx * 3],
        y: buckets[idx * 3 + 1],
        z: buckets[idx * 3 + 2],
        isZero: bucketsZero[idx],
      };
      let bucketSum = {
        x: bucketSums[k * 3],
        y: bucketSums[k * 3 + 1],
        z: bucketSums[k * 3 + 2],
        isZero: bucketSumsZero[k],
      };
      let partialSum = {
        x: partialSums[k * 3],
        y: partialSums[k * 3 + 1],
        z: partialSums[k * 3 + 2],
        isZero: partialSumsZero[k],
      };
      // TODO: this should have faster paths if a summand is zero
      // (bucket is zero pretty often; bucketSum at the beginning)
      addAssignProjective(scratchSpace, bucketSum, bucket);
      bucketSumsZero[k] = bucketSum.isZero;
      addAssignProjective(scratchSpace, partialSum, bucketSum);
      partialSumsZero[k] = partialSum.isZero;
    }
  }

  let nMul2 = multiplyCount.valueOf();
  resetMultiplyCount();

  // third loop -- compute final sum using horner's rule
  let k = K - 1;
  let partialSum = {
    x: partialSums[k * 3],
    y: partialSums[k * 3 + 1],
    z: partialSums[k * 3 + 2],
    isZero: partialSumsZero[k],
  };
  writePointInto(finalSum, partialSum);
  k--;
  for (; k >= 0; k--) {
    for (let j = 0; j < c; j++) {
      doubleInPlaceProjective(scratchSpace, finalSum);
    }
    let partialSum = {
      x: partialSums[k * 3],
      y: partialSums[k * 3 + 1],
      z: partialSums[k * 3 + 2],
      isZero: partialSumsZero[k],
    };
    addAssignProjective(scratchSpace, finalSum, partialSum);
  }

  let nMul3 = multiplyCount.valueOf();
  resetMultiplyCount();

  let [x, y, z] = toProjectiveOutput(finalSum);

  // TODO read out and return result
  resetPointers();

  return { nMul1, nMul2, nMul3, x, y, z, numberOfAdds, numberOfDoubles };
}

/**
 *
 * @param {number} n
 */
function randomCurvePoints(n) {
  let scratchSpace = getPointers(32);
  /**
   * @type {CompatiblePoint[]}
   */
  let points = Array(n);
  for (let i = 0; i < n; i++) {
    points[i] = randomCurvePoint(scratchSpace);
  }
  resetPointers();
  return points;
}

/**
 * @param {number[]} scratchSpace
 * @returns {CompatiblePoint}
 */
function randomCurvePoint([x, y, z, ...scratchSpace]) {
  let { mg1, mg4 } = constants;
  writeBigint(x, randomBaseField());
  let [ysquare] = scratchSpace;

  // let i = 0;
  while (true) {
    // compute y^2 = x^3 + 4
    multiply(ysquare, x, x);
    multiply(ysquare, ysquare, x);
    add(ysquare, ysquare, mg4);

    // try computing square root to get y (works half the time, because half the field elements are squares)
    let isRoot = sqrt(scratchSpace, y, ysquare);
    if (isRoot) break;
    // if it didn't work, increase x by 1 and try again
    add(x, x, mg1);
  }
  copy(z, mg1);
  let p = { x, y, z, isZero: false };
  let minusZP = takePoint(scratchSpace);
  // clear cofactor
  scaleProjective(scratchSpace, minusZP, scalar.asBits.minusZ, p); // -z*p
  addAssignProjective(scratchSpace, p, minusZP); // p = p - z*p = -(z - 1) * p
  // convert to affine point, back to normal representation and to byte arrays
  let affineP = takeAffinePoint(scratchSpace);
  toAffine(scratchSpace, affineP, p);
  fromMontgomery(affineP.x);
  fromMontgomery(affineP.y);
  return [
    readBytes(scratchSpace, affineP.x),
    readBytes(scratchSpace, affineP.y),
    false,
  ];
}

/**
 * @param {number[]} scratchSpace
 * @param {AffinePoint} affine affine representation
 * @param {Point} point projective representation
 */
function toAffine([zinv, ...scratchSpace], { x: x0, y: y0 }, { x, y, z }) {
  // return x/z, y/z
  inverse(scratchSpace[0], zinv, z);
  multiply(x0, x, zinv);
  multiply(y0, y, zinv);
}

/**
 * @param {number[]} scratchSpace
 * @param {ProjectivePoint} point projective representation
 */
function toAffineOutput([zinv, ...scratchSpace], { x, y, z }) {
  // return x/z, y/z
  inverse(scratchSpace[0], zinv, z);
  multiply(x, x, zinv);
  multiply(y, y, zinv);
  fromMontgomery(x);
  fromMontgomery(y);
  return [readBigInt(x), readBigInt(y)];
}

/**
 * @param {ProjectivePoint} point projective representation
 */
function toProjectiveOutput({ x, y, z }) {
  fromMontgomery(x);
  fromMontgomery(y);
  fromMontgomery(z);
  return [readBigInt(x), readBigInt(y), readBigInt(z)];
}

/**
 *
 * @param {Point} point
 * @param {AffinePoint} affinePoint
 */
function fromAffine({ x, y, z }, affinePoint) {
  copy(x, affinePoint.x);
  copy(y, affinePoint.y);
  copy(z, constants.mg1);
}

/**
 * @param {number[]} scratchSpace
 * @param {Point} result
 * @param {boolean[]} scalar
 * @param {Point} point
 */
function scaleProjective([x, y, z, ...scratchSpace], result, scalar, point) {
  writePointInto(result, curve.zero);
  point = writePointInto({ x, y, z }, point);
  for (let bit of scalar) {
    if (bit) {
      addAssignProjective(scratchSpace, result, point);
    }
    doubleInPlaceProjective(scratchSpace, point);
  }
}

/**
 * P *= 2
 * @param {number[]} scratchSpace
 * @param {Point} point
 */
function doubleInPlaceProjective(scratch, P) {
  // http://hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#doubling-dbl-1998-cmo-2
  // (with a = 0)
  let { x: X1, y: Y1, z: Z1, isZero } = P;
  if (isZero) return;
  numberOfDoubles++;
  let [tmp, w, s, ss, sss, Rx2, Bx4, h] = scratch;
  // w = 3*X1^2
  square(w, X1);
  add(tmp, w, w); // TODO efficient doubling
  add(w, tmp, w);
  // s = Y1*Z1
  multiply(s, Y1, Z1);
  // ss = s^2
  square(ss, s);
  // sss = s*ss
  multiply(sss, ss, s);
  // R = Y1*s, Rx2 = R + R
  multiply(Rx2, Y1, s);
  add(Rx2, Rx2, Rx2);
  // 2*(B = X1*R), Bx4 = 2*B+2*B
  multiply(Bx4, X1, Rx2);
  add(Bx4, Bx4, Bx4);
  // h = w^2-8*B = w^2 - Bx4 - Bx4
  square(h, w);
  subtract(h, h, Bx4); // TODO efficient doubling
  subtract(h, h, Bx4);
  // X3 = 2*h*s
  multiply(X1, h, s);
  add(X1, X1, X1);
  // Y3 = w*(4*B-h)-8*R^2 = (Bx4 - h)*w - (Rx2^2 + Rx2^2)
  subtract(Y1, Bx4, h);
  multiply(Y1, Y1, w);
  square(tmp, Rx2);
  add(tmp, tmp, tmp); // TODO efficient doubling
  subtract(Y1, Y1, tmp);
  // Z3 = 8*sss
  multiply(Z1, sss, constants.mg8);
}

/**
 * p1 += p2
 * @param {number[]} scratch
 * @param {Point} P1
 * @param {Point} p2
 */
function addAssignProjective(scratch, P1, P2) {
  // if (p1.isZero()) return p2;
  // if (p2.isZero()) return p1;
  let { x: x1, y: y1, z: z1, isZero: isZero1 } = P1;
  let { x: x2, y: y2, z: z2, isZero: isZero2 } = P2;
  if (isZero1) {
    writePointInto(P1, P2);
    return;
  }

  if (isZero2) return;
  numberOfAdds++;
  P1.isZero = false;

  let [u1, u2, v1, v2, u, v, vv, vvv, v2vv, w, a] = scratch;
  // const U1 = Y2.multiply(Z1);
  // const U2 = Y1.multiply(Z2);
  // const V1 = X2.multiply(Z1);
  // const V2 = X1.multiply(Z2);
  multiply(u1, y2, z1);
  multiply(u2, y1, z2);
  multiply(v1, x2, z1);
  multiply(v2, x1, z2);

  // x1/z1 = x2/z2  <==>  x1*z2 = x2*z1  <==>  v2 = v1
  if (isEqual(v1, v2) && isEqual(u1, u2)) {
    doubleInPlaceProjective(scratch, P1);
    return;
  }

  // if (V1.equals(V2) && U1.equals(U2)) return this.double();
  // if (V1.equals(V2)) return this.getZero();
  // const U = U1.subtract(U2);
  subtract(u, u1, u2);
  // const V = V1.subtract(V2);
  subtract(v, v1, v2);
  // const VV = V.multiply(V);
  square(vv, v);
  // const VVV = VV.multiply(V);
  multiply(vvv, vv, v);
  // const V2VV = V2.multiply(VV);
  multiply(v2vv, v2, vv);
  // const W = Z1.multiply(Z2);
  multiply(w, z1, z2);
  // const A = U.multiply(U).multiply(W).subtract(VVV).subtract(V2VV.multiply(2n));
  square(a, u);
  multiply(a, a, w);
  subtract(a, a, vvv);
  subtract(a, a, v2vv);
  subtract(a, a, v2vv);
  // const X3 = V.multiply(A);
  multiply(x1, v, a);
  // const Z3 = VVV.multiply(W);
  multiply(z1, vvv, w);
  // const Y3 = U.multiply(V2VV.subtract(A)).subtract(VVV.multiply(U2));
  subtract(v2vv, v2vv, a);
  multiply(vvv, vvv, u2);
  multiply(y1, u, v2vv);
  subtract(y1, y1, vvv);
}

/**
 *
 * @param {Point} targetPoint
 * @param {Point} point
 */
function writePointInto(targetPoint, point) {
  copy(targetPoint.x, point.x);
  copy(targetPoint.y, point.y);
  copy(targetPoint.z, point.z);
  targetPoint.isZero = point.isZero;
  return targetPoint;
}

/**
 *
 * @param {number[]} scratchSpace
 * @returns
 */
function takePoint(scratchSpace) {
  let [x, y, z] = scratchSpace.splice(0, 3);
  return { x, y, z };
}
/**
 *
 * @param {number[]} scratchSpace
 * @returns
 */
function takeAffinePoint(scratchSpace) {
  let [x, y] = scratchSpace.splice(0, 2);
  return { x, y };
}

function readProjectiveAsAffine(
  [x0, y0, ...scratchSpace],
  { x, y, z, isZero }
) {
  if (isZero) return { x: readBigInt(x), y: readBigInt(y), isZero: true };
  toAffine(scratchSpace, { x: x0, y: y0 }, { x, y, z, isZero });
  return {
    x: readBigInt(x0),
    y: readBigInt(y0),
    isZero,
  };
}

function readProjective(
  [tmpX, tmpY, tmpZ],
  { x, y, z, isZero },
  changeRepresentation = false
) {
  if (changeRepresentation) {
    multiply(tmpX, x, constants.one);
    multiply(tmpY, y, constants.one);
    multiply(tmpZ, z, constants.one);
    return {
      x: readBigInt(tmpX),
      y: readBigInt(tmpY),
      z: readBigInt(tmpZ),
      isZero,
    };
  }
  return {
    x: readBigInt(x),
    y: readBigInt(y),
    z: readBigInt(z),
    isZero,
  };
}
