import { mod, modInverse, p } from "../finite-field-js.js";
import { bigintFromBytes, bigintToBits } from "../util.js";

export { msmDumbAffine, doubleAffine, addAffine, scale };

/**
 *
 * @param {import("./src/curve.js").CompatibleScalar[]} scalars
 * @param {import("./src/curve.js").CompatiblePoint[]} points
 */
function msmDumbAffine(scalars, points) {
  let n = points.length;

  let pointsBigint = points.map((P) => {
    let x = bigintFromBytes(P[0]);
    let y = bigintFromBytes(P[1]);
    let isZero = P[2];
    return [x, y, isZero];
  });
  let scalarsBits = scalars.map((s) => {
    let bigint = bigintFromBytes(s);
    return bigintToBits(bigint);
  });
  let sum = [0n, 0n, true];
  for (let i = 0; i < n; i++) {
    let s = scalarsBits[i];
    let P = pointsBigint[i];
    let Q = scale(s, P);
    // console.log("scale result", Q);
    sum = addAffine(sum, Q);
  }
  return sum;
}

function addAffine([x1, y1, isZero1], [x2, y2, isZero2]) {
  if (isZero1) {
    return [x2, y2, isZero2];
  }
  if (isZero2) {
    return [x1, y1, isZero1];
  }
  if (x1 === x2) {
    if (y1 === y2) {
      // P1 + P1 --> we double
      return doubleAffine([x1, y1, isZero1]);
    }
    if (y1 === -y2) {
      // P1 - P1 --> return zero
      return [0n, 0n, true];
    }
  }
  // m = (y2 - y1)/(x2 - x1)
  let d = modInverse(x2 - x1, p);
  let m = mod((y2 - y1) * d, p);
  // x3 = m^2 - x1 - x2
  let x3 = mod(m * m - x1 - x2, p);
  // y3 = m*(x1 - x3) - y1
  let y3 = mod(m * (x1 - x3) - y1, p);
  return [x3, y3, false];
}
function doubleAffine([x, y, isZero]) {
  if (isZero) return [0n, 0n, true];
  // m = 3*x^2 / 2y
  let d = modInverse(2n * y, p);
  let m = mod(3n * x * x * d, p);
  // x2 = m^2 - 2x
  let x2 = mod(m * m - 2n * x, p);
  // y2 = m*(x - x2) - y
  let y2 = mod(m * (x - x2) - y, p);
  return [x2, y2, false];
}
/**
 *
 * @param {boolean[]} scalar
 * @param {[bigint, bigint, boolean]} point
 * @returns
 */
function scale(scalar, point) {
  /**
   * @type {[bigint, bigint, boolean]}
   */
  let result = [0n, 0n, true];
  for (let i = 0; i < scalar.length - 1; i++) {
    if (scalar[i]) result = addAffine(result, point);
    point = doubleAffine(point);
  }
  if (scalar[scalar.length - 1]) result = addAffine(result, point);
  return result;
}
