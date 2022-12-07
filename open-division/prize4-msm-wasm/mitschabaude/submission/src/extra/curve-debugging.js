/**
 * this contains some helpers for debugging, which are currently unused only because we aren't, at this moment, debugging something
 */
import {
  affineCoords,
  isZeroAffine,
  isZeroProjective,
  projectiveCoords,
} from "../curve.js";
import {
  readBigInt,
  mod,
  p,
  inverse,
  multiply,
  fromMontgomery,
} from "../finite-field.js";

export { readAffine, readProjective, readProjectiveAsAffine };

function readAffine(P) {
  let isZero = isZeroAffine(P);
  let [x, y] = affineCoords(P);
  return {
    x: mod(readBigInt(x), p),
    y: mod(readBigInt(y), p),
    isZero,
  };
}
function readProjective(P) {
  let isZero = isZeroProjective(P);
  let [x, y, z] = projectiveCoords(P);
  return {
    x: readBigInt(x),
    y: readBigInt(y),
    z: readBigInt(z),
    isZero,
  };
}
function readProjectiveAsAffine(scratchSpace, P) {
  let isZero = isZeroProjective(P);
  if (isZero) {
    let [x, y] = projectiveCoords(P);
    return { x: readBigInt(x), y: readBigInt(y), isZero: true };
  }
  let [x1, y1] = toAffineBigints(scratchSpace, P);
  return { x: x1, y: y1, isZero };
}

/**
 * @param {number[]} scratchSpace
 * @param {number} point projective representation
 */
function toAffineBigints([zinv, x1, y1, ...scratchSpace], P) {
  let [x, y, z] = projectiveCoords(P);
  // return x/z, y/z
  inverse(scratchSpace[0], zinv, z);
  multiply(x1, x, zinv);
  multiply(y1, y, zinv);
  return [mod(readBigInt(x1), p), mod(readBigInt(y1), p)];
}
