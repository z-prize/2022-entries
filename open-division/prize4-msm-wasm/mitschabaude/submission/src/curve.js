/**
 * this file contains hosts most of the basic curve operations we need --
 * with the notable exception of affine addition, which is implemented in wasm and just re-exported here
 */
import {
  addAffine,
  add,
  subtract,
  multiply,
  square,
  copy,
  constants,
  memoryBytes,
  n,
  isEqual,
} from "./finite-field.js";
export {
  addAffine,
  doubleAffine,
  addAssignProjective,
  doubleInPlaceProjective,
  sizeField,
  sizeAffine,
  sizeProjective,
  isZeroAffine,
  isZeroProjective,
  copyAffine,
  copyProjective,
  copyAffineToProjectiveNonZero,
  affineCoords,
  projectiveCoords,
  setNonZeroProjective,
  setIsNonZeroAffine,
};

let sizeField = 8 * n;
let sizeAffine = 16 * n + 8;
let sizeProjective = 24 * n + 8;

/**
 * affine EC doubling, H = 2*G
 *
 * assuming d = 1/(2*y) is given, and inputs aren't zero.
 *
 * this supports doubling a point in-place with H === G
 * @param {number[]} scratch
 * @param {number} H output point
 * @param {number} G input point (x, y)
 * @param {number} d 1/(2y)
 */
function doubleAffine([m, tmp, x2, y2], H, G, d) {
  let [x, y] = affineCoords(G);
  let [xOut, yOut] = affineCoords(H);
  // m = 3*x^2*d
  square(m, x);
  add(tmp, m, m); // TODO efficient doubling
  add(m, tmp, m);
  multiply(m, d, m);
  // x2 = m^2 - 2x
  square(x2, m);
  add(tmp, x, x); // TODO efficient doubling
  subtract(x2, x2, tmp);
  // y2 = (x - x2)*m - y
  subtract(y2, x, x2);
  multiply(y2, y2, m);
  subtract(y2, y2, y);
  // H = x2,y2
  copy(xOut, x2);
  copy(yOut, y2);
}

/**
 * projective point addition with assignement, P1 += P2
 *
 * @param {number[]} scratch
 * @param {number} P1
 * @param {number} P2
 */
function addAssignProjective(scratch, P1, P2) {
  if (isZeroProjective(P1)) {
    copyProjective(P1, P2);
    return;
  }
  if (isZeroProjective(P2)) return;
  setNonZeroProjective(P1);
  let [X1, Y1, Z1] = projectiveCoords(P1);
  let [X2, Y2, Z2] = projectiveCoords(P2);
  let [Y2Z1, Y1Z2, X2Z1, X1Z2, Z1Z2, u, uu, v, vv, vvv, R] = scratch;
  // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#addition-add-1998-cmo-2
  // Y1Z2 = Y1*Z2
  multiply(Y1Z2, Y1, Z2);
  // Y2Z1 = Y2*Z1
  multiply(Y2Z1, Y2, Z1);
  // X1Z2 = X1*Z2
  multiply(X1Z2, X1, Z2);
  // X2Z1 = X2*Z1
  multiply(X2Z1, X2, Z1);

  // double if the points are equal
  // x1*z2 = x2*z1 and y1*z2 = y2*z1
  // <==>  x1/z1 = x2/z2 and y1/z1 = y2/z2
  if (isEqual(X1Z2, X2Z1) && isEqual(Y1Z2, Y2Z1)) {
    doubleInPlaceProjective(scratch, P1);
    return;
  }
  // Z1Z2 = Z1*Z2
  multiply(Z1Z2, Z1, Z2);
  // u = Y2Z1-Y1Z2
  subtract(u, Y2Z1, Y1Z2);
  // uu = u^2
  square(uu, u);
  // v = X2Z1-X1Z2
  subtract(v, X2Z1, X1Z2);
  // vv = v^2
  square(vv, v);
  // vvv = v*vv
  multiply(vvv, v, vv);
  // R = vv*X1Z2
  multiply(R, vv, X1Z2);
  // A = uu*Z1Z2-vvv-2*R
  let A = uu;
  multiply(A, uu, Z1Z2);
  subtract(A, A, vvv);
  subtract(A, A, R);
  subtract(A, A, R);
  // X3 = v*A
  multiply(X1, v, A);
  // Y3 = u*(R-A)-vvv*Y1Z2
  subtract(R, R, A);
  multiply(Y1, u, R);
  multiply(Y1Z2, vvv, Y1Z2);
  subtract(Y1, Y1, Y1Z2);
  // Z3 = vvv*Z1Z2
  multiply(Z1, vvv, Z1Z2);
}

/**
 * projective point doubling with assignment, P *= 2
 *
 * @param {number[]} scratch
 * @param {Point} P
 */
function doubleInPlaceProjective(scratch, P) {
  if (isZeroProjective(P)) return;
  let [X1, Y1, Z1] = projectiveCoords(P);
  let [tmp, w, s, ss, sss, Rx2, Bx4, h] = scratch;
  // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-projective.html#doubling-dbl-1998-cmo-2
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
  // 2*B (= X1*R), Bx4 = 2*B+2*B
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
 * @param {number} pointer
 */
function isZeroAffine(pointer) {
  return !memoryBytes[pointer + 2 * sizeField];
}
/**
 * @param {number} pointer
 */
function isZeroProjective(pointer) {
  return !memoryBytes[pointer + 3 * sizeField];
}

/**
 *
 * @param {number} target
 * @param {number} source
 */
function copyAffine(target, source) {
  memoryBytes.copyWithin(target, source, source + sizeAffine);
}
/**
 *
 * @param {number} target
 * @param {number} source
 */
function copyProjective(target, source) {
  memoryBytes.copyWithin(target, source, source + sizeProjective);
}
/**
 * @param {number} P
 * @param {number} A
 */
function copyAffineToProjectiveNonZero(P, A) {
  // x,y = x,y
  memoryBytes.copyWithin(P, A, A + 2 * sizeField);
  // z = 1
  memoryBytes.copyWithin(
    P + 2 * sizeField,
    constants.mg1,
    constants.mg1 + sizeField
  );
  // isNonZero = 1
  memoryBytes[P + 3 * sizeField] = 1;
  // isInfinity = isInfinity
  // memoryBytes[P + 3 * sizeField] = memoryBytes[A + 2 * sizeField];
}

/**
 * @param {number} pointer
 */
function affineCoords(pointer) {
  return [pointer, pointer + sizeField];
}
/**
 * @param {number} pointer
 */
function projectiveCoords(pointer) {
  return [pointer, pointer + sizeField, pointer + 2 * sizeField];
}

/**
 * @param {number} pointer
 */
function setNonZeroProjective(pointer) {
  memoryBytes[pointer + 3 * sizeField] = 1;
}
/**
 * @param {number} pointer
 * @param {boolean} isNonZero
 */
function setIsNonZeroAffine(pointer, isNonZero) {
  memoryBytes[pointer + 2 * sizeField] = Number(isNonZero);
}
