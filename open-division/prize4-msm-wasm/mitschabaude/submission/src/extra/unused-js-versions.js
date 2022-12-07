/**
 * This contains some of the initial JS versions of functions used in the MSM, that are now implemented in Wasm
 *
 * The JS versions are way easier to debug and typically have very similar, or just slightly degraded, performance compared to their Wasm counter-parts.
 */
import { affineCoords, setIsNonZeroAffine } from "../curve.js";
import {
  inverse,
  multiply,
  square,
  subtract,
  subtractPositive,
} from "../finite-field.js";

/**
 * Given points G0,...,G(n-1) and H0,...,H(n-1), compute
 *
 * Si = Gi + Hi, i=0,...,n-1
 *
 * unsafe: this is a faster version which doesn't handle edge cases!
 * it assumes all the Gi, Hi are non-zero and we won't hit cases where Gi === +/-Hi
 *
 * this is a valid assumption in parts of the msm, for important applications like the prover side of a commitment scheme like KZG or IPA,
 * where inputs are independent and pseudo-random in significant parts of the msm algorithm
 * (we always use the safe version in those parts of the msm where the chance of edge cases is non-negligible)
 *
 * the performance improvement is in the ballpark of 1-3%
 *
 * @param {number[]} scratch
 * @param {Uint32Array} tmp pointers of length n
 * @param {Uint32Array} d pointers of length n
 * @param {Uint32Array} S
 * @param {Uint32Array} G
 * @param {Uint32Array} H
 * @param {number} n
 */
function batchAddUnsafeJs(scratch, tmp, d, S, G, H, n) {
  for (let i = 0; i < n; i++) {
    subtractPositive(tmp[i], H[i], G[i]);
  }
  batchInverseJs(scratch[0], d[0], tmp[0], n);
  for (let i = 0; i < n; i++) {
    addAffineJs(scratch[0], S[i], G[i], H[i], d[i]);
  }
}

/**
 * @param {number[]} scratch
 * @param {Uint32Array} invX inverted fields of at least length n
 * @param {Uint32Array} X fields to invert, at least length n
 * @param {number} n length
 */
function batchInverseJs([I, tmp], invX, X, n) {
  if (n === 0) return;
  if (n === 1) {
    inverse(tmp, invX[0], X[0]);
    return;
  }
  // invX = [_, x0*x1, ..., x0*....*x(n-2), x0*....*x(n-1)]
  // invX[i] = x0*...*xi
  multiply(invX[1], X[1], X[0]);
  for (let i = 2; i < n; i++) {
    multiply(invX[i], invX[i - 1], X[i]);
  }
  // I = 1/(x0*....*x(n-1)) = 1/invX[n-1]
  inverse(tmp, I, invX[n - 1]);

  for (let i = n - 1; i > 1; i--) {
    multiply(invX[i], invX[i - 1], I);
    multiply(I, I, X[i]);
  }
  // now I = 1/(x0*x1)
  multiply(invX[1], X[0], I);
  multiply(invX[0], I, X[1]);
}

/**
 * affine EC addition, G3 = G1 + G2
 *
 * assuming d = 1/(x2 - x1) is given, and inputs aren't zero, and x1 !== x2
 * (edge cases are handled one level higher, before batching)
 *
 * this supports addition with assignment where G3 === G1 (but not G3 === G2)
 * @param {number[]} scratch
 * @param {number} G3 (x3, y3)
 * @param {number} G1 (x1, y1)
 * @param {number} G2 (x2, y2)
 * @param {number} d 1/(x2 - x1)
 */
function addAffineJs([m, tmp], G3, G1, G2, d) {
  let [x1, y1] = affineCoords(G1);
  let [x2, y2] = affineCoords(G2);
  let [x3, y3] = affineCoords(G3);
  setIsNonZeroAffine(G3, true);
  // m = (y2 - y1)*d
  subtractPositive(m, y2, y1);
  multiply(m, m, d);
  // x3 = m^2 - x1 - x2
  square(tmp, m);
  subtract(x3, tmp, x1);
  subtract(x3, x3, x2);
  // y3 = (x2 - x3)*m - y2
  subtractPositive(y3, x2, x3);
  multiply(y3, y3, m);
  subtract(y3, y3, y2);
}
