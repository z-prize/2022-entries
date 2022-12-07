export let memory: WebAssembly.Memory;

export function multiply(out: number, x: number, y: number): void;
export function square(out: number, x: number): void;
export function leftShift(out: number, x: number, k: number): void;
/**
 * x*(y - z)
 */
export function multiplyDifference(
  out: number,
  x: number,
  y: number,
  z: number
): void;
export function barrett(x: number): void;
export function multiplySchoolbook(xy: number, x: number, y: number): void;
export function add(out: number, x: number, y: number): void;
export function subtract(out: number, x: number, y: number): void;
export function reduce(x: number): void;
export function addNoReduce(out: number, x: number, y: number): void;
export function subtractNoReduce(out: number, x: number, y: number): void;
export function subtractPositive(out: number, x: number, y: number): void;
export function makeOdd(u: number, s: number): number;
// helpers
export function isEqual(x: number, y: number): boolean;
export function isEqualNegative(x: number, y: number): boolean;
export function isZero(x: number): boolean;
export function isGreater(x: number, y: number): boolean;
export function copy(x: number, y: number): void;
export function toPackedBytes(bytes: number, x: number): void;
export function fromPackedBytes(x: number, bytes: number): void;

/**
 * affine EC addition, G3 = G1 + G2
 *
 * assuming d = 1/(x2 - x1) is given, and inputs aren't zero, and x1 !== x2
 * (edge cases are handled one level higher, before batching)
 *
 * this supports addition with assignment where G3 === G1 (but not G3 === G2)
 * @param scratch
 * @param G3 (x3, y3)
 * @param G1 (x1, y1)
 * @param G2 (x2, y2)
 * @param d 1/(x2 - x1)
 */
export function addAffine(
  scratch: number,
  G3: number,
  G1: number,
  G2: number,
  d: number
): void;

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
 * @param scratch
 * @param tmp pointers of length n
 * @param d pointers of length n
 * @param S
 * @param G
 * @param H
 * @param n
 */
export function batchAddUnsafe(
  scratch: number,
  tmp: number,
  d: number,
  S: number,
  G: number,
  H: number,
  n: number
): void;

export function endomorphism(Gout: number, G: number): void;

/**
 * compute r = a^(-1) * 2^k mod p, returns k
 *
 * @param scratch
 * @param r
 * @param a
 */
export function almostInverse(scratch: number, r: number, a: number): number;

/**
 * montgomery inverse, a 2^K -> a^(-1) 2^K (mod p)
 *
 * @param scratch
 * @param r
 * @param a
 */
export function inverse(scratch: number, r: number, a: number): void;

/**
 * montgomery batch inverse
 *
 * @param scratch
 * @param z inverses to be computed
 * @param x field elements x0, ..., x(n-1)
 * @param n
 */
export function batchInverse(
  scratch: number,
  z: number,
  x: number,
  n: number
): void;

export const multiplyCount: number;
export function resetMultiplyCount(): void;
export const inverseCount: number;
export function resetInverseCount(): void;
