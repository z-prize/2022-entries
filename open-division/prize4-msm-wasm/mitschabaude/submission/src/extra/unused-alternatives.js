/**
 * this contains alternative or earlier versions of some algorithms, that we didn't end up using
 * but that may still be interesting as inspiration or reference.
 */

import {
  addAssignProjective,
  copyAffineToProjectiveNonZero,
  sizeProjective,
} from "../curve.js";
import { getPointers, getZeroPointers } from "../finite-field.js";

/**
 * alternative, purely projective version of reducing buckets into partition sums
 *
 * @param {number[]} scratchSpace
 * @param {number[][][]} buckets
 * @param {{c: number, K: number, L: number}}
 */
function reduceBucketsSimple(scratchSpace, buckets, { K, L }) {
  /**
   * @type {number[][]}
   */
  let bucketSums = Array(K);
  for (let k = 0; k < K; k++) {
    bucketSums[k] = getPointers(L + 1, sizeProjective);
  }
  let runningSums = getZeroPointers(K, sizeProjective);
  let partialSums = getZeroPointers(K, sizeProjective);

  // sum up buckets to partial sums
  for (let l = L; l > 0; l--) {
    for (let k = 0; k < K; k++) {
      let bucket = buckets[k][l][0];
      let runningSum = runningSums[k];
      let partialSum = partialSums[k];
      if (bucket === undefined) {
        // bucket sum is zero => running sum stays the same
        addAssignProjective(scratchSpace, partialSum, runningSum);
      } else {
        // bucket sum is affine, we convert to projective here
        let bucketSum = bucketSums[k][l];
        copyAffineToProjectiveNonZero(bucketSum, bucket);
        addAssignProjective(scratchSpace, runningSum, bucketSum);
        addAssignProjective(scratchSpace, partialSum, runningSum);
      }
    }
  }
  return partialSums;
}
