import {
  addAffine,
  doubleAffine,
  addAssignProjective,
  doubleInPlaceProjective,
  copyAffine,
  copyProjective,
  copyAffineToProjectiveNonZero,
  isZeroAffine,
  projectiveCoords,
  setIsNonZeroAffine,
  isZeroProjective,
} from "./curve.js";
import {
  getPointers,
  n,
  memoryBytes,
  getZeroPointers,
  writeBytes,
  toMontgomery,
  resetPointers,
  constants,
  fromMontgomery,
  getPointer,
  getAndResetOpCounts,
  readBytes,
  getPointersInMemory,
  getEmptyPointersInMemory,
  packedSizeBytes,
} from "./finite-field.js";
import {
  multiply,
  copy,
  subtract,
  add,
  isEqual,
  subtractPositive,
  inverse,
  endomorphism,
  batchInverse,
  batchAddUnsafe,
} from "./wasm/finite-field.wasm.js";
import {
  decompose,
  writeBytesScalar,
  scalarSize,
  getPointerScalar,
  resetPointersScalar,
  extractBitSlice,
  scalarBitlength,
} from "./scalar-glv.js";
import { log2 } from "./util.js";

export { msmAffine, batchAdd };

/**
 * Memory layout of curve points
 * -------------
 *
 * a _field element_ x is represented as n limbs, where n is a parameter that depends on the field order and limb size.
 * in wasm memory, each limb is stored as an `i64`, i.e. takes up 8 bytes of space.
 * (usually only the lowest w bits of each `i64` are filled, where w <= 32 is some configured limb size;
 * but within computations we will sometimes fill up most or all of the 64 bits)
 *
 * an _affine point_ is layed out as `[x, y, isNonZero]` in memory, where x and y are field elements and
 * `isNonZero` is a flag used to track whether a point is zero / the point at infinity.
 * - x, y each have length sizeField = 8*n bytes
 * - `isNonZero` is either 0 or 1, but we nevertheless reserve 8 bytes (one `i64`) of space for it.
 *   this helps ensure that all memory addresses are multiples of 8, a property which is required by JS APIs like
 *   BigInt64Array, and which should also make memory accesses more efficient.
 *
 * in code, we represent an affine point by a pointer `p`.
 * a pointer is just a JS `number`, and can be easily passed between wasm and JS.
 * on the wasm side, a number appears as an `i32`, suitable as input to memory load/store operations.
 *
 * from `p`, we obtain pointers to the individual coordinates as
 * ```
 * x = p
 * y = p + sizeField
 * isNonZero = p + 2*sizeField
 * ```
 *
 * for a _projective point_, the layout is `[x, y, z, isNonZero]`.
 * we can obtain x, y from a pointer as before, and
 * ```
 * z = p + 2*sizeField
 * isNonZero = p + 3*sizeField
 * ```
 */
let sizeField = 8 * n; // a field element has n limbs, each of which is an int64 (= 8 bytes)
let sizeAffine = 16 * n + 8; // an affine point is 2 field elements + 1 int64 for isNonZero flag
let sizeProjective = 24 * n + 8;

/**
 * table of the form `[n]: (c, c0)`, which has msm parameters c, c0 for different n.
 * n is the log-size of scalar and point inputs.
 * table was optimized for the zprize evaluation environment.
 *
 * @param c window size
 * @param c0 log-size of sub-partitions used in the bucket reduction step
 */
let cTable = {
  [14]: [13, 8],
  [16]: [13, 9],
  [18]: [16, 9],
};

/**
 * @typedef {[xArray: Uint8Array, yArray: Uint8Array, isInfinity: boolean]} InputPoint
 */

/**
 * MSM (multi-scalar multiplication)
 * ----------------------------------
 *
 * given scalars `s_i` and points `G_i`, `i=0,...N-1`, compute
 *
 * `[s_0] G_0 + ... + [s_(N-1)] G_(N-1)`.
 *
 * broadly, our implementation uses the pippenger algorithm / bucket method, where scalars are sliced
 * into windows of size c, giving rise to K = [b/c] _partitions_ or "sub-MSMs" (where b is the scalar bit length).
 *
 * for each partition k, points `G_i` are sorted into `L = 2^(c-1)` _buckets_ according to the ḱth NAF slice of their scalar `s_i`.
 * in total, we end up with `K*L` buckets, which are indexed by `(k, l)` where `k = 0,...K-1` and `l = 1,...,L`.
 *
 * after sorting the points, computation proceeds in **three main steps:**
 * 1. each bucket is accumulated into a single point, the _bucket sum_ `B_(l,k)`, which is simply the sum of all points in the bucket.
 * 2. the bucket sums of each partition k are reduced into a partition sum `P_k = 1*B_(k, 1) + 2*B_(k, 2) + ... + L*B_(k, L)`.
 * 3. the partition sums are reduced into the final result, `S = P_0 + 2^c*P_1 + ... + 2^(c*(K-1))*P_(K-1)`
 *
 * ### High-level implementation
 *
 * - we use **batch-affine additions** for step 1 (bucket accumulation),
 *   as pioneered by Zac Williamson in Aztec's barretenberg library: https://github.com/AztecProtocol/barretenberg/pull/19.
 *   thus, in this step we loop over all buckets, collect pairs of points to add, and then do a batch-addition on all of them.
 *   this is done in multiple passes, until the points of each bucket are summed to a single point, in an implicit binary tree.
 *   (in each pass, empty buckets and buckets with 1 remaining point are skipped;
 *   also, buckets of uneven length have a dangling point at the end, which doesn't belong to a pair and is skipped and included in a later pass)
 * - as a novelty, we also use **batch-affine additions for all of step 2** (bucket reduction).
 *   we achieve this by splitting up each partition recursively into sub-partitions, which are reduced independently from each other.
 *   this gives us enough independent additions to amortize the cost of the inversion in the batch-add step.
 *   sub-partitions are recombined in a series of comparatively cheap, log-sized steps. for details, see {@link reduceBucketsAffine}.
 * - we switch from an affine to a projective point representation between steps 2 and 3. step 3 is so tiny (< 0.1% of the computation)
 *   that the performance of projective curve arithmetic becomes irrelevant.
 *
 * the algorithm has a significant **preparation phase**, which happens before step 1, where we split scalars and sort points and such.
 * before splitting scalars into length-c slices, we do a **GLV decomposition**, where each 256-bit scalar is split into two
 * 128-bit chunks as `s = s0 + s1*lambda`. multiplying a point by `lambda` is a curve endomorphism,
 * with an efficient implementation `[lambda] (x,y) = (beta*x, y) =: endo((x, y))`,
 * where `lambda` and `beta` are certain cube roots of 1 in their respective fields.
 * correspondingly, each point `G` becomes two points `G`, `endo(G)`.
 * we also store `-G` and `-endo(G)` which are used when the NAF slices of `s0`, `s1` are negative.
 *
 * other than processing inputs, the preparation phase is concerned with organizing points. this should be done in a way which:
 * 1. enables to efficiently collect independent point pairs to add, in multiple successive passes over all buckets;
 * 2. makes memory access efficient when batch-adding pairs => ideally, the 2 points that form a pair, as well as consecutive pairs, are stored next to each other
 *
 * we address these two goals by copying all points to K independent linear arrays; one for each partition.
 * ordering in each of these arrays is achieved by performing a _counting sort_ of all points with respect to their bucket `l` in partition `k`.
 *
 * between step 1 and 2, there is a similar re-organization step. at the end of step 1, bucket sums are accumulated into the `0` locations
 * of each original bucket, which are spread apart as far as the original buckets were long.
 * before step 2, we copy bucket sums to a new linear array from 1 to L, for each partition.
 *
 * finally, here's a rough breakdown of the time spent in the 5 different phases of the algorithm.
 * we split the preparation phase into two; the "summation steps" are the three steps also defined above.
 *
 * ```txt
 *  8% - preparation 1 (input processing)
 * 12% - preparation 2 (sorting points in bucket order)
 * 65% - summation step 1 (bucket accumulation)
 * 15% - summation step 2 (bucket reduction)
 *  0% - summation step 3 (final sum over partitions)
 * ```
 *
 * you can find more details on each phase and reasoning about performance in the comments below!
 *
 * @param {Uint8Array[]} inputScalars `s_0, ..., s_(N-1)`
 * @param {InputPoint[]} inputPoints `G_0, ..., G_(N-1)`
 * @param {{c: number, c0: number}?} options optional msm parameters `c`, `c0` (this is only needed when trying out different parameters
 * than our well-optimized, hard-coded ones; see {@link cTable})
 */
function msmAffine(inputScalars, inputPoints, { c: c_, c0: c0_ } = {}) {
  let N = inputScalars.length;
  let n = log2(N);
  let c = n - 1;
  if (c < 1) c = 1;
  let c0 = c >> 1;
  [c, c0] = cTable[n] || [c, c0];
  // if parameters for c and c0 were passed in, use those instead
  if (c_) c = c_;
  if (c0_) c0 = c0_;

  let K = Math.ceil((scalarBitlength + 1) / c); // number of partitions
  let L = 2 ** (c - 1); // number of buckets per partition, -1 (we'll skip the 0 bucket, but will have them in the array at index 0 to simplify access)
  let doubleL = 2 * L;

  let sizeAffine2 = 2 * sizeAffine;
  let sizeAffine4 = 4 * sizeAffine;
  let pointPtr = getPointer(N * sizeAffine4);
  let sizeScalar2 = 2 * scalarSize;
  let scalarPtr = getPointerScalar(2 * N * sizeScalar2);

  /**
   * @type {(number)[][]}
   */
  let bucketCounts = Array(K);
  for (let k = 0; k < K; k++) {
    bucketCounts[k] = Array(L + 1);
    for (let l = 0; l <= L; l++) {
      bucketCounts[k][l] = 0;
    }
  }
  let scratch = getPointers(30);

  let maxBucketSize = 0;
  let nPairs = 0; // we need to allocate space for one pointer per addition pair

  getAndResetOpCounts();

  /**
   * Preparation phase 1
   * --------------------
   *
   * this phase is where we process inputs:
   * - store input points in wasm memory, in the format we need
   *   - writing the bytes to wasm memory takes ~2% of total runtime
   *   - we also turn point coordinates x,y to montgomery form;
   *     takes ~1% of runtime (= small potential savings for not using montgomery)
   * - compute & store negative, endo, and negative-endo points
   * - decompose input scalars as `s = s0 + s1*lambda` and store s0, s1 in wasm memory
   * - walk over the c-bit windows of each scalar, to
   *   - count the number of points for each bucket
   *   - count the total number of pairs to add in the first batch addition
   *
   * note: actual copying into buckets is done in the next phase!
   * here, we just use the scalar slices to count bucket sizes, as first step of a counting sort.
   *
   * ### Performance
   *
   * this phase takes ~8% of the total, roughly made up of
   *
   * 2% write scalars & points to wasm memory
   * 1% bucket counts
   * 1% turn coordinates to montgomery form
   * 0.5% endomorphism
   * 0.5% split scalars to slices
   * 0.2% other processing of points (negation, copying)
   * 0.1% GLV-decompose scalar
   *
   * it's hard to get perfect data from the profiler because this phase is a hodgepodge of so many different small pieces.
   * also, there is ~2.7% of unexplained runtime which is spent somewhere in the JS logic.
   * that said, most of the effort here, like writing to wasm memory and processing points, is necessitated
   * by the architecture and can't be significantly reduced.
   */
  for (
    let i = 0, point = pointPtr, scalar = scalarPtr;
    i < N;
    i++, point += sizeAffine4, scalar += sizeScalar2
  ) {
    let inputScalar = inputScalars[i];
    let inputPoint = inputPoints[i];

    /**
     * store point in n-limb format and convert to montgomery representation.
     * see {@link sizeField} for the memory layout.
     */
    let x = point;
    let y = point + sizeField;
    // this line writes inputPoint[0] to a `scratch` location and uses `fromPackedBytes`
    // to turn the packed layout of the input field element into our wider n-limb layout.
    // the output field element is stored at `x`
    writeBytes(scratch, x, inputPoint[0]);
    //
    writeBytes(scratch, y, inputPoint[1]);
    let isNonZero = Number(!inputPoint[2]);
    memoryBytes[point + 2 * sizeField] = isNonZero;
    // do one multiplication on each coordinate to bring it into montgomery form
    toMontgomery(x);
    toMontgomery(y);

    // -point, endo(point), -endo(point)
    // this just takes 1 field multiplication for the endomorphism, and 1 subtraction
    let negPoint = point + sizeAffine;
    let endoPoint = negPoint + sizeAffine;
    let negEndoPoint = endoPoint + sizeAffine;
    copy(negPoint, x);
    subtract(negPoint + sizeField, constants.p, y);
    memoryBytes[negPoint + 2 * sizeField] = isNonZero;
    endomorphism(endoPoint, point);
    memoryBytes[endoPoint + 2 * sizeField] = isNonZero;
    copy(negEndoPoint, endoPoint);
    copy(negEndoPoint + sizeField, negPoint + sizeField);
    memoryBytes[negEndoPoint + 2 * sizeField] = isNonZero;

    // decompose scalar from one 32-byte into two 16-byte chunks
    writeBytesScalar(scalar, inputScalar);
    decompose(scalar);

    // partition each 16-byte scalar into c-bit slices
    for (let k = 0, carry0 = 0, carry1 = 0; k < K; k++) {
      // compute kth slice from first half scalar
      let l = extractBitSlice(scalar, k * c, c) + carry0;

      if (l > L) {
        l = doubleL - l;
        carry0 = 1;
      } else {
        carry0 = 0;
      }
      if (l !== 0) {
        // if the slice is non-zero, increase bucket count
        let bucketSize = ++bucketCounts[k][l];
        if ((bucketSize & 1) === 0) nPairs++;
        if (bucketSize > maxBucketSize) maxBucketSize = bucketSize;
      }
      // compute kth slice from second half scalar
      // note: we repeat this code instead of merging both into a loop of size 2,
      // because the latter would imply creating a throw-away array of size two for the scalars.
      // creating such throw-away objects has a garbage collection cost
      l = extractBitSlice(scalar + scalarSize, k * c, c) + carry1;

      if (l > L) {
        l = doubleL - l;
        carry1 = 1;
      } else {
        carry1 = 0;
      }
      if (l !== 0) {
        // if the slice is non-zero, increase bucket count
        let bucketSize = ++bucketCounts[k][l];
        if ((bucketSize & 1) === 0) nPairs++;
        if (bucketSize > maxBucketSize) maxBucketSize = bucketSize;
      }
    }
  }
  /**
   * Preparation phase 2
   * -------------------
   *
   * this phase basically consists of the second and third loops of the _counting sort_ algorithm shown here:
   * https://en.wikipedia.org/wiki/Counting_sort#Pseudocode
   *
   * we actually do K of these counting sorts -- one for each partition.
   *
   * note that the first loop in that link -- counting bucket sizes -- was already performed above,
   * and we have the `counts` stored in {@link bucketCounts}.
   *
   * here's how other parts of the linked algorithm correspond to our code:
   * - array `input`: in our case, this is the array of (scalar, point) pairs created in phase 1.
   *   note: when we say "array" here, we mean a range of memory locations which implicitly form an array.
   * - the `key(...)` function for mapping `input` elements to integer "keys":
   *   in our case, this is the function that computes the (kth) scalar slice belonging to each (scalar, point),
   *   i.e. {@link extractBitSlice} which we used above (loop 1) and which is re-executed in loop 3
   * - array `output`: in our case, we have one output array for each k. it's implicitly represented by a starting
   *   pointer which, right below, is stored at `buckets[k][0]`. by incrementing the pointer by the size of an affine point,
   *   we get to the next point.
   *
   * for our purposes, we don't only need sorting -- we also need to keep track of the indices
   * where one bucket ends and the next one begins, to form correct addition pairs.
   * these bucket bounds are stored in {@link buckets}.
   *
   * ## Performance
   *
   * this phase needs ~12% of total runtime (for 2^16 input points).
   *
   * this time is dominated by 8.5% for the {@link copyAffine} invocation at the end of 'loop #3'.
   * it writes to an unpredictable memory location (randomly distributed bucket) in each iteration.
   *
   * unfortunately, copying points scales superlinearly with input size:
   * for 2^18 input points, this phase already takes ~14% of runtime.
   *
   * as far as we can tell, this is still preferable to any other solutions we are aware of.
   * solutions that avoid the copying / sorting step seem to incur plenty of time for both random reads and
   * random writes during the bucket accumulation step, and end up being much slower -- especially or large inputs.
   * the counting sort solution almost entirely avoids random reads, with the exception of
   * reading random buckets from the relatively small {@link bucketCounts} helper array.
   *
   * there is not much other stuff happening in this phase:
   * - 'loop #2' is negligible at < 0.1% of runtime.
   * - 1-2% spent on {@link bucketCounts} reads/writes
   * - 0.5% on {@link extractBitSlice}
   *
   * @type {number[][]}
   */
  let buckets = Array(K);
  for (let k = 0; k < K; k++) {
    buckets[k] = Array(L + 1);
    // the starting pointer for the array of points, in bucket order
    buckets[k][0] = getPointer(2 * N * sizeAffine);
  }
  /**
   * loop #2 of counting sort (for each k).
   * "integrate" bucket counts, to become start / end indices (i.e., bucket bounds).
   * while we're at it, we fill an array `buckets` with the same bucket bounds but in a
   * more convenient format -- as memory addresses.
   */
  for (let k = 0; k < K; k++) {
    let counts = bucketCounts[k];
    let running = 0;
    let bucketsK = buckets[k];
    let runningIndex = bucketsK[0];
    for (let l = 1; l <= L; l++) {
      let count = counts[l];
      counts[l] = running;
      running += count;
      runningIndex += count * sizeAffine;
      bucketsK[l] = runningIndex;
    }
  }
  /**
   * loop #3 of counting sort (for each k).
   * we loop over the input elements and re-compute in which bucket `l` they belong.
   * by retrieving counts[l], we find the output position where a point should be stored in.
   * at the beginning, counts[l] will be the 0 index of bucket l, but when we store a point we increment count[l]
   * so that the next point in this bucket is stored at the next position.
   *
   * all in all, the result of this sorting is that points form a contiguous array, one bucket after another
   * => this is fantastic for the batch additions in the next step
   */
  for (
    // we loop over implicit arrays of points & scalars by taking their starting pointers and incrementing by the size of one element
    // note: this time, we treat `G` and `endo(G)` as separate points, and iterate over 2N points.
    let i = 0, point = pointPtr, scalar = scalarPtr;
    i < 2 * N;
    i++, point += sizeAffine2, scalar += scalarSize
  ) {
    // a point `A` and it's negation `-A` are stored next to each other
    let negPoint = point + sizeAffine;
    let carry = 0;
    /**
     * recomputing the scalar slices here with {@link extractBitSlice} is faster than storing & retrieving them!
     */
    for (let k = 0; k < K; k++) {
      let l = extractBitSlice(scalar, k * c, c) + carry;
      if (l > L) {
        l = doubleL - l;
        carry = 1;
      } else {
        carry = 0;
      }
      if (l === 0) continue;
      // compute the memory address in the bucket array where we want to store our point
      let ptr0 = buckets[k][0];
      let l0 = bucketCounts[k][l]++; // update start index, so the next point in this bucket lands at one position higher
      let newPtr = ptr0 + l0 * sizeAffine; // this is where the point should be copied to
      let ptr = carry === 1 ? negPoint : point; // this is the point that should be copied
      // copy point to the bucket array -- expensive operation! (but it pays off)
      copyAffine(newPtr, ptr);
    }
  }

  let [G, gPtr] = getEmptyPointersInMemory(nPairs); // holds first summands
  let [H, hPtr] = getEmptyPointersInMemory(nPairs); // holds second summands
  let denom = getPointer(nPairs * sizeField);
  let tmp = getPointer(nPairs * sizeField);

  // batch-add buckets into their first point, in `maxBucketSize` iterations
  for (let m = 1; m < maxBucketSize; m *= 2) {
    let p = 0;
    let sizeAffineM = m * sizeAffine;
    let sizeAffine2M = 2 * m * sizeAffine;
    // walk over all buckets to identify point-pairs to add
    for (let k = 0; k < K; k++) {
      let bucketsK = buckets[k];
      let nextBucket = bucketsK[0];
      for (let l = 1; l <= L; l++) {
        let point = nextBucket;
        nextBucket = bucketsK[l];
        for (; point + sizeAffineM < nextBucket; point += sizeAffine2M) {
          G[p] = point;
          H[p] = point + sizeAffineM;
          p++;
        }
      }
    }
    nPairs = p;
    // now (G,H) represents a big array of independent additions, which we batch-add
    batchAddUnsafe(scratch[0], tmp, denom, gPtr, gPtr, hPtr, nPairs);
  }
  // we're done!!
  // buckets[k][l-1] now contains the bucket sum (for non-empty buckets)

  let [nMul1, nInv1] = getAndResetOpCounts();

  // second stage
  let partialSums = reduceBucketsAffine(scratch, buckets, { c, c0, K, L });

  let [nMul2, nInv2] = getAndResetOpCounts();

  // third stage -- compute final sum
  let finalSum = getPointer(sizeProjective);
  let k = K - 1;
  let partialSum = partialSums[k];
  copyProjective(finalSum, partialSum);
  k--;
  for (; k >= 0; k--) {
    for (let j = 0; j < c; j++) {
      doubleInPlaceProjective(scratch, finalSum);
    }
    let partialSum = partialSums[k];
    addAssignProjective(scratch, finalSum, partialSum);
  }

  // convert final sum back to affine point
  let result = toAffineOutput(scratch, finalSum);

  let [nMul3, nInv3] = getAndResetOpCounts();
  resetPointers();
  resetPointersScalar();
  let statistics = { nMul1, nMul2, nMul3, nInv: nInv1 + nInv2 + nInv3 };

  return { result, statistics };
}

/**
 * reducing buckets into one sum per partition, using only batch-affine additions & doublings
 *
 * @param {number[]} scratch
 * @param {number[][]} oldBuckets
 * @param {{c: number, K: number, L: number}}
 * @param {number} depth
 */
function reduceBucketsAffine(scratch, oldBuckets, { c, c0, K, L }) {
  // D = 1 is the standard algorithm, just batch-added over the K partitions
  // D > 1 means that we're doing D * K = n adds at a time
  // => more efficient than doing just K at a time, since we amortize the cost of the inversion better
  let depth = c - 1 - c0;
  let D = 2 ** depth;
  let n = D * K;
  let L0 = 2 ** c0; // == L/D

  // normalize the way buckets are stored -- we'll store intermediate running sums there
  // copy bucket sums into new contiguous pointers to improve memory access
  /** @type {number[][]} */
  let buckets = Array(K);
  for (let k = 0; k < K; k++) {
    let newBuckets = getZeroPointers(L + 1, sizeAffine);
    buckets[k] = newBuckets;
    let oldBucketsK = oldBuckets[k];
    let nextBucket = oldBucketsK[0];
    for (let l = 1; l <= L; l++) {
      let bucket = nextBucket;
      nextBucket = oldBucketsK[l];
      if (bucket === nextBucket) continue;
      let newBucket = newBuckets[l];
      copyAffine(newBucket, bucket);
    }
  }

  let [runningSums] = getEmptyPointersInMemory(n);
  let [nextBuckets] = getEmptyPointersInMemory(n);
  let [d] = getPointersInMemory(K * L, sizeField);
  let [tmp] = getPointersInMemory(K * L, sizeField);

  // linear part of running sum computation / sums of the form x_(d*L0 + L0) + x(d*L0 + (L0-1)) + ...x_(d*L0 + 1), for d=0,...,D-1
  for (let l = L0 - 1; l > 0; l--) {
    // collect buckets to add into running sums
    let p = 0;
    for (let k = 0; k < K; k++) {
      for (let d = 0; d < D; d++, p++) {
        runningSums[p] = buckets[k][d * L0 + l + 1];
        nextBuckets[p] = buckets[k][d * L0 + l];
      }
    }
    // add them; we add-assign the running sum to the next bucket and not the other way;
    // building up a list of intermediary partial sums at the pointers that were the buckets before
    batchAdd(scratch, tmp, d, nextBuckets, nextBuckets, runningSums, n);
  }

  // logarithmic part (i.e., logarithmic # of batchAdds / inversions; the # of EC adds is linear in K*D = K * 2^(c - c0))
  // adding x_(d*2*L0 + 1) += x_((d*2 + 1)*L0 + 1), d = 0,...,D/2-1,  x_(d*2(2*L0) + 1) += x_((d*2 + 1)*(2*L0) + 1), d = 0,...,D/4-1, ...
  // until x_(d*2*2**(depth-1)*L0 + 1) += x_((d*2 + 1)*2**(depth-1)*L0 + 1), d = 0,...,(D/2^depth - 1) = 0
  // <===> x_1 += x_(L/2 + 1)
  // iterate over L1 = 2^0*L0, 2^1*L0, ..., 2^(depth-1)*L0 (= L/2) and D1 = 2^(depth-1), 2^(depth-2), ..., 2^0
  // (no-op if 2^(depth-1) < 1 <===> depth = 0)
  let minorSums = runningSums;
  let majorSums = nextBuckets;
  for (let L1 = L0, D1 = D >> 1; D1 > 0; L1 <<= 1, D1 >>= 1) {
    let p = 0;
    for (let k = 0; k < K; k++) {
      for (let d = 0; d < D1; d++, p++) {
        minorSums[p] = buckets[k][(d * 2 + 1) * L1 + 1];
        majorSums[p] = buckets[k][d * 2 * L1 + 1];
      }
    }
    batchAdd(scratch, tmp, d, majorSums, majorSums, minorSums, p);
  }
  // second logarithmic step: repeated doubling of some buckets until they hold square areas to fill up the triangle
  // first, double x_(d*L0 + 1), d=1,...,D-1, c0 times, so they all hold 2^c0 * x_(d*L0 + 1)
  // (no-op if depth=0 / D=1 / c0=c)
  let p = 0;
  for (let k = 0; k < K; k++) {
    for (let d = 1; d < D; d++, p++) {
      minorSums[p] = buckets[k][d * L0 + 1];
    }
  }
  if (D > 1) {
    for (let j = 0; j < c0; j++) {
      batchDoubleInPlace(scratch, tmp, d, minorSums, p);
    }
  }
  // now, double successively smaller sets of buckets until the biggest is 2^(c-1) * x_(2^(c-1) + 1)
  // x_(d*L0 + 1), d=2,4,...,D-2 / d=4,8,...,D-4 / ... / d=D/2 = 2^(c - c0 - 1)
  // (no-op if depth = 0, 1)
  for (let L1 = 2 * L0, D1 = D >> 1; D1 > 1; L1 <<= 1, D1 >>= 1) {
    let p = 0;
    for (let k = 0; k < K; k++) {
      for (let d = 1; d < D1; d++, p++) {
        majorSums[p] = buckets[k][d * L1 + 1];
      }
    }
    batchDoubleInPlace(scratch, tmp, d, majorSums, p);
  }

  // alright! now our buckets precisely fill up the big triangle
  // => sum them all in a big addition tree
  // we always batchAdd a list of pairs into the first element of each pair
  // round 0: (1,2), (3,4), (5,6), ..., (L-1, L);
  //      === (l,l+1) for l=1; l<L; i+=2
  // round 1: (l,l+2) for l=1; l<L; i+=4
  // round j: let m=2^j; (l,l+m) for l=1; l<L; l+=2*m
  // in the last round we want 1 pair (1, 1 + m=2^(c-1)), so we want m < 2**c = L

  let [G] = getEmptyPointersInMemory(K * L);
  let [H] = getEmptyPointersInMemory(K * L);

  for (let m = 1; m < L; m *= 2) {
    p = 0;
    for (let k = 0; k < K; k++) {
      for (let l = 1; l < L; l += 2 * m, p++) {
        G[p] = buckets[k][l];
        H[p] = buckets[k][l + m];
      }
    }
    batchAdd(scratch, tmp, d, G, G, H, p);
  }

  // finally, return the output sum of each partition as a projective point
  let partialSums = getZeroPointers(K, sizeProjective);
  for (let k = 0; k < K; k++) {
    if (isZeroAffine(buckets[k][1])) continue;
    copyAffineToProjectiveNonZero(partialSums[k], buckets[k][1]);
  }
  return partialSums;
}

/**
 * Given points G0,...,G(n-1) and H0,...,H(n-1), compute
 *
 * Si = Gi + Hi, i=0,...,n-1
 *
 * @param {number[]} scratch
 * @param {Uint32Array} tmp pointers of length n
 * @param {Uint32Array} d pointers of length n
 * @param {Uint32Array} S
 * @param {Uint32Array} G
 * @param {Uint32Array} H
 * @param {number} n
 */
function batchAdd(scratch, tmp, d, S, G, H, n) {
  let iAdd = Array(n);
  let iDouble = Array(n);
  let iBoth = Array(n);
  let nAdd = 0;
  let nDouble = 0;
  let nBoth = 0;

  for (let i = 0; i < n; i++) {
    // check G, H for zero
    if (isZeroAffine(G[i])) {
      copyAffine(S[i], H[i]);
      continue;
    }
    if (isZeroAffine(H[i])) {
      if (S[i] !== G[i]) copyAffine(S[i], G[i]);
      continue;
    }
    if (isEqual(G[i], H[i])) {
      // here, we handle the x1 === x2 case, in which case (x2 - x1) shouldn't be part of batch inversion
      // => batch-affine doubling G[p] in-place for the y1 === y2 cases, setting G[p] zero for y1 === -y2
      let y = G[i] + sizeField;
      if (!isEqual(y, H[i] + sizeField)) {
        setIsNonZeroAffine(S[i], false);
        continue;
      }
      add(tmp[nBoth], y, y); // TODO: efficient doubling
      iDouble[nDouble] = i;
      iBoth[i] = nBoth;
      nDouble++, nBoth++;
    } else {
      // typical case, where x1 !== x2 and we add the points
      subtractPositive(tmp[nBoth], H[i], G[i]);
      iAdd[nAdd] = i;
      iBoth[i] = nBoth;
      nAdd++, nBoth++;
    }
  }
  batchInverse(scratch[0], d[0], tmp[0], nBoth);
  for (let j = 0; j < nAdd; j++) {
    let i = iAdd[j];
    addAffine(scratch[0], S[i], G[i], H[i], d[iBoth[i]]);
  }
  for (let j = 0; j < nDouble; j++) {
    let i = iDouble[j];
    doubleAffine(scratch, S[i], G[i], d[iBoth[i]]);
  }
}

/**
 * Given points G0,...,G(n-1), compute
 *
 * Gi *= 2, i=0,...,n-1
 *
 * @param {number[]} scratch
 * @param {Uint32Array} tmp pointers of length n
 * @param {Uint32Array} d pointers of length n
 * @param {Uint32Array} G
 * @param {number} n
 */
function batchDoubleInPlace(scratch, tmp, d, G, n) {
  // maybe every curve point should have space for one extra field element so we have those tmp pointers ready?

  // check G for zero
  let G1 = Array(n);
  let n1 = 0;
  for (let i = 0; i < n; i++) {
    if (isZeroAffine(G[i])) continue;
    G1[n1] = G[i];
    // TODO: confirm that y === 0 can't happen, either bc 0 === x^3 + 4 has no solutions in the field or bc the (x,0) aren't in G1
    let y = G1[n1] + sizeField;
    add(tmp[n1], y, y); // TODO: efficient doubling
    n1++;
  }
  batchInverse(scratch[0], d[0], tmp[0], n1);
  for (let i = 0; i < n1; i++) {
    doubleAffine(scratch, G1[i], G1[i], d[i]);
  }
}

/**
 * converts projective point back to affine, and into the `InputPoint` format expected from the MSM
 *
 * @param {number[]} scratchSpace
 * @param {number} point projective representation
 * @returns {InputPoint}
 */
function toAffineOutput([zinv, ...scratchSpace], point) {
  if (isZeroProjective(point)) {
    return [
      new Uint8Array(packedSizeBytes),
      new Uint8Array(packedSizeBytes),
      true,
    ];
  }
  let [x, y, z] = projectiveCoords(point);
  // return x/z, y/z
  inverse(scratchSpace[0], zinv, z);
  multiply(x, x, zinv);
  multiply(y, y, zinv);
  fromMontgomery(x);
  fromMontgomery(y);
  return [readBytes(scratchSpace, x), readBytes(scratchSpace, y), false];
}
