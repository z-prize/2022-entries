/* From Filecoin
 *
 * Same multiexp algorithm used in Bellman, with some modifications.
 * https://github.com/zkcrypto/bellman/blob/10c5010fd9c2ca69442dc9775ea271e286e776d8/src/multiexp.rs#L174
 * The CPU version of multiexp parallelism is done by dividing the exponent
 * values into smaller windows, and then applying a sequence of rounds to each
 * window. The GPU kernel not only assigns a thread to each window but also
 * divides the bases into several groups which highly increases the number of
 * threads running in parallel for calculating a multiexp instance.
 */

KERNEL void POINT_bellman_multiexp(
    GLOBAL POINT_affine *bases,
    GLOBAL POINT_projective *buckets,
    GLOBAL POINT_projective *results,
    GLOBAL EXPONENT *exps,
    uint base_start,
    uint n,
    uint num_groups,
    uint num_windows,
    uint window_size) {

  // We have `num_windows` * `num_groups` threads per multiexp.
  uint gid = GET_GLOBAL_ID();
  if(gid >= num_windows * num_groups) return;

  // We have (2^window_size - 1) buckets.
  uint bucket_len = ((1 << window_size) - 1);

  // Each thread has its own set of buckets in global memory.
  buckets += bucket_len * gid;
  for(uint i = 0; i < bucket_len; i++) buckets[i] = POINT_ZERO;

  bases = bases + base_start;

  uint len = (uint)ceil(n / (float)num_groups); // Num of elements in each group

  // This thread runs the multiexp algorithm on elements from `nstart` to `nened`
  // on the window [`bits`, `bits` + `w`)
  uint nstart = len * (gid / num_windows);
  uint nend = min(nstart + len, n);
  uint bits = (gid % num_windows) * window_size;
  ushort w = min((ushort)window_size, (ushort)(EXPONENT_BITS - bits));

  POINT_projective res = POINT_ZERO;
  for(uint i = nstart; i < nend; i++) {
    EXPONENT exp = exps[i];
    uint ind = EXPONENT_get_bits(exp, bits, w);

    // O_o, weird optimization, having a single special case makes it
    // tremendously faster!
    // 511 is chosen because it's half of the maximum bucket len, but
    // any other number works... Bigger indices seems to be better...

    /* if(ind == 511) buckets[510] = POINT_add_mixed(buckets[510], bases[i]); */
    /* else if(ind--) buckets[ind] = POINT_add_mixed(buckets[ind], bases[i]); */

    if(ind--) buckets[ind] = POINT_add_mixed(buckets[ind], bases[i]);
  }

  // Summation by parts
  // e.g. 3a + 2b + 1c = a +
  //                    (a) + b +
  //                    ((a) + b) + c
  POINT_projective acc = POINT_ZERO;
  for(int j = bucket_len - 1; j >= 0; j--) {
    acc = POINT_add(acc, buckets[j]);
    res = POINT_add(res, acc);
  }

  results[gid] = res;
}

// to save cpu time, use gpu to sum all groups in a same window
// one local work group calculate one window

// REMINDER: local work group size <= lsize <= 32
KERNEL void POINT_group_acc(
    GLOBAL POINT_projective *results,
    uint num_groups,
    uint num_windows,
    uint lsize) {

  const uint groupid = GET_GROUP_ID();
  if(groupid >= num_windows) return;

  const uint gid = GET_GLOBAL_ID();
  const uint lid = GET_LOCAL_ID();

  if (lid >= 32) return;

  LOCAL POINT_projective l[32];
  l[lid] = POINT_ZERO;
  
  uint len = (uint)ceil(num_groups / (float)lsize); // Num of elements in each group
  uint nstart = len * lid;
  uint nend = min(nstart + len, num_groups);

  for (uint i=nstart; i < nend; i++) {
    uint idx = i * num_windows + groupid;
    l[lid] = POINT_add(results[idx], l[lid]);
  }

  BARRIER_LOCAL();

  if (lid == 0) {
    for (uint i=1; i < lsize; i++) {
      l[0] = POINT_add(l[0], l[i]);
    }
    results[groupid] = l[0];
  }
}

// This function calculates c = a + b * x
// `x` is a scalar, `a` is affine point vector, `b * x` is projective point vector
// a very inefficient implementation
KERNEL void POINT_scalar_mult(
    GLOBAL POINT_affine *bases_1,
    GLOBAL POINT_projective *bases_2,
    GLOBAL POINT_projective *results,
    EXPONENT exp_origin,
    uint n) {

  // We have `n` threads per scalar mult.
  const uint gid = GET_GLOBAL_ID();

  if(gid >= n) return;

  EXPONENT exp = EXPONENT_unmont(exp_origin);

  POINT_projective res = POINT_add_mixed(POINT_ZERO, bases_1[gid]);
  POINT_projective acc = bases_2[gid];

  for (int i=EXPONENT_BITS-1; i>=0; i--) {
    bool bit = EXPONENT_get_bit(exp, i);

    if (bit) {
      res = POINT_add(res, acc);
    }
    acc = POINT_double(acc);
  }
  results[gid] = res;
}

KERNEL void POINT_affine_to_projective(
    GLOBAL POINT_affine *g1,
    GLOBAL POINT_projective *g1_result,
    uint n) {

  const uint gid = GET_GLOBAL_ID();

  if(gid >= n) return;

  g1_result[gid] = POINT_add_mixed(POINT_ZERO, g1[gid]);
}

KERNEL void POINT_bellman_multiexp_precalc(
    GLOBAL POINT_affine *bases_precalc,
    GLOBAL POINT_projective *results,
    GLOBAL EXPONENT *exps,
    uint start,
    uint n,
    uint num_groups,
    uint num_windows,
    uint window_size) {

  // We have `num_windows` * `num_groups` threads per multiexp.
  const uint gid = GET_GLOBAL_ID();
  if(gid >= num_windows * num_groups) return;

  uint bucket_len = 1 << window_size;

  bases_precalc += start * bucket_len;

  uint len = (uint)ceil(n / (float)num_groups); // Num of elements in each group

  // This thread runs the multiexp algorithm on elements from `nstart` to `nened`
  // on the window [`bits`, `bits` + `w`)
  uint window_id = gid % num_windows;
  uint group_id = gid / num_windows;
  uint nstart = len * group_id;
  uint nend = min(nstart + len, n);
  uint bits = window_id * window_size;
  ushort w = min((ushort)window_size, (ushort)(EXPONENT_BITS - bits));

  POINT_projective acc = POINT_ZERO;

  for(uint i = nstart; i < nend; i++) {
    EXPONENT exp = exps[i];
    uint ind = EXPONENT_get_bits(exp, bits, w);
    uint pos = i * bucket_len + ind;

    if (ind) {
	acc = POINT_add_mixed(acc, bases_precalc[pos]);
    }
  }

  // reorg results layout to
  // [ |              window 0           | |             window 1         | |           window n        | ]
  // [ | group_0, group_1, ...           | | group_0, group_1, ...        | | group_0, group_1, ...     | ]
  results[num_groups * window_id + group_id] = acc;
}

KERNEL void POINT_multiexp_group_acc_iter(
    GLOBAL POINT_projective *src,
    GLOBAL POINT_projective *dst,
    uint total, // total represents total threads
    uint n, // n represents the number of total groups
    uint num_windows,
    uint max_p) {

  const uint gid = GET_GLOBAL_ID();

  if (gid >= total) return;

  POINT_projective acc = POINT_ZERO;

  // `window_id` indicates the id of window this thread is processing
  uint window_id = gid % num_windows;

  // `lid` indicates the id of thread processing the same window
  uint lid = gid / num_windows;

  // originally we have `n` groups
  // let's reduce them into `ceil(n / max_p)`
  uint num_groups = (uint)ceil(n / (float)max_p);
  uint nstart = lid * max_p;
  uint nend = min(nstart + max_p, n);

  // one thread only takes care of max_p points at most
  for (uint i=nstart; i<nend; i++) {
    uint src_pos = window_id * n + i;
    acc = POINT_add(acc, src[src_pos]);
  }

  uint dst_pos = window_id * num_groups + lid;
  dst[dst_pos] = acc;
}

// for extended twisted edwards curves
KERNEL void POINT_multiexp_ed_precalc(
    GLOBAL POINT_ed_affine *bases_precalc,
    GLOBAL POINT_ed_projective *results,
    GLOBAL EXPONENT *exps,
    uint start,
    uint n,
    uint num_groups,
    uint num_windows,
    uint window_size) {

  // We have `num_windows` * `num_groups` threads per multiexp.
  const uint gid = GET_GLOBAL_ID();
  if(gid >= num_windows * num_groups) return;

  uint bucket_len = 1 << window_size;

  bases_precalc += start * bucket_len;

  uint len = (uint)ceil(n / (float)num_groups); // Num of elements in each group

  // This thread runs the multiexp algorithm on elements from `nstart` to `nened`
  // on the window [`bits`, `bits` + `w`)
  uint window_id = gid % num_windows;
  uint group_id = gid / num_windows;
  uint nstart = len * group_id;
  uint nend = min(nstart + len, n);
  uint bits = window_id * window_size;
  ushort w = min((ushort)window_size, (ushort)(EXPONENT_BITS - bits));

  POINT_ed_projective acc = POINT_ED_ZERO;

  for(uint i = nstart; i < nend; i++) {
    EXPONENT exp = exps[i];
    uint ind = EXPONENT_get_bits(exp, bits, w);
    uint pos = i * bucket_len + ind;

    if (ind) {
	acc = POINT_ed_add_mixed_dedicated(acc, bases_precalc[pos]);
    }
  }

  // reorg results layout to
  // [ |              window 0           | |             window 1         | |           window n        | ]
  // [ | group_0, group_1, ...           | | group_0, group_1, ...        | | group_0, group_1, ...     | ]
  results[num_groups * window_id + group_id] = acc;
}

KERNEL void POINT_multiexp_ed_group_acc_iter(
    GLOBAL POINT_ed_projective *src,
    GLOBAL POINT_ed_projective *dst,
    uint total, // total represents total threads
    uint n, // n represents the number of total groups
    uint num_windows,
    uint max_p) {

  const uint gid = GET_GLOBAL_ID();

  if (gid >= total) return;

  POINT_ed_projective acc = POINT_ED_ZERO;

  // `window_id` indicates the id of window this thread is processing
  uint window_id = gid % num_windows;

  // `lid` indicates the id of thread processing the same window
  uint lid = gid / num_windows;

  // originally we have `n` groups
  // let's reduce them into `ceil(n / max_p)`
  uint num_groups = (uint)ceil(n / (float)max_p);
  uint nstart = lid * max_p;
  uint nend = min(nstart + max_p, n);

  uint dst_pos = window_id * num_groups + lid;

  // one thread only takes care of max_p points at most
  for (uint i=nstart; i<nend; i++) {
    uint src_pos = window_id * n + i;
    acc = POINT_ed_add(acc, src[src_pos]);
  }

  dst[dst_pos] = acc;
}

// for extended twisted edwards curves
KERNEL void POINT_multiexp_ed_neg_one_a_precalc(
    GLOBAL POINT_ed_affine *bases_precalc,
    GLOBAL POINT_ed_projective *results,
    GLOBAL EXPONENT *exps,
    uint start,
    uint n,
    uint num_groups,
    uint num_windows,
    uint window_size) {

  // We have `num_windows` * `num_groups` threads per multiexp.
  const uint gid = GET_GLOBAL_ID();
  if(gid >= num_windows * num_groups) return;

  uint bucket_len = 1 << window_size;

  bases_precalc += start * bucket_len;

  uint len = (uint)ceil(n / (float)num_groups); // Num of elements in each group

  // This thread runs the multiexp algorithm on elements from `nstart` to `nened`
  // on the window [`bits`, `bits` + `w`)
  uint window_id = gid % num_windows;
  uint group_id = gid / num_windows;
  uint nstart = len * group_id;
  uint nend = min(nstart + len, n);
  uint bits = window_id * window_size;
  ushort w = min((ushort)window_size, (ushort)(EXPONENT_BITS - bits));

  POINT_ed_projective acc = POINT_ED_ZERO;

  for(uint i = nstart; i < nend; i++) {
    EXPONENT exp = exps[i];
    uint ind = EXPONENT_get_bits(exp, bits, w);
    uint pos = i * bucket_len + ind - 1;

    if (ind) {
	acc = POINT_ed_neg_one_a_add_mixed_dedicated(acc, bases_precalc[pos]);
    }
  }

  // reorg results layout to
  // [ |              window 0           | |             window 1         | |           window n        | ]
  // [ | group_0, group_1, ...           | | group_0, group_1, ...        | | group_0, group_1, ...     | ]
  results[num_groups * window_id + group_id] = acc;
}

// 2-NAF to accelerate MSM
KERNEL void POINT_multiexp_ed_neg_one_a_precalc_naf(
    GLOBAL POINT_ed_affine *bases_precalc,
    GLOBAL POINT_ed_projective *results,
    GLOBAL EXPONENT *exps,
    uint start,
    uint n,
    uint num_groups,
    uint num_windows,
    uint window_size) {

  // We have `num_windows` * `num_groups` threads per multiexp.
  const uint gid = GET_GLOBAL_ID();
  if(gid >= num_windows * num_groups) return;

  int bucket_len = 1 << (window_size-1);

  bases_precalc += (uint)(start * bucket_len);

  uint len = (uint)ceil(n / (float)num_groups); // Num of elements in each group

  // This thread runs the multiexp algorithm on elements from `nstart` to `nened`
  // on the window [`bits`, `bits` + `w`)
  uint window_id = gid % num_windows;
  uint group_id = gid / num_windows;
  uint nstart = len * group_id;
  uint nend = min(nstart + len, n);
  uint bits = window_id * window_size;
  ushort w = min((ushort)window_size, (ushort)(EXPONENT_BITS - bits));

  POINT_ed_projective acc = POINT_ED_ZERO;

  for(uint i = nstart; i < nend; i++) {
    EXPONENT exp = exps[i];
    int ind = (int)(EXPONENT_get_bits(exp, bits, w));
    int pos = abs(bucket_len - ind);

    if (pos--) {
      acc = POINT_ed_neg_one_a_add_mixed_dedicated_naf(acc, bases_precalc[pos + i * bucket_len], (ind < bucket_len));
    }
  }

  // reorg results layout to
  // [ |              window 0           | |             window 1         | |           window n        | ]
  // [ | group_0, group_1, ...           | | group_0, group_1, ...        | | group_0, group_1, ...     | ]
  results[num_groups * window_id + group_id] = acc;
}

KERNEL void POINT_multiexp_ed_neg_one_a_group_acc_iter(
    GLOBAL POINT_ed_projective *src,
    GLOBAL POINT_ed_projective *dst,
    uint total, // total represents total threads
    uint n, // n represents the number of total groups
    uint num_windows,
    uint max_p) {

  const uint gid = GET_GLOBAL_ID();

  if (gid >= total) return;

  // `window_id` indicates the id of window this thread is processing
  uint window_id = gid % num_windows;

  // `lid` indicates the id of thread processing the same window
  uint lid = gid / num_windows;

  // originally we have `n` groups
  // let's reduce them into `ceil(n / max_p)`
  uint num_groups = (uint)ceil(n / (float)max_p);
  uint nstart = lid * max_p;

  POINT_ed_projective acc = src[window_id * n + nstart];

  uint nend = min(nstart + max_p, n);

  uint dst_pos = window_id * num_groups + lid;

  // one thread only takes care of max_p points at most
  for (uint i=nstart+1; i<nend; i++) {
    uint src_pos = window_id * n + i;
    acc = POINT_ed_neg_one_a_add(acc, src[src_pos]);
  }

  dst[dst_pos] = acc;
}
