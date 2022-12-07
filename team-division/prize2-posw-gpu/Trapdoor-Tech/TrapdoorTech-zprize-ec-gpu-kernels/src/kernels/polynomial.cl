/*
 * To put most of the polynomial arithmetics onto GPU
 *
 */

KERNEL void FIELD_poly_copy_from_to(
    GLOBAL FIELD *src,
    GLOBAL FIELD *dst,
    uint n
    ) {

  const uint gid = GET_GLOBAL_ID();

  if (gid >= n) return;

  dst[gid] = src[gid];
}

KERNEL void FIELD_poly_copy_from_offset_to(
    GLOBAL FIELD *src,
    GLOBAL FIELD *dst,
    uint offset,
    uint n
    ) {

  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;

  GLOBAL FIELD *src_offset = src;
  src_offset += offset;

  dst[gid] = src_offset[gid];
}

KERNEL void FIELD_poly_copy_from_to_offset(
    GLOBAL FIELD *src,
    GLOBAL FIELD *dst,
    uint offset,
    uint n
    ) {

  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;

  GLOBAL FIELD *dst_offset = dst;
  dst_offset += offset;

  dst_offset[gid] = src[gid];
}

KERNEL void FIELD_poly_all_zeroes(
    GLOBAL FIELD *poly,
    uint n
    ) {

  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;

  poly[gid] = FIELD_ZERO;
}

KERNEL void FIELD_poly_set_fe(
    GLOBAL FIELD *poly,
    FIELD fe,
    uint n
    ) {

  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;

  poly[gid] = fe;
}

KERNEL void FIELD_poly_add_at_offset(
    GLOBAL FIELD *poly,
    FIELD fr,
    uint offset
    ) {

  const uint gid = GET_GLOBAL_ID();
  if (gid >= 1) return;

  poly[offset] = FIELD_add(poly[offset], fr);
}


KERNEL void FIELD_poly_setup_vanishing(
    GLOBAL FIELD *poly,
    FIELD fr,
    uint idx,
    uint factor,
    uint n
    ) {

  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;

  uint m = gid % factor;
  if (m == idx) {
    poly[gid] = fr;
  }
}

KERNEL void FIELD_poly_setup_l0(
    GLOBAL FIELD *poly,
    uint n
    ) {

  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;

  if (gid == 0) {
    poly[gid] = FIELD_ONE;
  } else {
    poly[gid] = FIELD_ZERO;
  }
}

KERNEL void FIELD_poly_negate(
    GLOBAL FIELD *poly,
    uint n
    ) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;

  poly[gid] = FIELD_sub(FIELD_ZERO, poly[gid]);
}

KERNEL void FIELD_poly_square(
    GLOBAL FIELD *poly,
    uint n
    ) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;

  FIELD tmp = poly[gid];
  poly[gid] = FIELD_mul(tmp, tmp);
}

KERNEL void FIELD_poly_scale(
    GLOBAL FIELD *poly,
    FIELD scale,
    uint n
    ) {
  const uint gid = GET_GLOBAL_ID();

  if (gid >= n) return;

  poly[gid] = FIELD_mul(poly[gid], scale);
}

KERNEL void FIELD_poly_add_constant(
    GLOBAL FIELD *poly,
    FIELD c,
    uint n
    ) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;

  poly[gid] = FIELD_add(poly[gid], c);
}

KERNEL void FIELD_poly_sub_constant(
    GLOBAL FIELD *poly,
    FIELD c,
    uint n
    ) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;

  poly[gid] = FIELD_sub(poly[gid], c);
}

KERNEL void FIELD_poly_add_assign(
    GLOBAL FIELD *poly_1,
    GLOBAL FIELD *poly_2,
    uint n
    ) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;

  poly_1[gid] = FIELD_add(poly_1[gid], poly_2[gid]);
}

KERNEL void FIELD_poly_add_assign_scaled(
    GLOBAL FIELD *poly_1,
    GLOBAL FIELD *poly_2,
    FIELD scale,
    uint n
    ) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;

  FIELD tmp = FIELD_mul(poly_2[gid], scale);
  poly_1[gid] = FIELD_add(poly_1[gid], tmp);
}

KERNEL void FIELD_poly_sub_assign(
    GLOBAL FIELD *poly_1,
    GLOBAL FIELD *poly_2,
    uint n
    ) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;

  poly_1[gid] = FIELD_sub(poly_1[gid], poly_2[gid]);
}

KERNEL void FIELD_poly_sub_assign_scaled(
    GLOBAL FIELD *poly_1,
    GLOBAL FIELD *poly_2,
    FIELD scale,
    uint n
    ) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;

  FIELD tmp = FIELD_mul(poly_2[gid], scale);
  poly_1[gid] = FIELD_sub(poly_1[gid], tmp);
}

KERNEL void FIELD_poly_mul_assign(
    GLOBAL FIELD *poly_1,
    GLOBAL FIELD *poly_2,
    uint n
    ) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;

  poly_1[gid] = FIELD_mul(poly_1[gid], poly_2[gid]);
}

/*
 * distribute powers
 */
KERNEL void FIELD_poly_distribute_powers(
    GLOBAL FIELD *poly_1,
    GLOBAL FIELD *powers,
    FIELD g,
    uint offset,
    uint chunk_size,
    uint n
    ) {
  const uint gid = GET_GLOBAL_ID();
  if (gid * chunk_size >= n) return;

  uint start = gid * chunk_size;
  uint end = min(start + chunk_size, n);
  
  FIELD lpower = FIELD_pow_lookup(powers, start + offset);

  for (uint i=start; i<end; i++) {
    poly_1[i] = FIELD_mul(poly_1[i], lpower);

    lpower = FIELD_mul(g, lpower);
  }
}

KERNEL void FIELD_poly_generate_powers(
    GLOBAL FIELD *poly_1,
    GLOBAL FIELD *powers,
    uint n
    ) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;

  poly_1[gid] = FIELD_pow_lookup(powers, gid);
}

/*
 * evaluate
 */

KERNEL void FIELD_poly_evaluate_at(
    GLOBAL FIELD *poly_1,
    GLOBAL FIELD *powers,
    GLOBAL FIELD *result,
    FIELD g,
    uint chunk_size,
    uint n
    ) {
  const uint gid = GET_GLOBAL_ID();
  result[gid] = FIELD_ZERO;

  if (gid * chunk_size >= n) return;

  uint start = gid * chunk_size;
  uint end = min(start + chunk_size, n);
  
  FIELD lpower = FIELD_pow_lookup(powers, start);
  FIELD accumulator = FIELD_ZERO;
  FIELD tmp = FIELD_ZERO;

  for (uint i=start; i<end; i++) {
    tmp = FIELD_mul(poly_1[i], lpower);
    accumulator = FIELD_add(accumulator, tmp);
    lpower = FIELD_mul(g, lpower);
  }

  result[gid] = accumulator;
}

/*
 * for nearly inplace batch inversion, saving GPU buffers
 * use `buckets` array to save temporary result
 * each `gid` thread handle with `chunk_size` elements
 * saving temporary multiplication acc into `buckets`
 *
 * while doing inversion_part_2, we can quickly (get up to `i` elements acc) in this chunk via buckets and a small loop
 */

KERNEL void FIELD_poly_batch_inversion_part_1(
    GLOBAL FIELD *poly_1,
    GLOBAL FIELD *result,
    GLOBAL FIELD *buckets,
    uint buckets_num,
    uint chunk_size,
    uint n
    ) {
  uint gid = GET_GLOBAL_ID();

  result[gid] = FIELD_ONE;

  if (gid * chunk_size >= n) return;

  // offset for gid
  buckets = buckets + gid * buckets_num;
  uint bucket_size = (uint)ceil(chunk_size/(float)buckets_num);

  uint start = gid * chunk_size;
  uint end = min(start + chunk_size, n);
  uint bucket_idx = 0;
  
  FIELD acc = FIELD_ONE;

  for (uint i=start; i<end; i+=bucket_size) {
    uint idx_end = min(i+bucket_size, end);

    for (uint j=i; j<idx_end; j++) {
      acc = FIELD_mul(acc, poly_1[j]);
    }

    buckets[bucket_idx] = acc;
    bucket_idx++;
  }

  result[gid] = acc;
}

KERNEL void FIELD_poly_batch_inversion_part_2(
    GLOBAL FIELD *poly_1,
    GLOBAL FIELD *result,
    GLOBAL FIELD *buckets,
    uint buckets_num,
    uint chunk_size,
    uint n
    ) {
  uint gid = GET_GLOBAL_ID();

  if (gid * chunk_size >= n) return;

  uint start = gid * chunk_size;
  uint end = min(start + chunk_size, n);

  // offset for gid
  buckets = buckets + gid * buckets_num;
  uint bucket_size = (uint)ceil(chunk_size/(float)buckets_num);

  FIELD acc = result[gid];
  FIELD bucket_acc = FIELD_ONE;

  for (uint i=(end - 1); i>start; i--) {
    // get bucket index
    uint bucket_idx = (uint)((i - start)/(float)bucket_size);

    // get previous bucket
    if (bucket_idx > 0) {
      bucket_acc = buckets[bucket_idx-1];
    } else {
      bucket_acc = FIELD_ONE;
    }

    // acc_i = (previous bucket) * (up to i elements in this bucket)
    for (uint j=bucket_idx * bucket_size + gid * chunk_size; j<i; j++)
      bucket_acc = FIELD_mul(bucket_acc, poly_1[j]);

    FIELD tmp = poly_1[i];
    poly_1[i] = FIELD_mul(acc, bucket_acc);
    acc = FIELD_mul(acc, tmp);
  }

  poly_1[start] = acc;
}

/*
 * for grand_product
 */

KERNEL void FIELD_poly_grand_product_part_1(
    GLOBAL FIELD *poly_1,
    GLOBAL FIELD *result,
    uint chunk_size,
    uint n
    ) {
  uint gid = GET_GLOBAL_ID();
  result[gid] = FIELD_ONE;
  
  if (gid * chunk_size >= n) return;

  uint start = gid * chunk_size;
  uint end = min(start + chunk_size, n);
  
  FIELD acc = FIELD_ONE;

  for (uint i=start; i<end; i++) {
    acc = FIELD_mul(acc, poly_1[i]);
    poly_1[i] = acc;
  }

  result[gid] = acc;
}

KERNEL void FIELD_poly_grand_product_part_2(
    GLOBAL FIELD *poly_1,
    GLOBAL FIELD *result,
    uint chunk_size,
    uint n
    ) {
  uint gid = GET_GLOBAL_ID();

  if (gid == 0) return;
  
  uint start = gid * chunk_size;
  uint end = min(start + chunk_size, n);
 
  FIELD acc = result[gid - 1];

  for (uint i=start; i<end; i++) {
    poly_1[i] = FIELD_mul(acc, poly_1[i]);
  }
}

/*
 * for grand_sum
 */

KERNEL void FIELD_poly_grand_sum_part_1(
    GLOBAL FIELD *poly_1,
    GLOBAL FIELD *poly_2,
    GLOBAL FIELD *result,
    uint chunk_size,
    uint n
    ) {
  uint gid = GET_GLOBAL_ID();
  result[gid] = FIELD_ZERO;
  
  if (gid * chunk_size >= n) return;

  uint start = gid * chunk_size;
  uint end = min(start + chunk_size, n);
  
  FIELD acc = FIELD_ZERO;

  for (uint i=start; i<end; i++) {
    acc = FIELD_add(acc, poly_1[i]);
    poly_2[i] = acc;
  }

  result[gid] = acc;
}

KERNEL void FIELD_poly_grand_sum_part_2(
    GLOBAL FIELD *poly_1,
    GLOBAL FIELD *poly_2,
    GLOBAL FIELD *result,
    uint chunk_size,
    uint n
    ) {
  uint gid = GET_GLOBAL_ID();

  if (gid == 0) return;
  
  uint start = gid * chunk_size;
  uint end = min(start + chunk_size, n);
 
  FIELD acc = result[gid - 1];

  for (uint i=start; i<end; i++) {
    poly_2[i] = FIELD_add(acc, poly_2[i]);
  }
}

KERNEL void FIELD_poly_shift(
    GLOBAL FIELD *poly_1,
    GLOBAL FIELD *poly_2,
    uint shift,
    uint n
    ) {
  uint gid = GET_GLOBAL_ID();

  if (gid >= n) return;

  long idx = (long)gid + (long)shift;
  if (idx >= n)
    idx -= n;

  poly_2[gid] = poly_1[idx];
}

KERNEL void FIELD_poly_mont(
    GLOBAL FIELD *poly,
    uint n
    ) {
  uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;

  poly[gid] = FIELD_mont(poly[gid]);
}

KERNEL void FIELD_poly_unmont(
    GLOBAL FIELD *poly,
    uint n
    ) {
  uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;

  poly[gid] = FIELD_unmont(poly[gid]);
}

KERNEL void FIELD_poly_shrink_domain(
    GLOBAL FIELD *large_domain_poly,
    GLOBAL FIELD *small_domain_poly,
    uint factor,
    uint n				// small domain poly length
    ) {
  uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;

  small_domain_poly[gid] = large_domain_poly[gid * factor];
}
