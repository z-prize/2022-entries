DEVICE uint bitreverse(uint n, uint bits) {
  uint r = 0;
  for(int i = 0; i < bits; i++) {
    r = (r << 1) | (n & 1);
    n >>= 1;
  }
  return r;
}

/// Multiplies all of the elements by `field`
KERNEL void FIELD_mul_by_field(GLOBAL FIELD* elements,
                        uint n,
                        FIELD field) {
  uint gid = GET_GLOBAL_ID();
  elements[gid] = FIELD_mul(elements[gid], field);
}

KERNEL void FIELD_arrange_res(GLOBAL FIELD* src,
                          GLOBAL FIELD* dst,
                          uint lgn) {
  uint gid = GET_GLOBAL_ID();
  dst[gid] = src[bitreverse(gid, lgn)];
}

/*
 * Bitreverse elements before doing inplace FFT
 */
KERNEL void FIELD_reverse_bits(GLOBAL FIELD* a, // Source buffer
                           uint lgn) // Log2 of n
{
  uint k = GET_GLOBAL_ID();
  uint rk = bitreverse(k, lgn);
  if(k < rk) {
    FIELD old = a[rk];
    a[rk] = a[k];
    a[k] = old;
  }
}

/*
 * Inplace FFT algorithm, uses 1/2 less memory than radix-fft
 * Inspired from original bellman FFT implementation
 */
KERNEL void FIELD_inplace_fft(GLOBAL FIELD* a, // Source buffer
                          GLOBAL FIELD* omegas, // [omega, omega^2, omega^4, ...]
                          uint lgn,
                          uint lgm) // Log2 of n
{
  uint gid = GET_GLOBAL_ID();
  uint n = 1 << lgn;
  uint m = 1 << lgm;
  uint j = gid & (m - 1);
  uint k = 2 * m * (gid >> lgm);
  FIELD w = FIELD_pow_lookup(omegas, j << (lgn - 1 - lgm));
  FIELD t = FIELD_mul(a[k + j + m], w);
  FIELD tmp = a[k + j];
  tmp = FIELD_sub(tmp, t);
  a[k + j + m] = tmp;
  a[k + j] = FIELD_add(a[k + j], t);
}

/*
 * FFT algorithm is inspired from: http://www.bealto.com/gpu-fft_group-1.html
 */
KERNEL void FIELD_radix_fft(GLOBAL FIELD* x, // Source buffer
                        GLOBAL FIELD* y, // Destination buffer
                        GLOBAL FIELD* pq, // Precalculated twiddle factors
                        GLOBAL FIELD* omegas, // [omega, omega^2, omega^4, ...]
                        uint n, // Number of elements
                        uint lgp, // Log2 of `p` (Read more in the link above)
                        uint deg, // 1=>radix2, 2=>radix4, 3=>radix8, ...
                        uint max_deg) // Maximum degree supported, according to `pq` and `omegas`
{

  uint __lid = GET_LOCAL_ID();
  uint lsize = 1 << deg >> 1; // get_local_size(0) == 1 << max_deg
  uint lid = __lid & (lsize - 1);
  uint loffset = (__lid - lid);
  uint index = (GET_GROUP_ID() << (max_deg - deg)) + (loffset >> (deg - 1));
  uint t = n >> deg;
  uint p = 1 << lgp;
  uint k = index & (p - 1);

  x += index;
  y += ((index - k) << deg) + k;

  uint count = 1 << deg; // 2^deg
  uint counth = count >> 1; // Half of count

  uint counts = count / lsize * lid;
  uint counte = counts + count / lsize;

  __shared__ FIELD uu[256];
  FIELD* u = uu;

  u += 2*loffset;
  __shared__ FIELD pq_shared[128];
  // load pq_shared
  pq_shared[threadIdx.x] = pq[threadIdx.x];

  // Compute powers of twiddle
  uint t_power = (n >> lgp >> deg) * k;
  uint new_counts = counts + 1;
  FIELD tmp = FIELD_ONE;
  FIELD tmp_1 = FIELD_ONE;
  if (t_power != 0) {
    tmp = omegas[t_power * counts];
    tmp_1 = omegas[t_power * new_counts];
    u[counts] = FIELD_mul(tmp, x[counts * t]);
    u[counts + 1] = FIELD_mul(tmp_1, x[new_counts * t]);
  } else {
    u[counts] = x[counts * t];
    u[counts + 1] = x[new_counts * t];
  }
  BARRIER_LOCAL();

  uint pqshift = max_deg - deg;
  for(uint rnd = 0; rnd < deg; rnd++) {
    uint bit = counth >> rnd;
    for(uint i = counts >> 1; i < counte >> 1; i++) {
      uint di = i & (bit - 1);
      uint i0 = (i << 1) - di;
      uint i1 = i0 + bit;
      tmp = u[i0];
      u[i0] = FIELD_add(u[i0], u[i1]);
      u[i1] = FIELD_sub(tmp, u[i1]);
      if(di != 0) u[i1] = FIELD_mul(pq_shared[di << rnd << pqshift], u[i1]);
    }

    BARRIER_LOCAL();
  }

  for(uint i = counts >> 1; i < counte >> 1; i++) {
    y[i*p] = u[bitreverse(i, deg)];
    y[(i+counth)*p] = u[bitreverse(i + counth, deg)];
  }
}