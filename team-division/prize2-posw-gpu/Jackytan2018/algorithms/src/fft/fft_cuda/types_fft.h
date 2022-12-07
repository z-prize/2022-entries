#pragma once

#ifdef __SIZE_TYPE__
typedef __SIZE_TYPE__ size_t;
#else
#include <stddef.h>
#endif

#include <stdint.h>

typedef unsigned long long limb_t;
#define LIMB_T_BITS    64

#define TO_LIMB_T(limb64)     limb64

#define NLIMBS(bits)   (bits/LIMB_T_BITS)

#if !defined(restrict)
#if !defined(__STDC_VERSION__) || __STDC_VERSION__<199901
#if defined(__GNUC__) && __GNUC__>=2
#define restrict __restrict__
#elif defined(_MSC_VER)
#define restrict __restrict
#else
#define restrict
#endif
#endif
#endif

typedef limb_t blst_scalar[NLIMBS(256)];
typedef limb_t blst_fr[NLIMBS(256)];
typedef limb_t blst_fp[NLIMBS(384)];
typedef limb_t vec768[NLIMBS(768)];

#define ONE_MONT_P TO_LIMB_T(0x7d1c7ffffffffff3), \
                 TO_LIMB_T(0x7257f50f6ffffff2), \
                 TO_LIMB_T(0x16d81575512c0fee), \
                 TO_LIMB_T(0xd4bda322bbb9a9d)

__device__ static blst_fr BLS12_377_P = {
    TO_LIMB_T(0xa11800000000001), TO_LIMB_T(0x59aa76fed0000001),
    TO_LIMB_T(0x60b44d1e5c37b001), TO_LIMB_T(0x12ab655e9a2ca556),
};

__device__ static blst_fr BLS12_377_ZERO {
0};
__device__ static blst_fr BLS12_377_ONE {
ONE_MONT_P};
__device__ static blst_fr BLS12_377_R2 {
0x25d577bab861857b, 0xcc2c27b58860591f, 0xa7cc008fe5dc8593, 0x11fdae7eff1c939,};
__device__ static limb_t BLS12_377_p0 = (limb_t) 0xa117fffffffffff;

__device__ static const blst_fr BIGINT_ONE = { 1, 0, 0, 0 };
