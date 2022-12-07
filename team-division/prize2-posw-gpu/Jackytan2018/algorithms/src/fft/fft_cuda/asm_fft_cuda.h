#include "types_fft.h"

#pragma once

__device__ void mul_mont_256(blst_fr ret, blst_fr a, blst_fr b, blst_fr p, limb_t p_inv);

__device__ void sqr_mont_256(blst_fr ret, blst_fr a, blst_fr p, limb_t p_inv);

__device__ void add_mod_256(blst_fr ret, blst_fr a, blst_fr b, blst_fr p);

__device__ void sub_mod_256(blst_fr ret, blst_fr a, blst_fr b, blst_fr p);

__device__ void sub_mod_256_unsafe(blst_fr ret, blst_fr a, blst_fr b);

__device__ void add_mod_256_unsafe(blst_fr ret, blst_fr a, blst_fr b);

__device__ void div_by_2_mod_256(blst_fr ret, blst_fr a);

__device__ void cneg_mod_256(blst_fr ret, blst_fr a, bool flag, blst_fr p);

__device__ void pow_256(blst_fr * ret, blst_fr base, uint32_t exponent, blst_fr p, limb_t p_inv, blst_fr p_one);

__device__ void pow_256_lookup(blst_fr * ret, blst_fr * bases,
			       uint32_t exponent, blst_fr p, limb_t p_inv, blst_fr p_one);

__device__ void inverse_256(blst_fr out, blst_fr in);

__device__ static inline int is_gt_256(blst_fr left, blst_fr right)
{
    for (int i = 3; i >= 0; --i) {
	if (left[i] < right[i]) {
	    return 0;
	} else if (left[i] > right[i]) {
	    return 1;
	}
    }
    return 0;
}

__device__ static inline int is_blst_fr_zero(blst_fr p) {
    return p[0] == 0 &&
        p[1] == 0 &&
        p[2] == 0 &&
        p[3] == 0 ;
}

__device__ static inline int is_blst_fr_eq(blst_fr p1, const blst_fr p2) {
    return p1[0] == p2[0] &&
        p1[1] == p2[1] &&
        p1[2] == p2[2] &&
        p1[3] == p2[3];
}
