#ifndef __BLST_OPS_H__
#define __BLST_OPS_H__

#include "asm_fft_cuda.h"

__device__ static inline void blst_fr_add(blst_fr ret, blst_fr a, blst_fr b)
{
    add_mod_256(ret, a, b, BLS12_377_P);
}

__device__ static inline void blst_fr_add_unsafe(blst_fr ret, blst_fr a, blst_fr b)
{
    add_mod_256_unsafe(ret, a, b);
}

__device__ static inline void blst_fr_sub(blst_fr ret, blst_fr a, blst_fr b)
{
    sub_mod_256(ret, a, b, BLS12_377_P);
}

__device__ static inline void blst_fr_sub_unsafe(blst_fr ret, blst_fr a, blst_fr b)
{
    sub_mod_256_unsafe(ret, a, b);
}

__device__ static inline void blst_fr_cneg(blst_fr ret, blst_fr a, bool flag)
{
    cneg_mod_256(ret, a, flag, BLS12_377_P);
}

__device__ static inline void blst_fr_mul(blst_fr ret, blst_fr a, blst_fr b)
{
    mul_mont_256(ret, a, b, BLS12_377_P, BLS12_377_p0);
}

__device__ static inline void blst_fr_sqr(blst_fr ret, blst_fr a)
{
    sqr_mont_256(ret, a, BLS12_377_P, BLS12_377_p0);
}

__device__ static inline void blst_fr_pow(blst_fr * ret, blst_fr base, uint32_t exponent)
{
    pow_256(ret, base, exponent, BLS12_377_P, BLS12_377_p0, BLS12_377_ONE);
}

__device__ static inline void blst_fr_pow_lookup(blst_fr * ret, blst_fr * bases, uint32_t exponent)
{
    pow_256_lookup(ret, bases, exponent, BLS12_377_P, BLS12_377_p0, BLS12_377_ONE);
}
__device__ static inline void blst_fr_inverse(blst_fr out, blst_fr in)
{
    inverse_256(out, in);
}

__device__ void blst_opt_test();
#endif /* __BLST_OPS_H__ */
