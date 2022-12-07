#ifndef __BLS12_ASM_H__
#define __BLS12_ASM_H__

#include "structs.h"
            
#if defined(__ADX__) /* e.g. -march=broadwell */ && !defined(__BLST_PORTABLE__)\
                                                 && !defined(__BLST_NO_ASM__)
//# define mul_mont_sparse_256 mulx_mont_sparse_256
//# define sqr_mont_sparse_256 sqrx_mont_sparse_256
//# define from_mont_256 fromx_mont_256
//# define redc_mont_256 redcx_mont_256
# define mul_mont_384 mulx_mont_384
# define sqr_mont_384 sqrx_mont_384
//# define sqr_n_mul_mont_384 sqrx_n_mul_mont_384
# define sqr_n_mul_mont_383 sqrx_n_mul_mont_383
//# define mul_384 mulx_384
//# define sqr_384 sqrx_384
# define redc_mont_384 redcx_mont_384
//# define from_mont_384 fromx_mont_384
//# define sgn0_pty_mont_384 sgn0x_pty_mont_384
//# define sgn0_pty_mont_384x sgn0x_pty_mont_384x
# define ct_inverse_mod_383 ctx_inverse_mod_383
#elif defined(__BLST_NO_ASM__)
//# define ct_inverse_mod_383 ct_inverse_mod_384
#endif

#if defined(__GNUC__) || defined(__clang__)
# define launder(var) asm volatile("" : "+r"(var))
#else
# define launder(var)
#endif

void mul_mont_384(vec384 ret, const vec384 a, const vec384 b, const vec384 p, limb_t p0);
void sqr_mont_384(vec384 ret, const vec384 a, const vec384 p, limb_t p0);
void ct_inverse_mod_383(vec768 ret, const vec384 inp, const vec384 mod, const vec384 modx);
void redc_mont_384(vec384 ret, const vec768 a, const vec384 p, limb_t n0);

static void sqr_n_mul_mont_383(vec384 ret, const vec384 a, size_t count, const vec384 p, limb_t n0, const vec384 b);
static inline void mul_fp(vec384 ret, const vec384 a, const vec384 b)
{   mul_mont_384(ret, a, b, BLS12_377_P, p0);   }

static inline void sqr_fp(vec384 ret, const vec384 a)
{   sqr_mont_384(ret, a, BLS12_377_P, p0);   }

static inline void sqr_n_mul_fp(vec384 out, const vec384 a, size_t count, const vec384 b)
{   sqr_n_mul_mont_383(out, a, count, BLS12_377_P, p0, b);   }

static inline bool_t is_zero(limb_t l)
{
    limb_t ret = (~l & (l - 1)) >> (LIMB_T_BITS - 1);
    launder(ret);
    return ret;
}

static inline bool_t vec_is_zero(const void *a, size_t num)
{
    const limb_t *ap = (const limb_t *)a;
    limb_t acc;
    size_t i;

    num /= sizeof(limb_t);

    for (acc = 0, i = 0; i < num; i++)
        acc |= ap[i];

    return is_zero(acc);
}

static inline bool_t vec_is_equal(const void *a, const void *b, size_t num)
{
    const limb_t *ap = (const limb_t *)a;
    const limb_t *bp = (const limb_t *)b;
    limb_t acc;
    size_t i;

    num /= sizeof(limb_t);

    for (acc = 0, i = 0; i < num; i++)
        acc |= ap[i] ^ bp[i];

    return is_zero(acc);
}

static inline void vec_copy(void *restrict ret, const void *a, size_t num)
{
    limb_t *rp = (limb_t *)ret;
    const limb_t *ap = (const limb_t *)a;
    size_t i;

    num /= sizeof(limb_t);

    for (i = 0; i < num; i++)
        rp[i] = ap[i];
}

#endif