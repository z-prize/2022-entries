#include "fields.h"
#include "structs.h"

# define sqr(ret,a)		sqr_fp(ret,a)
# define mul(ret,a,b)		mul_fp(ret,a,b)
# define sqr_n_mul(ret,a,n,b)	sqr_n_mul_fp(ret,a,n,b)

# include "recip-addchain.h"
static void flt_reciprocal_fp(vec384 out, const vec384 inp)
{
    RECIPROCAL_MOD_BLS12_377_P(out, inp, vec384);
}
# undef RECIPROCAL_MOD_BLS12_377_P
# undef sqr_n_mul
# undef mul
# undef sqr


static void reciprocal_fp(vec384 out, const vec384 inp, const vec384 p, const limb_t p0)
{
    static const vec384 Px128 = {    /* left-aligned value of the modulus */
        TO_LIMB_T(0x2846000000000080), TO_LIMB_T(0xb85aea2180000004),
        TO_LIMB_T(0xf79b117dd04a4000), TO_LIMB_T(0xd116cf9807a89c78),
        TO_LIMB_T(0x31d82e03650a49d8), TO_LIMB_T(0xd71d230be288756L)
    };
#ifdef __BLST_NO_ASM__
# define RRx4 BLS12_377_RR
#else
    static const vec384 RRx4 = {
        TO_LIMB_T(0x5910e1b250033487), TO_LIMB_T(0xf59c95669010c6c6),
        TO_LIMB_T(0x6ba46215d15189b3), TO_LIMB_T(0xe55b1a1b0901fb21),
        TO_LIMB_T(0x47bf46009942e6ab), TO_LIMB_T(0x9b8e662801d37)
    };
#endif
    union { vec768 x; vec384 r[2]; } temp;

    ct_inverse_mod_383(temp.x, inp, p, Px128);
    redc_mont_384(temp.r[0], temp.x, p, p0);
    mul_mont_384(temp.r[0], temp.r[0], RRx4, p, p0);

#ifndef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
    /* sign goes straight to flt_reciprocal */
    mul_mont_384(temp.r[1], temp.r[0], inp, p, p0);
    if (vec_is_equal(temp.r[1],  BLS12_377_Rx, sizeof(vec384)) |
        vec_is_zero(temp.r[1], sizeof(vec384)))
        {
        vec_copy(out, temp.r[0], sizeof(vec384));
        }
    else
        flt_reciprocal_fp(out, inp);
#else
    vec_copy(out, temp.r[0], sizeof(vec384));
#endif
#undef RRx4
}

void blst_fp_mul(vec384 ret, const limb_t* a, const limb_t* b, const limb_t* p, const limb_t p0)
{
    mul_mont_384(ret, a, b, p, p0);
}

void blst_fp_sqr(vec384 ret, const limb_t* a, const limb_t* p, const limb_t p0)
{
    sqr_mont_384(ret, a, p, p0);
}

void blst_fp_eucl_inverse(vec384 ret, const vec384 a, const vec384 p, const limb_t p0)
{   
    reciprocal_fp(ret, a, p, p0);
}
