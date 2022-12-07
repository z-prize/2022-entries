/*
 * Copyright Supranational LLC
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __BLS12_381_ASM_CONST_H__
#define __BLS12_381_ASM_CONST_H__
#include "vect.h"

extern const vec384 BLS12_381_P;
extern const limb_t BLS12_381_p0;
//////////// 377
//static const limb_t p0 = (limb_t)0x89f3fffcfffcfffd;  /* -1/P */
static const limb_t p0 = (limb_t)0x8508bfffffffffff;  /* -1/P */
typedef union { vec384 p12[12]; vec384x p2; vec384 p; } radix384;
extern const radix384 BLS12_381_Rx; /* (1<<384)%P, "radix", one-in-Montgomery */
extern const vec384 BLS12_381_RR;   /* (1<<768)%P, "radix"^2, to-Montgomery   */

//////////// 381
//#define ONE_MONT_P TO_LIMB_T(0x760900000002fffd), \
//                   TO_LIMB_T(0xebf4000bc40c0002), \
//                   TO_LIMB_T(0x5f48985753c758ba), \
//                   TO_LIMB_T(0x77ce585370525745), \
//                   TO_LIMB_T(0x5c071a97a256ec6d), \
//                   TO_LIMB_T(0x15f65ec3fa80e493)

                   //////////// 377
#define ONE_MONT_P TO_LIMB_T(0x2cdffffffffff68), \
                   TO_LIMB_T(0x51409f837fffffb1), \
                   TO_LIMB_T(0x9f7db3a98a7d3ff2), \
                   TO_LIMB_T(0x7b4e97b76e7c6305), \
                   TO_LIMB_T(0x4cf495bf803c84e8), \
                   TO_LIMB_T(0x008d6661e2fdf49a)



#define ZERO_384 (BLS12_381_Rx.p2[1])

extern const vec256 BLS12_381_r;    /* order */
/// 381 官方
//static const limb_t r0 = (limb_t)0xfffffffeffffffff;  /* -1/r */
/// 377 ars
static const limb_t r0 = (limb_t)0xA117FFFFFFFFFFF;  /* -1/r */
extern const vec256 BLS12_381_rRR;  /* (1<<512)%r, "radix"^2, to-Montgomery   */

#endif
