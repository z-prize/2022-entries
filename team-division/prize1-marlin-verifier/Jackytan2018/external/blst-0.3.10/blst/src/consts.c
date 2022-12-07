/*
 * Copyright Supranational LLC
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "consts.h"

///////////// 381
///* z = -0xd201000000010000 */
//const vec384 BLS12_381_P = {    /* (z-1)^2 * (z^4 - z^2 + 1)/3 + z */
//    TO_LIMB_T(0xb9feffffffffaaab), TO_LIMB_T(0x1eabfffeb153ffff),
//    TO_LIMB_T(0x6730d2a0f6b0f624), TO_LIMB_T(0x64774b84f38512bf),
//    TO_LIMB_T(0x4b1ba7b6434bacd7), TO_LIMB_T(0x1a0111ea397fe69a)
//};

///////////// 377
/* z = -0xd201000000010000 */
//const vec384 BLS12_381_P = {    /* (z-1)^2 * (z^4 - z^2 + 1)/3 + z */
//    TO_LIMB_T(0x02cdffffffffff68),
//    TO_LIMB_T(0x51409f837fffffb1),
//    TO_LIMB_T(0x9f7db3a98a7d3ff2),
//    TO_LIMB_T(0x7b4e97b76e7c6305),
//    TO_LIMB_T(0x4cf495bf803c84e8),
//    TO_LIMB_T(0x008d6661e2fdf49a)
//};
const vec384 BLS12_381_P = {    /* (z-1)^2 * (z^4 - z^2 + 1)/3 + z */
    TO_LIMB_T(0x8508c00000000001),
    TO_LIMB_T(0x170b5d4430000000),
    TO_LIMB_T(0x1ef3622fba094800),
    TO_LIMB_T(0x1a22d9f300f5138f),
    TO_LIMB_T(0xc63b05c06ca1493b),
    TO_LIMB_T(0x1ae3a4617c510ea)
};


///////////// 381
//const limb_t BLS12_381_p0 = (limb_t)0x89f3fffcfffcfffd;  /* -1/P */

///////////// 377
const limb_t BLS12_381_p0 = (limb_t)0x8508bfffffffffff;  /* -1/P */

const radix384 BLS12_381_Rx = { /* (1<<384)%P, "radix", one-in-Montgomery */
  { { ONE_MONT_P },
    { 0 } }
};


///////////// 381
//const vec384 BLS12_381_RR = {   /* (1<<768)%P, "radix"^2, to-Montgomery */
//    TO_LIMB_T(0xf4df1f341c341746), TO_LIMB_T(0x0a76e6a609d104f1),
//    TO_LIMB_T(0x8de5476c4c95b6d5), TO_LIMB_T(0x67eb88a9939d83c0),
//    TO_LIMB_T(0x9a793e85b519952d), TO_LIMB_T(0x11988fe592cae3aa)
//};

///////////// 377
 const vec384 BLS12_381_RR = {   /* (1<<768)%P, "radix"^2, to-Montgomery */
    TO_LIMB_T(0xb786686c9400cd22), TO_LIMB_T(0x329fcaab00431b1),
    TO_LIMB_T(0x22a5f11162d6b46d), TO_LIMB_T(0xbfdf7d03827dc3ac),
    TO_LIMB_T(0x837e92f041790bf9), TO_LIMB_T(0x6dfccb1e914b88)
};


///////// 381 官方
//const vec256 BLS12_381_r = {    /* z^4 - z^2 + 1, group order */
//    TO_LIMB_T(0xffffffff00000001), TO_LIMB_T(0x53bda402fffe5bfe),
//    TO_LIMB_T(0x3339d80809a1d805), TO_LIMB_T(0x73eda753299d7d48)
//};

/////// 377 ars
//const R: BigInteger = BigInteger([
//        9015221291577245683u64,
//                8239323489949974514u64,
//                1646089257421115374u64,
//                958099254763297437u64,
//]);
//const vec256 BLS12_381_r = {    /* z^4 - z^2 + 1, group order */
//        TO_LIMB_T(0x7D1C7FFFFFFFFFF3), TO_LIMB_T(0x7257F50F6FFFFFF2),
//        TO_LIMB_T(0x16D81575512C0FEE), TO_LIMB_T(0xD4BDA322BBB9A9D)
//};
//const MODULUS: BigInteger = BigInteger([
//        725501752471715841u64,
//                6461107452199829505u64,
//                6968279316240510977u64,
//                1345280370688173398u64,
//]);

const vec256 BLS12_381_r = {    /* z^4 - z^2 + 1, group order */
        TO_LIMB_T(0X0A11800000000001), TO_LIMB_T(0x59AA76FED0000001),
        TO_LIMB_T(0x60B44D1E5C37B001), TO_LIMB_T(0x12AB655E9A2CA556)
};

/// 381 官方
//const vec256 BLS12_381_rRR = {  /* (1<<512)%r, "radix"^2, to-Montgomery */
//    TO_LIMB_T(0xc999e990f3f29c6d), TO_LIMB_T(0x2b6cedcb87925c23),
//    TO_LIMB_T(0x05d314967254398f), TO_LIMB_T(0x0748d9d99f59ff11)
//};

/// 377 ars
//#[allow(dead_code)]
//const R2: Scalar = Scalar(blst_fr {
//    l: [
//    2726216793283724667u64,
//            14712177743343147295u64,
//            12091039717619697043u64,
//            81024008013859129u64,
//    ],
//});
const vec256 BLS12_381_rRR = {  /* (1<<512)%r, "radix"^2, to-Montgomery */
    TO_LIMB_T(0x25D577BAB861857B), TO_LIMB_T(0xCC2C27B58860591F),
    TO_LIMB_T(0xA7CC008FE5DC8593), TO_LIMB_T(0x11FDAE7EFF1C939)
};


