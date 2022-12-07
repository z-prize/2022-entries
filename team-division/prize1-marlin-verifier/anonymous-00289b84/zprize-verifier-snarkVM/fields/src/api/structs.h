#ifndef  __BLS12_STRUCTS__
#define  __BLS12_STRUCTS__

#include <stddef.h>

#if defined(__x86_64__) || defined(__aarch64__)
/* These are available even in ILP32 flavours, but even then they are
 * capable of performing 64-bit operations as efficiently as in *P64. */
typedef unsigned long long limb_t;
# define LIMB_T_BITS    64

#elif defined(__BLST_NO_ASM__) || defined(__wasm64__)
typedef unsigned int limb_t;
# define LIMB_T_BITS    32
# ifndef __BLST_NO_ASM__
#  define __BLST_NO_ASM__
# endif

#else                   /* 32 bits on 32-bit platforms, 64 - on 64-bit */
typedef unsigned long limb_t;
#  ifdef _LP64
#   define LIMB_T_BITS   64
#  else
#   define LIMB_T_BITS   32
#   define __BLST_NO_ASM__
#  endif
#endif

#if LIMB_T_BITS == 64
# define TO_LIMB_T(limb64)     limb64
#else
# define TO_LIMB_T(limb64)     (limb_t)limb64,(limb_t)(limb64>>32)
#endif

#define NLIMBS(bits)   (bits/LIMB_T_BITS)

typedef limb_t vec256[NLIMBS(256)];
typedef limb_t vec512[NLIMBS(512)];
typedef limb_t vec384[NLIMBS(384)];
typedef limb_t vec768[NLIMBS(768)];
typedef vec384 vec384x[2];      /* 0 is "real" part, 1 is "imaginary" */

typedef limb_t bool_t;

static const vec384 BLS12_377_Rx = { // (1 << 384) % P
    TO_LIMB_T(0x2cdffffffffff68), TO_LIMB_T(0x51409f837fffffb1),
    TO_LIMB_T(0x9f7db3a98a7d3ff2), TO_LIMB_T(0x7b4e97b76e7c6305),
    TO_LIMB_T(0x4cf495bf803c84e8), TO_LIMB_T(0x8d6661e2fdf49a)
};

static const vec384 BLS12_377_RR = { //(1 << 768) % P
    TO_LIMB_T(0xb786686c9400cd22), TO_LIMB_T(0x329fcaab00431b1),
    TO_LIMB_T(0x22a5f11162d6b46d), TO_LIMB_T(0xbfdf7d03827dc3ac),
    TO_LIMB_T(0x837e92f041790bf9), TO_LIMB_T(0x6dfccb1e914b88)
};
static const limb_t p0 = TO_LIMB_T(0x8508bfffffffffff);
static const vec384 BLS12_377_P = {
    TO_LIMB_T(0x8508c00000000001), TO_LIMB_T(0x170b5d4430000000),
    TO_LIMB_T(0x1ef3622fba094800), TO_LIMB_T(0x1a22d9f300f5138f),
    TO_LIMB_T(0xc63b05c06ca1493b), TO_LIMB_T(0x1ae3a4617c510ea)
};
#endif