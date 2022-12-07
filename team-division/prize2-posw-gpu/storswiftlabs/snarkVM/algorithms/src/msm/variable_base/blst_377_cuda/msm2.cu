#include "blst_377_ops.h"
#include <stdio.h>
#include <stdint.h>
// #include <cooperative_groups.h>
// #include <cooperative_groups/reduce.h>
// namespace cg = cooperative_groups;

#define blst_fp_copy(desc, src)  { \
    (desc)[0] = (src)[0]; \
    (desc)[1] = (src)[1]; \
    (desc)[2] = (src)[2]; \
    (desc)[3] = (src)[3]; \
    (desc)[4] = (src)[4]; \
    (desc)[5] = (src)[5]; \
}

#define blst_p1_affine_copy( desc, src) {\
    blst_fp_copy((desc).X , (src).X);\
    blst_fp_copy((desc).Y , (src).Y);\
}

#define  blst_p1_copy(desc, src) { \
    blst_fp_copy((desc)->X , (src)->X);\
    blst_fp_copy((desc)->Y , (src)->Y);\
    blst_fp_copy((desc)->Z , (src)->Z);\
}


static const uint32_t MAX_WINDOW_SIZE = 482;
// static const uint32_t BLST_WIDTH = 253;
// __device__ const limb_t BLS12_377_FR_INV = TO_LIMB_T(0XA117FFFFFFFFFFF);
__device__ const blst_fr BLS12_377_FR_MODULUS = {
    TO_LIMB_T(0x0a11800000000001),
    TO_LIMB_T(0x59aa76fed0000001),
    TO_LIMB_T(0x60b44d1e5c37b001),
    TO_LIMB_T(0x12ab655e9a2ca556)
};

__device__ const blst_fr BLS12_377_FR_ONE = {
    TO_LIMB_T(9015221291577245683),
    TO_LIMB_T(8239323489949974514),
    TO_LIMB_T(1646089257421115374),
    TO_LIMB_T(958099254763297437),
};

#define blst_fr_copy(desc, src)  { \
    (desc)[0] = (src)[0]; \
    (desc)[1] = (src)[1]; \
    (desc)[2] = (src)[2]; \
    (desc)[3] = (src)[3]; \
}

__device__ inline int blst_fr_is_ge(const blst_fr left, const blst_fr right) {
    for (int i = 3; i >= 0; --i) {
        if (left[i] < right[i]) {
            return 0;
        } else if (left[i] > right[i]) {
            return 1;
        }
    }
    return 1;
}

__device__ inline void blst_fr_sub_unchecked(blst_fr ret,  const blst_fr a, const blst_fr b)
{
    asm(
        "sub.cc.u64  %0, %4, %8;\n\t" 
        "subc.cc.u64 %1, %5, %9;\n\t"
        "subc.cc.u64 %2, %6, %10;\n\t"
        "subc.u64    %3, %7, %11;\n\t"
        :"=l"(ret[0]),
        "=l"(ret[1]),
        "=l"(ret[2]),
        "=l"(ret[3])
        :"l"(a[0]),
        "l"(a[1]),
        "l"(a[2]),
        "l"(a[3]),
        "l"(b[0]),
        "l"(b[1]),
        "l"(b[2]),
        "l"(b[3])
    );
}


__device__ inline void blst_fr_add_unchecked(blst_fr ret,  const blst_fr a, const blst_fr b)
{
    asm(
      "add.cc.u64  %0, %4, %8;\n\t" 
      "addc.cc.u64 %1, %5, %9;\n\t" 
      "addc.cc.u64 %2, %6, %10;\n\t" 
      "addc.u64    %3, %7, %11;\n\t" 
      : "=l"(ret[0]),
      "=l"(ret[1]),
      "=l"(ret[2]),
      "=l"(ret[3])
      : "l"(a[0]),
      "l"(a[1]),
      "l"(a[2]),
      "l"(a[3]),
      "l"(b[0]),
      "l"(b[1]),
      "l"(b[2]),
      "l"(b[3])
    );
}


__device__ inline void blst_fr_add(blst_fr ret,  const blst_fr a, const blst_fr b)
{
    blst_fr_add_unchecked(ret, a, b);

    if (blst_fr_is_ge(ret, BLS12_377_FR_MODULUS)) {
        blst_fr_sub_unchecked(ret, ret, BLS12_377_FR_MODULUS);
    }
}


__device__ inline void blst_fr_sub(blst_fr ret,  const blst_fr a, const blst_fr b)
{
    if (blst_fr_is_ge(b, a)) {
        blst_fr added;
        blst_fr_add_unchecked(added, a, BLS12_377_FR_MODULUS);
        blst_fr_sub_unchecked(ret, added, b);
    }
    else
        blst_fr_sub_unchecked(ret, a, b);
}


__device__ inline void blst_fr_reduce(blst_fr ret, limb_t r[8])
{
    if (r[0] == 0 && r[1] == 0 && 
        r[2] == 0 && r[3] == 0 && 
        r[4] == 0 && r[5] == 0 && 
        r[6] == 0 && r[7] == 0) {
        ret[0] = 0;
        ret[1] = 0;
        ret[2] = 0;
        ret[3] = 0;
        return;
    }
    asm(
    "{\n\t"
    ".reg .u32 k0, k1, hi;\n\t"
    ".reg .u32 p1, p2, p3, p4, p5, p6, p7;\n\t"
    ".reg .u64 c, c2, r, inv, lo;\n\t"
    ".reg .u64 r0, r1, r2, r3, r4, r5, r6, r7, r8;\n\t"
    "mov.u64 inv, 0XA117FFFFFFFFFFF;\n\t"
    "mov.u64 lo, 0xFFFFFFFF;\n\t"
    "mov.u32 hi, 32;\n\t"
    "mov.u32 p1, 0x0a118000;\n\t"
    "mov.u32 p2, 0xd0000001;\n\t"
    "mov.u32 p3, 0x59aa76fe;\n\t"
    "mov.u32 p4, 0x5c37b001;\n\t"
    "mov.u32 p5, 0x60b44d1e;\n\t"
    "mov.u32 p6, 0x9a2ca556;\n\t"
    "mov.u32 p7, 0x12ab655e;\n\t"

    "mul.lo.u64 r, %4, inv;\n\t"  
    "mov.b64 {k0, k1}, r;\n\t"
    "and.b64 r, %4, lo;\n\t"   
    "cvt.u64.u32  c, k0;\n\t"          
    "add.u64 r0, c, r;\n\t"    
    "shr.u64 c, r0, hi;\n\t"
    "shr.u64 r, %4, hi;\n\t"   
    "mad.wide.u32 r, k0, p1, r;\n\t"   
    "add.u64 r1, c, r;\n\t"    
    "shr.u64 c, r1, hi;\n\t"
    "and.b64 r, %5, lo;\n\t"   
    "mad.wide.u32 r, k0, p2, r;\n\t"   
    "add.u64 r2, c, r;\n\t"    
    "shr.u64 c, r2, hi;\n\t"
    "shr.u64 r, %5, hi;\n\t"   
    "mad.wide.u32 r, k0, p3, r;\n\t"   
    "add.u64 r3, c, r;\n\t"    
    "shr.u64 c, r3, hi;\n\t"
    "and.b64 r, %6, lo;\n\t"   
    "mad.wide.u32 r, k0, p4, r;\n\t"   
    "add.u64 r4, c, r;\n\t"    
    "shr.u64 c, r4, hi;\n\t"
    "shr.u64 r, %6, hi;\n\t"   
    "mad.wide.u32 r, k0, p5, r;\n\t"   
    "add.u64 r5, c, r;\n\t"    
    "shr.u64 c, r5, hi;\n\t"
    "and.b64 r, %7, lo;\n\t"   
    "mad.wide.u32 r, k0, p6, r;\n\t"   
    "add.u64 r6, c, r;\n\t"    
    "shr.u64 c, r6, hi;\n\t"
    "shr.u64 r, %7, hi;\n\t"   
    "mad.wide.u32 r, k0, p7, r;\n\t"   
    "add.u64 r7, c, r;\n\t"    
    "shr.u64 c, r7, hi;\n\t"   
    "and.b64 r, %8, lo;\n\t"   
    "add.u64 r8, c, r;\n\t"    
    "shr.u64 c2,r8, hi;\n\t"
    "and.b64 r, r1, lo;\n\t"   
    "cvt.u64.u32  c, k1;\n\t"          
    "add.u64 r1, c, r;\n\t"    
    "shr.u64 c, r1, hi;\n\t"
    "and.b64 r, r2, lo;\n\t"   
    "mad.wide.u32 r, k1, p1, r;\n\t"   
    "add.u64 r2, c, r;\n\t"    
    "shr.u64 c, r2, hi;\n\t"
    "and.b64 r, r3, lo;\n\t"   
    "mad.wide.u32 r, k1, p2, r;\n\t"   
    "add.u64 r3, c, r;\n\t"    
    "shr.u64 c, r3, hi;\n\t"
    "and.b64 r, r4, lo;\n\t"   
    "mad.wide.u32 r, k1, p3, r;\n\t"   
    "add.u64 r4, c, r;\n\t"    
    "shr.u64 c, r4, hi;\n\t"
    "and.b64 r, r5, lo;\n\t"   
    "mad.wide.u32 r, k1, p4, r;\n\t"   
    "add.u64 r5, c, r;\n\t"    
    "shr.u64 c, r5, hi;\n\t"
    "and.b64 r, r6, lo;\n\t"   
    "mad.wide.u32 r, k1, p5, r;\n\t"   
    "add.u64 r6, c, r;\n\t"    
    "shr.u64 c, r6, hi;\n\t"
    "and.b64 r, r7, lo;\n\t"   
    "mad.wide.u32 r, k1, p6, r;\n\t"   
    "add.u64 r7, c, r;\n\t"    
    "shr.u64 c, r7, hi;\n\t"
    "and.b64 r, r8, lo;\n\t"   
    "mad.wide.u32 r, k1, p7, r;\n\t"   
    "add.u64 r8, c, r;\n\t"    
    "shr.u64 c, r8, hi;\n\t"
    "shr.u64 r, %8, hi;\n\t"   
    "add.u64 r, r, c2;\n\t"            
    "add.u64 r0, c, r;\n\t"    
    "shr.u64 c2,r0, hi;\n\t"
    "bfi.b64  r, r3, r2, hi, hi;\n\t"  
    "mul.lo.u64 r, r, inv;\n\t"  
    "mov.b64 {k0, k1}, r;\n\t" 
    "and.b64 r, r2, lo;\n\t"   
    "cvt.u64.u32  c, k0;\n\t"          
    "add.u64 r2, c, r;\n\t"    
    "shr.u64 c, r2, hi;\n\t"
    "and.b64 r, r3, lo;\n\t"   
    "mad.wide.u32 r, k0, p1, r;\n\t"   
    "add.u64 r3, c, r;\n\t"    
    "shr.u64 c, r3, hi;\n\t"
    "and.b64 r, r4, lo;\n\t"   
    "mad.wide.u32 r, k0, p2, r;\n\t"   
    "add.u64 r4, c, r;\n\t"    
    "shr.u64 c, r4, hi;\n\t"
    "and.b64 r, r5, lo;\n\t"   
    "mad.wide.u32 r, k0, p3, r;\n\t"   
    "add.u64 r5, c, r;\n\t"    
    "shr.u64 c, r5, hi;\n\t"
    "and.b64 r, r6, lo;\n\t"   
    "mad.wide.u32 r, k0, p4, r;\n\t"   
    "add.u64 r6, c, r;\n\t"    
    "shr.u64 c, r6, hi;\n\t"
    "and.b64 r, r7, lo;\n\t"   
    "mad.wide.u32 r, k0, p5, r;\n\t"   
    "add.u64 r7, c, r;\n\t"    
    "shr.u64 c, r7, hi;\n\t"
    "and.b64 r, r8, lo;\n\t"   
    "mad.wide.u32 r, k0, p6, r;\n\t"   
    "add.u64 r8, c, r;\n\t"    
    "shr.u64 c, r8, hi;\n\t"
    "and.b64 r, r0, lo;\n\t"   
    "mad.wide.u32 r, k0, p7, r;\n\t"   
    "add.u64 r0, c, r;\n\t"    
    "shr.u64 c, r0, hi;\n\t"
    "and.b64 r, %9, lo;\n\t"   
    "add.u64 r, r, c2;\n\t"            
    "add.u64 r1, c, r;\n\t"    
    "shr.u64 c2,r1, hi;\n\t"
    "and.b64 r, r3, lo;\n\t"    
    "cvt.u64.u32  c, k1;\n\t"          
    "add.u64 r3, c, r;\n\t"     
    "shr.u64 c, r3, hi;\n\t"
    "and.b64 r, r4, lo;\n\t"    
    "mad.wide.u32 r, k1, p1, r;\n\t"   
    "add.u64 r4, c, r;\n\t"     
    "shr.u64 c, r4, hi;\n\t"
    "and.b64 r, r5, lo;\n\t"    
    "mad.wide.u32 r, k1, p2, r;\n\t"   
    "add.u64 r5, c, r;\n\t"     
    "shr.u64 c, r5, hi;\n\t"
    "and.b64 r, r6, lo;\n\t"    
    "mad.wide.u32 r, k1, p3, r;\n\t"   
    "add.u64 r6, c, r;\n\t"     
    "shr.u64 c, r6, hi;\n\t"
    "and.b64 r, r7, lo;\n\t"    
    "mad.wide.u32 r, k1, p4, r;\n\t"   
    "add.u64 r7, c, r;\n\t"     
    "shr.u64 c, r7, hi;\n\t"
    "and.b64 r, r8, lo;\n\t"    
    "mad.wide.u32 r, k1, p5, r;\n\t"   
    "add.u64 r8, c, r;\n\t"     
    "shr.u64 c, r8, hi;\n\t"
    "and.b64 r, r0, lo;\n\t"    
    "mad.wide.u32 r, k1, p6, r;\n\t"   
    "add.u64 r0, c, r;\n\t"     
    "shr.u64 c, r0, hi;\n\t"
    "and.b64 r, r1, lo;\n\t"    
    "mad.wide.u32 r, k1, p7, r;\n\t"   
    "add.u64 r1, c, r;\n\t"     
    "shr.u64 c, r1, hi;\n\t"
    "shr.u64 r, %9, hi;\n\t"    
    "add.u64 r, r, c2;\n\t"            
    "add.u64 r2, c, r;\n\t"     
    "shr.u64 c2,r2, hi;\n\t"
    "bfi.b64  r, r5, r4, hi, hi;\n\t"  
    "mul.lo.u64 r, r, inv;\n\t"  
    "mov.b64 {k0, k1}, r;\n\t" 
    "and.b64 r,  r4, lo;\n\t"   
    "cvt.u64.u32  c, k0;\n\t"          
    "add.u64 r4, c, r;\n\t"     
    "shr.u64 c, r4, hi;\n\t"
    "and.b64 r,  r5, lo;\n\t"   
    "mad.wide.u32 r, k0, p1, r;\n\t"   
    "add.u64 r5, c, r;\n\t"     
    "shr.u64 c, r5, hi;\n\t"
    "and.b64 r,  r6, lo;\n\t"   
    "mad.wide.u32 r, k0, p2, r;\n\t"   
    "add.u64 r6, c, r;\n\t"     
    "shr.u64 c, r6, hi;\n\t"
    "and.b64 r,  r7, lo;\n\t"   
    "mad.wide.u32 r, k0, p3, r;\n\t"   
    "add.u64 r7, c, r;\n\t"     
    "shr.u64 c, r7, hi;\n\t"
    "and.b64 r,  r8, lo;\n\t"   
    "mad.wide.u32 r, k0, p4, r;\n\t"   
    "add.u64 r8, c, r;\n\t"     
    "shr.u64 c, r8, hi;\n\t"
    "and.b64 r,  r0, lo;\n\t"   
    "mad.wide.u32 r, k0, p5, r;\n\t"   
    "add.u64 r0, c, r;\n\t"     
    "shr.u64 c, r0, hi;\n\t"
    "and.b64 r,  r1, lo;\n\t"   
    "mad.wide.u32 r, k0, p6, r;\n\t"   
    "add.u64 r1, c, r;\n\t"     
    "shr.u64 c, r1, hi;\n\t"
    "and.b64 r,  r2, lo;\n\t"   
    "mad.wide.u32 r, k0, p7, r;\n\t"   
    "add.u64 r2, c, r;\n\t"     
    "shr.u64 c, r2, hi;\n\t"
    "and.b64 r, %10, lo;\n\t"   
    "add.u64 r, r, c2;\n\t"            
    "add.u64 r3, c, r;\n\t"     
    "shr.u64 c2,r3, hi;\n\t"
    "and.b64 r,  r5, lo;\n\t"   
    "cvt.u64.u32  c, k1;\n\t"          
    "add.u64 r5, c, r;\n\t"     
    "shr.u64 c, r5, hi;\n\t"
    "and.b64 r,  r6, lo;\n\t"   
    "mad.wide.u32 r, k1, p1, r;\n\t"   
    "add.u64 r6, c, r;\n\t"     
    "shr.u64 c, r6, hi;\n\t"
    "and.b64 r,  r7, lo;\n\t"   
    "mad.wide.u32 r, k1, p2, r;\n\t"   
    "add.u64 r7, c, r;\n\t"     
    "shr.u64 c, r7, hi;\n\t"
    "and.b64 r,  r8, lo;\n\t"   
    "mad.wide.u32 r, k1, p3, r;\n\t"   
    "add.u64 r8, c, r;\n\t"     
    "shr.u64 c, r8, hi;\n\t"
    "and.b64 r,  r0, lo;\n\t"   
    "mad.wide.u32 r, k1, p4, r;\n\t"   
    "add.u64 r0, c, r;\n\t"     
    "shr.u64 c, r0, hi;\n\t"
    "and.b64 r,  r1, lo;\n\t"   
    "mad.wide.u32 r, k1, p5, r;\n\t"   
    "add.u64 r1, c, r;\n\t"     
    "shr.u64 c, r1, hi;\n\t"
    "and.b64 r,  r2, lo;\n\t"   
    "mad.wide.u32 r, k1, p6, r;\n\t"   
    "add.u64 r2, c, r;\n\t"     
    "shr.u64 c, r2, hi;\n\t"
    "and.b64 r,  r3, lo;\n\t"   
    "mad.wide.u32 r, k1, p7, r;\n\t"   
    "add.u64 r3, c, r;\n\t"     
    "shr.u64 c, r3, hi;\n\t"
    "shr.u64 r, %10, hi;\n\t"   
    "add.u64 r, r, c2;\n\t"            
    "add.u64 r4, c, r;\n\t"     
    "shr.u64 c2,r4, hi;\n\t"
    "bfi.b64  r, r7, r6, hi, hi;\n\t"  
    "mul.lo.u64 r, r, inv;\n\t"  
    "mov.b64 {k0, k1}, r;\n\t" 
    "and.b64 r,  r6, lo;\n\t"   
    "cvt.u64.u32  c, k0;\n\t"          
    "add.u64 r6, c, r;\n\t"     
    "shr.u64 c, r6, hi;\n\t"
    "and.b64 r,  r7, lo;\n\t"   
    "mad.wide.u32 r, k0, p1, r;\n\t"   
    "add.u64 r7, c, r;\n\t"     
    "shr.u64 c, r7, hi;\n\t"
    "and.b64 r,  r8, lo;\n\t"   
    "mad.wide.u32 r, k0, p2, r;\n\t"   
    "add.u64 r8, c, r;\n\t"     
    "shr.u64 c, r8, hi;\n\t"
    "and.b64 r,  r0, lo;\n\t"   
    "mad.wide.u32 r, k0, p3, r;\n\t"   
    "add.u64 r0, c, r;\n\t"     
    "shr.u64 c, r0, hi;\n\t"
    "and.b64 r,  r1, lo;\n\t"   
    "mad.wide.u32 r, k0, p4, r;\n\t"   
    "add.u64 r1, c, r;\n\t"     
    "shr.u64 c, r1, hi;\n\t"
    "and.b64 r,  r2, lo;\n\t"   
    "mad.wide.u32 r, k0, p5, r;\n\t"   
    "add.u64 r2, c, r;\n\t"     
    "shr.u64 c, r2, hi;\n\t"
    "and.b64 r,  r3, lo;\n\t"   
    "mad.wide.u32 r, k0, p6, r;\n\t"   
    "add.u64 r3, c, r;\n\t"     
    "shr.u64 c, r3, hi;\n\t"
    "and.b64 r,  r4, lo;\n\t"   
    "mad.wide.u32 r, k0, p7, r;\n\t"   
    "add.u64 r4, c, r;\n\t"     
    "shr.u64 c, r4, hi;\n\t"
    "and.b64 r, %11, lo;\n\t"   
    "add.u64 r, r, c2;\n\t"            
    "add.u64 r5, c, r;\n\t"     
    "shr.u64 c2,r5, hi;\n\t"
    "and.b64 r,  r7, lo;\n\t"   
    "cvt.u64.u32  c, k1;\n\t"          
    "add.u64 r7, c, r;\n\t"     
    "shr.u64 c, r7, hi;\n\t"
    "and.b64 r,  r8, lo;\n\t"   
    "mad.wide.u32 r, k1, p1, r;\n\t"   
    "add.u64 r8, c, r;\n\t"     
    "shr.u64 c, r8, hi;\n\t"
    "and.b64 r,  r0, lo;\n\t"   
    "mad.wide.u32 r, k1, p2, r;\n\t"   
    "add.u64 r0, c, r;\n\t"     
    "shr.u64 c, r0, hi;\n\t"
    "and.b64 r,  r1, lo;\n\t"   
    "mad.wide.u32 r, k1, p3, r;\n\t"   
    "add.u64 r1, c, r;\n\t"     
    "shr.u64 c, r1, hi;\n\t"
    "and.b64 r,  r2, lo;\n\t"   
    "mad.wide.u32 r, k1, p4, r;\n\t"   
    "add.u64 r2, c, r;\n\t"     
    "shr.u64 c, r2, hi;\n\t"
    "and.b64 r,  r3, lo;\n\t"   
    "mad.wide.u32 r, k1, p5, r;\n\t"   
    "add.u64 r3, c, r;\n\t"     
    "shr.u64 c, r3, hi;\n\t"
    "and.b64 r,  r4, lo;\n\t"   
    "mad.wide.u32 r, k1, p6, r;\n\t"   
    "add.u64 r4, c, r;\n\t"     
    "shr.u64 c, r4, hi;\n\t"
    "and.b64 r,  r5, lo;\n\t"   
    "mad.wide.u32 r, k1, p7, r;\n\t"   
    "add.u64 r5, c, r;\n\t"     
    "shr.u64 c, r5, hi;\n\t"
    "shr.u64 r, %11, hi;\n\t"   
    "add.u64 r, r, c2;\n\t"            
    "add.u64 r6, c, r;\n\t"     
    "shr.u64 c2,r6, hi;\n\t"

    "bfi.b64  %0, r0, r8, hi, hi;\n\t"
    "bfi.b64  %1, r2, r1, hi, hi;\n\t"
    "bfi.b64  %2, r4, r3, hi, hi;\n\t"
    "bfi.b64  %3, r6, r5, hi, hi;\n\t"
    "}"
    : 
    "=l"(ret[0]),
    "=l"(ret[1]),
    "=l"(ret[2]),
    "=l"(ret[3])
    : 
    "l"(r[0]),
    "l"(r[1]),
    "l"(r[2]),
    "l"(r[3]),
    "l"(r[4]),
    "l"(r[5]),
    "l"(r[6]),
    "l"(r[7])
        );

    if (blst_fr_is_ge(ret, BLS12_377_FR_MODULUS)) {
        blst_fr_sub_unchecked(ret, ret, BLS12_377_FR_MODULUS);
    }
}


__device__ inline void blst_fr_mul(blst_fr ret, const blst_fr a, const blst_fr b) 
{
    limb_t r[16];
    uint32_t* a32 = (uint32_t*)a;
    uint32_t* b32 = (uint32_t*)b;
  
    asm(
    "{\n\t"
    ".reg .u64 c, t;\n\t"
    ".reg .u32 r;\n\t"
    "mov.u32 r, 32;\n\t"
    "mov.u64 t, 0xFFFFFFFF;\n\t"
    "mul.wide.u32 %0, %16, %24;\n\t"       
    "shr.u64  c, %0, r;\n\t"
    "mad.wide.u32 %1, %16, %25, c;\n\t"    
    "shr.u64  c, %1, r;\n\t"
    "mad.wide.u32 %2, %16, %26, c;\n\t"    
    "shr.u64  c, %2, r;\n\t"
    "mad.wide.u32 %3, %16, %27, c;\n\t"    
    "shr.u64  c, %3, r;\n\t"
    "mad.wide.u32 %4, %16, %28, c;\n\t"    
    "shr.u64  c, %4, r;\n\t"
    "mad.wide.u32 %5, %16, %29, c;\n\t"    
    "shr.u64  c, %5, r;\n\t"
    "mad.wide.u32 %6, %16, %30, c;\n\t"    
    "shr.u64  c, %6, r;\n\t"
    "mad.wide.u32 %7, %16, %31, c;\n\t"    
    "shr.u64 %8, %7, r;\n\t"
    "and.b64 c,  %1, t;\n\t"                                          
    "mad.wide.u32 %1, %17, %24, c;\n\t"    
    "shr.u64  c, %1, r;\n\t"   
    "and.b64 %2, %2, t;\n\t"               
    "add.u64 c, c, %2;\n\t"    
    "mad.wide.u32 %2, %17, %25, c;\n\t"    
    "shr.u64  c, %2, r;\n\t"
    "and.b64 %3, %3, t;\n\t"               
    "add.u64 c, c, %3;\n\t"    
    "mad.wide.u32 %3, %17, %26, c;\n\t"    
    "shr.u64  c, %3, r;\n\t"
    "and.b64 %4, %4, t;\n\t"               
    "add.u64 c, c, %4;\n\t"    
    "mad.wide.u32 %4, %17, %27, c;\n\t"    
    "shr.u64  c, %4, r;\n\t"
    "and.b64 %5, %5, t;\n\t"               
    "add.u64 c, c, %5;\n\t"    
    "mad.wide.u32 %5, %17, %28, c;\n\t"    
    "shr.u64  c, %5, r;\n\t"
    "and.b64 %6, %6, t;\n\t"               
    "add.u64 c, c, %6;\n\t"    
    "mad.wide.u32 %6, %17, %29, c;\n\t"    
    "shr.u64  c, %6, r;\n\t"
    "and.b64 %7, %7, t;\n\t"               
    "add.u64 c, c, %7;\n\t"    
    "mad.wide.u32 %7, %17, %30, c;\n\t"    
    "shr.u64  c, %7, r;\n\t"
    "add.u64 c, c, %8;\n\t"    
    "mad.wide.u32 %8, %17, %31, c;\n\t"    
    "shr.u64 %9, %8, r;\n\t"
    "and.b64  c, %2, t;\n\t"                                          
    "mad.wide.u32 %2, %18, %24, c;\n\t"    
    "shr.u64  c, %2, r;\n\t"
    "and.b64 %3, %3, t;\n\t"               
    "add.u64 c, c, %3;\n\t"    
    "mad.wide.u32 %3, %18, %25, c;\n\t"    
    "shr.u64  c, %3, r;\n\t"
    "and.b64 %4, %4, t;\n\t"               
    "add.u64 c, c, %4;\n\t"    
    "mad.wide.u32 %4, %18, %26, c;\n\t"    
    "shr.u64  c, %4, r;\n\t"
    "and.b64 %5, %5, t;\n\t"               
    "add.u64 c, c, %5;\n\t"    
    "mad.wide.u32 %5, %18, %27, c;\n\t"    
    "shr.u64  c, %5, r;\n\t"
    "and.b64 %6, %6, t;\n\t"               
    "add.u64 c, c, %6;\n\t"    
    "mad.wide.u32 %6, %18, %28, c;\n\t"    
    "shr.u64  c, %6, r;\n\t"
    "and.b64 %7, %7, t;\n\t"               
    "add.u64 c, c, %7;\n\t"    
    "mad.wide.u32 %7, %18, %29, c;\n\t"    
    "shr.u64  c, %7, r;\n\t"
    "and.b64 %8, %8, t;\n\t"               
    "add.u64 c, c, %8;\n\t"    
    "mad.wide.u32 %8, %18, %30, c;\n\t"    
    "shr.u64  c, %8, r;\n\t"
    "add.u64 c, c, %9;\n\t"    
    "mad.wide.u32 %9, %18, %31, c;\n\t"    
    "shr.u64 %10,%9, r;\n\t"
    "and.b64  c, %3, t;\n\t"                                          
    "mad.wide.u32 %3, %19,  %24, c;\n\t"    
    "shr.u64  c, %3,  r;\n\t"   
    "and.b64 %4, %4, t;\n\t"                
    "add.u64 c, c, %4;\n\t"    
    "mad.wide.u32 %4, %19,  %25, c;\n\t"    
    "shr.u64  c, %4,  r;\n\t"
    "and.b64 %5, %5, t;\n\t"                
    "add.u64 c, c, %5;\n\t"    
    "mad.wide.u32 %5, %19,  %26, c;\n\t"    
    "shr.u64  c, %5,  r;\n\t"
    "and.b64 %6, %6, t;\n\t"                
    "add.u64 c, c, %6;\n\t"    
    "mad.wide.u32 %6, %19,  %27, c;\n\t"    
    "shr.u64  c, %6,  r;\n\t"
    "and.b64 %7, %7, t;\n\t"                
    "add.u64 c, c, %7;\n\t"    
    "mad.wide.u32 %7, %19,  %28, c;\n\t"    
    "shr.u64  c, %7,  r;\n\t"
    "and.b64 %8, %8, t;\n\t"                
    "add.u64 c, c, %8;\n\t"    
    "mad.wide.u32 %8, %19,  %29, c;\n\t"    
    "shr.u64  c, %8,  r;\n\t"
    "and.b64 %9, %9, t;\n\t"                
    "add.u64 c, c, %9;\n\t"    
    "mad.wide.u32 %9, %19,  %30, c;\n\t"    
    "shr.u64  c, %9,  r;\n\t"
    "add.u64 c, c, %10;\n\t"   
    "mad.wide.u32 %10, %19, %31, c;\n\t"    
    "shr.u64 %11,%10, r;\n\t"
    "and.b64  c, %4, t;\n\t"                                          
    "mad.wide.u32 %4, %20,  %24, c;\n\t"   
    "shr.u64  c, %4,  r;\n\t"
    "and.b64 %5, %5, t;\n\t"               
    "add.u64 c, c, %5;\n\t"    
    "mad.wide.u32 %5, %20,  %25, c;\n\t"   
    "shr.u64  c, %5,  r;\n\t"
    "and.b64 %6, %6, t;\n\t"               
    "add.u64 c, c, %6;\n\t"    
    "mad.wide.u32 %6, %20,  %26, c;\n\t"   
    "shr.u64  c, %6,  r;\n\t"
    "and.b64 %7, %7, t;\n\t"               
    "add.u64 c, c, %7;\n\t"    
    "mad.wide.u32 %7, %20,  %27, c;\n\t"   
    "shr.u64  c, %7,  r;\n\t"
    "and.b64 %8, %8, t;\n\t"               
    "add.u64 c, c, %8;\n\t"    
    "mad.wide.u32 %8, %20,  %28, c;\n\t"   
    "shr.u64  c, %8,  r;\n\t"
    "and.b64 %9, %9, t;\n\t"               
    "add.u64 c, c, %9;\n\t"    
    "mad.wide.u32 %9, %20,  %29, c;\n\t"   
    "shr.u64  c, %9,  r;\n\t"
    "and.b64 %10, %10, t;\n\t"             
    "add.u64 c, c, %10;\n\t"   
    "mad.wide.u32 %10, %20, %30, c;\n\t"   
    "shr.u64  c, %10, r;\n\t"
    "add.u64 c, c, %11;\n\t"   
    "mad.wide.u32 %11, %20, %31, c;\n\t"    
    "shr.u64 %12,%11, r;\n\t"
    "and.b64  c, %5, t;\n\t"                                           
    "mad.wide.u32 %5, %21,  %24, c;\n\t"    
    "shr.u64  c, %5,  r;\n\t"   
    "and.b64 %6, %6, t;\n\t"                
    "add.u64 c, c, %6;\n\t"    
    "mad.wide.u32 %6, %21,  %25, c;\n\t"    
    "shr.u64  c, %6,  r;\n\t"
    "and.b64 %7, %7, t;\n\t"                
    "add.u64 c, c, %7;\n\t"    
    "mad.wide.u32 %7, %21,  %26, c;\n\t"    
    "shr.u64  c, %7,  r;\n\t"
    "and.b64 %8, %8, t;\n\t"                
    "add.u64 c, c, %8;\n\t"    
    "mad.wide.u32 %8, %21,  %27, c;\n\t"    
    "shr.u64  c, %8,  r;\n\t"
    "and.b64 %9, %9, t;\n\t"                
    "add.u64 c, c, %9;\n\t"    
    "mad.wide.u32 %9, %21,  %28, c;\n\t"    
    "shr.u64  c, %9,  r;\n\t"
    "and.b64 %10, %10, t;\n\t"              
    "add.u64 c, c, %10;\n\t"   
    "mad.wide.u32 %10, %21, %29, c;\n\t"    
    "shr.u64  c, %10, r;\n\t"
    "and.b64 %11, %11, t;\n\t"              
    "add.u64 c, c, %11;\n\t"   
    "mad.wide.u32 %11, %21, %30, c;\n\t"    
    "shr.u64  c, %11, r;\n\t"
    "add.u64 c, c, %12;\n\t"   
    "mad.wide.u32 %12, %21, %31, c;\n\t"    
    "shr.u64 %13,%12, r;\n\t"
    "and.b64  c, %6, t;\n\t"                                           
    "mad.wide.u32 %6, %22,  %24, c;\n\t"    
    "shr.u64  c, %6,  r;\n\t"
    "and.b64 %7, %7, t;\n\t"                
    "add.u64 c, c, %7;\n\t"    
    "mad.wide.u32 %7, %22,  %25, c;\n\t"    
    "shr.u64  c, %7,  r;\n\t"
    "and.b64 %8, %8, t;\n\t"                
    "add.u64 c, c, %8;\n\t"    
    "mad.wide.u32 %8, %22,  %26, c;\n\t"    
    "shr.u64  c, %8,  r;\n\t"
    "and.b64 %9, %9, t;\n\t"                
    "add.u64 c, c, %9;\n\t"    
    "mad.wide.u32 %9, %22,  %27, c;\n\t"    
    "shr.u64  c, %9,  r;\n\t"
    "and.b64 %10, %10, t;\n\t"              
    "add.u64 c, c, %10;\n\t"   
    "mad.wide.u32 %10, %22, %28, c;\n\t"    
    "shr.u64  c, %10, r;\n\t"
    "and.b64 %11, %11, t;\n\t"              
    "add.u64 c, c, %11;\n\t"   
    "mad.wide.u32 %11, %22, %29, c;\n\t"    
    "shr.u64  c, %11, r;\n\t"
    "and.b64 %12, %12, t;\n\t"              
    "add.u64 c, c, %12;\n\t"   
    "mad.wide.u32 %12, %22, %30, c;\n\t"    
    "shr.u64  c, %12, r;\n\t"
    "add.u64 c, c, %13;\n\t"   
    "mad.wide.u32 %13, %22, %31, c;\n\t"    
    "shr.u64 %14,%13, r;\n\t" 
    "and.b64  c, %7, t;\n\t"                
    "mad.wide.u32 %7, %23,  %24, c;\n\t"    
    "shr.u64  c, %7,  r;\n\t"   
    "and.b64 %8, %8, t;\n\t"                
    "add.u64 c, c, %8;\n\t"    
    "mad.wide.u32 %8, %23,  %25, c;\n\t"    
    "shr.u64  c, %8,  r;\n\t"
    "and.b64 %9, %9, t;\n\t"                
    "add.u64 c, c, %9;\n\t"    
    "mad.wide.u32 %9, %23,  %26, c;\n\t"    
    "shr.u64  c, %9,  r;\n\t"
    "and.b64 %10, %10, t;\n\t"              
    "add.u64 c, c, %10;\n\t"   
    "mad.wide.u32 %10, %23, %27, c;\n\t"    
    "shr.u64  c, %10, r;\n\t"
    "and.b64 %11, %11, t;\n\t"              
    "add.u64 c, c, %11;\n\t"   
    "mad.wide.u32 %11, %23, %28, c;\n\t"    
    "shr.u64  c, %11, r;\n\t"
    "and.b64 %12, %12, t;\n\t"              
    "add.u64 c, c, %12;\n\t"   
    "mad.wide.u32 %12, %23, %29, c;\n\t"    
    "shr.u64  c, %12, r;\n\t"
    "and.b64 %13, %13, t;\n\t"              
    "add.u64 c, c, %13;\n\t"   
    "mad.wide.u32 %13, %23, %30, c;\n\t"    
    "shr.u64  c, %13, r;\n\t" 
    "add.u64 c, c, %14;\n\t"   
    "mad.wide.u32 %14, %23, %31, c;\n\t"    
    "shr.u64 %15,%14, r;\n\t"
    "bfi.b64  %0, %1,  %0,  r, r;\n\t"
    "bfi.b64  %1, %3,  %2,  r, r;\n\t"
    "bfi.b64  %2, %5,  %4,  r, r;\n\t"
    "bfi.b64  %3, %7,  %6,  r, r;\n\t"
    "bfi.b64  %4, %9,  %8,  r, r;\n\t"
    "bfi.b64  %5, %11, %10, r, r;\n\t"
    "bfi.b64  %6, %13, %12, r, r;\n\t"
    "bfi.b64  %7, %15, %14, r, r;\n\t"
    "}"
    : 
    "+l"(r[0]),
    "+l"(r[1]),
    "+l"(r[2]),
    "+l"(r[3]),
    "+l"(r[4]),
    "+l"(r[5]),
    "+l"(r[6]),
    "+l"(r[7]),
    "+l"(r[8]),
    "+l"(r[9]),
    "+l"(r[10]),
    "+l"(r[11]),
    "+l"(r[12]),
    "+l"(r[13]),
    "+l"(r[14]),
    "+l"(r[15])
    : 
    "r"(a32[0]),
    "r"(a32[1]),
    "r"(a32[2]),
    "r"(a32[3]),
    "r"(a32[4]),
    "r"(a32[5]),
    "r"(a32[6]),
    "r"(a32[7]),
    "r"(b32[0]),
    "r"(b32[1]),
    "r"(b32[2]),
    "r"(b32[3]),
    "r"(b32[4]),
    "r"(b32[5]),
    "r"(b32[6]),
    "r"(b32[7])
    );

    blst_fr_reduce(ret, r);
}


__device__ inline void mont_fr_reduce(blst_fr ret,  const blst_fr a)
{
    limb_t r[8] ={a[0], a[1], a[2], a[3], 0, 0, 0, 0};
    blst_fr_reduce(ret, r);
}



extern "C" __global__  void msm4_evaluate(
    blst_fr* result,
    blst_fr* bases,
    blst_fr* bate,
    blst_fr* gamma
)
{
    size_t gid = blockIdx.x;
    blst_fr* gammah = gamma + 32768;
    blst_fr* group_powers[]={ bate, bate, gamma, gammah, gamma, gammah, gamma, gammah};    
    const uint32_t group_size[] =  { 32768, 32767, 32768, 32767, 32768,  32767, 32768,  32767 };
    const uint32_t group_start[] = { 0, 32768, 65535, 98303, 131070, 163838, 196605, 229373, 262140};

    blst_fr* _bases = bases + group_start[gid];
    blst_fr* _powers = group_powers[gid];

    uint32_t size = group_size[gid];
    size_t start = threadIdx.x << 7;
    size_t end = start + 128;
    if (start >= size)
        return;
    if (end > size)
        end = size;

    blst_fr ret = {0};
    blst_fr sum_ret = {0};
    do {
        blst_fr_mul(ret, _bases[start], _powers[start]);
        blst_fr_add(sum_ret, sum_ret, ret);
    } while (++start < end);

    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    blst_fr_copy(result[idx], sum_ret);
}


extern "C" __global__ void msm4_sum(
    blst_fr* result,
    blst_fr* bases,
    const uint32_t chunk_size,
    const uint32_t data_size
)
{
    uint32_t idx = (threadIdx.x + blockIdx.x * blockDim.x);
    uint32_t start = idx * chunk_size;
    uint32_t end = start + chunk_size; 

    blst_fr ret = {0};
    if (start >= data_size)
    {
        blst_fr_copy(result[idx], ret);
        return;
    }

    if (end > data_size)
        end = data_size;
    
    while(start < end)
    {
        blst_fr_add(ret, ret, bases[start]);
        start++;
    }
    
    blst_fr_copy(result[idx], ret);
}


extern "C" __global__ void msm4_powers_serial(
    blst_fr* bases, 
    blst_fr* roots,
    const uint32_t root_idx,
    const uint32_t size
)
{
    size_t index = (threadIdx.x + blockIdx.x * blockDim.x) << 3;
    if ( index >= size)
        return; 

    uint32_t loop = index;
    blst_fr res, b, r;
    blst_fr_copy(b, roots[root_idx]); 
    blst_fr_copy(r, roots[root_idx]);       
    blst_fr_copy(res, BLS12_377_FR_ONE);

    while(loop)
    {
        if(loop & 1)
            blst_fr_mul(res, res, b);
        blst_fr_mul(b, b, b);
        loop >>= 1;
    }
    blst_fr_copy(bases[index], res);  

    for (int i=1; i<8; i++)
    {
        if (index+i >= size)
            break;

        blst_fr_mul(res, res, r);
        blst_fr_copy(bases[index+i], res);  
    }
}

extern "C" __global__ void msm4_io_helper(
    blst_fr* bases,
    blst_fr* roots,
    const uint32_t chunk_size,
    const uint32_t chunk_num,
    const uint32_t gap,
    const uint32_t size
)
{
    size_t index = (threadIdx.x + blockIdx.x * blockDim.x) << 3;
    if ( index >= size)
        return;
    size_t end = index + 8;
    if (end > size)
        end = size;

    do {
    uint32_t cid = index & (chunk_num - 1);
    uint32_t rid = index & (~(chunk_num - 1));
    uint32_t lid = index / chunk_num + cid * chunk_size;

    blst_fr neg, lo, hi;
    blst_fr_copy(lo, bases[lid]);
    blst_fr_copy(hi, bases[lid + gap]);

    blst_fr_sub(neg, lo, hi);
    blst_fr_add(bases[lid], lo, hi);
    blst_fr_mul(bases[lid + gap], neg, roots[rid]);
    } while(++index < end);
}

extern "C" __global__ void msm4_oi_helper(
    blst_fr* bases,
    blst_fr* roots,
    const uint32_t chunk_size,
    const uint32_t chunk_num,
    const uint32_t gap,
    const uint32_t size
)
{
    size_t index = (threadIdx.x + blockIdx.x * blockDim.x) << 3;
    if ( index >= size)
        return;
    size_t end = index + 8;
    if (end > size)
        end = size;

    do {    
    uint32_t cid = index & (chunk_num - 1);
    uint32_t rid = index & (~(chunk_num - 1));
    uint32_t lid = index / chunk_num + cid * chunk_size;

    blst_fr neg, lo, hi;
    blst_fr_copy(lo, bases[lid]);
    blst_fr_copy(hi, bases[lid + gap]);

    blst_fr_mul(hi, hi, roots[rid]);
    blst_fr_sub(neg, lo, hi);

    blst_fr_add(bases[lid], lo, hi);
    blst_fr_copy(bases[lid + gap], neg);
    } while(++index < end);
}

extern "C" __global__ void msm4_mul_assign(
    blst_fr* base, 
    blst_fr* other,
    const uint32_t size)
{
    size_t start = (threadIdx.x + blockIdx.x * blockDim.x) << 3;
    size_t end = start + 8;
    if (start >= size)
        return;
    if (end > size)
        end = size;

    do {
        blst_fr_mul(base[start], base[start], other[0]);  
    } while (++start < end);
}

extern "C" __global__ void msm4_mul2_assign(
    blst_fr* base, 
    blst_fr* other,
    const uint32_t size)
{
    size_t start = (threadIdx.x + blockIdx.x * blockDim.x) << 3;
    size_t end = start + 8;
    if (start >= size)
        return;
    if (end > size)
        end = size;

    do {
        blst_fr_mul(base[start], base[start], other[start]);  
    } while (++start < end);
}


extern "C" __global__ void msm6_pixel(
    blst_p1* bucket_lists, 
    const blst_g1_affine* bases_in, 
    const blst_fr* scalars, 
    const uint32_t last_window_size, 
    const uint32_t window_count,
    const uint32_t window_size)
{
    limb_t index = threadIdx.x  >> 6;
    limb_t shift = threadIdx.x - (index << 6);
    limb_t mask = (limb_t) 1 << shift;

    blst_p1 bucket;
    blst_p1_copy(&bucket, &BLS12_377_ZERO_PROJECTIVE);

    uint32_t w_start = window_size * blockIdx.x;
    uint32_t w_size = (blockIdx.x == window_count -1)?last_window_size:window_size;
    // uint32_t w_end = w_start + w_size;

    __shared__ blst_fr mask_cache[MAX_WINDOW_SIZE];
    uint32_t cache_id = threadIdx.x ;
    if (cache_id < w_size)
        mont_fr_reduce(mask_cache[cache_id], scalars[w_start+cache_id]);
    cache_id += 253;
    if (cache_id < w_size)
        mont_fr_reduce(mask_cache[cache_id], scalars[w_start+cache_id]);
    __syncthreads();

    uint32_t activated_size = 0;    
    uint32_t activated_bases[MAX_WINDOW_SIZE];
    // we delay the actual additions to a second loop because it reduces warp divergence (20% practical gain)
    for (uint32_t i = 0; i < w_size; ++i) {
        if ((mask_cache[i][index] & mask) != 0)
            activated_bases[activated_size++] = w_start + i;
    }
    uint32_t i = 0;
    blst_p1 intermediate;    
    blst_p1_affine p1_affine, p2_affine;   
    for (; i < ((activated_size >>1) << 1); i += 2) {
        blst_p1_affine_copy(p1_affine, bases_in[activated_bases[i]]);
        blst_p1_affine_copy(p2_affine, bases_in[activated_bases[i + 1]]);
        blst_p1_add_affines_into_projective(&intermediate, &p1_affine, &p2_affine);
        blst_p1_add_projective_to_projective(&bucket, &bucket, &intermediate);
    }
    for (; i < activated_size; ++i) {
        blst_p1_affine_copy(p1_affine, bases_in[activated_bases[i]]);
        blst_p1_add_affine_to_projective(&bucket, &bucket, &p1_affine);
    }

    blst_p1_copy(&bucket_lists[threadIdx.x * window_count + blockIdx.x], &bucket);
}

extern "C" __global__ void msm6_collapse_rows(
    blst_p1* target, 
    const blst_p1* bucket_lists, 
    const uint32_t window_count) 
{
    uint32_t t_size = (window_count + 31) >> 5;
    uint32_t l_x = threadIdx.x >> 1;
    uint32_t l_y = threadIdx.x & 1;
    uint32_t p_y = (blockIdx.x << 1) + l_y;
    uint32_t p_x = l_x * t_size;
    uint32_t p_e = min(p_x+t_size, window_count);

    if (p_y >= 253)
        return;

    blst_p1 result;
    blst_p1 intermediate; 
    __shared__ blst_p1 cache[32];
    uint32_t c_x = threadIdx.x;
    uint32_t p_start = p_y * window_count + p_x;
    uint32_t p_end = p_y * window_count + p_e;

    if (p_x>=window_count) {
        blst_p1_copy(&result, &BLS12_377_ZERO_PROJECTIVE);
    }     
    else {
        blst_p1_copy(&result, &bucket_lists[p_start]);
    }
    for (uint32_t i=p_start+1; i<p_end; ++i) {
        blst_p1_copy(&intermediate, &bucket_lists[i]);
        blst_p1_add_projective_to_projective(&result, &result, &intermediate);
    }

    if (c_x >= 32) {
        blst_p1_copy(&cache[c_x-32], &result);
        return;
    }
    __syncthreads();
    blst_p1_copy(&intermediate, &cache[c_x]);
    blst_p1_add_projective_to_projective(&result, &result, &intermediate);

    if (c_x >= 16) {
        blst_p1_copy(&cache[c_x-16], &result);
        return;
    }
    __syncthreads();
    blst_p1_copy(&intermediate, &cache[c_x]);
    blst_p1_add_projective_to_projective(&result, &result, &intermediate);

    if (c_x >= 8) {
        blst_p1_copy(&cache[c_x-8], &result);
        return;
    }
    __syncthreads();
    blst_p1_copy(&intermediate, &cache[c_x]);
    blst_p1_add_projective_to_projective(&result, &result, &intermediate);

    if (c_x >= 4) {
        blst_p1_copy(&cache[c_x-4], &result);
        return;
    }
    __syncthreads();
    blst_p1_copy(&intermediate, &cache[c_x]);
    blst_p1_add_projective_to_projective(&result, &result, &intermediate);

    if (c_x >= 2) {
        blst_p1_copy(&cache[c_x-2], &result);
        return;
    }
    __syncthreads();
    blst_p1_copy(&intermediate, &cache[c_x]);
    blst_p1_add_projective_to_projective(&result, &result, &intermediate);

    blst_p1_copy(&target[p_y], &result);  
}
