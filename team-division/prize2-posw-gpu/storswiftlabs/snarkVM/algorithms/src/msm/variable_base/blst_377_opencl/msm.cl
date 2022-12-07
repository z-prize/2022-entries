#if defined(__NVCC__) || defined(__NV_CL_C_VERSION)
#define OPENCL_NVIDIA
#endif

typedef union {
    ulong2 vec[3];
    ulong  v64[6];    
    uint   v32[12];
}blst_fp;

typedef union {
    ulong2 vec[2];
    ulong  v64[4];    
    uint   v32[8];
}blst_fr;

typedef struct { blst_fp X, Y; } blst_p1_affine;
typedef struct { blst_fp X, Y, Z; } blst_p1;
typedef struct { blst_fp X, Y, ZZ, ZZZ; } blst_p1_ext;
typedef struct __attribute__ ((packed)) { blst_fp X, Y; ulong inf;} blst_g1_affine;

typedef uint limb_t;
#define FP_LIMBS 12
#define FR_LIMBS 8
#define LIMB(fp,i) fp.v32[i]

typedef ulong limb6_t;
#define FP_LIMBS6 6
#define FR_LIMBS6 4
#define LIMB6(fp,i) fp.v64[i]

#define TO_LIMB_T(limb64)     (limb_t)limb64
__constant const blst_fp FP_ZERO = { .v64={0} };
__constant const blst_fr FR_ZERO = { .v64={0} };

__constant const limb_t BLS12_377_P0 = TO_LIMB_T(0x8508bfffffffffff);

__constant const blst_fp BLS12_377_P =   { .v64 = {
    (0x8508c00000000001), (0x170b5d4430000000),
    (0x1ef3622fba094800), (0x1a22d9f300f5138f),
    (0xc63b05c06ca1493b), (0x01ae3a4617c510ea)
}};

__constant const blst_fp BLS12_377_ONE = { .v64 = {
    (0x02cdffffffffff68), (0x51409f837fffffb1), 
    (0x9f7db3a98a7d3ff2), (0x7b4e97b76e7c6305), 
    (0x4cf495bf803c84e8), (0x008d6661e2fdf49a)
 }};

__constant const limb_t BLS12_377_FR_M0 = TO_LIMB_T(0XA117FFFFFFFFFFF);

__constant const blst_fr BLS12_377_FR_M = {.v64 = {
    (0x0a11800000000001), (0x59aa76fed0000001),
    (0x60b44d1e5c37b001), (0x12ab655e9a2ca556)
}};

__constant const blst_fr BLS12_377_FR_ONE = {.v64 = {
    (9015221291577245683), (8239323489949974514),
    (1646089257421115374), (958099254763297437),
}};

static inline bool is_fp_zero(const blst_fp p) {
    limb6_t zero = LIMB6(p, 0);
    __attribute__((opencl_unroll_hint))
    for (int i=1; i<FP_LIMBS6; ++i)
        zero |= LIMB6(p,i);
    return (zero == 0);    
}

static inline bool is_fp_eq(const blst_fp p1, const blst_fp p2) {
    __attribute__((opencl_unroll_hint))
    for (int i=0; i<FP_LIMBS6; ++i) {
        if (LIMB6(p1, i) != LIMB6(p2, i))
            return false;
    }
    return true;    
}

static inline bool is_fp_ge(const blst_fp left, const blst_fp right) {
    __attribute__((opencl_unroll_hint))
    for (int i = FP_LIMBS6-1; i >= 0; --i) {
        if (LIMB6(left, i) < LIMB6(right, i)) {
            return false;
        } else if (LIMB6(left, i) > LIMB6(right, i)) {
            return true;
        }
    }
    return true;
}

static inline bool is_fr_one(const blst_fr p1) {
    __attribute__((opencl_unroll_hint))
    for (int i=0; i<FR_LIMBS6; ++i) {
        if (LIMB6(p1, i) != LIMB6(BLS12_377_FR_ONE, i))
            return false;
    }
    return true;    
}

static inline bool is_fr_ge(const blst_fr left, const blst_fr right) {
    __attribute__((opencl_unroll_hint))
    for (int i = FR_LIMBS6-1; i >= 0; --i) {
        if (LIMB6(left, i) < LIMB6(right, i)) {
            return false;
        } else if (LIMB6(left,i) > LIMB6(right, i)) {
            return true;
        }
    }
    return true;
}

static inline bool is_p1_zero(const blst_p1 *p) {
    return is_fp_zero(p->Z);
}

static inline bool is_p1_affine_zero(const blst_p1_affine *p) {
    return is_fp_zero(p->X);
}

#define blst_fp_copy(desc, src)  (desc) = (src)
#define blst_fr_copy(desc, src)  (desc) = (src)
#define blst_p1_copy(desc, src)  (*desc) = (*src)

static inline blst_fp fp_sub_unsafe(blst_fp a, const blst_fp b) {
#if defined(OPENCL_NVIDIA)
   __asm(
      "sub.cc.u64  %0, %0, %6;\n\t"
      "subc.cc.u64 %1, %1, %7;\n\t"
      "subc.cc.u64 %2, %2, %8;\n\t"
      "subc.cc.u64 %3, %3, %9;\n\t"
      "subc.cc.u64 %4, %4, %10;\n\t"
      "subc.u64    %5, %5, %11;\n\t"
      : 
      "+l"(a.v64[0]),
      "+l"(a.v64[1]),
      "+l"(a.v64[2]),
      "+l"(a.v64[3]),
      "+l"(a.v64[4]),
      "+l"(a.v64[5])
      :
      "l"(b.v64[0]),
      "l"(b.v64[1]),
      "l"(b.v64[2]),
      "l"(b.v64[3]),
      "l"(b.v64[4]),
      "l"(b.v64[5])
    );
#else
    bool borrow = 0;
    __attribute__((opencl_unroll_hint))
    for(uint i = 0; i < FP_LIMBS6; i++) {
      limb6_t old = LIMB6(a, i);
      LIMB6(a, i) -= LIMB6(b, i) + borrow;
      borrow = borrow ? old <= LIMB6(a, i) : old < LIMB6(a, i);
    }
#endif
    return a;
}

static inline blst_fp fp_add_unsafe(blst_fp a, const blst_fp b) {
#if defined(OPENCL_NVIDIA)    
    __asm(
      "add.cc.u64  %0, %0, %6;\n\t"
      "addc.cc.u64 %1, %1, %7;\n\t"
      "addc.cc.u64 %2, %2, %8;\n\t"
      "addc.cc.u64 %3, %3, %9;\n\t"
      "addc.cc.u64 %4, %4, %10;\n\t"
      "addc.u64    %5, %5, %11;\n\t"
      : 
      "+l"(a.v64[0]),
      "+l"(a.v64[1]),
      "+l"(a.v64[2]),
      "+l"(a.v64[3]),
      "+l"(a.v64[4]),
      "+l"(a.v64[5])
      : 
      "l"(b.v64[0]),
      "l"(b.v64[1]),
      "l"(b.v64[2]),
      "l"(b.v64[3]),
      "l"(b.v64[4]),
      "l"(b.v64[5])
    );
#else
    bool carry = 0;
    __attribute__((opencl_unroll_hint))
    for(uint i = 0; i < FP_LIMBS6; i++) {
      limb6_t old = LIMB6(a, i);
      LIMB6(a, i) += LIMB6(b, i) + carry;
      carry = carry ? old >= LIMB6(a, i) : old > LIMB6(a, i);
    }
#endif
    return a;
}

#define blst_fp_add(r, a, b) r = fp_add_(a, b) 
static inline blst_fp fp_add_(blst_fp a, const blst_fp b) {
    a = fp_add_unsafe(a, b);
    if (is_fp_ge(a, BLS12_377_P)) 
        return fp_sub_unsafe(a, BLS12_377_P);
    return a;
}

#define blst_fp_sub(r, a, b) r = fp_sub_(a, b) 
static inline blst_fp fp_sub_(blst_fp a, const blst_fp b) {
#if defined(OPENCL_NVIDIA)
   __asm(
      "{"
      ".reg.pred p;\n\t"
      ".reg.u64 c;\n\t"
      "sub.cc.u64  %0, %0, %6;\n\t"
      "subc.cc.u64 %1, %1, %7;\n\t"
      "subc.cc.u64 %2, %2, %8;\n\t"
      "subc.cc.u64 %3, %3, %9;\n\t"
      "subc.cc.u64 %4, %4, %10;\n\t"
      "subc.cc.u64 %5, %5, %11;\n\t"
      "subc.u64 c, 0, 0;\n\t"
      "setp.ne.u64 p, c, 0;\n\t"
      "@p add.cc.u64  %0, %0, %12;\n\t"
      "@p addc.cc.u64 %1, %1, %13;\n\t"
      "@p addc.cc.u64 %2, %2, %14;\n\t"
      "@p addc.cc.u64 %3, %3, %15;\n\t"
      "@p addc.cc.u64 %4, %4, %16;\n\t"
      "@p addc.u64    %5, %5, %17;\n\t"
      "}"
      : 
      "+l"(a.v64[0]),
      "+l"(a.v64[1]),
      "+l"(a.v64[2]),
      "+l"(a.v64[3]),
      "+l"(a.v64[4]),
      "+l"(a.v64[5])
      :
      "l"(b.v64[0]),
      "l"(b.v64[1]),
      "l"(b.v64[2]),
      "l"(b.v64[3]),
      "l"(b.v64[4]),
      "l"(b.v64[5]),
      "l"(BLS12_377_P.v64[0]),
      "l"(BLS12_377_P.v64[1]),
      "l"(BLS12_377_P.v64[2]),
      "l"(BLS12_377_P.v64[3]),
      "l"(BLS12_377_P.v64[4]),
      "l"(BLS12_377_P.v64[5])
    );
#else
    if (!is_fp_ge(a, b)) 
        a = fp_add_unsafe(a, BLS12_377_P);
    a = fp_sub_unsafe(a, b);
#endif
    return a;
}

#if defined(OPENCL_NVIDIA)
// The Montgomery reduction here is based on Algorithm 14.32 in
// Handbook of Applied Cryptography
// <http://cacr.uwaterloo.ca/hac/about/chap14.pdf>.
static inline blst_fp fp_reduce(ulong r[12]) 
{
    blst_fp ret;
   __asm(
    "{\n\t"
    ".reg .u32 k0, hi;\n\t"
    ".reg .u32 p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11;\n\t"
    ".reg .u64 c, c2, r, lo;\n\t"
    ".reg .u64 r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11;\n\t"
    "mov.u64 lo, 0xFFFFFFFF;\n\t"
    "mov.u32 hi, 32;\n\t"
    "mov.u32 p1, 0x8508c000;\n\t"
    "mov.u32 p2, 0x30000000;\n\t"
    "mov.u32 p3, 0x170b5d44;\n\t"
    "mov.u32 p4, 0xba094800;\n\t"
    "mov.u32 p5, 0x1ef3622f;\n\t"
    "mov.u32 p6, 0x00f5138f;\n\t"
    "mov.u32 p7, 0x1a22d9f3;\n\t"
    "mov.u32 p8, 0x6ca1493b;\n\t"
    "mov.u32 p9, 0xc63b05c0;\n\t"
    "mov.u32 p10, 0x17c510ea;\n\t"
    "mov.u32 p11, 0x01ae3a46;\n\t"       
    "and.b64 r, %6, lo;\n\t"     
    "shl.b64 c, r, hi;\n\t"     
    "sub.u64 c, c, r;\n\t"       
    "cvt.u32.u64 k0, c;\n\t"
    "and.b64 r, c, lo;\n\t"      
    "add.cc.u64  %6 , %6 , r;\n\t"
    "mul.wide.u32 r , k0, p2;\n\t"   
    "addc.cc.u64 %7 , %7 , r;\n\t"
    "mul.wide.u32 r , k0, p4;\n\t"   
    "addc.cc.u64 %8 , %8 , r;\n\t"
    "mul.wide.u32 r , k0, p6;\n\t"   
    "addc.cc.u64 %9 , %9 , r;\n\t"
    "mul.wide.u32 r , k0, p8;\n\t"   
    "addc.cc.u64 %10, %10, r;\n\t"
    "mul.wide.u32 r , k0, p10;\n\t"  
    "addc.cc.u64 %11, %11, r;\n\t"
    "addc.u64    %0 , 0  , 0;\n\t"
    "mul.wide.u32 r0, k0, p1;\n\t"
    "mul.wide.u32 r1, k0, p3;\n\t"
    "mul.wide.u32 r2, k0, p5;\n\t"
    "mul.wide.u32 r3, k0, p7;\n\t"
    "mul.wide.u32 r4, k0, p9;\n\t"
    "mul.wide.u32 r5, k0, p11;\n\t"
    "shr.u64 r, %6, hi;\n\t"     
    "add.cc.u64 r0, r0, r;\n\t"   
    "addc.u64 c2, 0, 0;\n\t"
    "and.b64 r, r0, lo;\n\t"     
    "shl.b64 c, r, hi;\n\t"       
    "sub.u64 c, c, r;\n\t"       
    "cvt.u32.u64 k0, c;\n\t"
    "and.b64 r, c,  lo;\n\t"     
    "add.cc.u64  r0, r0, r;\n\t"
    "mad.wide.u32 r, k0, p2, c2;\n\t"
    "addc.cc.u64 r1, r1, r;\n\t"
    "mul.wide.u32 r, k0, p4;\n\t"    
    "addc.cc.u64 r2, r2, r;\n\t"
    "mul.wide.u32 r, k0, p6;\n\t"    
    "addc.cc.u64 r3, r3, r;\n\t"
    "mul.wide.u32 r, k0, p8;\n\t"    
    "addc.cc.u64 r4, r4, r;\n\t"
    "mul.wide.u32 r, k0, p10;\n\t"   
    "addc.cc.u64 r5, r5, r;\n\t"
    "addc.u64    r6, 0 , 0;\n\t"
    "mul.wide.u32 r, k0, p1;\n\t"    
    "add.cc.u64  %7 , %7 , r;\n\t"
    "mul.wide.u32 r, k0, p3;\n\t"    
    "addc.cc.u64 %8 , %8 , r;\n\t"
    "mul.wide.u32 r, k0, p5;\n\t"    
    "addc.cc.u64 %9 , %9 , r;\n\t"
    "mul.wide.u32 r, k0, p7;\n\t"    
    "addc.cc.u64 %10, %10, r;\n\t" 
    "mul.wide.u32 r, k0, p9;\n\t"    
    "addc.cc.u64 %11, %11, r;\n\t"
    "mul.wide.u32 r, k0, p11;\n\t"   
    "addc.u64    %0 , %0 , r;\n\t" 
    "shr.u64 r, r0, hi;\n\t"     
    "add.cc.u64 %7, %7, r;\n\t"   
    "addc.u64 c2, 0, 0;\n\t"
    "and.b64 r, %7, lo;\n\t"     
    "shl.b64 c, r, hi;\n\t"       
    "sub.u64 c, c, r;\n\t"       
    "cvt.u32.u64 k0, c;\n\t"
    "and.b64 r, c,  lo;\n\t"     
    "add.cc.u64  %7 , %7 , r;\n\t"
    "mad.wide.u32 r, k0, p2, c2;\n\t"
    "addc.cc.u64 %8 , %8 , r;\n\t"
    "mul.wide.u32 r, k0, p4;\n\t"    
    "addc.cc.u64 %9 , %9 , r;\n\t"
    "mul.wide.u32 r, k0, p6;\n\t"    
    "addc.cc.u64 %10, %10, r;\n\t"
    "mul.wide.u32 r, k0, p8;\n\t"    
    "addc.cc.u64 %11, %11, r;\n\t"
    "mul.wide.u32 r, k0, p10;\n\t"   
    "addc.cc.u64 %0 , %0 , r;\n\t"
    "addc.u64    %1 , 0  , 0;\n\t"
    "mul.wide.u32 r, k0, p1;\n\t"    
    "add.cc.u64  r1, r1, r;\n\t"
    "mul.wide.u32 r, k0, p3;\n\t"    
    "addc.cc.u64 r2, r2, r;\n\t"
    "mul.wide.u32 r, k0, p5;\n\t"    
    "addc.cc.u64 r3, r3, r;\n\t"
    "mul.wide.u32 r, k0, p7;\n\t"    
    "addc.cc.u64 r4, r4, r;\n\t"
    "mul.wide.u32 r, k0, p9;\n\t"    
    "addc.cc.u64 r5, r5, r;\n\t"
    "mul.wide.u32 r, k0, p11;\n\t"   
    "addc.u64    r6, r6, r;\n\t"
    "shr.u64 r, %7, hi;\n\t"     
    "add.cc.u64 r1, r1, r;\n\t"   
    "addc.u64 c2, 0, 0;\n\t"
    "and.b64 r, r1, lo;\n\t"     
    "shl.b64 c, r, hi;\n\t"       
    "sub.u64 c, c, r;\n\t"       
    "cvt.u32.u64 k0, c;\n\t"
    "and.b64 r, c,  lo;\n\t"     
    "add.cc.u64  r1, r1, r;\n\t"
    "mad.wide.u32 r, k0, p2, c2;\n\t"
    "addc.cc.u64 r2, r2, r;\n\t"
    "mul.wide.u32 r, k0, p4;\n\t"    
    "addc.cc.u64 r3, r3, r;\n\t"
    "mul.wide.u32 r, k0, p6;\n\t"    
    "addc.cc.u64 r4, r4, r;\n\t"
    "mul.wide.u32 r, k0, p8;\n\t"    
    "addc.cc.u64 r5, r5, r;\n\t"
    "mul.wide.u32 r, k0, p10;\n\t"   
    "addc.cc.u64 r6, r6, r;\n\t"
    "addc.u64    r7, 0 , 0;\n\t"
    "mul.wide.u32 r, k0, p1;\n\t"    
    "add.cc.u64  %8 , %8 , r;\n\t"
    "mul.wide.u32 r, k0, p3;\n\t"    
    "addc.cc.u64 %9 , %9 , r;\n\t"
    "mul.wide.u32 r, k0, p5;\n\t"    
    "addc.cc.u64 %10, %10, r;\n\t"
    "mul.wide.u32 r, k0, p7;\n\t"    
    "addc.cc.u64 %11, %11, r;\n\t"
    "mul.wide.u32 r, k0, p9;\n\t"    
    "addc.cc.u64 %0 , %0 , r;\n\t"
    "mul.wide.u32 r, k0, p11;\n\t"   
    "addc.u64    %1 , %1 , r;\n\t"
    "shr.u64 r, r1, hi;\n\t"     
    "add.cc.u64 %8, %8, r;\n\t"   
    "addc.u64 c2, 0, 0;\n\t"
    "and.b64 r, %8, lo;\n\t"     
    "shl.b64 c, r, hi;\n\t"       
    "sub.u64 c, c, r;\n\t"       
    "cvt.u32.u64 k0, c;\n\t"
    "and.b64 r, c,  lo;\n\t"     
    "add.cc.u64  %8 , %8 , r;\n\t"
    "mad.wide.u32 r, k0, p2, c2;\n\t"
    "addc.cc.u64 %9 , %9 , r;\n\t"
    "mul.wide.u32 r, k0, p4;\n\t"    
    "addc.cc.u64 %10, %10, r;\n\t"
    "mul.wide.u32 r, k0, p6;\n\t"    
    "addc.cc.u64 %11, %11, r;\n\t"
    "mul.wide.u32 r, k0, p8;\n\t"    
    "addc.cc.u64 %0 , %0 , r;\n\t"
    "mul.wide.u32 r, k0, p10;\n\t"   
    "addc.cc.u64 %1 , %1 , r;\n\t"
    "addc.u64    %2 , 0  , 0;\n\t"
    "mul.wide.u32 r, k0, p1;\n\t"    
    "add.cc.u64  r2, r2, r;\n\t"
    "mul.wide.u32 r, k0, p3;\n\t"    
    "addc.cc.u64 r3, r3, r;\n\t"
    "mul.wide.u32 r, k0, p5;\n\t"    
    "addc.cc.u64 r4, r4, r;\n\t"
    "mul.wide.u32 r, k0, p7;\n\t"    
    "addc.cc.u64 r5, r5, r;\n\t"
    "mul.wide.u32 r, k0, p9;\n\t"    
    "addc.cc.u64 r6, r6, r;\n\t"
    "mul.wide.u32 r, k0, p11;\n\t"   
    "addc.u64    r7, r7, r;\n\t"
    "shr.u64 r, %8, hi;\n\t"     
    "add.cc.u64 r2, r2, r;\n\t"   
    "addc.u64 c2, 0, 0;\n\t"
    "and.b64 r, r2, lo;\n\t"     
    "shl.b64 c, r, hi;\n\t"       
    "sub.u64 c, c, r;\n\t"       
    "cvt.u32.u64 k0, c;\n\t"
    "and.b64 r, c,  lo;\n\t"     
    "add.cc.u64  r2, r2, r;\n\t"
    "mad.wide.u32 r, k0, p2, c2;\n\t"
    "addc.cc.u64 r3, r3, r;\n\t"
    "mul.wide.u32 r, k0, p4;\n\t"    
    "addc.cc.u64 r4, r4, r;\n\t"
    "mul.wide.u32 r, k0, p6;\n\t"    
    "addc.cc.u64 r5, r5, r;\n\t"
    "mul.wide.u32 r, k0, p8;\n\t"    
    "addc.cc.u64 r6, r6, r;\n\t"
    "mul.wide.u32 r, k0, p10;\n\t"   
    "addc.cc.u64 r7, r7, r;\n\t"
    "addc.u64    r8, 0 , 0;\n\t"
    "mul.wide.u32 r, k0, p1;\n\t"    
    "add.cc.u64  %9 , %9 , r;\n\t"
    "mul.wide.u32 r, k0, p3;\n\t"    
    "addc.cc.u64 %10, %10, r;\n\t"
    "mul.wide.u32 r, k0, p5;\n\t"    
    "addc.cc.u64 %11, %11, r;\n\t"
    "mul.wide.u32 r, k0, p7;\n\t"    
    "addc.cc.u64 %0 , %0 , r;\n\t"
    "mul.wide.u32 r, k0, p9;\n\t"    
    "addc.cc.u64 %1 , %1 , r;\n\t"
    "mul.wide.u32 r, k0, p11;\n\t"   
    "addc.u64    %2 , %2 , r;\n\t"
    "shr.u64 r, r2, hi;\n\t"     
    "add.cc.u64 %9, %9, r;\n\t"   
    "addc.u64 c2, 0, 0;\n\t"
    "and.b64 r, %9, lo;\n\t"     
    "shl.b64 c, r, hi;\n\t"       
    "sub.u64 c, c, r;\n\t"       
    "cvt.u32.u64 k0, c;\n\t"
    "and.b64 r, c,  lo;\n\t"     
    "add.cc.u64  %9 , %9 , r;\n\t"
    "mad.wide.u32 r, k0, p2, c2;\n\t"
    "addc.cc.u64 %10, %10, r;\n\t"
    "mul.wide.u32 r, k0, p4;\n\t"    
    "addc.cc.u64 %11, %11, r;\n\t"
    "mul.wide.u32 r, k0, p6;\n\t"    
    "addc.cc.u64 %0 , %0 , r;\n\t"
    "mul.wide.u32 r, k0, p8;\n\t"    
    "addc.cc.u64 %1 , %1 , r;\n\t"
    "mul.wide.u32 r, k0, p10;\n\t"   
    "addc.cc.u64 %2 , %2 , r;\n\t"
    "addc.u64    %3 , 0  , 0;\n\t"
    "mul.wide.u32 r, k0, p1;\n\t"    
    "add.cc.u64  r3, r3, r;\n\t"
    "mul.wide.u32 r, k0, p3;\n\t"    
    "addc.cc.u64 r4, r4, r;\n\t"
    "mul.wide.u32 r, k0, p5;\n\t"    
    "addc.cc.u64 r5, r5, r;\n\t"
    "mul.wide.u32 r, k0, p7;\n\t"    
    "addc.cc.u64 r6, r6, r;\n\t"
    "mul.wide.u32 r, k0, p9;\n\t"    
    "addc.cc.u64 r7, r7, r;\n\t"
    "mul.wide.u32 r, k0, p11;\n\t"   
    "addc.u64    r8, r8, r;\n\t"
    "shr.u64 r, %9, hi;\n\t"     
    "add.cc.u64 r3, r3, r;\n\t"   
    "addc.u64 c2, 0, 0;\n\t"
    "and.b64 r, r3, lo;\n\t"     
    "shl.b64 c, r, hi;\n\t"       
    "sub.u64 c, c, r;\n\t"       
    "cvt.u32.u64 k0, c;\n\t"
    "and.b64 r, c,  lo;\n\t"     
    "add.cc.u64  r3, r3, r;\n\t"
    "mad.wide.u32 r, k0, p2, c2;\n\t"
    "addc.cc.u64 r4, r4, r;\n\t"
    "mul.wide.u32 r, k0, p4;\n\t"    
    "addc.cc.u64 r5, r5, r;\n\t"
    "mul.wide.u32 r, k0, p6;\n\t"    
    "addc.cc.u64 r6, r6, r;\n\t"
    "mul.wide.u32 r, k0, p8;\n\t"    
    "addc.cc.u64 r7, r7, r;\n\t"
    "mul.wide.u32 r, k0, p10;\n\t"   
    "addc.cc.u64 r8, r8, r;\n\t"
    "addc.u64    r9, 0 , 0;\n\t"
    "mul.wide.u32 r, k0, p1;\n\t"    
    "add.cc.u64  %10, %10, r;\n\t"
    "mul.wide.u32 r, k0, p3;\n\t"    
    "addc.cc.u64 %11, %11, r;\n\t"
    "mul.wide.u32 r, k0, p5;\n\t"    
    "addc.cc.u64 %0 , %0 , r;\n\t"
    "mul.wide.u32 r, k0, p7;\n\t"    
    "addc.cc.u64 %1 , %1 , r;\n\t"
    "mul.wide.u32 r, k0, p9;\n\t"    
    "addc.cc.u64 %2 , %2 , r;\n\t"
    "mul.wide.u32 r, k0, p11;\n\t"   
    "addc.u64    %3 , %3 , r;\n\t"
    "shr.u64 r, r3, hi;\n\t"     
    "add.cc.u64 %10, %10, r;\n\t"   
    "addc.u64 c2, 0, 0;\n\t"
    "and.b64 r, %10, lo;\n\t"    
    "shl.b64 c, r, hi;\n\t"         
    "sub.u64 c, c, r;\n\t"       
    "cvt.u32.u64 k0, c;\n\t"
    "and.b64 r, c,  lo;\n\t"     
    "add.cc.u64  %10, %10, r;\n\t"
    "mad.wide.u32 r, k0, p2, c2;\n\t"
    "addc.cc.u64 %11, %11, r;\n\t"
    "mul.wide.u32 r, k0, p4;\n\t"    
    "addc.cc.u64 %0 , %0 , r;\n\t"
    "mul.wide.u32 r, k0, p6;\n\t"    
    "addc.cc.u64 %1 , %1 , r;\n\t"
    "mul.wide.u32 r, k0, p8;\n\t"    
    "addc.cc.u64 %2 , %2 , r;\n\t"
    "mul.wide.u32 r, k0, p10;\n\t"   
    "addc.cc.u64 %3 , %3 , r;\n\t"
    "addc.u64    %4 , 0  , 0;\n\t"
    "mul.wide.u32 r, k0, p1;\n\t"    
    "add.cc.u64  r4, r4, r;\n\t"
    "mul.wide.u32 r, k0, p3;\n\t"    
    "addc.cc.u64 r5, r5, r;\n\t"
    "mul.wide.u32 r, k0, p5;\n\t"    
    "addc.cc.u64 r6, r6, r;\n\t"
    "mul.wide.u32 r, k0, p7;\n\t"    
    "addc.cc.u64 r7, r7, r;\n\t"
    "mul.wide.u32 r, k0, p9;\n\t"    
    "addc.cc.u64 r8, r8, r;\n\t"
    "mul.wide.u32 r, k0, p11;\n\t"   
    "addc.u64    r9, r9, r;\n\t"
    "shr.u64 r, %10, hi;\n\t"    
    "add.cc.u64 r4, r4, r;\n\t"   
    "addc.u64 c2, 0, 0;\n\t"
    "and.b64 r, r4, lo;\n\t"     
    "shl.b64 c, r, hi;\n\t"       
    "sub.u64 c, c, r;\n\t"       
    "cvt.u32.u64 k0, c;\n\t"
    "and.b64 r, c,  lo;\n\t"     
    "add.cc.u64  r4 , r4, r;\n\t"
    "mad.wide.u32 r, k0, p2, c2;\n\t"
    "addc.cc.u64 r5 , r5, r;\n\t"
    "mul.wide.u32 r, k0, p4;\n\t"    
    "addc.cc.u64 r6 , r6, r;\n\t"
    "mul.wide.u32 r, k0, p6;\n\t"    
    "addc.cc.u64 r7 , r7, r;\n\t"
    "mul.wide.u32 r, k0, p8;\n\t"    
    "addc.cc.u64 r8 , r8, r;\n\t"
    "mul.wide.u32 r, k0, p10;\n\t"   
    "addc.cc.u64 r9 , r9, r;\n\t"
    "addc.u64    r10, 0 , 0;\n\t"
    "mul.wide.u32 r, k0, p1;\n\t"    
    "add.cc.u64  %11, %11, r;\n\t"
    "mul.wide.u32 r, k0, p3;\n\t"    
    "addc.cc.u64 %0 , %0 , r;\n\t"
    "mul.wide.u32 r, k0, p5;\n\t"    
    "addc.cc.u64 %1 , %1 , r;\n\t"
    "mul.wide.u32 r, k0, p7;\n\t"    
    "addc.cc.u64 %2 , %2 , r;\n\t"
    "mul.wide.u32 r, k0, p9;\n\t"    
    "addc.cc.u64 %3 , %3 , r;\n\t"
    "mul.wide.u32 r, k0, p11;\n\t"   
    "addc.u64    %4 , %4 , r;\n\t"
    "shr.u64 r, r4, hi;\n\t"     
    "add.cc.u64 %11, %11, r;\n\t"   
    "addc.u64 c2, 0, 0;\n\t"
    "and.b64 r, %11, lo;\n\t"    
    "shl.b64 c, r, hi;\n\t"         
    "sub.u64 c, c, r;\n\t"       
    "cvt.u32.u64 k0, c;\n\t"
    "and.b64 r, c,  lo;\n\t"     
    "add.cc.u64  %11, %11, r;\n\t"
    "mad.wide.u32 r, k0, p2, c2;\n\t"
    "addc.cc.u64 %0 , %0 , r;\n\t"
    "mul.wide.u32 r, k0, p4;\n\t"    
    "addc.cc.u64 %1 , %1 , r;\n\t"
    "mul.wide.u32 r, k0, p6;\n\t"    
    "addc.cc.u64 %2 , %2 , r;\n\t"
    "mul.wide.u32 r, k0, p8;\n\t"    
    "addc.cc.u64 %3 , %3 , r;\n\t"
    "mul.wide.u32 r, k0, p10;\n\t"   
    "addc.cc.u64 %4 , %4 , r;\n\t"
    "addc.u64    %5 , 0  , 0;\n\t"
    "mul.wide.u32 r, k0, p1;\n\t"    
    "add.cc.u64  r5 , r5 , r;\n\t"
    "mul.wide.u32 r, k0, p3;\n\t"    
    "addc.cc.u64 r6 , r6 , r;\n\t"
    "mul.wide.u32 r, k0, p5;\n\t"    
    "addc.cc.u64 r7 , r7 , r;\n\t"
    "mul.wide.u32 r, k0, p7;\n\t"    
    "addc.cc.u64 r8 , r8 , r;\n\t"
    "mul.wide.u32 r, k0, p9;\n\t"    
    "addc.cc.u64 r9 , r9 , r;\n\t"
    "mul.wide.u32 r, k0, p11;\n\t"   
    "addc.u64    r10, r10, r;\n\t"
    "shr.u64 r, %11, hi;\n\t"     
    "add.cc.u64 r5, r5, r;\n\t"   
    "addc.u64 c2, 0, 0;\n\t"
    "and.b64 r, r5, lo;\n\t"     
    "shl.b64 c, r, hi;\n\t"       
    "sub.u64 c, c, r;\n\t"       
    "cvt.u32.u64 k0, c;\n\t"
    "and.b64 r, c,  lo;\n\t"     
    "add.cc.u64  r5 , r5 , r;\n\t"
    "mad.wide.u32 r, k0, p2, c2;\n\t"
    "addc.cc.u64 r6 , r6 , r;\n\t"
    "mul.wide.u32 r, k0, p4;\n\t"    
    "addc.cc.u64 r7 , r7 , r;\n\t"
    "mul.wide.u32 r, k0, p6;\n\t"    
    "addc.cc.u64 r8 , r8 , r;\n\t"
    "mul.wide.u32 r, k0, p8;\n\t"    
    "addc.cc.u64 r9 , r9 , r;\n\t"
    "mul.wide.u32 r, k0, p10;\n\t"   
    "addc.cc.u64 r10, r10, r;\n\t"
    "addc.u64    r11, 0  , 0;\n\t"
    "mul.wide.u32 r, k0, p1;\n\t"    
    "add.cc.u64  %0 , %0 , r;\n\t"
    "mul.wide.u32 r, k0, p3;\n\t"    
    "addc.cc.u64 %1 , %1 , r;\n\t"
    "mul.wide.u32 r, k0, p5;\n\t"    
    "addc.cc.u64 %2 , %2 , r;\n\t"
    "mul.wide.u32 r, k0, p7;\n\t"    
    "addc.cc.u64 %3 , %3 , r;\n\t"
    "mul.wide.u32 r, k0, p9;\n\t"    
    "addc.cc.u64 %4 , %4 , r;\n\t"
    "mul.wide.u32 r, k0, p11;\n\t"   
    "addc.u64    %5 , %5 , r;\n\t"
    "shr.u64 c, r10, hi;\n\t"
    "shl.b64 r11, r11, hi;\n\t"
    "or.b64 r11, r11, c;\n\t"
    "shr.u64 c, r9 , hi;\n\t"
    "shl.b64 r10, r10, hi;\n\t"
    "or.b64 r10, r10, c;\n\t"
    "shr.u64 c, r8 , hi;\n\t"
    "shl.b64 r9 , r9 , hi;\n\t"
    "or.b64 r9 , r9 , c;\n\t"
    "shr.u64 c, r7 , hi;\n\t"
    "shl.b64 r8 , r8 , hi;\n\t"
    "or.b64 r8 , r8 , c;\n\t"
    "shr.u64 c, r6 , hi;\n\t"
    "shl.b64 r7 , r7 , hi;\n\t"
    "or.b64 r7 , r7 , c;\n\t"
    "shr.u64 c, r5 , hi;\n\t"
    "shl.b64 r6 , r6 , hi;\n\t"
    "or.b64 r6 , r6 , c;\n\t"
    "add.cc.u64  %0, %0, r6;\n\t"
    "addc.cc.u64 %1, %1, r7;\n\t"
    "addc.cc.u64 %2, %2, r8;\n\t"
    "addc.cc.u64 %3, %3, r9;\n\t"
    "addc.cc.u64 %4, %4, r10;\n\t"
    "addc.u64    %5, %5, r11;\n\t"
    "add.cc.u64  %0, %0, %12;\n\t"
    "addc.cc.u64 %1, %1, %13;\n\t"
    "addc.cc.u64 %2, %2, %14;\n\t"
    "addc.cc.u64 %3, %3, %15;\n\t"
    "addc.cc.u64 %4, %4, %16;\n\t"
    "addc.u64    %5, %5, %17;\n\t"
    "}"
    : 
    "=l"(ret.v64[0]),
    "=l"(ret.v64[1]),
    "=l"(ret.v64[2]),
    "=l"(ret.v64[3]),
    "=l"(ret.v64[4]),
    "=l"(ret.v64[5])
    : 
    "l"(r[0]),
    "l"(r[1]),
    "l"(r[2]),
    "l"(r[3]),
    "l"(r[4]),
    "l"(r[5]),
    "l"(r[6]),
    "l"(r[7]),
    "l"(r[8]),
    "l"(r[9]),
    "l"(r[10]),
    "l"(r[11])
    );

    if (is_fp_ge(ret, BLS12_377_P)) 
        return fp_sub_unsafe(ret, BLS12_377_P);
    return ret;
}
#else
// // Returns a * b + c, puts the carry in d
// static ulong mac(ulong a, ulong b, ulong c, ulong *d) {
//     ulong lo = a * b + c;
//     *d = mad_hi(a, b, (ulong)(lo < c));
//     return lo;
// }
// // Returns a * b + c + d, puts the carry in d
// static ulong mac_with_carry(ulong a, ulong b, ulong c, ulong *d) {
//     ulong lo = a * b + c;
//     ulong hi = mad_hi(a, b, (ulong)(lo < c));
//     a = lo;
//     lo += *d;
//     hi += (lo < a);
//     *d = hi;
//     return lo;
// }
// // Returns a + b, puts the carry in d
// static ulong add_with_carry(ulong a, ulong *b) {
//     ulong lo = a + *b;
//     *b = lo < a;
//     return lo;
// }


// Returns a * b + c, puts the carry in d
static inline uint mac(uint a, uint b, uint c, uint *d) {
  ulong res = convert_ulong(a) * convert_ulong(b) + convert_ulong(c);
  *d = convert_uint(res >> 32);
  return convert_uint(res);
}

// Returns a * b + c + d, puts the carry in d
static inline uint mac_with_carry(uint a, uint b, uint c, uint *d) {
  ulong res = convert_ulong(a) * convert_ulong(b) + convert_ulong(c) + convert_ulong(*d);
  *d = convert_uint(res >> 32);
  return convert_uint(res);
}

// Returns a + b, puts the carry in b
static inline uint add_with_carry(uint a, uint *b) {
    uint lo = a + *b;
    *b = lo < a;
    return lo;
}
#endif

#define blst_fp_mul(r, a, b) r = fp_mul_(a, b)
static inline blst_fp fp_mul_(const blst_fp a, const blst_fp b) 
{
#if defined(OPENCL_NVIDIA)
  ulong r[12];
 __asm (
    "{\n\t"
    ".reg .u64 c, t;\n\t"
    "mul.wide.u32 %0, %12, %25;\n\t"
    "mul.wide.u32 %1, %12, %27;\n\t"
    "mul.wide.u32 %2, %12, %29;\n\t"
    "mul.wide.u32 %3, %12, %31;\n\t"
    "mul.wide.u32 %4, %12, %33;\n\t"
    "mul.wide.u32 %5, %12, %35;\n\t"
    "mul.wide.u32 c, %13, %24;\n\t"   
    "add.cc.u64 %0, %0, c;\n\t"
    "mul.wide.u32 c, %13, %26;\n\t"   
    "addc.cc.u64 %1, %1, c;\n\t"
    "mul.wide.u32 c, %13, %28;\n\t"   
    "addc.cc.u64 %2, %2, c;\n\t"
    "mul.wide.u32 c, %13, %30;\n\t"   
    "addc.cc.u64 %3, %3, c;\n\t"
    "mul.wide.u32 c, %13, %32;\n\t"   
    "addc.cc.u64 %4, %4, c;\n\t"
    "mul.wide.u32 c, %13, %34;\n\t"   
    "addc.cc.u64 %5, %5, c;\n\t"
    "addc.u64 %6, 0 , 0;\n\t"
    "mul.wide.u32 c, %14, %25;\n\t"   
    "add.cc.u64 %1, %1, c;\n\t"
    "mul.wide.u32 c, %14, %27;\n\t"   
    "addc.cc.u64 %2, %2, c;\n\t"
    "mul.wide.u32 c, %14, %29;\n\t"   
    "addc.cc.u64 %3, %3, c;\n\t"
    "mul.wide.u32 c, %14, %31;\n\t"   
    "addc.cc.u64 %4, %4, c;\n\t"
    "mul.wide.u32 c, %14, %33;\n\t"   
    "addc.cc.u64 %5, %5, c;\n\t"
    "mul.wide.u32 c, %14, %35;\n\t"   
    "addc.cc.u64 %6, %6, c;\n\t"
    "addc.u64 %7, 0 , 0;\n\t"
    "mul.wide.u32 c, %15, %24;\n\t"   
    "add.cc.u64 %1, %1, c;\n\t"
    "mul.wide.u32 c, %15, %26;\n\t"   
    "addc.cc.u64 %2, %2, c;\n\t"
    "mul.wide.u32 c, %15, %28;\n\t"   
    "addc.cc.u64 %3, %3, c;\n\t"
    "mul.wide.u32 c, %15, %30;\n\t"   
    "addc.cc.u64 %4, %4, c;\n\t"
    "mul.wide.u32 c, %15, %32;\n\t"   
    "addc.cc.u64 %5, %5, c;\n\t"
    "mul.wide.u32 c, %15, %34;\n\t"   
    "addc.cc.u64 %6, %6, c;\n\t"
    "addc.u64 %7, %7, 0;\n\t"
    "mul.wide.u32 c, %16, %25;\n\t"   
    "add.cc.u64 %2, %2, c;\n\t"
    "mul.wide.u32 c, %16, %27;\n\t"   
    "addc.cc.u64 %3, %3, c;\n\t"
    "mul.wide.u32 c, %16, %29;\n\t"   
    "addc.cc.u64 %4, %4, c;\n\t"
    "mul.wide.u32 c, %16, %31;\n\t"   
    "addc.cc.u64 %5, %5, c;\n\t"
    "mul.wide.u32 c, %16, %33;\n\t"   
    "addc.cc.u64 %6, %6, c;\n\t"
    "mul.wide.u32 c, %16, %35;\n\t"   
    "addc.cc.u64 %7, %7, c;\n\t"
    "addc.u64 %8, 0 , 0;\n\t"
    "mul.wide.u32 c, %17, %24;\n\t"   
    "add.cc.u64 %2, %2, c;\n\t"
    "mul.wide.u32 c, %17, %26;\n\t"   
    "addc.cc.u64 %3, %3, c;\n\t"
    "mul.wide.u32 c, %17, %28;\n\t"   
    "addc.cc.u64 %4, %4, c;\n\t"
    "mul.wide.u32 c, %17, %30;\n\t"   
    "addc.cc.u64 %5, %5, c;\n\t"
    "mul.wide.u32 c, %17, %32;\n\t"   
    "addc.cc.u64 %6, %6, c;\n\t"
    "mul.wide.u32 c, %17, %34;\n\t"   
    "addc.cc.u64 %7, %7, c;\n\t"
    "addc.u64 %8, %8, 0;\n\t"
    "mul.wide.u32 c, %18, %25;\n\t"   
    "add.cc.u64 %3, %3, c;\n\t"
    "mul.wide.u32 c, %18, %27;\n\t"   
    "addc.cc.u64 %4, %4, c;\n\t"
    "mul.wide.u32 c, %18, %29;\n\t"   
    "addc.cc.u64 %5, %5, c;\n\t"
    "mul.wide.u32 c, %18, %31;\n\t"   
    "addc.cc.u64 %6, %6, c;\n\t"
    "mul.wide.u32 c, %18, %33;\n\t"   
    "addc.cc.u64 %7, %7, c;\n\t"
    "mul.wide.u32 c, %18, %35;\n\t"   
    "addc.cc.u64 %8, %8, c;\n\t"
    "addc.u64 %9, 0 , 0;\n\t"
    "mul.wide.u32 c, %19, %24;\n\t"   
    "add.cc.u64 %3, %3, c;\n\t"
    "mul.wide.u32 c, %19, %26;\n\t"   
    "addc.cc.u64 %4, %4, c;\n\t"
    "mul.wide.u32 c, %19, %28;\n\t"   
    "addc.cc.u64 %5, %5, c;\n\t"
    "mul.wide.u32 c, %19, %30;\n\t"   
    "addc.cc.u64 %6, %6, c;\n\t"
    "mul.wide.u32 c, %19, %32;\n\t"   
    "addc.cc.u64 %7, %7, c;\n\t"
    "mul.wide.u32 c, %19, %34;\n\t"   
    "addc.cc.u64 %8, %8, c;\n\t"
    "addc.u64 %9, %9, 0;\n\t"
    "mul.wide.u32 c, %20, %25;\n\t"   
    "add.cc.u64 %4, %4, c;\n\t"
    "mul.wide.u32 c, %20, %27;\n\t"   
    "addc.cc.u64 %5, %5, c;\n\t"
    "mul.wide.u32 c, %20, %29;\n\t"   
    "addc.cc.u64 %6, %6, c;\n\t"
    "mul.wide.u32 c, %20, %31;\n\t"   
    "addc.cc.u64 %7, %7, c;\n\t"
    "mul.wide.u32 c, %20, %33;\n\t"   
    "addc.cc.u64 %8, %8, c;\n\t"
    "mul.wide.u32 c, %20, %35;\n\t"   
    "addc.cc.u64 %9, %9, c;\n\t"
    "addc.u64 %10, 0, 0;\n\t"
    "mul.wide.u32 c, %21, %24;\n\t"   
    "add.cc.u64 %4, %4, c;\n\t"
    "mul.wide.u32 c, %21, %26;\n\t"   
    "addc.cc.u64 %5, %5, c;\n\t"
    "mul.wide.u32 c, %21, %28;\n\t"   
    "addc.cc.u64 %6, %6, c;\n\t"
    "mul.wide.u32 c, %21, %30;\n\t"   
    "addc.cc.u64 %7, %7, c;\n\t"
    "mul.wide.u32 c, %21, %32;\n\t"   
    "addc.cc.u64 %8, %8, c;\n\t"
    "mul.wide.u32 c, %21, %34;\n\t"   
    "addc.cc.u64 %9, %9, c;\n\t"
    "addc.u64 %10, %10, 0;\n\t"
    "mul.wide.u32 c, %22, %25;\n\t"   
    "add.cc.u64 %5 , %5 , c;\n\t"
    "mul.wide.u32 c, %22, %27;\n\t"   
    "addc.cc.u64 %6 , %6 , c;\n\t"
    "mul.wide.u32 c, %22, %29;\n\t"   
    "addc.cc.u64 %7 , %7 , c;\n\t"
    "mul.wide.u32 c, %22, %31;\n\t"   
    "addc.cc.u64 %8 , %8 , c;\n\t"
    "mul.wide.u32 c, %22, %33;\n\t"   
    "addc.cc.u64 %9 , %9 , c;\n\t"
    "mul.wide.u32 c, %22, %35;\n\t"   
    "addc.cc.u64 %10, %10, c;\n\t"
    "addc.u64 %11, 0, 0;\n\t"
    "mul.wide.u32 c, %23, %24;\n\t"   
    "add.cc.u64 %5 , %5 , c;\n\t"
    "mul.wide.u32 c, %23, %26;\n\t"   
    "addc.cc.u64 %6 , %6 , c;\n\t"
    "mul.wide.u32 c, %23, %28;\n\t"   
    "addc.cc.u64 %7 , %7 , c;\n\t"
    "mul.wide.u32 c, %23, %30;\n\t"   
    "addc.cc.u64 %8 , %8 , c;\n\t"
    "mul.wide.u32 c, %23, %32;\n\t"   
    "addc.cc.u64 %9 , %9 , c;\n\t"
    "mul.wide.u32 c, %23, %34;\n\t"   
    "addc.cc.u64 %10, %10, c;\n\t"
    "addc.u64 %11, %11, 0;\n\t"
    "shr.u64 c, %10, 32;\n\t"  
    "shl.b64 %11, %11, 32;\n\t"   
    "or.b64 %11, %11, c;\n\t"
    "shr.u64 c, %9 , 32;\n\t"  
    "shl.b64 %10, %10, 32;\n\t"   
    "or.b64 %10, %10, c;\n\t"
    "shr.u64 c, %8 , 32;\n\t"  
    "shl.b64 %9 , %9 , 32;\n\t"   
    "or.b64 %9 , %9 , c;\n\t"
    "shr.u64 c, %7 , 32;\n\t"  
    "shl.b64 %8 , %8 , 32;\n\t"   
    "or.b64 %8 , %8 , c;\n\t"
    "shr.u64 c, %6 , 32;\n\t"  
    "shl.b64 %7 , %7 , 32;\n\t"   
    "or.b64 %7 , %7 , c;\n\t"
    "shr.u64 c, %5 , 32;\n\t"  
    "shl.b64 %6 , %6 , 32;\n\t"   
    "or.b64 %6 , %6 , c;\n\t"
    "shr.u64 c, %4 , 32;\n\t"  
    "shl.b64 %5 , %5 , 32;\n\t"   
    "or.b64 %5 , %5 , c;\n\t"
    "shr.u64 c, %3 , 32;\n\t"  
    "shl.b64 %4 , %4 , 32;\n\t"   
    "or.b64 %4 , %4 , c;\n\t"
    "shr.u64 c, %2 , 32;\n\t"  
    "shl.b64 %3 , %3 , 32;\n\t"   
    "or.b64 %3 , %3 , c;\n\t"
    "shr.u64 c, %1 , 32;\n\t"  
    "shl.b64 %2 , %2 , 32;\n\t"   
    "or.b64 %2 , %2 , c;\n\t"
    "shr.u64 c, %0 , 32;\n\t"  
    "shl.b64 %1 , %1 , 32;\n\t"   
    "or.b64 %1 , %1 , c;\n\t"
    "shl.b64 %0 , %0 , 32;\n\t"
    "mul.wide.u32 c, %12, %24;\n\t"     
    "add.cc.u64 %0, %0, c;\n\t"
    "mul.wide.u32 c, %12, %26;\n\t"     
    "addc.cc.u64 %1, %1, c;\n\t"
    "mul.wide.u32 c, %12, %28;\n\t"     
    "addc.cc.u64 %2, %2, c;\n\t"
    "mul.wide.u32 c, %12, %30;\n\t"     
    "addc.cc.u64 %3, %3, c;\n\t"
    "mul.wide.u32 c, %12, %32;\n\t"     
    "addc.cc.u64 %4, %4, c;\n\t"
    "mul.wide.u32 c, %12, %34;\n\t"     
    "addc.cc.u64 %5, %5, c;\n\t"
    "addc.u64 t, 0, 0;\n\t"
    "mul.wide.u32 c, %13, %25;\n\t"     
    "add.cc.u64 %1, %1, c;\n\t"
    "mul.wide.u32 c, %13, %27;\n\t"     
    "addc.cc.u64 %2, %2, c;\n\t"
    "mul.wide.u32 c, %13, %29;\n\t"     
    "addc.cc.u64 %3, %3, c;\n\t"
    "mul.wide.u32 c, %13, %31;\n\t"     
    "addc.cc.u64 %4, %4, c;\n\t"
    "mul.wide.u32 c, %13, %33;\n\t"     
    "addc.cc.u64 %5, %5, c;\n\t"
    "mad.wide.u32 c, %13, %35, t;\n\t"  
    "addc.cc.u64 %6, %6, c;\n\t"
    "addc.u64 t, 0, 0;\n\t"
    "mul.wide.u32 c, %14, %24;\n\t"     
    "add.cc.u64 %1, %1, c;\n\t"
    "mul.wide.u32 c, %14, %26;\n\t"     
    "addc.cc.u64 %2, %2, c;\n\t"
    "mul.wide.u32 c, %14, %28;\n\t"     
    "addc.cc.u64 %3, %3, c;\n\t"
    "mul.wide.u32 c, %14, %30;\n\t"     
    "addc.cc.u64 %4, %4, c;\n\t"
    "mul.wide.u32 c, %14, %32;\n\t"     
    "addc.cc.u64 %5, %5, c;\n\t"
    "mul.wide.u32 c, %14, %34;\n\t"     
    "addc.cc.u64 %6, %6, c;\n\t"
    "addc.u64 t, t, 0;\n\t"
    "mul.wide.u32 c, %15, %25;\n\t"     
    "add.cc.u64 %2, %2, c;\n\t"
    "mul.wide.u32 c, %15, %27;\n\t"     
    "addc.cc.u64 %3, %3, c;\n\t"
    "mul.wide.u32 c, %15, %29;\n\t"     
    "addc.cc.u64 %4, %4, c;\n\t"
    "mul.wide.u32 c, %15, %31;\n\t"     
    "addc.cc.u64 %5, %5, c;\n\t"
    "mul.wide.u32 c, %15, %33;\n\t"     
    "addc.cc.u64 %6, %6, c;\n\t"
    "mad.wide.u32 c, %15, %35, t;\n\t"  
    "addc.cc.u64 %7, %7, c;\n\t"
    "addc.u64 t, 0, 0;\n\t"
    "mul.wide.u32 c, %16, %24;\n\t"     
    "add.cc.u64 %2, %2, c;\n\t"
    "mul.wide.u32 c, %16, %26;\n\t"     
    "addc.cc.u64 %3, %3, c;\n\t"
    "mul.wide.u32 c, %16, %28;\n\t"     
    "addc.cc.u64 %4, %4, c;\n\t"
    "mul.wide.u32 c, %16, %30;\n\t"     
    "addc.cc.u64 %5, %5, c;\n\t"
    "mul.wide.u32 c, %16, %32;\n\t"     
    "addc.cc.u64 %6, %6, c;\n\t"
    "mul.wide.u32 c, %16, %34;\n\t"     
    "addc.cc.u64 %7, %7, c;\n\t"
    "addc.u64 t, t, 0;\n\t"
    "mul.wide.u32 c, %17, %25;\n\t"     
    "add.cc.u64 %3, %3, c;\n\t"
    "mul.wide.u32 c, %17, %27;\n\t"     
    "addc.cc.u64 %4, %4, c;\n\t"
    "mul.wide.u32 c, %17, %29;\n\t"     
    "addc.cc.u64 %5, %5, c;\n\t"
    "mul.wide.u32 c, %17, %31;\n\t"     
    "addc.cc.u64 %6, %6, c;\n\t"
    "mul.wide.u32 c, %17, %33;\n\t"     
    "addc.cc.u64 %7, %7, c;\n\t"
    "mad.wide.u32 c, %17, %35, t;\n\t"  
    "addc.cc.u64 %8, %8, c;\n\t"
    "addc.u64 t, 0, 0;\n\t"
    "mul.wide.u32 c, %18, %24;\n\t"     
    "add.cc.u64 %3, %3, c;\n\t"
    "mul.wide.u32 c, %18, %26;\n\t"     
    "addc.cc.u64 %4, %4, c;\n\t"
    "mul.wide.u32 c, %18, %28;\n\t"     
    "addc.cc.u64 %5, %5, c;\n\t"
    "mul.wide.u32 c, %18, %30;\n\t"     
    "addc.cc.u64 %6, %6, c;\n\t"
    "mul.wide.u32 c, %18, %32;\n\t"     
    "addc.cc.u64 %7, %7, c;\n\t"
    "mul.wide.u32 c, %18, %34;\n\t"     
    "addc.cc.u64 %8, %8, c;\n\t"
    "addc.u64 t, t, 0;\n\t"
    "mul.wide.u32 c, %19, %25;\n\t"     
    "add.cc.u64 %4, %4, c;\n\t"
    "mul.wide.u32 c, %19, %27;\n\t"     
    "addc.cc.u64 %5, %5, c;\n\t"
    "mul.wide.u32 c, %19, %29;\n\t"     
    "addc.cc.u64 %6, %6, c;\n\t"
    "mul.wide.u32 c, %19, %31;\n\t"     
    "addc.cc.u64 %7, %7, c;\n\t"
    "mul.wide.u32 c, %19, %33;\n\t"     
    "addc.cc.u64 %8, %8, c;\n\t"
    "mad.wide.u32 c, %19, %35, t;\n\t"  
    "addc.cc.u64 %9, %9, c;\n\t"
    "addc.u64 t, 0, 0;\n\t"
    "mul.wide.u32 c, %20, %24;\n\t"     
    "add.cc.u64 %4, %4, c;\n\t"
    "mul.wide.u32 c, %20, %26;\n\t"     
    "addc.cc.u64 %5, %5, c;\n\t"
    "mul.wide.u32 c, %20, %28;\n\t"     
    "addc.cc.u64 %6, %6, c;\n\t"
    "mul.wide.u32 c, %20, %30;\n\t"     
    "addc.cc.u64 %7, %7, c;\n\t"
    "mul.wide.u32 c, %20, %32;\n\t"     
    "addc.cc.u64 %8, %8, c;\n\t"
    "mul.wide.u32 c, %20, %34;\n\t"     
    "addc.cc.u64 %9, %9, c;\n\t"
    "addc.u64 t, t, 0;\n\t"
    "mul.wide.u32 c, %21, %25;\n\t"     
    "add.cc.u64 %5 , %5 , c;\n\t"
    "mul.wide.u32 c, %21, %27;\n\t"     
    "addc.cc.u64 %6 , %6 , c;\n\t"
    "mul.wide.u32 c, %21, %29;\n\t"     
    "addc.cc.u64 %7 , %7 , c;\n\t"
    "mul.wide.u32 c, %21, %31;\n\t"     
    "addc.cc.u64 %8 , %8 , c;\n\t"
    "mul.wide.u32 c, %21, %33;\n\t"     
    "addc.cc.u64 %9 , %9 , c;\n\t"
    "mad.wide.u32 c, %21, %35, t;\n\t"  
    "addc.cc.u64 %10, %10, c;\n\t"
    "addc.u64 t, 0, 0;\n\t"
    "mul.wide.u32 c, %22, %24;\n\t"     
    "add.cc.u64 %5 , %5 , c;\n\t"
    "mul.wide.u32 c, %22, %26;\n\t"     
    "addc.cc.u64 %6 , %6 , c;\n\t"
    "mul.wide.u32 c, %22, %28;\n\t"     
    "addc.cc.u64 %7 , %7 , c;\n\t"
    "mul.wide.u32 c, %22, %30;\n\t"     
    "addc.cc.u64 %8 , %8 , c;\n\t"
    "mul.wide.u32 c, %22, %32;\n\t"     
    "addc.cc.u64 %9 , %9 , c;\n\t"
    "mul.wide.u32 c, %22, %34;\n\t"     
    "addc.cc.u64 %10, %10, c;\n\t"
    "addc.u64 t, t, 0;\n\t"
    "mul.wide.u32 c, %23, %25;\n\t"      
    "add.cc.u64 %6 , %6 , c;\n\t"
    "mul.wide.u32 c, %23, %27;\n\t"      
    "addc.cc.u64 %7 , %7 , c;\n\t"
    "mul.wide.u32 c, %23, %29;\n\t"      
    "addc.cc.u64 %8 , %8 , c;\n\t"
    "mul.wide.u32 c, %23, %31;\n\t"      
    "addc.cc.u64 %9 , %9 , c;\n\t"
    "mul.wide.u32 c, %23, %33;\n\t"      
    "addc.cc.u64 %10, %10, c;\n\t"
    "mad.wide.u32 c, %23, %35, t;\n\t"   
    "addc.cc.u64 %11, %11, c;\n\t"
    "}"
    : 
    "=l"(r[0]),
    "=l"(r[1]),
    "=l"(r[2]),
    "=l"(r[3]),
    "=l"(r[4]),
    "=l"(r[5]),
    "=l"(r[6]),
    "=l"(r[7]),
    "=l"(r[8]),
    "=l"(r[9]),
    "=l"(r[10]),
    "=l"(r[11])
    : 
    "r"(a.v32[0]),
    "r"(a.v32[1]),
    "r"(a.v32[2]),
    "r"(a.v32[3]),
    "r"(a.v32[4]),
    "r"(a.v32[5]),
    "r"(a.v32[6]),
    "r"(a.v32[7]),
    "r"(a.v32[8]),
    "r"(a.v32[9]),
    "r"(a.v32[10]),
    "r"(a.v32[11]),
    "r"(b.v32[0]),
    "r"(b.v32[1]),
    "r"(b.v32[2]),
    "r"(b.v32[3]),
    "r"(b.v32[4]),
    "r"(b.v32[5]),
    "r"(b.v32[6]),
    "r"(b.v32[7]),
    "r"(b.v32[8]),
    "r"(b.v32[9]),
    "r"(b.v32[10]),
    "r"(b.v32[11])
    );

   return fp_reduce(r);
#else
    limb_t carry1, carry2;
    blst_fp fp = {.v64={0}};
    __attribute__((opencl_unroll_hint))
    for(uint i = 0; i < FP_LIMBS; i++) {
        LIMB(fp, 0) = mac(LIMB(a, 0), LIMB(b, i), LIMB(fp, 0), &carry1);
        limb_t m = BLS12_377_P0 * LIMB(fp, 0);
        mac(m, LIMB(BLS12_377_P, 0), LIMB(fp, 0), &carry2);
        __attribute__((opencl_unroll_hint))
        for(uint j = 1; j < FP_LIMBS; j++) {
            LIMB(fp, j) = mac_with_carry(LIMB(a, j), LIMB(b, i), LIMB(fp, j), &carry1);
            LIMB(fp, j - 1) = mac_with_carry(m, LIMB(BLS12_377_P, j), LIMB(fp, j), &carry2);
        }
        LIMB(fp, FP_LIMBS - 1) = carry1 + carry2;
    }

    if (is_fp_ge(fp, BLS12_377_P)) 
        return fp_sub_unsafe(fp, BLS12_377_P);
    return fp;
#endif
}

#define blst_fp_sqr(r, a) r = fp_sqr_(a)
static inline blst_fp fp_sqr_(const blst_fp a)
{
#if defined(OPENCL_NVIDIA)
    union { ulong v64[12]; uint v32[24];} r;
    __asm(
    "{\n\t"
    ".reg .u64 r;\n\t"
    ".reg .u32 c, l, h;\n\t"
    "mul.lo.u32 %1 , %24, %25;\n\t"
    "mul.hi.u32 %2 , %24, %25;\n\t"
    "mul.lo.u32 %3 , %24, %27;\n\t"
    "mul.hi.u32 %4 , %24, %27;\n\t"
    "mul.lo.u32 %5 , %24, %29;\n\t"
    "mul.hi.u32 %6 , %24, %29;\n\t"
    "mul.lo.u32 %7 , %24, %31;\n\t"
    "mul.hi.u32 %8 , %24, %31;\n\t"
    "mul.lo.u32 %9 , %24, %33;\n\t"
    "mul.hi.u32 %10, %24, %33;\n\t"
    "mul.lo.u32 %11, %24, %35;\n\t"
    "mul.hi.u32 %12, %24, %35;\n\t"
    "mad.lo.cc.u32 %3 , %25, %26, %3 ;\n\t"
    "madc.hi.cc.u32 %4 , %25, %26, %4 ;\n\t"
    "madc.lo.cc.u32 %5 , %25, %28, %5 ;\n\t"
    "madc.hi.cc.u32 %6 , %25, %28, %6 ;\n\t"
    "madc.lo.cc.u32 %7 , %25, %30, %7 ;\n\t"
    "madc.hi.cc.u32 %8 , %25, %30, %8 ;\n\t"
    "madc.lo.cc.u32 %9 , %25, %32, %9 ;\n\t"
    "madc.hi.cc.u32 %10, %25, %32, %10;\n\t"
    "madc.lo.cc.u32 %11, %25, %34, %11;\n\t"
    "madc.hi.cc.u32 %12, %25, %34, %12;\n\t"
    "addc.u32 %13,  0 , 0;\n\t"
    "mad.lo.cc.u32 %5 , %26, %27, %5 ;\n\t"
    "madc.hi.cc.u32 %6 , %26, %27, %6 ;\n\t"
    "madc.lo.cc.u32 %7 , %26, %29, %7 ;\n\t"
    "madc.hi.cc.u32 %8 , %26, %29, %8 ;\n\t"
    "madc.lo.cc.u32 %9 , %26, %31, %9 ;\n\t"
    "madc.hi.cc.u32 %10, %26, %31, %10;\n\t"
    "madc.lo.cc.u32 %11, %26, %33, %11;\n\t"
    "madc.hi.cc.u32 %12, %26, %33, %12;\n\t"
    "madc.lo.cc.u32 %13, %26, %35, %13;\n\t"
    "madc.hi.u32 %14, %26, %35, 0;\n\t"
    "mad.lo.cc.u32 %7 , %27, %28, %7 ;\n\t"
    "madc.hi.cc.u32 %8 , %27, %28, %8 ;\n\t"
    "madc.lo.cc.u32 %9 , %27, %30, %9 ;\n\t"
    "madc.hi.cc.u32 %10, %27, %30, %10;\n\t"
    "madc.lo.cc.u32 %11, %27, %32, %11;\n\t"
    "madc.hi.cc.u32 %12, %27, %32, %12;\n\t"
    "madc.lo.cc.u32 %13, %27, %34, %13;\n\t"
    "madc.hi.cc.u32 %14, %27, %34, %14;\n\t"
    "addc.u32 %15,  0 , 0;\n\t"
    "mad.lo.cc.u32 %9 , %28, %29, %9 ;\n\t"
    "madc.hi.cc.u32 %10, %28, %29, %10;\n\t"
    "madc.lo.cc.u32 %11, %28, %31, %11;\n\t"
    "madc.hi.cc.u32 %12, %28, %31, %12;\n\t"
    "madc.lo.cc.u32 %13, %28, %33, %13;\n\t"
    "madc.hi.cc.u32 %14, %28, %33, %14;\n\t"
    "madc.lo.cc.u32 %15, %28, %35, %15;\n\t"
    "madc.hi.u32 %16, %28, %35, 0;\n\t"
    "mad.lo.cc.u32 %11, %29, %30, %11;\n\t"
    "madc.hi.cc.u32 %12, %29, %30, %12;\n\t"
    "madc.lo.cc.u32 %13, %29, %32, %13;\n\t"
    "madc.hi.cc.u32 %14, %29, %32, %14;\n\t"
    "madc.lo.cc.u32 %15, %29, %34, %15;\n\t"
    "madc.hi.cc.u32 %16, %29, %34, %16;\n\t"
    "addc.u32 %17,  0 , 0;\n\t"
    "mad.lo.cc.u32 %13, %30, %31, %13;\n\t"
    "madc.hi.cc.u32 %14, %30, %31, %14;\n\t"
    "madc.lo.cc.u32 %15, %30, %33, %15;\n\t"
    "madc.hi.cc.u32 %16, %30, %33, %16;\n\t"
    "madc.lo.cc.u32 %17, %30, %35, %17;\n\t"
    "madc.hi.u32 %18, %30, %35, 0;\n\t"
    "mad.lo.cc.u32 %15, %31, %32, %15;\n\t"
    "madc.hi.cc.u32 %16, %31, %32, %16;\n\t"
    "madc.lo.cc.u32 %17, %31, %34, %17;\n\t"
    "madc.hi.cc.u32 %18, %31, %34, %18;\n\t"
    "addc.u32 %19,  0 , 0;\n\t"
    "mad.lo.cc.u32 %17, %32, %33, %17;\n\t"
    "madc.hi.cc.u32 %18, %32, %33, %18;\n\t"
    "madc.lo.cc.u32 %19, %32, %35, %19;\n\t"
    "madc.hi.u32 %20, %32, %35, 0;\n\t"
    "mad.lo.cc.u32 %19, %33, %34, %19;\n\t"
    "madc.hi.cc.u32 %20, %33, %34, %20;\n\t"
    "addc.u32 %21,  0 , 0;\n\t"
    "mad.lo.cc.u32 %21, %34, %35, %21;\n\t"
    "madc.hi.u32 %22, %34, %35, 0;\n\t"
    "mad.lo.cc.u32 %2 , %24, %26, %2 ;\n\t"
    "madc.hi.cc.u32 %3 , %24, %26, %3 ;\n\t"
    "madc.lo.cc.u32 %4 , %24, %28, %4 ;\n\t"
    "madc.hi.cc.u32 %5 , %24, %28, %5 ;\n\t"
    "madc.lo.cc.u32 %6 , %24, %30, %6 ;\n\t"
    "madc.hi.cc.u32 %7 , %24, %30, %7 ;\n\t"
    "madc.lo.cc.u32 %8 , %24, %32, %8 ;\n\t"
    "madc.hi.cc.u32 %9 , %24, %32, %9 ;\n\t"
    "madc.lo.cc.u32 %10, %24, %34, %10;\n\t"
    "madc.hi.cc.u32 %11, %24, %34, %11;\n\t"
    "addc.u32 c, 0, 0;\n\t"
    "mad.lo.cc.u32 %4 , %25, %27, %4 ;\n\t"
    "madc.hi.cc.u32 %5 , %25, %27, %5 ;\n\t"
    "madc.lo.cc.u32 %6 , %25, %29, %6 ;\n\t"
    "madc.hi.cc.u32 %7 , %25, %29, %7 ;\n\t"
    "madc.lo.cc.u32 %8 , %25, %31, %8 ;\n\t"
    "madc.hi.cc.u32 %9 , %25, %31, %9 ;\n\t"
    "madc.lo.cc.u32 %10, %25, %33, %10;\n\t"
    "madc.hi.cc.u32 %11, %25, %33, %11;\n\t"
    "madc.lo.cc.u32 l  , %25, %35, c;\n\t"
    "madc.hi.u32 h  , %25, %35, 0;\n\t"
    "add.cc.u32 %12, %12, l;\n\t"
    "addc.cc.u32 %13, %13, h;\n\t"
    "addc.u32 c, 0, 0;\n\t"
    "mad.lo.cc.u32 %6 , %26, %28, %6 ;\n\t"
    "madc.hi.cc.u32 %7 , %26, %28, %7 ;\n\t"
    "madc.lo.cc.u32 %8 , %26, %30, %8 ;\n\t"
    "madc.hi.cc.u32 %9 , %26, %30, %9 ;\n\t"
    "madc.lo.cc.u32 %10, %26, %32, %10;\n\t"
    "madc.hi.cc.u32 %11, %26, %32, %11;\n\t"
    "madc.lo.cc.u32 %12, %26, %34, %12;\n\t"
    "madc.hi.cc.u32 %13, %26, %34, %13;\n\t"
    "addc.u32 c, c, 0;\n\t"
    "mad.lo.cc.u32 %8 , %27, %29, %8 ;\n\t"
    "madc.hi.cc.u32 %9 , %27, %29, %9 ;\n\t"
    "madc.lo.cc.u32 %10, %27, %31, %10;\n\t"
    "madc.hi.cc.u32 %11, %27, %31, %11;\n\t"
    "madc.lo.cc.u32 %12, %27, %33, %12;\n\t"
    "madc.hi.cc.u32 %13, %27, %33, %13;\n\t"
    "madc.lo.cc.u32 l  , %27, %35, c;\n\t"
    "madc.hi.u32 h  , %27, %35, 0;\n\t"
    "add.cc.u32 %14, %14, l;\n\t"
    "addc.cc.u32 %15, %15, h;\n\t"
    "addc.u32 c, 0, 0;\n\t"
    "mul.wide.u32 r, %28, %30;\n\t"
    "mov.b64 {l, h}, r;\n\t"
    "add.cc.u32 %10, %10, l;\n\t"  
    "addc.cc.u32 %11, %11, h;\n\t"
    "mul.wide.u32 r, %28, %32;\n\t"
    "mov.b64 {l, h}, r;\n\t"
    "addc.cc.u32 %12, %12, l;\n\t" 
    "addc.cc.u32 %13, %13, h;\n\t"
    "mul.wide.u32 r, %28, %34;\n\t"
    "mov.b64 {l, h}, r;\n\t"
    "addc.cc.u32 %14, %14, l;\n\t" 
    "addc.cc.u32 %15, %15, h;\n\t"
    "addc.u32 c, c, 0;\n\t"
    "mul.wide.u32 r, %29, %31;\n\t"
    "mov.b64 {l, h}, r;\n\t"
    "add.cc.u32 %12, %12, l;\n\t"  
    "addc.cc.u32 %13, %13, h;\n\t"
    "mul.wide.u32 r, %29, %33;\n\t"
    "mov.b64 {l, h}, r;\n\t"
    "addc.cc.u32 %14, %14, l;\n\t" 
    "addc.cc.u32 %15, %15, h;\n\t"
    "mul.wide.u32 r, %29, %35;\n\t"
    "mov.b64 {l, h}, r;\n\t"
    "addc.cc.u32 l, l, c;\n\t"     
    "addc.u32 h, h, 0;\n\t"
    "add.cc.u32 %16, %16, l;\n\t"
    "addc.cc.u32 %17, %17, h;\n\t"
    "addc.u32 c, 0, 0;\n\t"
    "mul.wide.u32 r, %30, %32;\n\t"
    "mov.b64 {l, h}, r;\n\t"
    "add.cc.u32 %14, %14, l;\n\t"  
    "addc.cc.u32 %15, %15, h;\n\t"
    "mul.wide.u32 r, %30, %34;\n\t"
    "mov.b64 {l, h}, r;\n\t"
    "addc.cc.u32 %16, %16, l;\n\t" 
    "addc.cc.u32 %17, %17, h;\n\t"
    "addc.u32 c, c, 0;\n\t"
    "mul.wide.u32 r, %31, %33;\n\t"
    "mov.b64 {l, h}, r;\n\t"
    "add.cc.u32 %16, %16, l;\n\t"  
    "addc.cc.u32 %17, %17, h;\n\t"
    "mul.wide.u32 r, %31, %35;\n\t"
    "mov.b64 {l, h}, r;\n\t"
    "addc.cc.u32 l, l, c;\n\t"     
    "addc.u32 h, h, 0;\n\t"
    "add.cc.u32 %18, %18, l;\n\t"
    "addc.cc.u32 %19, %19, h;\n\t"
    "addc.u32 c, 0, 0;\n\t"
    "mul.wide.u32 r, %32, %34;\n\t"
    "mov.b64 {l, h}, r;\n\t"
    "add.cc.u32 %18, %18, l;\n\t"  
    "addc.cc.u32 %19, %19, h;\n\t"
    "addc.u32 c, c, 0;\n\t"
    "mul.wide.u32 r, %33, %35;\n\t"
    "mov.b64 {l, h}, r;\n\t"
    "add.cc.u32 l, l, c;\n\t"      
    "addc.u32 h, h, 0;\n\t"
    "add.cc.u32 %20, %20, l;\n\t"
    "addc.cc.u32 %21, %21, h;\n\t"
    "addc.u32 %22, %22, 0;\n\t"
    "shr.u32 %23, %22, 31;\n\t"   
    "shr.u32  c , %21, 31;\n\t"
    "shl.b32 %22, %22, 1;\n\t"  
    "or.b32 %22, %22, c;\n\t"
    "shr.u32  c , %20, 31;\n\t"
    "shl.b32 %21, %21, 1;\n\t"  
    "or.b32 %21, %21, c;\n\t"
    "shr.u32  c , %19, 31;\n\t"
    "shl.b32 %20, %20, 1;\n\t"  
    "or.b32 %20, %20, c;\n\t"
    "shr.u32  c , %18, 31;\n\t"
    "shl.b32 %19, %19, 1;\n\t"  
    "or.b32 %19, %19, c;\n\t"
    "shr.u32  c , %17, 31;\n\t"
    "shl.b32 %18, %18, 1;\n\t"  
    "or.b32 %18, %18, c;\n\t"
    "shr.u32  c , %16, 31;\n\t"
    "shl.b32 %17, %17, 1;\n\t"  
    "or.b32 %17, %17, c;\n\t"
    "shr.u32  c , %15, 31;\n\t"
    "shl.b32 %16, %16, 1;\n\t"  
    "or.b32 %16, %16, c;\n\t"
    "shr.u32  c , %14, 31;\n\t"
    "shl.b32 %15, %15, 1;\n\t"  
    "or.b32 %15, %15, c;\n\t"
    "shr.u32  c , %13, 31;\n\t"
    "shl.b32 %14, %14, 1;\n\t"  
    "or.b32 %14, %14, c;\n\t"
    "shr.u32  c , %12, 31;\n\t"
    "shl.b32 %13, %13, 1;\n\t"  
    "or.b32 %13, %13, c;\n\t"
    "shr.u32  c , %11, 31;\n\t"
    "shl.b32 %12, %12, 1;\n\t"  
    "or.b32 %12, %12, c;\n\t"
    "shr.u32  c , %10, 31;\n\t"
    "shl.b32 %11, %11, 1;\n\t"  
    "or.b32 %11, %11, c;\n\t"
    "shr.u32  c , %9 , 31;\n\t"
    "shl.b32 %10, %10, 1;\n\t"  
    "or.b32 %10, %10, c;\n\t"
    "shr.u32  c , %8 , 31;\n\t"
    "shl.b32 %9 , %9 , 1;\n\t"  
    "or.b32 %9 , %9 , c;\n\t"
    "shr.u32  c , %7 , 31;\n\t"
    "shl.b32 %8 , %8 , 1;\n\t"  
    "or.b32 %8 , %8 , c;\n\t"
    "shr.u32  c , %6 , 31;\n\t"
    "shl.b32 %7 , %7 , 1;\n\t"  
    "or.b32 %7 , %7 , c;\n\t"
    "shr.u32  c , %5 , 31;\n\t"
    "shl.b32 %6 , %6 , 1;\n\t"  
    "or.b32 %6 , %6 , c;\n\t"
    "shr.u32  c , %4 , 31;\n\t"
    "shl.b32 %5 , %5 , 1;\n\t"  
    "or.b32 %5 , %5 , c;\n\t"
    "shr.u32  c , %3 , 31;\n\t"
    "shl.b32 %4 , %4 , 1;\n\t"  
    "or.b32 %4 , %4 , c;\n\t"
    "shr.u32  c , %2 , 31;\n\t"
    "shl.b32 %3 , %3 , 1;\n\t"  
    "or.b32 %3 , %3 , c;\n\t"
    "shr.u32  c , %1 , 31;\n\t"
    "shl.b32 %2 , %2 , 1;\n\t"  
    "or.b32 %2 , %2 , c;\n\t"
    "shl.b32 %1 , %1 , 1;\n\t"  
    "mul.lo.u32 %0 , %24, %24;\n\t"
    "mad.hi.cc.u32 %1 , %24, %24, %1 ;\n\t"
    "madc.lo.cc.u32 %2 , %25, %25, %2 ;\n\t"
    "madc.hi.cc.u32 %3 , %25, %25, %3 ;\n\t"
    "madc.lo.cc.u32 %4 , %26, %26, %4 ;\n\t"
    "madc.hi.cc.u32 %5 , %26, %26, %5 ;\n\t"
    "madc.lo.cc.u32 %6 , %27, %27, %6 ;\n\t"
    "madc.hi.cc.u32 %7 , %27, %27, %7 ;\n\t"
    "madc.lo.cc.u32 %8 , %28, %28, %8 ;\n\t"
    "madc.hi.cc.u32 %9 , %28, %28, %9 ;\n\t"
    "madc.lo.cc.u32 %10, %29, %29, %10;\n\t"
    "madc.hi.cc.u32 %11, %29, %29, %11;\n\t"
    "madc.lo.cc.u32 %12, %30, %30, %12;\n\t"
    "madc.hi.cc.u32 %13, %30, %30, %13;\n\t"
    "madc.lo.cc.u32 %14, %31, %31, %14;\n\t"
    "madc.hi.cc.u32 %15, %31, %31, %15;\n\t"
    "madc.lo.cc.u32 %16, %32, %32, %16;\n\t"
    "madc.hi.cc.u32 %17, %32, %32, %17;\n\t"
    "madc.lo.cc.u32 %18, %33, %33, %18;\n\t"
    "madc.hi.cc.u32 %19, %33, %33, %19;\n\t"
    "madc.lo.cc.u32 %20, %34, %34, %20;\n\t"
    "madc.hi.cc.u32 %21, %34, %34, %21;\n\t"
    "madc.lo.cc.u32 %22, %35, %35, %22;\n\t"
    "madc.hi.u32 %23, %35, %35, %23;\n\t"
    "}\n\t"
    :
    "=r"(r.v32[0]),
    "=r"(r.v32[1]),
    "=r"(r.v32[2]),
    "=r"(r.v32[3]),
    "=r"(r.v32[4]),
    "=r"(r.v32[5]),
    "=r"(r.v32[6]),
    "=r"(r.v32[7]),
    "=r"(r.v32[8]),
    "=r"(r.v32[9]),
    "=r"(r.v32[10]),
    "=r"(r.v32[11]),
    "=r"(r.v32[12]),
    "=r"(r.v32[13]),
    "=r"(r.v32[14]),
    "=r"(r.v32[15]),
    "=r"(r.v32[16]),
    "=r"(r.v32[17]),
    "=r"(r.v32[18]),
    "=r"(r.v32[19]),
    "=r"(r.v32[20]),
    "=r"(r.v32[21]),
    "=r"(r.v32[22]),
    "=r"(r.v32[23])
    :
    "r"(a.v32[0]),
    "r"(a.v32[1]),
    "r"(a.v32[2]),
    "r"(a.v32[3]),
    "r"(a.v32[4]),
    "r"(a.v32[5]),
    "r"(a.v32[6]),
    "r"(a.v32[7]),
    "r"(a.v32[8]),
    "r"(a.v32[9]),
    "r"(a.v32[10]),
    "r"(a.v32[11])
   );

    return fp_reduce(r.v64);
#else
    // limb_t r[FP_LIMBS+FP_LIMBS];
    // limb_t carry=0;
    // r[1 ] = mac_with_carry(LIMB(a, 0), LIMB(a, 1 ), 0, &carry);
    // r[2 ] = mac_with_carry(LIMB(a, 0), LIMB(a, 2 ), 0, &carry);
    // r[3 ] = mac_with_carry(LIMB(a, 0), LIMB(a, 3 ), 0, &carry);
    // r[4 ] = mac_with_carry(LIMB(a, 0), LIMB(a, 4 ), 0, &carry);
    // r[5 ] = mac_with_carry(LIMB(a, 0), LIMB(a, 5 ), 0, &carry);
    // r[6 ] = mac_with_carry(LIMB(a, 0), LIMB(a, 6 ), 0, &carry);
    // r[7 ] = mac_with_carry(LIMB(a, 0), LIMB(a, 7 ), 0, &carry);
    // r[8 ] = mac_with_carry(LIMB(a, 0), LIMB(a, 8 ), 0, &carry);
    // r[9 ] = mac_with_carry(LIMB(a, 0), LIMB(a, 9 ), 0, &carry);
    // r[10] = mac_with_carry(LIMB(a, 0), LIMB(a, 10), 0, &carry);
    // r[11] = mac_with_carry(LIMB(a, 0), LIMB(a, 11), 0, &carry);
    // r[12] = carry;


    // carry=0;
    // r[3 ] = mac_with_carry(LIMB(a, 1), LIMB(a, 2 ), r[3 ], &carry);
    // r[4 ] = mac_with_carry(LIMB(a, 1), LIMB(a, 3 ), r[4 ], &carry);
    // r[5 ] = mac_with_carry(LIMB(a, 1), LIMB(a, 4 ), r[5 ], &carry);
    // r[6 ] = mac_with_carry(LIMB(a, 1), LIMB(a, 5 ), r[6 ], &carry);
    // r[7 ] = mac_with_carry(LIMB(a, 1), LIMB(a, 6 ), r[7 ], &carry);
    // r[8 ] = mac_with_carry(LIMB(a, 1), LIMB(a, 7 ), r[8 ], &carry);
    // r[9 ] = mac_with_carry(LIMB(a, 1), LIMB(a, 8 ), r[9 ], &carry);
    // r[10] = mac_with_carry(LIMB(a, 1), LIMB(a, 9 ), r[10], &carry);
    // r[11] = mac_with_carry(LIMB(a, 1), LIMB(a, 10), r[11], &carry);
    // r[12] = mac_with_carry(LIMB(a, 1), LIMB(a, 11), r[12], &carry);
    // r[13] = carry;


    // carry=0;
    // r[5 ] = mac_with_carry(LIMB(a, 2), LIMB(a, 3 ), r[5 ], &carry);
    // r[6 ] = mac_with_carry(LIMB(a, 2), LIMB(a, 4 ), r[6 ], &carry);
    // r[7 ] = mac_with_carry(LIMB(a, 2), LIMB(a, 5 ), r[7 ], &carry);
    // r[8 ] = mac_with_carry(LIMB(a, 2), LIMB(a, 6 ), r[8 ], &carry);
    // r[9 ] = mac_with_carry(LIMB(a, 2), LIMB(a, 7 ), r[9 ], &carry);
    // r[10] = mac_with_carry(LIMB(a, 2), LIMB(a, 8 ), r[10], &carry);
    // r[11] = mac_with_carry(LIMB(a, 2), LIMB(a, 9 ), r[11], &carry);
    // r[12] = mac_with_carry(LIMB(a, 2), LIMB(a, 10), r[12], &carry);
    // r[13] = mac_with_carry(LIMB(a, 2), LIMB(a, 11), r[13], &carry);
    // r[14] = carry;

    return fp_mul_(a,a);
#endif
}

static inline blst_fr fr_sub_unchecked(blst_fr a, const blst_fr b)
{
#if defined(OPENCL_NVIDIA)    
   __asm(
      "sub.cc.u64  %0, %0, %4;\n\t"
      "subc.cc.u64 %1, %1, %5;\n\t"
      "subc.cc.u64 %2, %2, %6;\n\t"
      "subc.u64    %3, %3, %7;\n\t"
      : 
      "+l"(a.v64[0]),
      "+l"(a.v64[1]),
      "+l"(a.v64[2]),
      "+l"(a.v64[3])
      : 
      "l"(b.v64[0]),
      "l"(b.v64[1]),
      "l"(b.v64[2]),
      "l"(b.v64[3])
    );
#else
    bool borrow = 0;
    __attribute__((opencl_unroll_hint))
    for(uint i = 0; i < FR_LIMBS6; i++) {
      limb6_t old = LIMB6(a, i);
      LIMB6(a, i) -= LIMB6(b, i) + borrow;
      borrow = borrow ? old <= LIMB6(a, i) : old < LIMB6(a, i);
    }
#endif
   
    return a;
}

static inline blst_fr fr_add_unchecked(blst_fr a, const blst_fr b)
{
#if defined(OPENCL_NVIDIA)
    __asm(
      "add.cc.u64  %0, %4, %8;\n\t"
      "addc.cc.u64 %1, %5, %9;\n\t"
      "addc.cc.u64 %2, %6, %10;\n\t"
      "addc.u64    %3, %7, %11;\n\t"
      : 
      "+l"(a.v64[0]),
      "+l"(a.v64[1]),
      "+l"(a.v64[2]),
      "+l"(a.v64[3])
      : 
      "l"(b.v64[0]),
      "l"(b.v64[1]),
      "l"(b.v64[2]),
      "l"(b.v64[3])
    );
#else
    bool carry = 0;
    __attribute__((opencl_unroll_hint))
    for(uint i = 0; i < FR_LIMBS6; i++) {
      limb6_t old = LIMB6(a, i);
      LIMB6(a, i) += LIMB6(b, i) + carry;
      carry = carry ? old >= LIMB6(a, i) : old > LIMB6(a, i);
    }
#endif
    return a;
}

#define blst_fr_sub(r, a, b) r = fr_sub_(a, b)
static inline blst_fr fr_sub_(blst_fr a, const blst_fr b)
{
    if (!is_fr_ge(a, b)) 
        a = fr_add_unchecked(a, BLS12_377_FR_M);
    return fr_sub_unchecked(a, b);
}

#define blst_fr_add(r, a, b) r = fr_add_(a, b)
static inline blst_fr fr_add_(blst_fr a, const blst_fr b)
{
    a = fr_add_unchecked(a, b);
    if (is_fr_ge(a, BLS12_377_FR_M))
        return fr_sub_unchecked(a, BLS12_377_FR_M);
    return a;
}

#if defined(OPENCL_NVIDIA)
static inline blst_fr fr_reduce(ulong r[8])
{
    blst_fr ret;
    __asm(
    "{\n\t"
    ".reg .u32 k0, hi;\n\t"
    ".reg .u32 p1, p2, p3, p4, p5, p6, p7;\n\t"
    ".reg .u64 c, c2, r, lo;\n\t"
    ".reg .u64 r0, r1, r2, r3, r4, r5, r6, r7;\n\t"
    "mov.u64 lo, 0xFFFFFFFF;\n\t"
    "mov.u32 hi, 32;\n\t"
    "mov.u32 p1, 0x0a118000;\n\t"
    "mov.u32 p2, 0xd0000001;\n\t"
    "mov.u32 p3, 0x59aa76fe;\n\t"
    "mov.u32 p4, 0x5c37b001;\n\t"
    "mov.u32 p5, 0x60b44d1e;\n\t"
    "mov.u32 p6, 0x9a2ca556;\n\t"
    "mov.u32 p7, 0x12ab655e;\n\t"
    "and.b64 r, %4, lo;\n\t"      
    "shl.b64 c, r, hi;\n\t"
    "sub.u64 c, c, r;\n\t"        
    "cvt.u32.u64 k0, c;\n\t"
    "and.b64 r, c, lo;\n\t"       
    "add.cc.u64  %4 , %4 , r;\n\t"
    "mul.wide.u32 r , k0, p2;\n\t"    
    "addc.cc.u64 %5 , %5 , r;\n\t"
    "mul.wide.u32 r , k0, p4;\n\t"    
    "addc.cc.u64 %6 , %6 , r;\n\t"
    "mul.wide.u32 r , k0, p6;\n\t"    
    "addc.cc.u64 %7 , %7 , r;\n\t"
    "addc.u64    %0 , 0  , 0;\n\t"
    "mul.wide.u32 r0, k0, p1;\n\t"
    "mul.wide.u32 r1, k0, p3;\n\t"
    "mul.wide.u32 r2, k0, p5;\n\t"
    "mul.wide.u32 r3, k0, p7;\n\t"
    "shr.u64 r, %4, hi;\n\t"     
    "add.cc.u64 r0, r0, r;\n\t"   
    "addc.u64 c2, 0, 0;\n\t"
    "and.b64 r, r0, lo;\n\t"     
    "shl.b64 c, r, hi;\n\t"       
    "sub.u64 c, c, r;\n\t"       
    "cvt.u32.u64 k0, c;\n\t"
    "and.b64 r, c,  lo;\n\t"     
    "add.cc.u64  r0, r0, r;\n\t"
    "mad.wide.u32 r, k0, p2, c2;\n\t"
    "addc.cc.u64 r1, r1, r;\n\t"
    "mul.wide.u32 r, k0, p4;\n\t"    
    "addc.cc.u64 r2, r2, r;\n\t"
    "mul.wide.u32 r, k0, p6;\n\t"    
    "addc.cc.u64 r3, r3, r;\n\t"
    "addc.u64    r4, 0 , 0;\n\t"
    "mul.wide.u32 r, k0, p1;\n\t"    
    "add.cc.u64  %5, %5, r;\n\t"
    "mul.wide.u32 r, k0, p3;\n\t"    
    "addc.cc.u64 %6, %6, r;\n\t"
    "mul.wide.u32 r, k0, p5;\n\t"    
    "addc.cc.u64 %7, %7, r;\n\t"
    "mul.wide.u32 r, k0, p7;\n\t"    
    "addc.u64    %0, %0, r;\n\t" 
    "shr.u64 r, r0, hi;\n\t"     
    "add.cc.u64 %5, %5, r;\n\t"   
    "addc.u64 c2, 0, 0;\n\t"
    "and.b64 r, %5, lo;\n\t"     
    "shl.b64 c, r, hi;\n\t"       
    "sub.u64 c, c, r;\n\t"       
    "cvt.u32.u64 k0, c;\n\t"
    "and.b64 r, c,  lo;\n\t"     
    "add.cc.u64  %5, %5, r;\n\t"
    "mad.wide.u32 r, k0, p2, c2;\n\t"
    "addc.cc.u64 %6, %6, r;\n\t"
    "mul.wide.u32 r, k0, p4;\n\t"    
    "addc.cc.u64 %7, %7, r;\n\t"
    "mul.wide.u32 r, k0, p6;\n\t"    
    "addc.cc.u64 %0, %0, r;\n\t"
    "addc.u64    %1, 0 , 0;\n\t"
    "mul.wide.u32 r, k0, p1;\n\t"    
    "add.cc.u64  r1, r1, r;\n\t"
    "mul.wide.u32 r, k0, p3;\n\t"    
    "addc.cc.u64 r2, r2, r;\n\t"
    "mul.wide.u32 r, k0, p5;\n\t"    
    "addc.cc.u64 r3, r3, r;\n\t"
    "mul.wide.u32 r, k0, p7;\n\t"    
    "addc.u64    r4, r4, r;\n\t"
    "shr.u64 r, %5, hi;\n\t"     
    "add.cc.u64 r1, r1, r;\n\t"   
    "addc.u64 c2, 0, 0;\n\t"
    "and.b64 r, r1, lo;\n\t"     
    "shl.b64 c, r, hi;\n\t"       
    "sub.u64 c, c, r;\n\t"       
    "cvt.u32.u64 k0, c;\n\t"
    "and.b64 r, c,  lo;\n\t"     
    "add.cc.u64  r1, r1, r;\n\t"
    "mad.wide.u32 r, k0, p2, c2;\n\t"
    "addc.cc.u64 r2, r2, r;\n\t"
    "mul.wide.u32 r, k0, p4;\n\t"    
    "addc.cc.u64 r3, r3, r;\n\t"
    "mul.wide.u32 r, k0, p6;\n\t"    
    "addc.cc.u64 r4, r4, r;\n\t"
    "addc.u64    r5, 0 , 0;\n\t"
    "mul.wide.u32 r, k0, p1;\n\t"    
    "add.cc.u64  %6, %6, r;\n\t"
    "mul.wide.u32 r, k0, p3;\n\t"    
    "addc.cc.u64 %7, %7, r;\n\t"
    "mul.wide.u32 r, k0, p5;\n\t"    
    "addc.cc.u64 %0, %0, r;\n\t"
    "mul.wide.u32 r, k0, p7;\n\t"    
    "addc.u64    %1, %1, r;\n\t"
    "shr.u64 r, r1, hi;\n\t"     
    "add.cc.u64 %6, %6, r;\n\t"   
    "addc.u64 c2, 0, 0;\n\t"
    "and.b64 r, %6, lo;\n\t"     
    "shl.b64 c, r, hi;\n\t"       
    "sub.u64 c, c, r;\n\t"       
    "cvt.u32.u64 k0, c;\n\t"
    "and.b64 r, c,  lo;\n\t"     
    "add.cc.u64  %6, %6, r;\n\t"
    "mad.wide.u32 r, k0, p2, c2;\n\t"
    "addc.cc.u64 %7, %7, r;\n\t"
    "mul.wide.u32 r, k0, p4;\n\t"    
    "addc.cc.u64 %0, %0, r;\n\t"
    "mul.wide.u32 r, k0, p6;\n\t"    
    "addc.cc.u64 %1, %1, r;\n\t"
    "addc.u64    %2, 0 , 0;\n\t"
    "mul.wide.u32 r, k0, p1;\n\t"    
    "add.cc.u64  r2, r2, r;\n\t"
    "mul.wide.u32 r, k0, p3;\n\t"    
    "addc.cc.u64 r3, r3, r;\n\t"
    "mul.wide.u32 r, k0, p5;\n\t"    
    "addc.cc.u64 r4, r4, r;\n\t"
    "mul.wide.u32 r, k0, p7;\n\t"    
    "addc.u64    r5, r5, r;\n\t"
    "shr.u64 r, %6, hi;\n\t"     
    "add.cc.u64 r2, r2, r;\n\t"   
    "addc.u64 c2, 0, 0;\n\t"
    "and.b64 r, r2, lo;\n\t"     
    "shl.b64 c, r, hi;\n\t"       
    "sub.u64 c, c, r;\n\t"       
    "cvt.u32.u64 k0, c;\n\t"
    "and.b64 r, c,  lo;\n\t"     
    "add.cc.u64  r2, r2, r;\n\t"
    "mad.wide.u32 r, k0, p2, c2;\n\t"
    "addc.cc.u64 r3, r3, r;\n\t"
    "mul.wide.u32 r, k0, p4;\n\t"    
    "addc.cc.u64 r4, r4, r;\n\t"
    "mul.wide.u32 r, k0, p6;\n\t"    
    "addc.cc.u64 r5, r5, r;\n\t"
    "addc.u64    r6, 0 , 0;\n\t"
    "mul.wide.u32 r, k0, p1;\n\t"    
    "add.cc.u64  %7, %7, r;\n\t"
    "mul.wide.u32 r, k0, p3;\n\t"    
    "addc.cc.u64 %0, %0, r;\n\t"
    "mul.wide.u32 r, k0, p5;\n\t"    
    "addc.cc.u64 %1, %1, r;\n\t"
    "mul.wide.u32 r, k0, p7;\n\t"    
    "addc.u64    %2, %2, r;\n\t"
    "shr.u64 r, r2, hi;\n\t"     
    "add.cc.u64 %7, %7, r;\n\t"   
    "addc.u64 c2, 0, 0;\n\t"
    "and.b64 r, %7, lo;\n\t"     
    "shl.b64 c, r, hi;\n\t"       
    "sub.u64 c, c, r;\n\t"       
    "cvt.u32.u64 k0, c;\n\t"
    "and.b64 r, c,  lo;\n\t"     
    "add.cc.u64  %7, %7, r;\n\t"
    "mad.wide.u32 r, k0, p2, c2;\n\t"
    "addc.cc.u64 %0, %0, r;\n\t"
    "mul.wide.u32 r, k0, p4;\n\t"    
    "addc.cc.u64 %1, %1, r;\n\t"
    "mul.wide.u32 r, k0, p6;\n\t"    
    "addc.cc.u64 %2, %2, r;\n\t"
    "addc.u64    %3, 0 , 0;\n\t"
    "mul.wide.u32 r, k0, p1;\n\t"    
    "add.cc.u64  r3, r3, r;\n\t"
    "mul.wide.u32 r, k0, p3;\n\t"    
    "addc.cc.u64 r4, r4, r;\n\t"
    "mul.wide.u32 r, k0, p5;\n\t"    
    "addc.cc.u64 r5, r5, r;\n\t"
    "mul.wide.u32 r, k0, p7;\n\t"    
    "addc.u64    r6, r6, r;\n\t"
    "shr.u64 r, %7, hi;\n\t"     
    "add.cc.u64 r3, r3, r;\n\t"   
    "addc.u64 c2, 0, 0;\n\t"
    "and.b64 r, r3, lo;\n\t"     
    "shl.b64 c, r, hi;\n\t"       
    "sub.u64 c, c, r;\n\t"       
    "cvt.u32.u64 k0, c;\n\t"
    "and.b64 r, c,  lo;\n\t"     
    "add.cc.u64  r3, r3, r;\n\t"
    "mad.wide.u32 r, k0, p2, c2;\n\t"
    "addc.cc.u64 r4, r4, r;\n\t"
    "mul.wide.u32 r, k0, p4;\n\t"    
    "addc.cc.u64 r5, r5, r;\n\t"
    "mul.wide.u32 r, k0, p6;\n\t"    
    "addc.cc.u64 r6, r6, r;\n\t"
    "addc.u64    r7, 0 , 0;\n\t"
    "mul.wide.u32 r, k0, p1;\n\t"    
    "add.cc.u64  %0, %0, r;\n\t"
    "mul.wide.u32 r, k0, p3;\n\t"    
    "addc.cc.u64 %1, %1, r;\n\t"
    "mul.wide.u32 r, k0, p5;\n\t"    
    "addc.cc.u64 %2, %2, r;\n\t"
    "mul.wide.u32 r, k0, p7;\n\t"    
    "addc.u64    %3, %3, r;\n\t"
    "shr.u64 c, r6 , hi;\n\t"   
    "shl.b64 r7 , r7 , hi;\n\t"  
    "or.b64 r7 , r7 , c;\n\t"
    "shr.u64 c, r5 , hi;\n\t"   
    "shl.b64 r6 , r6 , hi;\n\t"  
    "or.b64 r6 , r6 , c;\n\t"
    "shr.u64 c, r4 , hi;\n\t"   
    "shl.b64 r5 , r5 , hi;\n\t"  
    "or.b64 r5 , r5 , c;\n\t"
    "shr.u64 c, r3 , hi;\n\t"   
    "shl.b64 r4 , r4 , hi;\n\t"  
    "or.b64 r4 , r4 , c;\n\t"
    "add.cc.u64  %0, %0, r4;\n\t"
    "addc.cc.u64 %1, %1, r5;\n\t"
    "addc.cc.u64 %2, %2, r6;\n\t"
    "addc.cc.u64 %3, %3, r7;\n\t"
    "add.cc.u64  %0, %0, %8;\n\t"
    "addc.cc.u64 %1, %1, %9;\n\t"
    "addc.cc.u64 %2, %2, %10;\n\t"
    "addc.cc.u64 %3, %3, %11;\n\t"
    "}"
    : 
    "=l"(ret.v64[0]),
    "=l"(ret.v64[1]),
    "=l"(ret.v64[2]),
    "=l"(ret.v64[3])
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

    if (is_fr_ge(ret, BLS12_377_FR_M)) 
        return fr_sub_unchecked(ret, BLS12_377_FR_M);
    return ret;
}
#endif

static inline blst_fr fr_repr(blst_fr a) 
{
#if defined(OPENCL_NVIDIA)
    ulong r[8] ={a.v64[0], a.v64[1], a.v64[2], a.v64[3], 0, 0, 0, 0};
    return fr_reduce(r);
#else

    __attribute__((opencl_unroll_hint))
    for(uint i = 0; i < FR_LIMBS; i++) {
        limb_t carry;    
        limb_t m = BLS12_377_FR_M0 * LIMB(a, i);
        mac(m, LIMB(BLS12_377_FR_M, 0), LIMB(a, i), &carry);
        __attribute__((opencl_unroll_hint))
        for(uint j = 1; j < FR_LIMBS; j++) {
            uint idx = ((j+i) & (FR_LIMBS-1));
            LIMB(a, idx) = mac_with_carry(m, LIMB(BLS12_377_FR_M, j), LIMB(a, idx), &carry);
        }
        LIMB(a, i) = carry;
    }
    return a;
#endif
}

#define blst_fr_mul(r, a, b)  r = fr_mul_(a, b)
static inline blst_fr fr_mul_(const blst_fr a, const blst_fr b) 
{
#if defined(OPENCL_NVIDIA)
  ulong r[8];
  __asm(
    "{\n\t"
    ".reg .u64 c, t;\n\t"
    "mul.wide.u32 %0, %8, %17;\n\t"
    "mul.wide.u32 %1, %8, %19;\n\t"
    "mul.wide.u32 %2, %8, %21;\n\t"
    "mul.wide.u32 %3, %8, %23;\n\t"
    "mul.wide.u32 c, %9, %16;\n\t"   
    "add.cc.u64 %0, %0, c;\n\t"
    "mul.wide.u32 c, %9, %18;\n\t"   
    "addc.cc.u64 %1, %1, c;\n\t"
    "mul.wide.u32 c, %9, %20;\n\t"   
    "addc.cc.u64 %2, %2, c;\n\t"
    "mul.wide.u32 c, %9, %22;\n\t"   
    "addc.cc.u64 %3, %3, c;\n\t"
    "addc.u64 %4, 0 , 0;\n\t"
    "mul.wide.u32 c, %10, %17;\n\t"   
    "add.cc.u64 %1, %1, c;\n\t"
    "mul.wide.u32 c, %10, %19;\n\t"   
    "addc.cc.u64 %2, %2, c;\n\t"
    "mul.wide.u32 c, %10, %21;\n\t"   
    "addc.cc.u64 %3, %3, c;\n\t"
    "mul.wide.u32 c, %10, %23;\n\t"   
    "addc.cc.u64 %4, %4, c;\n\t"
    "addc.u64 %5, 0 , 0;\n\t"
    "mul.wide.u32 c, %11, %16;\n\t"   
    "add.cc.u64 %1, %1, c;\n\t"
    "mul.wide.u32 c, %11, %18;\n\t"   
    "addc.cc.u64 %2, %2, c;\n\t"
    "mul.wide.u32 c, %11, %20;\n\t"   
    "addc.cc.u64 %3, %3, c;\n\t"
    "mul.wide.u32 c, %11, %22;\n\t"   
    "addc.cc.u64 %4, %4, c;\n\t"
    "addc.u64 %5, %5, 0;\n\t"
    "mul.wide.u32 c, %12, %17;\n\t"   
    "add.cc.u64 %2, %2, c;\n\t"
    "mul.wide.u32 c, %12, %19;\n\t"   
    "addc.cc.u64 %3, %3, c;\n\t"
    "mul.wide.u32 c, %12, %21;\n\t"   
    "addc.cc.u64 %4, %4, c;\n\t"
    "mul.wide.u32 c, %12, %23;\n\t"   
    "addc.cc.u64 %5, %5, c;\n\t"
    "addc.u64 %6, 0 , 0;\n\t"
    "mul.wide.u32 c, %13, %16;\n\t"   
    "add.cc.u64 %2, %2, c;\n\t"
    "mul.wide.u32 c, %13, %18;\n\t"   
    "addc.cc.u64 %3, %3, c;\n\t"
    "mul.wide.u32 c, %13, %20;\n\t"   
    "addc.cc.u64 %4, %4, c;\n\t"
    "mul.wide.u32 c, %13, %22;\n\t"   
    "addc.cc.u64 %5, %5, c;\n\t"
    "addc.u64 %6, %6, 0;\n\t"
    "mul.wide.u32 c, %14, %17;\n\t"   
    "add.cc.u64 %3, %3, c;\n\t"
    "mul.wide.u32 c, %14, %19;\n\t"   
    "addc.cc.u64 %4, %4, c;\n\t"
    "mul.wide.u32 c, %14, %21;\n\t"   
    "addc.cc.u64 %5, %5, c;\n\t"
    "mul.wide.u32 c, %14, %23;\n\t"   
    "addc.cc.u64 %6, %6, c;\n\t"
    "addc.u64 %7, 0 , 0;\n\t"
    "mul.wide.u32 c, %15, %16;\n\t"   
    "add.cc.u64 %3, %3, c;\n\t"
    "mul.wide.u32 c, %15, %18;\n\t"   
    "addc.cc.u64 %4, %4, c;\n\t"
    "mul.wide.u32 c, %15, %20;\n\t"   
    "addc.cc.u64 %5, %5, c;\n\t"
    "mul.wide.u32 c, %15, %22;\n\t"   
    "addc.cc.u64 %6, %6, c;\n\t"
    "addc.u64 %7, %7, 0;\n\t"
    "shr.u64 c, %6 , 32;\n\t"   
    "shl.b64 %7 , %7 , 32;\n\t"  
    "or.b64 %7 , %7 , c;\n\t"
    "shr.u64 c, %5 , 32;\n\t"   
    "shl.b64 %6 , %6 , 32;\n\t"  
    "or.b64 %6 , %6 , c;\n\t"
    "shr.u64 c, %4 , 32;\n\t"   
    "shl.b64 %5 , %5 , 32;\n\t"  
    "or.b64 %5 , %5 , c;\n\t"
    "shr.u64 c, %3 , 32;\n\t"   
    "shl.b64 %4 , %4 , 32;\n\t"  
    "or.b64 %4 , %4 , c;\n\t"
    "shr.u64 c, %2 , 32;\n\t"   
    "shl.b64 %3 , %3 , 32;\n\t"  
    "or.b64 %3 , %3 , c;\n\t"
    "shr.u64 c, %1 , 32;\n\t"   
    "shl.b64 %2 , %2 , 32;\n\t"  
    "or.b64 %2 , %2 , c;\n\t"
    "shr.u64 c, %0 , 32;\n\t"   
    "shl.b64 %1 , %1 , 32;\n\t"  
    "or.b64 %1 , %1 , c;\n\t"
    "shl.b64 %0 , %0 , 32;\n\t"
    "mul.wide.u32 c, %8, %16;\n\t"     
    "add.cc.u64 %0, %0, c;\n\t"
    "mul.wide.u32 c, %8, %18;\n\t"     
    "addc.cc.u64 %1, %1, c;\n\t"
    "mul.wide.u32 c, %8, %20;\n\t"     
    "addc.cc.u64 %2, %2, c;\n\t"
    "mul.wide.u32 c, %8, %22;\n\t"     
    "addc.cc.u64 %3, %3, c;\n\t"
    "addc.u64 t, 0, 0;\n\t"
    "mul.wide.u32 c, %9, %17;\n\t"     
    "add.cc.u64 %1, %1, c;\n\t"
    "mul.wide.u32 c, %9, %19;\n\t"     
    "addc.cc.u64 %2, %2, c;\n\t"
    "mul.wide.u32 c, %9, %21;\n\t"     
    "addc.cc.u64 %3, %3, c;\n\t"
    "mad.wide.u32 c, %9, %23, t;\n\t"  
    "addc.cc.u64 %4, %4, c;\n\t"
    "addc.u64 t, 0, 0;\n\t"
    "mul.wide.u32 c, %10, %16;\n\t"     
    "add.cc.u64 %1, %1, c;\n\t"
    "mul.wide.u32 c, %10, %18;\n\t"     
    "addc.cc.u64 %2, %2, c;\n\t"
    "mul.wide.u32 c, %10, %20;\n\t"     
    "addc.cc.u64 %3, %3, c;\n\t"
    "mul.wide.u32 c, %10, %22;\n\t"     
    "addc.cc.u64 %4, %4, c;\n\t"
    "addc.u64 t, t, 0;\n\t"
    "mul.wide.u32 c, %11, %17;\n\t"     
    "add.cc.u64 %2, %2, c;\n\t"
    "mul.wide.u32 c, %11, %19;\n\t"     
    "addc.cc.u64 %3, %3, c;\n\t"
    "mul.wide.u32 c, %11, %21;\n\t"     
    "addc.cc.u64 %4, %4, c;\n\t"
    "mad.wide.u32 c, %11, %23, t;\n\t"  
    "addc.cc.u64 %5, %5, c;\n\t"
    "addc.u64 t, 0, 0;\n\t"
    "mul.wide.u32 c, %12, %16;\n\t"     
    "add.cc.u64 %2, %2, c;\n\t"
    "mul.wide.u32 c, %12, %18;\n\t"     
    "addc.cc.u64 %3, %3, c;\n\t"
    "mul.wide.u32 c, %12, %20;\n\t"     
    "addc.cc.u64 %4, %4, c;\n\t"
    "mul.wide.u32 c, %12, %22;\n\t"     
    "addc.cc.u64 %5, %5, c;\n\t"
    "addc.u64 t, t, 0;\n\t"
    "mul.wide.u32 c, %13, %17;\n\t"     
    "add.cc.u64 %3, %3, c;\n\t"
    "mul.wide.u32 c, %13, %19;\n\t"     
    "addc.cc.u64 %4, %4, c;\n\t"
    "mul.wide.u32 c, %13, %21;\n\t"     
    "addc.cc.u64 %5, %5, c;\n\t"
    "mad.wide.u32 c, %13, %23, t;\n\t"  
    "addc.cc.u64 %6, %6, c;\n\t"
    "addc.u64 t, 0, 0;\n\t"
    "mul.wide.u32 c, %14, %16;\n\t"     
    "add.cc.u64 %3, %3, c;\n\t"
    "mul.wide.u32 c, %14, %18;\n\t"     
    "addc.cc.u64 %4, %4, c;\n\t"
    "mul.wide.u32 c, %14, %20;\n\t"     
    "addc.cc.u64 %5, %5, c;\n\t"
    "mul.wide.u32 c, %14, %22;\n\t"     
    "addc.cc.u64 %6, %6, c;\n\t"
    "addc.u64 t, t, 0;\n\t"
    "mul.wide.u32 c, %15, %17;\n\t"     
    "add.cc.u64 %4, %4, c;\n\t"
    "mul.wide.u32 c, %15, %19;\n\t"     
    "addc.cc.u64 %5, %5, c;\n\t"
    "mul.wide.u32 c, %15, %21;\n\t"     
    "addc.cc.u64 %6, %6, c;\n\t"
    "mad.wide.u32 c, %15, %23, t;\n\t"  
    "addc.cc.u64 %7, %7, c;\n\t"
   "}"
    : 
    "=l"(r[0]),
    "=l"(r[1]),
    "=l"(r[2]),
    "=l"(r[3]),
    "=l"(r[4]),
    "=l"(r[5]),
    "=l"(r[6]),
    "=l"(r[7])
    : 
    "r"(a.v32[0]),
    "r"(a.v32[1]),
    "r"(a.v32[2]),
    "r"(a.v32[3]),
    "r"(a.v32[4]),
    "r"(a.v32[5]),
    "r"(a.v32[6]),
    "r"(a.v32[7]),
    "r"(b.v32[0]),
    "r"(b.v32[1]),
    "r"(b.v32[2]),
    "r"(b.v32[3]),
    "r"(b.v32[4]),
    "r"(b.v32[5]),
    "r"(b.v32[6]),
    "r"(b.v32[7])
    );
    return fr_reduce(r);
#else
    limb_t carry1, carry2;
    blst_fr fr = {.v64={0}};
    __attribute__((opencl_unroll_hint))
    for(uint i = 0; i < FR_LIMBS; i++) {
        LIMB(fr, 0) = mac(LIMB(a, 0), LIMB(b, i), LIMB(fr, 0), &carry1);
        limb_t m = BLS12_377_FR_M0 * LIMB(fr, 0);
        mac(m, LIMB(BLS12_377_FR_M, 0), LIMB(fr, 0), &carry2);
        __attribute__((opencl_unroll_hint))
        for(uint j = 1; j < FR_LIMBS; j++) {
            LIMB(fr, j) = mac_with_carry(LIMB(a, j), LIMB(b, i), LIMB(fr, j), &carry1);
            LIMB(fr, j - 1) = mac_with_carry(m, LIMB(BLS12_377_FR_M, j), LIMB(fr, j), &carry2);
        }
        LIMB(fr, FR_LIMBS - 1) = carry1 + carry2;
    }

    if (is_fr_ge(fr, BLS12_377_FR_M)) 
        return fr_sub_unchecked(fr, BLS12_377_FR_M);
    return fr;
#endif
}

////////////////////////////////////////////////////////////
void p1_double(blst_p1* out, const blst_p1* in) {
    if (is_p1_zero(in)) {
        blst_p1_copy(out, in);
        return;
    }

    // Z3 = 2*Y1*Z1
    blst_fp_mul(out->Z, in->Y, in->Z);
    blst_fp_add(out->Z, out->Z, out->Z);

    // A = X1^2
    blst_fp a;
    blst_fp_sqr(a, in->X);
    
    // B = Y1^2
    blst_fp b;
    blst_fp_sqr(b, in->Y);

    // C = B^2
    // blst_fp c;
    // blst_fp_sqr(c, b);

    // B2 = B*2
    blst_fp b2;
    blst_fp_add(b2, b, b);

    // D = 2*((X1+B)^2-A-C) 2*(2*X1*B)
    blst_fp d;
    blst_fp_mul(d, in->X, b2);
    blst_fp_add(d, d, d);

    // E = 3*A
    blst_fp e;
    blst_fp_add(e, a, a);
    blst_fp_add(e, e, a);

    // F = E^2
    blst_fp f;
    blst_fp_sqr(f, e);

    // X3 = F-2*D
    blst_fp_add(out->X, d, d);
    blst_fp_sub(out->X, f, out->X);

    // Y3 = E*(D-X3)-8*C
    blst_fp_sub(out->Y, d, out->X);
    blst_fp_mul(out->Y, out->Y, e);

    blst_fp c3;
    blst_fp_sqr(c3, b2); // 4c
    blst_fp_add(c3, c3, c3); // 8c
    blst_fp_sub(out->Y, out->Y, c3);
}

void p1_add_affine(blst_p1 *p1, const blst_p1_affine *p2) {
    if (is_fp_zero(p2->X)) {
        return;
    }

    if (is_p1_zero(p1)) {
        blst_fp_copy(p1->X, p2->X);
        blst_fp_copy(p1->Y, p2->Y);
        blst_fp_copy(p1->Z, BLS12_377_ONE);
        return;
    }
  
    // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
    // Works for all curves.

    // Z1Z1 = Z1^2
    blst_fp z1z1;
    blst_fp_sqr(z1z1, p1->Z);

    // U2 = X2*Z1Z1
    blst_fp u2;
    blst_fp_mul(u2, p2->X, z1z1);

    // S2 = Y2*Z1*Z1Z1
    blst_fp s2;
    blst_fp_mul(s2, p2->Y, p1->Z);
    blst_fp_mul(s2, s2, z1z1);

    if (is_fp_eq(p1->X, u2) && is_fp_eq(p1->Y, s2)) {
        p1_double(p1, p1);
        return;
    }
    // H = U2-X1
    blst_fp h;
    blst_fp_sub(h, u2, p1->X);

    // HH = H^2
    // blst_fp hh;
    // blst_fp_sqr(hh, h);

    // I = 4*HH
    blst_fp i;
    blst_fp_add(i, h, h);
    blst_fp_sqr(i, i);

    // J = H*I
    blst_fp j;
    blst_fp_mul(j, h, i);

    // r = 2*(S2-Y1)
    blst_fp r;
    blst_fp_sub(r, s2, p1->Y);
    blst_fp_add(r, r, r);

    // V = X1*I
    blst_fp v;
    blst_fp_mul(v, p1->X, i);

    // X3 = r^2 - J - 2*V
    blst_fp_sqr(p1->X, r);
    blst_fp_sub(p1->X, p1->X, j);
    blst_fp_sub(p1->X, p1->X, v);
    blst_fp_sub(p1->X, p1->X, v);

    // Y3 = r*(V-X3)-2*Y1*J
    blst_fp_mul(j, p1->Y, j);
    blst_fp_add(j, j, j);
    blst_fp_sub(p1->Y, v, p1->X);
    blst_fp_mul(p1->Y, p1->Y, r);
    blst_fp_sub(p1->Y, p1->Y, j);

    // Z3 = (Z1+H)^2-Z1Z1-HH    2*Z1*H
    blst_fp_mul(p1->Z, p1->Z, h);
    blst_fp_add(p1->Z, p1->Z, p1->Z);

    // blst_fp_add(p1->Z, p1->Z, h);
    // blst_fp_sqr(p1->Z, p1->Z);
    // blst_fp_sub(p1->Z, p1->Z, z1z1);
    // blst_fp_sub(p1->Z, p1->Z, hh);
}

void p1_add_p1(blst_p1 *p1, const blst_p1 *p2) {
    if (is_p1_zero(p2)) {
        return;
    }

    if (is_p1_zero(p1)) {
        blst_p1_copy(p1, p2);
        return;
    }

    if (is_fp_eq(p2->Z, BLS12_377_ONE)) {
        blst_p1_affine p2_affine;
        blst_fp_copy(p2_affine.X, p2->X);
        blst_fp_copy(p2_affine.Y, p2->Y);
        p1_add_affine(p1, &p2_affine);
        return;
    }
  
    // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
    // Works for all curves.

    // Z1Z1 = Z1^2
    blst_fp z1z1;
    blst_fp_sqr(z1z1, p1->Z);

    // Z2Z2 = Z2^2
    blst_fp z2z2;
    blst_fp_sqr(z2z2, p2->Z);

    // U1 = X1*Z2Z2
    blst_fp u1;
    blst_fp_mul(u1, p1->X, z2z2);

    // U2 = X2*Z1Z1
    blst_fp u2;
    blst_fp_mul(u2, p2->X, z1z1);

    // S1 = Y1*Z2*Z2Z2
    blst_fp s1;
    blst_fp_mul(s1, p1->Y, p2->Z);
    blst_fp_mul(s1, s1, z2z2);

    // S2 = Y2*Z1*Z1Z1
    blst_fp s2;
    blst_fp_mul(s2, p2->Y, p1->Z);
    blst_fp_mul(s2, s2, z1z1);

    if (is_fp_eq(u1, u2) && is_fp_eq(s1, s2)) {
        p1_double(p1, p1);
        return;
    }

    // H = U2-U1
    blst_fp h;
    blst_fp_sub(h, u2, u1);

    // HH = H^2
    // blst_fp hh;
    // blst_fp_sqr(hh, h);

    // I = 4*HH
    blst_fp i;
    blst_fp_add(i, h, h);
    blst_fp_sqr(i, i);

    // J = H*I
    blst_fp j;
    blst_fp_mul(j, h, i);

    // r = 2*(S2-S1)
    blst_fp r;
    blst_fp_sub(r, s2, s1);
    blst_fp_add(r, r, r);

    // V = U1*I
    blst_fp v;
    blst_fp_mul(v, u1, i);

    // X3 = r^2 - J - 2*V
    blst_fp_sqr(p1->X, r);
    blst_fp_sub(p1->X, p1->X, j);
    blst_fp_sub(p1->X, p1->X, v);
    blst_fp_sub(p1->X, p1->X, v);

    // Y3 = r*(V-X3)-2*S1*J
    blst_fp_mul(j, s1, j);
    blst_fp_add(j, j, j);
    blst_fp_sub(p1->Y, v, p1->X);
    blst_fp_mul(p1->Y, p1->Y, r);
    blst_fp_sub(p1->Y, p1->Y, j);

    // Z3 = ((Z1+Z2)^2-Z1Z1-Z2Z2)*H = 2*Z1*Z2*H
    blst_fp_mul(p1->Z, p1->Z, p2->Z);
    blst_fp_mul(p1->Z, p1->Z, h);    
    blst_fp_add(p1->Z, p1->Z, p1->Z);
    
    // blst_fp_add(p1->Z, p1->Z, p2->Z);
    // blst_fp_sqr(p1->Z, p1->Z);
    // blst_fp_sub(p1->Z, p1->Z, z1z1);
    // blst_fp_sub(p1->Z, p1->Z, z2z2);
    // blst_fp_mul(p1->Z, p1->Z, h);
}

///////////////////////////////////////////////////////////////////////////////////////
__kernel void msm4_evaluate(
    __global blst_fr* result,
    __global blst_fr* bases,
    __global blst_fr* bate,
    __global blst_fr* gamma
)
{
    size_t gid = get_group_id(0);
    __global blst_fr* gammah = gamma + 32768;
    __global blst_fr* group_powers[]={ bate, bate, gamma, gammah, gamma, gammah, gamma, gammah};    
    const uint group_size[] =  { 32768, 32767, 32768, 32767, 32768,  32767, 32768,  32767 };
    const uint group_start[] = { 0, 32768, 65535, 98303, 131070, 163838, 196605, 229373, 262140};

    __global blst_fr* _bases = bases + group_start[gid];
    __global blst_fr* _powers = group_powers[gid];

    uint size = group_size[gid];
    size_t start = get_local_id(0) << 7;
    size_t end = start + 128;
    if (start >= size)
        return;
    if (end > size)
        end = size;

    blst_fr ret;
    blst_fr sum_ret = FR_ZERO;
    do {
        blst_fr_mul(ret, _bases[start], _powers[start]);
        blst_fr_add(sum_ret, sum_ret, ret);
    } while (++start < end);

    blst_fr_copy(result[get_global_id(0)], sum_ret);
}


__kernel void msm4_sum(
    __global blst_fr* result,
    __global blst_fr* bases,
    const uint chunk_size,
    const uint data_size
)
{
    uint start = get_global_id(0) * chunk_size;
    uint end = start + chunk_size; 

    blst_fr ret = FR_ZERO;
    if (start >= data_size)
    {
        blst_fr_copy(result[get_global_id(0)], ret);
        return;
    }

    if (end > data_size)
        end = data_size;
    
    while(start < end)
    {
        blst_fr_add(ret, ret, bases[start]);
        start++;
    }
    
    blst_fr_copy(result[get_global_id(0)], ret);
}


__kernel void msm4_powers_serial(
    __global blst_fr* bases, 
    __global blst_fr* roots,
    const uint root_idx,
    const uint size
)
{
    size_t index = get_global_id(0) << 3;
    if ( index >= size)
        return; 

    uint loop = index;
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

__kernel void msm4_io_helper(
    __global blst_fr* bases,
    __global blst_fr* roots,
    const uint chunk_size,
    const uint chunk_num,
    const uint gap,
    const uint size
)
{
    size_t index = get_global_id(0) << 3;
    if ( index >= size)
        return;
    size_t end = index + 8;
    if (end > size)
        end = size;

    do {
    uint cid = index & (chunk_num - 1);
    uint rid = index & (~(chunk_num - 1));
    uint lid = index / chunk_num + cid * chunk_size;

    blst_fr neg, lo, hi;
    blst_fr_copy(lo, bases[lid]);
    blst_fr_copy(hi, bases[lid + gap]);

    blst_fr_sub(neg, lo, hi);
    blst_fr_add(bases[lid], lo, hi);
    blst_fr_mul(bases[lid + gap], neg, roots[rid]);
    } while(++index < end);
}

__kernel void msm4_oi_helper(
    __global blst_fr* bases,
    __global blst_fr* roots,
    const uint chunk_size,
    const uint chunk_num,
    const uint gap,
    const uint size
)
{
    size_t index = get_global_id(0) << 3;
    if ( index >= size)
        return;
    size_t end = index + 8;
    if (end > size)
        end = size;

    do {    
    uint cid = index & (chunk_num - 1);
    uint rid = index & (~(chunk_num - 1));
    uint lid = index / chunk_num + cid * chunk_size;

    blst_fr lo, hi;
    blst_fr_copy(lo, bases[lid]);
    blst_fr_copy(hi, bases[lid + gap]);

    blst_fr_mul(hi, hi, roots[rid]);
    blst_fr_add(bases[lid], lo, hi);
    blst_fr_sub(bases[lid + gap], lo, hi);
    } while(++index < end);
}

__kernel void msm4_mul_assign(
    __global blst_fr* base, 
    __global blst_fr* other,
    const uint size)
{
    size_t start = get_global_id(0) << 3;
    size_t end = start + 8;
    if (start >= size)
        return;
    if (end > size)
        end = size;

    do {
        blst_fr_mul(base[start], base[start], other[0]);  
    } while (++start < end);
}

__kernel void msm4_mul2_assign(
    __global blst_fr* base, 
    __global blst_fr* other,
    const uint size)
{
    size_t start = get_global_id(0) << 3;
    size_t end = start + 8;
    if (start >= size)
        return;
    if (end > size)
        end = size;

    do {
        blst_fr_mul(base[start], base[start], other[start]);  
    } while (++start < end);
}

#define NBITS 253
#define NTHRBITS  8
#define NTHREADS  256

static inline int get_wval(const blst_fr d, uint off, uint bits)
{
    uint top = off + bits - 1;
    ulong ret = (convert_ulong(d.v32[top/32]) << 32) | convert_ulong(d.v32[off/32]);
    return convert_int(ret >> (off%32)) & ((1<<bits) - 1);
}

static inline int is_unique(__local int* wvals, int wval, int dir)
{
    // __local int* wvals = (__local int*)scratch;
    const uint tid = get_local_id(0);
    dir &= 1;   // force logical operations on predicates

    barrier(CLK_LOCAL_MEM_FENCE);
    wvals[tid] = wval;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Straightforward scan appears to be the fastest option for NTHREADS.
    // Bitonic sort complexity, a.k.a. amount of iterations, is [~3x] lower,
    // but each step is [~5x] slower...
    int negatives = 0;
    int uniq = 1;
    #pragma unroll 16
    for (uint i=0; i<NTHREADS; i++) {
        int b = wvals[i];   // compiled as 128-bit [broadcast] loads:-)
        if (((i<tid)^dir) && i!=tid && wval==b)
            uniq = 0;
        negatives += (b < 0);
    }

    return uniq | (int)(NTHREADS-1-negatives)>>31;
    // return value is 1, 0 or -1.
}


__kernel void per_scalars(
    __global blst_fr* scalars, 
    const uint block_size, 
    const uint npoints)
{
    uint idx = get_group_id(0) * block_size;
    uint end = min(idx + block_size, npoints);
    idx += get_local_id(0);

    #pragma unroll 1
    while (idx < end) {
        scalars[idx] = fr_repr(scalars[idx]);
        idx += NTHREADS;
    }    
}

__kernel void per_scalars_one(
    __global blst_p1 *ones, 
    __global blst_g1_affine* points, 
    __global blst_fr* scalars, 
    const uint block_size, 
    const uint npoints)
{
    uint idx = get_group_id(0) * block_size;
    uint end = min(idx + block_size, npoints);
    idx += get_local_id(0);

    blst_p1 res;
    res.Z = FP_ZERO;
    blst_p1_affine p;
    while (idx < end) {
        blst_fr s = scalars[idx];
        if (is_fr_one(s)) {
            vstore2(vload2(0, points[idx].X.v64), 0, p.X.v64);
            vstore2(vload2(1, points[idx].X.v64), 1, p.X.v64);
            vstore2(vload2(2, points[idx].X.v64), 2, p.X.v64);
            vstore2(vload2(0, points[idx].Y.v64), 0, p.Y.v64);
            vstore2(vload2(1, points[idx].Y.v64), 1, p.Y.v64);
            vstore2(vload2(2, points[idx].Y.v64), 2, p.Y.v64);
            p1_add_affine(&res, &p);
            scalars[idx] = FR_ZERO;
        } else {
            scalars[idx] = fr_repr(s);
        }
        idx += NTHREADS;
    }

    __local blst_p1 scratch[128];
    uint tid = get_local_id(0);
    for (size_t i=128; i>0; i>>=1) {
        if (tid >= i) scratch[tid - i] = res;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (tid >= i) return;

        blst_p1 s = scratch[tid];
        p1_add_p1(&res, &s);
    }
    ones[get_group_id(0)] = res;
}

__kernel void msm_cl(
    __global blst_p1* buckets,
    __global blst_p1* rets,
    __global blst_g1_affine* points, 
    __global blst_fr* scalars, 
    uint npoints,
    const uint WBITS,
    const uint group_num
) {
    const uint bucket_size = 1 <<WBITS;
    const uint window_num = (NBITS + WBITS -1)/WBITS;
    uint tid = get_local_id(0);
    uint bid = get_group_id(0);
    const uint bit0 = (bid % window_num) * WBITS;

    if (group_num > 1) {
        uint ngroup = bid / window_num;
        uint delta = (npoints + group_num - 1) / group_num;
        uint off = delta * ngroup;

        points  += off;
        scalars += off;
        if (ngroup == group_num-1)
            npoints -= off;
        else
            npoints = delta;
    }

    __global blst_p1* row = buckets + (bid * bucket_size);

    #pragma unroll 16
    for (uint i = tid; i < bucket_size; i += NTHREADS)
        row[i].Z =FP_ZERO;
    
    int wbits = (bit0 > NBITS-WBITS) ? NBITS-bit0 : WBITS;
    int bias  = (tid >> max((int)(wbits+NTHRBITS-WBITS), 0)) << max(wbits, (int)(WBITS-NTHRBITS));
    __local blst_p1 scratch[NTHREADS];

    int dir = 1;
    for (uint i = tid; true; ) {
        int wval = -1;
        blst_p1_affine point;
        if (i < npoints) {
            wval = get_wval(scalars[i], bit0, wbits);
            wval += wval ? bias : 0;
            vstore2(vload2(0, points[i].X.v64), 0, point.X.v64);
            vstore2(vload2(1, points[i].X.v64), 1, point.X.v64);
            vstore2(vload2(2, points[i].X.v64), 2, point.X.v64);
            vstore2(vload2(0, points[i].Y.v64), 0, point.Y.v64);
            vstore2(vload2(1, points[i].Y.v64), 1, point.Y.v64);
            vstore2(vload2(2, points[i].Y.v64), 2, point.Y.v64);
        }

        int uniq = is_unique((__local int*)scratch, wval, dir^=1) | (wval==0);
        if (uniq < 0)   // all |wval|-s are negative, all done
            break;

        if (i < npoints && uniq) {
            if (wval) {
                blst_p1 p1 = row[wval-1];
                p1_add_affine(&p1, &point);
                row[wval-1]= p1;
            }
            i += NTHREADS;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint i = 1<<(WBITS-NTHRBITS);
    row += tid * i;
    blst_p1 acc = row[--i];
    blst_p1 res = acc;
    while (i--) {
        blst_p1 p = row[i];
        p1_add_p1(&acc, &p);
        p1_add_p1(&res, &acc);
    }
    bias = wbits+NTHRBITS-WBITS;
    #pragma unroll
    for (int ic=0; ic<8; ++ic)
    {
        uint sid = tid & 0xfe;
        if ((tid & 1) == 0) {
            scratch[sid] = res;
            scratch[sid+1] = acc;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((tid & 1) == 0) return;

        tid >>= 1;
        blst_p1 p = scratch[sid];
        p1_add_p1(&res, &p);
        if (ic < bias) {
            blst_p1 raise = acc;
            for (size_t j = 0; j < WBITS-NTHRBITS+ic; j++)
                p1_double(&raise, &raise);
            p1_add_p1(&res, &raise);
            p = scratch[sid+1];
            p1_add_p1(&acc, &p);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    rets[bid] = res;
}
