#include <string.h>
#include <stdio.h>
#include "asm_fft_cuda.h"

__device__ static inline int is_ge_256( blst_fr left,  blst_fr right) 
{
    for (int i = 3; i >= 0; --i) {
        if (left[i] < right[i]) {
            return 0;
        } else if (left[i] > right[i]) {
            return 1;
        }
    }
    return 1;
}

__device__ static inline void sub_mod_256_unchecked(blst_fr ret,  blst_fr a,  blst_fr b) 
{
    asm(
        "sub.cc.u64 %0, %4, %8;\n\t"
        "subc.cc.u64 %1, %5, %9;\n\t"
        "subc.cc.u64 %2, %6, %10;\n\t"
        "subc.u64 %3, %7, %11;"
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

__device__ static inline void reduce(blst_fr x,  blst_fr p) 
{
    if (is_ge_256(x, p)) {
        blst_fr x_sub;
        sub_mod_256_unchecked(x_sub, x, p);
        memcpy(x, x_sub, sizeof(blst_fr));
    }
}

__device__ static inline void mont_256(blst_fr ret, limb_t r[8],  blst_fr p,  limb_t p_inv) 
{    
    limb_t k = r[0] * p_inv;
    
    limb_t cross_carry = 0;
    
    asm(
        "{\n\t"
        ".reg .u64 c;\n\t"
        ".reg .u64 t;\n\t"
        ".reg .u64 nc;\n\t"
        
        "mad.lo.cc.u64 c, %10, %6, %0;\n\t"
        "madc.hi.cc.u64 c, %10, %6, 0;\n\t"
        
        "addc.cc.u64 t, %1, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %1, %10, %7, t;\n\t"
        "madc.hi.cc.u64 c, %10, %7, nc;\n\t"
        
        "addc.cc.u64 t, %2, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %2, %10, %8, t;\n\t"
        "madc.hi.cc.u64 c, %10, %8, nc;\n\t"
        
        "addc.cc.u64 t, %3, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %3, %10, %9, t;\n\t"
        "madc.hi.cc.u64 c, %10, %9, nc;\n\t"
        
        "addc.cc.u64 %4, %4, c;\n\t"
        "addc.u64 %5, 0, 0;\n\t"
        "}"
        : "+l"(r[0]),
        "+l"(r[1]),
        "+l"(r[2]),
        "+l"(r[3]),
        "+l"(r[4]),
        "=l"(cross_carry)
        : "l"(p[0]),
        "l"(p[1]),
        "l"(p[2]),
        "l"(p[3]),
        "l"(k)
    );
    
    k = r[1] * p_inv;
    
    asm(
        "{\n\t"
        ".reg .u64 c;\n\t"
        ".reg .u64 t;\n\t"
        ".reg .u64 nc;\n\t"
        
        "mad.lo.cc.u64 c, %10, %6, %0;\n\t"
        "madc.hi.cc.u64 c, %10, %6, 0;\n\t"
        
        "addc.cc.u64 t, %1, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %1, %10, %7, t;\n\t"
        "madc.hi.cc.u64 c, %10, %7, nc;\n\t"
        
        "addc.cc.u64 t, %2, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %2, %10, %8, t;\n\t"
        "madc.hi.cc.u64 c, %10, %8, nc;\n\t"
        
        "addc.cc.u64 t, %3, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %3, %10, %9, t;\n\t"
        "madc.hi.cc.u64 c, %10, %9, nc;\n\t"
        
        "addc.cc.u64 c, c, %5;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "addc.cc.u64 %4, %4, c;\n\t"
        "addc.u64 %5, nc, 0;\n\t"
        "}"
        
        : "+l"(r[1]),
        "+l"(r[2]),
        "+l"(r[3]),
        "+l"(r[4]),
        "+l"(r[5]),
        "+l"(cross_carry)
        : "l"(p[0]),
        "l"(p[1]),
        "l"(p[2]),
        "l"(p[3]),
        "l"(k)
    );
    
    k = r[2] * p_inv;
    
    asm(
        "{\n\t"
        ".reg .u64 c;\n\t"
        ".reg .u64 t;\n\t"
        ".reg .u64 nc;\n\t"
        
        "mad.lo.cc.u64 c, %10, %6, %0;\n\t"
        "madc.hi.cc.u64 c, %10, %6, 0;\n\t"
        
        "addc.cc.u64 t, %1, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %1, %10, %7, t;\n\t"
        "madc.hi.cc.u64 c, %10, %7, nc;\n\t"
        
        "addc.cc.u64 t, %2, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %2, %10, %8, t;\n\t"
        "madc.hi.cc.u64 c, %10, %8, nc;\n\t"
        
        "addc.cc.u64 t, %3, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %3, %10, %9, t;\n\t"
        "madc.hi.cc.u64 c, %10, %9, nc;\n\t"
        
        
        "addc.cc.u64 c, c, %5;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "addc.cc.u64 %4, %4, c;\n\t"
        "addc.u64 %5, nc, 0;\n\t"
        "}"
        : "+l"(r[2]),
        "+l"(r[3]),
        "+l"(r[4]),
        "+l"(r[5]),
        "+l"(r[6]),
        "+l"(cross_carry)
        : "l"(p[0]),
        "l"(p[1]),
        "l"(p[2]),
        "l"(p[3]),
        "l"(k)
    );
    
    k = r[3] * p_inv;
    
    asm(
        "{\n\t"
        ".reg .u64 c;\n\t"
        ".reg .u64 t;\n\t"
        ".reg .u64 nc;\n\t"
        
        "mad.lo.cc.u64 c, %10, %6, %0;\n\t"
        "madc.hi.cc.u64 c, %10, %6, 0;\n\t"
        
        "addc.cc.u64 t, %1, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %1, %10, %7, t;\n\t"
        "madc.hi.cc.u64 c, %10, %7, nc;\n\t"
        
        "addc.cc.u64 t, %2, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %2, %10, %8, t;\n\t"
        "madc.hi.cc.u64 c, %10, %8, nc;\n\t"
        
        "addc.cc.u64 t, %3, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %3, %10, %9, t;\n\t"
        "madc.hi.cc.u64 c, %10, %9, nc;\n\t"
        
        
        
        "addc.cc.u64 c, c, %5;\n\t"
        "add.u64 %4, %4, c;\n\t" // and this to be add.cc
        "}"
        : "+l"(r[3]),
        "+l"(r[4]),
        "+l"(r[5]),
        "+l"(r[6]),
        "+l"(r[7])
        : "l"(cross_carry),
        "l"(p[0]),
        "l"(p[1]),
        "l"(p[2]),
        "l"(p[3]),
        "l"(k)
    );    

    memcpy(ret, r + 4, sizeof(limb_t) * 4);
    reduce(ret, p);
}

__device__ void mul_mont_256(blst_fr ret, blst_fr a, blst_fr b,  blst_fr p, limb_t p_inv) 
{
    limb_t r[8] = {0};
    
    asm(
        "{\n\t"
        ".reg .u64 c;\n\t"
        ".reg .u64 nc;\n\t"
        ".reg .u64 t;\n\t"
        
        "mad.lo.cc.u64 %0, %8, %12, 0;\n\t"
        "madc.hi.cc.u64 c, %8, %12, 0;\n\t"
        
        "madc.lo.cc.u64 %1, %8, %13, c;\n\t"
        "madc.hi.cc.u64 c, %8, %13, 0;\n\t"
        
        "madc.lo.cc.u64 %2, %8, %14, c;\n\t"
        "madc.hi.cc.u64 c, %8, %14, 0;\n\t"
        
        "madc.lo.cc.u64 %3, %8, %15, c;\n\t"
        "madc.hi.cc.u64 %4, %8, %15, 0;\n\t"
        
        
        
        "mad.lo.cc.u64 %1, %9, %12, %1;\n\t"
        "madc.hi.cc.u64 c, %9, %12, 0;\n\t"
        
        "addc.cc.u64 t, %2, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %2, %9, %13, t;\n\t"
        "madc.hi.cc.u64 c, %9, %13, nc;\n\t"
        
        "addc.cc.u64 t, %3, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %3, %9, %14, t;\n\t"
        "madc.hi.cc.u64 c, %9, %14, nc;\n\t"
        
        "addc.cc.u64 t, %4, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %4, %9, %15, t;\n\t"
        "madc.hi.cc.u64 %5, %9, %15, nc;\n\t"
        
        
        
        
        "mad.lo.cc.u64 %2, %10, %12, %2;\n\t"
        "madc.hi.cc.u64 c, %10, %12, 0;\n\t"
        
        "addc.cc.u64 t, %3, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %3, %10, %13, t;\n\t"
        "madc.hi.cc.u64 c, %10, %13, nc;\n\t"
        
        "addc.cc.u64 t, %4, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %4, %10, %14, t;\n\t"
        "madc.hi.cc.u64 c, %10, %14, nc;\n\t"
        
        "addc.cc.u64 t, %5, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %5, %10, %15, t;\n\t"
        "madc.hi.cc.u64 %6, %10, %15, nc;\n\t"
        
        
        
        "mad.lo.cc.u64 %3, %11, %12, %3;\n\t"
        "madc.hi.cc.u64 c, %11, %12, 0;\n\t"
        
        "addc.cc.u64 t, %4, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %4, %11, %13, t;\n\t"
        "madc.hi.cc.u64 c, %11, %13, nc;\n\t"
        
        "addc.cc.u64 t, %5, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %5, %11, %14, t;\n\t"
        "madc.hi.cc.u64 c, %11, %14, nc;\n\t"
        
        "addc.cc.u64 t, %6, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %6, %11, %15, t;\n\t"
        "madc.hi.cc.u64 %7, %11, %15, nc;\n\t"
        
        "}"
        : "+l"(r[0]),
        "+l"(r[1]),
        "+l"(r[2]),
        "+l"(r[3]),
        "+l"(r[4]),
        "+l"(r[5]),
        "+l"(r[6]),
        "+l"(r[7])
        : "l"(a[0]),
        "l"(a[1]),
        "l"(a[2]),
        "l"(a[3]),
        "l"(b[0]),
        "l"(b[1]),
        "l"(b[2]),
        "l"(b[3])
    );
    
    mont_256(ret, r, p, p_inv);
}

__device__ void sqr_mont_256(blst_fr ret,  blst_fr a,  blst_fr p, limb_t p_inv) 
{
    limb_t r[8] = {0};
    
    asm(
        "{\n\t"
        ".reg .u64 c;\n\t"
        ".reg .u64 nc;\n\t"
        ".reg .u64 t;\n\t"
        
        "mad.lo.cc.u64 %1, %8, %9, 0;\n\t"
        "madc.hi.cc.u64 c, %8, %9, 0;\n\t"
        
        "madc.lo.cc.u64 %2, %8, %10, c;\n\t"
        "madc.hi.cc.u64 c, %8, %10, 0;\n\t"
        
        "madc.lo.cc.u64 %3, %8, %11, c;\n\t"
        "madc.hi.cc.u64 %4, %8, %11, 0;\n\t"
        
        
        "mad.lo.cc.u64 %3, %9, %10, %3;\n\t"
        "madc.hi.cc.u64 c, %9, %10, 0;\n\t"
        
        "addc.cc.u64 t, %4, c;\n\t"
        "addc.u64 nc, 0, 0;\n\t"
        "mad.lo.cc.u64 %4, %9, %11, t;\n\t"
        "madc.hi.cc.u64 %5, %9, %11, nc;\n\t"
        
        
        "mad.lo.cc.u64 %5, %10, %11, %5;\n\t"
        "madc.hi.cc.u64 %6, %10, %11, 0;\n\t"
        
        "}"
        : "+l"(r[0]),
        "+l"(r[1]),
        "+l"(r[2]),
        "+l"(r[3]),
        "+l"(r[4]),
        "+l"(r[5]),
        "+l"(r[6]),
        "+l"(r[7])
        : "l"(a[0]),
        "l"(a[1]),
        "l"(a[2]),
        "l"(a[3])
    );
    
    
    r[7] = r[6] >> 63;
    r[6] = (r[6] << 1) | (r[5] >> 63);
    r[5] = (r[5] << 1) | (r[4] >> 63);
    r[4] = (r[4] << 1) | (r[3] >> 63);
    r[3] = (r[3] << 1) | (r[2] >> 63);
    r[2] = (r[2] << 1) | (r[1] >> 63);
    r[1] = r[1] << 1;
    
    
    asm(
        "{\n\t"
        
        "mad.lo.cc.u64 %0, %8, %8, 0;\n\t"
        "madc.hi.cc.u64 %1, %8, %8, %1;\n\t"
        
        "madc.lo.cc.u64 %2, %9, %9, %2;\n\t"
        "madc.hi.cc.u64 %3, %9, %9, %3;\n\t"
        
        "madc.lo.cc.u64 %4, %10, %10, %4;\n\t"
        "madc.hi.cc.u64 %5, %10, %10, %5;\n\t"
        
        "madc.lo.cc.u64 %6, %11, %11, %6;\n\t"
        "madc.hi.cc.u64 %7, %11, %11, %7;\n\t"
        
        "}"
        : "+l"(r[0]),
        "+l"(r[1]),
        "+l"(r[2]),
        "+l"(r[3]),
        "+l"(r[4]),
        "+l"(r[5]),
        "+l"(r[6]),
        "+l"(r[7])
        : "l"(a[0]),
        "l"(a[1]),
        "l"(a[2]),
        "l"(a[3])
    );
    
    mont_256(ret, r, p, p_inv);
}


__device__ static inline void add_mod_256_unchecked(blst_fr ret,  blst_fr a,  blst_fr b) 
{
    asm(
        "add.cc.u64 %0, %4, %8;\n\t"
        "addc.cc.u64 %1, %5, %9;\n\t"
        "addc.cc.u64 %2, %6, %10;\n\t"
        "addc.u64 %3, %7, %11;"
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

__device__ void add_mod_256(blst_fr ret,  blst_fr a,  blst_fr b,  blst_fr p) 
{
    add_mod_256_unchecked(ret, a, b);
    
    reduce(ret, p);
}

__device__ void sub_mod_256(blst_fr ret,  blst_fr a,  blst_fr b,  blst_fr p) 
{
    blst_fr added;
    
    memcpy(added, a, sizeof(blst_fr));
    if (is_gt_256(b, a)) {
        add_mod_256_unchecked(added, added, p);
    }
    
    sub_mod_256_unchecked(ret, added, b);
}

__device__ void sub_mod_256_unsafe(blst_fr ret,  blst_fr a,  blst_fr b) 
{
    sub_mod_256_unchecked(ret, a, b);
}

__device__ void add_mod_256_unsafe(blst_fr ret,  blst_fr a,  blst_fr b) 
{
    add_mod_256_unchecked(ret, a, b);
}

__device__ static inline void _rshift_256(blst_fr ret,  blst_fr value) 
{
    ret[0] = (value[1] << 63) | (value[0] >> 1);
    ret[1] = (value[2] << 63) | (value[1] >> 1);
    ret[2] = (value[3] << 63) | (value[2] >> 1);
    ret[3] = value[3] >> 1;
}

__device__ void div_by_2_mod_256(blst_fr ret,  blst_fr a) 
{
    _rshift_256(ret, a);
}

__device__ void cneg_mod_256(blst_fr ret,  blst_fr a, bool flag,  blst_fr p) 
{
    if (flag) {
        sub_mod_256(ret, p, a, p);
    } else {
        memcpy(ret, a, 4 * sizeof(limb_t));
    }
}

__device__ void pow_256(blst_fr* ret, blst_fr base, uint32_t exponent,  blst_fr p,  limb_t p_inv,  blst_fr p_one) 
{
    blst_fr tmp_base;

    memcpy(tmp_base, base, sizeof(blst_fr));
    memcpy(*ret, p_one, sizeof(blst_fr));

    while(exponent > 0) {
        if (exponent & 1) {
            mul_mont_256(*ret, *ret, tmp_base, p, p_inv);
        }
        exponent = exponent >> 1;
        sqr_mont_256(tmp_base, tmp_base, p, p_inv);
    }
    
}

__device__ void pow_256_lookup(blst_fr* ret, blst_fr*  bases, uint32_t exponent,  blst_fr p,  limb_t p_inv,  blst_fr p_one) 
{
    uint32_t i = 0;
    
    memcpy(*ret, p_one, sizeof(blst_fr));
    while(exponent > 0) {
        if (exponent & 1) {
            mul_mont_256(*ret, *ret, bases[i], p, p_inv);
        }
        exponent = exponent >> 1;
        i++;
    }

}

__device__ void inverse_256(blst_fr out, blst_fr in) {
    if (is_blst_fr_zero(in)) {
        *((int*)NULL);
    }

    blst_fr u;
    memcpy(u, in, sizeof(blst_fr));
    blst_fr v;
    memcpy(v, BLS12_377_P, sizeof(blst_fr));
    blst_fr b;
    memcpy(b, BLS12_377_R2, sizeof(blst_fr));
    blst_fr c;
    memset(c, 0, sizeof(blst_fr));

    while (!is_blst_fr_eq(u, BIGINT_ONE) && !is_blst_fr_eq(v, BIGINT_ONE)) {
        while ((u[0] & 1) == 0) {
            div_by_2_mod_256(u, u);

            if ((b[0] & 1) != 0) {
               add_mod_256_unsafe(b, b, BLS12_377_P);
            }
            div_by_2_mod_256(b, b);
        }

        while ((v[0] & 1) == 0) {
            div_by_2_mod_256(v, v);

            if ((c[0] & 1) != 0) {
                add_mod_256_unsafe(c, c, BLS12_377_P);
            }
            div_by_2_mod_256(c, c);
        }

        if (is_gt_256(u, v)) {
            sub_mod_256_unsafe(u, u, v);
            
            sub_mod_256(b, b, c, BLS12_377_P);
        } else {
            sub_mod_256_unsafe(v, v, u);

            sub_mod_256(c, c, b,BLS12_377_P);
        }
    }
    if (is_blst_fr_eq(u, BIGINT_ONE)) {
        memcpy(out, b, sizeof(blst_fr));
    } else {
        memcpy(out, c, sizeof(blst_fr));
    }
}