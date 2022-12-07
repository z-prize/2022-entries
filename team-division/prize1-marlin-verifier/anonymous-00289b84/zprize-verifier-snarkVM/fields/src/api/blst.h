#ifndef __BLST_H__
#define __BLST_H__

#include "structs.h"

void blst_fp_mul(vec384 ret, const limb_t* a, const limb_t* b, const limb_t* p, const limb_t p0);
void blst_fp_sqr(vec384 ret, const limb_t* a, const limb_t* p, const limb_t p0);
void blst_fp_eucl_inverse(vec384 ret, const limb_t* a, const limb_t* p, const limb_t p0);

#endif