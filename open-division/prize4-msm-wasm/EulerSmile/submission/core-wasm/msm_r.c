/*
 * Copyright (c) 2022 EulerSmile ( see https://github.com/EulerSmile).
 *
 * Dual-licensed under both the MIT and Apache-2.0 licenses;
 * you may not use this file except in compliance with the License.
 */

#include <string.h>
#include <stdio.h>
#include "ecp_BLS12381.h"
#include "fp_BLS12381.h"
#define BIG_I32_LEN 14
#define ECP_I32_LEN 45

const int POINT_BYTES = 2 * MODBYTES_384_29 + 1;

// Get the best window bit
// Since the range is less than 2^18, so the window is less than 14
int best_w(int n){
    int k = 4;
    int min = 1 << 30;
    int add ;
    for(int i=4;i<20;i++){
        add = 256*(n+2*((1<<i)-1)+1)/i;
        if (add < min){
            min = add;
            k = i;
        }
    }
    if (k > 14) k = 14;
    return k;
}
/* Kill an octet string - Zeroise it for security */
void OCT_clear(octet *w)
{
    int i;
    for (i = 0; i < w->max; i++) w->val[i] = 0;
    w->len = 0;
}

void OCT_fromStr(octet *dst, char *src, int n)
{
    int i;
    unsigned char ch;
    OCT_clear(dst);
    dst->len = n;
    for (i = 0; i < n; i++) {
        ch = src[i];
        dst->val[i] = ch;
    }
}

// Set a big number from bytes
void BIG_384_29_fromArkBytes(BIG_384_29 *t, char *c, unsigned char **s)
{
    memcpy(&c[16], *s, 32); // first 16 byte should be zero.
    BIG_384_29_fromBytes(*t, c);
    *s += 32;
}

// Set a point from bytes
int ECP_BLS12381_fromBytes(ECP_BLS12381 *t, octet *U, unsigned char **s)
{
    ECP_BLS12381_inf(t);
    OCT_fromStr(U, *s, POINT_BYTES);
    *s += POINT_BYTES;

    return ECP_BLS12381_fromOctet(t, U);
}

// The main msm function
void ECP_BLS12381_msm_i(ECP_BLS12381 *P, int n, int32_t *i32X, int32_t *i32e, int32_t *i32B){
    int i,j,k,nb,ret;
    BIG_384_29 t,mt;
    ECP_BLS12381 S,R;
    int window = best_w(n);
    
    ECP_BLS12381_inf(P);

    ECP_BLS12381* T;
    ECP_BLS12381* B;

    BIG_384_29* te;
    te = (BIG_384_29*)i32e;
    BIG_384_29_copy(mt,te);  BIG_384_29_norm(mt);
    for (i=1;i<n;i++)
    { // find biggest
        te = (BIG_384_29*)(i32e+i*BIG_I32_LEN);
        BIG_384_29_copy(t,te[0]); BIG_384_29_norm(t);
        k=BIG_384_29_comp(t,mt);
        BIG_384_29_cmove(mt,t,(k+1)/2);
    }
    nb=(BIG_384_29_nbits(mt)+window-1)/window;
    for (int i=nb-1;i>=0;i--)
    { // Pippenger's algorithm
        for (j=0;j<(1<<window);j++){
            B = (ECP_BLS12381*)(i32B+j*ECP_I32_LEN);
            ECP_BLS12381_inf(B);
        }
            
        for (j=0;j<n;j++)
        {
            te = (BIG_384_29*)(i32e+j*BIG_I32_LEN);
            BIG_384_29_copy(mt,te[0]); BIG_384_29_norm(mt);
            BIG_384_29_shr(mt,i*window);
            k=BIG_384_29_lastbits(mt,window);

            T = (ECP_BLS12381*)(i32X+j*ECP_I32_LEN);
            B = (ECP_BLS12381*)(i32B+k*ECP_I32_LEN);
            ECP_BLS12381_add(B,T);
        }
        ECP_BLS12381_inf(&R); ECP_BLS12381_inf(&S);
        for (j=((1<<window)-1);j>=1;j--)
        {
            B = (ECP_BLS12381*)(i32B+j*ECP_I32_LEN);
            ECP_BLS12381_add(&R,B);
            ECP_BLS12381_add(&S,&R);
        }
        for (j=0;j<window;j++){
            ECP_BLS12381_dbl(P);
        }
        ECP_BLS12381_add(P,&S);
    }
}

// Transfer the points and scalas from char array to i32 array
// which be uesd in the miracl-core lib.
void ECP_BLS12381_C2I(int n, 
            const unsigned char *X, int32_t *i32X,
            const unsigned char *e, int32_t *i32e)
{
    int i,j;
    BIG_384_29 TA;
    ECP_BLS12381 PA;

    char u[POINT_BYTES];
    char c[48] = { 0 };
    octet U = {0, sizeof(u), u};
    
    char *ep = (char *)e;
    char *xp = (char *)X;
    
    int32_t *i32;
    for(i=0;i<n;i++){    
        BIG_384_29_fromArkBytes(&TA, c, &ep);
        i32 = (int32_t *)&TA;
        for(j=0;j<BIG_I32_LEN;j++){
            i32e[i*BIG_I32_LEN + j] = i32[j];
        }
        ECP_BLS12381_fromBytes(&PA, &U, &xp);
        i32 = (int32_t *)&PA;
        for(j=0;j<ECP_I32_LEN;j++){
            i32X[i*ECP_I32_LEN + j] = i32[j];
        }
    }
}

// Run msm and return a point in miracl-core model
void ECP_muln_impl(ECP_BLS12381 *P, int n, 
            const unsigned char *X, int32_t *i32X,
            const unsigned char *e, int32_t *i32e,
            int32_t *i32B){           
    ECP_BLS12381_C2I(n, X, i32X, e, i32e);            
    ECP_BLS12381_msm_i(P, n, i32X, i32e, i32B);
}

/** @brief The interface for rust.
 *
 * input
   @param n the length of the point/scalar array
   @param X the points in char array
   @param i32X the space to store the points X in i32 array
   @param e the scalars in char array
   @param i32e the space to store the scalars e in i32 array
   @param i32B the space to store the windows used in msm
 * output
    @param P a point in the char array
*/
void ECP_muln_rust(unsigned char *P, int n, 
            const unsigned char *X, int32_t *i32X,
            const unsigned char *e, int32_t *i32e,
            int32_t *i32B)
{
    ECP_BLS12381 Out;
    ECP_muln_impl(&Out, n, X, i32X, e, i32e, i32B);
    
    char u[2*MODBYTES_384_29+1];
    octet U = {0, sizeof(u), u};
    ECP_BLS12381_toOctet(&U, &Out, false);
    for(int i=0;i<U.len;i++){
        P[i] = (unsigned char)U.val[i];
    }
}