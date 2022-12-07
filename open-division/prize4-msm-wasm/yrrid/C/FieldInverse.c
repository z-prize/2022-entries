/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Author(s):  Niall Emmart

***/

static inline uint32_t sad(uint32_t a, uint32_t b) {
  return (a>=b) ? a-b : b-a;
}

static inline uint64_t sadll(uint64_t a, uint64_t b) {
  return (a>=b) ? a-b : b-a;
}

static inline bool useTopBottom(uint64_t* aTop, uint64_t* bTop, PackedField* a, PackedField* b) {
  uint64_t f0, f1, f2, f3, f4, f5;
  uint64_t a0, a1, b0, b1;
  uint32_t cnt;

  f5=a->f5 | b->f5;
  f4=a->f4 | b->f4;
  f3=a->f3 | b->f3;
  f2=a->f2 | b->f2;
  f1=a->f1 | b->f1;
  f0=a->f0 | b->f0;

  if(f5!=0) {
    cnt=clzll(f5);
    a1=a->f5;
    a0=a->f4;
    b1=b->f5;
    b0=b->f4;
  }
  else if(f4!=0) {
    cnt=clzll(f4);
    a1=a->f4;
    a0=a->f3;
    b1=b->f4;
    b0=b->f3;
  }
  else if(f3!=0) {
    cnt=clzll(f3);
    a1=a->f3;
    a0=a->f2;
    b1=b->f3;
    b0=b->f2;
  }
  else if(f2!=0) {
    cnt=clzll(f2);
    a1=a->f2;
    a0=a->f1;
    b1=b->f2;
    b0=b->f1;
  }
  else if(f1!=0) {
    cnt=clzll(f1);
    a1=a->f1;
    a0=a->f0;
    b1=b->f1;
    b0=b->f0;
  }
  else
    return false;

  if(cnt<=31) {
    cnt=31-cnt;
    *aTop=a1>>cnt;
    *bTop=b1>>cnt;
  }
  else {
    cnt=cnt-31;
    *aTop=(a1<<cnt) | (a0>>(64-cnt));
    *bTop=(b1<<cnt) | (b0>>(64-cnt));
  }
  return true;
}

static inline Coeffs stepBottom(uint64_t a, uint64_t b, uint32_t maxBits) {
  bool     swap, resultSwap=true;
  uint64_t f, g, m=0xFFFFFFFFull;
  uint32_t ctz=0, nextCTZ, count=0;
  uint64_t swapA, swapB, coeffSum;
  Coeffs   res;

  f=0x000000001ull;
  g=0x100000000ull;

  do {

    swapB=a;
    swapA=b-a;
    nextCTZ=ctzll(swapA);
    swap=a<b;

    a=a-b;
    a=swap ? swapA : a;
    b=swap ? swapB : b;

    resultSwap=resultSwap!=swap;

    g=g<<ctz;
    coeffSum=f+g;
    g=swap ? f : g;

    ctz=nextCTZ;
    count+=ctz;
    a=a>>ctz;
    f=coeffSum;
  } while(a!=b && count<maxBits);

  uint64_t swapF=g, swapG=f;

  f=resultSwap ? f : swapF;
  g=resultSwap ? g : swapG;

  res.aCoeffA=f & m;
  res.aCoeffB=f>>32;
  res.bCoeffA=g & m;
  res.bCoeffB=g>>32;
  
  return res;
}

static inline Coeffs stepTopBottom(uint64_t aTop, uint64_t bTop, uint64_t aBottom, uint64_t bBottom, uint32_t maxBits) {
  bool     swap, resultSwap=false;
  uint64_t a, b, f, g, m=0xFFFFFFFFull, limit=1ull<<(maxBits+1);
  uint32_t ctz=0, nextCTZ, count=0;
  uint64_t swapA, swapB, coeffSum;
  Coeffs   res;

  a=(aTop<<31) | (aBottom & 0x7FFFFFFFull);
  b=(bTop<<31) | (bBottom & 0x7FFFFFFFull);
  f=0x000000001ull;
  g=0x100000000ull;

  do {
    swapB=a;
    swapA=b-a;
    nextCTZ=ctzll(swapA);
    swap=(a<b);

    a=a-b;
    a=swap ? swapA : a;
    b=swap ? swapB : b;

    g=g<<ctz;
    if(a<limit) 
      break;

    resultSwap=resultSwap!=swap;

    coeffSum=f+g;
    g=swap ? f : g;

    ctz=nextCTZ;
    count+=ctz;
    a=a>>ctz;
    f=coeffSum;
  } while(count<maxBits);

  uint64_t swapF=g, swapG=f;

  f=resultSwap ? swapF : f;
  g=resultSwap ? swapG : g;

  res.aCoeffA=f & m;
  res.aCoeffB=f>>32;
  res.bCoeffA=g & m;
  res.bCoeffB=g>>32;

  return res;
}

#if !defined(OLD_INVERSE)
void fieldInverseFullReduce(Field* r, Field* f) { 
  bool        useBottom=false;
  PackedField A, B, T, U;
  PackedField X={1, 0, 0, 0, 0, 0}, Y={0, 0, 0, 0, 0, 0};
  Coeffs      c;
  uint64_t    aTop, bTop, aBottom, bBottom, max, min, highWords=0, m=0xFFFFFFFFull;
  uint32_t    aCount, bCount, count=0, maxBits=31;

  fieldFullReduce(f);
  packedFieldFromField(&A, f);
  packedFieldSetN(&B);

  if(packedFieldLimbOR(&A)==0) {
    fieldSetZero(r);
    return;
  }

  while(true) {
    // since the inverse exists, we will always end with A=1, B=1 on the last pass
    // which means neither A nor B can be 0 here.

    aCount=packedFieldCTZ(&A);
    bCount=packedFieldCTZ(&B);
    if(bCount>aCount) 
      packedFieldShiftLeft(&X, bCount-aCount);
    else if(aCount>bCount) 
      packedFieldShiftLeft(&Y, aCount-bCount);
    count+=(aCount>=bCount) ? aCount : bCount;
    packedFieldShiftRight(&A, aCount, highWords>>32);
    packedFieldShiftRight(&B, bCount, highWords & m);

    if(useBottom || !useTopBottom(&aTop, &bTop, &A, &B)) {
      useBottom=true;
      if(A.f0==B.f0)
        break;
      c=stepBottom(A.f0, B.f0, maxBits);
    }
    else {
      max=(aTop>=bTop) ? aTop : bTop;
      min=(aTop<=bTop) ? aTop : bTop;
      if(max-min>3)
        c=stepTopBottom(aTop, bTop, A.f0, B.f0, maxBits);
      else {
        c.aCoeffA=1;
        c.aCoeffB=packedFieldGreater(&B, &A) ? 0 : 1;
        c.bCoeffA=1-c.aCoeffB;
        c.bCoeffB=1;
      }
    }

    highWords=packedFieldMultiplySubtract(&T, &U, c, &A, &B);
    A=T;
    B=U;

    packedFieldMultiplyAdd(&T, &U, c, &X, &Y);
    X=T;
    Y=U;
  }

  packedFieldSetN(&X);
  packedFieldSub(&Y, &X, &Y);  
  packedFieldToField(r, &Y);
  fieldDivTwoExpResolve(r, r, count-390);
  fieldToMontgomeryFullReduce(r, r);
}
#endif
