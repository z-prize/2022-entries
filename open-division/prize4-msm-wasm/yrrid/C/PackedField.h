/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Author(s):  Niall Emmart
            Sougata Bhattacharya
            Antony Suresh

***/

static inline uint32_t ctz(uint32_t x) {
  return __builtin_ctz(x);
}

static inline uint32_t ctzll(uint64_t x) {
  return __builtin_ctzll(x);
}

static inline uint32_t clz(uint32_t x) {
  return __builtin_clz(x);
}

static inline uint32_t clzll(uint64_t x) {
  return __builtin_clzll(x);
}

static inline void packedFieldFromField(PackedField* pf, Field* f) {
  pf->f0=f->f0 + (f->f1 << 30) + (f->f2 << 60);
  pf->f1=(f->f2 >> 4) + (f->f3 << 26) + (f->f4 << 56);
  pf->f2=(f->f4 >> 8) + (f->f5 << 22) + (f->f6 << 52);
  pf->f3=(f->f6 >> 12) + (f->f7 << 18) + (f->f8 << 48);
  pf->f4=(f->f8 >> 16) + (f->f9 << 14) + (f->f10 << 44);
  pf->f5=(f->f10 >> 20) + (f->f11 << 10) + (f->f12 << 40);
}

static inline void packedFieldToField(Field* f, PackedField* pf) {
  uint64_t mask=0x3FFFFFFF;

  f->f0 = pf->f0 & mask;
  f->f1 = (pf->f0 >> 30) & mask;
  f->f2 = (pf->f0 >> 60) + (pf->f1 << 4) & mask;
  f->f3 = (pf->f1 >> 26) & mask;
  f->f4 = (pf->f1 >> 56) + (pf->f2 << 8) & mask;
  f->f5 = (pf->f2 >> 22) & mask;
  f->f6 = (pf->f2 >> 52) + (pf->f3 << 12) & mask;
  f->f7 = (pf->f3 >> 18) & mask;
  f->f8 = (pf->f3 >> 48) + (pf->f4 << 16) & mask;
  f->f9 = (pf->f4 >> 14) & mask;
  f->f10 = (pf->f4 >> 44) + (pf->f5 << 20) & mask;
  f->f11 = (pf->f5 >> 10) & mask;
  f->f12 = pf->f5 >> 40;
}

static inline uint64_t packedFieldLimbOR(PackedField* pf) {
  return pf->f0 | pf->f1 | pf->f2 | pf->f3 | pf->f4 | pf->f5;
}

static inline int32_t packedFieldCTZ(PackedField* pf) {
  if(pf->f0!=0)
    return ctzll(pf->f0);
  if(pf->f1!=0)
    return ctzll(pf->f1) + 64;
  if(pf->f2!=0)
    return ctzll(pf->f2) + 128;
  if(pf->f3!=0)
    return ctzll(pf->f3) + 192;
  if(pf->f4!=0)
    return ctzll(pf->f4) + 256;
  if(pf->f5!=0)
    return ctzll(pf->f5) + 320;
  return -1;
}

static inline void packedFieldSetN(PackedField* pf) {
  pf->f0=0xB9FEFFFFFFFFAAABull;
  pf->f1=0x1EABFFFEB153FFFFull;
  pf->f2=0x6730D2A0F6B0F624ull;
  pf->f3=0x64774B84F38512BFull;
  pf->f4=0x4B1BA7B6434BACD7ull;
  pf->f5=0x1A0111EA397FE69Aull;
}

static inline void packedFieldShiftRightOne(PackedField* pf) {
  pf->f0=(pf->f0 >> 1) + (pf->f1 << 63);
  pf->f1=(pf->f1 >> 1) + (pf->f2 << 63);
  pf->f2=(pf->f2 >> 1) + (pf->f3 << 63);
  pf->f3=(pf->f3 >> 1) + (pf->f4 << 63);
  pf->f4=(pf->f4 >> 1) + (pf->f5 << 63);
  pf->f5=pf->f5 >> 1;
}

static inline void packedFieldShiftLeft(PackedField* pf, uint32_t count) {
  uint32_t left=count & 0x3F, right=64-left;

  if(left!=0) {
    pf->f5=(pf->f5 << left) + (pf->f4 >> right);
    pf->f4=(pf->f4 << left) + (pf->f3 >> right);
    pf->f3=(pf->f3 << left) + (pf->f2 >> right);
    pf->f2=(pf->f2 << left) + (pf->f1 >> right);
    pf->f1=(pf->f1 << left) + (pf->f0 >> right);
    pf->f0=pf->f0 << left;
  }
 
  while(count>=64) {
    count-=64;
    pf->f5=pf->f4; pf->f4=pf->f3; pf->f3=pf->f2; pf->f2=pf->f1; pf->f1=pf->f0; pf->f0=0;
  }
}

static inline void packedFieldShiftRight(PackedField* pf, uint32_t count, uint64_t fill) {
  uint32_t right=count & 0x3F, left=64-right;

  if(right!=0) {
    pf->f0=(pf->f0 >> right) + (pf->f1 << left);
    pf->f1=(pf->f1 >> right) + (pf->f2 << left);
    pf->f2=(pf->f2 >> right) + (pf->f3 << left);
    pf->f3=(pf->f3 >> right) + (pf->f4 << left);
    pf->f4=(pf->f4 >> right) + (pf->f5 << left);
    pf->f5=(pf->f5 >> right) + (fill << left);
  }

  while(count>=64) {
    count-=64;
    pf->f0=pf->f1; pf->f1=pf->f2; pf->f2=pf->f3; pf->f3=pf->f4; pf->f4=pf->f5; pf->f5=0;
  }
}

static inline void packedFieldAddN(PackedField* pf) {
  uint64_t s0, s1, s2, s3, s4, s5;

  s0=pf->f0 + 0xB9FEFFFFFFFFAAABull;
  s1=pf->f1 + 0x1EABFFFEB153FFFFull;
  s2=pf->f2 + 0x6730D2A0F6B0F624ull;
  s3=pf->f3 + 0x64774B84F38512BFull;
  s4=pf->f4 + 0x4B1BA7B6434BACD7ull;
  s5=pf->f5 + 0x1A0111EA397FE69Aull;

  pf->f5=s5 + (s4 < pf->f4 ? 1 : 0);
  pf->f4=s4 + (s3 < pf->f3 ? 1 : 0);
  pf->f3=s3 + (s2 < pf->f2 ? 1 : 0);
  pf->f2=s2 + (s1 < pf->f1 ? 1 : 0);
  pf->f1=s1 + (s0 < pf->f0 ? 1 : 0);
  pf->f0=s0;
}

static inline bool packedFieldSub(PackedField* r, PackedField* a, PackedField* b) {
  uint64_t s0, s1, s2, s3, s4, s5;
  uint64_t c0, c1, c2, c3, c4;
  bool     c5;

  s0=a->f0 + ~b->f0;
  s1=a->f1 + ~b->f1;
  s2=a->f2 + ~b->f2;
  s3=a->f3 + ~b->f3;
  s4=a->f4 + ~b->f4;
  s5=a->f5 + ~b->f5;

  c0=(s0+1 <= a->f0) ? 1 : 0;
  c1=(s1 < a->f1 || s1+c0 < c0) ? 1 : 0;
  c2=(s2 < a->f2 || s2+c1 < c1) ? 1 : 0;
  c3=(s3 < a->f3 || s3+c2 < c2) ? 1 : 0;
  c4=(s4 < a->f4 || s4+c3 < c3) ? 1 : 0;
  c5=s5 < a->f5 || s5+c4 < c4;

  r->f0 = s0 + 1;
  r->f1 = s1 + c0;
  r->f2 = s2 + c1;
  r->f3 = s3 + c2;
  r->f4 = s4 + c3;
  r->f5 = s5 + c4;

  return c5;
}

static inline bool packedFieldGreater(PackedField* a, PackedField* b) {
  if(a->f5 > b->f5) return true;
  if(a->f5 < b->f5) return false;
  if(a->f4 > b->f4) return true;
  if(a->f4 < b->f4) return false;
  if(a->f3 > b->f3) return true;
  if(a->f3 < b->f3) return false;
  if(a->f2 > b->f2) return true;
  if(a->f2 < b->f2) return false;
  if(a->f1 > b->f1) return true;
  if(a->f1 < b->f1) return false;
  return a->f0 > b->f0;
}

static inline uint64_t packedFieldMultiplySubtract(PackedField* T, PackedField* U, Coeffs c, PackedField* A, PackedField* B) {
  int64_t  tAcc, uAcc;
  uint64_t al, ah, bl, bh, m=0xFFFFFFFFull;
  uint64_t tl, ul; 

  // f0 processing
  al=A->f0&m; ah=A->f0>>32; bl=B->f0&m; bh=B->f0>>32;
  tAcc=al*c.aCoeffA - bl*c.aCoeffB;
  uAcc=bl*c.bCoeffB - al*c.bCoeffA;
  tl=tAcc & m;
  ul=uAcc & m;

  tAcc=(tAcc>>32) + ah*c.aCoeffA - bh*c.aCoeffB;
  uAcc=(uAcc>>32) + bh*c.bCoeffB - ah*c.bCoeffA;

  T->f0 = (tAcc<<32) + tl;
  U->f0 = (uAcc<<32) + ul;

  // f1 processing
  al=A->f1&m; ah=A->f1>>32; bl=B->f1&m; bh=B->f1>>32;
  tAcc=(tAcc>>32) + al*c.aCoeffA - bl*c.aCoeffB;
  uAcc=(uAcc>>32) + bl*c.bCoeffB - al*c.bCoeffA;
  tl=tAcc & m;
  ul=uAcc & m;

  tAcc=(tAcc>>32) + ah*c.aCoeffA - bh*c.aCoeffB;
  uAcc=(uAcc>>32) + bh*c.bCoeffB - ah*c.bCoeffA;

  T->f1 = (tAcc<<32) + tl;
  U->f1 = (uAcc<<32) + ul;
 
  // f2 processing
  al=A->f2&m; ah=A->f2>>32; bl=B->f2&m; bh=B->f2>>32;
  tAcc=(tAcc>>32) + al*c.aCoeffA - bl*c.aCoeffB;
  uAcc=(uAcc>>32) + bl*c.bCoeffB - al*c.bCoeffA;
  tl=tAcc & m;
  ul=uAcc & m;

  tAcc=(tAcc>>32) + ah*c.aCoeffA - bh*c.aCoeffB;
  uAcc=(uAcc>>32) + bh*c.bCoeffB - ah*c.bCoeffA;

  T->f2 = (tAcc<<32) + tl;
  U->f2 = (uAcc<<32) + ul;
 
  // f3 processing
  al=A->f3&m; ah=A->f3>>32; bl=B->f3&m; bh=B->f3>>32;
  tAcc=(tAcc>>32) + al*c.aCoeffA - bl*c.aCoeffB;
  uAcc=(uAcc>>32) + bl*c.bCoeffB - al*c.bCoeffA;
  tl=tAcc & m;
  ul=uAcc & m;

  tAcc=(tAcc>>32) + ah*c.aCoeffA - bh*c.aCoeffB;
  uAcc=(uAcc>>32) + bh*c.bCoeffB - ah*c.bCoeffA;

  T->f3 = (tAcc<<32) + tl;
  U->f3 = (uAcc<<32) + ul;

  // f4 processing
  al=A->f4&m; ah=A->f4>>32; bl=B->f4&m; bh=B->f4>>32;
  tAcc=(tAcc>>32) + al*c.aCoeffA - bl*c.aCoeffB;
  uAcc=(uAcc>>32) + bl*c.bCoeffB - al*c.bCoeffA;
  tl=tAcc & m;
  ul=uAcc & m;

  tAcc=(tAcc>>32) + ah*c.aCoeffA - bh*c.aCoeffB;
  uAcc=(uAcc>>32) + bh*c.bCoeffB - ah*c.bCoeffA;

  T->f4 = (tAcc<<32) + tl;
  U->f4 = (uAcc<<32) + ul;

  // f5 processing
  al=A->f5&m; ah=A->f5>>32; bl=B->f5&m; bh=B->f5>>32;
  tAcc=(tAcc>>32) + al*c.aCoeffA - bl*c.aCoeffB;
  uAcc=(uAcc>>32) + bl*c.bCoeffB - al*c.bCoeffA;
  tl=tAcc & m;
  ul=uAcc & m;

  tAcc=(tAcc>>32) + ah*c.aCoeffA - bh*c.aCoeffB;
  uAcc=(uAcc>>32) + bh*c.bCoeffB - ah*c.bCoeffA;

  T->f5 = (tAcc<<32) + tl;
  U->f5 = (uAcc<<32) + ul;

  return (tAcc & ~m) | (uAcc>>32);
}

static inline uint64_t packedFieldMultiplyAdd(PackedField* T, PackedField* U, Coeffs c, PackedField* A, PackedField* B) {
  int64_t  tAcc, uAcc;
  uint64_t al, ah, bl, bh, m=0xFFFFFFFFull;
  uint64_t tl, ul; 

  // f0 processing
  al=A->f0&m; ah=A->f0>>32; bl=B->f0&m; bh=B->f0>>32;
  tAcc=al*c.aCoeffA + bl*c.aCoeffB;
  uAcc=al*c.bCoeffA + bl*c.bCoeffB;
  tl=tAcc & m;
  ul=uAcc & m;

  tAcc=(tAcc>>32) + ah*c.aCoeffA + bh*c.aCoeffB;
  uAcc=(uAcc>>32) + ah*c.bCoeffA + bh*c.bCoeffB;

  T->f0 = (tAcc<<32) + tl;
  U->f0 = (uAcc<<32) + ul;

  // f1 processing
  al=A->f1&m; ah=A->f1>>32; bl=B->f1&m; bh=B->f1>>32;
  tAcc=(tAcc>>32) + al*c.aCoeffA + bl*c.aCoeffB;
  uAcc=(uAcc>>32) + al*c.bCoeffA + bl*c.bCoeffB;
  tl=tAcc & m;
  ul=uAcc & m;

  tAcc=(tAcc>>32) + ah*c.aCoeffA + bh*c.aCoeffB;
  uAcc=(uAcc>>32) + ah*c.bCoeffA + bh*c.bCoeffB;

  T->f1 = (tAcc<<32) + tl;
  U->f1 = (uAcc<<32) + ul;
 
  // f2 processing
  al=A->f2&m; ah=A->f2>>32; bl=B->f2&m; bh=B->f2>>32;
  tAcc=(tAcc>>32) + al*c.aCoeffA + bl*c.aCoeffB;
  uAcc=(uAcc>>32) + al*c.bCoeffA + bl*c.bCoeffB;
  tl=tAcc & m;
  ul=uAcc & m;

  tAcc=(tAcc>>32) + ah*c.aCoeffA + bh*c.aCoeffB;
  uAcc=(uAcc>>32) + ah*c.bCoeffA + bh*c.bCoeffB;

  T->f2 = (tAcc<<32) + tl;
  U->f2 = (uAcc<<32) + ul;
 
  // f3 processing
  al=A->f3&m; ah=A->f3>>32; bl=B->f3&m; bh=B->f3>>32;
  tAcc=(tAcc>>32) + al*c.aCoeffA + bl*c.aCoeffB;
  uAcc=(uAcc>>32) + al*c.bCoeffA + bl*c.bCoeffB;
  tl=tAcc & m;
  ul=uAcc & m;

  tAcc=(tAcc>>32) + ah*c.aCoeffA + bh*c.aCoeffB;
  uAcc=(uAcc>>32) + ah*c.bCoeffA + bh*c.bCoeffB;

  T->f3 = (tAcc<<32) + tl;
  U->f3 = (uAcc<<32) + ul;

  // f4 processing
  al=A->f4&m; ah=A->f4>>32; bl=B->f4&m; bh=B->f4>>32;
  tAcc=(tAcc>>32) + al*c.aCoeffA + bl*c.aCoeffB;
  uAcc=(uAcc>>32) + al*c.bCoeffA + bl*c.bCoeffB;
  tl=tAcc & m;
  ul=uAcc & m;

  tAcc=(tAcc>>32) + ah*c.aCoeffA + bh*c.aCoeffB;
  uAcc=(uAcc>>32) + ah*c.bCoeffA + bh*c.bCoeffB;

  T->f4 = (tAcc<<32) + tl;
  U->f4 = (uAcc<<32) + ul;

  // f5 processing
  al=A->f5&m; ah=A->f5>>32; bl=B->f5&m; bh=B->f5>>32;
  tAcc=(tAcc>>32) + al*c.aCoeffA + bl*c.aCoeffB;
  uAcc=(uAcc>>32) + al*c.bCoeffA + bl*c.bCoeffB;
  tl=tAcc & m;
  ul=uAcc & m;

  tAcc=(tAcc>>32) + ah*c.aCoeffA + bh*c.aCoeffB;
  uAcc=(uAcc>>32) + ah*c.bCoeffA + bh*c.bCoeffB;

  T->f5 = (tAcc<<32) + tl;
  U->f5 = (uAcc<<32) + ul;

  return (tAcc & ~m) | (uAcc>>32);
}

static inline void packedFieldSwap(PackedField* a, PackedField* b) {
  uint64_t swap;

  swap=a->f0; a->f0=b->f0; b->f0=swap;
  swap=a->f1; a->f1=b->f1; b->f1=swap;
  swap=a->f2; a->f2=b->f2; b->f2=swap;
  swap=a->f3; a->f3=b->f3; b->f3=swap;
  swap=a->f4; a->f4=b->f4; b->f4=swap;
  swap=a->f5; a->f5=b->f5; b->f5=swap;
}
