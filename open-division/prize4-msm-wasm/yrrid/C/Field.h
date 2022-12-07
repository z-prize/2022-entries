/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Author(s):  Niall Emmart

***/

void fieldLoad(Field* f, uint32_t* source);
void fieldStore(uint32_t* destination, Field* f);

void fieldMul(Field* r, Field* a, Field* b);
void fieldMulResolve(Field* r, Field* a, Field* b);
void fieldSqr(Field* r, Field* a);
void FieldSqrResolve(Field* r, Field* a);

void fieldMul64PartialReduce(Field* r, Field* f);
void fieldMul64FullReduce(Field* r, Field* f);

bool fieldIsZero(Field* f);
void fieldPartialReduce(Field* f);
void fieldFullReduce(Field* f);

void fieldToMontgomeryFullReduce(Field* r, Field* f);
void fieldFromMontgomeryFullReduce(Field* r, Field* f);
void fieldInverseFullReduce(Field* r, Field* f);

void fromMontgomery(uint32_t* r, Field *f) ;
void getField(uint32_t* packed, Field* f, bool convertFromMontgomery);
void fieldDumpInternal(Field* f);
void fieldDump(Field* f, bool convertFromMontgomery);

extern Field N;

static inline uint64_t computeNP0(uint32_t x) {
  uint64_t inv=x;

  inv=inv*(inv*x+14);
  inv=inv*(inv*x+2);
  inv=inv*(inv*x+2);
  inv=inv*(inv*x+2);
  return inv & 0x3FFFFFFF;
}

static inline void fieldSetZero(Field* r) {
  r->f0 = 0;
  r->f1 = 0;
  r->f2 = 0;
  r->f3 = 0;
  r->f4 = 0;
  r->f5 = 0;
  r->f6 = 0;
  r->f7 = 0;
  r->f8 = 0;
  r->f9 = 0;
  r->f10 = 0;
  r->f11 = 0;
  r->f12 = 0;
}

static inline void fieldSetR(Field* r) {
  r->f0 = 0x00d1ff2e;
  r->f1 = 0x19d80000;
  r->f2 = 0x34800ac4;
  r->f3 = 0x2e00cde6;
  r->f4 = 0x02431c84;
  r->f5 = 0x269f83a2;
  r->f6 = 0x3dcf80dd;
  r->f7 = 0x09b42da0;
  r->f8 = 0x25eec26c;
  r->f9 = 0x15d98f12;
  r->f10 = 0x04b29f14;
  r->f11 = 0x259fcfa0;
  r->f12 = 0x00015de9;
}

static inline void fieldSetN(Field* r) {
  r->f0 = 0x3fffaaab;
  r->f1 = 0x27fbffff;
  r->f2 = 0x153ffffb;
  r->f3 = 0x2affffac;
  r->f4 = 0x30f6241e;
  r->f5 = 0x034a83da;
  r->f6 = 0x112bf673;
  r->f7 = 0x12e13ce1;
  r->f8 = 0x2cd76477;
  r->f9 = 0x1ed90d2e;
  r->f10 = 0x29a4b1ba;
  r->f11 = 0x3a8e5ff9;
  r->f12 = 0x001a0111;
}

static inline void fieldSet(Field* r, Field* f) {
  r->f0 = f->f0;
  r->f1 = f->f1;
  r->f2 = f->f2;
  r->f3 = f->f3;
  r->f4 = f->f4;
  r->f5 = f->f5;
  r->f6 = f->f6;
  r->f7 = f->f7;
  r->f8 = f->f8;
  r->f9 = f->f9;
  r->f10 = f->f10;
  r->f11 = f->f11;
  r->f12 = f->f12;
}

static inline void fieldMask(Field* f) {
  uint64_t mask=0x3FFFFFFFu;

  f->f0 &= mask;
  f->f1 &= mask;
  f->f2 &= mask;
  f->f3 &= mask;
  f->f4 &= mask;
  f->f5 &= mask;
  f->f6 &= mask;
  f->f7 &= mask;
  f->f8 &= mask;
  f->f9 &= mask;
  f->f10 &= mask;
  f->f11 &= mask;
}

static inline void fieldAddResolve(Field* r, Field* a, Field* b) {
  r->f0 = a->f0 + b->f0;
  r->f1 = a->f1 + b->f1 + (r->f0>>30);
  r->f2 = a->f2 + b->f2 + (r->f1>>30);
  r->f3 = a->f3 + b->f3 + (r->f2>>30);
  r->f4 = a->f4 + b->f4 + (r->f3>>30);
  r->f5 = a->f5 + b->f5 + (r->f4>>30);
  r->f6 = a->f6 + b->f6 + (r->f5>>30);
  r->f7 = a->f7 + b->f7 + (r->f6>>30);
  r->f8 = a->f8 + b->f8 + (r->f7>>30);
  r->f9 = a->f9 + b->f9 + (r->f8>>30);
  r->f10 = a->f10 + b->f10 + (r->f9>>30);
  r->f11 = a->f11 + b->f11 + (r->f10>>30);
  r->f12 = a->f12 + b->f12 + (r->f11>>30);
  fieldMask(r);
}

static inline void fieldSubResolve(Field* r, Field* a, Field* b) {
  r->f0 = a->f0 - b->f0;
  r->f1 = a->f1 - b->f1 + (((int64_t)r->f0)>>30);
  r->f2 = a->f2 - b->f2 + (((int64_t)r->f1)>>30);
  r->f3 = a->f3 - b->f3 + (((int64_t)r->f2)>>30);
  r->f4 = a->f4 - b->f4 + (((int64_t)r->f3)>>30);
  r->f5 = a->f5 - b->f5 + (((int64_t)r->f4)>>30);
  r->f6 = a->f6 - b->f6 + (((int64_t)r->f5)>>30);
  r->f7 = a->f7 - b->f7 + (((int64_t)r->f6)>>30);
  r->f8 = a->f8 - b->f8 + (((int64_t)r->f7)>>30);
  r->f9 = a->f9 - b->f9 + (((int64_t)r->f8)>>30);
  r->f10 = a->f10 - b->f10 + (((int64_t)r->f9)>>30);
  r->f11 = a->f11 - b->f11 + (((int64_t)r->f10)>>30);
  r->f12 = a->f12 - b->f12 + (((int64_t)r->f11)>>30);
  fieldMask(r);
}
