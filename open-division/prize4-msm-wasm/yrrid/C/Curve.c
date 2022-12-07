/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Author(s):  Niall Emmart

***/

void loadXY(PointXY* r, uint32_t* source) {
  fieldLoad(&r->x, source);
  fieldLoad(&r->y, source+12);
}

void storeXY(uint32_t* destination, PointXY* point) {
  fieldStore(destination, &point->x);
  fieldStore(destination+12, &point->y);
}

void loadXYZZ(PointXYZZ* r, uint32_t* source) {
  fieldLoad(&r->x, source);
  fieldLoad(&r->y, source+12);
  fieldLoad(&r->zz, source+24);
  fieldLoad(&r->zzz, source+36);
}

void storeXYZZ(uint32_t* destination, PointXYZZ* point) {
  fieldStore(destination, &point->x);
  fieldStore(destination+12, &point->y);
  fieldStore(destination+24, &point->zz);
  fieldStore(destination+36, &point->zzz);
}

void storeAccumulatorXYZZ(uint32_t* destination, AccumulatorXYZZ* accumulator) {
  if(accumulator->infinity) {
    for(int i=0;i<48;i++)
      destination[i]=0;
  }
  else if(accumulator->affine) {
    Field local;

    fieldSetR(&local);
    fieldStore(destination, &accumulator->xyzz.x);
    fieldStore(destination+12, &accumulator->xyzz.y);
    fieldStore(destination+24, &local);   // hopefully the compiler generates decent code for this
    fieldStore(destination+36, &local);
  }
  else {
    fieldStore(destination, &accumulator->xyzz.x);
    fieldStore(destination+12, &accumulator->xyzz.y);
    fieldStore(destination+24, &accumulator->xyzz.zz);
    fieldStore(destination+36, &accumulator->xyzz.zzz);
  }
}

void copyXY(PointXY* r, PointXY* point) {
  fieldSet(&r->x, &point->x);
  fieldSet(&r->y, &point->y);
}

void copyXYZZ(PointXYZZ* r, PointXYZZ* point) {
  fieldSet(&r->x, &point->x);
  fieldSet(&r->y, &point->y);
  fieldSet(&r->zz, &point->zz);
  fieldSet(&r->zzz, &point->zzz);
}

void negateXY(PointXY* affine) {
  // point.y = -point.y + N
  affine->y.f0 = - affine->y.f0 + 0xFFFFAAABu;
  affine->y.f1 = - affine->y.f1 + 0xE7FBFFFCu + (affine->y.f0>>30);
  affine->y.f2 = - affine->y.f2 + 0xD53FFFF8u + (affine->y.f1>>30);
  affine->y.f3 = - affine->y.f3 + 0xEAFFFFA9u + (affine->y.f2>>30);
  affine->y.f4 = - affine->y.f4 + 0xF0F6241Bu + (affine->y.f3>>30);
  affine->y.f5 = - affine->y.f5 + 0xC34A83D7u + (affine->y.f4>>30);
  affine->y.f6 = - affine->y.f6 + 0xD12BF670u + (affine->y.f5>>30);
  affine->y.f7 = - affine->y.f7 + 0xD2E13CDEu + (affine->y.f6>>30);
  affine->y.f8 = - affine->y.f8 + 0xECD76474u + (affine->y.f7>>30);
  affine->y.f9 = - affine->y.f9 + 0xDED90D2Bu + (affine->y.f8>>30);
  affine->y.f10 = - affine->y.f10 + 0xE9A4B1B7u + (affine->y.f9>>30);
  affine->y.f11 = - affine->y.f11 + 0xFA8E5FF6u + (affine->y.f10>>30);
  affine->y.f12 = - affine->y.f12 + 0x001A010Eu + (affine->y.f11>>30);
  fieldMask(&affine->y);
}

void negateXYZZ(PointXYZZ* point) {
  // point.y = -point.y + 5N
  point->y.f0 = - point->y.f0 + 0xFFFE5557u;
  point->y.f1 = - point->y.f1 + 0xC7EBFFFCu + (point->y.f0>>30);
  point->y.f2 = - point->y.f2 + 0xEA3FFFE7u + (point->y.f1>>30);
  point->y.f3 = - point->y.f3 + 0xD6FFFe5Au + (point->y.f2>>30);
  point->y.f4 = - point->y.f4 + 0xF4CEB496u + (point->y.f3>>30);
  point->y.f5 = - point->y.f5 + 0xD0749342u + (point->y.f4>>30);
  point->y.f6 = - point->y.f6 + 0xD5DBD03Cu + (point->y.f5>>30);
  point->y.f7 = - point->y.f7 + 0xDE663063u + (point->y.f6>>30);
  point->y.f8 = - point->y.f8 + 0xE034F651u + (point->y.f7>>30);
  point->y.f9 = - point->y.f9 + 0xDA3D41E6u + (point->y.f8>>30);
  point->y.f10 = - point->y.f10 + 0xD03778A1u + (point->y.f9>>30);
  point->y.f11 = - point->y.f11 + 0xE4C7DFDDu + (point->y.f10>>30);
  point->y.f12 = - point->y.f12 + 0x00820556u + (point->y.f11>>30);
  fieldMask(&point->y);
}

void dumpXY(PointXY* point) {
  logString("  x=");
  fieldDump(&point->x, true);
  logString("  y=");
  fieldDump(&point->y, true);
}

void dumpXYZZ(PointXYZZ* point) {
  logString("  x=");
  fieldDump(&point->x, true);
  logString("  y=");
  fieldDump(&point->y, true);
  logString(" zz=");
  fieldDump(&point->zz, true);
  logString("zzz=");
  fieldDump(&point->zzz, true);
}

void initializeAccumulatorXYZZ(AccumulatorXYZZ* accumulator) {
  accumulator->infinity=true;
  accumulator->affine=false;
}

void doubleXY(AccumulatorXYZZ* a, PointXY* p) {
  Field T0, T1;

  a->infinity=false;
  a->affine=false;

  fieldAddResolve(&T0, &p->y, &p->y);
  fieldSqrResolve(&a->xyzz.zz, &T0);
  fieldMulResolve(&a->xyzz.zzz, &T0, &a->xyzz.zz);

  fieldMulResolve(&T0, &p->x, &a->xyzz.zz);

  fieldSqrResolve(&a->xyzz.x, &p->x); 
  fieldMulResolve(&a->xyzz.y, &p->y, &a->xyzz.zzz);

  // T1 = a.x*3
  T1.f0 = a->xyzz.x.f0*3;
  T1.f1 = a->xyzz.x.f1*3 + (T1.f0>>30);
  T1.f2 = a->xyzz.x.f2*3 + (T1.f1>>30);
  T1.f3 = a->xyzz.x.f3*3 + (T1.f2>>30);
  T1.f4 = a->xyzz.x.f4*3 + (T1.f3>>30);
  T1.f5 = a->xyzz.x.f5*3 + (T1.f4>>30);
  T1.f6 = a->xyzz.x.f6*3 + (T1.f5>>30);
  T1.f7 = a->xyzz.x.f7*3 + (T1.f6>>30);
  T1.f8 = a->xyzz.x.f8*3 + (T1.f7>>30);
  T1.f9 = a->xyzz.x.f9*3 + (T1.f8>>30);
  T1.f10 = a->xyzz.x.f10*3 + (T1.f9>>30);
  T1.f11 = a->xyzz.x.f11*3 + (T1.f10>>30);
  T1.f12 = a->xyzz.x.f12*3 + (T1.f11>>30);
  fieldMask(&T1);

  fieldSqr(&a->xyzz.x, &T1);

  // a.x = a.x - a.zzz - 2*T0 + 3N
  a->xyzz.x.f0 = a->xyzz.x.f0 - T0.f0*2 + 0xFFFF0001u;
  a->xyzz.x.f1 = a->xyzz.x.f1 - T0.f1*2 + 0xF7F3FFFCu + (a->xyzz.x.f0>>30);
  a->xyzz.x.f2 = a->xyzz.x.f2 - T0.f2*2 + 0xFFBFFFEFu + (a->xyzz.x.f1>>30);
  a->xyzz.x.f3 = a->xyzz.x.f3 - T0.f3*2 + 0xC0FFFF01u + (a->xyzz.x.f2>>30);
  a->xyzz.x.f4 = a->xyzz.x.f4 - T0.f4*2 + 0xD2E26C59u + (a->xyzz.x.f3>>30);
  a->xyzz.x.f5 = a->xyzz.x.f5 - T0.f5*2 + 0xC9DF8B8Du + (a->xyzz.x.f4>>30);
  a->xyzz.x.f6 = a->xyzz.x.f6 - T0.f6*2 + 0xF383E356u + (a->xyzz.x.f5>>30);
  a->xyzz.x.f7 = a->xyzz.x.f7 - T0.f7*2 + 0xF8A3B6A0u + (a->xyzz.x.f6>>30);
  a->xyzz.x.f8 = a->xyzz.x.f8 - T0.f8*2 + 0xC6862D62u + (a->xyzz.x.f7>>30);
  a->xyzz.x.f9 = a->xyzz.x.f9 - T0.f9*2 + 0xDC8B2789u + (a->xyzz.x.f8>>30);
  a->xyzz.x.f10 = a->xyzz.x.f10 - T0.f10*2 + 0xFCEE152Cu + (a->xyzz.x.f9>>30);
  a->xyzz.x.f11 = a->xyzz.x.f11 - T0.f11*2 + 0xEFAB1FE9u + (a->xyzz.x.f10>>30);
  a->xyzz.x.f12 = a->xyzz.x.f12 - T0.f12*2 + 0x004E0332u + (a->xyzz.x.f11>>30);
  fieldMask(&a->xyzz.x);

  fieldPartialReduce(&a->xyzz.x);

  // T0 = T0 - a.x + 2N
  T0.f0 = T0.f0 - a->xyzz.x.f0 + 0xFFFF5556u;
  T0.f1 = T0.f1 - a->xyzz.x.f1 + 0xCFF7FFFCu + (T0.f0>>30);
  T0.f2 = T0.f2 - a->xyzz.x.f2 + 0xEA7FFFF4u + (T0.f1>>30);
  T0.f3 = T0.f3 - a->xyzz.x.f3 + 0xD5FFFF55u + (T0.f2>>30);
  T0.f4 = T0.f4 - a->xyzz.x.f4 + 0xE1EC483Au + (T0.f3>>30);
  T0.f5 = T0.f5 - a->xyzz.x.f5 + 0xC69507B2u + (T0.f4>>30);
  T0.f6 = T0.f6 - a->xyzz.x.f6 + 0xE257ECE3u + (T0.f5>>30);
  T0.f7 = T0.f7 - a->xyzz.x.f7 + 0xE5C279BFu + (T0.f6>>30);
  T0.f8 = T0.f8 - a->xyzz.x.f8 + 0xD9AEC8EBu + (T0.f7>>30);
  T0.f9 = T0.f9 - a->xyzz.x.f9 + 0xFDB21A5Au + (T0.f8>>30);
  T0.f10 = T0.f10 - a->xyzz.x.f10 + 0xD3496371u + (T0.f9>>30);
  T0.f11 = T0.f11 - a->xyzz.x.f11 + 0xF51CBFF0u + (T0.f10>>30);
  T0.f12 = T0.f12 - a->xyzz.x.f12 + 0x00340220u + (T0.f11>>30);
  fieldMask(&T0);

  fieldMul(&T0, &T0, &T1);

  // a.y = T0 - a.y + 2N
  a->xyzz.y.f0 = T0.f0 - a->xyzz.y.f0 + 0xFFFF5556u;
  a->xyzz.y.f1 = T0.f1 - a->xyzz.y.f1 + 0xCFF7FFFCu + (a->xyzz.y.f0>>30);
  a->xyzz.y.f2 = T0.f2 - a->xyzz.y.f2 + 0xEA7FFFF4u + (a->xyzz.y.f1>>30);
  a->xyzz.y.f3 = T0.f3 - a->xyzz.y.f3 + 0xD5FFFF55u + (a->xyzz.y.f2>>30);
  a->xyzz.y.f4 = T0.f4 - a->xyzz.y.f4 + 0xE1EC483Au + (a->xyzz.y.f3>>30);
  a->xyzz.y.f5 = T0.f5 - a->xyzz.y.f5 + 0xC69507B2u + (a->xyzz.y.f4>>30);
  a->xyzz.y.f6 = T0.f6 - a->xyzz.y.f6 + 0xE257ECE3u + (a->xyzz.y.f5>>30);
  a->xyzz.y.f7 = T0.f7 - a->xyzz.y.f7 + 0xE5C279BFu + (a->xyzz.y.f6>>30);
  a->xyzz.y.f8 = T0.f8 - a->xyzz.y.f8 + 0xD9AEC8EBu + (a->xyzz.y.f7>>30);
  a->xyzz.y.f9 = T0.f9 - a->xyzz.y.f9 + 0xFDB21A5Au + (a->xyzz.y.f8>>30);
  a->xyzz.y.f10 = T0.f10 - a->xyzz.y.f10 + 0xD3496371u + (a->xyzz.y.f9>>30);
  a->xyzz.y.f11 = T0.f11 - a->xyzz.y.f11 + 0xF51CBFF0u + (a->xyzz.y.f10>>30);
  a->xyzz.y.f12 = T0.f12 - a->xyzz.y.f12 + 0x00340220u + (a->xyzz.y.f11>>30);
  fieldMask(&a->xyzz.y);
}

void doubleXYZZ(AccumulatorXYZZ* a, PointXYZZ* p) {
  Field T0, T1, T2;

  a->infinity=false;
  a->affine=false;

  fieldAddResolve(&T1, &p->y, &p->y);
  fieldPartialReduce(&T1);

  fieldSqrResolve(&T0, &T1);      
  fieldMulResolve(&T1, &T0, &T1);  

  fieldMulResolve(&a->xyzz.zz, &p->zz, &T0);
  fieldMulResolve(&a->xyzz.zzz, &p->zzz, &T1);

  fieldMulResolve(&T0, &p->x, &T0);
  fieldSqrResolve(&a->xyzz.x, &p->x);
  fieldMulResolve(&a->xyzz.y, &p->y, &T1);

  // T1 = a.x*3
  T1.f0 = a->xyzz.x.f0*3;
  T1.f1 = a->xyzz.x.f1*3 + (T1.f0>>30);
  T1.f2 = a->xyzz.x.f2*3 + (T1.f1>>30);
  T1.f3 = a->xyzz.x.f3*3 + (T1.f2>>30);
  T1.f4 = a->xyzz.x.f4*3 + (T1.f3>>30);
  T1.f5 = a->xyzz.x.f5*3 + (T1.f4>>30);
  T1.f6 = a->xyzz.x.f6*3 + (T1.f5>>30);
  T1.f7 = a->xyzz.x.f7*3 + (T1.f6>>30);
  T1.f8 = a->xyzz.x.f8*3 + (T1.f7>>30);
  T1.f9 = a->xyzz.x.f9*3 + (T1.f8>>30);
  T1.f10 = a->xyzz.x.f10*3 + (T1.f9>>30);
  T1.f11 = a->xyzz.x.f11*3 + (T1.f10>>30);
  T1.f12 = a->xyzz.x.f12*3 + (T1.f11>>30);
  fieldMask(&T1);

  fieldSqr(&a->xyzz.x, &T1);

  // a.x = a.x - a.zzz - 2*T0 + 3N
  a->xyzz.x.f0 = a->xyzz.x.f0 - T0.f0*2 + 0xFFFF0001u;
  a->xyzz.x.f1 = a->xyzz.x.f1 - T0.f1*2 + 0xF7F3FFFCu + (a->xyzz.x.f0>>30);
  a->xyzz.x.f2 = a->xyzz.x.f2 - T0.f2*2 + 0xFFBFFFEFu + (a->xyzz.x.f1>>30);
  a->xyzz.x.f3 = a->xyzz.x.f3 - T0.f3*2 + 0xC0FFFF01u + (a->xyzz.x.f2>>30);
  a->xyzz.x.f4 = a->xyzz.x.f4 - T0.f4*2 + 0xD2E26C59u + (a->xyzz.x.f3>>30);
  a->xyzz.x.f5 = a->xyzz.x.f5 - T0.f5*2 + 0xC9DF8B8Du + (a->xyzz.x.f4>>30);
  a->xyzz.x.f6 = a->xyzz.x.f6 - T0.f6*2 + 0xF383E356u + (a->xyzz.x.f5>>30);
  a->xyzz.x.f7 = a->xyzz.x.f7 - T0.f7*2 + 0xF8A3B6A0u + (a->xyzz.x.f6>>30);
  a->xyzz.x.f8 = a->xyzz.x.f8 - T0.f8*2 + 0xC6862D62u + (a->xyzz.x.f7>>30);
  a->xyzz.x.f9 = a->xyzz.x.f9 - T0.f9*2 + 0xDC8B2789u + (a->xyzz.x.f8>>30);
  a->xyzz.x.f10 = a->xyzz.x.f10 - T0.f10*2 + 0xFCEE152Cu + (a->xyzz.x.f9>>30);
  a->xyzz.x.f11 = a->xyzz.x.f11 - T0.f11*2 + 0xEFAB1FE9u + (a->xyzz.x.f10>>30);
  a->xyzz.x.f12 = a->xyzz.x.f12 - T0.f12*2 + 0x004E0332u + (a->xyzz.x.f11>>30);
  fieldMask(&a->xyzz.x);

  fieldPartialReduce(&a->xyzz.x);

  // T0 = T0 - a.x + 2N
  T0.f0 = T0.f0 - a->xyzz.x.f0 + 0xFFFF5556u;
  T0.f1 = T0.f1 - a->xyzz.x.f1 + 0xCFF7FFFCu + (T0.f0>>30);
  T0.f2 = T0.f2 - a->xyzz.x.f2 + 0xEA7FFFF4u + (T0.f1>>30);
  T0.f3 = T0.f3 - a->xyzz.x.f3 + 0xD5FFFF55u + (T0.f2>>30);
  T0.f4 = T0.f4 - a->xyzz.x.f4 + 0xE1EC483Au + (T0.f3>>30);
  T0.f5 = T0.f5 - a->xyzz.x.f5 + 0xC69507B2u + (T0.f4>>30);
  T0.f6 = T0.f6 - a->xyzz.x.f6 + 0xE257ECE3u + (T0.f5>>30);
  T0.f7 = T0.f7 - a->xyzz.x.f7 + 0xE5C279BFu + (T0.f6>>30);
  T0.f8 = T0.f8 - a->xyzz.x.f8 + 0xD9AEC8EBu + (T0.f7>>30);
  T0.f9 = T0.f9 - a->xyzz.x.f9 + 0xFDB21A5Au + (T0.f8>>30);
  T0.f10 = T0.f10 - a->xyzz.x.f10 + 0xD3496371u + (T0.f9>>30);
  T0.f11 = T0.f11 - a->xyzz.x.f11 + 0xF51CBFF0u + (T0.f10>>30);
  T0.f12 = T0.f12 - a->xyzz.x.f12 + 0x00340220u + (T0.f11>>30);
  fieldMask(&T0);

  fieldMul(&T0, &T0, &T1);

  // a.y = T0 - a.y + 2N
  a->xyzz.y.f0 = T0.f0 - a->xyzz.y.f0 + 0xFFFF5556u;
  a->xyzz.y.f1 = T0.f1 - a->xyzz.y.f1 + 0xCFF7FFFCu + (a->xyzz.y.f0>>30);
  a->xyzz.y.f2 = T0.f2 - a->xyzz.y.f2 + 0xEA7FFFF4u + (a->xyzz.y.f1>>30);
  a->xyzz.y.f3 = T0.f3 - a->xyzz.y.f3 + 0xD5FFFF55u + (a->xyzz.y.f2>>30);
  a->xyzz.y.f4 = T0.f4 - a->xyzz.y.f4 + 0xE1EC483Au + (a->xyzz.y.f3>>30);
  a->xyzz.y.f5 = T0.f5 - a->xyzz.y.f5 + 0xC69507B2u + (a->xyzz.y.f4>>30);
  a->xyzz.y.f6 = T0.f6 - a->xyzz.y.f6 + 0xE257ECE3u + (a->xyzz.y.f5>>30);
  a->xyzz.y.f7 = T0.f7 - a->xyzz.y.f7 + 0xE5C279BFu + (a->xyzz.y.f6>>30);
  a->xyzz.y.f8 = T0.f8 - a->xyzz.y.f8 + 0xD9AEC8EBu + (a->xyzz.y.f7>>30);
  a->xyzz.y.f9 = T0.f9 - a->xyzz.y.f9 + 0xFDB21A5Au + (a->xyzz.y.f8>>30);
  a->xyzz.y.f10 = T0.f10 - a->xyzz.y.f10 + 0xD3496371u + (a->xyzz.y.f9>>30);
  a->xyzz.y.f11 = T0.f11 - a->xyzz.y.f11 + 0xF51CBFF0u + (a->xyzz.y.f10>>30);
  a->xyzz.y.f12 = T0.f12 - a->xyzz.y.f12 + 0x00340220u + (a->xyzz.y.f11>>30);
  fieldMask(&a->xyzz.y);
}

void doubleAccumulatorXYZZ(AccumulatorXYZZ* a, AccumulatorXYZZ* p) {
  if(p->infinity) {
    a->infinity=true;
    a->affine=false;
  }
  else if(p->affine) {
    PointXY local;

    fieldSet(&local.x, &p->xyzz.x);
    fieldSet(&local.y, &p->xyzz.y);
    doubleXY(a, &local);
  }
  else
    doubleXYZZ(a, &p->xyzz);
}

static inline void _add_XY_XY(AccumulatorXYZZ* a, PointXY* p) {
  Field T0, T1;

  // T0 = p.x - a.x + N
  T0.f0 = p->x.f0 - a->xyzz.x.f0 + 0xFFFFAAABu;
  T0.f1 = p->x.f1 - a->xyzz.x.f1 + 0xE7FBFFFCu + (T0.f0>>30);
  T0.f2 = p->x.f2 - a->xyzz.x.f2 + 0xD53FFFF8u + (T0.f1>>30);
  T0.f3 = p->x.f3 - a->xyzz.x.f3 + 0xEAFFFFA9u + (T0.f2>>30);
  T0.f4 = p->x.f4 - a->xyzz.x.f4 + 0xF0F6241Bu + (T0.f3>>30);
  T0.f5 = p->x.f5 - a->xyzz.x.f5 + 0xC34A83D7u + (T0.f4>>30);
  T0.f6 = p->x.f6 - a->xyzz.x.f6 + 0xD12BF670u + (T0.f5>>30);
  T0.f7 = p->x.f7 - a->xyzz.x.f7 + 0xD2E13CDEu + (T0.f6>>30);
  T0.f8 = p->x.f8 - a->xyzz.x.f8 + 0xECD76474u + (T0.f7>>30);
  T0.f9 = p->x.f9 - a->xyzz.x.f9 + 0xDED90D2Bu + (T0.f8>>30);
  T0.f10 = p->x.f10 - a->xyzz.x.f10 + 0xE9A4B1B7u + (T0.f9>>30);
  T0.f11 = p->x.f11 - a->xyzz.x.f11 + 0xFA8E5FF6u + (T0.f10>>30);
  T0.f12 = p->x.f12 - a->xyzz.x.f12 + 0x001A010Eu + (T0.f11>>30);
  fieldMask(&T0);

  // T1 = p.y - a.y + N
  T1.f0 = p->y.f0 - a->xyzz.y.f0 + 0xFFFFAAABu;
  T1.f1 = p->y.f1 - a->xyzz.y.f1 + 0xE7FBFFFCu + (T1.f0>>30);
  T1.f2 = p->y.f2 - a->xyzz.y.f2 + 0xD53FFFF8u + (T1.f1>>30);
  T1.f3 = p->y.f3 - a->xyzz.y.f3 + 0xEAFFFFA9u + (T1.f2>>30);
  T1.f4 = p->y.f4 - a->xyzz.y.f4 + 0xF0F6241Bu + (T1.f3>>30);
  T1.f5 = p->y.f5 - a->xyzz.y.f5 + 0xC34A83D7u + (T1.f4>>30);
  T1.f6 = p->y.f6 - a->xyzz.y.f6 + 0xD12BF670u + (T1.f5>>30);
  T1.f7 = p->y.f7 - a->xyzz.y.f7 + 0xD2E13CDEu + (T1.f6>>30);
  T1.f8 = p->y.f8 - a->xyzz.y.f8 + 0xECD76474u + (T1.f7>>30);
  T1.f9 = p->y.f9 - a->xyzz.y.f9 + 0xDED90D2Bu + (T1.f8>>30);
  T1.f10 = p->y.f10 - a->xyzz.y.f10 + 0xE9A4B1B7u + (T1.f9>>30);
  T1.f11 = p->y.f11 - a->xyzz.y.f11 + 0xFA8E5FF6u + (T1.f10>>30);
  T1.f12 = p->y.f12 - a->xyzz.y.f12 + 0x001A010Eu + (T1.f11>>30);
  fieldMask(&T1);

  if(fieldIsZero(&T0) && fieldIsZero(&T1)) {
    doubleXY(a, p);
    return;
  }

  fieldSqrResolve(&a->xyzz.zz, &T0);
  fieldMulResolve(&a->xyzz.zzz, &T0, &a->xyzz.zz);
  fieldMulResolve(&T0, &a->xyzz.x, &a->xyzz.zz);
  fieldMulResolve(&a->xyzz.y, &a->xyzz.y, &a->xyzz.zzz);

  fieldSqr(&a->xyzz.x, &T1);

  // a.x = a.x - a.zzz - 2*T0 + 4N
  a->xyzz.x.f0 = a->xyzz.x.f0 - a->xyzz.zzz.f0 - T0.f0*2 + 0xFFFEAAACu;
  a->xyzz.x.f1 = a->xyzz.x.f1 - a->xyzz.zzz.f1 - T0.f1*2 + 0xDFEFFFFCu + (a->xyzz.x.f0>>30);
  a->xyzz.x.f2 = a->xyzz.x.f2 - a->xyzz.zzz.f2 - T0.f2*2 + 0xD4FFFFEBu + (a->xyzz.x.f1>>30);
  a->xyzz.x.f3 = a->xyzz.x.f3 - a->xyzz.zzz.f3 - T0.f3*2 + 0xEBFFFEAEu + (a->xyzz.x.f2>>30);
  a->xyzz.x.f4 = a->xyzz.x.f4 - a->xyzz.zzz.f4 - T0.f4*2 + 0xC3D89077u + (a->xyzz.x.f3>>30);
  a->xyzz.x.f5 = a->xyzz.x.f5 - a->xyzz.zzz.f5 - T0.f5*2 + 0xCD2A0F68u + (a->xyzz.x.f4>>30);
  a->xyzz.x.f6 = a->xyzz.x.f6 - a->xyzz.zzz.f6 - T0.f6*2 + 0xC4AFD9C9u + (a->xyzz.x.f5>>30);
  a->xyzz.x.f7 = a->xyzz.x.f7 - a->xyzz.zzz.f7 - T0.f7*2 + 0xCB84F382u + (a->xyzz.x.f6>>30);
  a->xyzz.x.f8 = a->xyzz.x.f8 - a->xyzz.zzz.f8 - T0.f8*2 + 0xF35D91DAu + (a->xyzz.x.f7>>30);
  a->xyzz.x.f9 = a->xyzz.x.f9 - a->xyzz.zzz.f9 - T0.f9*2 + 0xFB6434B7u + (a->xyzz.x.f8>>30);
  a->xyzz.x.f10 = a->xyzz.x.f10 - a->xyzz.zzz.f10 - T0.f10*2 + 0xE692C6E6u + (a->xyzz.x.f9>>30);
  a->xyzz.x.f11 = a->xyzz.x.f11 - a->xyzz.zzz.f11 - T0.f11*2 + 0xEA397FE3u + (a->xyzz.x.f10>>30);
  a->xyzz.x.f12 = a->xyzz.x.f12 - a->xyzz.zzz.f12 - T0.f12*2 + 0x00680444u + (a->xyzz.x.f11>>30);
  fieldMask(&a->xyzz.x);

  fieldPartialReduce(&a->xyzz.x);

  // T0 = T0 - a.x + 2N
  T0.f0 = T0.f0 - a->xyzz.x.f0 + 0xFFFF5556u;
  T0.f1 = T0.f1 - a->xyzz.x.f1 + 0xCFF7FFFCu + (T0.f0>>30);
  T0.f2 = T0.f2 - a->xyzz.x.f2 + 0xEA7FFFF4u + (T0.f1>>30);
  T0.f3 = T0.f3 - a->xyzz.x.f3 + 0xD5FFFF55u + (T0.f2>>30);
  T0.f4 = T0.f4 - a->xyzz.x.f4 + 0xE1EC483Au + (T0.f3>>30);
  T0.f5 = T0.f5 - a->xyzz.x.f5 + 0xC69507B2u + (T0.f4>>30);
  T0.f6 = T0.f6 - a->xyzz.x.f6 + 0xE257ECE3u + (T0.f5>>30);
  T0.f7 = T0.f7 - a->xyzz.x.f7 + 0xE5C279BFu + (T0.f6>>30);
  T0.f8 = T0.f8 - a->xyzz.x.f8 + 0xD9AEC8EBu + (T0.f7>>30);
  T0.f9 = T0.f9 - a->xyzz.x.f9 + 0xFDB21A5Au + (T0.f8>>30);
  T0.f10 = T0.f10 - a->xyzz.x.f10 + 0xD3496371u + (T0.f9>>30);
  T0.f11 = T0.f11 - a->xyzz.x.f11 + 0xF51CBFF0u + (T0.f10>>30);
  T0.f12 = T0.f12 - a->xyzz.x.f12 + 0x00340220u + (T0.f11>>30);
  fieldMask(&T0);

  fieldMul(&T0, &T0, &T1);

  // a.y = T0 - a.y + 2N
  a->xyzz.y.f0 = T0.f0 - a->xyzz.y.f0 + 0xFFFF5556u;
  a->xyzz.y.f1 = T0.f1 - a->xyzz.y.f1 + 0xCFF7FFFCu + (a->xyzz.y.f0>>30);
  a->xyzz.y.f2 = T0.f2 - a->xyzz.y.f2 + 0xEA7FFFF4u + (a->xyzz.y.f1>>30);
  a->xyzz.y.f3 = T0.f3 - a->xyzz.y.f3 + 0xD5FFFF55u + (a->xyzz.y.f2>>30);
  a->xyzz.y.f4 = T0.f4 - a->xyzz.y.f4 + 0xE1EC483Au + (a->xyzz.y.f3>>30);
  a->xyzz.y.f5 = T0.f5 - a->xyzz.y.f5 + 0xC69507B2u + (a->xyzz.y.f4>>30);
  a->xyzz.y.f6 = T0.f6 - a->xyzz.y.f6 + 0xE257ECE3u + (a->xyzz.y.f5>>30);
  a->xyzz.y.f7 = T0.f7 - a->xyzz.y.f7 + 0xE5C279BFu + (a->xyzz.y.f6>>30);
  a->xyzz.y.f8 = T0.f8 - a->xyzz.y.f8 + 0xD9AEC8EBu + (a->xyzz.y.f7>>30);
  a->xyzz.y.f9 = T0.f9 - a->xyzz.y.f9 + 0xFDB21A5Au + (a->xyzz.y.f8>>30);
  a->xyzz.y.f10 = T0.f10 - a->xyzz.y.f10 + 0xD3496371u + (a->xyzz.y.f9>>30);
  a->xyzz.y.f11 = T0.f11 - a->xyzz.y.f11 + 0xF51CBFF0u + (a->xyzz.y.f10>>30);
  a->xyzz.y.f12 = T0.f12 - a->xyzz.y.f12 + 0x00340220u + (a->xyzz.y.f11>>30);
  fieldMask(&a->xyzz.y);
}

static inline void _add_XYZZ_XY(AccumulatorXYZZ* a, PointXY* p) {
  Field T0, T1, T2;

  if(fieldIsZero(&a->xyzz.zz)) {
    a->affine=true;
    fieldSet(&a->xyzz.x, &p->x);
    fieldSet(&a->xyzz.y, &p->y);
    return;
  }

  fieldMul(&T0, &p->x, &a->xyzz.zz);
  fieldMul(&T1, &p->y, &a->xyzz.zzz);

  // T0 = T0 - a.x + 2N
  T0.f0 = T0.f0 - a->xyzz.x.f0 + 0xFFFF5556u;
  T0.f1 = T0.f1 - a->xyzz.x.f1 + 0xCFF7FFFCu + (T0.f0>>30);
  T0.f2 = T0.f2 - a->xyzz.x.f2 + 0xEA7FFFF4u + (T0.f1>>30);
  T0.f3 = T0.f3 - a->xyzz.x.f3 + 0xD5FFFF55u + (T0.f2>>30);
  T0.f4 = T0.f4 - a->xyzz.x.f4 + 0xE1EC483Au + (T0.f3>>30);
  T0.f5 = T0.f5 - a->xyzz.x.f5 + 0xC69507B2u + (T0.f4>>30);
  T0.f6 = T0.f6 - a->xyzz.x.f6 + 0xE257ECE3u + (T0.f5>>30);
  T0.f7 = T0.f7 - a->xyzz.x.f7 + 0xE5C279BFu + (T0.f6>>30);
  T0.f8 = T0.f8 - a->xyzz.x.f8 + 0xD9AEC8EBu + (T0.f7>>30);
  T0.f9 = T0.f9 - a->xyzz.x.f9 + 0xFDB21A5Au + (T0.f8>>30);
  T0.f10 = T0.f10 - a->xyzz.x.f10 + 0xD3496371u + (T0.f9>>30);
  T0.f11 = T0.f11 - a->xyzz.x.f11 + 0xF51CBFF0u + (T0.f10>>30);
  T0.f12 = T0.f12 - a->xyzz.x.f12 + 0x00340220u + (T0.f11>>30);
  fieldMask(&T0);

  // T1 = T1 - a.y + 5N
  T1.f0 = T1.f0 - a->xyzz.y.f0 + 0xFFFE5557u;
  T1.f1 = T1.f1 - a->xyzz.y.f1 + 0xC7EBFFFCu + (T1.f0>>30);
  T1.f2 = T1.f2 - a->xyzz.y.f2 + 0xEA3FFFE7u + (T1.f1>>30);
  T1.f3 = T1.f3 - a->xyzz.y.f3 + 0xD6FFFe5Au + (T1.f2>>30);
  T1.f4 = T1.f4 - a->xyzz.y.f4 + 0xF4CEB496u + (T1.f3>>30);
  T1.f5 = T1.f5 - a->xyzz.y.f5 + 0xD0749342u + (T1.f4>>30);
  T1.f6 = T1.f6 - a->xyzz.y.f6 + 0xD5DBD03Cu + (T1.f5>>30);
  T1.f7 = T1.f7 - a->xyzz.y.f7 + 0xDE663063u + (T1.f6>>30);
  T1.f8 = T1.f8 - a->xyzz.y.f8 + 0xE034F651u + (T1.f7>>30);
  T1.f9 = T1.f9 - a->xyzz.y.f9 + 0xDA3D41E6u + (T1.f8>>30);
  T1.f10 = T1.f10 - a->xyzz.y.f10 + 0xD03778A1u + (T1.f9>>30);
  T1.f11 = T1.f11 - a->xyzz.y.f11 + 0xE4C7DFDDu + (T1.f10>>30);
  T1.f12 = T1.f12 - a->xyzz.y.f12 + 0x00820556u + (T1.f11>>30);
  fieldMask(&T1);

  if(fieldIsZero(&T0) && fieldIsZero(&T1)) {
    doubleXY(a, p);
    return;
  }

  fieldSqrResolve(&T2, &T0);
  fieldMulResolve(&T0, &T0, &T2);
  fieldMulResolve(&a->xyzz.zz, &a->xyzz.zz, &T2);
  fieldMulResolve(&a->xyzz.zzz, &a->xyzz.zzz, &T0);
  fieldMulResolve(&T2, &a->xyzz.x, &T2);
  fieldMulResolve(&a->xyzz.y, &a->xyzz.y, &T0);

  fieldSqr(&a->xyzz.x, &T1);

  // a.x = a.x - T0 - T2*2 + 5N
  a->xyzz.x.f0 = a->xyzz.x.f0 - T0.f0 - T2.f0*2 + 0xFFFE5557u;
  a->xyzz.x.f1 = a->xyzz.x.f1 - T0.f1 - T2.f1*2 + 0xC7EBFFFCu + (a->xyzz.x.f0>>30);
  a->xyzz.x.f2 = a->xyzz.x.f2 - T0.f2 - T2.f2*2 + 0xEA3FFFE7u + (a->xyzz.x.f1>>30);
  a->xyzz.x.f3 = a->xyzz.x.f3 - T0.f3 - T2.f3*2 + 0xD6FFFe5Au + (a->xyzz.x.f2>>30);
  a->xyzz.x.f4 = a->xyzz.x.f4 - T0.f4 - T2.f4*2 + 0xF4CEB496u + (a->xyzz.x.f3>>30);
  a->xyzz.x.f5 = a->xyzz.x.f5 - T0.f5 - T2.f5*2 + 0xD0749342u + (a->xyzz.x.f4>>30);
  a->xyzz.x.f6 = a->xyzz.x.f6 - T0.f6 - T2.f6*2 + 0xD5DBD03Cu + (a->xyzz.x.f5>>30);
  a->xyzz.x.f7 = a->xyzz.x.f7 - T0.f7 - T2.f7*2 + 0xDE663063u + (a->xyzz.x.f6>>30);
  a->xyzz.x.f8 = a->xyzz.x.f8 - T0.f8 - T2.f8*2 + 0xE034F651u + (a->xyzz.x.f7>>30);
  a->xyzz.x.f9 = a->xyzz.x.f9 - T0.f9 - T2.f9*2 + 0xDA3D41E6u + (a->xyzz.x.f8>>30);
  a->xyzz.x.f10 = a->xyzz.x.f10 - T0.f10 - T2.f10*2 + 0xD03778A1u + (a->xyzz.x.f9>>30);
  a->xyzz.x.f11 = a->xyzz.x.f11 - T0.f11 - T2.f11*2 + 0xE4C7DFDDu + (a->xyzz.x.f10>>30);
  a->xyzz.x.f12 = a->xyzz.x.f12 - T0.f12 - T2.f12*2 + 0x00820556u + (a->xyzz.x.f11>>30);
  fieldMask(&a->xyzz.x);

  fieldPartialReduce(&a->xyzz.x);

  // T2 = T2 - a.x + 2N
  T2.f0 = T2.f0 - a->xyzz.x.f0 + 0xFFFF5556u;
  T2.f1 = T2.f1 - a->xyzz.x.f1 + 0xCFF7FFFCu + (T2.f0>>30);
  T2.f2 = T2.f2 - a->xyzz.x.f2 + 0xEA7FFFF4u + (T2.f1>>30);
  T2.f3 = T2.f3 - a->xyzz.x.f3 + 0xD5FFFF55u + (T2.f2>>30);
  T2.f4 = T2.f4 - a->xyzz.x.f4 + 0xE1EC483Au + (T2.f3>>30);
  T2.f5 = T2.f5 - a->xyzz.x.f5 + 0xC69507B2u + (T2.f4>>30);
  T2.f6 = T2.f6 - a->xyzz.x.f6 + 0xE257ECE3u + (T2.f5>>30);
  T2.f7 = T2.f7 - a->xyzz.x.f7 + 0xE5C279BFu + (T2.f6>>30);
  T2.f8 = T2.f8 - a->xyzz.x.f8 + 0xD9AEC8EBu + (T2.f7>>30);
  T2.f9 = T2.f9 - a->xyzz.x.f9 + 0xFDB21A5Au + (T2.f8>>30);
  T2.f10 = T2.f10 - a->xyzz.x.f10 + 0xD3496371u + (T2.f9>>30);
  T2.f11 = T2.f11 - a->xyzz.x.f11 + 0xF51CBFF0u + (T2.f10>>30);
  T2.f12 = T2.f12 - a->xyzz.x.f12 + 0x00340220u + (T2.f11>>30);
  fieldMask(&T2);

  fieldMul(&T2, &T1, &T2);

  // a.y = T2 - a.y + 2N
  a->xyzz.y.f0 = T2.f0 - a->xyzz.y.f0 + 0xFFFF5556u;
  a->xyzz.y.f1 = T2.f1 - a->xyzz.y.f1 + 0xCFF7FFFCu + (a->xyzz.y.f0>>30);
  a->xyzz.y.f2 = T2.f2 - a->xyzz.y.f2 + 0xEA7FFFF4u + (a->xyzz.y.f1>>30);
  a->xyzz.y.f3 = T2.f3 - a->xyzz.y.f3 + 0xD5FFFF55u + (a->xyzz.y.f2>>30);
  a->xyzz.y.f4 = T2.f4 - a->xyzz.y.f4 + 0xE1EC483Au + (a->xyzz.y.f3>>30);
  a->xyzz.y.f5 = T2.f5 - a->xyzz.y.f5 + 0xC69507B2u + (a->xyzz.y.f4>>30);
  a->xyzz.y.f6 = T2.f6 - a->xyzz.y.f6 + 0xE257ECE3u + (a->xyzz.y.f5>>30);
  a->xyzz.y.f7 = T2.f7 - a->xyzz.y.f7 + 0xE5C279BFu + (a->xyzz.y.f6>>30);
  a->xyzz.y.f8 = T2.f8 - a->xyzz.y.f8 + 0xD9AEC8EBu + (a->xyzz.y.f7>>30);
  a->xyzz.y.f9 = T2.f9 - a->xyzz.y.f9 + 0xFDB21A5Au + (a->xyzz.y.f8>>30);
  a->xyzz.y.f10 = T2.f10 - a->xyzz.y.f10 + 0xD3496371u + (a->xyzz.y.f9>>30);
  a->xyzz.y.f11 = T2.f11 - a->xyzz.y.f11 + 0xF51CBFF0u + (a->xyzz.y.f10>>30);
  a->xyzz.y.f12 = T2.f12 - a->xyzz.y.f12 + 0x00340220u + (a->xyzz.y.f11>>30);
  fieldMask(&a->xyzz.y);
}

static inline void _add_XYZZ_XYZZ(AccumulatorXYZZ* a, PointXYZZ* p) {
  Field T0, T1, T2;

  if(fieldIsZero(&p->zz))
    return;

  if(fieldIsZero(&a->xyzz.zz)) {
    fieldSet(&a->xyzz.x, &p->x);
    fieldSet(&a->xyzz.y, &p->y);
    fieldSet(&a->xyzz.zz, &p->zz);
    fieldSet(&a->xyzz.zzz, &p->zzz);
    return;
  }
  
  fieldMul(&T0, &a->xyzz.zz, &p->x);
  fieldMul(&T1, &a->xyzz.zzz, &p->y);
  fieldMulResolve(&a->xyzz.x, &a->xyzz.x, &p->zz);
  fieldMulResolve(&a->xyzz.y, &a->xyzz.y, &p->zzz);

  // T0 = T0 - a.x + 2N
  T0.f0 = T0.f0 - a->xyzz.x.f0 + 0xFFFF5556u;
  T0.f1 = T0.f1 - a->xyzz.x.f1 + 0xCFF7FFFCu + (T0.f0>>30);
  T0.f2 = T0.f2 - a->xyzz.x.f2 + 0xEA7FFFF4u + (T0.f1>>30);
  T0.f3 = T0.f3 - a->xyzz.x.f3 + 0xD5FFFF55u + (T0.f2>>30);
  T0.f4 = T0.f4 - a->xyzz.x.f4 + 0xE1EC483Au + (T0.f3>>30);
  T0.f5 = T0.f5 - a->xyzz.x.f5 + 0xC69507B2u + (T0.f4>>30);
  T0.f6 = T0.f6 - a->xyzz.x.f6 + 0xE257ECE3u + (T0.f5>>30);
  T0.f7 = T0.f7 - a->xyzz.x.f7 + 0xE5C279BFu + (T0.f6>>30);
  T0.f8 = T0.f8 - a->xyzz.x.f8 + 0xD9AEC8EBu + (T0.f7>>30);
  T0.f9 = T0.f9 - a->xyzz.x.f9 + 0xFDB21A5Au + (T0.f8>>30);
  T0.f10 = T0.f10 - a->xyzz.x.f10 + 0xD3496371u + (T0.f9>>30);
  T0.f11 = T0.f11 - a->xyzz.x.f11 + 0xF51CBFF0u + (T0.f10>>30);
  T0.f12 = T0.f12 - a->xyzz.x.f12 + 0x00340220u + (T0.f11>>30);
  fieldMask(&T0);

  // T1 = T1 - a.y + 2N
  T1.f0 = T1.f0 - a->xyzz.y.f0 + 0xFFFF5556u;
  T1.f1 = T1.f1 - a->xyzz.y.f1 + 0xCFF7FFFCu + (T1.f0>>30);
  T1.f2 = T1.f2 - a->xyzz.y.f2 + 0xEA7FFFF4u + (T1.f1>>30);
  T1.f3 = T1.f3 - a->xyzz.y.f3 + 0xD5FFFF55u + (T1.f2>>30);
  T1.f4 = T1.f4 - a->xyzz.y.f4 + 0xE1EC483Au + (T1.f3>>30);
  T1.f5 = T1.f5 - a->xyzz.y.f5 + 0xC69507B2u + (T1.f4>>30);
  T1.f6 = T1.f6 - a->xyzz.y.f6 + 0xE257ECE3u + (T1.f5>>30);
  T1.f7 = T1.f7 - a->xyzz.y.f7 + 0xE5C279BFu + (T1.f6>>30);
  T1.f8 = T1.f8 - a->xyzz.y.f8 + 0xD9AEC8EBu + (T1.f7>>30);
  T1.f9 = T1.f9 - a->xyzz.y.f9 + 0xFDB21A5Au + (T1.f8>>30);
  T1.f10 = T1.f10 - a->xyzz.y.f10 + 0xD3496371u + (T1.f9>>30);
  T1.f11 = T1.f11 - a->xyzz.y.f11 + 0xF51CBFF0u + (T1.f10>>30);
  T1.f12 = T1.f12 - a->xyzz.y.f12 + 0x00340220u + (T1.f11>>30);
  fieldMask(&T1);

  if(fieldIsZero(&T0) && fieldIsZero(&T1)) {
    doubleXYZZ(a, p);
    return;
  }
  
  fieldSqrResolve(&T2, &T0);
  fieldMulResolve(&T0, &T0, &T2);

  fieldMulResolve(&a->xyzz.zz, &a->xyzz.zz, &p->zz);
  fieldMulResolve(&a->xyzz.zz, &a->xyzz.zz, &T2);
  fieldMulResolve(&a->xyzz.zzz, &a->xyzz.zzz, &p->zzz);
  fieldMulResolve(&a->xyzz.zzz, &a->xyzz.zzz, &T0);

  fieldMulResolve(&T2, &a->xyzz.x, &T2);
  fieldMulResolve(&a->xyzz.y, &a->xyzz.y, &T0);

  fieldSqr(&a->xyzz.x, &T1);

  // a.x = a.x - T0 - T2*2 + 5N
  a->xyzz.x.f0 = a->xyzz.x.f0 - T0.f0 - T2.f0*2 + 0xFFFE5557u;
  a->xyzz.x.f1 = a->xyzz.x.f1 - T0.f1 - T2.f1*2 + 0xC7EBFFFCu + (a->xyzz.x.f0>>30);
  a->xyzz.x.f2 = a->xyzz.x.f2 - T0.f2 - T2.f2*2 + 0xEA3FFFE7u + (a->xyzz.x.f1>>30);
  a->xyzz.x.f3 = a->xyzz.x.f3 - T0.f3 - T2.f3*2 + 0xD6FFFe5Au + (a->xyzz.x.f2>>30);
  a->xyzz.x.f4 = a->xyzz.x.f4 - T0.f4 - T2.f4*2 + 0xF4CEB496u + (a->xyzz.x.f3>>30);
  a->xyzz.x.f5 = a->xyzz.x.f5 - T0.f5 - T2.f5*2 + 0xD0749342u + (a->xyzz.x.f4>>30);
  a->xyzz.x.f6 = a->xyzz.x.f6 - T0.f6 - T2.f6*2 + 0xD5DBD03Cu + (a->xyzz.x.f5>>30);
  a->xyzz.x.f7 = a->xyzz.x.f7 - T0.f7 - T2.f7*2 + 0xDE663063u + (a->xyzz.x.f6>>30);
  a->xyzz.x.f8 = a->xyzz.x.f8 - T0.f8 - T2.f8*2 + 0xE034F651u + (a->xyzz.x.f7>>30);
  a->xyzz.x.f9 = a->xyzz.x.f9 - T0.f9 - T2.f9*2 + 0xDA3D41E6u + (a->xyzz.x.f8>>30);
  a->xyzz.x.f10 = a->xyzz.x.f10 - T0.f10 - T2.f10*2 + 0xD03778A1u + (a->xyzz.x.f9>>30);
  a->xyzz.x.f11 = a->xyzz.x.f11 - T0.f11 - T2.f11*2 + 0xE4C7DFDDu + (a->xyzz.x.f10>>30);
  a->xyzz.x.f12 = a->xyzz.x.f12 - T0.f12 - T2.f12*2 + 0x00820556u + (a->xyzz.x.f11>>30);
  fieldMask(&a->xyzz.x);

  fieldPartialReduce(&a->xyzz.x);

  // T2 = T2 - a.x + 2N
  T2.f0 = T2.f0 - a->xyzz.x.f0 + 0xFFFF5556u;
  T2.f1 = T2.f1 - a->xyzz.x.f1 + 0xCFF7FFFCu + (T2.f0>>30);
  T2.f2 = T2.f2 - a->xyzz.x.f2 + 0xEA7FFFF4u + (T2.f1>>30);
  T2.f3 = T2.f3 - a->xyzz.x.f3 + 0xD5FFFF55u + (T2.f2>>30);
  T2.f4 = T2.f4 - a->xyzz.x.f4 + 0xE1EC483Au + (T2.f3>>30);
  T2.f5 = T2.f5 - a->xyzz.x.f5 + 0xC69507B2u + (T2.f4>>30);
  T2.f6 = T2.f6 - a->xyzz.x.f6 + 0xE257ECE3u + (T2.f5>>30);
  T2.f7 = T2.f7 - a->xyzz.x.f7 + 0xE5C279BFu + (T2.f6>>30);
  T2.f8 = T2.f8 - a->xyzz.x.f8 + 0xD9AEC8EBu + (T2.f7>>30);
  T2.f9 = T2.f9 - a->xyzz.x.f9 + 0xFDB21A5Au + (T2.f8>>30);
  T2.f10 = T2.f10 - a->xyzz.x.f10 + 0xD3496371u + (T2.f9>>30);
  T2.f11 = T2.f11 - a->xyzz.x.f11 + 0xF51CBFF0u + (T2.f10>>30);
  T2.f12 = T2.f12 - a->xyzz.x.f12 + 0x00340220u + (T2.f11>>30);
  fieldMask(&T2);

  fieldMul(&T2, &T1, &T2);

  // a.y = T2 - a.y + 2N
  a->xyzz.y.f0 = T2.f0 - a->xyzz.y.f0 + 0xFFFF5556u;
  a->xyzz.y.f1 = T2.f1 - a->xyzz.y.f1 + 0xCFF7FFFCu + (a->xyzz.y.f0>>30);
  a->xyzz.y.f2 = T2.f2 - a->xyzz.y.f2 + 0xEA7FFFF4u + (a->xyzz.y.f1>>30);
  a->xyzz.y.f3 = T2.f3 - a->xyzz.y.f3 + 0xD5FFFF55u + (a->xyzz.y.f2>>30);
  a->xyzz.y.f4 = T2.f4 - a->xyzz.y.f4 + 0xE1EC483Au + (a->xyzz.y.f3>>30);
  a->xyzz.y.f5 = T2.f5 - a->xyzz.y.f5 + 0xC69507B2u + (a->xyzz.y.f4>>30);
  a->xyzz.y.f6 = T2.f6 - a->xyzz.y.f6 + 0xE257ECE3u + (a->xyzz.y.f5>>30);
  a->xyzz.y.f7 = T2.f7 - a->xyzz.y.f7 + 0xE5C279BFu + (a->xyzz.y.f6>>30);
  a->xyzz.y.f8 = T2.f8 - a->xyzz.y.f8 + 0xD9AEC8EBu + (a->xyzz.y.f7>>30);
  a->xyzz.y.f9 = T2.f9 - a->xyzz.y.f9 + 0xFDB21A5Au + (a->xyzz.y.f8>>30);
  a->xyzz.y.f10 = T2.f10 - a->xyzz.y.f10 + 0xD3496371u + (a->xyzz.y.f9>>30);
  a->xyzz.y.f11 = T2.f11 - a->xyzz.y.f11 + 0xF51CBFF0u + (a->xyzz.y.f10>>30);
  a->xyzz.y.f12 = T2.f12 - a->xyzz.y.f12 + 0x00340220u + (a->xyzz.y.f11>>30);
  fieldMask(&a->xyzz.y);
}

void addXY(AccumulatorXYZZ* accumulator, PointXY* point) {
  if(accumulator->infinity) {
    accumulator->infinity=false;
    accumulator->affine=true;
    fieldSet(&accumulator->xyzz.x, &point->x);
    fieldSet(&accumulator->xyzz.y, &point->y);
  }
  else if(accumulator->affine) {
    accumulator->affine=false;
    _add_XY_XY(accumulator, point);
  }
  else {
    _add_XYZZ_XY(accumulator, point);
  }
}

void addXYZZ(AccumulatorXYZZ* accumulator, PointXYZZ* point) {
  if(accumulator->infinity) {
    accumulator->infinity=false;
    accumulator->affine=false;
    fieldSet(&accumulator->xyzz.x, &point->x);
    fieldSet(&accumulator->xyzz.y, &point->y);
    fieldSet(&accumulator->xyzz.zz, &point->zz);
    fieldSet(&accumulator->xyzz.zzz, &point->zzz);
  }
  else if(accumulator->affine) {
    // not a common case, don't optimize
    accumulator->affine=false;
    fieldSetR(&accumulator->xyzz.zz);
    fieldSetR(&accumulator->xyzz.zzz);
    _add_XYZZ_XYZZ(accumulator, point);
  }
  else {
    _add_XYZZ_XYZZ(accumulator, point);
  }
}

void getAccumulator(PointXYZZ* r, AccumulatorXYZZ* accumulator) {
  if(accumulator->infinity) {
    fieldSetZero(&r->x);
    fieldSetZero(&r->y);
    fieldSetZero(&r->zz);
    fieldSetZero(&r->zzz);
  }
  else if(accumulator->affine) {
    fieldSet(&r->x, &accumulator->xyzz.x);
    fieldSet(&r->y, &accumulator->xyzz.y);
    fieldSetR(&r->zz);
    fieldSetR(&r->zzz);
  }
  else {
    fieldSet(&r->x, &accumulator->xyzz.x);
    fieldSet(&r->y, &accumulator->xyzz.y);
    fieldSet(&r->zz, &accumulator->xyzz.zz);
    fieldSet(&r->zzz, &accumulator->xyzz.zzz);
  }
}

void normalizeXYZZ(PointXY* r, PointXYZZ* point) {
  Field inv;

  if(fieldIsZero(&point->zz)) {
    fieldSetZero(&r->x);
    fieldSetZero(&r->y);
    return;
  }

  fieldInverseFullReduce(&inv, &point->zzz);
  fieldMulResolve(&r->y, &inv, &point->y);
  fieldFullReduce(&r->y);    // overkill, since r < 2N

  fieldMulResolve(&inv, &inv, &point->zz);
  fieldSqrResolve(&inv, &inv);
  fieldMulResolve(&r->x, &inv, &point->x);
  fieldFullReduce(&r->x);    // overkill, since r < 2N 
}

void scaleXY(PointXYZZ* r, PointXY* point, uint32_t* scalar, uint32_t bits) {
  AccumulatorXYZZ acc;

  initializeAccumulatorXYZZ(&acc);
  for(int i=bits-1;i>=0;i--) {
    int word=i>>5, bit=i & 0x1F;

    if(!acc.infinity) {
      if(acc.affine) {
        doubleXY(&acc, point);
        acc.affine=false;
      }
      else
        doubleXYZZ(&acc, &acc.xyzz);
    }

    if(((scalar[word]>>bit) & 0x01)!=0) 
      addXY(&acc, point);
  }
  getAccumulator(r, &acc);
}

void scaleXYZZ(PointXYZZ* r, PointXYZZ* point, uint32_t* scalar, uint32_t bits) {
  AccumulatorXYZZ acc;

  initializeAccumulatorXYZZ(&acc);
  for(int i=bits-1;i>=0;i--) {
    int word=i>>5, bit=i & 0x1F;

    if(!acc.infinity) 
      doubleXYZZ(&acc, &acc.xyzz);
    if(((scalar[word]>>bit) & 0x01)!=0) 
      addXYZZ(&acc, point);
  }
  getAccumulator(r, &acc);
}

void scaleByLambdaXY(PointXY* r, PointXY* point) {
  fieldMulResolve(&r->x, &point->x, &lambdaTerm);
  if((r->x.f12 << 30) + r->x.f11 >= 0x680447A8E5FF9ull) {
    Field local;

    fieldSubResolve(&local, &r->x, &N);
    if(((int64_t)local.f12)>=0) 
      fieldSet(&r->x, &local);
  }
  fieldSet(&r->y, &point->y);
}

// Batched Accumulate XY APIs
void initializeAccumulatorXY(AccumulatorXY* accumulator) {
  fieldSetZero(&accumulator->xy.x);
  accumulator->xy.x.f12=0x00800000;
  fieldSetZero(&accumulator->xy.y);
}

void initializeFieldState(Field* state) {
  fieldSetZero(state);
  state->f12=0x00800000;  
}

void addXYPhaseOne(Field* state, AccumulatorXY* acc, PointXY* affine, uint32_t* inverses) {
  Field deltaX, deltaY;

  if(((acc->xy.x.f12 | affine->x.f12) & 0x00800000)!=0)
    return;

  // deltaX = affine.x - acc.xy.x + 2N
  deltaX.f0 = affine->x.f0 - acc->xy.x.f0 + 0xFFFF5556u;
  deltaX.f1 = affine->x.f1 - acc->xy.x.f1 + 0xCFF7FFFCu + (deltaX.f0>>30);
  deltaX.f2 = affine->x.f2 - acc->xy.x.f2 + 0xEA7FFFF4u + (deltaX.f1>>30);
  deltaX.f3 = affine->x.f3 - acc->xy.x.f3 + 0xD5FFFF55u + (deltaX.f2>>30);
  deltaX.f4 = affine->x.f4 - acc->xy.x.f4 + 0xE1EC483Au + (deltaX.f3>>30);
  deltaX.f5 = affine->x.f5 - acc->xy.x.f5 + 0xC69507B2u + (deltaX.f4>>30);
  deltaX.f6 = affine->x.f6 - acc->xy.x.f6 + 0xE257ECE3u + (deltaX.f5>>30);
  deltaX.f7 = affine->x.f7 - acc->xy.x.f7 + 0xE5C279BFu + (deltaX.f6>>30);
  deltaX.f8 = affine->x.f8 - acc->xy.x.f8 + 0xD9AEC8EBu + (deltaX.f7>>30);
  deltaX.f9 = affine->x.f9 - acc->xy.x.f9 + 0xFDB21A5Au + (deltaX.f8>>30);
  deltaX.f10 = affine->x.f10 - acc->xy.x.f10 + 0xD3496371u + (deltaX.f9>>30);
  deltaX.f11 = affine->x.f11 - acc->xy.x.f11 + 0xF51CBFF0u + (deltaX.f10>>30);
  deltaX.f12 = affine->x.f12 - acc->xy.x.f12 + 0x00340220u + (deltaX.f11>>30);
  fieldMask(&deltaX);

  if(fieldIsZero(&deltaX)) {
    // deltaY = affine.y - acc.xy.y + 4N
    deltaY.f0 = affine->y.f0 - acc->xy.y.f0 + 0xFFFEAAACu;
    deltaY.f1 = affine->y.f1 - acc->xy.y.f1 + 0xDFEFFFFCu + (deltaY.f0>>30);
    deltaY.f2 = affine->y.f2 - acc->xy.y.f2 + 0xD4FFFFEBu + (deltaY.f1>>30);
    deltaY.f3 = affine->y.f3 - acc->xy.y.f3 + 0xEBFFFEAEu + (deltaY.f2>>30);
    deltaY.f4 = affine->y.f4 - acc->xy.y.f4 + 0xC3D89077u + (deltaY.f3>>30);
    deltaY.f5 = affine->y.f5 - acc->xy.y.f5 + 0xCD2A0F68u + (deltaY.f4>>30);
    deltaY.f6 = affine->y.f6 - acc->xy.y.f6 + 0xC4AFD9C9u + (deltaY.f5>>30);
    deltaY.f7 = affine->y.f7 - acc->xy.y.f7 + 0xCB84F382u + (deltaY.f6>>30);
    deltaY.f8 = affine->y.f8 - acc->xy.y.f8 + 0xF35D91DAu + (deltaY.f7>>30);
    deltaY.f9 = affine->y.f9 - acc->xy.y.f9 + 0xFB6434B7u + (deltaY.f8>>30);
    deltaY.f10 = affine->y.f10 - acc->xy.y.f10 + 0xE692C6E6u + (deltaY.f9>>30);
    deltaY.f11 = affine->y.f11 - acc->xy.y.f11 + 0xEA397FE3u + (deltaY.f10>>30);
    deltaY.f12 = affine->y.f12 - acc->xy.y.f12 + 0x00680444u + (deltaY.f11>>30);
    fieldMask(&deltaY);

    if(!fieldIsZero(&deltaY))
      return;

    // if deltaX is zero, we need to invert 2y
    fieldAddResolve(&deltaX, &affine->y, &affine->y);
  }

  if((state->f12 & 0x00800000)!=0) {
    Field localR;

    fieldSetR(&localR);
    fieldStore(inverses, &localR);      
    fieldSet(state, &deltaX);
  }
  else {
    fieldStore(inverses, state);
    fieldMulResolve(state, state, &deltaX);
  }
}

void addXYPhaseTwo(Field* state, AccumulatorXY* acc, PointXY* affine, uint32_t* inverses) {
  Field deltaX, deltaY;
  Field inverse;
  Field s, ss;

  if(((acc->xy.x.f12 | affine->x.f12) & 0x00800000)!=0) {
    if((affine->x.f12 & 0x00800000)==0) {
      fieldSet(&acc->xy.x, &affine->x);
      fieldSet(&acc->xy.y, &affine->y);
    }
    return;
  }

  fieldLoad(&inverse, inverses);
  fieldMulResolve(&inverse, state, &inverse);

  // deltaX = affine.x - acc.xy.x + 2N
  deltaX.f0 = affine->x.f0 - acc->xy.x.f0 + 0xFFFF5556u;
  deltaX.f1 = affine->x.f1 - acc->xy.x.f1 + 0xCFF7FFFCu + (deltaX.f0>>30);
  deltaX.f2 = affine->x.f2 - acc->xy.x.f2 + 0xEA7FFFF4u + (deltaX.f1>>30);
  deltaX.f3 = affine->x.f3 - acc->xy.x.f3 + 0xD5FFFF55u + (deltaX.f2>>30);
  deltaX.f4 = affine->x.f4 - acc->xy.x.f4 + 0xE1EC483Au + (deltaX.f3>>30);
  deltaX.f5 = affine->x.f5 - acc->xy.x.f5 + 0xC69507B2u + (deltaX.f4>>30);
  deltaX.f6 = affine->x.f6 - acc->xy.x.f6 + 0xE257ECE3u + (deltaX.f5>>30);
  deltaX.f7 = affine->x.f7 - acc->xy.x.f7 + 0xE5C279BFu + (deltaX.f6>>30);
  deltaX.f8 = affine->x.f8 - acc->xy.x.f8 + 0xD9AEC8EBu + (deltaX.f7>>30);
  deltaX.f9 = affine->x.f9 - acc->xy.x.f9 + 0xFDB21A5Au + (deltaX.f8>>30);
  deltaX.f10 = affine->x.f10 - acc->xy.x.f10 + 0xD3496371u + (deltaX.f9>>30);
  deltaX.f11 = affine->x.f11 - acc->xy.x.f11 + 0xF51CBFF0u + (deltaX.f10>>30);
  deltaX.f12 = affine->x.f12 - acc->xy.x.f12 + 0x00340220u + (deltaX.f11>>30);
  fieldMask(&deltaX);

  // deltaY = affine.y - acc.xy.y + 4N
  deltaY.f0 = affine->y.f0 - acc->xy.y.f0 + 0xFFFEAAACu;
  deltaY.f1 = affine->y.f1 - acc->xy.y.f1 + 0xDFEFFFFCu + (deltaY.f0>>30);
  deltaY.f2 = affine->y.f2 - acc->xy.y.f2 + 0xD4FFFFEBu + (deltaY.f1>>30);
  deltaY.f3 = affine->y.f3 - acc->xy.y.f3 + 0xEBFFFEAEu + (deltaY.f2>>30);
  deltaY.f4 = affine->y.f4 - acc->xy.y.f4 + 0xC3D89077u + (deltaY.f3>>30);
  deltaY.f5 = affine->y.f5 - acc->xy.y.f5 + 0xCD2A0F68u + (deltaY.f4>>30);
  deltaY.f6 = affine->y.f6 - acc->xy.y.f6 + 0xC4AFD9C9u + (deltaY.f5>>30);
  deltaY.f7 = affine->y.f7 - acc->xy.y.f7 + 0xCB84F382u + (deltaY.f6>>30);
  deltaY.f8 = affine->y.f8 - acc->xy.y.f8 + 0xF35D91DAu + (deltaY.f7>>30);
  deltaY.f9 = affine->y.f9 - acc->xy.y.f9 + 0xFB6434B7u + (deltaY.f8>>30);
  deltaY.f10 = affine->y.f10 - acc->xy.y.f10 + 0xE692C6E6u + (deltaY.f9>>30);
  deltaY.f11 = affine->y.f11 - acc->xy.y.f11 + 0xEA397FE3u + (deltaY.f10>>30);
  deltaY.f12 = affine->y.f12 - acc->xy.y.f12 + 0x00680444u + (deltaY.f11>>30);
  fieldMask(&deltaY);

  if(fieldIsZero(&deltaX)) {
    if(!fieldIsZero(&deltaY)) {
      // if deltaY==0 ==> acc=-affine, result should be pt at infinity
      initializeAccumulatorXY(acc);
      return;
    }

    // Otherwise, acc=affine, and it's point doubling
    // Processing is almost the same, except s=3*affine.x^2 / 2*affine.y

    // set deltaY = 3*affine.x^2
    fieldSqrResolve(&deltaY, &affine->x);

    deltaY.f0 = deltaY.f0*3;
    deltaY.f1 = deltaY.f1*3 + (deltaY.f0>>30);
    deltaY.f2 = deltaY.f2*3 + (deltaY.f1>>30);
    deltaY.f3 = deltaY.f3*3 + (deltaY.f2>>30);
    deltaY.f4 = deltaY.f4*3 + (deltaY.f3>>30);
    deltaY.f5 = deltaY.f5*3 + (deltaY.f4>>30);
    deltaY.f6 = deltaY.f6*3 + (deltaY.f5>>30);
    deltaY.f7 = deltaY.f7*3 + (deltaY.f6>>30);
    deltaY.f8 = deltaY.f8*3 + (deltaY.f7>>30);
    deltaY.f9 = deltaY.f9*3 + (deltaY.f8>>30);
    deltaY.f10 = deltaY.f10*3 + (deltaY.f9>>30);
    deltaY.f11 = deltaY.f11*3 + (deltaY.f10>>30);
    deltaY.f12 = deltaY.f12*3 + (deltaY.f11>>30);
    fieldMask(&deltaY);

    // set deltaX = 2*affine->y
    fieldAddResolve(&deltaX, &affine->y, &affine->y);
  }

  // get the state ready for the next iteration
  fieldMulResolve(state, state, &deltaX);

  // compute s and ss
  fieldMulResolve(&s, &deltaY, &inverse);
  fieldSqrResolve(&ss, &s);
  
  // acc.xy.x = ss - affine.x - acc.xy.x + 3N
  acc->xy.x.f0 = ss.f0 - affine->x.f0 - acc->xy.x.f0 + 0xFFFF0001u;
  acc->xy.x.f1 = ss.f1 - affine->x.f1 - acc->xy.x.f1 + 0xF7F3FFFCu + (acc->xy.x.f0>>30);
  acc->xy.x.f2 = ss.f2 - affine->x.f2 - acc->xy.x.f2 + 0xFFBFFFEFu + (acc->xy.x.f1>>30);
  acc->xy.x.f3 = ss.f3 - affine->x.f3 - acc->xy.x.f3 + 0xC0FFFF01u + (acc->xy.x.f2>>30);
  acc->xy.x.f4 = ss.f4 - affine->x.f4 - acc->xy.x.f4 + 0xD2E26C59u + (acc->xy.x.f3>>30);
  acc->xy.x.f5 = ss.f5 - affine->x.f5 - acc->xy.x.f5 + 0xC9DF8B8Du + (acc->xy.x.f4>>30);
  acc->xy.x.f6 = ss.f6 - affine->x.f6 - acc->xy.x.f6 + 0xF383E356u + (acc->xy.x.f5>>30);
  acc->xy.x.f7 = ss.f7 - affine->x.f7 - acc->xy.x.f7 + 0xF8A3B6A0u + (acc->xy.x.f6>>30);
  acc->xy.x.f8 = ss.f8 - affine->x.f8 - acc->xy.x.f8 + 0xC6862D62u + (acc->xy.x.f7>>30);
  acc->xy.x.f9 = ss.f9 - affine->x.f9 - acc->xy.x.f9 + 0xDC8B2789u + (acc->xy.x.f8>>30);
  acc->xy.x.f10 = ss.f10 - affine->x.f10 - acc->xy.x.f10 + 0xFCEE152Cu + (acc->xy.x.f9>>30);
  acc->xy.x.f11 = ss.f11 - affine->x.f11 - acc->xy.x.f11 + 0xEFAB1FE9u + (acc->xy.x.f10>>30);
  acc->xy.x.f12 = ss.f12 - affine->x.f12 - acc->xy.x.f12 + 0x004E0332u + (acc->xy.x.f11>>30);
  fieldMask(&acc->xy.x);

  fieldPartialReduce(&acc->xy.x);
  
  // deltaX = affine.x - acc.xy.x + 2N
  deltaX.f0 = affine->x.f0 - acc->xy.x.f0 + 0xFFFF5556u;
  deltaX.f1 = affine->x.f1 - acc->xy.x.f1 + 0xCFF7FFFCu + (deltaX.f0>>30);
  deltaX.f2 = affine->x.f2 - acc->xy.x.f2 + 0xEA7FFFF4u + (deltaX.f1>>30);
  deltaX.f3 = affine->x.f3 - acc->xy.x.f3 + 0xD5FFFF55u + (deltaX.f2>>30);
  deltaX.f4 = affine->x.f4 - acc->xy.x.f4 + 0xE1EC483Au + (deltaX.f3>>30);
  deltaX.f5 = affine->x.f5 - acc->xy.x.f5 + 0xC69507B2u + (deltaX.f4>>30);
  deltaX.f6 = affine->x.f6 - acc->xy.x.f6 + 0xE257ECE3u + (deltaX.f5>>30);
  deltaX.f7 = affine->x.f7 - acc->xy.x.f7 + 0xE5C279BFu + (deltaX.f6>>30);
  deltaX.f8 = affine->x.f8 - acc->xy.x.f8 + 0xD9AEC8EBu + (deltaX.f7>>30);
  deltaX.f9 = affine->x.f9 - acc->xy.x.f9 + 0xFDB21A5Au + (deltaX.f8>>30);
  deltaX.f10 = affine->x.f10 - acc->xy.x.f10 + 0xD3496371u + (deltaX.f9>>30);
  deltaX.f11 = affine->x.f11 - acc->xy.x.f11 + 0xF51CBFF0u + (deltaX.f10>>30);
  deltaX.f12 = affine->x.f12 - acc->xy.x.f12 + 0x00340220u + (deltaX.f11>>30);
  fieldMask(&deltaX);

  fieldMul(&acc->xy.y, &s, &deltaX);

  // acc.xy.y = acc.xy.y - affine.y + 2N
  acc->xy.y.f0 = acc->xy.y.f0 - affine->y.f0 + 0xFFFF5556u;
  acc->xy.y.f1 = acc->xy.y.f1 - affine->y.f1 + 0xCFF7FFFCu + (acc->xy.y.f0>>30);
  acc->xy.y.f2 = acc->xy.y.f2 - affine->y.f2 + 0xEA7FFFF4u + (acc->xy.y.f1>>30);
  acc->xy.y.f3 = acc->xy.y.f3 - affine->y.f3 + 0xD5FFFF55u + (acc->xy.y.f2>>30);
  acc->xy.y.f4 = acc->xy.y.f4 - affine->y.f4 + 0xE1EC483Au + (acc->xy.y.f3>>30);
  acc->xy.y.f5 = acc->xy.y.f5 - affine->y.f5 + 0xC69507B2u + (acc->xy.y.f4>>30);
  acc->xy.y.f6 = acc->xy.y.f6 - affine->y.f6 + 0xE257ECE3u + (acc->xy.y.f5>>30);
  acc->xy.y.f7 = acc->xy.y.f7 - affine->y.f7 + 0xE5C279BFu + (acc->xy.y.f6>>30);
  acc->xy.y.f8 = acc->xy.y.f8 - affine->y.f8 + 0xD9AEC8EBu + (acc->xy.y.f7>>30);
  acc->xy.y.f9 = acc->xy.y.f9 - affine->y.f9 + 0xFDB21A5Au + (acc->xy.y.f8>>30);
  acc->xy.y.f10 = acc->xy.y.f10 - affine->y.f10 + 0xD3496371u + (acc->xy.y.f9>>30);
  acc->xy.y.f11 = acc->xy.y.f11 - affine->y.f11 + 0xF51CBFF0u + (acc->xy.y.f10>>30);
  acc->xy.y.f12 = acc->xy.y.f12 - affine->y.f12 + 0x00340220u + (acc->xy.y.f11>>30);
  fieldMask(&acc->xy.y);
}



