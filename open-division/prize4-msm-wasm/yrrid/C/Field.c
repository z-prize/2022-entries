/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Author(s):  Niall Emmart

***/

Field fieldNMultiples[10]={
  {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
   0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
  {0x3fffaaab, 0x27fbffff, 0x153ffffb, 0x2affffac, 0x30f6241e, 0x034a83da,
   0x112bf673, 0x12e13ce1, 0x2cd76477, 0x1ed90d2e, 0x29a4b1ba, 0x3a8e5ff9, 0x001a0111},
  {0x3fff5556, 0x0ff7ffff, 0x2a7ffff7, 0x15ffff58, 0x21ec483d, 0x069507b5,
   0x2257ece6, 0x25c279c2, 0x19aec8ee, 0x3db21a5d, 0x13496374, 0x351cbff3, 0x00340223},
  {0x3fff0001, 0x37f3ffff, 0x3fbffff2, 0x00ffff04, 0x12e26c5c, 0x09df8b90,
   0x3383e359, 0x38a3b6a3, 0x06862d65, 0x1c8b278c, 0x3cee152f, 0x2fab1fec, 0x004e0335},
  {0x3ffeaaac, 0x1fefffff, 0x14ffffee, 0x2bfffeb1, 0x03d8907a, 0x0d2a0f6b,
   0x04afd9cc, 0x0b84f385, 0x335d91dd, 0x3b6434ba, 0x2692c6e9, 0x2a397fe6, 0x00680447},
  {0x3ffe5557, 0x07ebffff, 0x2a3fffea, 0x16fffe5d, 0x34ceb499, 0x10749345,
   0x15dbd03f, 0x1e663066, 0x2034f654, 0x1a3d41e9, 0x103778a4, 0x24c7dfe0, 0x00820559},
  {0x3ffe0002, 0x2fe7ffff, 0x3f7fffe5, 0x01fffe09, 0x25c4d8b8, 0x13bf1720,
   0x2707c6b2, 0x31476d47, 0x0d0c5acb, 0x39164f18, 0x39dc2a5e, 0x1f563fd9, 0x009c066b},
  {0x3ffdaaad, 0x17e3ffff, 0x14bfffe1, 0x2cfffdb6, 0x16bafcd6, 0x17099afb,
   0x3833bd25, 0x0428aa28, 0x39e3bf43, 0x17ef5c46, 0x2380dc19, 0x19e49fd3, 0x00b6077d},
  {0x3ffd5558, 0x3fdfffff, 0x29ffffdc, 0x17fffd62, 0x07b120f5, 0x1a541ed6,
   0x095fb398, 0x1709e70a, 0x26bb23ba, 0x36c86975, 0x0d258dd3, 0x1472ffcd, 0x00d0088f},
  {0x3ffd0003, 0x27dbffff, 0x3f3fffd8, 0x02fffd0e, 0x38a74514, 0x1d9ea2b0,
   0x1a8baa0b, 0x29eb23eb, 0x13928831, 0x15a176a4, 0x36ca3f8e, 0x0f015fc6, 0x00ea09a1},
};

Field N={
  0x3fffaaab, 0x27fbffff, 0x153ffffb, 0x2affffac, 0x30f6241e, 0x034a83da,
  0x112bf673, 0x12e13ce1, 0x2cd76477, 0x1ed90d2e, 0x29a4b1ba, 0x3a8e5ff9, 0x001a0111
};

Field oneTerm={
  0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000
};

Field rSquaredTerm={
  0x0510070f, 0x3b19070d, 0x0132243a, 0x299bb0e8, 0x3507af6e, 0x3b81ec77,
  0x21b145ef, 0x0a487bb4, 0x370a4144, 0x05dcb4cb, 0x18c97900, 0x3812b364, 0x000f696e
};

Field lambdaTerm={
  0x1c907181, 0x3cbde486, 0x26574c3e, 0x332475ec, 0x1c3ebc1b, 0x39ee6864,
  0x16ffa856, 0x2c3499ff, 0x0550bd16, 0x14cbac30, 0x17d18c86, 0x215959f7, 0x0009c6d4
};

uint32_t modmul=0;
uint32_t modsqr=0;

void fieldLoad(Field* f, uint32_t* source) {
  uint32_t mask=0x3FFFFFFF;  
  uint32_t d0=source[0], d1=source[1], d2=source[2], d3=source[3], d4=source[4], d5=source[5];
  uint32_t d6=source[6], d7=source[7], d8=source[8], d9=source[9], d10=source[10], d11=source[11]; 

  f->f0=d0 & mask;
  f->f1=(d0>>30) + (d1<<2) & mask;
  f->f2=(d1>>28) + (d2<<4) & mask;
  f->f3=(d2>>26) + (d3<<6) & mask;
  f->f4=(d3>>24) + (d4<<8) & mask;
  f->f5=(d4>>22) + (d5<<10) & mask;
  f->f6=(d5>>20) + (d6<<12) & mask;
  f->f7=(d6>>18) + (d7<<14) & mask;
  f->f8=(d7>>16) + (d8<<16) & mask;
  f->f9=(d8>>14) + (d9<<18) & mask;
  f->f10=(d9>>12) + (d10<<20) & mask;
  f->f11=(d10>>10) + (d11<<22) & mask;
  f->f12=(d11>>8);
}

void fieldStore(uint32_t* destination, Field* f) {
  uint32_t d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12;

  d0=(uint32_t)(f->f0 + (f->f1<<30));
  d1=(uint32_t)((f->f1>>2) + (f->f2<<28));
  d2=(uint32_t)((f->f2>>4) + (f->f3<<26)); 
  d3=(uint32_t)((f->f3>>6) + (f->f4<<24)); 
  d4=(uint32_t)((f->f4>>8) + (f->f5<<22)); 
  d5=(uint32_t)((f->f5>>10) + (f->f6<<20)); 
  d6=(uint32_t)((f->f6>>12) + (f->f7<<18)); 
  d7=(uint32_t)((f->f7>>14) + (f->f8<<16)); 
  d8=(uint32_t)((f->f8>>16) + (f->f9<<14)); 
  d9=(uint32_t)((f->f9>>18) + (f->f10<<12)); 
  d10=(uint32_t)((f->f10>>20) + (f->f11<<10)); 
  d11=(uint32_t)((f->f11>>22) + (f->f12<<8));

  destination[0]=d0; destination[1]=d1; destination[2]=d2; destination[3]=d3; destination[4]=d4; destination[5]=d5;
  destination[6]=d6; destination[7]=d7; destination[8]=d8; destination[9]=d9; destination[10]=d10; destination[11]=d11;
}

bool fieldIsZero(Field* f) {
  uint64_t top, estimate, check;
  Field*   compare;

  top=((f->f12 << 30) + f->f11);
  estimate=top*10>>54;
  check=estimate*(0xD0088F51CBFF34Dull);
  if(top!=check>>9)
    return false;

  compare=fieldNMultiples+estimate;
  return ((f->f0 ^ compare->f0) | (f->f1 ^ compare->f1) | (f->f2 ^ compare->f2) | (f->f3 ^ compare->f3) | 
          (f->f4 ^ compare->f4) | (f->f5 ^ compare->f5) | (f->f6 ^ compare->f6) | (f->f7 ^ compare->f7) |
          (f->f8 ^ compare->f8) | (f->f9 ^ compare->f9) | (f->f10 ^ compare->f10))==0;
}

void fieldPartialReduce(Field* f) {
  uint64_t estimate;

  // Returns f mod P (more or less)
  // Input:  f < 2^384   Output:  f mod P, 0 <= f <= P*1.000001
  
  estimate=((f->f12 << 8) + (f->f11 >> 22)) * 0x9D835D2Fu >> 60;
  if(estimate>0) 
    fieldSubResolve(f, f, &fieldNMultiples[estimate]);
}

void fieldFullReduce(Field* f) {
  fieldPartialReduce(f);
  if((f->f12 << 30) + f->f11 >= 0x680447A8E5FF9ull) {
    Field local;

    fieldSubResolve(&local, f, &N);
    if(((int64_t)local.f12)>=0) 
      fieldSet(f, &local);
  }
}

void fieldMul(Field* r, Field* a, Field* b) {
  uint64_t p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13,
           p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24;
  uint64_t q, mask=0x3FFFFFFFu;

  p0=0; p1=0; p2=0; p3=0; p4=0; p5=0; p6=0; p7=0; p8=0; p9=0; p10=0; p11=0; p12=0; 
  p13=0; p14=0; p15=0; p16=0; p17=0; p18=0; p19=0; p20=0; p21=0; p22=0; p23=0; p24=0;

  // #include "./generated/grade_school_mult_v1.c"
  #include "./generated/karatsuba_mult_v1.c"

  #include "./generated/grade_school_red_v1.c"
  // #include "./generated/karatsuba_red_v1.c"
  #include "./generated/no_resolve.c"

  modmul++;
}

void fieldMulResolve(Field* r, Field* a, Field* b) {
  uint64_t p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13,
           p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24;
  uint64_t q, mask=0x3FFFFFFFu;

  p0=0; p1=0; p2=0; p3=0; p4=0; p5=0; p6=0; p7=0; p8=0; p9=0; p10=0; p11=0; p12=0; 
  p13=0; p14=0; p15=0; p16=0; p17=0; p18=0; p19=0; p20=0; p21=0; p22=0; p23=0; p24=0;

  // #include "./generated/grade_school_mult_v1.c"
  #include "./generated/karatsuba_mult_v1.c"

  #include "./generated/grade_school_red_v1.c"
  // #include "./generated/karatsuba_red_v1.c"
  #include "./generated/resolve.c"

  modmul++;
}

void fieldSqr(Field* r, Field* a) {
  uint64_t p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13,
           p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24;
  uint64_t q, mask=0x3FFFFFFFu;

  p0=0; p1=0; p2=0; p3=0; p4=0; p5=0; p6=0; p7=0; p8=0; p9=0; p10=0; p11=0; p12=0; 
  p13=0; p14=0; p15=0; p16=0; p17=0; p18=0; p19=0; p20=0; p21=0; p22=0; p23=0; p24=0;

  #include "./generated/fast_square_mult_v1.c"
  #include "./generated/grade_school_red_v1.c"
  // #include "./generated/karatsuba_red_v1.c"
  #include "./generated/no_resolve.c"

  modsqr++;
}

void fieldSqrResolve(Field* r, Field* a) {
  uint64_t p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13,
           p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24;
  uint64_t q, mask=0x3FFFFFFFu;

  p0=0; p1=0; p2=0; p3=0; p4=0; p5=0; p6=0; p7=0; p8=0; p9=0; p10=0; p11=0; p12=0; 
  p13=0; p14=0; p15=0; p16=0; p17=0; p18=0; p19=0; p20=0; p21=0; p22=0; p23=0; p24=0;

  #include "./generated/fast_square_mult_v1.c"
  #include "./generated/grade_school_red_v1.c"
  // #include "./generated/karatsuba_red_v1.c"
  #include "./generated/resolve.c"

  modsqr++;
}

void fieldMul64PartialReduce(Field* r, Field* f) {
  uint64_t estimate;
  Field    subtract;

  // in essence, this routine converts from Montgomery R=2^384 to Montgomery R=2^390

  // multiply by 8
  r->f0 = f->f0<<3;
  r->f1 = (f->f1<<3) + (r->f0>>30);
  r->f2 = (f->f2<<3) + (r->f1>>30);
  r->f3 = (f->f3<<3) + (r->f2>>30);
  r->f4 = (f->f4<<3) + (r->f3>>30);
  r->f5 = (f->f5<<3) + (r->f4>>30);
  r->f6 = (f->f6<<3) + (r->f5>>30);
  r->f7 = (f->f7<<3) + (r->f6>>30);
  r->f8 = (f->f8<<3) + (r->f7>>30);
  r->f9 = (f->f9<<3) + (r->f8>>30);
  r->f10 = (f->f10<<3) + (r->f9>>30);
  r->f11 = (f->f11<<3) + (r->f10>>30);
  r->f12 = (f->f12<<3) + (r->f11>>30);
  fieldMask(r);

  // estimate multiples of N
  estimate=((f->f12 << 8) + (f->f11 >> 22)) * 0x9D835D2Fu >> 60;
  fieldSet(&subtract, &fieldNMultiples[estimate]);

  // subtract and multiply result by 8
  r->f0 = (r->f0 - subtract.f0)<<3;
  r->f1 = ((r->f1 - subtract.f1)<<3) + (((int64_t)r->f0)>>30);
  r->f2 = ((r->f2 - subtract.f2)<<3) + (((int64_t)r->f1)>>30);
  r->f3 = ((r->f3 - subtract.f3)<<3) + (((int64_t)r->f2)>>30);
  r->f4 = ((r->f4 - subtract.f4)<<3) + (((int64_t)r->f3)>>30);
  r->f5 = ((r->f5 - subtract.f5)<<3) + (((int64_t)r->f4)>>30);
  r->f6 = ((r->f6 - subtract.f6)<<3) + (((int64_t)r->f5)>>30);
  r->f7 = ((r->f7 - subtract.f7)<<3) + (((int64_t)r->f6)>>30);
  r->f8 = ((r->f8 - subtract.f8)<<3) + (((int64_t)r->f7)>>30);
  r->f9 = ((r->f9 - subtract.f9)<<3) + (((int64_t)r->f8)>>30);
  r->f10 = ((r->f10 - subtract.f10)<<3) + (((int64_t)r->f9)>>30);
  r->f11 = ((r->f11 - subtract.f11)<<3) + (((int64_t)r->f10)>>30);
  r->f12 = ((r->f12 - subtract.f12)<<3) + (((int64_t)r->f11)>>30);
  fieldMask(r);

  // estimate and remove multiples of N
  estimate=((f->f12 << 8) + (f->f11 >> 22)) * 0x9D835D2Fu >> 60;
  fieldSubResolve(r, r, &fieldNMultiples[estimate]);
}

void fieldMul64FullReduce(Field* r, Field* f) {
  fieldMul64PartialReduce(r, f);
  if((r->f12 << 30) + r->f11 >= 0x680447A8E5FF9ull) {
    Field local;

    fieldSubResolve(&local, r, &N);
    if(((int64_t)local.f12)>=0) 
      fieldSet(r, &local);
  }
}

void fieldDivTwoExpResolve(Field* r, Field* a, int32_t exp) {
  uint64_t shifter[25];
  int32_t  words, bits;
  uint64_t p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13,
           p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24;
  uint64_t q, mask=0x3FFFFFFFu;

  for(int i=0;i<25;i++)
    shifter[i]=0;

  exp=390-exp;
  if(exp<0 || exp>390) 
    return;
  words=exp/30;
  bits=exp-words*30;
  if(bits==0 && words>0) {
    bits=30;
    words--;
  }

  shifter[0+words]=a->f0<<bits; shifter[1+words]=a->f1<<bits; shifter[2+words]=a->f2<<bits; shifter[3+words]=a->f3<<bits;
  shifter[4+words]=a->f4<<bits; shifter[5+words]=a->f5<<bits; shifter[6+words]=a->f6<<bits; shifter[7+words]=a->f7<<bits;
  shifter[8+words]=a->f8<<bits; shifter[9+words]=a->f9<<bits; shifter[10+words]=a->f10<<bits; shifter[11+words]=a->f11<<bits;
  shifter[12+words]=a->f12<<bits;

  p0=shifter[0]; p1=shifter[1]; p2=shifter[2]; p3=shifter[3]; p4=shifter[4]; p5=shifter[5]; p6=shifter[6]; p7=shifter[7];
  p8=shifter[8]; p9=shifter[9]; p10=shifter[10]; p11=shifter[11]; p12=shifter[12]; p13=shifter[13]; p14=shifter[14]; p15=shifter[15];
  p16=shifter[16]; p17=shifter[17]; p18=shifter[18]; p19=shifter[19]; p20=shifter[20]; p21=shifter[21]; p22=shifter[22]; p23=shifter[23];
  p24=shifter[24];

  #include "./generated/grade_school_red_v1.c"
  #include "./generated/resolve.c"
}

void fieldToMontgomeryFullReduce(Field* r, Field* f) {
  fieldMulResolve(r, f, &rSquaredTerm);
  if((r->f12 << 30) + r->f11 >= 0x680447A8E5FF9ull) {
    Field local;

    fieldSubResolve(&local, r, &N);
    if(((int64_t)local.f12)>=0) 
      fieldSet(r, &local);
  }
}

void fieldFromMontgomeryFullReduce(Field* r, Field* f) {
  fieldMulResolve(r, f, &oneTerm);
  if((r->f12 << 30) + r->f11 >= 0x680447A8E5FF9ull) {
    Field local;

    fieldSubResolve(&local, r, &N);
    if(((int64_t)local.f12)>=0) 
      fieldSet(r, &local);
  }
}

#if defined(OLD_INVERSE)
void fieldInverseFullReduce(Field* r, Field* f) {
  PackedField A, B;
  PackedField T0={1, 0, 0, 0, 0, 0}, T1={0, 0, 0, 0, 0, 0};
  Field       local;

  fieldFromMontgomeryFullReduce(&local, f);
  packedFieldFromField(&A, &local);
  packedFieldSetN(&B);

  // this alg is sorta slow, but short to implement

  while(packedFieldLimbOR(&A)!=0) {
    if((A.f0 & 0x01)!=0) {
      if(packedFieldGreater(&B, &A)) {
        packedFieldSwap(&A, &B);
        packedFieldSwap(&T0, &T1);
      }
      packedFieldSub(&A, &A, &B);
      if(!packedFieldSub(&T0, &T0, &T1))
        packedFieldAddN(&T0);
    }
    packedFieldShiftRightOne(&A);
    if((T0.f0 & 0x01)!=0)
      packedFieldAddN(&T0);
    packedFieldShiftRightOne(&T0);
  }
  
  packedFieldToField(r, &T1);
  fieldToMontgomeryFullReduce(r, r);
}
#endif

void fieldDumpInternal(Field* f) {
  logHex64(f->f0);
  logString(" ");
  logHex64(f->f1);
  logString(" ");
  logHex64(f->f2);
  logString(" ");
  logHex64(f->f3);
  logString(" ");
  logHex64(f->f4);
  logString(" ");
  logHex64(f->f5);
  logString(" ");
  logHex64(f->f6);
  logString(" ");
  logHex64(f->f7);
  logString(" ");
  logHex64(f->f8);
  logString(" ");
  logHex64(f->f9);
  logString(" ");
  logHex64(f->f10);
  logString(" ");
  logHex64(f->f11);
  logString(" ");
  logHex64(f->f12);
  logString("\n");
}

void fieldDump(Field* f, bool convertFromMontgomery) {
  uint32_t packed[12];

  if(convertFromMontgomery) 
    fromMontgomery(packed, f);
  else
    fieldStore(packed, f);

  for(int i=11;i>=0;i--) 
    logHex(packed[i]);
  logString("\n");
}

void getField(uint32_t* packed, Field* f, bool convertFromMontgomery) {
  if(convertFromMontgomery) 
    fromMontgomery(packed, f);
  else
    fieldStore(packed, f);
}

void fromMontgomery(uint32_t* r, Field *f) {
  Field local;

  fieldFromMontgomeryFullReduce( &local, f);
  fieldStore(r, &local);
}

