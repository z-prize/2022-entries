/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Author(s):  Niall Emmart

***/

void scalarSubN(uint32_t* scalar) {
  int64_t carry=0;

  carry=carry - 0x00000001u + scalar[0];
  scalar[0]=carry;  carry=carry>>32;
  carry=carry - 0xFFFFFFFFu + scalar[1];
  scalar[1]=carry;  carry=carry>>32;
  carry=carry - 0xFFFE5BFEu + scalar[2];
  scalar[2]=carry;  carry=carry>>32;
  carry=carry - 0x53BDA402u + scalar[3];
  scalar[3]=carry;  carry=carry>>32;
  carry=carry - 0x09A1D805u + scalar[4];
  scalar[4]=carry;  carry=carry>>32;
  carry=carry - 0x3339D808u + scalar[5];
  scalar[5]=carry;  carry=carry>>32;
  carry=carry - 0x299D7D48u + scalar[6];
  scalar[6]=carry;  carry=carry>>32;
  carry=carry - 0x73EDA753u + scalar[7];
  scalar[7]=carry; 
}

void scalarNegateAddN(uint32_t* scalar) {
  int64_t carry=0;

  carry=carry + 0x00000001u - scalar[0];
  scalar[0]=carry;  carry=carry>>32;
  carry=carry + 0xFFFFFFFFu - scalar[1];
  scalar[1]=carry;  carry=carry>>32;
  carry=carry + 0xFFFE5BFEu - scalar[2];
  scalar[2]=carry;  carry=carry>>32;
  carry=carry + 0x53BDA402u - scalar[3];
  scalar[3]=carry;  carry=carry>>32;
  carry=carry + 0x09A1D805u - scalar[4];
  scalar[4]=carry;  carry=carry>>32;
  carry=carry + 0x3339D808u - scalar[5];
  scalar[5]=carry;  carry=carry>>32;
  carry=carry + 0x299D7D48u - scalar[6];
  scalar[6]=carry;  carry=carry>>32;
  carry=carry + 0x73EDA753u - scalar[7];
  scalar[7]=carry; 
}

void scalarNegate(uint32_t* scalar) {
  int64_t carry=0;

  carry=carry - scalar[0];
  scalar[0]=carry;  carry=carry>>32;
  carry=carry - scalar[1];
  scalar[1]=carry;  carry=carry>>32;
  carry=carry - scalar[2];
  scalar[2]=carry;  carry=carry>>32;
  carry=carry - scalar[3];
  scalar[3]=carry;  carry=carry>>32;
  carry=carry - scalar[4];
  scalar[4]=carry;  carry=carry>>32;
  carry=carry - scalar[5];
  scalar[5]=carry;  carry=carry>>32;
  carry=carry - scalar[6];
  scalar[6]=carry;  carry=carry>>32;
  carry=carry - scalar[7];
  scalar[7]=carry; 
}

void lambdaIncrement(uint32_t* halfScalar) {
  halfScalar[0]++;
  if(halfScalar[0]==0) {
    halfScalar[1]++;
    halfScalar[2]+=(halfScalar[1]==0);
    halfScalar[3]+=((halfScalar[1] | halfScalar[2])==0);
  }
}

void lambdaNegateAddLambda(uint32_t* halfScalar) {
  int64_t carry=0;

  carry=carry + 0xFFFFFFFFu - halfScalar[0];
  halfScalar[0]=carry;  carry=carry>>32;
  carry=carry + 0x00000000u - halfScalar[1];
  halfScalar[1]=carry;  carry=carry>>32;
  carry=carry + 0x0001A402u - halfScalar[2];
  halfScalar[2]=carry;  carry=carry>>32;
  carry=carry + 0xAC45A401u - halfScalar[3];
  halfScalar[3]=carry; 
}
void lambdaQR(uint32_t* q, uint32_t* r, uint32_t* scalar) {
  uint32_t d0=scalar[0], d1=scalar[1], d2=scalar[2], d3=scalar[3], d4=scalar[4], d5=scalar[5], d6=scalar[6], d7=scalar[7];
  uint32_t add, r0, r1, r2, r3, r4, r5, r6, r7, r8;
  uint64_t i0=0x36cfee30, i1=0x0fdb948b, i2=0x01faadd6, i3=0x1afb3c78, i4=0x0000017c;
  uint64_t l0=0x3fffffff, l1=0x00000003, l2=0x001a4020, l3=0x11690040, l4=0x000000ac;
  uint64_t pl0=0xffffffffu, pl1=0x00000000u, pl2=0x0001a402u, pl3=0xac45a401u;
  uint64_t h0, h1, h2, h3, h4;
  uint64_t p0, p1, p2, p3, p4;
  uint64_t t, mask30=0x3FFFFFFF, mask32=0xFFFFFFFFu;
  int64_t  carry;

  // uses Barrett's method to compute q & r

  // extract upper 128 bits from 255 bit scalar, multiply by the inverse of lambda, i0-i4
  h0=(d3>>31) + (d4<<1) & mask30;
  h1=(d4>>29) + (d5<<3) & mask30;
  h2=(d5>>27) + (d6<<5) & mask30;
  h3=(d6>>25) + (d7<<7) & mask30;
  h4=d7>>23;

  t=h0*i3;
  t+=h1*i2;
  t+=h2*i1;
  t+=h3*i0;

  p0=h0*i4;
  p0+=h1*i3;
  p0+=h2*i2;
  p0+=h3*i1;
  p0+=h4*i0;
  p0+=(t>>30);

  p1=h1*i4;
  p1+=h2*i3;
  p1+=h3*i2;
  p1+=h4*i1;
  p1+=(p0>>30);

  p2=h2*i4;
  p2+=h3*i3;
  p2+=h4*i2;
  p2+=(p1>>30);

  p3=h3*i4;
  p3+=h4*i3;
  p3+=(p2>>30);

  p4=h4*i4;
  p4+=(p3>>30);

  p0=p0 & mask30;
  p1=p1 & mask30;
  p2=p2 & mask30;
  p3=p3 & mask30;
  p4=p4 & mask30;

  // q estimate is in p0-p4

  // next shift p0-p4 by 129 bits (note, we only computed the top 5 words of the product, so this is just a shift by 9 bits)

  h0=(p0>>9) + (p1<<21) & mask30;
  h1=(p1>>9) + (p2<<21) & mask30;
  h2=(p2>>9) + (p3<<21) & mask30;
  h3=(p3>>9) + (p4<<21) & mask30;
  h4=p4>>9;

  // pack the q estimate up into 32 bit words

  r0=(uint32_t)(h0 + (h1<<30));
  r1=(uint32_t)((h1>>2) + (h2<<28));
  r2=(uint32_t)((h2>>4) + (h3<<26));
  r3=(uint32_t)((h3>>6) + (h4<<24));

  // compute q estimate * lambda

  p0=h0*l0;

  p1=h0*l1;
  p1+=h1*l0;
  p1+=(p0>>30);

  p2=h0*l2;
  p2+=h1*l1;
  p2+=h2*l0;
  p2+=(p1>>30);

  p3=h0*l3;
  p3+=h1*l2;
  p3+=h2*l1;
  p3+=h3*l0;
  p3+=(p2>>30);

  p4=h0*l4;
  p4+=h1*l3;
  p4+=h2*l2;
  p4+=h3*l1;
  p4+=h4*l0;
  p4+=(p3>>30);

  p0=p0 & mask30;
  p1=p1 & mask30;
  p2=p2 & mask30;
  p3=p3 & mask30;
  p4=p4 & mask30;

  // pack q*lambda into h0-h4

  h0=p0 + (p1<<30) & mask32;
  h1=(p1>>2) + (p2<<28) & mask32;
  h2=(p2>>4) + (p3<<26) & mask32;
  h3=(p3>>6) + (p4<<24) & mask32;
  h4=(p4>>8) & mask32;
 
  // compute scalar - q*lambda 

  carry=((uint64_t)d0)-h0;
  h0=carry & mask32;
  carry=carry>>32;
  carry+=((uint64_t)d1)-h1;
  h1=carry & mask32;
  carry=carry>>32;
  carry+=((uint64_t)d2)-h2;
  h2=carry & mask32;
  carry=carry>>32;
  carry+=((uint64_t)d3)-h3;
  h3=carry & mask32;
  carry=carry>>32;
  carry+=((uint64_t)d4)-h4;
  h4=carry & 0xFFFFu;

  // correction steps... could be as many as 3 (yes, 3 not 2)

  add=0;
  while(true) {
    carry=h0-pl0;
    p0=carry & mask32;
    carry=carry>>32;
    carry+=h1-pl1;
    p1=carry & mask32;
    carry=carry>>32;
    carry+=h2-pl2;
    p2=carry & mask32;
    carry=carry>>32;
    carry+=h3-pl3;
    p3=carry & mask32;
    carry=carry>>32;
    if(carry<0 && h4==0)  // went negative
      break;
    h4+=carry;
    add++;
    h0=p0; h1=p1; h2=p2; h3=p3;    
  }
  
  // if we ran any correction steps, update q and pack it away.

  r0+=add;
  if(r0<add) {
    r1++;
    r2+=(r1==0);
    r3+=((r1 | r2)==0);
  }
  r4=(uint32_t)h0;
  r5=(uint32_t)h1;
  r6=(uint32_t)h2;
  r7=(uint32_t)h3;

  // store everything out

  q[0]=r0; q[1]=r1; q[2]=r2; q[3]=r3;
  r[0]=r4; r[1]=r5; r[2]=r6; r[3]=r7;
}
