/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Author(s):  Niall Emmart
            Sougata Bhattacharya
            Antony Suresh

***/

static inline void cmSlice10(uint32_t* sliced, uint64_t* packed) {
  uint64_t mask=0x03FF, low=packed[0], high=packed[1];

  sliced[0]=low & mask;
  sliced[1]=(low>>10) & mask;
  sliced[2]=(low>>20) & mask;
  sliced[3]=(low>>30) & mask;
  sliced[4]=(low>>40) & mask;
  sliced[5]=(low>>50) & mask;
  sliced[6]=(low>>60) + (high<<4) & mask;
  sliced[7]=(high>>6) & mask;
  sliced[8]=(high>>16) & mask;
  sliced[9]=(high>>26) & mask;
  sliced[10]=(high>>36) & mask;
  sliced[11]=(high>>46) & mask;
  sliced[12]=high>>56;

  // sliced[12] must be positive
  for(int i=0;i<12;i++) {
    if(sliced[i]>0x200) {
      sliced[i+1]++;
      sliced[i]=(sliced[i]==0x400) ? 0 : (sliced[i] ^ 0x800003FF);
    }
  }
}

static inline void cmSlice11(uint32_t* sliced, uint64_t* packed) {
  uint64_t mask=0x07FF, low=packed[0], high=packed[1];

  sliced[0]=low & mask;
  sliced[1]=(low>>11) & mask;
  sliced[2]=(low>>22) & mask;
  sliced[3]=(low>>33) & mask;
  sliced[4]=(low>>44) & mask;
  sliced[5]=(low>>55) + (high<<9) & mask;
  sliced[6]=(high>>2) & mask;
  sliced[7]=(high>>13) & mask;
  sliced[8]=(high>>24) & mask;
  sliced[9]=(high>>35) & mask;
  sliced[10]=(high>>46) & mask;
  sliced[11]=high>>57;

  // sliced[11] must be positive
  for(int i=0;i<11;i++) {
    if(sliced[i]>0x400) {
      sliced[i+1]++;
      sliced[i]=(sliced[i]==0x800) ? 0 : (sliced[i] ^ 0x800007FF);
    }
  }
}

static inline void cmSlice12(uint32_t* sliced, uint64_t* packed) {
  uint64_t mask=0x0FFF, low=packed[0], high=packed[1];

  sliced[0]=low & mask;
  sliced[1]=(low>>12) & mask;
  sliced[2]=(low>>24) & mask;
  sliced[3]=(low>>36) & mask;
  sliced[4]=(low>>48) & mask;
  sliced[5]=(low>>60) + (high<<4) & mask;
  sliced[6]=(high>>8) & mask;
  sliced[7]=(high>>20) & mask;
  sliced[8]=(high>>32) & mask;
  sliced[9]=(high>>44) & mask;
  sliced[10]=high>>56;

  // sliced[10] must be positive
  for(int i=0;i<10;i++) {
    if(sliced[i]>0x800) {
      sliced[i+1]++;
      sliced[i]=(sliced[i]==0x1000) ? 0 : (sliced[i] ^ 0x80000FFF);
    }
  }
}

static inline void cmSlice13(uint32_t* sliced, uint64_t* packed) {
  uint64_t mask=0x1FFF, low=packed[0], high=packed[1];

  sliced[0]=low & mask;
  sliced[1]=(low>>13) & mask;
  sliced[2]=(low>>26) & mask;
  sliced[3]=(low>>39) & mask;
  sliced[4]=(low>>52) + (high<<12) & mask;
  sliced[5]=(high>>1) & mask;
  sliced[6]=(high>>14) & mask;
  sliced[7]=(high>>27) & mask;
  sliced[8]=(high>>40) & mask;
  sliced[9]=high>>53;

  // sliced[9] must be positive
  for(int i=0;i<9;i++) {
    if(sliced[i]>0x1000) {
      sliced[i+1]++;
      sliced[i]=(sliced[i]==0x2000) ? 0 : (sliced[i] ^ 0x80001FFF);
    }
  }
}

static inline void cmSlice15(uint32_t* sliced, uint64_t* packed) {
  uint64_t mask=0x7FFF, low=packed[0], high=packed[1];

  sliced[0]=low & mask;
  sliced[1]=(low>>15) & mask;
  sliced[2]=(low>>30) & mask;
  sliced[3]=(low>>45) & mask;
  sliced[4]=(low>>60) + (high<<4) & mask;
  sliced[5]=(high>>11) & mask;
  sliced[6]=(high>>26) & mask;
  sliced[7]=(high>>41) & mask;
  sliced[8]=high>>56;

  // sliced[8] must be positive
  for(int i=0;i<8;i++) {
    if(sliced[i]>0x4000) {
      sliced[i+1]++;
      sliced[i]=(sliced[i]==0x8000) ? 0 : (sliced[i] ^ 0x80007FFF);
    }
  }
}

static inline void cmSlice16(uint32_t* sliced, uint64_t* packed) {
  uint64_t mask=0xFFFF, low=packed[0], high=packed[1];

  sliced[0]=low & mask;
  sliced[1]=(low>>16) & mask;
  sliced[2]=(low>>32) & mask;
  sliced[3]=low>>48;
  sliced[4]=high & mask;
  sliced[5]=(high>>16) & mask;
  sliced[6]=(high>>32) & mask;
  sliced[7]=high>>48;

  // sliced[7] must be positive
  for(int i=0;i<7;i++) {
    if(sliced[i]>0x8000) {
      sliced[i+1]++;
      sliced[i]=(sliced[i]==0x10000) ? 0 : (sliced[i] ^ 0x8000FFFF);
    }
  }
}

void cmProcessSlices(CMState* state, PointXY* point, uint32_t* normalSlices, uint32_t* phiSlices, uint32_t count, uint32_t shift, bool negate) {
  if(negate)
    negateXY(point);

  cmUpdatePoint(state, point);
  for(int i=0;i<count;i++) 
    if(((int32_t)normalSlices[i])>0)
      cmAddPointToBucket(state, (i<<shift) + normalSlices[i] - 1, point);

  negateXY(point);

  cmUpdatePoint(state, point);
  for(int i=0;i<count;i++)
    if(((int32_t)normalSlices[i])<0)
      cmAddPointToBucket(state, (i<<shift) + (normalSlices[i] & 0xFFFF), point);

  // process "phi" points
  if(negate)
    negateXY(point);

  scaleByLambdaXY(point, point);

  cmUpdatePoint(state, point);
  for(int i=0;i<count;i++)
    if(((int32_t)phiSlices[i])<0) 
      cmAddPointToBucket(state, (i<<shift) + (phiSlices[i] & 0xFFFF), point);

  negateXY(point);

  cmUpdatePoint(state, point);
  for(int i=0;i<count;i++)
    if(((int32_t)phiSlices[i])>0)
      cmAddPointToBucket(state, (i<<shift) + phiSlices[i] - 1, point);
}

void cmPreprocessScalar(uint32_t* scalar) {
  if(scalar[7]>0x73EDA753) {
    scalarSubN(scalar);
    if(scalar[7]>0x73EDA753)
      scalarSubN(scalar);
  }
}

uint32_t* cmMSM10(CMRun* run) {
  CMState  state;
  uint64_t normal[2], phi[2];
  uint32_t normalSlices[13], phiSlices[13];
  PointXY  point;

  cmInitialize(&state, 10, run->maxBatch, run->maxCollisions);

  state.points=run->points;

  for(int i=0;i<run->pointCount;i++) {
    #if defined(WASM_BUILD)
    //  if(*(uint8_t*)(run->points + i*26 + 24)!=0)
    //    continue;
      loadXY(&point, run->points + i*24);
      fieldToMontgomeryFullReduce(&point.x, &point.x);
      fieldToMontgomeryFullReduce(&point.y, &point.y);
    #endif

    #if defined(C_BUILD)
      loadXY(&point, run->points + i*24);
    #endif

    // use endomorphism to break scalar into normal & phi
    // cmPreprocessScalar(run->scalars + i*8);
    lambdaQR((uint32_t*)phi, (uint32_t*)normal, run->scalars + i*8);
    cmSlice10(normalSlices, normal);
    cmSlice10(phiSlices, phi);
    cmProcessSlices(&state, &point, normalSlices, phiSlices, 13, 9, false);
  }

  cmCompleteProcessing(&state);
  return cmMSMReduce(&state);
}

uint32_t* cmMSM11(CMRun* run) {
  CMState  state;
  uint64_t normal[2], phi[2];
  uint32_t normalSlices[12], phiSlices[12];
  PointXY  point;

  cmInitialize(&state, 11, run->maxBatch, run->maxCollisions);

  state.points=run->points;

  for(int i=0;i<run->pointCount;i++) {
    #if defined(WASM_BUILD)
      //if(*(uint8_t*)(run->points + i*26 + 24)!=0)
      //  continue;
      loadXY(&point, run->points + i*24);
      fieldToMontgomeryFullReduce(&point.x, &point.x);
      fieldToMontgomeryFullReduce(&point.y, &point.y);
    #endif

    #if defined(C_BUILD)
      loadXY(&point, run->points + i*24);
    #endif

    // cmPreprocessScalar(run->scalars + i*8);
    lambdaQR((uint32_t*)phi, (uint32_t*)normal, run->scalars + i*8);
    cmSlice11(normalSlices, normal);
    cmSlice11(phiSlices, phi);
    cmProcessSlices(&state, &point, normalSlices, phiSlices, 12, 10, false);
  }

  cmCompleteProcessing(&state);
  return cmMSMReduce(&state);
}

uint32_t* cmMSM12(CMRun* run) {
  CMState  state;
  uint64_t normal[2], phi[2];
  uint32_t normalSlices[11], phiSlices[11];
  PointXY  point;

  cmInitialize(&state, 12, run->maxBatch, run->maxCollisions);

  state.points=run->points;

  for(int i=0;i<run->pointCount;i++) {
    #if defined(WASM_BUILD)
      //if(*(uint8_t*)(run->points + i*26 + 24)!=0)
      //  continue;
      loadXY(&point, run->points + i*24);
      fieldToMontgomeryFullReduce(&point.x, &point.x);
      fieldToMontgomeryFullReduce(&point.y, &point.y);
    #endif

    #if defined(C_BUILD)
      loadXY(&point, run->points + i*24);
    #endif

    // cmPreprocessScalar(run->scalars + i*8);
    lambdaQR((uint32_t*)phi, (uint32_t*)normal, run->scalars + i*8);
    cmSlice12(normalSlices, normal);
    cmSlice12(phiSlices, phi);
    cmProcessSlices(&state, &point, normalSlices, phiSlices, 11, 11, false);
  }

  cmCompleteProcessing(&state);
  return cmMSMReduce(&state);
}

uint32_t* cmMSM13(CMRun* run) {
  CMState  state;
  uint64_t normal[2], phi[2];
  uint32_t normalSlices[10], phiSlices[10];
  PointXY  point;

  cmInitialize(&state, 13, run->maxBatch, run->maxCollisions);

  state.points=run->points;

  for(int i=0;i<run->pointCount;i++) {
    #if defined(WASM_BUILD)
      //if(*(uint8_t*)(run->points + i*26 + 24)!=0)
      //  continue;
      loadXY(&point, run->points + i*24);
      fieldToMontgomeryFullReduce(&point.x, &point.x);
      fieldToMontgomeryFullReduce(&point.y, &point.y);
    #endif

    #if defined(C_BUILD)
      loadXY(&point, run->points + i*24);
    #endif

    // cmPreprocessScalar(run->scalars + i*8);
    lambdaQR((uint32_t*)phi, (uint32_t*)normal, run->scalars + i*8);
    cmSlice13(normalSlices, normal);
    cmSlice13(phiSlices, phi);
    cmProcessSlices(&state, &point, normalSlices, phiSlices, 10, 12, false);
  }

  cmCompleteProcessing(&state);
  return cmMSMReduce(&state);
}

uint32_t* cmMSM15(CMRun* run) {
  CMState  state;
  uint64_t normal[2], phi[2];
  uint32_t normalSlices[9], phiSlices[9];
  PointXY  point;

  cmInitialize(&state, 15, run->maxBatch, run->maxCollisions);

  state.points=run->points;

  for(int i=0;i<run->pointCount;i++) {
    #if defined(WASM_BUILD)
      //if(*(uint8_t*)(run->points + i*26 + 24)!=0)
      //  continue;
      loadXY(&point, run->points + i*24);
      fieldToMontgomeryFullReduce(&point.x, &point.x);
      fieldToMontgomeryFullReduce(&point.y, &point.y);
    #endif

    #if defined(C_BUILD)
      loadXY(&point, run->points + i*24);
    #endif

    // cmPreprocessScalar(run->scalars + i*8);
    lambdaQR((uint32_t*)phi, (uint32_t*)normal, run->scalars + i*8);
    cmSlice15(normalSlices, normal);
    cmSlice15(phiSlices, phi);
    cmProcessSlices(&state, &point, normalSlices, phiSlices, 9, 14, false);
  }

  cmCompleteProcessing(&state);
  return cmMSMReduce(&state);
}

uint32_t* cmMSM16(CMRun* run) {
  CMState  state;
  uint32_t localScalar[8];
  uint32_t normal[4], phi[4];
  uint32_t normalSlices[8], phiSlices[8];
  PointXY  point;
  bool     negate;

  cmInitialize(&state, 16, run->maxBatch, run->maxCollisions);

  state.points=run->points;

  for(int i=0;i<run->pointCount;i++) {
    #if defined(WASM_BUILD)
      //if(*(uint8_t*)(run->points + i*26 + 24)!=0)
      //  continue;
      loadXY(&point, run->points + i*24);
      fieldToMontgomeryFullReduce(&point.x, &point.x);
      fieldToMontgomeryFullReduce(&point.y, &point.y);
    #endif

    #if defined(C_BUILD)
      loadXY(&point, run->points + i*24);
    #endif

    for(int j=0;j<8;j++)
      localScalar[j]=run->scalars[i*8 + j];

    // cmPreprocessScalar(localScalar);
    if(localScalar[7]>=0x3FFFFFFF) {
      // scalar is too big -- use the relation, kP = (N-k)(-P), to shrink it
      scalarNegateAddN(localScalar);
      if(localScalar[7]>=0x80000000) {
        // haha, we had k>N.... someone is probably messing with us
        scalarNegate(localScalar);
      }
      else 
        negateXY(&point);
    }

    lambdaQR(phi, normal, localScalar);

    negate=false;
    if(normal[3]>=0x80000000) {
      lambdaIncrement(phi);
      lambdaNegateAddLambda(normal);
      negate=true;
    }

    cmSlice16(normalSlices, (uint64_t*)normal);
    cmSlice16(phiSlices, (uint64_t*)phi);
    cmProcessSlices(&state, &point, normalSlices, phiSlices, 8, 15, negate);
  }

  cmCompleteProcessing(&state);
  return cmMSMReduce(&state);
}

uint32_t* cmMSM(CMRun* run) {
  if(run->windowBits<=10)
    return cmMSM10(run);
  else if(run->windowBits==11)
    return cmMSM11(run);
  else if(run->windowBits==12)
    return cmMSM12(run);
  else if(run->windowBits==13 || run->windowBits==14)
    return cmMSM13(run);
  else if(run->windowBits==15)
    return cmMSM15(run);
  else 
    return cmMSM16(run);
}
