/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Author(s):  Niall Emmart
            Sougata Bhattacharya
            Antony Suresh

***/

void cmGroupReduce(CMState* state) {
  AccumulatorXY sosAcc, sumAcc;
  PointXY       bucket, sum;
  uint32_t      k=(state->windows << state->windowBits)>>6;
  uint32_t*     buckets=cmBuckets(state);
  uint32_t*     sosScratch=cmScratchPoints(state);
  uint32_t*     sumScratch=cmScratchPoints(state) + k*24;
  uint32_t*     inverses=cmInverses(state);

  // initialize sos to 0 and sum to bucket j*64 + 63
  for(int j=0;j<k;j++) {
    sosScratch[j*24 + 11]=0x80000000;
    for(int i=0;i<24;i++) 
      sumScratch[j*24 + i]=buckets[j*1536 + 1512 + i];   // last bucket in each group
  }

  // batched invert algorithm
  for(int i=62;i>=0;i--) {
    initializeFieldState(&state->inverseState); 
    for(int j=0;j<k;j++) {  
      loadXY(&sosAcc.xy, sosScratch + j*24);
      loadXY(&sumAcc.xy, sumScratch + j*24);
      loadXY(&bucket, buckets + j*1536 + i*24);
      fieldPartialReduce(&sumAcc.xy.y);
      fieldPartialReduce(&bucket.y);

      addXYPhaseOne(&state->inverseState, &sumAcc, &bucket, inverses + j*24);
      addXYPhaseOne(&state->inverseState, &sosAcc, &sumAcc.xy, inverses + j*24 + 12);
    }
    fieldInverseFullReduce(&state->inverseState, &state->inverseState);
    for(int j=k-1;j>=0;j--) {  
      loadXY(&sosAcc.xy, sosScratch + j*24);
      loadXY(&sumAcc.xy, sumScratch + j*24);
      loadXY(&bucket, buckets + j*1536 + i*24);
      fieldPartialReduce(&sumAcc.xy.y);
      fieldPartialReduce(&bucket.y);

      addXYPhaseTwo(&state->inverseState, &sosAcc, &sumAcc.xy, inverses + j*24 + 12);
      addXYPhaseTwo(&state->inverseState, &sumAcc, &bucket, inverses + j*24);

      storeXY(sosScratch + j*24, &sosAcc.xy);
      storeXY(sumScratch + j*24, &sumAcc.xy);
    }
  }

  // need to add sum to sos one last time
  initializeFieldState(&state->inverseState); 
  for(int j=0;j<k;j++) {  
    loadXY(&sosAcc.xy, sosScratch + j*24);
    loadXY(&sum, sumScratch + j*24);
    fieldPartialReduce(&sum.y);
    addXYPhaseOne(&state->inverseState, &sosAcc, &sum, inverses + j*12);
  }
  fieldInverseFullReduce(&state->inverseState, &state->inverseState);
  for(int j=k-1;j>=0;j--) {  
    loadXY(&sosAcc.xy, sosScratch + j*24);
    loadXY(&sum, sumScratch + j*24);
    fieldPartialReduce(&sum.y);
    addXYPhaseTwo(&state->inverseState, &sosAcc, &sum, inverses + j*12);
    storeXY(sosScratch + j*24, &sosAcc.xy);
  }
}

void cmWindowReduce(CMState* state, AccumulatorXYZZ* sumOfSums, uint32_t window) {
  AccumulatorXYZZ sum; 
  uint32_t        groupCount=1<<(state->windowBits-6), windows=state->windows;
  uint32_t*       groupSOSBase=cmScratchPoints(state) + window*groupCount*24;
  uint32_t*       groupSumBase=cmScratchPoints(state) + (state->windows+window)*groupCount*24;
  PointXY         point;
  PointXYZZ       pointXYZZ;

  // Compute the bucket reduction for a window

  initializeAccumulatorXYZZ(&sum);
  initializeAccumulatorXYZZ(sumOfSums);

  // window reduction = 64 * sum of sums{group sum} + sum{group sos}

  getAccumulator(&pointXYZZ, &sum);
  for(int i=groupCount-1;i>=1;i--) {
    loadXY(&point, groupSumBase + i*24);
    if((point.x.f12 & 0x00800000)==0) {
      fieldFullReduce(&point.x);
      fieldFullReduce(&point.y);
      addXY(&sum, &point);
      getAccumulator(&pointXYZZ, &sum);
    }
    addXYZZ(sumOfSums, &pointXYZZ);
  }

  // multiply by 64
  for(int i=0;i<6;i++)
    doubleAccumulatorXYZZ(sumOfSums, sumOfSums);

  // add SOS terms
  for(int i=groupCount-1;i>=0;i--) {
    loadXY(&point, groupSOSBase + i*24);
    if((point.x.f12 & 0x00800000)==0) {
      fieldFullReduce(&point.x);
      fieldFullReduce(&point.y);
      addXY(sumOfSums, &point);
    }
  }
}

uint32_t* cmMSMReduce(CMState* state) {
  uint32_t        windowBits=state->windowBits, windows=state->windows;
  AccumulatorXYZZ msm;
  AccumulatorXYZZ sumOfSums;
  PointXYZZ       pointXYZZ;
  PointXY         result;

  cmGroupReduce(state);

  initializeAccumulatorXYZZ(&msm);
  for(int window=windows-1;window>=0;window--) {
    if(window!=windows-1)
      for(int i=0;i<=windowBits;i++)
        doubleAccumulatorXYZZ(&msm, &msm);
    cmWindowReduce(state, &sumOfSums, window);
    getAccumulator(&pointXYZZ, &sumOfSums);
    addXYZZ(&msm, &pointXYZZ);
  }
  getAccumulator(&pointXYZZ, &msm);
  normalizeXYZZ(&result, &pointXYZZ);

  #if defined(WASM_BUILD)
    fromMontgomery(cmResults(state), &result.x);
    fromMontgomery(cmResults(state)+12, &result.y);
  #endif

  #if defined(C_BUILD)
    storeXY(cmResults(state), &result);
  #endif

  return cmResults(state);
}
