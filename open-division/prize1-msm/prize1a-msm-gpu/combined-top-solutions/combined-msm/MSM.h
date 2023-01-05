/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Written by Niall Emmart.

***/

class MemoryLayout {
  public:
  uint32_t   overlay1, overlay2, overlay3, overlay4, overlay5;

  void*      scalars;
  void*      processedScalars;
  void*      pages;
  void*      points;
  void*      unsortedTriple;     // output from partition
  void*      sortedTriple;       // output from count sort
  void*      scratch;
  
  void*      buckets;
  void*      results;

  void*      atomics;
  void*      sizes;
  void*      prefixSumSizes;
  void*      counters;
  void*      histogram;
};

class MSMContext {
  public:
  MemoryLayout ml;
  int32_t      errorState;
  uint32_t     maxPoints;
  uint32_t     maxBatches;
  uint32_t     smCount;
  uint32_t     preprocessedPoints;
  void*        gpuPlanningMemory;
  void*        gpuPointsMemory;
  void*        cpuReduceResults;
  
  cudaStream_t runStream, memoryStream;
  cudaEvent_t  planningComplete, lastRoundPlanningComplete, writeComplete;

  cudaEvent_t  timer0, timer1, timer2, timer3, timer4;

  MSMContext(uint32_t _maxPoints, uint32_t _maxBatches);
  ~MSMContext();

  int32_t  msmPreprocessPoints(void* affinePointsPtr, uint32_t points);
  int32_t  msmRun(uint64_t* projectiveResultsPtr, void* scalarsPtr,  uint32_t scalars);

  private:
  void     cudaError(const char* call, const char* file, uint32_t line);
  uint64_t memoryLayoutSize();
  int32_t  initializeGPU();
  int32_t  initializeMemoryLayout();
  void     hostReduce(uint64_t* projectiveResultsPtr, uint32_t batch);
};

// C Interface used from Rust

extern "C" {

#if defined(SUPPORT_READING)
  int32_t MSMReadHexPoints(uint8_t* pointsPtr, uint32_t count, const char* path);
  int32_t MSMReadHexScalars(uint8_t* scalarsPtr, uint32_t count, const char* path);
#endif

  void* MSMAllocContext(int32_t maxPoints, int32_t maxBatches);
  int32_t MSMFreeContext(void* context);
  int32_t MSMPreprocessPoints(void* context, void* affinePointsPtr, uint32_t points);
  int32_t MSMRun(void* context, uint64_t* projectiveResultsPtr, void* scalarsPtr, uint32_t scalars);
}

// We always use a 23 bit window size
// NBUCKETS is 1<<(windowBits-1)

#define NBUCKETS  0x00400000u
#define PAGE_SIZE 31744           // a valid page size should be P=n*60+4, and P%128=0 some poss (16384, 24064, 31744)

#define SCRATCH_MAX_COUNT 9126
#define SIZE_LIMIT ((SCRATCH_MAX_COUNT-256)*32)
#define SCRATCH_REQUIRED (SCRATCH_MAX_COUNT*160)

