/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Written by Niall Emmart.

***/

#include <stdio.h>
#include <stdint.h>

#include "MSM.h"

#include "asm.cu"
#include "Support.cu"
#include "Chain.cu"
#include "MP.cu"
#include "Curve.cu"
#include "SHM.cu"

#include "PrecomputePoints.cu"
#include "ProcessSignedDigits.cu"
#include "InitializeCountersAndSizes.cu"
#include "Partition1024.cu"
#include "Partition4096.cu"
#include "SortCounts.cu"

#include "ComputeBucketSums.cu"
#include "ReduceBuckets.cu"

#include "HostCurve.cpp"
#include "HostReduce.cpp"

#define ROUND128(x) (x + 127 & 0xFFFFFF80)
#define ROUND256(x) (x + 255 & 0xFFFFFF00)

#define CUDA_CHECK(call) if((errorState=call)!=0) { cudaError("Call \"" #call "\" failed.", __FILE__, __LINE__); return errorState; }

uint32_t MAX128(uint32_t a, uint32_t b) {
  return ROUND128(a>=b ? a : b);
}

uint32_t MAX128(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t max;
  
  max=(a>=b ? a : b);
  max=(max>=c ? max : c);
  return ROUND128(max);
}

void* advanceBytes(void*& current, uint32_t bytes) {
  uint8_t* prior=(uint8_t*)current;
  
  current=(void*)(prior + bytes); 
  return (void*)prior;
}

void* advanceScalars(void* scalars, uint32_t count) {
  uint8_t* ptr8=(uint8_t*)scalars;
  
  return (void*)(ptr8 + ((uint64_t)32)*((uint64_t)count));
}

void* advanceFields(void* results, uint32_t fieldCount) {
  uint8_t* ptr8=(uint8_t*)results;
  
  return (void*)(ptr8 + fieldCount*48);
}

#if defined(SUPPORT_READING)

#include "Reader.cpp"

// C interface implementations
int32_t MSMReadHexPoints(uint8_t* pointsPtr, uint32_t count, const char* path) {
  FILE* f=fopen(path, "r");
  
  // absolutely horrible, but I don't know how to load points in Rust
  
  if(f==NULL) {
    fprintf(stderr, "Failed to open path '%s' for reading\n", path);
    return -1;
  }
  
  for(uint32_t i=0;i<count;i++) {
    for(uint32_t j=0;j<2;j++) {
      if(!parseHex(pointsPtr + j*48, f, 48)) {
        fprintf(stderr, "Points file parse failed\n");
        return -1;
      }
    }
    for(uint32_t j=0;j<8;j++)
      pointsPtr[j + 96]=0;     
    pointsPtr+=104;
  }
  
  fclose(f);
  return 0;
}

int32_t MSMReadHexScalars(uint8_t* scalarsPtr, uint32_t count, const char* path) {
  FILE* f=fopen(path, "r");
  
  // absolutely horrible, but I don't know how to load points in Rust
  
  if(f==NULL) {
    fprintf(stderr, "Failed to open path '%s' for reading\n", path);
    return -1;
  }
  
  for(uint32_t i=0;i<count;i++) {
    if(!parseHex(scalarsPtr, f, 32)) {
      fprintf(stderr, "Scalar file parse failed\n");
      return -1;
    }
    scalarsPtr += 32;
  }
  
  fclose(f);
  return 0;
}

#endif

void* MSMAllocContext(int32_t maxPoints, int32_t maxBatches) {
  return (void*)(new MSMContext(maxPoints, maxBatches));
}

int32_t MSMFreeContext(void* context) {
  delete (MSMContext*)context;
  return 0;
}

int32_t MSMPreprocessPoints(void* context, void* affinePointsPtr, uint32_t points) {
  return ((MSMContext*)context)->msmPreprocessPoints(affinePointsPtr, points);
}

int32_t MSMRun(void* context, uint64_t* projectiveResultsPtr, void* scalarsPtr, uint32_t scalars) {
  return ((MSMContext*)context)->msmRun(projectiveResultsPtr, scalarsPtr, scalars);
}

MSMContext::MSMContext(uint32_t _maxPoints, uint32_t _maxBatches) {
  maxPoints=ROUND256(_maxPoints);
  maxBatches=_maxBatches;
  smCount=0;
  errorState=0;
  gpuPlanningMemory=NULL;
  gpuPointsMemory=NULL;
  cpuReduceResults=NULL;
}

MSMContext::~MSMContext() {
  // FIX FIX FIX!  Need to clean up streams and events
  
  if(gpuPlanningMemory!=NULL)
    cudaFree(gpuPlanningMemory);
  if(gpuPointsMemory!=NULL)
    cudaFree(gpuPointsMemory);
  if(cpuReduceResults!=NULL)
    cudaFreeHost(cpuReduceResults);
}

void MSMContext::cudaError(const char* call, const char* file, uint32_t line) {
  fprintf(stderr, "CUDA Error %d occurred on \"%s\", in %s:%d\n", errorState, call, file, line);
}

size_t MSMContext::memoryLayoutSize() {
  size_t    totalBytes;
  uint32_t  counters, pointsPerPage, pageCount, sizeCount;
  uint32_t  overlay1a, overlay1b, overlay2a, overlay3a, overlay3b, overlay3c;

  pointsPerPage=(PAGE_SIZE-4)/5;
  counters=11*1024 + 128;
  sizeCount=11*1024;
  pageCount=(maxPoints*11 + pointsPerPage - 1)/pointsPerPage + 11*1024; 
     
  // Processing performed:
  //   scalars -> processedScalars            
  //   processedScalars -> pages              
  //   pages -> points + unsortedTriple (uses scratch)  
  //   unsortedTriple -> sortedTriple
  ///  points + sortedTriple -> buckets
   
  //  Overlay 1:   scalars / pages
  //  Overlay 2:   points + unsorted triple
  //  Overlay 3:   processedScalars / scratch / sortedTriple 
  //  Overlay 4:   buckets
  //  Overlay 5:   misc
  
  // hopefully none of these exceed 2^32
  overlay1a=maxPoints*32;
  overlay1b=pageCount*PAGE_SIZE;
  ml.overlay1=MAX128(overlay1a, overlay1b);
 
  overlay2a=maxPoints*44;
  overlay2a+=NBUCKETS*(11+11)*4;
  ml.overlay2=overlay2a;
  
  overlay3a = maxPoints*33;
  overlay3b = smCount*ROUND128(SCRATCH_REQUIRED);
  overlay3c = NBUCKETS*(2+12+12)*4 + 32*(1+6+6)*4;
  ml.overlay3 = MAX128(overlay3a, overlay3b, overlay3c);
    
  ml.overlay4 = (NBUCKETS+NBUCKETS+32)*192;
  
  ml.overlay5 = ROUND128(128*4 + counters*8 + sizeCount*4 + sizeCount*4 + 1024*4 + smCount*8*3*192*maxBatches);
  
  totalBytes=ml.overlay1; 
  totalBytes+=ml.overlay2;
  totalBytes+=ml.overlay3;
  totalBytes+=ml.overlay4;
  totalBytes+=ml.overlay5;

  if(totalBytes<104ull * maxPoints)
    totalBytes=104ull * maxPoints;
  return totalBytes;
}

int32_t MSMContext::initializeMemoryLayout() {
  uint32_t  counters, pointsPerPage, pageCount, sizeCount;
  void*     overlay=gpuPlanningMemory;
  void*     current;
  
  pointsPerPage=(PAGE_SIZE-4)/5;
  counters=11*1024 + 128;
  sizeCount=11*1024;
  pageCount=(maxPoints*11 + pointsPerPage - 1)/pointsPerPage + 11*1024; 

  // Processing performed:
  //   scalars -> processedScalars            
  //   processedScalars -> pages              
  //   pages -> points + unsortedTriple (uses scratch)  
  //   unsortedTriple + points -> sortedTriple
  ///  points + sortedTriple -> buckets
   
  //  Overlay 1:   scalars / pages
  //  Overlay 2:   points + unsorted triple
  //  Overlay 3:   processedScalars / scratch / sortedTriple
  //  Overlay 4:   buckets
  //  Overlay 5:   misc

  // OVERLAY 1
    current=overlay;
    ml.scalars=advanceBytes(current, maxPoints*32);

    current=overlay;
    ml.pages=advanceBytes(current, pageCount*PAGE_SIZE);
 
    advanceBytes(overlay, ml.overlay1);
  
  // OVERLAY 2
    current=overlay;
    ml.points=advanceBytes(current, maxPoints*44);
    ml.unsortedTriple=advanceBytes(current, NBUCKETS*(11+11)*4);
    
    advanceBytes(overlay, ml.overlay2);
  
  // OVERLAY 3
    current=overlay;
    ml.processedScalars=advanceBytes(current, maxPoints*33);
  
    current=overlay;
    ml.scratch=advanceBytes(current, smCount*ROUND128(SCRATCH_REQUIRED));
     
    current=overlay;
    ml.sortedTriple=advanceBytes(current, NBUCKETS*(2+12+12)*4 + 32*(1+6+6)*4);

    advanceBytes(overlay, ml.overlay3);

  // OVERLAY 4 
    current=overlay;
    ml.buckets=advanceBytes(current, (NBUCKETS+NBUCKETS+32)*192);
    
    advanceBytes(overlay, ml.overlay4);
    
  // OVERLAY 5 
    current=overlay;
    ml.atomics=advanceBytes(current, 128*4);
    ml.counters=advanceBytes(current, counters*8);
    ml.sizes=advanceBytes(current, sizeCount*4);
    ml.prefixSumSizes=advanceBytes(current, sizeCount*4);
    ml.histogram=advanceBytes(current, 1024*4);
    ml.results=advanceBytes(current, smCount*8*3*192*maxBatches);
  
    advanceBytes(overlay, ml.overlay5);

  return 0;  
}
  
int32_t MSMContext::initializeGPU() {
  cudaDeviceProp properties;
  
  if(errorState!=0)
    return errorState;
  
  if(smCount!=0) {
    // we're already initialized
    return 0;
  }

  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

  smCount=properties.multiProcessorCount;
  
  CUDA_CHECK(cudaFuncSetAttribute(partition1024Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 64*1024));
  CUDA_CHECK(cudaFuncSetAttribute(partition4096Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 64*1024));
  CUDA_CHECK(cudaFuncSetAttribute(sortCountsKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 96*1024));
  CUDA_CHECK(cudaFuncSetAttribute(computeBucketSums, cudaFuncAttributeMaxDynamicSharedMemorySize, 96*1024));
  
  CUDA_CHECK(cudaStreamCreate(&runStream));
  CUDA_CHECK(cudaStreamCreate(&memoryStream));
  
  CUDA_CHECK(cudaEventCreate(&planningComplete));
  CUDA_CHECK(cudaEventCreate(&lastRoundPlanningComplete));
  CUDA_CHECK(cudaEventCreate(&writeComplete));
  
  CUDA_CHECK(cudaEventCreate(&timer0));
  CUDA_CHECK(cudaEventCreate(&timer1));
  CUDA_CHECK(cudaEventCreate(&timer2));
  CUDA_CHECK(cudaEventCreate(&timer3));
  CUDA_CHECK(cudaEventCreate(&timer4));
    
  CUDA_CHECK(cudaMalloc(&gpuPlanningMemory, memoryLayoutSize()));

#if defined(SMALL)
  CUDA_CHECK(cudaMalloc(&gpuPointsMemory, 96ull * 6 * 65536));
#else
  CUDA_CHECK(cudaMalloc(&gpuPointsMemory, 96ull * 6 * maxPoints));
#endif

  if(initializeMemoryLayout()!=0)
    return errorState;

  CUDA_CHECK(cudaMallocHost((void**)&cpuReduceResults, maxBatches*smCount*8*3*192));
  return 0;
}

void MSMContext::hostReduce(uint64_t* projectiveResultsPtr, uint32_t batch) {
  typedef Host::BLS12377::G1Montgomery Field;
  typedef Host::HostReduce<Field>      HostReduce;

  HostReduce hostReduce(2, 23, smCount*8);

  hostReduce.reduce(projectiveResultsPtr + batch*6*3, (uint32_t*)advanceFields(cpuReduceResults, batch*smCount*8*3*4));   
}

int32_t MSMContext::msmPreprocessPoints(void* affinePointsPtr, uint32_t points) {
  uint32_t basePoints;
  
  if(errorState!=0)
    return errorState;

  if(initializeGPU()<0)
    return errorState;

  if(points>maxPoints) {
    fprintf(stderr, "Point count exceeded max points\n");
    return -1;
  }
  
  if(points%65536!=0) {
    fprintf(stderr, "Point count must be evenly divisible by 65536\n");
    return -1;
  }
  
  #if defined(SMALL)
    basePoints=65536;
  #else
    basePoints=points;  
  #endif
  
  // This kernel pre-computes the following for each input point, Pi:
  //   2^46*Pi, 2^92*Pi, 2^138*Pi, 2^184*Pi, 2^230*Pi, 2^276*Pi
  // These precomputed points let us reduce the computation from 11 window @ 23 bits per window
  // down to 2 windows.  This pre-computation is part of the setup routine and is not timed.
  
  CUDA_CHECK(cudaMemcpy(gpuPlanningMemory, affinePointsPtr, ((uint64_t)104)*((uint64_t)basePoints), cudaMemcpyHostToDevice));
  precomputePointsKernel<<<smCount, 256, 1536>>>(gpuPointsMemory, gpuPlanningMemory, basePoints);
  CUDA_CHECK(cudaDeviceSynchronize());

  preprocessedPoints=points;
  
  return 0;
}

int32_t MSMContext::msmRun(uint64_t* projectiveResultsPtr, void* scalarsPtr,  uint32_t scalars) {
  uint32_t  points=preprocessedPoints, batches=scalars/points;
  void*     partition1024Args[5]={&ml.pages, &ml.sizes, &ml.counters, &ml.processedScalars, &points};
  void*     sizesPrefixSumArgs[5]={&ml.pages, &ml.prefixSumSizes, &ml.sizes, &ml.counters, &ml.atomics};
  void*     nextScalarsPtr=scalarsPtr;
  void*     nextResultsPtr=ml.results;
  
  if(errorState!=0)
    return errorState;
    
  if(scalars%points!=0) {
    fprintf(stderr, "Scalar count must be a multiply of point count\n");
    return -1;
  }
  
  if(batches>maxBatches) {
    fprintf(stderr, "Batch count exceed max batches\n");
    return -1;
  }
  
  if(preprocessedPoints!=points) {
    fprintf(stderr, "Points count does not match preprocessed points\n");
    return -1;
  }
  
  // The initial copy time of 2^26 scalars is pretty slow.
  
  CUDA_CHECK(cudaEventRecord(timer0, runStream));  
  CUDA_CHECK(cudaMemcpy(ml.scalars, nextScalarsPtr, points*32u, cudaMemcpyHostToDevice));
  nextScalarsPtr=advanceScalars(nextScalarsPtr, points);
  CUDA_CHECK(cudaEventRecord(timer1, runStream));  

  for(uint32_t batch=1;batch<=batches;batch++) {    
    if(batch>0)
      CUDA_CHECK(cudaStreamWaitEvent(runStream, writeComplete));

    // This kernel breaks each scalar value into 11 windows, and does signed-digit processing.  The window value
    // has a sign bit and a 22-bit scalar value (the Pippenger bucket).  Note, 23 evenly divides 253, so use a 
    // small trick -- if the high bit of the scalar is set, we negate the point and change the scalar to
    //    s'=m - s
    // where m is the order of the field.  The new scalar, s', will have the high bit clear.   This works since:
    //    s' (-Pi) = (m - s) (-Pi) = -s -Pi = s Pi.
    
    processSignedDigitsKernel<<<points/256, 256, 8928, runStream>>>(ml.processedScalars, ml.scalars, points);
 
    // These next 6 kernels are a replacement for CUB sorting.  Their implementation is involved, but conceptually 
    // what they do is quite simple:  they group all the points together that need to go into the same bucket.  What
    // comes is 11x 2^22 lists of points.  The buckets are then sorted such that buckets that will accumulate the most
    // points are processed first.
  
    initializeCountersSizesAtomicsHistogramKernel<<<smCount, 256, 0, runStream>>>(ml.counters, ml.sizes, ml.atomics, ml.histogram);
    CUDA_CHECK(cudaLaunchCooperativeKernel((const void*)partition1024Kernel, dim3(smCount), dim3(1024), partition1024Args, 64*1024, runStream));
    CUDA_CHECK(cudaLaunchCooperativeKernel((const void*)sizesPrefixSumKernel, dim3(11), dim3(1024), sizesPrefixSumArgs, 0, runStream));
    partition4096Kernel<<<smCount, 1024, 64*1024, runStream>>>(ml.points, ml.unsortedTriple, ml.scratch, ml.prefixSumSizes, ml.sizes, ml.pages, ml.atomics, points);
    histogramPrefixSumKernel<<<smCount, 1024, 0, runStream>>>(ml.histogram, ml.unsortedTriple);
    sortCountsKernel<<<smCount, 1024, 96*1024, runStream>>>(ml.sortedTriple, ml.histogram, ml.unsortedTriple);

    if(batch!=batches) {
      // DO NOT REMOVE BRACKETS
      CUDA_CHECK(cudaEventRecord(planningComplete, runStream));
    }
    else {
      // DO NOT REMOVE BRACKETS
      CUDA_CHECK(cudaEventRecord(lastRoundPlanningComplete, runStream));
    }

    // ComputeBucketSums processes the lists, and computes a bucket sum for each list.  The kernel works by assigning a
    // thread to each bucket, and uses an EC add routines, based on XYZZ representation.  Since the buckets have been sorted 
    // by the number of points in each bucket, it's almost always the case that all the threads in each warp and converged 
    // and we can take advantage of copying for the first point in a bucket and use the faster "zz=1/zzz=1" addition
    // for the second point in each bucket.  Finally, again because the buckets are converged, all threads in the
    // warp write their results to global memory at the same time.
    // ReduceBuckets launches smCount*8 warps.  Half the warps are used to reduce each window.  

    computeBucketSums<<<smCount, 384, 96*1024, runStream>>>(ml.buckets, gpuPointsMemory, ml.sortedTriple, ml.points, ml.atomics);
    reduceBuckets<<<smCount, 256, 256*96 + 1536, runStream>>>(nextResultsPtr, ml.buckets);
    nextResultsPtr=advanceFields(nextResultsPtr, smCount*8*3*4);
    
    if(batch!=batches) {  
      CUDA_CHECK(cudaStreamWaitEvent(memoryStream, planningComplete))
      if(batch==0) {
        CUDA_CHECK(cudaMemcpyAsync(ml.scalars, nextScalarsPtr, (points-points/4)*32u, cudaMemcpyHostToDevice, memoryStream));
        nextScalarsPtr=advanceScalars(nextScalarsPtr, points-points/4);
      }
      else {
        CUDA_CHECK(cudaMemcpyAsync(ml.scalars, nextScalarsPtr, points*32u, cudaMemcpyHostToDevice, memoryStream));
        nextScalarsPtr=advanceScalars(nextScalarsPtr, points);
      }
      CUDA_CHECK(cudaEventRecord(writeComplete, memoryStream));
    }
  }

  CUDA_CHECK(cudaEventSynchronize(lastRoundPlanningComplete));
  CUDA_CHECK(cudaMemcpyAsync(cpuReduceResults, ml.results, (batches-1)*smCount*8*3*192, cudaMemcpyDeviceToHost, memoryStream));
  CUDA_CHECK(cudaStreamSynchronize(memoryStream));

  // all but the last one
  for(uint32_t batch=0;batch<batches-1;batch++)
    hostReduce(projectiveResultsPtr, batch);
  
  CUDA_CHECK(cudaStreamSynchronize(runStream));
  CUDA_CHECK(cudaMemcpy(advanceFields(cpuReduceResults, (batches-1)*smCount*8*3*4),
                        advanceFields(ml.results, (batches-1)*smCount*8*3*4),
                        smCount*8*3*192, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(timer2, 0));

  // reduce the very last result
  hostReduce(projectiveResultsPtr, batches-1);

  float ms;
  cudaEventElapsedTime(&ms, timer0, timer1);
  printf("Initial copy: %f ms\n", ms);
  cudaEventElapsedTime(&ms, timer0, timer2);
  printf("Total time: %f ms\n", ms);
  
  return 0;
}
