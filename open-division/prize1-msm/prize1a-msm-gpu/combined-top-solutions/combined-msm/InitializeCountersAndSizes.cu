/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Written by Niall Emmart.

***/

#include <cooperative_groups.h>

__global__ void initializeCountersSizesAtomicsHistogramKernel(void* countersPtr, void* sizesPtr, void* atomicsPtr, void* histogramPtr) {
  uint32_t  globalTID=blockIdx.x*blockDim.x+threadIdx.x, globalStride=blockDim.x*gridDim.x;
  uint64_t* counters=(uint64_t*)countersPtr;
  uint32_t* sizes=(uint32_t*)sizesPtr;
  uint32_t* atomics=(uint32_t*)atomicsPtr;
  uint32_t* histogram=(uint32_t*)histogramPtr;
  
  if(blockIdx.x==0 && threadIdx.x<128)
    atomics[threadIdx.x]=0;
    
  for(uint32_t i=globalTID;i<=11*1024;i+=globalStride) {
    if(i<11*1024)
      sizes[i]=0;
      
    if(i==0)
      counters[i]=make_wide(11*1024, 0);
    else
      counters[i]=make_wide(0, i-1);
  }
  
  for(uint32_t i=globalTID;i<1024;i+=globalStride)
    histogram[i]=0;
}

// 11*1024 points to prefix sum

__launch_bounds__(1024)
__global__ void sizesPrefixSumKernel(void* pagesPtr, void* prefixSumSizesPtr, void* sizesPtr, void* countersPtr, void* atomicsPtr) {
  int32_t   globalTID=blockIdx.x*blockDim.x + threadIdx.x;
  uint32_t* prefixSumSizes=(uint32_t*)prefixSumSizesPtr;
  uint32_t* sizes=(uint32_t*)sizesPtr;
  uint64_t* counters=(uint64_t*)countersPtr;
  
  uint64_t  page;
  void*     pageBase;
  
  uint32_t  pageCount, lastPageBytes, size, totalSize;
  
  __shared__ uint32_t warpTotals[32];
  __shared__ uint32_t blockMax;
  
  blockMax=0;
  
  __syncthreads();
  
  pageCount=sizes[globalTID];
  page=counters[globalTID + 1];
  lastPageBytes=ulow(page);
  size=(PAGE_SIZE-4)/5*pageCount + udiv5(lastPageBytes);
  
  // set the nextPage pointer to 0
  pageBase=byteOffset(pagesPtr, uhigh(page), PAGE_SIZE);
  *(uint32_t*)byteOffset(pageBase, PAGE_SIZE-4)=0;
  
  totalSize=multiwarpPrefixSum(warpTotals, size, 32);
  
  if(threadIdx.x==1023)
    prefixSumSizes[blockIdx.x]=totalSize;
  
  atomicMax(&blockMax, size);

  cooperative_groups::this_grid().sync();
  
  if(threadIdx.x<11)
    warpTotals[threadIdx.x]=prefixSumSizes[threadIdx.x];
  
  __syncthreads();

  #pragma unroll 1
  for(int32_t i=0;i<blockIdx.x;i++)
    totalSize+=warpTotals[i];
 
  sizes[globalTID]=size;
  prefixSumSizes[globalTID]=totalSize-size;
  
  if(threadIdx.x==0)
    atomicMax((uint32_t*)atomicsPtr, blockMax);
}
