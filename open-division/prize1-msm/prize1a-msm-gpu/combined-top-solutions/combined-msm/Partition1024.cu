/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Written by Niall Emmart.

***/

#include <cooperative_groups.h>

__device__ __forceinline__ uint64_t atomicAdd(uint64_t* ptr, uint64_t amt) {
  return (uint64_t)atomicAdd((unsigned long long*)ptr, (unsigned long long)amt);
}

__device__ __forceinline__ uint64_t atomicExch(uint64_t* ptr, uint64_t amt) {
  return (uint64_t)atomicExch((unsigned long long*)ptr, (unsigned long long)amt);
}

__device__ __forceinline__ uint32_t nextPage(void* countersPtr) {
  return atomicAdd((uint32_t*)countersPtr, 1u);
}

__device__ __forceinline__ void initializeShared() {
  #pragma unroll 1
  for(int32_t localBin=threadIdx.x;localBin<1024;localBin+=blockDim.x) 
    store_shared_u32(localBin*4, 0);

  #pragma unroll 1
  for(int32_t i=threadIdx.x;i<15360;i+=blockDim.x)            // initialize the shared mem buffers
    store_shared_u32(4096 + i*4, 0xFFFFFFFF);  
}

__device__ __forceinline__ void shared_copy_bytes(void* global, uint32_t sAddr, uint32_t bytes) {
  uint32_t warpThread=threadIdx.x & 0x1F;
  
  if(warpThread<bytes)
    *(uint8_t*)byteOffset(global, warpThread)=load_shared_byte(sAddr + warpThread);
  if(warpThread+32<bytes)
    *(uint8_t*)byteOffset(global, warpThread + 32)=load_shared_byte(sAddr + warpThread + 32);
}
 
__device__ __forceinline__ void cleanup(void* pagesPtr, void* sizesPtr, void* countersPtr, uint32_t window, uint32_t localBin, uint32_t writeBytes) {
  uint32_t warpThread=threadIdx.x & 0x1F, globalBin=window*1024 + localBin;
  uint64_t page;
  uint32_t newPage=0, mask, thread, currentWriteBytes, binOffset=0;
  uint32_t shufflePageLow, shufflePageHigh, shuffleMemoryOffset, shuffleWriteBytes;
  void*    pageBase;
  
  currentWriteBytes=0;
  while(currentWriteBytes==0 && writeBytes>0) {
    page=atomicAdd((uint64_t*)byteOffset(countersPtr, globalBin*8 + 8), (uint64_t)writeBytes);
    if(ulow(page)<PAGE_SIZE-4 && ulow(page)+writeBytes>=PAGE_SIZE-4) {
      currentWriteBytes=PAGE_SIZE-ulow(page)-4;

      newPage=nextPage(countersPtr);
      atomicExch((uint64_t*)byteOffset(countersPtr, globalBin*8 + 8), make_wide(writeBytes-currentWriteBytes, newPage));
      atomicAdd((uint32_t*)byteOffset(sizesPtr, globalBin*4), 1);

      pageBase=byteOffset(pagesPtr, uhigh(page), PAGE_SIZE);
      *(uint32_t*)byteOffset(pageBase, PAGE_SIZE-4)=newPage;
      break;
    }
    else if(ulow(page)+writeBytes<PAGE_SIZE-4) {
      currentWriteBytes=writeBytes;
      break;
    }  
  }
      
  __syncwarp(0xFFFFFFFF);
    
  while(true) {
    mask=__ballot_sync(0xFFFFFFFF, currentWriteBytes>0);
    if(mask==0)
      return;
    thread=31-__clz(mask);
      
    shufflePageLow=__shfl_sync(0xFFFFFFFF, ulow(page), thread);
    shufflePageHigh=__shfl_sync(0xFFFFFFFF, uhigh(page), thread);
    shuffleMemoryOffset=__shfl_sync(0xFFFFFFFF, 4096 + localBin*60 + binOffset, thread);
    shuffleWriteBytes=__shfl_sync(0xFFFFFFFF, currentWriteBytes, thread);
  
    pageBase=byteOffset(pagesPtr, shufflePageHigh, PAGE_SIZE);
    shared_copy_bytes(byteOffset(pageBase, shufflePageLow), shuffleMemoryOffset, shuffleWriteBytes);
    if(warpThread==thread) {
      binOffset+=currentWriteBytes;
      writeBytes-=currentWriteBytes;
      currentWriteBytes=writeBytes;
      page=make_wide(0, newPage);
    }
  }
}

__device__ __forceinline__ void cleanup(void* pagesPtr, void* sizesPtr, void* countersPtr, uint32_t window) {
  #pragma unroll 1
  for(int32_t localBin=threadIdx.x;localBin<1024;localBin+=blockDim.x)
    cleanup(pagesPtr, sizesPtr, countersPtr, window, localBin, load_shared_u32(localBin*4));
}


__device__ __forceinline__ void processWrites(void* pagesPtr, void* sizesPtr, void* countersPtr, bool writeRequired, uint32_t globalBin, uint32_t highBitMask) {
  uint32_t warpThread=threadIdx.x & 0x1F, localBin=globalBin & 0x3FF;
  uint64_t page;
  uint32_t newPage=0, mask, thread, data, writeThreads;
  uint32_t shufflePageLow, shufflePageHigh, shuffleMemoryOffset; 
  void*    pageBase;
  
  if(writeRequired) {
    while(true) {
      page=atomicAdd((uint64_t*)byteOffset(countersPtr, globalBin*8 + 8), 60ull);
      if(ulow(page)==PAGE_SIZE-64) {
        newPage=nextPage(countersPtr);
        atomicExch((uint64_t*)byteOffset(countersPtr, globalBin*8 + 8), make_wide(0, newPage));
        atomicAdd((uint32_t*)byteOffset(sizesPtr, globalBin*4), 1);
        break;
      }
      else if(ulow(page)<PAGE_SIZE-64) 
        break;
    }
  }
  
  __syncwarp(0xFFFFFFFF);
  
  while(true) {
    mask=__ballot_sync(0xFFFFFFFF, writeRequired);
    if(mask==0)
      return;
    thread=31-__clz(mask);      
    
    shuffleMemoryOffset=__shfl_sync(0xFFFFFFFF, 4096 + localBin*60, thread);
    shufflePageLow=__shfl_sync(0xFFFFFFFF, ulow(page), thread);
    shufflePageHigh=__shfl_sync(0xFFFFFFFF, uhigh(page), thread);    
    data=__shfl_sync(0xFFFFFFFF, newPage, thread);

    if(warpThread<15) {
      while(true) {
        data=load_shared_u32(shuffleMemoryOffset + warpThread*4);
        if(__all_sync(0x00007FFF, (data & highBitMask)==0))
          break;        
      }
      data=shared_atomic_exch_u32(shuffleMemoryOffset + warpThread*4, 0xFFFFFFFF);
    }
    
    __syncwarp(0xFFFFFFFF);
    
    if(warpThread==thread) {
      shared_atomic_exch_u32(localBin*4, 0);
      writeRequired=false;
    }
    pageBase=byteOffset(pagesPtr, shufflePageHigh, PAGE_SIZE);
    writeThreads=(shufflePageLow<PAGE_SIZE-64 ? 15 : 16);
    if(warpThread<writeThreads)
      *(uint32_t*)byteOffset(pageBase, shufflePageLow + warpThread*4)=data;
  }
}

__launch_bounds__(1024)
__global__ void partition1024Kernel(void* pagesPtr, void* sizesPtr, void* countersPtr, void* processedScalarsPtr, uint32_t points) {
  uint32_t warpThread=threadIdx.x & 0x1F, chunk, window=0, priorWindow=0;
  uint32_t point, scalar, lowBits, middleBits, highBits, signBit, highBitMask;
  uint32_t writeLowBytes, writeHighByte, bin, sharedBase, offset, chunkSize=16384;
  bool     processed;
  void*    chunkBase;

  extern __shared__ uint32_t counts[];
  
//if(blockIdx.x==0 && threadIdx.x==0)
//  printf("Pointers: %lx %lx %lx %lx\n", (uint64_t)pagesPtr, (uint64_t)sizesPtr, (uint64_t)countersPtr, (uint64_t)processedScalarsPtr);
  
  // divide into 1024 bins per window

  highBitMask=warpThread%5;
  if(highBitMask>0)
    highBitMask=0x80<<highBitMask*8-8;

  initializeShared();
  
  __syncthreads();
  
  chunk=chunkSize*blockIdx.x;
  
  while(true) {
    if(window*points+points<=chunk) {
      __syncthreads();
      while(window<11 && window*points+points<=chunk) {
        window++;
        cooperative_groups::this_grid().sync();
      }
      cleanup(pagesPtr, sizesPtr, countersPtr, priorWindow);
      __syncthreads();
      initializeShared();
      priorWindow=window;
      __syncthreads();
    }
    if(window>=11)
      break;

    chunkBase=byteOffset(processedScalarsPtr, chunk, 3);
    for(int32_t i=threadIdx.x;i<chunkSize;i+=blockDim.x) {
      point=(chunk-window*points)+i;
      if(warpThread<24)
        scalar=*(uint32_t*)byteOffset(chunkBase, (i>>5)*96 + warpThread*4);
      scalar=uncompress(scalar);

      // scalar consists of:  sign bit, plus 23 bits with 0 through 2^22 (inclusive)
      // if scalar is zero, sign bit is always 0.

      processed=(scalar==0);   
      scalar--;     
      
      lowBits=scalar & 0x1F;            // low 5 bits
      middleBits=(scalar>>5) & 0x7F;    // middle 7 bits
      highBits=(scalar>>12) & 0x3FF;    // upper 10 bits    -- used to pick page
      signBit=(scalar>>23) & 0x01;      // sign bit         -- stored
      
      writeLowBytes=(lowBits<<27) + (signBit<<26) + point;
      writeHighByte=middleBits;

      bin=window*1024 + highBits;
      sharedBase=highBits*60 + 4096;
      
      // repeat until all threads have written their 5 bytes
      while(!__all_sync(0xFFFFFFFF, processed)) {
        offset=0;
        if(!processed) {
          offset=atomicAdd(&counts[highBits], 5);
          if(offset<=55) {
            // hopefully, the compiler is not allowed to reorder these
            store_shared_byte(sharedBase + offset + 0, writeLowBytes);
            store_shared_byte(sharedBase + offset + 1, writeLowBytes>>8);
            store_shared_byte(sharedBase + offset + 2, writeLowBytes>>16);
            store_shared_byte(sharedBase + offset + 3, writeLowBytes>>24);
            store_shared_byte(sharedBase + offset + 4, writeHighByte);        // important, write the high byte last

            processed=true;
          }
        }
        processWrites(pagesPtr, sizesPtr, countersPtr, offset==55, bin, highBitMask);
      }        
    }
    chunk+=chunkSize*gridDim.x;
  }
}
