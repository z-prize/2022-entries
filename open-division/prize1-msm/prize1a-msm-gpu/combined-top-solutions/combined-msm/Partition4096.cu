/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Written by Niall Emmart.

***/


// Defined in MSM.h
//   #define SCRATCH_MAX_COUNT 9126
//   #define SIZE_LIMIT ((SCRATCH_MAX_COUNT-256)*32)
//   #define SCRATCH_REQUIRED (SCRATCH_MAX_COUNT*160)

class ProcessFinalBinsOffsets {
  public:
  static const uint64_t countersOffset=0;                // 8 entries of 4 bytes
  static const uint32_t bufferCountsOffset=32;           // 256 entries of 4 bytes
  static const uint32_t pointCountsOffset=1056;          // 4096 entries of 4 bytes
  static const uint32_t mapOffset=17440;                 // 9216 entries of 1 byte
  static const uint32_t buffersOffset=26656;             // 256 entries of 128 bytes
  
  // these all overlap buffers
  static const uint32_t prefixCountsOffset=26656;        // 256 entries of 4 bytes   
  static const uint32_t warpSumsOffset=27680;            // 8 entries of 4 bytes
  static const uint32_t sortedMapOffset=27712;           // 9216 entries of 4 bytes  
};

__device__ __forceinline__ uint32_t round128(uint32_t x) {
  return x+127 & 0xFFFFFF80;
}

__device__ __forceinline__ void read640(uint32_t* data, void* pagesPtr, void* pageBase, uint32_t offset) {
  uint32_t warpThread=threadIdx.x & 0x1F;
  uint32_t nextPage, subtract;
  
  if(offset+640<=PAGE_SIZE-4) {
    data[0]=*(uint32_t*)byteOffset(pageBase, offset+warpThread*4+0);
    data[1]=*(uint32_t*)byteOffset(pageBase, offset+warpThread*4+128);
    data[2]=*(uint32_t*)byteOffset(pageBase, offset+warpThread*4+256);
    data[3]=*(uint32_t*)byteOffset(pageBase, offset+warpThread*4+384);
    data[4]=*(uint32_t*)byteOffset(pageBase, offset+warpThread*4+512);
  }
  else {
    subtract=0;
    nextPage=*(uint32_t*)byteOffset(pageBase, PAGE_SIZE-4);
    #pragma unroll
    for(int32_t i=0;i<5;i++) {
      if(offset+warpThread*4+i*128>=PAGE_SIZE-4) {
        pageBase=byteOffset(pagesPtr, nextPage, PAGE_SIZE);
        subtract=PAGE_SIZE-4;
      }
      data[i]=*(uint32_t*)byteOffset(pageBase, offset+warpThread*4+i*128-subtract);
    }
  }
}

__device__ __forceinline__ void unpackData(uint32_t* lowBits, uint32_t* highBits, uint32_t* data) {
  uint32_t quad=(threadIdx.x & 0x1F)>>2, quadThread=threadIdx.x & 0x03, shift=quadThread*8, src=quad*5+quadThread;
  uint32_t lo0, lo1, hi0, hi1, lo, hi;
  
  #pragma unroll
  for(int32_t i=0;i<4;i++) {
    lo0=__shfl_sync(0xFFFFFFFF, data[i], 8*i+src);
    lo1=__shfl_sync(0xFFFFFFFF, data[i+1], 8*i+src);
    hi0=__shfl_sync(0xFFFFFFFF, data[i], 8*i+src+1);
    hi1=__shfl_sync(0xFFFFFFFF, data[i+1], 8*i+src+1);
    lo=(8*i+src<32) ? lo0 : lo1;
    hi=(8*i+src<31) ? hi0 : hi1;
    lowBits[i]=__funnelshift_r(lo, hi, shift);                 // 27 bits of point, 5 bits of bucket
    highBits[i]=(hi>>shift) & 0x7F;                            // 7 bits of bucket
  }
}

__device__ __forceinline__ void shared_copy_u4(void* global, uint32_t sAddr, uint32_t count) {
  #pragma unroll 1
  for(uint32_t i=threadIdx.x;i<count;i+=blockDim.x)
    *(uint4*)byteOffset(global, i*16)=load_shared_u4(sAddr + i*16);
}

template<class Offsets>
__device__ __forceinline__ void initializeShared(uint32_t block) {
  if(threadIdx.x<256)
    store_shared_u32(Offsets::bufferCountsOffset + threadIdx.x*4, 0);
      
  #pragma unroll 1
  for(int32_t i=threadIdx.x;i<2048;i+=blockDim.x)
    store_shared_u4(Offsets::buffersOffset + i*16, make_uint4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF));

  #pragma unroll 1
  for(int32_t i=threadIdx.x;i<1024;i+=blockDim.x)
    store_shared_u4(Offsets::pointCountsOffset + i*16, make_uint4(0, 0, 0, 0));

  if(threadIdx.x<8)
    store_shared_u32(Offsets::countersOffset + threadIdx.x*4, 0);      
}

template<class Offsets>
__device__ __forceinline__ void prefixSumBuckets(uint32_t block, uint32_t base) {
  uint32_t halfWarp=threadIdx.x>>4, halfWarpThread=threadIdx.x & 0x0F, halfWarps=blockDim.x>>4;
  uint32_t counts[8];      // we have at most 16 counts per warp
  uint32_t stop, count, sum;
  
  if(blockDim.x==512) stop=8;
  if(blockDim.x==1024) stop=4;
  
  #pragma unroll
  for(int32_t i=0;i<8;i++) {
    if(i<stop) {
      counts[i]=load_shared_u32(Offsets::pointCountsOffset + i*blockDim.x*4 + threadIdx.x*4);
      sum=warpPrefixSum(counts[i], 16);
      if(halfWarpThread==15)
        store_shared_u32(Offsets::bufferCountsOffset + i*halfWarps*4 + halfWarp*4, sum);
      counts[i]=sum-counts[i];   // make it an exclusive prefix sum
    }
  }

  __syncthreads();
  
  if(threadIdx.x<256) {
    count=load_shared_u32(Offsets::bufferCountsOffset + threadIdx.x*4);
    sum=multiwarpPrefixSum(Offsets::warpSumsOffset, count, 8);
    store_shared_u32(Offsets::prefixCountsOffset + threadIdx.x*4, sum - count);
  }
 
  __syncthreads();

  #pragma unroll
  for(int32_t i=0;i<8;i++) {
    if(i<stop) {
      counts[i]+=load_shared_u32(Offsets::prefixCountsOffset + i*halfWarps*4 + halfWarp*4) + base;
      store_shared_u32(Offsets::pointCountsOffset + i*blockDim.x*4 + threadIdx.x*4, counts[i]);
    }
  }
}

template<class Offsets>
__device__ __forceinline__ void sortMap(uint32_t block, uint32_t scratchCount) {
  uint32_t count, sum, bin, nextIndex, mapEntry;
  
  // bufferCounts contains the total number of points written to that buffer
  
  if(threadIdx.x<256) {
    count=load_shared_u32(Offsets::bufferCountsOffset + threadIdx.x*4);
    count=(count+31)>>5;                                                             // convert point count to number of 128-byte segments
    sum=multiwarpPrefixSum(Offsets::warpSumsOffset, count, 8);                       // compute the prefix sum of 128-byte segments
    store_shared_u32(Offsets::prefixCountsOffset + threadIdx.x*4, sum - count);      // over-write prefix counts
  }

  __syncthreads();
  
  for(uint32_t scratchIndex=threadIdx.x;scratchIndex<scratchCount;scratchIndex+=blockDim.x) {
    bin=load_shared_byte(Offsets::mapOffset + scratchIndex);
    nextIndex=shared_atomic_add_u32(Offsets::prefixCountsOffset + bin*4, 1);
    mapEntry=(bin<<24) + scratchIndex;
    store_shared_u32(Offsets::sortedMapOffset + nextIndex*4, mapEntry);
  }
}

template<class Offsets>
__device__ __forceinline__ void cleanupShared(uint32_t block, void* scratchPtr) {
  uint32_t warp=threadIdx.x>>5, warpThread=threadIdx.x & 0x1F, warps=blockDim.x>>5;
  uint32_t count, scratchIndex, data;
  
  #pragma unroll 1
  for(uint32_t bin=warp;bin<256;bin+=warps) {
    count=load_shared_u32(Offsets::bufferCountsOffset + bin*4);
    if(count>0) {
      if(warpThread==0) {
        scratchIndex=shared_atomic_add_u32(Offsets::countersOffset, 1);
        store_shared_byte(Offsets::mapOffset + scratchIndex, bin);
      }
      scratchIndex=__shfl_sync(0xFFFFFFFF, scratchIndex, 0);
      data=shared_atomic_exch_u32(Offsets::buffersOffset + bin*128 + warpThread*4, 0xFFFFFFFF);
      *(uint32_t*)byteOffset(scratchPtr, scratchIndex*128 + warpThread*4)=data;
    }
  }
}

template<class Offsets>
__device__ __forceinline__ void writePointToShared(uint32_t block, void* scratchPtr, uint32_t lowBits, uint32_t highBits, bool valid) {
  uint32_t  warpThread=threadIdx.x & 0x1F;
  uint32_t  offset, bin, scratchIndex, mask, thread, shuffleBin, shuffleIndex, data;
  bool      processed=!valid, writeRequired;
  
  while(!__all_sync(0xFFFFFFFF, processed)) {
    offset=0;
    if(!processed) {
      bin=__funnelshift_l(lowBits, highBits, 1);
      offset=shared_atomic_add_u32(Offsets::bufferCountsOffset + bin*4, 4);
      if(offset<=124) {
        // if(shared_atomic_exch_u32(Offsets::buffersOffset + bin*128 + offset, lowBits & 0x7FFFFFFF)!=0xFFFFFFFF)
        //   printf("That's not right!\n");
        store_shared_u32(Offsets::buffersOffset + bin*128 + offset, lowBits & 0x7FFFFFFF);
        processed=true;
      }
    }

    writeRequired=(offset==124);
    if(writeRequired) {
      scratchIndex=shared_atomic_add_u32(Offsets::countersOffset, 1);
      store_shared_byte(Offsets::mapOffset + scratchIndex, bin);
    }
 
    while(true) {
      mask=__ballot_sync(0xFFFFFFFF, writeRequired);
      if(mask==0)
        break;
      thread=31-__clz(mask);
     
      shuffleBin=__shfl_sync(0xFFFFFFFF, bin, thread);
      shuffleIndex=__shfl_sync(0xFFFFFFFF, scratchIndex, thread);
     
      data=shared_atomic_exch_u32(Offsets::buffersOffset + shuffleBin*128 + warpThread*4, 0xFFFFFFFF);
      while(data==0xFFFFFFFF) 
        data=shared_atomic_exch_u32(Offsets::buffersOffset + shuffleBin*128 + warpThread*4, 0xFFFFFFFF);
      
      __syncwarp(0xFFFFFFFF);
      
      *(uint32_t*)byteOffset(scratchPtr, shuffleIndex*128 + warpThread*4)=data;
      
      if(warpThread==thread) {
        shared_atomic_exch_u32(Offsets::bufferCountsOffset + bin*4, 0);
        writeRequired=false;
      }      
    }
  }
}

template<class Offsets>
__device__ __forceinline__ void partitionPagesToScratch(uint32_t block, void* scratchPtr, void* pagesPtr, uint32_t size) {
  uint32_t warp=threadIdx.x>>5, warps=blockDim.x>>5, warpThread=threadIdx.x & 0x1F;
  uint32_t offset, idx, nextPage;
  uint32_t data[5], lowBits[4], highBits[4]; 
  void*    currentPage;
  
  // 1. reads from input pages
  // 2. writes to a shared memory buffer, using the top 8 bits of the bucket as the buffer index
  // 3. copies full buffers from shared to scratch
  // 4. writes any partial buffers out to scratch at the end
  
  currentPage=byteOffset(pagesPtr, block, PAGE_SIZE);
  offset=warp*640;
  idx=warp*128;
  while(idx<size) {
    while(offset>=PAGE_SIZE-4) {
      nextPage=*(uint32_t*)byteOffset(currentPage, PAGE_SIZE-4);
      currentPage=byteOffset(pagesPtr, nextPage, PAGE_SIZE);
      offset-=PAGE_SIZE-4;
    }
    if(idx<size)
      read640(data, pagesPtr, currentPage, offset);
    unpackData(lowBits, highBits, data);
    #pragma unroll
    for(int32_t i=0;i<4;i++) {
      bool valid=idx+i*32+warpThread<size;
        
      if(valid)
        shared_reduce_add_u32(Offsets::pointCountsOffset + __funnelshift_r(lowBits[i], highBits[i], 27)*4, 1);
      writePointToShared<Offsets>(block, scratchPtr, lowBits[i], highBits[i], valid);
    }
    offset+=warps*640;
    idx+=warps*128;
  }
  
  __syncthreads();
    
  cleanupShared<Offsets>(block, scratchPtr);
}

template<class Offsets>
__device__ __forceinline__ void partitionScratchToPoints(uint32_t block, void* pointsPtr, void* scratchPtr, uint32_t scratchCount, uint32_t points) {
  uint32_t warp=threadIdx.x>>5, warps=blockDim.x>>5, warpThread=threadIdx.x & 0x1F;
  uint32_t mapEntry, bucket, point, pointOffset, sign, pointGroup=(block>>11)*points;
  
  #pragma unroll 1
  for(uint32_t scratchIndex=warp;scratchIndex<scratchCount;scratchIndex+=warps) {
    mapEntry=load_shared_u32(Offsets::sortedMapOffset + scratchIndex*4);
    point=*(uint32_t*)byteOffset(scratchPtr, (mapEntry & 0x00FFFFFF)*128 + warpThread*4);
    bucket=(mapEntry>>20) + (point>>27);
    if((point & 0x80000000)==0) {
      pointOffset=shared_atomic_add_u32(Offsets::pointCountsOffset + bucket*4, 1);
      sign=(point & 0x04000000)<<5;
      *(uint32_t*)byteOffset(pointsPtr, pointOffset*4)=((point & 0x03FFFFFF) | sign) + pointGroup;
    }
  }
}

template<class Offsets>
__device__ __forceinline__ void countFromPages(uint32_t block, void* pagesPtr, uint32_t size) {
  uint32_t warp=threadIdx.x>>5, warps=blockDim.x>>5, warpThread=threadIdx.x & 0x1F;
  uint32_t offset, idx, nextPage;
  uint32_t data[5], lowBits[4], highBits[4]; 
  void*    currentPage;
  
  currentPage=byteOffset(pagesPtr, block, PAGE_SIZE);
  offset=warp*640;
  idx=warp*128;
  while(idx<size) {
    while(offset>=PAGE_SIZE-4) {
      nextPage=*(uint32_t*)byteOffset(currentPage, PAGE_SIZE-4);
      currentPage=byteOffset(pagesPtr, nextPage, PAGE_SIZE);
      offset-=PAGE_SIZE-4;
    }
    if(idx<size)
      read640(data, pagesPtr, currentPage, offset);
    unpackData(lowBits, highBits, data);
    #pragma unroll
    for(int32_t i=0;i<4;i++) {
      if(idx+i*32+warpThread<size)
        shared_reduce_add_u32(Offsets::pointCountsOffset + __funnelshift_r(lowBits[i], highBits[i], 27)*4, 1);
    }
    offset+=warps*640;
    idx+=warps*128;
  }
}

template<class Offsets>
__device__ __forceinline__ void partitionPagesToPoints(uint32_t block, void* pointsPtr, void* pagesPtr, uint32_t size, uint32_t points) {
  uint32_t warp=threadIdx.x>>5, warps=blockDim.x>>5, warpThread=threadIdx.x & 0x1F;
  uint32_t offset, idx, nextPage, index, sign, pointGroup=(block>>11)*points;
  uint32_t data[5], lowBits[4], highBits[4];
  void*    currentPage;
  
  currentPage=byteOffset(pagesPtr, block, PAGE_SIZE);
  offset=warp*640;
  idx=warp*128;
  while(idx<size) {
    while(offset>=PAGE_SIZE-4) {
      nextPage=*(uint32_t*)byteOffset(currentPage, PAGE_SIZE-4);
      currentPage=byteOffset(pagesPtr, nextPage, PAGE_SIZE);
      offset-=PAGE_SIZE-4;
    }
    if(idx<size)
      read640(data, pagesPtr, currentPage, offset);
    unpackData(lowBits, highBits, data);
    #pragma unroll
    for(int32_t i=0;i<4;i++) {
      if(idx+i*32+warpThread<size) {
        index=shared_atomic_add_u32(Offsets::pointCountsOffset + __funnelshift_r(lowBits[i], highBits[i], 27)*4, 1);
        sign=(lowBits[i] & 0x04000000)<<5;
        *(uint32_t*)byteOffset(pointsPtr, index*4)=((lowBits[i] & 0x03FFFFFF) | sign) + pointGroup;
      }
    }
    offset+=warps*640;
    idx+=warps*128;
  }
}

__launch_bounds__(1024)
__global__ void partition4096Kernel(void* pointsPtr, void* unsortedTriplePtr, void* scratchPtr, void* prefixSumSizesPtr, 
                                    void* sizesPtr, void* pagesPtr, void* atomicsPtr, uint32_t points) {                                 
  uint4*    unsortedCounts=(uint4*)unsortedTriplePtr;
  uint4*    unsortedIndexes=(uint4*)byteOffset(unsortedTriplePtr, NBUCKETS*11*4);
  uint32_t  block, size, prefixSumSize, scratchCount;
  
  extern __shared__ uint32_t shmem[];
  
  typedef ProcessFinalBinsOffsets Offsets;
  
  scratchPtr=byteOffset(scratchPtr, blockIdx.x, round128(SCRATCH_REQUIRED));

  __syncthreads();
  
  while(true) {
    if(threadIdx.x==0) {
      block=atomicAdd((uint32_t*)byteOffset(atomicsPtr, 4), 1);
      store_shared_u32(Offsets::countersOffset + 4, block);
    }
    
    __syncthreads();

    block=load_shared_u32(Offsets::countersOffset + 4);
      
    if(block>=11*1024)
      break;

    initializeShared<Offsets>(block);

    size=*(uint32_t*)byteOffset(sizesPtr, block*4);
    prefixSumSize=*(uint32_t*)byteOffset(prefixSumSizesPtr, block*4);
     
    __syncthreads();
  
    if(size<SIZE_LIMIT)
      partitionPagesToScratch<Offsets>(block, scratchPtr, pagesPtr, size);
    else
      countFromPages<Offsets>(block, pagesPtr, size);                                   // shouldn't happen with random data
      
    __syncthreads();
    
    shared_copy_u4(byteOffset(unsortedCounts, block*16384), Offsets::pointCountsOffset, 1024);

    __syncthreads();

    prefixSumBuckets<Offsets>(block, prefixSumSize);

    __syncthreads();

    shared_copy_u4(byteOffset(unsortedIndexes, block*16384), Offsets::pointCountsOffset, 1024);
    
    __syncthreads();
    
    if(size<SIZE_LIMIT) {
      scratchCount=load_shared_u32(Offsets::countersOffset);
      sortMap<Offsets>(block, scratchCount);
    }

    __syncthreads();
    
    if(size<SIZE_LIMIT)
      partitionScratchToPoints<Offsets>(block, pointsPtr, scratchPtr, scratchCount, points);
    else
      partitionPagesToPoints<Offsets>(block, pointsPtr, pagesPtr, size, points);        // wicked slow, shouldn't happen with random data

    __syncthreads();
    
#if 0
    // super useful debugging code for catching race conditions
    
    #pragma unroll 1
    for(int32_t i=threadIdx.x;i<4094;i+=blockDim.x) {
      uint32_t localCounter=load_shared_u32(Offsets::pointCountsOffset + i*4);
      uint32_t expectedCounter=*(uint32_t*)byteOffset(unsortedIndexes, i*4 + block*16384 + 4);
      
      if(localCounter!=expectedCounter)
        printf("Post write prefix mismatch at block=%d index=%d local=%d expected=%d size=%d\n", block, i, localCounter, expectedCounter, size);
    }    

    __syncthreads();
#endif
  }
}