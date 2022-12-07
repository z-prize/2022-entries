/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Written by Niall Emmart.

***/

__device__ __constant__ uint32_t divisorApprox[24]={
  0x00000000, 0xFFFFFFFF, 0x7FFFFFFF, 0x55555555, 0x3FFFFFFF, 0x33333333,
  0x2AAAAAAA, 0x24924924, 0x1FFFFFFF, 0x1C71C71C, 0x19999999, 0x1745D174, 
  0x15555555, 0x13B13B13, 0x12492492, 0x11111111, 0x0FFFFFFF, 0x0F0F0F0F, 
  0x0E38E38E, 0x0D79435E, 0x0CCCCCCC, 0x0C30C30C, 0x0BA2E8BA, 0x0B21642C,
};

// GROUP MUST BE 27 OR LESS!
#define GROUP 13

__device__ __forceinline__ uint32_t copyCountsAndIndexes(uint32_t countsAndIndexesOffset, uint4* sortedCountsAndIndexes, uint32_t bucket) {
  uint32_t count;
  uint4    load;

  // copy 1536 bytes
  load=sortedCountsAndIndexes[0];
  count=load.x + load.z;
  load.y=load.y<<2;
  load.w=load.w<<2;
  store_shared_u2(countsAndIndexesOffset, make_uint2(load.x, load.y));
  store_shared_u2(countsAndIndexesOffset + 256, make_uint2(load.z, load.w));
  load=sortedCountsAndIndexes[1];
  count+=load.x + load.z;
  load.y=load.y<<2;
  load.w=load.w<<2;
  store_shared_u2(countsAndIndexesOffset + 512, make_uint2(load.x, load.y));
  store_shared_u2(countsAndIndexesOffset + 768, make_uint2(load.z, load.w));
  load=sortedCountsAndIndexes[2];
  count+=load.x + load.z;
  load.y=load.y<<2;
  load.w=load.w<<2;
  store_shared_u2(countsAndIndexesOffset + 1024, make_uint2(load.x, load.y));
  store_shared_u2(countsAndIndexesOffset + 1280, make_uint2(load.z, load.w));
  return count;
}

__device__ __forceinline__ void copyPointIndexes(uint32_t& sequence, uint32_t countsAndIndexesOffset, uint32_t pointIndexOffset, void* pointIndexes, uint32_t bucket) {
  uint32_t remaining, shift, available; 
  uint2    countAndIndex;
  uint4    quad;
  
  remaining=GROUP;
  available=0;
  countAndIndex.x=0;
  while(remaining>0) {
    // when we enter here, available is zero
    if(countAndIndex.x==0 && sequence==1536)
      break;
    if(countAndIndex.x==0) {
      countAndIndex=load_shared_u2(countsAndIndexesOffset + sequence);
      sequence+=256;
      shift=countAndIndex.y & 0x0F;
      countAndIndex.y=countAndIndex.y & 0xFFFFFFF0;
      quad=*(uint4*)byteOffset(pointIndexes, countAndIndex.y);
      
      shift=shift>>2;
      available=umin(countAndIndex.x, 4-shift);
      countAndIndex.y+=(shift + available)<<2;
      quad.x=(shift>=2) ? quad.z : quad.x;
      quad.y=(shift>=2) ? quad.w : quad.y;
      shift=shift & 0x01;
    }
    else {
      quad=*(uint4*)byteOffset(pointIndexes, countAndIndex.y);
      available=umin(countAndIndex.x, 4);
      countAndIndex.y+=available<<2;
      shift=0;
    }      
    countAndIndex.x-=available;
      
    while(remaining>0 && available>0) {
      quad.x=(shift>0) ? quad.y : quad.x;
      quad.y=(shift>0) ? quad.z : quad.y;
      quad.z=(shift>0) ? quad.w : quad.z;
      store_shared_u32(pointIndexOffset, quad.x);
      pointIndexOffset+=4;
      available--;
      remaining--;
      shift=1;
    }
  }
  __syncwarp(0xFFFFFFFF);
  countAndIndex.x+=available;
  countAndIndex.y-=available<<2;
  if(countAndIndex.x>0) {
    sequence-=256;
    store_shared_u2(countsAndIndexesOffset + sequence, countAndIndex);
  }
}

__device__ __forceinline__ void prefetch(uint32_t storeOffset, uint32_t pointIndex, void* pointsPtr) {
  uint32_t loadIndex, loadIndex0, loadIndex1, oddEven=threadIdx.x & 0x01;
  void*    p0;
  void*    p1;
  
  #if defined(SMALL) 
    loadIndex=(pointIndex & 0xFFFF) | ((pointIndex & 0x7C000000) >> 10);
  #else
    loadIndex=pointIndex & 0x7FFFFFFF;
  #endif
    
  loadIndex0=loadIndex;
  loadIndex1=__shfl_xor_sync(0xFFFFFFFF, loadIndex, 1);
  if(oddEven!=0) {
    storeOffset-=80;
    loadIndex0=loadIndex1;
    loadIndex1=loadIndex;
  }
  
  p0=byteOffset(pointsPtr, loadIndex0, 96);
  p1=byteOffset(pointsPtr, loadIndex1, 96);
  
  shared_async_copy_u4(storeOffset+0, byteOffset(p0, 0 + oddEven*16));
  shared_async_copy_u4(storeOffset+32, byteOffset(p0, 32 + oddEven*16));
  shared_async_copy_u4(storeOffset+64, byteOffset(p0, 64 + oddEven*16));
  shared_async_copy_u4(storeOffset+96, byteOffset(p1, 0 + oddEven*16));
  shared_async_copy_u4(storeOffset+128, byteOffset(p1, 32 + oddEven*16));
  shared_async_copy_u4(storeOffset+160, byteOffset(p1, 64 + oddEven*16));
  shared_async_copy_commit();
}

__device__ __forceinline__ void prefetchWait() {
  shared_async_copy_wait();
}

__launch_bounds__(384)
__global__ void computeBucketSums(void* bucketsPtr, void* pointsPtr, void* sortedTriplePtr, void* pointIndexesPtr, void* atomicsPtr) {
  typedef BLS12377::G1Montgomery           Field;
  typedef CurveXYZZ::HighThroughput<Field> AccumulatorXYZZ;
  typedef PointXY<Field>                   PointXY;
  typedef PointXYZZ<Field>                 PointXYZZ;

  AccumulatorXYZZ  acc;
  PointXY          point;
  
  uint32_t         warp=threadIdx.x>>5, warpThread=threadIdx.x & 0x1F;
  uint32_t*        atomics=((uint32_t*)atomicsPtr)+2;
  uint32_t*        sortedBuckets=(uint32_t*)sortedTriplePtr;
  uint4*           sortedCountsAndIndexes=(uint4*)(sortedBuckets + NBUCKETS*2 + 32);
  uint4*           pointIndexes=(uint4*)pointIndexesPtr;
  
  uint32_t         next, bucket, count, sequence, pointIndex;
  uint32_t         countsAndIndexesOffset, pointIndexesOffset, pointsOffset;
  
  // shared memory offsets:
  //       0 -  1535     1536         constants required by field routines
  //    1536 - 19967    18432         counts and indexes for the 384 threads (48 bytes per thread)
  //   19968 - end     GROUP*4*384    point indexes
  //    xxxx - yyyyy    36864         point data
  
  
  copyToShared((uint4*)SHMData);
  
  countsAndIndexesOffset=warp*1536 + warpThread*8 + 1536;  
  pointIndexesOffset=threadIdx.x*GROUP*4 + 19968;
  pointsOffset=threadIdx.x*96 + 384*GROUP*4 + 19968;
  
  while(true) {
    if(warpThread==0)
      next=atomicAdd(atomics, 32); 
    next=__shfl_sync(0xFFFFFFFF, next, 0);
    if(next>=NBUCKETS*2) {
      int32_t warps=gridDim.x*blockDim.x>>5;
      
      if(next>=NBUCKETS*2 + (warps-1)*32)
        atomics[0]=0;  
      break;
    }
    next=next + warpThread;

    bucket=sortedBuckets[next];
    count=copyCountsAndIndexes(countsAndIndexesOffset, sortedCountsAndIndexes + next*3, bucket);
    
    acc.setZero();
    
    sequence=0;
    while(__any_sync(0xFFFFFFFF, count>0)) {
      copyPointIndexes(sequence, countsAndIndexesOffset, pointIndexesOffset, pointIndexes, bucket);
      __syncwarp(0xFFFFFFFF);
      pointIndex=(count==0) ? 0 : load_shared_u32(pointIndexesOffset);
      prefetch(pointsOffset, pointIndex, pointsPtr);
      #pragma unroll 1
      for(int i=1;__any_sync(0xFFFFFFFF, count>0) && i<=GROUP;i++) {
        prefetchWait();
        __syncwarp(0xFFFFFFFF);
        point.loadShared(pointsOffset);
        if((pointIndex & 0x80000000)!=0)
          point.negate();
        __syncwarp(0xFFFFFFFF);
        pointIndex=(count==0) ? 0 : load_shared_u32(pointIndexesOffset + i*4);
        if(i<GROUP) 
          prefetch(pointsOffset, pointIndex, pointsPtr);
        __syncwarp(0xFFFFFFFF);
        acc.add(point, count>0);
        if(count>0)
          count--;
      }
    }
    
    #if 1
      acc.accumulator().store(byteOffset(bucketsPtr, bucket, 192));
    #else
      PointXY result;
      
      result=acc.normalize();
      result.fromInternal();
      result.store(byteOffset(bucketsPtr, bucket, 96));
    #endif
  }
}

