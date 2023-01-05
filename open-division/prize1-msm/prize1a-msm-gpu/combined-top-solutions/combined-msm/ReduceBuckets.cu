/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Written by Niall Emmart.

***/

#define WINDOWS 2
#define BUCKETS_PER_WINDOW 0x400000

__device__ __forceinline__ void prefetchXY(uint32_t storeOffset, uint32_t bucketIndex, void* bucketsPtr, uint32_t limit) {
  uint32_t bucketIndex0, bucketIndex1, oddEven=threadIdx.x & 0x01;
  void*    p0;
  void*    p1;
  
  bucketIndex0=bucketIndex;
  bucketIndex1=__shfl_xor_sync(0xFFFFFFFF, bucketIndex, 1);
  if(oddEven!=0) {
    storeOffset-=80;
    bucketIndex0=bucketIndex1;
    bucketIndex1=bucketIndex;
  }

  p0=byteOffset(bucketsPtr, bucketIndex0, 192);
  p1=byteOffset(bucketsPtr, bucketIndex1, 192);
  
  if(bucketIndex0<limit) {
    shared_async_copy_u4(storeOffset+0, byteOffset(p0, 0 + oddEven*16));
    shared_async_copy_u4(storeOffset+32, byteOffset(p0, 32 + oddEven*16));
    shared_async_copy_u4(storeOffset+64, byteOffset(p0, 64 + oddEven*16));
  }
  if(bucketIndex1<limit) {
    shared_async_copy_u4(storeOffset+96, byteOffset(p1, 0 + oddEven*16));
    shared_async_copy_u4(storeOffset+128, byteOffset(p1, 32 + oddEven*16));
    shared_async_copy_u4(storeOffset+160, byteOffset(p1, 64 + oddEven*16));
  }
  shared_async_copy_commit();
}

__device__ __forceinline__ void prefetchZZ(uint32_t storeOffset, uint32_t bucketIndex, void* bucketsPtr, uint32_t limit) {
  uint32_t bucketIndex0, bucketIndex1, oddEven=threadIdx.x & 0x01;
  void*    p0;
  void*    p1;
  
  bucketIndex0=bucketIndex;
  bucketIndex1=__shfl_xor_sync(0xFFFFFFFF, bucketIndex, 1);
  if(oddEven!=0) {
    storeOffset-=80;
    bucketIndex0=bucketIndex1;
    bucketIndex1=bucketIndex;
  }

  p0=byteOffset(bucketsPtr, bucketIndex0, 192);
  p1=byteOffset(bucketsPtr, bucketIndex1, 192);

  if(bucketIndex0<limit) {
    shared_async_copy_u4(storeOffset+0, byteOffset(p0, 96 + oddEven*16));
    shared_async_copy_u4(storeOffset+32, byteOffset(p0, 128 + oddEven*16));
    shared_async_copy_u4(storeOffset+64, byteOffset(p0, 160 + oddEven*16));
  }
  if(bucketIndex1<limit) {
    shared_async_copy_u4(storeOffset+96, byteOffset(p1, 96 + oddEven*16));
    shared_async_copy_u4(storeOffset+128, byteOffset(p1, 128 + oddEven*16));
    shared_async_copy_u4(storeOffset+160, byteOffset(p1, 160 + oddEven*16));
  }
  shared_async_copy_commit();
}

/*  Already declared in ComputeBucketSums
__device__ __forceinline__ void prefetchWait() {
  shared_async_copy_wait();
}
*/

__launch_bounds__(256)
__global__ void reduceBuckets(void* reduced, void* bucketsPtr) {
  typedef BLS12377::G1Montgomery           Field;
  typedef CurveXYZZ::HighThroughput<Field> AccumulatorXYZZ;
  typedef PointXYZZ<Field>                 PointXYZZ;

  uint32_t warps=gridDim.x*blockDim.x>>5, warp=blockIdx.x*blockDim.x + threadIdx.x>>5, warpThread=threadIdx.x & 0x1F;  
  uint32_t warpsPerWindow=warps / WINDOWS;
  uint32_t bucketsPerThread=(BUCKETS_PER_WINDOW + warpsPerWindow*32 - 1)/(warpsPerWindow*32);
  uint32_t window=warp/warpsPerWindow;
  uint32_t pointOffset=threadIdx.x*96 + 1536;
  int32_t  stop=window*BUCKETS_PER_WINDOW + ((warp-window*warpsPerWindow)*32 + warpThread)*bucketsPerThread;
  int32_t  start=stop+bucketsPerThread-1;
  int32_t  limit=(window+1)*BUCKETS_PER_WINDOW;
  
  AccumulatorXYZZ sum, sumOfSums;
  PointXYZZ       point;  
  
  copyToShared((uint4*)SHMData);

  if(window>WINDOWS)
    return;
        
  if(start<limit)
    point.load(byteOffset(bucketsPtr, start, 192));
    
  #pragma unroll 1
  for(int32_t i=start;i>=stop;i--) {
    if(i>stop)
      prefetchXY(pointOffset, i-1, bucketsPtr, limit);
    sum.add(point, i<limit);
    prefetchWait();
    point.loadSharedXY(pointOffset);
    if(i>stop)
      prefetchZZ(pointOffset, i-1, bucketsPtr, limit);
    sumOfSums.add(PointXYZZ(sum.x, sum.y, sum.zz, sum.zzz));
    prefetchWait();
    point.loadSharedZZ(pointOffset);
  }

  // This next chunk of code is a little bit tricky.  In the first iteration, it writes
  // sum(sumOfSum) and stores it to offset 0.  In the second, it computes sum(sum) and stores it
  // to offset 1, and in the final iteration it computes sum(warpThread*sum) and stores it to
  // offset 2.
  // Since everything is inlined, this approach minimizes the number of copies of the add routine.
  
  #pragma unroll 1
  for(int32_t j=0;j<3;j++) {
    #pragma unroll 1
    for(int32_t i=1;i<=16;i=i<<1) {
      point=PointXYZZ(sumOfSums.x, sumOfSums.y, sumOfSums.zz, sumOfSums.zzz);
      point.warpShuffle(warpThread ^ i);
      sumOfSums.add(point);
    }
 
    if(warpThread==0)
      sumOfSums.accumulator().store(byteOffset(reduced, warp*3 + j, 192));
      
    if(j==0) 
      sumOfSums=sum;
    else if(j==1) {
      // set sumOfSums = warpThread * sum
      point=PointXYZZ(sum.x, sum.y, sum.zz, sum.zzz);
      sumOfSums.setZero();
      #pragma unroll 1
      for(int32_t i=16;i>=1;i=i>>1) {
        sumOfSums.add(point, (warpThread & i)!=0);
        if(i>1)
          sumOfSums.dbl();
      }
    }
  }
}

