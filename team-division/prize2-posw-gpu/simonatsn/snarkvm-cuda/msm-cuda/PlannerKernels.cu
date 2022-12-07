__device__ uint32_t uright_wrap(uint32_t lo, uint32_t hi, uint32_t amt) {
  uint32_t r;

  #if __CUDA_ARCH__>=320
    asm volatile ("shf.r.wrap.b32 %0,%1,%2,%3;" : "=r"(r) : "r"(lo), "r"(hi), "r"(amt));
  #else
    amt=amt & 0x1F;
    r=lo>>amt;
    r=r | (hi<<32-amt);
  #endif
  return r;
}

__global__ void kernelBuildBucketsAndPoints(MSMParams params, GPUPointers pointers, uint32_t* aData) {
  uint64_t *bucketsPoints=pointers.unsortedBucketsPoints;
  uint32_t gridTID=blockIdx.x*blockDim.x+threadIdx.x;
  uint32_t gridStride=blockDim.x*gridDim.x;
  // Width of windows
  uint32_t firstC=params.windowBits();
  uint32_t lastC=params.tailBits();
  if (params.tailBits() == 0) {
    lastC = firstC;
  }
  uint32_t firstMask=(1<<firstC)-1;
  uint32_t lastMask=(1<<lastC)-1;
  uint32_t words=(params._exponentBits+31)/32;
  uint32_t lo, hi;
  int32_t  groupCount=params.allWindows();

  // launch geometry
  //  SM_COUNT*4, 128 

  // Iterate through scalars
  for(int32_t i=gridTID;i<params._pointCount;i+=gridStride) {
    // and windows
    for(int32_t g=0;g<groupCount;g++) {
      uint32_t c=(g!=groupCount-1) ? firstC : lastC;
      uint32_t mask=(g!=groupCount-1) ? firstMask : lastMask;
      uint32_t bitOffset=g*params.windowBits();
      uint32_t wordOffset=words*i + (bitOffset>>5);
      uint32_t bucket;

      bitOffset=bitOffset & 0x1F;
      if(bitOffset+c<=32) 
        bucket=aData[wordOffset]>>bitOffset;
      else {
        lo=aData[wordOffset];
        hi=aData[wordOffset+1];
        bucket=uright_wrap(lo, hi, bitOffset);
      }
      bucket=(bucket & mask);
      uint32_t point_idx = g * params.precompStride() + (i + 1);
      bucketsPoints[g*params._pointCount+i]=((uint64_t)bucket << 32) + point_idx;
    }
  }
}

// Clear the buckets and counts array
//   bucket | count << 32
// count is initialized to the total number of points in the MSM
__global__ void kernelZeroCounts(MSMParams params, GPUPointers pointers) { 
  uint64_t *countsAndBuckets=pointers.unsortedCountsAndBuckets;
  uint32_t gridTID=blockIdx.x*blockDim.x+threadIdx.x;
  uint32_t gridStride=blockDim.x*gridDim.x;
  int32_t  bucketCount=params.buckets();
  
  if(gridTID<32)
    pointers.workCounts[gridTID]=0;
    
  for(int32_t i=gridTID;i<bucketCount;i+=gridStride) 
    countsAndBuckets[i]=make_wide(i, params._pointCount);
}

// Given a sorted list of buckets, determine the size and start offset of each bucket.
// Each thread processes 16 strided points in a chunk, then moves to the next set of points
// The algorithm works by loading the previous, current, and next bucket
// At the start of a new bucket sequence it stores the index in bucketOffsets and increments the counter by index
// At the end of a bucket sequence it decrements the counter by the index + 1
// The counter now contains the number of points - sequence length
__global__ void kernelBuildBucketOffsets(MSMParams params, GPUPointers pointers) {
  uint32_t *countsAndBuckets=(uint32_t*)pointers.unsortedCountsAndBuckets;
  uint32_t *bucketOffsets=pointers.bucketOffsets;
  uint64_t *bucketsPoints = pointers.bucketsPoints;
  uint32_t tid=blockIdx.x*blockDim.x*16 + threadIdx.x;
  uint32_t stride=blockDim.x*gridDim.x*16;
  uint32_t points=params.points();
  uint32_t chunkLength=blockDim.x*16;
  uint32_t mask=params.bucketsPerWindow()-1;
  uint32_t bucketBefore, bucket, bucketAfter;
  
  for(int32_t i=tid;i<points;i+=stride) {
    for(int32_t j=0;j<chunkLength;j+=blockDim.x) {
      uint32_t idx=i+j;
      
      if(idx<points) {
        if(idx>0)
          bucketBefore=bucketsPoints[idx-1] >> 32;
        else
          bucketBefore=0xFFFFFFFF;
        bucket=bucketsPoints[idx] >> 32;
        if(idx<points-1)
          bucketAfter=bucketsPoints[idx+1] >> 32;
        else
          bucketAfter=0xFFFFFFFF;
        
        if(bucketBefore!=bucket) {
          bucketOffsets[bucket]=idx;
          if((bucket & mask)!=0)
            atomicAdd(&countsAndBuckets[bucket*2+1], idx);
        }
        if(bucket!=bucketAfter) {
          if((bucket & mask)!=0)
            atomicAdd(&countsAndBuckets[bucket*2+1], 0-(idx+1));
        }
      }
    }
  }
}

// Scan the bucket counts.
// They should be sorted in ascending order, and since the count is actually
// number of points - length the largest buckets are ordered first.
// Determine the number of buckets with size > 1024 by looking for the boundary
// between 1024 and < 1024.
__global__ void kernelBuildWorkCounts(MSMParams params, GPUPointers pointers) {
  uint32_t *workCounts=pointers.workCounts;
  uint32_t *countsAndBuckets=(uint32_t*)pointers.countsAndBuckets;
  uint32_t gridTID=blockIdx.x*blockDim.x + threadIdx.x;
  uint32_t gridStride=blockDim.x*gridDim.x;
  uint32_t buckets=params.buckets();
  uint32_t beforeCount=1024, count;
  
  for(int32_t i=gridTID;i<buckets;i+=gridStride) {
    if(i>0)
      beforeCount=params._pointCount-countsAndBuckets[i*2-1];
    count=params._pointCount-countsAndBuckets[i*2+1];
    if(beforeCount>=1024 && count<1024) {
      workCounts[2]=0;                 // block task count
      workCounts[3]=(buckets-0+31)>>5; // warp task count
    }
    if(count<1024)
      break;
  }
}

// Extract the largest bucket in each warp
// Store the bucket size * 32 (32 threads in warp) to indicate
// storage needed for that warp task
__global__ void kernelBuildTaskOffsets(MSMParams params, GPUPointers pointers) {
  uint32_t *workCounts=pointers.workCounts;
  uint32_t *countsAndBuckets=(uint32_t*)pointers.countsAndBuckets;
  uint32_t *taskOffsets=pointers.unsummedTaskOffsets;
  uint32_t gridTID=blockIdx.x*blockDim.x+threadIdx.x;
  uint32_t gridStride=blockDim.x*gridDim.x;
  uint32_t blockTaskCount=pointers.workCounts[2];
  uint32_t warpTaskCount=pointers.workCounts[3];
  uint32_t bucketCount=params.buckets();
  uint32_t count, total=0;

  for(int32_t i=gridTID;i<bucketCount;i+=gridStride) {
    count=0;
    if(i<warpTaskCount)
      count=params._pointCount-countsAndBuckets[blockTaskCount*2 + i*64 + 1];
    taskOffsets[i]=count*32;
    total+=count*32;
  }
  atomicAdd(workCounts+4, total);
}

// The task offsets (which contains the length of each block) should be run through an
// exclusive prefix scan to compute the the offset of each block
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

__global__ void kernelBuildTaskData(MSMParams params, GPUPointers pointers) {
  uint32_t *bucketOffsets=pointers.bucketOffsets;
  uint32_t *taskOffsets=pointers.taskOffsets;
  uint32_t *taskData=pointers.taskData;
  uint64_t *bucketsPoints=pointers.bucketsPoints;
  uint64_t *countsAndBuckets=pointers.countsAndBuckets;

  uint32_t blockID=blockIdx.x;
  uint32_t blockStride=gridDim.x;
  uint32_t warpID=threadIdx.x>>5;
  uint32_t warpStride=blockDim.x>>5;
  uint32_t warpThread=threadIdx.x & 0x1F;
  
  int32_t  count, maxCount, zeroCount;
  uint32_t cbOffset, offset, base, bucket, point;
  uint64_t countAndBucket;
  uint32_t blockTaskCount=pointers.workCounts[2];
  uint32_t warpTaskCount=pointers.workCounts[3];
  uint32_t bucketCount=params.buckets();

  // Shared memory, per block, to gather data efficiently
  __shared__ uint32_t localBases[32];
  __shared__ uint32_t localCounts[32];
  __shared__ uint32_t localData[32*33];
  
  for(int32_t i=blockID;i<warpTaskCount;i+=blockStride) {
    cbOffset=blockTaskCount + i*32;
    offset=taskOffsets[i];
    countAndBucket=countsAndBuckets[cbOffset];
    maxCount=params._pointCount - uhigh(countAndBucket);

    __syncthreads();

    // Step 1
    // Prepare a warp's worth of work (pointers, counts)
    // The entire block will work on filling this in 
    if(threadIdx.x<32) {
      countAndBucket=make_wide(0, params._pointCount);
      if(cbOffset+warpThread<bucketCount)
        countAndBucket=countsAndBuckets[cbOffset+warpThread];
      count=params._pointCount - uhigh(countAndBucket);
      bucket=ulow(countAndBucket);
      base=bucketOffsets[bucket];

      localCounts[threadIdx.x]=count;
      localBases[threadIdx.x]=base - (maxCount - count);
    }

    // Step 2
    // Iterate through the max bucket size and gather work data
    for(int32_t j=0;j<maxCount;j+=32) {
      __syncthreads(); // block level sync
      
      for(int32_t col=warpID;col<32;col+=warpStride) {
        count=localCounts[col];
        base=localBases[col];
        zeroCount=maxCount-count;

        point=0;
        if(j+warpThread>=zeroCount && j+warpThread<maxCount) 
          point=bucketsPoints[base + j + warpThread] & (0xffffffff);

        localData[warpThread*33 + col]=point;
      }
      
      __syncthreads();
      
      // write the data
      count=min(maxCount-j, 32);
      for(int32_t row=warpID;row<count;row+=warpStride) {
        point=localData[row*33 + warpThread];
        taskData[offset + (j+row)*32 + warpThread]=point;
      }
    }
  }
}
