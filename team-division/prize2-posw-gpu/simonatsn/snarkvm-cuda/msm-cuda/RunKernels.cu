__device__
void gpu_print_point(uint32_t* in) {
#ifdef __CUDA_ARCH__
  const size_t limbs = 12;
  printf("  X mont: 0x");
  for (int i = 0; i < limbs; i++) {
    printf("%08x", in[limbs - i - 1]);
  }
  printf("\n");
  in += limbs;
  printf("  Y mont: 0x");
  for (int i = 0; i < limbs; i++) {
    printf("%08x", in[limbs - i - 1]);
  }
  printf("\n");
#endif
}

// Convert points from affine with infinity to affine plus precomputation
__launch_bounds__(256)
__global__ void precomputePointsKernel(void* pointsPtr, void* affinePointsPtr,
                                       uint32_t pointCount, uint32_t windowBits,
                                       uint32_t allWindows) {
  typedef BLS12377::G1Montgomery           Field;
  typedef CurveXYZZ::HighThroughput<Field> AccumulatorXYZZ;
  typedef PointXY<Field>                   PointXY;
  
  AccumulatorXYZZ  acc;
  PointXY          point, result;
  uint32_t         globalTID = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t         globalStride = blockDim.x * gridDim.x;
  
  copyToShared((uint4*)SHMData);
  
  for (uint32_t i = globalTID; i < pointCount; i += globalStride) {
    point.loadUnaligned(byteOffset(affinePointsPtr, i, sizeof(p1_affine_t)));
    if (pointsPtr != affinePointsPtr) {
      point.store(byteOffset(pointsPtr, i, sizeof(p1_affine_t)));
    }
    
    acc.setZero();
    acc.add(point, true);

    // Iterate through the windows
    #pragma unroll 1
    for (uint32_t j = 1; j < allWindows; j++) {
      // Scale
      #pragma unroll 1
      for(uint32_t k = 0; k < windowBits; k++)
        acc.dbl();
      acc.normalize().store(byteOffset(pointsPtr, pointCount * j + i, sizeof(p1_affine_t)));
    }
  }
}

__device__ __forceinline__
void computeBucketsXYZZ(void *buckets, MSMParams msmParams,
                        GPUPointers pointers, void* points) {
  typedef BLS12377::G1Montgomery                            Field;
  typedef CurveXYZZ::HighThroughput<BLS12377::G1Montgomery> AccumulatorXYZZ;
  typedef PointXY<BLS12377::G1Montgomery>                   PointXY;
  typedef PointXYZZ<BLS12377::G1Montgomery>                 PointXYZZ;
  
  uint32_t* taskOffsets      = pointers.taskOffsets;
  uint32_t* taskData         = pointers.taskData;
  uint64_t* countsAndBuckets = pointers.countsAndBuckets;
  
  AccumulatorXYZZ accumulator;
  PointXY         point;
  int32_t         warpThread     = threadIdx.x & 0x1F;
  uint32_t        blockTaskCount = pointers.workCounts[2];
  uint32_t        warpTaskCount  = pointers.workCounts[3];
  uint32_t        bucketCount    = msmParams.buckets();
  uint32_t        bucket, count, /*roundCount,*/ pointIndex, task, totalAdds=0;
  uint64_t        countAndBucket;

  copyToShared((uint4*)SHMData);

  // launch smCount @ 256 threads

  if (blockTaskCount > 0) {
    printf("WARNING: unexpected block task count");
  }
  
  // handle warp tasks
  // 32 threads will accumulate across 32 buckets - the points are
  // strided for efficient access
  while (true) {
    if (warpThread == 0) {
      task = atomicAdd(pointers.workCounts + 1, 1);
    }
    task = __shfl_sync(0xFFFFFFFF, task, 0);
  
    if (task >= warpTaskCount)
      break;
      
    accumulator.setZero();

    countAndBucket = countsAndBuckets[blockTaskCount + task * 32];
    count          = msmParams._pointCount - uhigh(countAndBucket);
    if (count > 0) {
      uint32_t* pointIndexes = taskData + taskOffsets[task];
      pointIndex = pointIndexes[warpThread];
  
      if(pointIndex!=0)
        point.load(byteOffset(points, pointIndex - 1, Field::bytes * 2));
      accumulator.add(point, pointIndex != 0);

      #pragma unroll 1
      for (int32_t i = 1;i < count; i++) {
        pointIndex = pointIndexes[warpThread + i * 32];
        if(pointIndex!=0)
          point.load(byteOffset(points, pointIndex - 1, Field::bytes * 2));
        accumulator.add(point, pointIndex != 0);
      }
      totalAdds += count - 1;
    }
    
    if (blockTaskCount + task * 32 + warpThread < bucketCount) {
      countAndBucket = countsAndBuckets[blockTaskCount + task * 32 + warpThread];
      bucket = ulow(countAndBucket);
      
      PointXYZZ acc = accumulator.accumulator();
      accumulator.accumulator().store(byteOffset(buckets, bucket, Field::bytes * 4));
    }
  }

  atomicAdd(pointers.workCounts + 5, totalAdds);
}
__launch_bounds__(256)
__global__ void computeBucketsXYZZ_G1(void* buckets, MSMParams msmParams, GPUPointers pointers, void* points) {
  computeBucketsXYZZ(buckets, msmParams, pointers, points);
}

__launch_bounds__(256)
__global__ void reduceBuckets(void* reduced, void* buckets, MSMParams msmParams, uint32_t warps) {
  typedef BLS12377::G1Montgomery           Field;
  typedef CurveXYZZ::HighThroughput<Field> AccumulatorXYZZ;
  typedef PointXYZZ<Field>                 PointXYZZ;

  uint32_t warp             = blockIdx.x * blockDim.x + threadIdx.x >> 5;
  uint32_t warpThread       = threadIdx.x & 0x1F;
  uint32_t warpsPerWindow   = warps / msmParams.precompWindows();
  uint32_t bucketsPerThread = ((msmParams.bucketsPerWindow() + warpsPerWindow * 32 - 1) /
                               (warpsPerWindow * 32));
  uint32_t window           = warp / warpsPerWindow;
  
  // Accumulation will run from the greatest bucket to least. Stop at
  // the least bucket for this thread.
  int32_t  stop  = (window * msmParams.bucketsPerWindow() +
                    // Plus bucket for this thread within the window
                    ((warp - window * warpsPerWindow) * 32 + warpThread) * bucketsPerThread);
  int32_t  start = stop + bucketsPerThread - 1;

  // Skip 0th bucket
  stop++;
  start++;

  AccumulatorXYZZ sum;
  AccumulatorXYZZ sumOfSums;
  PointXYZZ       point;  
  
  copyToShared((uint4*)SHMData);

  if (window > msmParams.precompWindows())
    return;

  // Iterate through buckets. Compute sum and sumOfSums
  #pragma unroll 1
  for (int32_t i = start; i >= stop; i--) {
    // Invalid once bucket goes beyond buckets per window
    bool valid = i < (window + 1) * msmParams.bucketsPerWindow();
    point.load(byteOffset(buckets, i, 192));

    sum.add(point, valid);
    PointXYZZ sos_acc;
    sos_acc.assign(sum.x, sum.y, sum.zz, sum.zzz);
    sumOfSums.add(sos_acc);
  }
  
  #pragma unroll 1
  for (int32_t j = 0; j < 3; j++) {
    #pragma unroll 1
    for (int32_t i = 1; i <= 16; i = i << 1) {
      point = PointXYZZ(sumOfSums.x, sumOfSums.y, sumOfSums.zz, sumOfSums.zzz);
      point.warpShuffle(warpThread ^ i);
      sumOfSums.add(point);
    }

    if (warpThread == 0)
      sumOfSums.accumulator().store(byteOffset(reduced, warp * 3 + j, 192));
      
    if (j == 0) 
      sumOfSums = sum;
    
    else if (j == 1) {
      point = PointXYZZ(sum.x, sum.y, sum.zz, sum.zzz);
      sumOfSums.setZero();
      #pragma unroll 1
      for (int32_t i = 16; i >= 1; i = i >> 1) {
        sumOfSums.add(point, (warpThread & i) != 0);
        if (i > 1)
          sumOfSums.dbl();
      }
    }
  }
}
