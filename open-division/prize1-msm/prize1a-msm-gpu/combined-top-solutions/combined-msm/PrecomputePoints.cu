/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Written by Niall Emmart.

***/

__launch_bounds__(256)
__global__ void precomputePointsKernel(void* pointsPtr, void* affinePointsPtr, uint32_t pointCount) {
  typedef BLS12377::G1Montgomery           Field;
  typedef CurveXYZZ::HighThroughput<Field> AccumulatorXYZZ;
  typedef PointXY<Field>                   PointXY;
  
  AccumulatorXYZZ  acc;
  PointXY          point, result;
  uint32_t         globalTID=blockIdx.x*blockDim.x+threadIdx.x, globalStride=blockDim.x*gridDim.x;
  
  copyToShared((uint4*)SHMData);
  
  for(uint32_t i=globalTID;i<pointCount;i+=globalStride) {
    point.loadUnaligned(byteOffset(affinePointsPtr, i, 104));
    #if 0   // FIX FIX FIX
      point.toInternal();
    #endif
    point.store(byteOffset(pointsPtr, i, 96));
    
    acc.setZero();
    acc.add(point, true);
    #pragma unroll 1
    for(uint32_t j=1;j<6;j++) {
      #pragma unroll 1
      for(uint32_t k=0;k<46;k++)
        acc.dbl();
      acc.normalize().store(byteOffset(pointsPtr, pointCount*j + i, 96));
    }
  }
}
