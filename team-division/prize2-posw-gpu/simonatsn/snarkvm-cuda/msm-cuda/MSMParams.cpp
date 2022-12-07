#include "MSMParams.hpp"

#ifndef ROUND64
#define ROUND64(x) (x + 63 & 0xFFFFFFC0)
#endif

uint64_t setMSMParams(MSMParams& params, uint32_t exponentBits,
                      uint32_t windowBits, uint32_t pointCount,
                      uint32_t precompWindows, uint32_t precompStride) {
  size_t    size, overlaySize1, overlaySize2, overlaySize3;

  params.set(exponentBits, windowBits, pointCount, precompWindows, precompStride);

  uint32_t  buckets = params.buckets();
  uint32_t  points  = params.points();
  
  overlaySize1=ROUND64(points)*2;
  overlaySize2=ROUND64(buckets) + ROUND64(buckets*2)*2;
  overlaySize3=ROUND64(buckets) + ROUND64(buckets*2) + ROUND64((buckets+31)/32)*2 + ROUND64(points + 32768);

  size=overlaySize1;
  if(overlaySize2>size)
    size=overlaySize2;
  if(overlaySize3>size)
    size=overlaySize3;

  // size in words
  size+=64 + ROUND64(points)*2;

  return size*4;
}

int32_t setMSMReduce(MSMReduce& reduce, uint32_t sumsPerBlock,
                     uint32_t threadsPerSum, uint32_t bucketsPerThread,
                     uint32_t bucketsPerSum,
                     uint32_t sumsPerWindow, uint32_t numBucketSumPoints,
                     uint32_t blocks) {
  if(sumsPerBlock*threadsPerSum!=256) {
    printf("Illegal params in setMSMReduce!\n");
    exit(1);
  }

  // printf("setMSMReduce\n");
  // printf("  sumsPerBlock %d\n", sumsPerBlock);
  // printf("  threadsPerSum %d\n", threadsPerSum);
  // printf("  bucketsPerThread %d\n", bucketsPerThread);
  // printf("  sumsPerWindow %d\n", sumsPerWindow);
  // printf("  numBucketSumPoints %d\n", numBucketSumPoints);
  // printf("  blocks %d\n", blocks);
  
  reduce.sumsPerBlock=sumsPerBlock;
  reduce.threadsPerSum=threadsPerSum;
  reduce.bucketsPerThread=bucketsPerThread;
  reduce.bucketsPerSum=bucketsPerSum;
  reduce.blocks=blocks;
  reduce.sumsPerWindow = sumsPerWindow;
  reduce.numBucketSumPoints = numBucketSumPoints;
  
  return 0;
}

static bool printParams = true;

int32_t computeMSMParams(MSMParams &params, MSMReduce& bodyReduce,
                         MSMReduce& tailReduce,
                         uint32_t desiredSMs, uint32_t windowBits,
                         uint32_t pointCount) {
  uint32_t sumsPerBlock       = 8;
  uint32_t threadsPerSum      = 32;
  uint32_t precompWindows     = 1;
  uint32_t tailBits           = 0;
  uint32_t blocksPerWindow    = 8;
  uint32_t sumsPerWindow      = 64;
  uint32_t bucketsPerSum      = 512;
  uint32_t bucketsPerThread   = 16;
  uint32_t numBucketSumPoints = 195;
  uint32_t sumsPerBlockTail   = 1;
  uint32_t threadsPerSumTail  = 256;
  uint32_t bucketsPerThreadTail = 16;
  uint32_t tailReduceBlocks   = (tailBits > 0) ? 1 : 0;

  setMSMParams(params, SCALAR_BITS, windowBits, pointCount,
               precompWindows, pointCount);

  setMSMReduce(bodyReduce, sumsPerBlock, threadsPerSum,
               bucketsPerThread, bucketsPerSum,
               sumsPerWindow, numBucketSumPoints,
               blocksPerWindow * precompWindows);

  // This numBucketSumPoints is not used - the tail count is included
  // in the body count. 
  setMSMReduce(tailReduce, sumsPerBlockTail,
               threadsPerSumTail, bucketsPerThreadTail,
               // Not actually used by tail reduce
               threadsPerSumTail * bucketsPerThreadTail,
               0, 0, tailReduceBlocks);

  return 0;
}

uint32_t reduceBlocks(const MSMParams& params, MSMReduce& body,
                      MSMReduce& tail) {
  return body.blocks + tail.blocks;
}
