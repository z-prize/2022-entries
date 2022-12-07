#define ROUND32(x) (x+31 & 0xFFFFFFE0)
#define ROUND64(x) (x+63 & 0xFFFFFFC0)

const size_t FP_BYTES = 48;
const size_t AFFINE_BYTES = 96;
const size_t EXT_JACOBIAN_BYTES = 192;
const size_t SCALAR_BYTES = 32;
const uint32_t SCALAR_BITS = 253; // BLS12-377

typedef struct {
   uint32_t* workCounts;                // 64
   uint64_t* bucketsPoints;             // points

   uint64_t* countsAndBuckets;          // 2*buckets

   uint32_t* bucketOffsets;             // buckets
   uint32_t* taskOffsets;               // (buckets+31)/32
   uint32_t* taskData;                  // points + 32768

   uint64_t* unsortedBucketsPoints;
   uint64_t* unsortedCountsAndBuckets;  // 2*buckets
   uint32_t* unsummedTaskOffsets;       // (buckets+31)/32
} GPUPointers;

#include "util.h"
#include "types.hpp"
#include "MSMParams.cpp"
#include "asm.cu"
#include "General.cu"
#include "Chain.cu"
#include "MP.cu"
#include "SHM.cu"
#include "Curve.cu"
#include "PlannerKernels.cu"
#include "RunKernels.cu"
#include "GPUPlanner.cpp"
#include "HostReduceSppark.cpp"

