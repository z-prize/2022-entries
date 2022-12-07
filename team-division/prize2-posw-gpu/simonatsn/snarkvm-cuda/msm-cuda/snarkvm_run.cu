#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <bits/stdc++.h>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <atomic>
#include "MSM.h"

struct svm_context {
  int         smCount;
  int         rdcSmCount;
  size_t      maxN;
  GPUPlanner* planner;
  MSMParams   params;
  MSMReduce   bodyReduce;
  MSMReduce   tailReduce;

  svm_context(size_t _maxN, int _smCount) {
    maxN = _maxN;
    smCount = _smCount;
    rdcSmCount = min(8, _smCount);

    planner = new GPUPlanner(0);
    planner->_smCount = smCount;

    int windowBits = 15;
    computeMSMParams(params, bodyReduce, tailReduce,
                     rdcSmCount, windowBits, maxN);
  }
  ~svm_context() {
    delete planner;
  }
};

extern "C"
svm_context *msm_cuda_create_context(size_t _maxN, int _smCount) {
  svm_context *ctx = new svm_context(_maxN, _smCount);
  return ctx;
}

extern "C"
void msm_cuda_delete_context(svm_context *ctx) {
  delete ctx;
}

extern "C"
void msm_cuda_precomp_params(svm_context *ctx, uint32_t* windowBits, uint32_t* allWindows) {
  *windowBits = ctx->params.windowBits();
  *allWindows = ctx->params.allWindows();
}

// Kernel storage needs for bucket sums (host) and buckets + bucketSums (gpu)
extern "C"
void msm_cuda_storage(svm_context *ctx, size_t *host, size_t *gpu) {
  *host = ctx->bodyReduce.numBucketSumPoints * EXT_JACOBIAN_BYTES;
  *gpu  = (ctx->params.buckets() * EXT_JACOBIAN_BYTES +
           ctx->bodyReduce.numBucketSumPoints * EXT_JACOBIAN_BYTES);
}


__global__
void msm_print_fr(uint32_t* in) {
#ifdef __CUDA_ARCH__
  const size_t limbs = 8;
  printf("  scalar: 0x");
  for (int i = 0; i < limbs; i++) {
    printf("%08x", in[limbs - i - 1]);
  }
  printf("\n");
#endif
}

__global__
void msm_print_point(uint32_t* in) {
#ifdef __CUDA_ARCH__
  gpu_print_point(in);
#endif
}

void hostReduce(MSMParams &params, uint32_t warps,
                p1_xyz_t* result, p1_xyzz_t* bucketSums) {
#ifndef __CUDA_ARCH__
  Host::HostReduceSppark hostReduceSppark(params, warps);
  hostReduceSppark.reduce(result, (Host::bucket_t*)bucketSums);
#endif
}

extern "C"
void msm_cuda_precompute_bases(svm_context *ctx, size_t N,
                               p1_affine_t *d_precomp, p1_affine_t *d_bases) {
  uint32_t launch_tpb = 256;
  assert(N == ctx->params.precompStride());
  precomputePointsKernel<<<ctx->smCount, launch_tpb, 1536, 0>>>
    ((void*)d_precomp, (void*)d_bases, N,
     ctx->params.windowBits(), ctx->params.allWindows());
}

extern "C"
void msm_cuda_launch(svm_context *ctx,
                     size_t    N,
                     p1_xyz_t *result,
                     scalar_t *d_scalars,
                     p1_affine_t *d_points,
                     p1_xyzz_t *d_buckets,
                     p1_xyzz_t *h_bucketSums,
                     bool scalars_are_fr,
                     const cudaStream_t &stream) {
  uint32_t launch_tpb = 256;
  assert(N <= ctx->maxN);
  assert(ctx->params.precompStride() > 0); // Precomp must be configured

  GPUPlanner &gpuP = *ctx->planner;
  gpuP._stream = stream;
  MSMParams  &params = ctx->params;

  uint32_t rdcBlockCount;

  setMSMParams(params, SCALAR_BITS, params.windowBits(), N,
               params.precompWindows(), params.precompStride());
  rdcBlockCount = reduceBlocks(params, ctx->bodyReduce, ctx->tailReduce);
  p1_xyzz_t* d_bucketSums = &d_buckets[params.buckets()];

  gpuP.runWithDeviceData(params, (uint32_t*)d_scalars);

  computeBucketsXYZZ_G1<<<ctx->smCount, launch_tpb, 1536, stream>>>
    ((void*)d_buckets, params, gpuP._pointers, (void*)d_points);
  reduceBuckets<<<rdcBlockCount, launch_tpb, 1536, stream>>>
    (d_bucketSums, d_buckets, params, rdcBlockCount * launch_tpb / 32);
  $CUDA(cudaMemcpyAsync(h_bucketSums, d_bucketSums,
                        ctx->bodyReduce.numBucketSumPoints * EXT_JACOBIAN_BYTES,
                        cudaMemcpyDeviceToHost, stream));
  $CUDA(cudaStreamSynchronize(stream));

  hostReduce(params, rdcBlockCount * launch_tpb / 32, result, h_bucketSums);
}
