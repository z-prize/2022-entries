// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __EXPORT_H__
#define __EXPORT_H__

extern "C" {
void *msm_cuda_create_context(size_t _maxN, int _smCount);
void msm_cuda_delete_context(void *ctx);
void msm_cuda_precomp_params(void *ctx, uint32_t* windowBits, uint32_t* allWindows);
void msm_cuda_storage(void *ctx, size_t *host, size_t *gpu);
void msm_cuda_precompute_bases(void *ctx, size_t N,
                               affine_noinf_t *d_precomp, affine_noinf_t *d_bases);
void msm_cuda_launch(void*           ctx,
                     size_t          N,
                     point_t*        result, // XYZ
                     scalar_t*       d_scalars,
                     affine_noinf_t* d_points,
                     bucket_t*       d_buckets,
                     bucket_t*       h_bucketSums,
                     bool            scalars_are_fr,
                     const cudaStream_t &stream);

};

#endif
