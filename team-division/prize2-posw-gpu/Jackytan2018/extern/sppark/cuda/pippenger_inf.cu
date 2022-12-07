// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>
#include <sys/mman.h>

# include <ff/bls12-377.hpp>

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_inf_t affine_t;
typedef fr_t scalar_t;

#include <msm/pippenger.cuh>

#ifndef __CUDA_ARCH__

static const size_t NUM_BATCH_THREADS = 2;
static thread_pool_t batch_pool(NUM_BATCH_THREADS);

typedef pippenger_t<bucket_t, point_t, affine_t, scalar_t> pipp_t;

// MSM context used store persistent state
template<class bucket_t, class affine_t, class scalar_t>
struct Context {
    pipp_t pipp;
    typename pipp_t::MSMConfig config;

    size_t ffi_affine_sz;
    size_t d_points_idx;
    size_t d_buckets_idx;
    size_t d_scalars_idxs;
    scalar_t *h_scalars;

    typename pipp_t::result_container_t res0;
};

template<class bucket_t, class affine_t, class scalar_t>
struct RustContext {
    Context<bucket_t, affine_t, scalar_t> *context;
};

// Initialization function
// Allocate device storage, transfer bases
extern "C"
RustError mult_pippenger_init(RustContext<bucket_t, affine_t, scalar_t> *context, int dev_id,
                              const affine_t points[], size_t npoints,
                              size_t ffi_affine_sz)
{    
    context->context = new Context<bucket_t, affine_t, scalar_t>();
    Context<bucket_t, affine_t, scalar_t> *ctx = context->context;
    if (dev_id > 0){
        pipp_t *pp = new pippenger_t<bucket_t, point_t, affine_t, scalar_t>(dev_id);
        ctx->pipp = *pp;
    }

    ctx->ffi_affine_sz = ffi_affine_sz;
    try {        
        ctx->config = ctx->pipp.init_msm(npoints); 

        // Allocate GPU storage
        ctx->d_points_idx = ctx->pipp.allocate_d_bases(ctx->config);
        ctx->d_buckets_idx = ctx->pipp.allocate_d_buckets(ctx->config);
        ctx->d_scalars_idxs = ctx->pipp.allocate_d_scalars(ctx->config);
        
        // Allocate pinned memory on host
        CUDA_OK(cudaMallocHost(&ctx->h_scalars, ctx->pipp.get_size_scalars(ctx->config)));
        
        ctx->pipp.transfer_bases_to_device(ctx->config, ctx->d_points_idx, points, ffi_affine_sz);

        ctx->res0 = ctx->pipp.get_result_container(ctx->config);
    } catch (const cuda_error& e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()}
#endif
    }
    return RustError{cudaSuccess};
}

// Peform MSM on a batch of scalars over fixed bases
extern "C"
RustError mult_pippenger_inf(RustContext<bucket_t, affine_t, scalar_t> *context,
                             point_t* out,
                             size_t npoints, size_t batches,
                             const scalar_t scalars[],
                             size_t ffi_affine_sz)
{
    Context<bucket_t, affine_t, scalar_t> *ctx = context->context;
    assert(ctx->config.npoints == npoints);
    assert(ctx->ffi_affine_sz == ffi_affine_sz);
    assert(batches > 0);

    cudaStream_t stream = ctx->pipp.default_stream;
    stream_t aux_stream(ctx->pipp.get_device());

    try {
        // Set results to infinity in case of failure along the way
        out[0].inf();

        typename pipp_t::result_container_t *kernel_res = &ctx->res0; 

        // The following loop overlaps bucket computation on the GPU with transfer
        // of the next set of scalars.        
        ctx->pipp.transfer_scalars_to_device(ctx->config, ctx->d_scalars_idxs, &scalars[0], aux_stream);
        
        CUDA_OK(cudaStreamSynchronize(aux_stream));
       
        ctx->pipp.launch_kernel(ctx->config, ctx->d_points_idx, ctx->d_scalars_idxs, ctx->d_buckets_idx, false);        
        ctx->pipp.transfer_buckets_to_host(ctx->config, *kernel_res, ctx->d_buckets_idx);        
        ctx->pipp.synchronize_stream();
        
        ctx->pipp.accumulate(ctx->config, out[0], *kernel_res);
    } catch (const cuda_error& e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()}
#endif
    }
    return RustError{cudaSuccess};
}

extern "C"
RustError mult_pippenger_update(RustContext<bucket_t, affine_t, scalar_t> *context, const affine_t points[])
{
    Context<bucket_t, affine_t, scalar_t> *ctx = context->context;
    try {       
        ctx->pipp.transfer_bases_to_device(ctx->config, ctx->d_points_idx, points, ctx->ffi_affine_sz);
    } catch (const cuda_error& e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()}
#endif
    }
    return RustError{cudaSuccess};
}

#endif  //  __CUDA_ARCH__
