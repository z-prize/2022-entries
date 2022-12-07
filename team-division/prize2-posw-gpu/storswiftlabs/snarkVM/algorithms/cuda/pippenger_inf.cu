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
typedef bucket_t::affine_t affine_t;
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
    size_t d_dones_idx;
    size_t d_ones_idx;
    size_t d_scalars_idxs[NUM_BATCH_THREADS];
    scalar_t *h_scalars;

    typename pipp_t::result_container_t res0;
    typename pipp_t::result_container_t res1;

    bucket_t h_ones[16];  
    Context(size_t device, size_t affine)
        : pipp(device)
        , ffi_affine_sz(affine) {
    }
};

template<class bucket_t, class affine_t, class scalar_t>
struct RustContext {
    Context<bucket_t, affine_t, scalar_t> *context;
};

// Initialization function
// Allocate device storage, transfer bases
extern "C"
RustError mult_pippenger_init(RustContext<bucket_t, affine_t, scalar_t> *context,
                              //const affine_t points[],
                              size_t npoints,
                              size_t device_index,
                              size_t ffi_affine_sz,
                              int acc_level)
{
    context->context = new Context<bucket_t, affine_t, scalar_t>(device_index, ffi_affine_sz);
    Context<bucket_t, affine_t, scalar_t> *ctx = context->context;

    try {
        ctx->pipp.set_acc_level(acc_level);
        ctx->config = ctx->pipp.init_msm(npoints, device_index);

        // Allocate GPU storage
        ctx->d_points_idx = ctx->pipp.allocate_d_bases(ctx->config);
        ctx->d_buckets_idx = ctx->pipp.allocate_d_buckets(ctx->config);
        ctx->d_dones_idx = ctx->pipp.allocate_d_dones(ctx->config);
        ctx->d_ones_idx = ctx->pipp.allocate_d_ones(ctx->config);        
        for (size_t i = 0; i < NUM_BATCH_THREADS; i++) {
            ctx->d_scalars_idxs[i] = ctx->pipp.allocate_d_scalars(ctx->config);
        }
        // Allocate pinned memory on host
        CUDA_OK(cudaMallocHost(&ctx->h_scalars, ctx->pipp.get_size_scalars(ctx->config)));

        ctx->pipp.allocate_roots();

/*         ctx->pipp.transfer_bases_to_device(ctx->config, ctx->d_points_idx, points,
                                           ffi_affine_sz); */

        ctx->res0 = ctx->pipp.get_result_container(ctx->config);
        ctx->res1 = ctx->pipp.get_result_container(ctx->config);
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
                             point_t* out, const affine_t points[],
                             size_t npoints, size_t batches,
                             const scalar_t scalars[],
                             size_t ffi_affine_sz)
{
    (void)points; // Silence unused param warning

    Context<bucket_t, affine_t, scalar_t> *ctx = context->context;
    //assert(ctx->config.npoints == npoints);
    assert(ctx->ffi_affine_sz == ffi_affine_sz);
    assert(batches > 0);

    ctx->config = ctx->pipp.init_msm(npoints, 0);

    try {
        // Set results to infinity in case of failure along the way
        for (size_t i = 0; i < batches; i++) {
            out[i].inf();
        }

        typename pipp_t::result_container_t *kernel_res = &ctx->res0;

        size_t d_scalars_compute = ctx->d_scalars_idxs[0];

        // The following loop overlaps bucket computation on the GPU with transfer
        // of the next set of scalars.

        ctx->pipp.transfer_scalars_to_device(ctx->config, d_scalars_compute,
                                             scalars);

        ctx->pipp.transfer_bases_to_device(ctx->config, ctx->d_points_idx, points,
                                               ffi_affine_sz);

        ctx->pipp.launch_kernel(ctx->config, ctx->d_points_idx, d_scalars_compute,
                                ctx->d_buckets_idx, ctx->d_dones_idx, true);
        ctx->pipp.transfer_buckets_to_host(ctx->config, *kernel_res, ctx->d_dones_idx);
             
        if (npoints <32000) 
            ctx->pipp.transfer_res_to_host(ctx->config, ctx->h_ones, ctx->d_ones_idx, 16);
        ctx->pipp.synchronize_stream();

        // Accumulate the final result
        ctx->pipp.accumulate(ctx->config, out[0], *kernel_res, ctx->h_ones);

        ctx->pipp.free_all_ptrs();

    } catch (const cuda_error& e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }
    return RustError{cudaSuccess};

}

extern "C" RustError mult_io_helper(RustContext<bucket_t, affine_t, scalar_t> *context,
                         scalar_t inout[], const scalar_t roots[], int npoints)
{
    try {
    context->context->pipp.launch_io_helper(inout, roots, npoints);
    } catch (const cuda_error& e) {
        return RustError{e.code()};
    }
    return RustError{cudaSuccess};
}

extern "C" RustError mult_oi_helper(RustContext<bucket_t, affine_t, scalar_t> *context,
                         scalar_t inout[], const scalar_t roots[], int npoints)
{
    try {
    context->context->pipp.launch_oi_helper(inout, roots, npoints);
    } catch (const cuda_error& e) {
        return RustError{e.code()};
    }
    return RustError{cudaSuccess};
}

#endif  //  __CUDA_ARCH__
