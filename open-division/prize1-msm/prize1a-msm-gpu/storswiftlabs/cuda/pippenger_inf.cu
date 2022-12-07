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
    size_t d_scalars_idxs[NUM_BATCH_THREADS];
    scalar_t *h_scalars;

    typename pipp_t::result_container_t res0;
    typename pipp_t::result_container_t res1;
};

template<class bucket_t, class affine_t, class scalar_t>
struct RustContext {
    Context<bucket_t, affine_t, scalar_t> *context;
};

// Initialization function
// Allocate device storage, transfer bases
extern "C"
RustError mult_pippenger_init(RustContext<bucket_t, affine_t, scalar_t> *context,
                              const affine_t points[], size_t npoints,
                              size_t ffi_affine_sz)
{
    context->context = new Context<bucket_t, affine_t, scalar_t>();
    Context<bucket_t, affine_t, scalar_t> *ctx = context->context;

    ctx->ffi_affine_sz = ffi_affine_sz;
    try {
        ctx->config = ctx->pipp.init_msm(npoints);

        // Allocate GPU storage
        ctx->d_points_idx = ctx->pipp.allocate_d_bases(ctx->config);
        ctx->d_buckets_idx = ctx->pipp.allocate_d_buckets(ctx->config);
        ctx->d_dones_idx = ctx->pipp.allocate_d_dones(ctx->config);        
        for (size_t i = 0; i < NUM_BATCH_THREADS; i++) {
            ctx->d_scalars_idxs[i] = ctx->pipp.allocate_d_scalars(ctx->config);
        }
        // Allocate pinned memory on host
        CUDA_OK(cudaMallocHost(&ctx->h_scalars, ctx->pipp.get_size_scalars(ctx->config)));
        
        ctx->pipp.transfer_bases_to_device(ctx->config, ctx->d_points_idx, points,
                                           ffi_affine_sz);
        
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
    
    (void)ffi_affine_sz;
    Context<bucket_t, affine_t, scalar_t> *ctx = context->context;
    assert(ctx->config.npoints == npoints);
    assert(ctx->ffi_affine_sz == ffi_affine_sz);
    assert(batches > 0);

    cudaStream_t stream = ctx->pipp.default_stream;
    stream_t aux_stream(ctx->pipp.get_device());
    
    try {
        // Set results to infinity in case of failure along the way
        for (size_t i = 0; i < batches; i++) {
            out[i].inf();
        }

        typename pipp_t::result_container_t *kernel_res = &ctx->res0;
        typename pipp_t::result_container_t *accum_res = &ctx->res1;
        size_t d_scalars_xfer = ctx->d_scalars_idxs[0];
        size_t d_scalars_compute = ctx->d_scalars_idxs[1];
            
        channel_t<size_t> ch;
        size_t scalars_sz = ctx->pipp.get_size_scalars(ctx->config);

        // The following loop overlaps bucket computation on the GPU with transfer
        // of the next set of scalars. 
        int work = 0;
        ctx->pipp.transfer_scalars_to_device(ctx->config, d_scalars_compute,
                                             &scalars[work * npoints], aux_stream);

        for (; work < (int)batches; work++) {
            // Launch the GPU kernel, transfer the results back
            batch_pool.spawn([&]() {
                CUDA_OK(cudaStreamSynchronize(aux_stream));
                ctx->pipp.launch_kernel(ctx->config, ctx->d_points_idx, d_scalars_compute,
                                ctx->d_buckets_idx, ctx->d_dones_idx, false);
                ctx->pipp.transfer_buckets_to_host(ctx->config, *kernel_res, ctx->d_dones_idx);
                ctx->pipp.synchronize_stream();
                ch.send(work);
            });

            // Transfer the next set of scalars, accumulate the previous result
            batch_pool.spawn([&]() {
                // Start next scalar transfer
                if (work + 1 < (int)batches) {
                    // Copy into pinned memory
                    memcpy(ctx->h_scalars, &scalars[(work + 1) * npoints], scalars_sz);
                    ctx->pipp.transfer_scalars_to_device(ctx->config,
                                                         d_scalars_xfer, ctx->h_scalars,
                                                         aux_stream);
                }
                // Accumulate the previous result
                if (work - 1 >= 0) {
                    ctx->pipp.accumulate(ctx->config, out[work - 1], *accum_res);
                }
                
                ch.send(work);
            });
            ch.recv();
            ch.recv();
            std::swap(kernel_res, accum_res);
            std::swap(d_scalars_xfer, d_scalars_compute);
        }

        // Accumulate the final result
        ctx->pipp.accumulate(ctx->config, out[batches - 1], *accum_res);

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
