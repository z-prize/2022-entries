// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>
#include <sys/mman.h>
#include <cub/cub.cuh>

#include <ff/bls12-377.hpp>
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_inf_t affine_t;
typedef fr_t scalar_t;

#include <msm/pippenger.cuh>

extern "C"
void mult_pippenger_faster_inf2()
{
    uint32_t *d_wval = nullptr; 
    uint32_t *d_wval_out = nullptr; 
    uint32_t *d_idx = nullptr; 
    uint32_t *d_idx_out = nullptr;
    uint32_t *d_offset_a = nullptr;
    uint32_t *d_offset_b = nullptr;
    void *d_temp = NULL;
    size_t temp_size = 0;

    cub::DeviceRadixSort::SortPairs(d_temp, temp_size, d_wval, d_wval_out, d_idx, d_idx_out, 1);  
    cub::DeviceSegmentedRadixSort::SortPairs(d_temp, temp_size, d_wval, d_wval_out, d_idx, d_idx_out, 1, 1, d_offset_a, d_offset_b); 
    cub::DeviceSelect::Flagged(d_temp, temp_size, d_idx, d_wval, d_idx_out, d_offset_a, 1);
    cub::DeviceScan::InclusiveSum(d_temp, temp_size, d_wval, d_idx, 1);
    cub::DeviceRunLengthEncode::Encode(d_temp, temp_size, d_wval, d_idx, d_idx_out, d_wval_out, 1);
}


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
    size_t d_scalars_idxs[NUM_BATCH_THREADS];
    size_t d_wval_idx;
    size_t d_idx_idx;
    size_t d_wval_out_idx;
    size_t d_idx_out_idx;
    size_t d_wval_count_idx;
    size_t d_wval_unique_idx;
    size_t d_wval_ptr_idx;
    size_t d_wval_unique_count_idx;
    size_t d_buckets_idx;
    size_t d_tres_idx;
    size_t d_tres_l_idx;
    size_t d_res_idx;

    size_t d_cub_sort_idx;
    size_t d_cub_flag_idx;
    size_t d_cub_encode_idx;
    size_t d_cub_sum_idx;

    scalar_t *h_scalars;

    typename pipp_t::result_container_t_faster fres0;
    typename pipp_t::result_container_t_faster fres1;
};

template<class bucket_t, class affine_t, class scalar_t>
struct RustContext {
    Context<bucket_t, affine_t, scalar_t> *context;
};

// Initialization function
// Allocate device storage, transfer bases
extern "C"
RustError mult_pippenger_faster_init(RustContext<bucket_t, affine_t, scalar_t> *context,
                              const affine_t points[], size_t npoints,
                              size_t ffi_affine_sz)
{
    context->context = new Context<bucket_t, affine_t, scalar_t>();
    Context<bucket_t, affine_t, scalar_t> *ctx = context->context;

    ctx->ffi_affine_sz = ffi_affine_sz;
    try {
        ctx->config = ctx->pipp.init_msm_faster(npoints);

        // Allocate GPU storage
        ctx->d_points_idx = ctx->pipp.allocate_d_bases(ctx->config);
        for (size_t i = 0; i < NUM_BATCH_THREADS; i++) {
            ctx->d_scalars_idxs[i] = ctx->pipp.allocate_d_scalars(ctx->config);
        }
        ctx->d_wval_idx = ctx->pipp.allocate_d_wval_faster(ctx->config);
        ctx->d_idx_idx = ctx->pipp.allocate_d_idx_faster(ctx->config);
        ctx->d_wval_out_idx = ctx->pipp.allocate_d_wval_faster(ctx->config);
        ctx->d_idx_out_idx = ctx->pipp.allocate_d_idx_faster(ctx->config);

        ctx->d_wval_count_idx = ctx->pipp.allocate_d_wval_count_faster();
        ctx->d_wval_unique_idx = ctx->pipp.allocate_d_wval_unique_faster();
        ctx->d_wval_ptr_idx = ctx->pipp.allocate_d_wval_ptr_faster();
        ctx->d_wval_unique_count_idx = ctx->pipp.allocate_d_wval_unique_count_faster();

        ctx->d_buckets_idx = ctx->pipp.allocate_d_buckets_faster();
        ctx->d_tres_idx = ctx->pipp.allocate_d_tres_faster(ctx->config);
        ctx->d_tres_l_idx = ctx->pipp.allocate_d_tres_faster(ctx->config);
        ctx->d_res_idx = ctx->pipp.allocate_d_res_faster();

        ctx->d_cub_sort_idx = ctx->pipp.allocate_d_cub_sort_faster(ctx->config);
        ctx->d_cub_flag_idx = ctx->pipp.allocate_d_cub_flag_faster(ctx->config);
        ctx->d_cub_encode_idx = ctx->pipp.allocate_d_cub_encode_faster(ctx->config);
        ctx->d_cub_sum_idx = ctx->pipp.allocate_d_cub_sum_faster();


        // Allocate pinned memory on host
        CUDA_OK(cudaMallocHost(&ctx->h_scalars, ctx->pipp.get_size_scalars(ctx->config)));
        
        ctx->pipp.transfer_bases_to_device(ctx->config, ctx->d_points_idx, points,
                                           ffi_affine_sz);
        
        ctx->fres0 = ctx->pipp.get_result_container_faster();
        ctx->fres1 = ctx->pipp.get_result_container_faster();
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
RustError mult_pippenger_faster_inf(RustContext<bucket_t, affine_t, scalar_t> *context,
                             point_t* out, const affine_t points[],
                             size_t npoints, size_t batches,
                             const scalar_t scalars[],
                             size_t ffi_affine_sz)
{
    (void)points; // Silence unused param warning
    
    Context<bucket_t, affine_t, scalar_t> *ctx = context->context;
    assert(ctx->config.npoints == npoints);
    assert(ctx->ffi_affine_sz == ffi_affine_sz);
    assert(batches > 0);

    cudaStream_t stream = ctx->pipp.default_stream;
    stream_t aux_stream(ctx->pipp.get_device());

    try {
        for (size_t i = 0; i < batches; i++) {
            out[i].inf();
        }

        typename pipp_t::result_container_t_faster *kernel_res = &ctx->fres0;
        typename pipp_t::result_container_t_faster *accum_res = &ctx->fres1;
        size_t d_scalars_xfer = ctx->d_scalars_idxs[0];
        size_t d_scalars_compute = ctx->d_scalars_idxs[1];
        
        channel_t<size_t> ch;
        size_t scalars_sz = ctx->pipp.get_size_scalars(ctx->config);

        int work = 0;
        ctx->pipp.transfer_scalars_to_device(ctx->config, d_scalars_compute,
                                             &scalars[work * npoints], aux_stream);
        CUDA_OK(cudaStreamSynchronize(aux_stream));


        for (; work < (int)batches; work++) {
            // Launch the GPU kernel, transfer the results back
            batch_pool.spawn([&]() {

                CUDA_OK(cudaStreamSynchronize(aux_stream));
                ctx->pipp.launch_kernel_faster_1(ctx->config, d_scalars_compute,
                                        ctx->d_buckets_idx, ctx->d_wval_idx, ctx->d_idx_idx);

                uint32_t *d_wval = ctx->pipp.d_wval_ptrs[ctx->d_wval_idx];
                uint32_t *d_wval_out = ctx->pipp.d_wval_ptrs[ctx->d_wval_out_idx];
                uint32_t *d_idx = ctx->pipp.d_idx_ptrs[ctx->d_idx_idx];
                uint32_t *d_idx_out = ctx->pipp.d_idx_ptrs[ctx->d_idx_out_idx];

                uint32_t nscalars = npoints;

                /*
                    [in]   wval: 'NWINS - 1' groups of sub-scalars. 
                    [in]   idx:  'NWINS - 1' groups of scalar indexes.
                    [out]  wval_out: 'NWINS - 1' groups of sorted sub-scalars. 
                        (allocated in the GPU global memory, and each group has 'nscalars' elements)
                    [out]  idx_out: 'NWINS - 1' groups of sorted scalar indexes. 
                        (allocated in the GPU global memory, and each group has 'nscalars' elements)
                */
                void *d_temp = NULL;
                size_t sort_size = 0;
                cub::DeviceRadixSort::SortPairs(d_temp, sort_size, d_wval, d_wval_out, d_idx, d_idx_out, nscalars, 0, 24, stream);  // Determine temporary device storage requirements
                void *d_cub_sort = (void *)ctx->pipp.d_cub_ptrs[ctx->d_cub_sort_idx];
                for(size_t k=0; k < NWINS - 1; k++)
                {
                    size_t ptr = k * nscalars;
                    cub::DeviceRadixSort::SortPairs(d_cub_sort, sort_size, d_wval + ptr, d_wval_out + ptr, d_idx + ptr, d_idx_out + ptr, nscalars, 0 ,24, stream);
                }

                uint32_t *d_wval_count = ctx->pipp.d_wval_ptrs[ctx->d_wval_count_idx];
                uint32_t *d_wval_unique = ctx->pipp.d_wval_ptrs[ctx->d_wval_unique_idx];
                uint32_t *d_wval_ptr = ctx->pipp.d_wval_ptrs[ctx->d_wval_ptr_idx];
                uint32_t *d_wval_unique_count = ctx->pipp.d_wval_ptrs[ctx->d_wval_unique_count_idx];

                /*
                    [in]  wval_out: 'NWINS - 1' groups of sorted sub-scalars. 
                    [out] wval_count: the number of sub-scalars with the same value in each group of sorted sub-scalars. 
                        (allocated in the GPU global memory with '(NWINS - 1) * (1 << WBITS)' elements)
                    [out] wval_unique: 'NWINS - 1' groups of unique sub-scalars.  
                        (allocated in the GPU global memory, and each group has (1 << WBITS) elements)
                    [out] wval_unique_count: the number of elements in each group of unique sub-scalars. 
                        (allocated in the GPU global memory with 'NWINS - 1' elements)
                */
                d_temp = NULL;
                size_t encode_size = 0;
                cub::DeviceRunLengthEncode::Encode(d_temp, encode_size, d_wval_out, d_wval_unique, d_wval_count, d_wval_unique_count, nscalars, stream);
                void *d_cub_encode = (void *)ctx->pipp.d_cub_ptrs[ctx->d_cub_encode_idx];
                for(uint32_t k=0; k < NWINS - 1; k++)
                {
                    uint32_t ptr = k * nscalars;
                    uint32_t cptr = k * (1 << WBITS);
                    cub::DeviceRunLengthEncode::Encode(d_cub_encode, encode_size, d_wval_out + ptr, d_wval_unique + cptr, d_wval_count + cptr, d_wval_unique_count + k, nscalars, stream);
                }

                /*
                    [in]  wval_count: the number of sub-scalars with the same value in each group of sorted sub-scalars.
                    [out] wval_ptr: the prefix sum of "wval_count".  
                        (allocated in the GPU global memory with '(NWINS - 1) * (1 << WBITS)' elements)
                */
                d_temp = NULL;
                size_t sum_size = 0;
                cub::DeviceScan::InclusiveSum(d_temp, sum_size, d_wval_count, d_wval_ptr, 1 << WBITS, stream);
                void *d_cub_sum = (void *)ctx->pipp.d_cub_ptrs[ctx->d_cub_sum_idx];
                for(uint32_t k=0; k < NWINS - 1; k++)
                {
                    uint32_t ptr = k * (1 << WBITS);
                    cub::DeviceScan::InclusiveSum(d_cub_sum, sum_size, d_wval_count + ptr, d_wval_ptr + ptr, 1 << WBITS, stream);
                }

                ctx->pipp.launch_kernel_faster_2(ctx->config, ctx->d_points_idx, ctx->d_buckets_idx, 
                                        ctx->d_wval_out_idx, ctx->d_idx_out_idx, 
                                        ctx->d_wval_count_idx, ctx->d_wval_unique_idx, ctx->d_wval_ptr_idx,
                                        ctx->d_wval_unique_count_idx);


                ctx->pipp.launch_kernel_faster_3(ctx->config, ctx->d_buckets_idx, ctx->d_tres_idx);

                /*
                    [in]   idx:  the last group of scalar indexes.
                    [in]   wval: the last group of sub-scalars. 
                    [out]  idx_out:  the corresponding scalar indexes to the selected sub-scalars.
                        (allocated in the GPU global memory with 'nscalars' elements)
                    [out]  s_count: the number of selected sub-scalars.
                */
                size_t ptr = (NWINS - 1) * nscalars;
                d_temp = NULL;
                size_t flag_size = 0;
                cub::DeviceSelect::Flagged(d_temp, flag_size, d_idx + ptr, d_wval + ptr, d_idx_out + ptr, d_wval_out + ptr, nscalars, stream);
                void *d_cub_flag = (void *)ctx->pipp.d_cub_ptrs[ctx->d_cub_flag_idx];
                cub::DeviceSelect::Flagged(d_cub_flag, flag_size, d_idx + ptr, d_wval + ptr, d_idx_out + ptr, d_wval_out + ptr, nscalars, stream);

                ctx->pipp.launch_kernel_faster_4(ctx->config, ctx->d_points_idx,
                            ctx->d_wval_out_idx, ctx->d_idx_out_idx, ctx->d_tres_l_idx);

                ctx->pipp.launch_kernel_faster_5(ctx->config, ctx->d_tres_idx, ctx->d_res_idx);
                ctx->pipp.launch_kernel_faster_6(ctx->config, ctx->d_tres_l_idx, ctx->d_res_idx);

                ctx->pipp.transfer_res_to_host_faster(*kernel_res, ctx->d_res_idx);
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
                    ctx->pipp.accumulate_faster(out[work - 1], *accum_res);
                }
                ch.send(work);
            });
            ch.recv();
            ch.recv();
            std::swap(kernel_res, accum_res);
            std::swap(d_scalars_xfer, d_scalars_compute);
        }

        // Accumulate the final result
        ctx->pipp.accumulate_faster(out[batches - 1], *accum_res);

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
