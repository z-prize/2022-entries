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
#include <algorithm>

#ifndef __CUDA_ARCH__

typedef pippenger_t<bucket_t, point_t, affine_t, scalar_t> pipp_t;

// MSM context used store persistent state
template<class bucket_t, class affine_t, class scalar_t>
struct Context 
{
    pipp_t pipp;
    typename pipp_t::MSMConfig config;

    size_t ffi_affine_sz;
    size_t d_points_idx;
    size_t d_scalars_idxs;
};

template<class bucket_t, class affine_t, class scalar_t>
struct RustContext 
{
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
    try 
	{
        ctx->config = ctx->pipp.init_msm(npoints);

        // Allocate GPU storage
        ctx->d_points_idx = ctx->pipp.allocate_d_bases(ctx->config);
		
        ctx->pipp.transfer_bases_to_device(ctx->config, ctx->d_points_idx, points, ffi_affine_sz);
		
        ctx->d_scalars_idxs = ctx->pipp.allocate_d_scalars(ctx->config);		
		ctx->pipp.allocate_tables(ctx->config);
    }
	catch (const cuda_error& e) 
	{
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()}
#endif
    }
    return RustError{cudaSuccess};
}

static int numLaunch = 0;
// Peform MSM on a batch of scalars over fixed bases
extern "C"
RustError mult_pippenger_inf(RustContext<bucket_t, affine_t, scalar_t> *context,
                             point_t* out, const affine_t points[],
                             size_t npoints, size_t batches,
                             const scalar_t scalars[],
                             size_t ffi_affine_sz)
{
	double totalT = omp_get_wtime();

	numLaunch++;
    (void)points; // Silence unused param warning

    Context<bucket_t, affine_t, scalar_t> *ctx = context->context;
    assert(ctx->config.npoints == npoints);
    assert(ctx->ffi_affine_sz == ffi_affine_sz);
    assert(batches > 0);


    try {
        // Set results to infinity in case of failure along the way
        for (size_t i = 0; i < batches; ++i)
            out[i].inf();

        const size_t d_scalars_compute = ctx->d_scalars_idxs;        			
        ctx->pipp.transfer_scalars_to_device(d_scalars_compute, scalars, ctx->config.npoints, 0);		
				
		int bucketsDone = 0, allDone = 0;
        for (int work = 0; work < (int)batches; work++) 
		{
			bool isLast = ((batches - 1) == work);
			ctx->pipp.launch_best(work, bucketsDone, allDone, out, ctx->config, ctx->d_points_idx, d_scalars_compute, (isLast ? NULL : scalars + (work + 1) * npoints));
        }

        // Accumulate the final result
		ctx->pipp.waitAll();
		ctx->pipp.accumulateAsync(out, 0, batches);
		
    } catch (const cuda_error& e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()}
#endif
    }
	
	totalT = omp_get_wtime() - totalT;
	printf("== total batch time %f ms, avg %f ms\n", totalT * 1000., totalT * 1000. / batches);

	if (numLaunch == 2)
		exit(0);
    return RustError{cudaSuccess};
}

#endif  //  __CUDA_ARCH__
