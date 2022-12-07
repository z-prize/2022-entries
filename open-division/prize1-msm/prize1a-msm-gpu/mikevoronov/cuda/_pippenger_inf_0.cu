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

static const size_t NUM_BATCH_THREADS = 1;
static thread_pool_t batch_pool(NUM_BATCH_THREADS);

typedef pippenger_t<bucket_t, point_t, affine_t, scalar_t> pipp_t;

// MSM context used store persistent state
template<class bucket_t, class affine_t, class scalar_t>
struct Context 
{
    pipp_t pipp;
    typename pipp_t::MSMConfig config;

    size_t ffi_affine_sz;
    size_t d_points_idx;
    size_t d_buckets_idx;
    size_t d_scalars_idxs[NUM_BATCH_THREADS];
    scalar_t *h_scalars;

    typename pipp_t::result_container_t res0;
    typename pipp_t::result_container_t res1;
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
		/*FILE* save = fopen("points_26.bin","wb");
		fwrite(points, sizeof(affine_t), npoints, save);
		fclose(save);		*/
		
        ctx->pipp.transfer_bases_to_device(ctx->config, ctx->d_points_idx, points, ffi_affine_sz);
		
		/*affine_t *points_c = new affine_t[npoints];
		FILE* save = fopen("points_26.bin","rb");
		fread(points_c, sizeof(affine_t), npoints, save);
		fclose(save);		
        ctx->pipp.transfer_bases_to_device(ctx->config, ctx->d_points_idx, points_c, ffi_affine_sz);
        delete []points_c;*/
		
        for (size_t i = 0; i < NUM_BATCH_THREADS; i++)
            ctx->d_scalars_idxs[i] = ctx->pipp.allocate_d_scalars(ctx->config);		
		ctx->pipp.allocate_tables(ctx->config);
		
        // Allocate pinned memory on host
        //CUDA_OK(cudaMallocHost(&ctx->h_scalars, ctx->pipp.get_size_scalars(ctx->config)));
        
        ctx->res0 = ctx->pipp.get_result_container(ctx->config);
        ctx->res1 = ctx->pipp.get_result_container(ctx->config);
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

int numLaunch = 0;
//scalar_t* scalars = NULL;
// Peform MSM on a batch of scalars over fixed bases
extern "C"
RustError mult_pippenger_inf(RustContext<bucket_t, affine_t, scalar_t> *context,
                             point_t* out, const affine_t points[],
                             size_t npoints, size_t batches,
                             const scalar_t scalars[],
                             size_t ffi_affine_sz)
{
	/*vector<set<int>> bIdx;
	for (int b = 0; b < 253; ++b)
	{
		set<int> group;
		for (int z = 0; z < npoints; ++z)
		{
			int d = get_wval((const unsigned*)(scalars + z), b, 1);
			if (d)
				group.insert(z);
		}
		bIdx.push_back(group);
		printf("%d -> %d\n", b, group.size());
	}	
	exit(0);*/
	
	/*if (scalars == NULL)
		CUDA_OK(cudaMallocHost(&scalars, sizeof(scalar_t) * npoints * batches));
	CUDA_OK(cudaMemcpy(scalars, scalars_in, sizeof(scalar_t) * npoints * batches, cudaMemcpyHostToHost));*/
	
	double totalT = omp_get_wtime();
	
	numLaunch++;
    (void)points; // Silence unused param warning
#if PRINT	
	printf("\n");
#endif
    Context<bucket_t, affine_t, scalar_t> *ctx = context->context;
    assert(ctx->config.npoints == npoints);
    assert(ctx->ffi_affine_sz == ffi_affine_sz);
    assert(batches > 0);

    cudaStream_t stream = ctx->pipp.default_stream;
    stream_t aux_stream(ctx->pipp.get_device());


    try {
        // Set results to infinity in case of failure along the way
        for (size_t i = 0; i < batches; i++)
            out[i].inf();

        typename pipp_t::result_container_t *kernel_res = &ctx->res0;
        typename pipp_t::result_container_t *accum_res = &ctx->res1;        
        size_t d_scalars_compute = ctx->d_scalars_idxs[0];
		size_t d_scalars_xfer = NUM_BATCH_THREADS > 1 ? ctx->d_scalars_idxs[1] : 0;
            
        channel_t<size_t> ch;
        size_t scalars_sz = ctx->pipp.get_size_scalars(ctx->config);

        // The following loop overlaps bucket computation on the GPU with transfer of the next set of scalars. 
		double t = omp_get_wtime();
		int countSubWorks = 1;
			
        ctx->pipp.transfer_scalars_to_device(d_scalars_compute, scalars, ctx->config.npoints / countSubWorks, 0);		
		//ctx->pipp.transfer_scalars_to_device_first(ctx->config, d_scalars_compute, ctx->h_scalars, scalars);
		t = omp_get_wtime() - t;
		//printf("first copy %f ms \n", 1000.0*t);

		int bucketsDone = 0;
		int allDone = 0;
        for (int work = 0; work < (int)batches; work++) 
		{
			bool isLast = (batches - 1) == work;
			
			if (work == 0 && countSubWorks > 1)
			{
				unsigned part = npoints / countSubWorks;
				for (int k = 0; k < countSubWorks; ++k)
				{
					bucketsDone = 0;
					bool last = (k == (countSubWorks - 1));
					ctx->pipp.launch_best(work, bucketsDone, allDone, out, ctx->config, ctx->d_points_idx, d_scalars_compute, d_scalars_xfer, ctx->h_scalars, 
										  scalars + npoints + k * part, (last ? NULL : scalars + (k + 1) * part),
										  false, k * part, part, last);
				}

				/*bucketsDone = 0;
				ctx->pipp.launch_best(work, bucketsDone, allDone, out, ctx->config, ctx->d_points_idx, d_scalars_compute, d_scalars_xfer, ctx->h_scalars, 
									  NULL, //scalars + (work + 1) * npoints, 
									  true, 0, 0);
				exit(0);*/
			}
			else
			{
				
				ctx->pipp.launch_best(work, bucketsDone, allDone, out, ctx->config, ctx->d_points_idx, d_scalars_compute, d_scalars_xfer, ctx->h_scalars, 
									  isLast ? NULL : scalars + (work + 1) * npoints, NULL, 
									  true, 0, 0);
			}
            std::swap(d_scalars_xfer, d_scalars_compute);
        }

        // Accumulate the final result
		t = omp_get_wtime();

		//ctx->pipp.accumulate_(batches - 1, out[batches - 1]);
		//ctx->pipp.intergateAsync(0, 4 * NWINS);
		ctx->pipp.waitAll();
		t = omp_get_wtime() - t;
		
		double t1 = omp_get_wtime();
		ctx->pipp.accumulateAsync(out, 0, batches);
		t1 = omp_get_wtime() - t1;
#if PRINT			
		printf("last acc %f ms, wait %f ms \n", 1000.0*t1, 1000.0*t);
#endif
		
    } catch (const cuda_error& e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()}
#endif
    }
	
	totalT = omp_get_wtime() - totalT;
#if 1
//PRINT		
	printf("== total batch time %f ms, avg %f ms\n", totalT * 1000., totalT * 1000. / batches);
#endif
	//CUDA_OK(cudaFreeHost(scalars));
	
	if (numLaunch == 2)
		exit(0);
    return RustError{cudaSuccess};

}

#endif  //  __CUDA_ARCH__
