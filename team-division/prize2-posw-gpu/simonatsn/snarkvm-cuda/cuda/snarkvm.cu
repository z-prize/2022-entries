// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>

#if defined(FEATURE_BLS12_381)
# include <ff/bls12-381.hpp>
#elif defined(FEATURE_BLS12_377)
# include <ff/bls12-377.hpp>
#elif defined(FEATURE_BN254)
# include <ff/alt_bn128.hpp>
#else
# error "no FEATURE"
#endif

#include "../ntt-cuda/ntt/ntt.cuh"
#include "../poly-cuda/polynomial.cuh"

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_inf_t affine_t;
typedef fr_t scalar_t;
typedef bucket_t::affine_t affine_noinf_t;

#include <msm/pippenger.cuh>
#include "../msm-cuda/export.h"
#include "compute_t_poly.cu"
#include "compute_matrix_sumcheck.cu"

__global__
void compute_r_alpha_x_evals(fr_t* out, fr_t* vanish_x, fr_t* evals, uint32_t ratio, fr_t* denoms) {
    index_t idx = threadIdx.x + blockDim.x * (index_t)blockIdx.x;
    fr_t num = *vanish_x;
    int phase = idx % ratio;
    if (phase != 0) {
        num = num - evals[phase];
    }
    //out[idx] = num;
    fr_t denom = denoms[idx];
    out[idx] = num * denom;
}

// A simple way to allocate a host pointer without having to
// care about freeing it.
template<typename T> class host_ptr_t {
    T* h_ptr;
public:
    host_ptr_t(size_t nelems) : h_ptr(nullptr)
    {
        if (nelems) {
            CUDA_OK(cudaMallocHost(&h_ptr, nelems * sizeof(T)));
        }
    }
    ~host_ptr_t() { if (h_ptr) cudaFreeHost((void*)h_ptr); }

    inline operator const T*() const            { return h_ptr; }
    inline operator T*() const                  { return h_ptr; }
    inline operator void*() const               { return (void*)h_ptr; }
    inline const T& operator[](size_t i) const  { return h_ptr[i]; }
    inline T& operator[](size_t i)              { return h_ptr[i]; }
};


#ifndef __CUDA_ARCH__
#include <vector>
#include <chrono>
#include <unistd.h>
#include <atomic>

typedef std::chrono::high_resolution_clock Clock;

using namespace std;

class snarkvm_t {
public:
    size_t max_lg_domain;
    size_t max_lg_blowup;

    struct resource_t {
        int dev;
        int stream;
        resource_t(int _dev, int _stream) {
            dev = _dev;
            stream = _stream;
        }
    };
    channel_t<resource_t*> resources;
    uint32_t num_gpus;

    // Memory will be structured as:
    //     GPU          Host
    //     stream0      stream0
    //       MSM          MSM
    //       Poly         Poly
    //     stream1      stream1
    //       MSM          MSM
    //       Poly         Poly
    //     batch_eval
    std::vector<dev_ptr_t<fr_t>*> d_mem;
    std::vector<host_ptr_t<fr_t>*> h_mem;
    size_t d_msm_elements_per_stream;
    size_t h_msm_elements_per_stream;
    size_t d_poly_elements_per_stream;
    size_t h_poly_elements_per_stream;

    fr_t *d_addr_msm(uint32_t dev, uint32_t stream) {
        fr_t* dmem = *d_mem[dev];
        return &dmem[(d_msm_elements_per_stream + d_poly_elements_per_stream) * stream];
    }
    fr_t *d_addr_poly(uint32_t dev, uint32_t stream) {
        fr_t* dmem = *d_mem[dev];
        return &dmem[(d_msm_elements_per_stream + d_poly_elements_per_stream) * stream +
                     d_msm_elements_per_stream];
    }
    fr_t *h_addr_msm(uint32_t dev, uint32_t stream) {
        fr_t* hmem = *h_mem[dev];
        return &hmem[(h_msm_elements_per_stream + h_poly_elements_per_stream) * stream];
    }
    fr_t *h_addr_poly(uint32_t dev, uint32_t stream) {
        fr_t* hmem = *h_mem[dev];
        return &hmem[(h_msm_elements_per_stream + h_poly_elements_per_stream) * stream +
                     h_msm_elements_per_stream];
    }

    // Cache for MSM bases.
    static const size_t msm_cache_entries = 4;
    // Key to cached msm_t mapping
    uint64_t msm_keys[msm_cache_entries];
    // Max number of points to support
    static const size_t msm_cache_npoints = 65537;
    // Current cache entry
    size_t cur_msm_cache_entry;
    struct dev_msm_point_cache_t {
        dev_ptr_t<affine_noinf_t>* cache[msm_cache_entries];
    };
    std::vector<dev_msm_point_cache_t> msm_point_cache;
    

    // Cache batch_eval_unnormalized_bivariate_lagrange_poly_with_diff_inputs_over_domain values
    size_t batch_eval_ratio = 4; // mul_domain / constraint_domain
    size_t d_batch_eval_domain = 131072;
    std::vector<fr_t*> d_batch_eval;

    // MSM kernel context, per device and stream
    struct msm_cuda_ctx_t {
        void* ctxs[gpu_t::FLIP_FLOP];
    };
    std::vector<msm_cuda_ctx_t> msm_cuda_ctxs;

    // Cache for poly_t inputs - device pointers
    std::vector<poly_t_cache_t> poly_t_cache;
    // Cache for arithmetization
    std::vector<arith_cache_t> arith_cache;

    snarkvm_t() {
        max_lg_domain = 17;
        size_t domain_size = (size_t)1 << max_lg_domain;
        size_t ext_domain_size = domain_size;

        // Set up MSM kernel context
        // Set up polynomial division kernel
        num_gpus = 1;
        msm_cuda_ctxs.resize(num_gpus);
        for (size_t j = 0; j < gpu_t::FLIP_FLOP; j++) {
            for (size_t dev = 0; dev < num_gpus; dev++) {
                auto &gpu = select_gpu(dev);
                msm_cuda_ctxs[dev].ctxs[j] = msm_cuda_create_context(msm_cache_npoints, gpu.sm_count());
                resources.send(new resource_t(dev, j));
            }
        }

        // The msm library needs storage internally:
        //   gpu - buckets + partial sums
        //   cpu - partial sums
        size_t msm_cuda_host_bytes;
        size_t msm_cuda_gpu_bytes;
        msm_cuda_storage(msm_cuda_ctxs[0].ctxs[0], &msm_cuda_host_bytes, &msm_cuda_gpu_bytes);

        // For MSM we need additional space for scalars. Points are cached.
        msm_cuda_gpu_bytes += msm_cache_npoints * sizeof(fr_t);
        msm_cuda_host_bytes += msm_cache_npoints * (sizeof(fr_t) + sizeof(affine_t));

        // Ensure GPU/CPU storage for either kernel.
        size_t msm_gpu_elements = (msm_cuda_gpu_bytes + sizeof(fr_t) - 1) / sizeof(fr_t);
        size_t msm_host_elements = (msm_cuda_host_bytes + sizeof(fr_t) - 1) / sizeof(fr_t);
        
        // Determine storage needed for polynomial operations
        size_t num_elements = ext_domain_size + domain_size;
        // Storage to operate on up to 5 polynomials at a time
        num_elements *= 5;
        
        d_poly_elements_per_stream = num_elements;
        h_poly_elements_per_stream = num_elements;
        d_msm_elements_per_stream = msm_gpu_elements;
        h_msm_elements_per_stream = msm_host_elements;

        size_t elements_per_stream_gpu = d_poly_elements_per_stream + d_msm_elements_per_stream;
        size_t elements_per_stream_cpu = h_poly_elements_per_stream + h_msm_elements_per_stream;

        // Determine storage needed per device, and host side per device
        size_t elements_per_dev = elements_per_stream_gpu * gpu_t::FLIP_FLOP;
        size_t elements_per_dev_cpu = elements_per_stream_cpu * gpu_t::FLIP_FLOP;
        // Add batch eval values on top since we need to preserve those (per device)
        elements_per_dev += batch_eval_ratio;
        
        d_mem.resize(num_gpus);
        h_mem.resize(num_gpus);
        d_batch_eval.resize(num_gpus);

        for (size_t dev = 0; dev < num_gpus; dev++) {
            auto &gpu = select_gpu(dev);
            d_mem[dev] = new dev_ptr_t<fr_t>(elements_per_dev);
            h_mem[dev] = new host_ptr_t<fr_t>(elements_per_dev_cpu);
            // Batch evals sit at the end of the memory allocation
            d_batch_eval[dev] = &(*d_mem[dev])[elements_per_dev - batch_eval_ratio];
            gpu.sync();
        }

        // Set up the MSM cache
        cur_msm_cache_entry = 0;
        msm_point_cache.resize(num_gpus);
        for (size_t i = 0; i < num_gpus; i++) {
            for (size_t j = 0; j < msm_cache_entries; j++) {
                msm_point_cache[i].cache[j] = nullptr;
            }
        }
        
        // For batch_eval_unnormalized_bivariate_lagrange_poly_with_diff_inputs_over_domain,
        // the pattern of subtractions depends the ratio of the domain sizes
        // ratio = 4
        // pow((g**4)%p, 131072//ratio, p)-1
        fr_t g("e805156baeaf0c35750f3f45a4e8d566148cb16a8de9973f8522249260678ff");
        vector<fr_t> evals;
        evals.resize(batch_eval_ratio);
        // Populate initial values
        evals[0] = fr_t::one();
        evals[1] = g;
        for (size_t i = 2; i < batch_eval_ratio; i++) {
            evals[i] = evals[i - 1] * g;
        }
        // Exponentiate
        fr_t one = fr_t::one();
        for (size_t i = 0; i < batch_eval_ratio; i++) {
            evals[i] = (evals[i] ^ (d_batch_eval_domain / 4)) - one;
        }
        for (size_t dev = 0; dev < num_gpus; dev++) {
            auto &gpu = select_gpu(dev);
            gpu.HtoD(d_batch_eval[dev], &evals[0], batch_eval_ratio);
            gpu.sync();
        }
    }
    ~snarkvm_t() {
        for (size_t dev = 0; dev < num_gpus; dev++) {
            select_gpu(dev);
            delete d_mem[dev];
            delete h_mem[dev];

            // Free MSM caches
            for (size_t i = 0; i < cur_msm_cache_entry; i++) {
                delete msm_point_cache[dev].cache[i];
            }
            //msm_cuda_delete_context(msm_cuda_ctx);
        }
    }

private:
    // a * b - c * d
    // a in evaluation form in mul domain
    // constraint domain 32768
    // mul domain 131072
    RustError CalculateLHS_internal(const gpu_t& gpu, stream_t& stream,
                                    fr_t* dmem,
                                    fr_t* out,
                                    fr_t* d_evals,
                                    fr_t* vanish_x,
                                    fr_t* denoms, size_t denoms_len,
                                    fr_t* b, size_t b_len,
                                    fr_t* c, size_t c_len,
                                    fr_t* d, size_t d_len,
                                    uint32_t lg_domain_size,
                                    uint32_t lg_ext_domain_size) {
        try {
            gpu.select();

            size_t domain_size = (size_t)1 << lg_domain_size;
            size_t ext_domain_size = (size_t)1 << lg_ext_domain_size;

            fr_t* d_a = dmem;
            fr_t* d_b = &dmem[ext_domain_size];
            fr_t* d_c = &dmem[ext_domain_size * 2];
            fr_t* d_d = &dmem[ext_domain_size * 3];
            fr_t* d_out = &dmem[ext_domain_size * 4];
            fr_t* d_vanish_x = d_b; // Temporarily us b for vanish_x

            // Copy the input data
            stream.HtoD(d_a, denoms, denoms_len);
            stream.HtoD(d_vanish_x, vanish_x, 1);

            // First compute the a value (r_alpha_x_evals)
            // - Create numerators array size 131072
            //   - if i%ratio == 0: vanish_x
            //   - else: vanish_x - evals[i % 4]
            // - denom *= numerator
            compute_r_alpha_x_evals<<<ext_domain_size / 1024, 1024, 0, stream>>>
                (d_a, d_vanish_x, d_evals,
                 1 << (lg_ext_domain_size - lg_domain_size),
                 d_a);

            // Copy the rest of the data now that the buffers are available
            cudaMemsetAsync(d_b, 0, sizeof(fr_t) * ext_domain_size * 3, stream);
            stream.HtoD(d_b, b, b_len);
            stream.HtoD(d_c, c, c_len);
            stream.HtoD(d_d, d, d_len);
            
            // Perform NTT on the input data
            NTT::NTT_device(d_b, lg_ext_domain_size,
                            NTT::InputOutputOrder::NR, NTT::Direction::forward, stream);
            NTT::NTT_device(d_c, lg_ext_domain_size,
                            NTT::InputOutputOrder::NR, NTT::Direction::forward, stream);
            NTT::NTT_device(d_d, lg_ext_domain_size,
                            NTT::InputOutputOrder::NR, NTT::Direction::forward, stream);

            // Bit reverse a so it's out of order
            // Requires an aux buffer so use out
            NTT::bit_rev(d_out, d_a, lg_ext_domain_size, stream);

            // a * b
            polynomial_inner_multiply<<<ext_domain_size / 1024, 1024, 0, stream>>>
                (d_out, d_out, d_b);
            // c * d
            polynomial_inner_multiply<<<ext_domain_size / 1024, 1024, 0, stream>>>
                (d_c, d_c, d_d);
            // a * b - c * d
            polynomial_sub<<<ext_domain_size / 1024, 1024, 0, stream>>>
                (d_out, d_out, d_c);

            // Perform iNTT on the result
            NTT::NTT_device(d_out, lg_ext_domain_size,
                            NTT::InputOutputOrder::RN, NTT::Direction::inverse, stream);
            
            // Copy the output data
            stream.DtoH(out, d_out, ext_domain_size);
            stream.sync();
        } catch (const cuda_error& e) {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};
        
    }

    // Inputs: z_a(32768), z_b(32768), eta_c(1), eta_b_over_eta_c:(1)
    // Outputs: summed_z_m(65536)
    // z_b  = z_b:P * eta_c:F
    // z_b += 1
    // summed_z_m = z_a:P * eta_c_z_b_plus_one:P - Increases domain by one
    // z_b -= 1
    // summed_z_m += eta_b_over_eta_c:F * z_b:P
    RustError CalculateSummedZM_internal(const gpu_t& gpu, stream_t& stream,
                                         fr_t* dmem, fr_t* out, 
                                         fr_t* z_a,
                                         fr_t* z_b,
                                         fr_t* eta_c,
                                         fr_t* eta_b_over_eta_c,
                                         uint32_t lg_domain_size) {
        try {
            gpu.select();

            size_t domain_size = (size_t)1 << lg_domain_size;
            size_t ext_domain_size = domain_size * 2;

            // z_a and z_b in ext domain since we will expand them
            fr_t* d_z_a = dmem;
            fr_t* d_z_b = &dmem[ext_domain_size];
            fr_t* d_eta_c = &dmem[ext_domain_size * 2];
            fr_t* d_eta_b_over_eta_c = &dmem[ext_domain_size * 2 + 1];
            fr_t* d_out = &dmem[ext_domain_size * 2 + 2];

            // Clear the inputs to the extended domain
            cudaMemsetAsync(d_z_a, 0, sizeof(fr_t) * ext_domain_size * 1, stream);
            // And output as well, since we stage NTT data there
            cudaMemsetAsync(d_out, 0, sizeof(fr_t) * ext_domain_size, stream);
            
            // Copy the input data
            stream.HtoD(d_z_a, z_a, domain_size);
            stream.HtoD(d_z_b, z_b, domain_size);
            stream.HtoD(d_eta_c, eta_c, 1);
            stream.HtoD(d_eta_b_over_eta_c, eta_b_over_eta_c, 1);

            // z_b  = z_b:P * eta_c:F
            polynomial_scale<<<domain_size / 1024, 1024, 0, stream>>>
                (d_z_b, d_z_b, d_eta_c);

            // z_b += 1
            polynomial_incr<<<1, 1, 0, stream>>>(d_z_b, d_z_b);
            
            // summed_z_m = z_a:P * eta_c_z_b_plus_one:P - Increases domain by one
            cudaMemcpyAsync(d_out, d_z_b, sizeof(fr_t) * domain_size,
                            cudaMemcpyDeviceToDevice, stream);
            Polynomial::MulDev(stream, d_out, d_z_a, d_out, lg_domain_size);

            // z_b -= 1
            polynomial_decr<<<1, 1, 0, stream>>>(d_z_b, d_z_b);

            // summed_z_m += eta_b_over_eta_c:F * z_b:P
            polynomial_scale<<<domain_size / 1024, 1024, 0, stream>>>
                (d_z_b, d_z_b, d_eta_b_over_eta_c);
            polynomial_add<<<domain_size / 1024, 1024, 0, stream>>>
                (d_out, d_out, d_z_b);
            
            // Copy the output data
            stream.DtoH(out, d_out, ext_domain_size);
            stream.sync();
        } catch (const cuda_error& e) {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};
    }

    static const size_t FP_BYTES = 48;
    struct rust_p1_affine_t {
        uint8_t x[FP_BYTES];
        uint8_t y[FP_BYTES];
        uint8_t inf;
        uint8_t pad[7];
    };

    // Fill the cache
    // - npoints - Number of points to copy into the staging buffer
    // - max_cache - Will be used for precomp stride. For simplicity all caches
    //               use the same value.
    void populate_msm_points(int dev, size_t npoints, size_t max_cache,
                             const rust_p1_affine_t* points,
                             affine_noinf_t *h_points, affine_noinf_t *d_points,
                             size_t ffi_affine_sz) {
        auto& gpu = select_gpu(dev);
        // Copy bases, omitting infinity
        assert(sizeof(rust_p1_affine_t) == ffi_affine_sz);
        for (unsigned i = 0; i < npoints; i++) {
            memcpy((uint8_t*)&h_points[i], (uint8_t*)&points[i], sizeof(affine_noinf_t));
        }
        gpu.HtoD(d_points, h_points, max_cache);
        gpu.sync();

        msm_cuda_precompute_bases(msm_cuda_ctxs[dev].ctxs[0], max_cache, d_points, d_points);
    }

public:
    RustError NTT(fr_t* inout,
                  uint32_t lg_domain_size,
                  NTT::InputOutputOrder ntt_order,
                  NTT::Direction ntt_direction) {
        assert(lg_domain_size <= max_lg_domain);
        resource_t *resource = resources.recv();
        int dev = resource->dev;
        auto& gpu = select_gpu(dev);
        int stream_idx = resource->stream;
        stream_t& stream = gpu[stream_idx];

        // Copy data to pinned staging buffer
        size_t domain_size = (size_t)1 << lg_domain_size;
        fr_t* h_inout = h_addr_poly(dev, stream_idx);
        memcpy(h_inout, inout, sizeof(fr_t) * domain_size);

        fr_t* d_poly = d_addr_poly(dev, stream_idx);

        RustError ret = RustError{cudaSuccess};
        try {
            stream.HtoD(d_poly, h_inout, domain_size);
            
            NTT::NTT_device(d_poly, lg_domain_size,
                            ntt_order, ntt_direction, stream);
            stream.DtoH(h_inout, d_poly, domain_size);
            stream.sync();
            memcpy(inout, h_inout, sizeof(fr_t) * domain_size);
        } catch (const cuda_error& e) {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            ret = RustError{e.code(), e.what()};
#else
            ret = RustError{e.code()};
#endif
        }
            
        resources.send(resource);
        return ret;
    }

    RustError PolyMul(fr_t* out, fr_t* in0, fr_t* in1, uint32_t lg_domain_size) {
        resource_t *resource = resources.recv();
        int dev = resource->dev;
        auto& gpu = select_gpu(dev);
        int stream_idx = resource->stream;
        stream_t& stream = gpu[stream_idx];

        // Copy data to pinned staging buffer
        size_t domain_size = (size_t)1 << lg_domain_size;
        fr_t* h_in0 = h_addr_poly(dev, stream_idx);
        fr_t* h_in1 = &h_in0[domain_size];
        memcpy(h_in0, in0, sizeof(fr_t) * domain_size);
        memcpy(h_in1, in1, sizeof(fr_t) * domain_size);
        
        fr_t* d_poly = d_addr_poly(dev, stream_idx);
        RustError e = Polynomial::Mul(gpu, stream, d_poly, h_in0, h_in0, h_in1, lg_domain_size);
        if (e.code != cudaSuccess) {
            resources.send(resource);
            return e;
        }
        memcpy(out, h_in0, sizeof(fr_t) * domain_size * 2);

        resources.send(resource);
        return RustError{cudaSuccess};
    }

    RustError CalculateSummedZM(fr_t* out,
                                fr_t* z_a,
                                fr_t* z_b,
                                fr_t* eta_c,
                                fr_t* eta_b_over_eta_c,
                                uint32_t lg_domain_size) {
        resource_t *resource = resources.recv();
        int dev = resource->dev;
        auto& gpu = select_gpu(dev);
        int stream_idx = resource->stream;
        stream_t& stream = gpu[stream_idx];

        // Copy data to pinned staging buffer
        size_t domain_size = (size_t)1 << lg_domain_size;
        size_t ext_domain_size = domain_size * 2;
        //fr_t* h_z_a = *h_mem[dev];
        fr_t* h_z_a = h_addr_poly(dev, stream_idx);
        fr_t* h_z_b = &h_z_a[domain_size];
        memcpy(h_z_a, z_a, sizeof(fr_t) * domain_size);
        memcpy(h_z_b, z_b, sizeof(fr_t) * domain_size);

        // z_b will contain the result
        fr_t* d_poly = d_addr_poly(dev, stream_idx);
        RustError e = CalculateSummedZM_internal(gpu, stream, d_poly,
                                                 h_z_b, h_z_a, h_z_b,
                                                 eta_c, eta_b_over_eta_c,
                                                 lg_domain_size);
        memcpy(out, h_z_b, sizeof(fr_t) * ext_domain_size);

        resources.send(resource);
        return e;
    }

    size_t count_trailing_val(size_t len, fr_t* a, const fr_t &val) {
        size_t count = 0;

        for (size_t i = len - 1; i > 0; i--) {
            if(a[i] == val) {
                    count++;
            } else {
                break;
            }
        }
        return count;
    }

    void prepare_arith_cache(arith_cache_arr_t &cache, const char* name, size_t len,
                             fr_t* arr, fr_t* d_arr) {
        fr_t zero;
        zero.zero();
        size_t trailing_zeros = count_trailing_val(len, arr, zero);
        size_t trailing_ones = count_trailing_val(len, arr, fr_t::one());

        cache.skip_zeros = len - trailing_zeros;
        cache.skip_ones = len - trailing_ones;
        cache.d_val = d_arr;
        cache.d_evals = nullptr;
        
        // Copy the array data to GPU
        cudaMemcpy(d_arr, arr, sizeof(fr_t) * len, cudaMemcpyHostToDevice);
    }
    void prepare_arith_evals(arith_cache_arr_t &cache, fr_t* d_evals, size_t lg_domain, stream_t& stream) {
        // We get the evaluations, so need to first iNTT 
        size_t domain = (size_t)1 << lg_domain;
        cudaMemcpyAsync(d_evals, cache.d_val, domain * sizeof(fr_t), cudaMemcpyDeviceToDevice, stream);
        NTT::NTT_device(d_evals, lg_domain,
                        NTT::InputOutputOrder::NN, NTT::Direction::inverse, stream);
        cudaMemsetAsync(&d_evals[domain], 0, domain * sizeof(fr_t), stream);
        // Leave in bit reversed order
        NTT::NTT_device(d_evals, lg_domain + 1,
                        NTT::InputOutputOrder::NR, NTT::Direction::forward, stream);
        cache.d_evals = d_evals;
    }
    
    void CachePolyTInputs(size_t a_len, uint32_t* a_r, uint32_t* a_c, fr_t* a_coeff,
                          size_t b_len, uint32_t* b_r, uint32_t* b_c, fr_t* b_coeff,
                          size_t c_len, uint32_t* c_r, uint32_t* c_c, fr_t* c_coeff,

                          fr_t* a_arith_row_on_k,
                          fr_t* a_arith_col_on_k,
                          fr_t* a_arith_row_col_on_k,
                          fr_t* a_arith_val,
                          fr_t* a_arith_evals_on_k,
                          
                          fr_t* b_arith_row_on_k,
                          fr_t* b_arith_col_on_k,
                          fr_t* b_arith_row_col_on_k,
                          fr_t* b_arith_val,
                          fr_t* b_arith_evals_on_k,
                          
                          fr_t* c_arith_row_on_k,
                          fr_t* c_arith_col_on_k,
                          fr_t* c_arith_row_col_on_k,
                          fr_t* c_arith_val,
                          fr_t* c_arith_evals_on_k
                          ) {
        const size_t lg_arith_domain_size = 16;
        size_t arith_domain_size = (size_t)1 << lg_arith_domain_size;

        poly_t_cache.resize(num_gpus);

        compute_poly_t_work(poly_t_cache[0],
                            a_len, a_r, a_c, a_coeff,
                            b_len, b_r, b_c, b_coeff,
                            c_len, c_r, c_c, c_coeff);

        uint32_t constraint_domain_size = 1 << poly_t_cache[0].lg_constraint_domain_size;

        // Copy params and host pointers
        for (size_t i = 0; i < num_gpus; i++) {
            poly_t_cache[i] = poly_t_cache[0];
        }
        // Push data to GPU
        for (size_t i = 0; i < num_gpus; i++) {
            auto &gpu = select_gpu(i);
            cudaMalloc(&poly_t_cache[i].d_coeffs, sizeof(fr_t) * poly_t_cache[i].work_len);
            cudaMalloc(&poly_t_cache[i].d_work,   sizeof(poly_t_work_item_t) * poly_t_cache[i].work_len);
            cudaMalloc(&poly_t_cache[i].d_tasks,  sizeof(poly_t_idx_task_t) * poly_t_cache[i].task_count);
            cudaMalloc(&poly_t_cache[i].d_final_sum_map,
                       sizeof(poly_t_final_sum_map_t) * poly_t_cache[i].task_count);
            cudaMalloc(&poly_t_cache[i].d_final_start_count,
                       sizeof(poly_t_final_start_count_t) * constraint_domain_size);

            cudaMemcpy(poly_t_cache[i].d_coeffs, poly_t_cache[i].coeffs,
                       sizeof(fr_t) * poly_t_cache[i].work_len, cudaMemcpyHostToDevice);
            cudaMemcpy(poly_t_cache[i].d_work, poly_t_cache[i].work,
                       sizeof(poly_t_work_item_t) * poly_t_cache[i].work_len, cudaMemcpyHostToDevice);
            cudaMemcpy(poly_t_cache[i].d_tasks, poly_t_cache[i].tasks,
                       sizeof(poly_t_idx_task_t) * poly_t_cache[i].task_count, cudaMemcpyHostToDevice);
            cudaMemcpy(poly_t_cache[i].d_final_sum_map, poly_t_cache[i].final_sum_map,
                       sizeof(poly_t_final_sum_map_t) * poly_t_cache[i].task_count, cudaMemcpyHostToDevice);
            cudaMemcpy(poly_t_cache[i].d_final_start_count, poly_t_cache[i].final_start_count,
                       sizeof(poly_t_final_start_count_t) * constraint_domain_size, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
        }

        // Arithmetization cache
        size_t elements_per_dev = (5 * 3 * arith_domain_size +
                                   // row, col, val in ext domain eval form
                                   3 * 3 * arith_domain_size * 2);
        arith_cache.resize(num_gpus);
        for (size_t i = 0; i < num_gpus; i++) {
            auto &gpu = select_gpu(i);

            arith_cache_t& cache = arith_cache[i];
            
            dev_ptr_t<fr_t>* d_dev_ptr = new dev_ptr_t<fr_t>(elements_per_dev);
            fr_t* d_ptr = *d_dev_ptr;
            size_t idx = 0;
            
            prepare_arith_cache(cache.vars[ARITH_A].row_on_k,     "a_arith_row_on_k",     arith_domain_size, a_arith_row_on_k,     &d_ptr[idx * arith_domain_size]); idx++;
            prepare_arith_cache(cache.vars[ARITH_A].col_on_k,     "a_arith_col_on_k",     arith_domain_size, a_arith_col_on_k,     &d_ptr[idx * arith_domain_size]); idx++;
            prepare_arith_cache(cache.vars[ARITH_A].row_col_on_k, "a_arith_row_col_on_k", arith_domain_size, a_arith_row_col_on_k, &d_ptr[idx * arith_domain_size]); idx++;
            prepare_arith_cache(cache.vars[ARITH_A].val,          "a_arith_val",          arith_domain_size, a_arith_val,          &d_ptr[idx * arith_domain_size]); idx++;
            prepare_arith_cache(cache.vars[ARITH_A].evals_on_k,   "a_arith_evals_on_k",   arith_domain_size, a_arith_evals_on_k,   &d_ptr[idx * arith_domain_size]); idx++;
            
            prepare_arith_cache(cache.vars[ARITH_B].row_on_k,     "b_arith_row_on_k",     arith_domain_size, b_arith_row_on_k,     &d_ptr[idx * arith_domain_size]); idx++;
            prepare_arith_cache(cache.vars[ARITH_B].col_on_k,     "b_arith_col_on_k",     arith_domain_size, b_arith_col_on_k,     &d_ptr[idx * arith_domain_size]); idx++;
            prepare_arith_cache(cache.vars[ARITH_B].row_col_on_k, "b_arith_row_col_on_k", arith_domain_size, b_arith_row_col_on_k, &d_ptr[idx * arith_domain_size]); idx++;
            prepare_arith_cache(cache.vars[ARITH_B].val,          "b_arith_val",          arith_domain_size, b_arith_val,          &d_ptr[idx * arith_domain_size]); idx++;
            prepare_arith_cache(cache.vars[ARITH_B].evals_on_k,   "b_arith_evals_on_k",   arith_domain_size, b_arith_evals_on_k,   &d_ptr[idx * arith_domain_size]); idx++;
            
            prepare_arith_cache(cache.vars[ARITH_C].row_on_k,     "c_arith_row_on_k",     arith_domain_size, c_arith_row_on_k,     &d_ptr[idx * arith_domain_size]); idx++;
            prepare_arith_cache(cache.vars[ARITH_C].col_on_k,     "c_arith_col_on_k",     arith_domain_size, c_arith_col_on_k,     &d_ptr[idx * arith_domain_size]); idx++;
            prepare_arith_cache(cache.vars[ARITH_C].row_col_on_k, "c_arith_row_col_on_k", arith_domain_size, c_arith_row_col_on_k, &d_ptr[idx * arith_domain_size]); idx++;
            prepare_arith_cache(cache.vars[ARITH_C].val,          "c_arith_val",          arith_domain_size, c_arith_val,          &d_ptr[idx * arith_domain_size]); idx++;
            prepare_arith_cache(cache.vars[ARITH_C].evals_on_k,   "c_arith_evals_on_k",   arith_domain_size, c_arith_evals_on_k,   &d_ptr[idx * arith_domain_size]); idx++;

            // Synchronize - we need the data on the GPU for the next step
            cudaDeviceSynchronize();

            for (size_t j = ARITH_A; j <= ARITH_C; j++) {
                prepare_arith_evals(cache.vars[j].row_on_k,     &d_ptr[idx * arith_domain_size], lg_arith_domain_size, gpu); idx += 2;
                prepare_arith_evals(cache.vars[j].col_on_k,     &d_ptr[idx * arith_domain_size], lg_arith_domain_size, gpu); idx += 2;
                prepare_arith_evals(cache.vars[j].row_col_on_k, &d_ptr[idx * arith_domain_size], lg_arith_domain_size, gpu); idx += 2;

            }
            
            cudaDeviceSynchronize();
        };
    }

    RustError ComputePolyT_internal(const gpu_t& gpu, stream_t& stream, poly_t_cache_t* cache,
                                    fr_t* d_out, fr_t* d_poly, uint32_t* d_atomic_task_count,
                                    fr_t* h_sums, fr_t* h_poly,
                                    uint32_t lg_constraint_domain_size,
                                    uint32_t lg_input_domain_size) {
        try {
            gpu.select();
            size_t constraint_domain_size = (size_t)1 << lg_constraint_domain_size;
            
            // eta_b, eta_c, r_alpha_x_evals
            stream.HtoD(d_poly, h_poly, constraint_domain_size + 2);
            cudaMemsetAsync(d_atomic_task_count, 0, sizeof(uint32_t), stream);

            compute_poly_t_partial<<<constraint_domain_size / 1024, 1024, 0, stream>>>
                (*cache, d_atomic_task_count,
                 d_out, d_poly,
                 1 << lg_constraint_domain_size, 1 << lg_input_domain_size,
                 lg_constraint_domain_size - lg_input_domain_size);

            // d_poly will hold the final sums
            cudaMemsetAsync(d_poly, 0, sizeof(fr_t) * constraint_domain_size, stream);
            compute_poly_t_final<<<(cache->final_nonzero_count + 1024 - 1) / 1024, 1024, 0, stream>>>
                (*cache, d_poly, d_out);

            NTT::NTT_device(d_poly, lg_constraint_domain_size,
                            NTT::InputOutputOrder::NR, NTT::Direction::inverse, stream);
            NTT::bit_rev(d_out, d_poly, lg_constraint_domain_size, stream);
            
            stream.DtoH(h_poly, d_out, constraint_domain_size);
            stream.sync();
            
        } catch (const cuda_error& e) {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};
    }

    RustError ComputePolyT(fr_t* out,
                           fr_t* eta_b,
                           fr_t* eta_c,
                           fr_t* r_alpha_x_evals,
                           uint32_t lg_constraint_domain_size,
                           uint32_t lg_input_domain_size) {
        resource_t *resource = resources.recv();
        int dev = resource->dev;
        auto& gpu = select_gpu(dev);
        int stream_idx = resource->stream;
        stream_t& stream = gpu[stream_idx];

        // Copy data to pinned staging buffer
        size_t constraint_domain_size = (size_t)1 << lg_constraint_domain_size;
        fr_t* h_poly = h_addr_poly(dev, stream_idx);
        memcpy(&h_poly[0], eta_b, sizeof(fr_t));
        memcpy(&h_poly[1], eta_c, sizeof(fr_t));
        memcpy(&h_poly[2], r_alpha_x_evals, constraint_domain_size * sizeof(fr_t));
        fr_t* d_poly = d_addr_poly(dev, stream_idx);
        fr_t* d_out = &d_poly[constraint_domain_size + 2];
        uint32_t* d_atomic_task_count = (uint32_t*)&d_poly[constraint_domain_size * 2 + 2];
        
        fr_t* h_out = &h_poly[constraint_domain_size + 2];
        fr_t* h_tmp = &h_poly[constraint_domain_size * 2 + 2];

        RustError e = ComputePolyT_internal(gpu, stream, &poly_t_cache[dev],
                                            d_out, d_poly, d_atomic_task_count,
                                            h_tmp, h_poly,
                                            lg_constraint_domain_size, lg_input_domain_size);
        if (e.code != cudaSuccess) {
            resources.send(resource);
            return e;
        }
        memcpy(out, h_poly, sizeof(fr_t) * constraint_domain_size);

        resources.send(resource);
        return RustError{cudaSuccess};
    }

    RustError CalculateLHS(fr_t* out,
                           fr_t* vanish_x,
                           fr_t* denoms, size_t denoms_len,
                           fr_t* b, size_t b_len,
                           fr_t* c, size_t c_len,
                           fr_t* d, size_t d_len,
                           uint32_t lg_domain_size,
                           uint32_t lg_ext_domain_size) {
        resource_t *resource = resources.recv();
        int dev = resource->dev;
        auto& gpu = select_gpu(dev);
        int stream_idx = resource->stream;
        stream_t& stream = gpu[stream_idx];

        // Copy data to pinned staging buffer
        size_t domain_size = (size_t)1 << lg_domain_size;
        size_t ext_domain_size = (size_t)1 << lg_ext_domain_size;
        assert(denoms_len == ext_domain_size);
        fr_t* h_denoms = h_addr_poly(dev, stream_idx);
        fr_t* h_b = &h_denoms[ext_domain_size];
        fr_t* h_c = &h_denoms[ext_domain_size * 2];
        fr_t* h_d = &h_denoms[ext_domain_size * 3];
        memcpy(h_denoms, denoms, sizeof(fr_t) * denoms_len);
        memcpy(h_b, b, sizeof(fr_t) * b_len);
        memcpy(h_c, c, sizeof(fr_t) * c_len);
        memcpy(h_d, d, sizeof(fr_t) * d_len);
        
        fr_t* d_poly = d_addr_poly(dev, stream_idx);
        RustError e = CalculateLHS_internal(gpu, stream, d_poly,
                                            h_denoms, // will hold the result
                                            d_batch_eval[dev], vanish_x,
                                            h_denoms, denoms_len, h_b, b_len,
                                            h_c, c_len, h_d, d_len,
                                            lg_domain_size, lg_ext_domain_size);
        if (e.code != cudaSuccess) {
            resources.send(resource);
            return e;
        }
        memcpy(out, h_denoms, sizeof(fr_t) * ext_domain_size);

        resources.send(resource);
        return RustError{cudaSuccess};
    }

    RustError MatrixSumcheck_internal(arith_cache_var_t& cache,
                                      const gpu_t& gpu, stream_t& stream,
                                      fr_t* h_scalars, fr_t* d_scalars,
                                      fr_t* h_f_poly,
                                      fr_t* h_h_poly,
                                      fr_t* d_inverses,
                                      fr_t* d_alpha,
                                      fr_t* d_beta,
                                      fr_t* d_v_H_alpha_v_H_beta,
                                      fr_t* d_alpha_beta,
                                      uint32_t lg_domain_size) {
        size_t domain_size = (size_t)1 << lg_domain_size;
        try {
            gpu.select();

            // Temporary
            fr_t* d_b_poly             = &d_scalars[domain_size * 0]; // 2x
            fr_t* d_b_evals            = &d_scalars[domain_size * 1];
            fr_t* d_f_poly             = &d_scalars[domain_size * 2]; // 2x
            fr_t* d_a_poly             = &d_scalars[domain_size * 4]; // 2x

            stream.HtoD(d_inverses, h_scalars, domain_size + 4);

            const int tpb = 1024;

            // Step 1
            //   inverses = inverses * evals_on_k
            //   b_evals = alpha_beta - alpha * row_on_k - beta * col_on_k + row_col_on_k
            //   drop
            //     row_on_k -> b_evals
            //     col_on_k -> a_poly
            //     row_col_on_k -> b_poly, row_on_k -> f_poly, evals_on_k -> free
            //   a_poly = arith_val * v_H_alpha_beta_v_H_beta
            //     arith_val -> b_poly ext domain
            compute_matrix_sumcheck_step1_1<<<(domain_size * 2) / 256, 256, 0, stream>>>
                (domain_size, cache, d_b_poly, d_a_poly, 
                 d_inverses, d_alpha, d_beta,
                 d_alpha_beta, d_v_H_alpha_v_H_beta);

            compute_matrix_sumcheck_step1_2<<<domain_size / tpb, tpb, 0, stream>>>
                (domain_size, cache, d_b_poly, d_a_poly, 
                 d_inverses, d_alpha, d_beta,
                 d_alpha_beta, d_v_H_alpha_v_H_beta);
            
            // f_poly = iNTT(inverses)
            NTT::bit_rev(d_f_poly, d_inverses, lg_domain_size, stream);
            NTT::NTT_device(d_f_poly, lg_domain_size,
                            NTT::InputOutputOrder::RN, NTT::Direction::inverse, stream);
            // f_poly is an output
            stream.DtoH(h_f_poly, d_f_poly, domain_size);

            // First half of h is a, second half zero
            fr_t* d_h_poly = d_a_poly;
            cudaMemsetAsync(&d_h_poly[domain_size], 0,
                            sizeof(fr_t) * domain_size, stream);
            
            // Clear the 2nd half of the multiply inputs
            cudaMemsetAsync(&d_f_poly[domain_size], 0,
                            sizeof(fr_t) * domain_size, stream);

            //Polynomial::MulDev(stream, d_b_poly, d_b_poly, d_f_poly, lg_domain_size);
            {
                fr_t* d_in0 = d_b_poly;
                fr_t* d_in1 = d_f_poly;
                fr_t* d_out = d_b_poly;
                NTT::NTT_device(d_in1, lg_domain_size + 1,
                                NTT::InputOutputOrder::NR, NTT::Direction::forward, stream);
        
                // Inner multiply
                polynomial_inner_multiply<<<domain_size * 2 / tpb, tpb, 0, stream>>>
                    (d_out, d_in0, d_in1);
        
                // Perform iNTT on the result
                NTT::NTT_device(d_out, lg_domain_size + 1,
                                NTT::InputOutputOrder::RN, NTT::Direction::inverse, stream);
            }

            polynomial_sub<<<domain_size * 2 / tpb, tpb, 0, stream>>>
                (d_h_poly, d_h_poly, d_b_poly);

            // h_poly is an output
            stream.DtoH(h_h_poly, d_h_poly, domain_size * 2);
            stream.sync();
        } catch (const cuda_error& e) {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};
    }
        
    RustError MatrixSumcheck
      ( uint32_t lg_domain_size,
        ArithVar cache_var,
       
        //fr_t* f_coeff0,           // F
        fr_t* h_poly,             // 65535
        fr_t* g_poly,             // 65536
        
        fr_t* alpha,              // F
        fr_t* beta,               // F
        fr_t* v_H_alpha_v_H_beta, // F
        fr_t* inverses)           // 65536
    {
        size_t domain_size = (size_t)1 << lg_domain_size;
        
        resource_t *resource = resources.recv();
        int dev = resource->dev;
        auto& gpu = select_gpu(dev);
        int stream_idx = resource->stream;
        stream_t& stream = gpu[stream_idx];

        fr_t* d_scalars      = d_addr_poly(dev, stream_idx);
        fr_t* h_scalars      = h_addr_poly(dev, stream_idx);
        
        // Inputs
        fr_t* d_inverses           = &d_scalars[domain_size * 6];
        fr_t* d_alpha              = &d_scalars[domain_size * 7 + 0];
        fr_t* d_beta               = &d_scalars[domain_size * 7 + 1];
        fr_t* d_v_H_alpha_v_H_beta = &d_scalars[domain_size * 7 + 2];
        fr_t* d_alpha_beta         = &d_scalars[domain_size * 7 + 3];

        fr_t* h_inverses           = &h_scalars[domain_size * 0];
        fr_t* h_alpha              = &h_scalars[domain_size * 1 + 0];
        fr_t* h_beta               = &h_scalars[domain_size * 1 + 1];
        fr_t* h_v_H_alpha_v_H_beta = &h_scalars[domain_size * 1 + 2];
        fr_t* h_alpha_beta         = &h_scalars[domain_size * 1 + 3];

        // Outputs
        fr_t* h_f_poly             = &h_scalars[domain_size * 0];
        fr_t* h_h_poly             = &h_scalars[domain_size * 1]; // 2x
        
        fr_t  alpha_beta           = *alpha * *beta;
        
        memcpy(h_alpha,               alpha,              sizeof(fr_t));
        memcpy(h_beta,                beta,               sizeof(fr_t));
        memcpy(h_v_H_alpha_v_H_beta,  v_H_alpha_v_H_beta, sizeof(fr_t));
        memcpy(h_alpha_beta,         &alpha_beta,         sizeof(fr_t));
        memcpy(h_inverses,            inverses,           sizeof(fr_t) * domain_size);

        arith_cache_t& cache = arith_cache[dev];

        RustError e = MatrixSumcheck_internal(cache.vars[cache_var],
                                              gpu, stream, h_scalars, d_scalars,
                                              h_f_poly, h_h_poly,
                                              d_inverses,
                                              d_alpha,
                                              d_beta,
                                              d_v_H_alpha_v_H_beta,
                                              d_alpha_beta,
                                              lg_domain_size);
        if (e.code != cudaSuccess) {
            resources.send(resource);
            return e;
        }
        memcpy(h_poly, h_h_poly, sizeof(fr_t) * domain_size * 2);
        // f_poly
        memcpy(g_poly, h_f_poly, sizeof(fr_t) * domain_size);

        resources.send(resource);
        return RustError{cudaSuccess};
        
    }
    
    RustError MSMCacheBases(const affine_t points[], size_t bases_len, size_t ffi_affine_sz) {
        uint64_t key = ((uint64_t *)points)[0];
        assert (cur_msm_cache_entry < msm_cache_entries);
        assert(bases_len <= msm_cache_npoints);
        for (size_t devt = 0; devt < num_gpus; devt++) {
            auto& gpu = select_gpu(devt);
            uint32_t windowBits;
            uint32_t allWindows;
            msm_cuda_precomp_params(msm_cuda_ctxs[devt].ctxs[0], &windowBits, &allWindows);

            // Allocate the max cache size, though we might not use it all. This
            // simplifies precomputation on the GPU since they can all be the same
            // size and stride. 
            msm_point_cache[devt].cache[cur_msm_cache_entry] =
                new dev_ptr_t<affine_noinf_t>(msm_cache_npoints * allWindows);
            fr_t* h_buf = *h_mem[devt];
            // Cache the entire set of bases
            populate_msm_points(devt, bases_len, msm_cache_npoints, (const rust_p1_affine_t*)points,
                                (affine_noinf_t*)h_buf, // host buffer
                                *msm_point_cache[devt].cache[cur_msm_cache_entry], // device buffer
                                ffi_affine_sz);

            // Ensure all transfers are complete. Could remove this if precomp used the right streams
            cudaDeviceSynchronize();
        }
        // Store the key
        msm_keys[cur_msm_cache_entry] = key;

        cur_msm_cache_entry++;
        return RustError{cudaSuccess};
    }
    
    RustError MSM(point_t* out, const affine_t points[],
                  size_t npoints, size_t bases_len,
                  const scalar_t scalars[], size_t ffi_affine_sz)
    {
        RustError result;
        
        resource_t *resource = resources.recv();
        int dev = resource->dev;
        auto& gpu = select_gpu(dev);
        int stream_idx = resource->stream;
        stream_t& stream = gpu[stream_idx];

        // See if these bases are cached
        uint64_t key = ((uint64_t *)points)[0];
        dev_ptr_t<affine_noinf_t>* cached_points = nullptr;
        for (size_t i = 0; i < msm_cache_entries && i < cur_msm_cache_entry; i++) {
            if (key == msm_keys[i]) {
                cached_points = msm_point_cache[dev].cache[i];
                break;
            }
        }
        // Create a new cached msm_t
        // Not MT safe - if we populate the cache from a single thread
        //       prior to going MT this will be ok.
        if (cached_points == nullptr && cur_msm_cache_entry < msm_cache_entries) {
            MSMCacheBases(points, bases_len, ffi_affine_sz);
            // Re-select the target gpu
            select_gpu(dev);
            // And the points cached for the kernel
            cached_points = msm_point_cache[dev].cache[cur_msm_cache_entry - 1];
        }
        assert (cached_points != nullptr);

        fr_t* h_scalars = h_addr_msm(dev, stream_idx);
        memcpy(h_scalars, scalars, sizeof(fr_t) * npoints);
            
        fr_t* d_scalars = d_addr_msm(dev, stream_idx);
        // Must have room for scalars, buckets, bucketsums
        bucket_t* d_buckets = (bucket_t*)&d_scalars[npoints];
        bucket_t* h_bucketSums = (bucket_t*)h_scalars;
        stream.HtoD(d_scalars, h_scalars, npoints);
        affine_noinf_t* cached_points_ptr = *cached_points;

        msm_cuda_launch(msm_cuda_ctxs[dev].ctxs[stream_idx], npoints, out,
                        d_scalars, cached_points_ptr,
                        d_buckets, h_bucketSums,
                        false, stream);

        result = RustError{cudaSuccess};

        resources.send(resource);
        return result;
    }
};

bool amlive = true;
static void shutdown() {
    amlive = false;
}
static bool alive() {
    return amlive;
}

snarkvm_t *snarkvm = nullptr;

extern "C" {
    void snarkvm_init_gpu() {
        snarkvm = new snarkvm_t();
        assert(snarkvm);
    }

    void snarkvm_cleanup_gpu() {
        shutdown();
        sleep(1); // Shutting down - delay while process exits
    }

    void* snarkvm_alloc_pinned(size_t bytes) {
        void* ptr = nullptr;
        if (cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault) != cudaSuccess) {
            return nullptr;
        }
        return ptr;
    }

    void snarkvm_free_pinned(void *ptr) {
        cudaFreeHost(ptr);
    }

    RustError snarkvm_ntt_batch(fr_t* inout, size_t N, uint32_t lg_domain_size,
                                NTT::InputOutputOrder ntt_order, NTT::Direction ntt_direction,
                                NTT::Type ntt_type)
    {
        if (!alive()) {
            sleep(10); // Shutting down - delay while process exits
            return RustError{0};
        }
        assert(N == 1);
        RustError err = RustError{0};
        try {
            err = snarkvm->NTT(inout, lg_domain_size, ntt_order,
                               ntt_direction/*, ntt_type*/);
        } catch(exception &exc) {
            if (!QUIET) {
                cout << "Exception at " << __FILE__ << ":" << __LINE__ << endl;
            }
            shutdown();
            sleep(10); // Shutting down - delay while process exits
        }
        return err;
    }

    RustError snarkvm_polymul(fr_t* out, fr_t* in0, fr_t* in1, uint32_t lg_domain_size) {
        if (!alive()) {
            sleep(10); // Shutting down - delay while process exits
            return RustError{0};
        }
        RustError err = RustError{0};
        try {
            err = snarkvm->PolyMul(out, in0, in1, lg_domain_size);
        } catch(exception &exc) {
            if (!QUIET) {
                cout << "Exception at " << __FILE__ << ":" << __LINE__ << endl;
            }
            shutdown();
            sleep(10); // Shutting down - delay while process exits
        }
        return err;
    }

    RustError snarkvm_calculate_summed_z_m(fr_t* out,
                                           fr_t* z_a,
                                           fr_t* z_b,
                                           fr_t* eta_c,
                                           fr_t* eta_b_over_eta_c,
                                           uint32_t lg_domain_size) {
        if (!alive()) {
            sleep(10); // Shutting down - delay while process exits
            return RustError{0};
        }
        RustError err = RustError{0};
        try {
            err = snarkvm->CalculateSummedZM(out, z_a, z_b,
                                             eta_c, eta_b_over_eta_c,
                                             lg_domain_size);
        } catch(exception &exc) {
            if (!QUIET) {
                cout << "Exception at " << __FILE__ << ":" << __LINE__ << endl;
            }
            shutdown();
            sleep(10); // Shutting down - delay while process exits
        }
        return err;
    }

    void snarkvm_cache_poly_t_inputs(size_t a_len, uint32_t* a_r, uint32_t* a_c, fr_t* a_coeff,
                                     size_t b_len, uint32_t* b_r, uint32_t* b_c, fr_t* b_coeff,
                                     size_t c_len, uint32_t* c_r, uint32_t* c_c, fr_t* c_coeff,

                                     fr_t* a_arith_row_on_k,
                                     fr_t* a_arith_col_on_k,
                                     fr_t* a_arith_row_col_on_k,
                                     fr_t* a_arith_val,
                                     fr_t* a_arith_evals_on_k,
                                     
                                     fr_t* b_arith_row_on_k,
                                     fr_t* b_arith_col_on_k,
                                     fr_t* b_arith_row_col_on_k,
                                     fr_t* b_arith_val,
                                     fr_t* b_arith_evals_on_k,
                                     
                                     fr_t* c_arith_row_on_k,
                                     fr_t* c_arith_col_on_k,
                                     fr_t* c_arith_row_col_on_k,
                                     fr_t* c_arith_val,
                                     fr_t* c_arith_evals_on_k
                                     ) {
        if (!alive()) {
            sleep(10); // Shutting down - delay while process exits
            return;
        }
        try {
            snarkvm->CachePolyTInputs(a_len, a_r, a_c, a_coeff,
                                      b_len, b_r, b_c, b_coeff,
                                      c_len, c_r, c_c, c_coeff,

                                      a_arith_row_on_k,
                                      a_arith_col_on_k,
                                      a_arith_row_col_on_k,
                                      a_arith_val,
                                      a_arith_evals_on_k,
                                      
                                      b_arith_row_on_k,
                                      b_arith_col_on_k,
                                      b_arith_row_col_on_k,
                                      b_arith_val,
                                      b_arith_evals_on_k,
                                      
                                      c_arith_row_on_k,
                                      c_arith_col_on_k,
                                      c_arith_row_col_on_k,
                                      c_arith_val,
                                      c_arith_evals_on_k
                                      );
        } catch(exception &exc) {
            if (!QUIET) {
                cout << "Exception at " << __FILE__ << ":" << __LINE__ << endl;
            }
            shutdown();
            sleep(10); // Shutting down - delay while process exits
        }
    }

    RustError snarkvm_compute_poly_t(fr_t* out,
                                     fr_t* eta_b,
                                     fr_t* eta_c,
                                     fr_t* r_alpha_x_evals,
                                     uint32_t lg_constraint_domain_size,
                                     uint32_t lg_input_domain_size) {
        if (!alive()) {
            sleep(10); // Shutting down - delay while process exits
            return RustError{0};
        }
        RustError err = RustError{0};
        try {
            err = snarkvm->ComputePolyT(out, eta_b, eta_c, r_alpha_x_evals,
                                        lg_constraint_domain_size, lg_input_domain_size);
        } catch(exception &exc) {
            if (!QUIET) {
                cout << "Exception at " << __FILE__ << ":" << __LINE__ << endl;
            }
            shutdown();
            sleep(10); // Shutting down - delay while process exits
        }
        return err;
    }
    
    RustError snarkvm_calculate_lhs(fr_t* out,
                                    fr_t* vanish_x,
                                    fr_t* denoms, size_t denoms_len,
                                    fr_t* b, size_t b_len,
                                    fr_t* c, size_t c_len,
                                    fr_t* d, size_t d_len,
                                    uint32_t lg_domain_size,
                                    uint32_t lg_ext_domain_size) {
        if (!alive()) {
            sleep(10); // Shutting down - delay while process exits
            return RustError{0};
        }
        RustError err = RustError{0};
        try {
            err = snarkvm->CalculateLHS(out, vanish_x, denoms, denoms_len, b, b_len,
                                        c, c_len, d, d_len,
                                        lg_domain_size, lg_ext_domain_size);
        } catch(exception &exc) {
            if (!QUIET) {
                cout << "Exception at " << __FILE__ << ":" << __LINE__ << endl;
            }
            shutdown();
            sleep(10); // Shutting down - delay while process exits
        }
        return err;
    }

    RustError snarkvm_matrix_sumcheck
      ( uint32_t lg_domain_size,
        ArithVar cache_var,
       
        fr_t* h_poly,
        fr_t* g_poly,
        
        fr_t* alpha,
        fr_t* beta,
        fr_t* v_H_alpha_v_H_beta,
        fr_t* inverses) {
        if (!alive()) {
            sleep(10); // Shutting down - delay while process exits
            return RustError{0};
        }
        RustError err = RustError{0};
        try {
            err = snarkvm->MatrixSumcheck(lg_domain_size,
                                          cache_var,
                                          
                                          //f_coeff0,
                                          h_poly,
                                          g_poly,
                                          
                                          alpha,
                                          beta,
                                          v_H_alpha_v_H_beta,
                                          inverses);
        } catch(exception &exc) {
            if (!QUIET) {
                cout << "Exception at " << __FILE__ << ":" << __LINE__ << endl;
            }
            shutdown();
            sleep(10); // Shutting down - delay while process exits
        }
        return err;
        
    }

    RustError snarkvm_msm(point_t* out, const affine_t points[], size_t npoints, size_t bases_len,
                          const scalar_t scalars[], size_t ffi_affine_size) {
        if (!alive()) {
            sleep(10); // Shutting down - delay while process exits
            return RustError{0};
        }
        
        RustError err = RustError{0};
        try {
            err =snarkvm->MSM(out, points, npoints, bases_len, scalars, ffi_affine_size);
        } catch(exception &exc) {
            if (!QUIET) {
                cout << "Exception at " << __FILE__ << ":" << __LINE__ << endl;
            }
            shutdown();
            sleep(10); // Shutting down - delay while process exits
        }
        return err;
    }

    RustError snarkvm_msm_cache(const affine_t points[], size_t bases_len, size_t ffi_affine_size) {
        return snarkvm->MSMCacheBases(points, bases_len, ffi_affine_size);
    }
}


#endif
