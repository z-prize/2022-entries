// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __POLYNOMIAL_CUH__
#define __POLYNOMIAL_CUH__

#ifndef __CUDA_ARCH__

# include <util/exception.cuh>
# include <util/rusterror.h>
# include <util/gpu_t.cuh>

#endif

__global__
void polynomial_inner_multiply(fr_t* out, fr_t* in0, fr_t* in1) {
    index_t idx = threadIdx.x + blockDim.x * (index_t)blockIdx.x;
    fr_t x = in0[idx];
    fr_t y = in1[idx];
    out[idx] = x * y;
}

__global__
void polynomial_scale(fr_t* out, fr_t* in0, fr_t* elmt) {
    index_t idx = threadIdx.x + blockDim.x * (index_t)blockIdx.x;
    fr_t x = in0[idx];
    out[idx] = x * *elmt;
}

__global__
void polynomial_add(fr_t* out, fr_t* in0, fr_t* in1) {
    index_t idx = threadIdx.x + blockDim.x * (index_t)blockIdx.x;
    fr_t x = in0[idx];
    fr_t y = in1[idx];
    out[idx] = x + y;
}

__global__
void polynomial_sub(fr_t* out, fr_t* in0, fr_t* in1) {
    index_t idx = threadIdx.x + blockDim.x * (index_t)blockIdx.x;
    fr_t x = in0[idx];
    fr_t y = in1[idx];
    out[idx] = x - y;
}

__global__
void polynomial_incr(fr_t* out, fr_t* in0) {
    index_t idx = threadIdx.x + blockDim.x * (index_t)blockIdx.x;
    fr_t x = in0[idx];
    out[idx] = x + fr_t::one();
}

__global__
void polynomial_decr(fr_t* out, fr_t* in0) {
    index_t idx = threadIdx.x + blockDim.x * (index_t)blockIdx.x;
    fr_t x = in0[idx];
    out[idx] = x - fr_t::one();
}

#ifdef __CUDA_ARCH__
__device__
void print_fr_dev(fr_t* in) {
    fr_t x = *in;
    x.from();
    printf("0x");
    for (size_t i = 0; i < x.len(); i++) {
        printf("%08x", x[x.len() - i - 1]);
    }
    printf("\n");
}
#endif

__global__
void print_fr(fr_t* in) {
#ifdef __CUDA_ARCH__
    fr_t x = *in;
    x.from();
    printf("0x");
    for (size_t i = 0; i < x.len(); i++) {
        printf("%08x", x[x.len() - i - 1]);
    }
    printf("\n");
#endif
}

#ifndef __CUDA_ARCH__

class Polynomial {
public:
    // in0 and in1 are in lg_domain_size
    // out will be in lg_domain_size + 1
    static void MulDev(stream_t& stream,
                       fr_t* d_out, fr_t* d_in0, fr_t* d_in1, // Device pointers
                       uint32_t lg_domain_size) {
        size_t domain_size = (size_t)1 << lg_domain_size;
        size_t ext_domain_size = domain_size * 2;
        
        // Perform NTT on the input data
        NTT::NTT_device(d_in0, lg_domain_size + 1,
                        NTT::InputOutputOrder::NR, NTT::Direction::forward, stream);
        NTT::NTT_device(d_in1, lg_domain_size + 1,
                        NTT::InputOutputOrder::NR, NTT::Direction::forward, stream);
        
        // Inner multiply
        polynomial_inner_multiply<<<ext_domain_size / 1024, 1024, 0, stream>>>
            (d_out, d_in0, d_in1);
        
        // Perform iNTT on the result
        NTT::NTT_device(d_out, lg_domain_size + 1,
                        NTT::InputOutputOrder::RN, NTT::Direction::inverse, stream);
    }
    
    // in0 and in1 are in lg_domain_size
    // out will be in lg_domain_size + 1
    static RustError Mul(const gpu_t& gpu, stream_t& stream,
                         fr_t* dmem,
                         fr_t* out, fr_t* in0, fr_t* in1,
                         uint32_t lg_domain_size) {
        try {
            gpu.select();

            size_t domain_size = (size_t)1 << lg_domain_size;
            size_t ext_domain_size = domain_size * 2;

            fr_t* d_in0 = dmem;
            fr_t* d_in1 = &dmem[ext_domain_size];
            fr_t* d_out = &dmem[ext_domain_size * 2];

            // Copy the input data
            cudaMemsetAsync(d_in0, 0, sizeof(fr_t) * ext_domain_size * 3, stream);
            stream.HtoD(d_in0, in0, domain_size);
            stream.HtoD(d_in1, in1, domain_size);

            MulDev(stream, d_out, d_in0, d_in1, lg_domain_size);

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
};
#endif //  __CUDA_ARCH__

#endif
