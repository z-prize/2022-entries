// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>

#ifndef WARP_SZ
# define WARP_SZ 32
#endif

#ifndef NTHREADS
# define NTHREADS 128
#endif
#if NTHREADS < 32 || (NTHREADS & (NTHREADS-1)) != 0
# error "bad NTHREADS value"
#endif

constexpr static int log2(int n)
{   int ret=0; while (n>>=1) ret++; return ret;   }

static const int NTHRBITS = log2(NTHREADS);

#ifndef NBITS
# define NBITS 253
#endif
#ifndef WBITS
# define WBITS 11
#endif
#define NWINS ((NBITS+WBITS-1)/WBITS)   // ceil(NBITS/WBITS)

#ifndef LARGE_L1_CODE_CACHE
# define LARGE_L1_CODE_CACHE 0
#endif

//
// To be launched as 'pippenger<<<dim3(NWINS, N), NTHREADS>>>(...)'  with
// |npoints| being around N*2**20 and N*NWINS*NTHREADS not exceeding the
// occupancy limit.
//
__global__ void pippenger(const affine_t* points, size_t npoints,
               const scalar_t* scalars, bool mont, bucket_t (*buckets)[NWINS][1<<WBITS],
               bucket_t (*ret)[NWINS][NTHREADS][2] = nullptr, const int acc_level = 0);
__global__ void pre_scalars(scalar_t* scalars_, const size_t npoints);
__global__ void pre_scalars_one(bucket_t *ones, const affine_t* points,
                                scalar_t* scalars, const size_t npoints);
__global__ void io_helper(fr_t* bases,fr_t* roots, 
    const int32_t chunk_size, const int32_t chunk_num, const int32_t gap, const int32_t size);

__global__ void oi_helper(fr_t* bases, fr_t* roots,
    const int32_t chunk_size, const int32_t chunk_num, const int32_t gap, const int32_t size);
#ifdef __CUDA_ARCH__

#include <cooperative_groups.h>

static __shared__ bucket_t scratch[NTHREADS];

// Transposed scalar_t
class scalar_T {
    uint32_t val[sizeof(scalar_t)/sizeof(uint32_t)][WARP_SZ];

public:
    __device__ uint32_t& operator[](size_t i)              { return val[i][0]; }
    __device__ const uint32_t& operator[](size_t i) const  { return val[i][0]; }
    __device__ scalar_T& operator=(const scalar_t& rhs)
    {
        for (size_t i = 0; i < sizeof(scalar_t)/sizeof(uint32_t); i++)
            val[i][0] = rhs[i];
        return *this;
    }
};

class scalars_T {
    scalar_T* ptr;

public:
    __device__ scalars_T(void* rhs) { ptr = (scalar_T*)rhs; }
    __device__ scalar_T& operator[](size_t i)
    {   return *(scalar_T*)&(&ptr[i/WARP_SZ][0])[i%WARP_SZ];   }
    __device__ const scalar_T& operator[](size_t i) const
    {   return *(const scalar_T*)&(&ptr[i/WARP_SZ][0])[i%WARP_SZ];   }
};

constexpr static __device__ int dlog2(int n)
{   int ret=0; while (n>>=1) ret++; return ret;   }

static __device__ int is_unique(int wval, int dir=0)
{
    int* const wvals = (int *)scratch;
    const uint32_t tid = threadIdx.x;
    dir &= 1;   // force logical operations on predicates

    NTHREADS > WARP_SZ ? __syncthreads() : __syncwarp();
    wvals[tid] = wval;
    NTHREADS > WARP_SZ ? __syncthreads() : __syncwarp();

    // Straightforward scan appears to be the fastest option for NTHREADS.
    // Bitonic sort complexity, a.k.a. amount of iterations, is [~3x] lower,
    // but each step is [~5x] slower...
    int negatives = 0;
    int uniq = 1;
    #pragma unroll 16
    for (uint32_t i=0; i<NTHREADS; i++) {
        int b = wvals[i];   // compiled as 128-bit [broadcast] loads:-)
        if (((i<tid)^dir) && i!=tid && wval==b)
            uniq = 0;
        negatives += (b < 0);
    }

    return uniq | (int)(NTHREADS-1-negatives)>>31;
    // return value is 1, 0 or -1.
}

#if WBITS==16
template<class scalar_t>
static __device__ int get_wval(const scalar_t& d, uint32_t off, uint32_t bits)
{
    uint32_t ret = d[off/32];
    return (ret >> (off%32)) & ((1<<bits) - 1);
}
#else
template<class scalar_t>
static __device__ int get_wval(const scalar_t& d, uint32_t off, uint32_t bits)
{
    uint32_t top = off + bits - 1;
    uint64_t ret = ((uint64_t)d[top/32] << 32) | d[off/32];

    return (int)(ret >> (off%32)) & ((1<<bits) - 1);
}
#endif

__global__
void pre_scalars(scalar_t* scalars, const size_t npoints)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t gsize = gridDim.x * blockDim.x;
    #pragma unroll 1
    while (gid < npoints) {
        scalar_t s = scalars[gid];
        s.from();
        scalars[gid] = s;
        gid += gsize;
    }    
}

__global__
void pre_scalars_one(bucket_t *ones, const affine_t* points,
                    scalar_t* scalars, const size_t npoints)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t gsize = gridDim.x * blockDim.x;
    bucket_t res;
    res.inf();
    while (gid < npoints) {
        scalar_t s = scalars[gid];
        if (s.is_one()) {
            res.add(points[gid]);
            scalars[gid].zero();
        } else {
            s.from();
            scalars[gid] = s;  
        }
        gid += gsize;
    }

    for (size_t i=(blockDim.x>>1); i>0; i>>=1) {
        if (threadIdx.x >= i) {
            scratch[threadIdx.x - i] = res;
            return;
        }
        __syncthreads();
        bucket_t s = scratch[threadIdx.x];
        res.add(s);
    }
    ones[blockIdx.x] = res;
}

__global__
void pippenger(const affine_t* points, size_t npoints,
               const scalar_t* scalars_, bool mont,
               bucket_t (*buckets)[NWINS][1<<WBITS],
               bucket_t (*ret)[NWINS][NTHREADS][2] /*= nullptr*/,
               const int acc_level)
{
    assert(blockDim.x == NTHREADS);
    assert(gridDim.x == NWINS);
    assert(npoints == (uint32_t)npoints);

    if (gridDim.y > 1) {
        uint32_t delta = ((uint32_t)npoints + gridDim.y - 1) / gridDim.y;
        delta = (delta+WARP_SZ-1) & (0U-WARP_SZ);
        uint32_t off = delta * blockIdx.y;

        points   += off;
        scalars_ += off;
        if (blockIdx.y == gridDim.y-1)
            npoints -= off;
        else
            npoints = delta;
    }

    uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    const uint32_t bit0 = bid * WBITS;
    bucket_t* row = buckets[blockIdx.y][bid];

    // scalars_T scalars = const_cast<scalar_t*>(scalars_);
    // if (mont) {
    //     #pragma unroll 1
    //     for (uint32_t i = NTHREADS*bid + tid; i < npoints; i += NTHREADS*NWINS) {
    //         scalar_t s = scalars_[i];
    //         s.from();
    //         scalars[i] = s;
    //     }
    //     cooperative_groups::this_grid().sync();
    // } else { // if (typeid(scalars) != typeid(scalars_)) {
    //     #pragma unroll 1
    //     for (uint32_t i = NTHREADS*bid + tid; i < npoints; i += NTHREADS*NWINS) {
    //         scalar_t s = scalars_[i];
    //         __syncwarp();
    //         scalars[i] = s;
    //     }
    //     cooperative_groups::this_grid().sync();
    // }

#if (__CUDACC_VER_MAJOR__-0) >= 11
    __builtin_assume(tid<NTHREADS);
#endif
    #pragma unroll 4
    for (uint32_t i = tid; i < 1<<WBITS; i += NTHREADS)
        row[i].inf();

    int wbits = (bit0 > NBITS-WBITS) ? NBITS-bit0 : WBITS;
    int bias  = (tid >> max(wbits+NTHRBITS-WBITS, 0)) << max(wbits, WBITS-NTHRBITS);

    int dir = 1;
    for (uint32_t i = tid; true; ) {
        int wval = -1;

        affine_t point;
        if (i < npoints) {
            wval = get_wval(scalars_[i], bit0, wbits);
            wval += wval ? bias : 0;
            point = points[i];
        }

        int uniq = is_unique(wval, dir^=1) | wval==0;
        if (uniq < 0)   // all |wval|-s are negative, all done
            break;

        if (i < npoints && uniq) {
            if (wval) {
                row[wval-1].add(point);
            }
            i += NTHREADS;
        }
    }
    __syncthreads();

    uint32_t i = 1<<(WBITS-NTHRBITS);
    row += tid * i;
    bucket_t acc = row[--i];
    scratch[tid] = acc;
    #pragma unroll 1
    while (i--) {
        bucket_t p = row[i];
        acc.add(p);
        scratch[tid].add(acc);
    }

    bucket_t res = scratch[tid];
    bias = wbits+NTHRBITS-WBITS;
    #pragma unroll 1
    for (int ic=0; ic < acc_level; ++ic)
    {
        uint32_t sid = tid & 0xfe;
        if ((tid & 1) == 0) {
            scratch[sid] = res;
            scratch[sid+1] = acc;
            return;
        }
        __syncthreads();

        tid >>= 1;
        res.add(scratch[sid]);
        if (ic < bias) {
            bucket_t raise = acc;
            for (size_t j = 0; j < WBITS-NTHRBITS+ic; j++)
                raise.dbl();
            res.add(raise);
            acc.add(scratch[sid+1]);
        }
        __syncthreads();
    }

    // if (ret == nullptr) {
    //     cooperative_groups::this_grid().sync();
    //     ret = reinterpret_cast<decltype(ret)>(buckets);
    // }

    ret[blockIdx.y][bid][tid][0] = res;
    ret[blockIdx.y][bid][tid][1] = acc;
}

__global__ void io_helper(
    fr_t* bases,
    fr_t* roots,
    const int32_t chunk_size,
    const int32_t chunk_num,
    const int32_t gap,
    const int32_t size
)
{
    int32_t index = blockIdx.x *blockDim.x + threadIdx.x;
    if ( index >= size)
        return;

    int32_t cid = index & (chunk_num - 1);
    int32_t rid = index & (~(chunk_num - 1));
    int32_t lid = index / chunk_num + cid * chunk_size;
    fr_t omega =  roots[rid & 0xff] * roots[(rid>>8) + (rid>0xff)*0xff];

    fr_t lo = bases[lid];
    fr_t hi = bases[lid + gap];
    fr_t neg = lo - hi;
    bases[lid] = lo + hi;
    // bases[lid + gap] = neg * roots[rid];
    bases[lid + gap] = neg * omega;
}

__global__ void oi_helper(
    fr_t* bases,
    fr_t* roots,
    const int32_t chunk_size,
    const int32_t chunk_num,
    const int32_t gap,
    const int32_t size
)
{
    int32_t index = blockIdx.x *blockDim.x + threadIdx.x;
    if ( index >= size)
        return;

    int32_t cid = index & (chunk_num - 1);
    int32_t rid = index & (~(chunk_num - 1));
    int32_t lid = index / chunk_num + cid * chunk_size;
    fr_t omega =  roots[rid & 0xff] * roots[(rid>>8) + (rid>0xff)*0xff];

    fr_t lo = bases[lid];
    fr_t hi = bases[lid + gap];
    // hi = hi * roots[rid];
    hi = hi * omega;
    bases[lid] = lo + hi;
    bases[lid + gap] = lo - hi;
}

#else

#include <cassert>
#include <vector>
using namespace std;

#include <util/exception.cuh>
#include <util/rusterror.h>
#include <util/thread_pool_t.hpp>
#include <util/host_pinned_allocator_t.hpp>

static point_t integrate_row(const bucket_t row[NTHREADS][2], int wbits = WBITS, int acc_level=0)
{
    size_t i = (NTHREADS>>acc_level)-1;
    size_t mask = (1U << max(wbits+NTHRBITS-WBITS-acc_level, 0)) - 1;

    if (mask == 0) {
        bucket_t res = row[i][0];
        while (i--)
            res.add(row[i][0]);
        return res;
    }

    point_t ret, res = row[i][0];
    bucket_t acc = row[i][1];
    ret.inf();
    while (i--) {
        point_t raise = acc;
        for (int j = 0; j < WBITS-NTHRBITS+acc_level; j++)
            raise.dbl();
        res.add(raise);
        res.add(row[i][0]);
        if (i & mask) {
            acc.add(row[i][1]);
        } else {
            ret.add(res);
            if (i-- == 0)
                break;
            res = row[i][0];
            acc = row[i][1];
        }
    }

    return ret;
}

#if 0
static point_t pippenger_final(const bucket_t ret[NWINS][NTHREADS][2])
{
    size_t i = NWINS-1;
    point_t res = integrate_row(ret[i], NBITS%WBITS ? NBITS%WBITS : WBITS);

    while (i--) {
        for (size_t j = 0; j < WBITS; j++)
            res.dbl();
        res.add(integrate_row(ret[i]));
    }

    return res;
}
#endif

template<typename... Types>
inline void launch_coop(void(*f)(Types...),
                        dim3 gridDim, dim3 blockDim, cudaStream_t stream,
                        Types... args)
{
    void* va_args[sizeof...(args)] = { &args... };
    CUDA_OK(cudaLaunchCooperativeKernel((const void*)f, gridDim, blockDim,
                                        va_args, 0, stream));
}


template<typename... Types>
inline void launch(void(*f)(Types...),
                        dim3 gridDim, dim3 blockDim, cudaStream_t stream,
                        Types... args)
{
    void* va_args[sizeof...(args)] = { &args... };
    CUDA_OK(cudaLaunchKernel((const void*)f, gridDim, blockDim, va_args, 0, stream));
}

class stream_t {
    cudaStream_t stream;
public:
    stream_t(int device)  {
        CUDA_OK(cudaSetDevice(device));
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    }
    ~stream_t() { cudaStreamDestroy(stream); }

    void init(int device)  {
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
        CUDA_OK(cudaSetDevice(device));
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    }
    inline operator decltype(stream)() { return stream; }
};

template<class bucket_t> class result_t {
    bucket_t ret[NWINS][NTHREADS][2];
public:
    result_t() {}
    inline operator decltype(ret)&() { return ret; }
};

template<class T>
class device_ptr_list_t {
    vector<T*> d_ptrs;
public:
    device_ptr_list_t() {}
    ~device_ptr_list_t() {
        for(T *ptr: d_ptrs) {
            cudaFree(ptr);
        }
    }
    size_t allocate(size_t bytes) {
        T *d_ptr;
        CUDA_OK(cudaMalloc(&d_ptr, bytes));
        d_ptrs.push_back(d_ptr);
        return d_ptrs.size() - 1;
    }
    size_t size() {
        return d_ptrs.size();
    }
    T* operator[](size_t i) {
        if (i > d_ptrs.size() - 1) {
            CUDA_OK(cudaErrorInvalidDevicePointer);
        }
        return d_ptrs[i];
    }
    
};
// TODO: Move to device class eventually
static thread_pool_t *da_pool = nullptr;
// Pippenger MSM class

template<class bucket_t, class point_t, class affine_t, class scalar_t>
class pippenger_t {
public:
    typedef vector<result_t<bucket_t>,
                   host_pinned_allocator_t<result_t<bucket_t>>> result_container_t;

private:
    size_t sm_count;
    bool init_done = false;
    device_ptr_list_t<affine_t> d_base_ptrs;
    device_ptr_list_t<scalar_t> d_scalar_ptrs;
    device_ptr_list_t<bucket_t> d_bucket_ptrs;
    int acc_level;
    // GPU device number
    int device;
    

public:
    // Default stream for operations
    stream_t default_stream;

    // Parameters for an MSM operation
    class MSMConfig {
        friend pippenger_t;
    public:
        size_t npoints;
    private:
        size_t N;
        size_t n;
        int acc_level;
    };

    pippenger_t() : default_stream(0) {
        device = 0;
        acc_level = 0;
    }

    pippenger_t(int _device)
        : default_stream(_device) {
        device = _device;
        acc_level = 0;
    }

    // Initialize instance. Throws cuda_error on error.
    void init(int device_index) {
        if (!init_done) {
            set_device(device_index);

            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess || prop.major < 7)
                CUDA_OK(cudaErrorInvalidDevice);
            sm_count = prop.multiProcessorCount;

            if (da_pool == nullptr) {
                da_pool = new thread_pool_t();
            }
            
            init_done = true;
        }
    }

    int get_device() {
        return device;
    }

    void set_device(int _device) {
        if (_device != device) {
            default_stream.init(_device);
            device = _device;
        }
    }

    void set_acc_level(int acc_level_) {
        acc_level = min(acc_level_, NTHRBITS);
    }
    
    // Initialize parameters for a specific size MSM. Throws cuda_error on error.
    MSMConfig init_msm(size_t npoints, size_t device_index) {
        init(device_index);
        
        MSMConfig config;
        config.npoints = npoints;

        config.n = (npoints+WARP_SZ-1) & ((size_t)0-WARP_SZ);
  
        config.N = 1;
        size_t delta = ((npoints+(config.N)-1)/(config.N)+WARP_SZ-1) & (0U-WARP_SZ);
        config.N = (npoints+delta-1) / delta;
        config.acc_level = acc_level;

        return config;
    }
    
    size_t get_size_bases(MSMConfig& config) {
        return config.n * sizeof(affine_t);
    }
    size_t get_size_scalars(MSMConfig& config) {
        return config.n * sizeof(scalar_t);
    }
    size_t get_size_buckets(MSMConfig& config) {
        return config.N * sizeof(bucket_t) * NWINS * (1 << WBITS);
    }

    size_t get_size_dones(MSMConfig& config) {
        return config.N * sizeof(bucket_t) * NWINS * NTHREADS * 2;
    }

    result_container_t get_result_container(MSMConfig& config) {
        result_container_t res(config.N);
        return res;
    }

    // Allocate storage for bases on device. Throws cuda_error on error.
    // Returns index of the allocated base storage.
    size_t allocate_d_bases(MSMConfig& config) {
        return d_base_ptrs.allocate(get_size_bases(config));
    }

    // Allocate storage for scalars on device. Throws cuda_error on error.
    // Returns index of the allocated base storage.
    size_t allocate_d_scalars(MSMConfig& config) {
        return d_scalar_ptrs.allocate(get_size_scalars(config));
    }

    // Allocate storage for buckets on device. Throws cuda_error on error.
    // Returns index of the allocated base storage.
    size_t allocate_d_buckets(MSMConfig& config) {
        return d_bucket_ptrs.allocate(get_size_buckets(config));
    }

    size_t allocate_d_dones(MSMConfig& config) {
        return d_bucket_ptrs.allocate(get_size_dones(config));
    }

    size_t allocate_d_ones(MSMConfig& config) {
        (void)config;
        return d_bucket_ptrs.allocate(16 * sizeof(bucket_t));
    }    

    size_t allocate_roots() {
        d_scalar_ptrs.allocate(128 * 1024 * sizeof(scalar_t));
        return d_scalar_ptrs.allocate(512 * sizeof(scalar_t));
    }

    size_t get_num_base_ptrs() {
        return d_base_ptrs.size();
    }
    size_t get_num_scalar_ptrs() {
        return d_scalar_ptrs.size();
    }
    size_t get_num_bucket_ptrs() {
        return d_bucket_ptrs.size();
    }

    void free_all_ptrs() {
         //d_base_ptrs.free();
         //d_scalar_ptrs.free();
         //d_bucket_ptrs.free();
    }


    // Transfer bases to device. Throws cuda_error on error.
    void transfer_bases_to_device(MSMConfig& config, size_t d_bases_idx, const affine_t points[],
                                  size_t ffi_affine_sz = sizeof(affine_t),
                                  cudaStream_t s = nullptr) {
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        affine_t *d_points = d_base_ptrs[d_bases_idx];
        CUDA_OK(cudaSetDevice(device));
        size_t d_affine_sz = sizeof(*d_points);
        if (ffi_affine_sz != d_affine_sz)
            CUDA_OK(cudaMemcpy2DAsync(d_points, d_affine_sz,
                                      points, ffi_affine_sz,
                                      d_affine_sz, config.npoints,
                                      cudaMemcpyHostToDevice, stream));
        else
            CUDA_OK(cudaMemcpyAsync(d_points, points, config.npoints*sizeof(*d_points),
                                    cudaMemcpyHostToDevice, stream));
    }

    // Transfer scalars to device. Throws cuda_error on error.
    void transfer_scalars_to_device(MSMConfig& config,
                                    size_t d_scalars_idx, const scalar_t scalars[],
                                    cudaStream_t s = nullptr) {
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        scalar_t *d_scalars = d_scalar_ptrs[d_scalars_idx];
        CUDA_OK(cudaSetDevice(device));
        CUDA_OK(cudaMemcpyAsync(d_scalars, scalars, config.npoints*sizeof(*d_scalars),
                                cudaMemcpyHostToDevice, stream));
    }

    // Transfer buckets from device. Throws cuda_error on error.
    void transfer_buckets_to_host(MSMConfig& config,
                                  result_container_t &res, size_t d_buckets_idx,
                                  cudaStream_t s = nullptr) {
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        bucket_t *d_buckets = d_bucket_ptrs[d_buckets_idx];
        CUDA_OK(cudaSetDevice(device));
        CUDA_OK(cudaMemcpyAsync(res[0], d_buckets, config.N*sizeof(res[0]),
                                cudaMemcpyDeviceToHost, stream));
    }

    // Transfer buckets from device. Throws cuda_error on error.
    void transfer_res_to_host(MSMConfig& config,
                        bucket_t *res, size_t d_buckets_idx,
                        size_t npoints) {
        (void)config;
        CUDA_OK(cudaMemcpyAsync(res, d_bucket_ptrs[d_buckets_idx], 
        npoints*sizeof(bucket_t), cudaMemcpyDeviceToHost, default_stream));
    }

    void synchronize_stream() {
        CUDA_OK(cudaSetDevice(device));
        CUDA_OK(cudaStreamSynchronize(default_stream));
    }
    
    // Perform accumulation into buckets on GPU. Throws cuda_error on error.
    void launch_kernel(MSMConfig& config,
                       size_t d_bases_idx, size_t d_scalars_idx, size_t d_buckets_idx, size_t d_dones_idx,
                       bool mont = true, cudaStream_t s = nullptr)
    {
        assert(WBITS > NTHRBITS);
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        const affine_t *d_points = d_base_ptrs[d_bases_idx];
        scalar_t *d_scalars = d_scalar_ptrs[d_scalars_idx];
        bucket_t (*d_buckets)[NWINS][1<<WBITS] =
            reinterpret_cast<decltype(d_buckets)>(d_bucket_ptrs[d_buckets_idx]);

        bucket_t (*d_none)[NWINS][NTHREADS][2] = 
            reinterpret_cast<decltype(d_none)>(d_bucket_ptrs[d_dones_idx]);
        
        CUDA_OK(cudaSetDevice(device));

        if (config.npoints <32000) {
            bucket_t *d_ones = d_bucket_ptrs[d_dones_idx+1];
            launch(pre_scalars_one, 16, NTHREADS, stream, d_ones, d_points, d_scalars, config.npoints);
        } else {
            launch(pre_scalars, sm_count, NTHREADS, stream, d_scalars, config.npoints);
        }

        launch(pippenger, dim3(NWINS, config.N), NTHREADS, stream,
                    d_points, config.npoints,
                    (const scalar_t*)d_scalars, mont,
                    d_buckets, d_none, config.acc_level);
    }
    
    // Perform final accumulation on CPU.
    void accumulate(MSMConfig& config, point_t &out, result_container_t &res, bucket_t h_ones[]) {
        struct tile_t {
            size_t x, y, dy;
            point_t p;
            tile_t() {}
        };
        vector<tile_t> grid(NWINS*config.N);
        
        size_t y = NWINS-1, total = 0;
        
        while (total < config.N) {
            grid[total].x  = total;
            grid[total].y  = y;
            grid[total].dy = NBITS - y*WBITS;
            total++;
        }
        
        while (y--) {
            for (size_t i = 0; i < config.N; i++, total++) {
                grid[total].x  = grid[i].x;
                grid[total].y  = y;
                grid[total].dy = WBITS;
            }
        }
        
        vector<atomic<size_t>> row_sync(NWINS); /* zeroed */
        counter_t<size_t> counter(0);
        channel_t<size_t> ch;
        
        auto n_workers = min(da_pool->size(), total);
        while (n_workers--) {
            da_pool->spawn([&, total, counter]() {
                for (size_t work; (work = counter++) < total;) {
                    auto item = &grid[work];
                    auto y = item->y;
                    item->p = integrate_row((res[item->x])[y], item->dy, config.acc_level);
                    if (++row_sync[y] == config.N)
                        ch.send(y);
                }
            });
        }
        

        bucket_t all_ones;
        all_ones.inf();
        if (config.npoints <32000) {
            for (size_t i=0; i<16; i++) {
                all_ones.add(h_ones[i]);
            }
        }
        out.inf();
        size_t row = 0, ny = NWINS;
        while (ny--) {
            auto y = ch.recv();
            row_sync[y] = -1U;
            while (grid[row].y == y) {
                while (row < total && grid[row].y == y)
                    out.add(grid[row++].p);
                if (y == 0)
                    break;
                for (size_t i = 0; i < WBITS; i++)
                    out.dbl();
                if (row_sync[--y] != -1U)
                    break;
            }
        }

        out.add(all_ones);
    }

    void launch_io_helper(scalar_t inout[], const scalar_t root[], int npoints)
    {
        try {
        CUDA_OK(cudaSetDevice(device));

        int dsize = npoints >> 1;
        int group = dsize >> 8;
        if ((dsize & 0xff) > 0)
            group += 1;

        scalar_t roots[512];
        roots[0] = scalar_t::one();
        for (int i=1; i<512; i++) {
            if ((i-256) * 256 > dsize)
                break;
            roots[i] = roots[i-1] * (i<=256 ? root[0]: roots[256]);            
        }
        scalar_t *d_scalars = d_scalar_ptrs[2];
        scalar_t *d_roots = d_scalar_ptrs[3];

        CUDA_OK(cudaMemcpyAsync(d_scalars, inout, npoints*sizeof(scalar_t),cudaMemcpyHostToDevice, default_stream));
        CUDA_OK(cudaMemcpyAsync(d_roots, roots, 512*sizeof(scalar_t),cudaMemcpyHostToDevice, default_stream));

        int gap = dsize;
        while (gap > 0) {
            int chunk_size = gap << 1;
            int chunk_num = npoints / chunk_size;
            launch(io_helper, group, 256, default_stream, d_scalars, d_roots, chunk_size, chunk_num, gap, dsize);
            gap >>= 1;
        }

        CUDA_OK(cudaMemcpyAsync(inout, d_scalars, npoints*sizeof(scalar_t),  cudaMemcpyDeviceToHost, default_stream));
        synchronize_stream();

        } catch (const cuda_error& e) {
            printf("fio error %s\n", e.what());
        }
    }


    void launch_oi_helper(scalar_t inout[], const scalar_t root[], int npoints)
    {
        try {
        CUDA_OK(cudaSetDevice(device));

        int dsize = npoints >> 1;
        int group = dsize >> 8;
        if ((dsize & 0xff) > 0)
            group += 1;

        scalar_t roots[512];
        roots[0] = scalar_t::one();
        for (int i=1; i<512; i++) {
            if ((i-256) * 256 > dsize)
                break;
            roots[i] = roots[i-1] * (i<=256 ? root[0]: roots[256]);            
        }
        scalar_t *d_scalars = d_scalar_ptrs[2];
        scalar_t *d_roots = d_scalar_ptrs[3];

        CUDA_OK(cudaMemcpyAsync(d_scalars, inout, npoints*sizeof(scalar_t),cudaMemcpyHostToDevice, default_stream));
        CUDA_OK(cudaMemcpyAsync(d_roots, roots, 512*sizeof(scalar_t),cudaMemcpyHostToDevice, default_stream));

        int gap = 1;
        while (gap < npoints) {
            int chunk_size = gap << 1;
            int chunk_num = npoints / chunk_size;
            launch(oi_helper, group, 256, default_stream, d_scalars, d_roots, chunk_size, chunk_num, gap, dsize);
            gap <<= 1;
        }

        CUDA_OK(cudaMemcpyAsync(inout, d_scalars, npoints*sizeof(scalar_t),  cudaMemcpyDeviceToHost, default_stream));
        synchronize_stream();

        } catch (const cuda_error& e) {
            printf("foi error %s\n", e.what());
        }
    }




    // Perform a full MSM computation.
    RustError msm(point_t &out,
                  const affine_t points[], size_t npoints,
                  const scalar_t scalars[], bool mont = true,
                  size_t ffi_affine_sz = sizeof(affine_t)) {
        bucket_t (*d_none)[NWINS][NTHREADS][2] = nullptr;

        try {
            MSMConfig config = init_msm(npoints, 0);

            size_t d_bases = 0;
            size_t d_scalars = 0;
            size_t d_buckets = 0;
            size_t d_dones = 1;

            // Ensure device buffers are allocated
            if (get_num_base_ptrs() == 0) {
                allocate_d_bases(config);
            }
            if (get_num_scalar_ptrs() == 0) {
                allocate_d_scalars(config);
            }
            if (get_num_bucket_ptrs() == 0) {
                allocate_d_buckets(config);
                allocate_d_dones(config);
            }
            
            transfer_bases_to_device(config, d_bases, points, ffi_affine_sz);
            transfer_scalars_to_device(config, d_scalars, scalars);
            launch_kernel(config, d_bases, d_scalars, d_buckets, d_dones, mont);

            result_container_t res(config.N);
            transfer_buckets_to_host(config, res, d_dones);
            synchronize_stream();

            accumulate(config, out, res);
        } catch (const cuda_error& e) {
            synchronize_stream();
            out.inf();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()}
#endif
        }
        
        return RustError{cudaSuccess};
    }
};

#endif
