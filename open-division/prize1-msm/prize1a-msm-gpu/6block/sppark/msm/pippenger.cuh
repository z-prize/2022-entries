// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>

#ifndef WARP_SZ
# define WARP_SZ 32
#endif

#ifndef NTHREADS
# define NTHREADS 256
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
# define WBITS 17
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
__global__
void pippenger(const affine_t* points, size_t npoints,
               const scalar_t* scalars, bool mont,
               bucket_t (*buckets)[NWINS][1<<WBITS],
               bucket_t (*ret)[NWINS][NTHREADS][2] = nullptr);

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
void pippenger(const affine_t* points, size_t npoints,
               const scalar_t* scalars_, bool mont,
               bucket_t (*buckets)[NWINS][1<<WBITS],
               bucket_t (*ret)[NWINS][NTHREADS][2] /*= nullptr*/)
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

    scalars_T scalars = const_cast<scalar_t*>(scalars_);

    const int NTHRBITS = dlog2(NTHREADS);
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    const uint32_t bit0 = bid * WBITS;
    bucket_t* row = buckets[blockIdx.y][bid];

    if (mont) {
        #pragma unroll 1
        for (uint32_t i = NTHREADS*bid + tid; i < npoints; i += NTHREADS*NWINS) {
            scalar_t s = scalars_[i];
            s.from();
            scalars[i] = s;
        }
        cooperative_groups::this_grid().sync();
    } else { // if (typeid(scalars) != typeid(scalars_)) {
        #pragma unroll 1
        for (uint32_t i = NTHREADS*bid + tid; i < npoints; i += NTHREADS*NWINS) {
            scalar_t s = scalars_[i];
            __syncwarp();
            scalars[i] = s;
        }
        cooperative_groups::this_grid().sync();
    }

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
            wval = get_wval(scalars[i], bit0, wbits);
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
    if (NTHREADS > WARP_SZ && sizeof(bucket_t) > 128)
        __syncthreads();

    uint32_t i = 1<<(WBITS-NTHRBITS);
    row += tid * i;
    bucket_t res, acc = row[--i];
    if (sizeof(res) <= 128)
        res = acc;
    else
        scratch[tid] = acc;
    #pragma unroll 1
    while (i--) {
        bucket_t p = row[i];
        #pragma unroll 1
        for (int pc = 0; pc < 2; pc++) {
            if (sizeof(res) <= 128) {
                acc.add(p);
                p = res;
                res = acc;
            } else {
                acc.add(p);
                p = scratch[tid];
                scratch[tid] = acc;
            }
        }
        acc = p;
    }

    if (ret == nullptr) {
        cooperative_groups::this_grid().sync();
        ret = reinterpret_cast<decltype(ret)>(buckets);
    }

    if (sizeof(res) <= 128)
        ret[blockIdx.y][bid][tid][0] = res;
    else
        ret[blockIdx.y][bid][tid][0] = scratch[tid];
    ret[blockIdx.y][bid][tid][1] = acc;
}

#else

#include <cassert>
#include <vector>
using namespace std;

#include <util/exception.cuh>
#include <util/rusterror.h>
#include <util/thread_pool_t.hpp>
#include <util/host_pinned_allocator_t.hpp>

static point_t integrate_row(const bucket_t row[NTHREADS][2], int wbits = WBITS)
{
    size_t i = NTHREADS-1;
    size_t mask = (1U << max(wbits+NTHRBITS-WBITS, 0)) - 1;

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
        for (size_t j = 0; j < WBITS-NTHRBITS; j++)
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

class stream_t {
    cudaStream_t stream;
public:
    stream_t(int device)  {
        CUDA_OK(cudaSetDevice(device));
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    }
    ~stream_t() { cudaStreamDestroy(stream); }
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

    // GPU device number
    int device;

    // TODO: Move to device class eventually
    thread_pool_t *da_pool = nullptr;

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
    };

    pippenger_t() : default_stream(0) {
        device = 0;
    }

    pippenger_t(int _device, thread_pool_t *pool = nullptr)
        : default_stream(_device) {
        da_pool = pool;
        device = _device;
    }

    // Initialize instance. Throws cuda_error on error.
    void init() {
        if (!init_done) {
            CUDA_OK(cudaSetDevice(device));
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

    // Initialize parameters for a specific size MSM. Throws cuda_error on error.
    MSMConfig init_msm(size_t npoints) {
        init();

        MSMConfig config;
        config.npoints = npoints;

        config.n = (npoints+WARP_SZ-1) & ((size_t)0-WARP_SZ);

        config.N = (sm_count*256) / (NTHREADS*NWINS);
        size_t delta = ((npoints+(config.N)-1)/(config.N)+WARP_SZ-1) & (0U-WARP_SZ);
        config.N = (npoints+delta-1) / delta;

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

    size_t get_num_base_ptrs() {
        return d_base_ptrs.size();
    }
    size_t get_num_scalar_ptrs() {
        return d_scalar_ptrs.size();
    }
    size_t get_num_bucket_ptrs() {
        return d_bucket_ptrs.size();
    }

    // Transfer bases to device. Throws cuda_error on error.
    void transfer_bases_to_device(MSMConfig& config, size_t d_bases_idx, const affine_t points[],
                                  size_t ffi_affine_sz = sizeof(affine_t),
                                  cudaStream_t s = nullptr) {
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        affine_t *d_points = d_base_ptrs[d_bases_idx];
        CUDA_OK(cudaSetDevice(device));
        if (ffi_affine_sz != sizeof(*d_points))
            CUDA_OK(cudaMemcpy2DAsync(d_points, sizeof(*d_points),
                                      points, ffi_affine_sz,
                                      ffi_affine_sz, config.npoints,
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

    void synchronize_stream() {
        CUDA_OK(cudaSetDevice(device));
        CUDA_OK(cudaStreamSynchronize(default_stream));
    }

    // Perform accumulation into buckets on GPU. Throws cuda_error on error.
    void launch_kernel(MSMConfig& config,
                       size_t d_bases_idx, size_t d_scalars_idx, size_t d_buckets_idx,
                       bool mont = true, cudaStream_t s = nullptr)
    {
        assert(WBITS > NTHRBITS);
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        affine_t *d_points = d_base_ptrs[d_bases_idx];
        scalar_t *d_scalars = d_scalar_ptrs[d_scalars_idx];
        bucket_t (*d_buckets)[NWINS][1<<WBITS] =
            reinterpret_cast<decltype(d_buckets)>(d_bucket_ptrs[d_buckets_idx]);

        bucket_t (*d_none)[NWINS][NTHREADS][2] = nullptr;

        CUDA_OK(cudaSetDevice(device));
        launch_coop(pippenger, dim3(NWINS, config.N), NTHREADS, stream,
                    (const affine_t*)d_points, config.npoints,
                    (const scalar_t*)d_scalars, mont,
                    d_buckets, d_none);
    }

    // Perform final accumulation on CPU.
    void accumulate(MSMConfig& config, point_t &out, result_container_t &res) {
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
                    item->p = integrate_row((res[item->x])[y], item->dy);
                    if (++row_sync[y] == config.N)
                        ch.send(y);
                }
            });
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
    }

    // Perform a full MSM computation.
    RustError msm(point_t &out,
                  const affine_t points[], size_t npoints,
                  const scalar_t scalars[], bool mont = true,
                  size_t ffi_affine_sz = sizeof(affine_t)) {
        bucket_t (*d_none)[NWINS][NTHREADS][2] = nullptr;

        try {
            MSMConfig config = init_msm(npoints);

            size_t d_bases = 0;
            size_t d_scalars = 0;
            size_t d_buckets = 0;

            // Ensure device buffers are allocated
            if (get_num_base_ptrs() == 0) {
                allocate_d_bases(config);
            }
            if (get_num_scalar_ptrs() == 0) {
                allocate_d_scalars(config);
            }
            if (get_num_bucket_ptrs() == 0) {
                allocate_d_buckets(config);
            }

            transfer_bases_to_device(config, d_bases, points, ffi_affine_sz);
            transfer_scalars_to_device(config, d_scalars, scalars);
            launch_kernel(config, d_bases, d_scalars, d_buckets, mont);

            result_container_t res(config.N);
            transfer_buckets_to_host(config, res, d_buckets);
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
