// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>


#ifndef WARP_SZ
# define WARP_SZ 32
#endif

#ifndef NTHREADS
# define NTHREADS 32
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
# define WBITS 21
#endif
#define NWINS 13  // ((NBITS+WBITS-1)/WBITS)   // ceil(NBITS/WBITS)

#ifndef LARGE_L1_CODE_CACHE
# define LARGE_L1_CODE_CACHE 0
#endif


/*
[in]  scalars_: randomly sampled BLS12-377 scalars.
[in]  nscalars: the number of scalars. 
[out] wval: 'NWINS' groups of sub-scalars. 
    (allocated in the GPU global memory and each group has 'nscalars' elements)
[out] idx:  'NWINS' groups of scalar indexes.
    (allocated in the GPU global memory and each group has 'nscalars' elements)
*/
__global__
void pippenger_faster_1(const scalar_t* scalars, size_t nscalars,
               bucket_t (*buckets)[1<<WBITS], uint32_t *wval, uint32_t *idx);

/*
[in]  points: BLS12-377 points.
[in]  npoints: The number of BLS12-377 points.
[in]  wval_out: 'NWINS - 1' groups of sorted sub-scalars. 
[in]  idx_out: 'NWINS - 1' groups of scalar indexes.
[in]  wval_count: the number of sub-scalars with the same value in each group of sorted sub-scalars.
[in]  wval_ptr: the prefix sum of the "wval_count".  
[in]  wval_unique: 'NWINS - 1' groups of unique sub-scalars.  
[in]  wval_unique_count: the number of elements in each group of unique sub-scalars. 
[out] buckets: (NWINS - 1) groups of buckets used to store BLS12-377 points. 
        (allocated in the GPU global memory, and each group has (1 << WBITS) buckets)
*/
__global__
void pippenger_faster_2(affine_t* points, size_t npoints,
               bucket_t (*buckets)[1<<WBITS],
               uint32_t *wval_out,
               uint32_t *idx_out,
               uint32_t *wval_count,
               uint32_t *wval_unique,
               uint32_t *wval_ptr,
               uint32_t *wval_unique_count);

/*
[in]  buckets: (NWINS - 1) groups of buckets used to store BLS12-377 points. 
[out] tres: the calculation results of all threads individually
    (allocated in the GPU global memory with '(NWINS - 1) * N * NTHREADS' elements)
*/
__global__
void pippenger_faster_3(bucket_t (*buckets)[1<<WBITS], bucket_t* tres);

/*
[in]  points: BLS12-377 points.
[in]  npoints: the number of BLS12-377 points.
[in]  s_count: the number of selected sub-scalars.
[in]  idx_out: the corresponding scalar indexes to the selected sub-scalars.
[out] tres:  the calculation results of all threads individually.
    (allocated in the GPU global memory with '(NWINS - 1) * N * NTHREADS' elements)
*/
__global__
void pippenger_faster_4(affine_t* points, size_t npoints,
               uint32_t *s_count, uint32_t *idx, bucket_t* tres);

/*
[in]  tres: the calculation results of all threads individually.
[out] res: results of the first 'NWINS - 1' subtasks.  
*/
__global__ 
void CSum(bucket_t *res, bucket_t *tres);

/*
[in]  tres: the calculation results of all threads individually.
[out] res: the result of the last subtask.
*/
__global__ 
void LSum(bucket_t *res, bucket_t *tres);

#ifdef __CUDA_ARCH__

#include <cooperative_groups.h>

static __shared__ bucket_t scratch[NTHREADS];
static __shared__ affine_t spoints[NTHREADS];
static __shared__ bucket_t accs[NTHREADS];
static __shared__ bucket_t ress[NTHREADS];

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


static __device__ uint32_t max_bits(uint32_t scalar)
{
    uint32_t max = 32;
    return max;
}

static __device__ bool test_bit(uint32_t scalar, uint32_t bitno)
{
    if (bitno >= 32)
        return false;
    return ((scalar >> bitno) & 0x1);
}

template<class bucket_t>
static __device__ void mul(bucket_t& res, const bucket_t& base, uint32_t scalar)
{
    res.inf();

    bool found_one = false;
    uint32_t mb = max_bits(scalar);
    for (int32_t i = mb - 1; i >= 0; --i)
    {
        if (found_one)
        {
            res.add(res);
        }

        if (test_bit(scalar, i))
        {
            found_one = true;
            res.add(base);
        }
    }
}

/*
[in]  scalars_: randomly sampled BLS12-377 scalars.
[in]  nscalars: the number of scalars. 
[out] wval: 'NWINS' groups of sub-scalars. 
    (allocated in the GPU global memory and each group has 'nscalars' elements)
[out] idx:  'NWINS' groups of scalar indexes.
    (allocated in the GPU global memory and each group has 'nscalars' elements)
*/
__global__
void pippenger_faster_1(const scalar_t* scalars_, size_t nscalars, bucket_t (*buckets)[1<<WBITS],
               uint32_t *wval,
               uint32_t *idx)
{
    const uint32_t tnum = blockDim.x * gridDim.x;
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint32_t i = tid; i < nscalars; i += tnum) 
    {
        for(uint32_t j = 0; j < NWINS; j++)
        {
            uint32_t bit0 = j * WBITS;
            uint32_t wbits = (bit0 > NBITS-WBITS) ? NBITS-bit0 : WBITS;
            uint32_t w = get_wval(scalars_[i], bit0, wbits);
            uint32_t ptr = j * nscalars;
            wval[ptr + i] = w;
            idx[ptr + i] = i;
        }
    }

    for(uint32_t j = 0; j < NWINS - 1; j++)
    {
        bucket_t* row = buckets[j];
        for (uint32_t i = tid; i < (1<<WBITS); i+= tnum)
            row[i].inf();
    }
}

/*
[in]  points: BLS12-377 points.
[in]  npoints: The number of BLS12-377 points.
[in]  wval_out: 'NWINS - 1' groups of sorted sub-scalars. 
[in]  idx_out: 'NWINS - 1' groups of scalar indexes.
[in]  wval_count: the number of sub-scalars with the same value in each group of sorted sub-scalars.
[in]  wval_ptr: the prefix sum of the "wval_count".  
[in]  wval_unique: 'NWINS - 1' groups of unique sub-scalars.  
[in]  wval_unique_count: the number of elements in each group of unique sub-scalars. 
[out] buckets: (NWINS - 1) groups of buckets used to store BLS12-377 points. 
        (allocated in the GPU global memory, and each group has (1 << WBITS) buckets)
*/
__global__
void pippenger_faster_2(affine_t* points, size_t npoints,
               bucket_t (*buckets)[1<<WBITS],
               uint32_t *wval_out,
               uint32_t *idx_out,
               uint32_t *wval_count,
               uint32_t *wval_unique,
               uint32_t *wval_ptr,
               uint32_t *wval_unique_count)
{
    const uint32_t tnum = blockDim.x * gridDim.y;
    const uint32_t tid = blockIdx.y * blockDim.x + threadIdx.x;
    const uint32_t bid = blockIdx.x;
    bucket_t* row = buckets[bid];

    uint32_t ptr = bid * npoints;
    uint32_t cptr = bid * (1 << WBITS);
    uint32_t pptr = bid * (1 << WBITS);

    /* Each thread processes the BLS12-377 points whose corresponding sub-scalars 
       are 's'-th to 'e'-th sub-scalars in 'bid'-th group of sorted sub-scalars. */
    uint32_t s = (npoints + tnum - 1) / tnum * tid;
    uint32_t e = (npoints + tnum - 1) / tnum * (tid + 1);
    if(s >= npoints) s = npoints;
    if(e >= npoints) e = npoints;

    /* Adjust 's' and 'e' to garantee that different threads will not put points into the same bucket. */
    if(s != 0)
    {
        while(s < npoints && wval_out[ptr + s] == wval_out[ptr + s - 1])
            s++;
    }
    while(e < npoints && wval_out[ptr + e] == wval_out[ptr + e - 1])
        e++;

    /* 'is' is the 's'-th sub-scalar in 'bid'-th group of sorted sub-scalar, and
       'ie' is the 'e'-th sub-scalar in 'bid'-th group of sorted sub-scalar */
    int32_t is, ie;
    if(s == npoints) is = (1 << WBITS);
    else is =  wval_out[ptr + s];
    if(e == npoints) ie = (1 << WBITS);
    else ie =  wval_out[ptr + e];

    /* the 'iis'-th sub-scalar in 'bid'-th group of unique sub-scalars is equal to 'is', and
       the 'iie'-th sub-scalar in 'bid'-th group of unique sub-scalars is equal to 'ie' */
    int32_t unique_count = (int32_t)wval_unique_count[bid];
    int32_t unique_sub = (1 << WBITS) - unique_count;
    int32_t iis = is - unique_sub;
    int32_t iie = ie - unique_sub;
    if(iis < 0) iis = 0;
    if(iie < 0) iie = 0;
    while(iis != unique_count && wval_unique[pptr + iis] != is) iis++;
    while(iie != unique_count && wval_unique[pptr + iie] != ie) iie++;


    /* Put BLS12-377 points with the same corresponding sub-scalars into the same bucket. */
    /* Each thread processes the BLS12-377 points whose coressponding sub-scalars 
       have the same value as 'iis'-th to 'iie'-th sub-scalars in 'bid'-th group of unique sub-scalars
       , which is equivalent to that each thread processes the BLS12-377 points whose corresponding sub-scalars 
       is 's'-th to 'e'-th sub-scalars in 'bid'-th group of sorted sub-scalars. */
    /* Note: this step helps to ensure load balancing and prevent data contention across all of threads. */
    uint32_t i = iis;
    uint32_t wcount;
    uint32_t wptr;
    uint32_t w;
    bucket_t mid;
    mid.inf();
    if(i < iie)
    {
        wcount = wval_count[cptr + i];
        wptr = wval_ptr[pptr + i];
        w = wval_out[ptr + wptr - wcount];
    }

    while(i < iie)
    {
        if(wcount == 0)
        {
            wcount = wval_count[cptr + i];
            wptr += wcount; 
            w = wval_out[ptr + wptr - wcount];
            mid.inf();
        }

        uint32_t j = wptr - wcount;
        uint32_t index = idx_out[ptr + j];
        spoints[threadIdx.x] = points[index];
        mid.add(spoints[threadIdx.x]);
        wcount--;
        if(wcount == 0)
        {
            row[w] = mid;
            i++;
        }
    }
}

/*
[in]  buckets: (NWINS - 1) groups of buckets used to store BLS12-377 points. 
[out] tres: the calculation results of all threads individually
    (allocated in the GPU global memory with '(NWINS - 1) * N * NTHREADS' elements)
*/
__global__
void pippenger_faster_3(bucket_t (*buckets)[1<<WBITS],
               bucket_t* tres)
{
    const uint32_t tnum = blockDim.x * gridDim.y;
    const uint32_t tid = blockIdx.y * blockDim.x + threadIdx.x;
    const uint32_t bid = blockIdx.x;
    bucket_t* row = buckets[bid];

    /* each thread processes 's'-th to 'e'-th buckets in 'bid'-th group of buckets. */
    int32_t s = ((1<<WBITS) + tnum - 1) / tnum * tid;
    int32_t e = ((1<<WBITS) + tnum - 1) / tnum * (1 + tid);
    if(s > (1 << WBITS)) s = 1<<WBITS;
    if(e > (1 << WBITS)) e = 1<<WBITS;

    /*  after the following steps, 
        accs[threadIdx.x] = row[e - 1] + row[e - 2] + ... + row[s + 1] + row[s]
        ress[threadIdx.x] = (e - s) * row[e - 1] + (e - s + 1)row[e - 2] + ... + 2 * row[s + 1] + row[s]
    */
    ress[threadIdx.x].inf();
    accs[threadIdx.x].inf();
    for(int32_t i = e - 1; i >= s && i > 0; i-- )
    {
        scratch[threadIdx.x] = row[i];
        accs[threadIdx.x].add(scratch[threadIdx.x]);
        ress[threadIdx.x].add(accs[threadIdx.x]);
    }

    /*  after the following steps, 
        smid[threadIdx.x] = (s - 1) * accs[threadIdx.x]
                          = (s - 1) * row[e - 1] + (s - 1) * row[e - 2] + ... + (s - 1) * row[s + 1] + (s - 1) * row[s]
        ress[threadIdx.x] = (e - 1) * row[e - 1] + (e - 2)row[e - 2] + ... + (s + 1) * row[s + 1] + s * row[s]
    */
    if(s != 0)
    {
        mul(scratch[threadIdx.x], accs[threadIdx.x], (s - 1));
        ress[threadIdx.x].add(scratch[threadIdx.x]);
    }

    uint32_t bptr = bid * tnum;
    tres[bptr + tid] = ress[threadIdx.x];

}

/*
[in]  points: BLS12-377 points.
[in]  npoints: the number of BLS12-377 points.
[in]  s_count: the number of selected sub-scalars.
[in]  idx_out: the corresponding scalar indexes to the selected sub-scalars.
[out] tres:  the calculation results of all threads individually.
    (allocated in the GPU global memory with '(NWINS - 1) * N * NTHREADS' elements)
*/
__global__
void pippenger_faster_4(affine_t* points, size_t npoints,
               uint32_t *s_count, uint32_t *idx, bucket_t* tres)
{
    const uint32_t tnum = blockDim.x * gridDim.y;
    const uint32_t tid = blockIdx.y * blockDim.x + threadIdx.x;
    const uint32_t bnum = gridDim.x;
    const uint32_t bid = blockIdx.x;

    uint32_t ptr = (NWINS - 1) * npoints;

    /* each thread group process "bs"-th to "be"-th selected sub-scalars. */
    uint32_t total = *s_count;
    uint32_t bs = (total + bnum - 1) / bnum * bid;
    uint32_t be = (total + bnum - 1) / bnum * (bid + 1);
    if(bs > total) bs = total;
    if(be > total) be = total;

    bucket_t mid;
    mid.inf();
    for(uint32_t i = bs + tid; i < be; i += tnum)
    {
        uint32_t index = idx[ptr + i];
        spoints[threadIdx.x] = points[index];
        mid.add(spoints[threadIdx.x]);
    }
    uint32_t bptr = bid * tnum;
    tres[bptr + tid] = mid;
}


/*
[in]  tres: the calculation results of all threads individually.
[out] res: the result of the first 'NWINS - 1' subtask.
*/
__global__ void CSum(bucket_t *res, bucket_t *tres)
{
    const uint32_t tnum = blockDim.x * gridDim.y;
    const uint32_t tid = blockIdx.y * blockDim.x + threadIdx.x;
    const uint32_t bid = blockIdx.x;

    uint32_t ptr = bid * tnum * 2;
    uint32_t cur = 1;
    while(2 * tnum > cur)
    {
        if((tid % cur == 0) && (tid * 2 + cur < 2 * tnum))
        {
            tres[ptr + tid * 2].add(tres[ptr + tid * 2 + cur]);
        }
        cur *= 2;
        cooperative_groups::this_grid().sync();
    }
    if(tid == 0) res[bid] = tres[ptr];
}

/*
[in]  tres: the calculation results of all threads individually.
[out] res: the result of the last subtask.
*/
__global__ void LSum(bucket_t *res, bucket_t *tres)
{
    const uint32_t tnum = blockDim.x * gridDim.y;
    const uint32_t tid = blockIdx.y * blockDim.x + threadIdx.x;
    const uint32_t bid = blockIdx.x;

    uint32_t ptr = bid * tnum * 2;
    uint32_t cur = 1;
    while(2 * tnum > cur)
    {
        if((tid % cur == 0) && (tid * 2 + cur < 2 * tnum))
        {
            tres[ptr + tid * 2].add(tres[ptr + tid * 2 + cur]);
        }
        cur *= 2;
        cooperative_groups::this_grid().sync();
    }

    cooperative_groups::this_grid().sync();
    if(bid == 0 && tid == 0)
    {
        res[NWINS - 1].inf();
        for(uint32_t i=0; i < NWINS - 1; i++) 
            res[NWINS - 1].add(tres[i * tnum * 2]);
    }
        
}

#else

#include <cassert>
#include <vector>
using namespace std;

#include <util/exception.cuh>
#include <util/rusterror.h>
#include <util/thread_pool_t.hpp>
#include <util/host_pinned_allocator_t.hpp>


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


template<class bucket_t> class result_t_faster {
    bucket_t ret[NWINS];
public:
    result_t_faster() {}
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
    typedef vector<result_t_faster<bucket_t>,
                   host_pinned_allocator_t<result_t_faster<bucket_t>>> result_container_t_faster;

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

    device_ptr_list_t<uint32_t> d_wval_ptrs;
    device_ptr_list_t<uint32_t> d_idx_ptrs;
    device_ptr_list_t<unsigned char> d_cub_ptrs;

    // Parameters for an MSM operation
    class MSMConfig {
        friend pippenger_t;
    public:
        size_t npoints;
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
    MSMConfig init_msm_faster(size_t npoints) {
        init();

        MSMConfig config;
        config.npoints = npoints;
        config.n = (npoints+WARP_SZ-1) & ((size_t)0-WARP_SZ);
        config.N = (sm_count*256) / (NTHREADS*(NWINS-1));
        size_t delta = ((npoints+(config.N)-1)/(config.N)+WARP_SZ-1) & (0U-WARP_SZ);
        config.N = (npoints+delta-1) / delta;

        if(config.N % 2 == 1) config.N -= 1;
        return config;
    }

    size_t get_size_bases(MSMConfig& config) {
        return config.n * sizeof(affine_t);
    }
    size_t get_size_scalars(MSMConfig& config) {
        return config.n * sizeof(scalar_t);
    }

    size_t get_size_wval_faster(MSMConfig& config) {
        return config.n * sizeof(uint32_t) * NWINS;
    }
    size_t get_size_idx_faster(MSMConfig& config) {
        return config.n * sizeof(uint32_t) * NWINS;
    }

    size_t get_size_wval_count_faster() {
        return sizeof(uint32_t) * (NWINS - 1) * (1 << WBITS);
    }
    size_t get_size_wval_ptr_faster() {
        return sizeof(uint32_t) * (NWINS - 1) * (1 << WBITS);
    }
    size_t get_size_wval_unique_faster() {
        return sizeof(uint32_t) * (NWINS - 1) * (1 << WBITS);
    }
    size_t get_size_wval_unique_count_faster() {
        return sizeof(uint32_t) * (NWINS - 1);
    }

    size_t get_size_buckets_faster() {
        return sizeof(bucket_t) * (NWINS - 1) * (1 << WBITS);
    }
    size_t get_size_tres_faster(MSMConfig& config) {
        return sizeof(bucket_t) * (NWINS - 1) * (config.N * NTHREADS);
    }
    size_t get_size_res_faster() {
        return sizeof(bucket_t) * (NWINS);
    }

    size_t get_size_cub_sort_faster(MSMConfig& config){
        uint32_t *d_wval = nullptr; 
        uint32_t *d_wval_out = nullptr; 
        uint32_t *d_idx = nullptr; 
        uint32_t *d_idx_out = nullptr;
        void *d_temp = NULL;
        size_t temp_size = 0;
        cub::DeviceRadixSort::SortPairs(d_temp, temp_size, d_wval, d_wval_out, d_idx, d_idx_out, config.n, 0, 24);
        return temp_size;
    }
    size_t get_size_cub_flag_faster(MSMConfig& config){
        uint32_t *d_wval = nullptr; 
        uint32_t *d_wval_out = nullptr; 
        uint32_t *d_idx = nullptr; 
        uint32_t *d_idx_out = nullptr;
        void *d_temp = NULL;
        size_t temp_size = 0;
        cub::DevicePartition::Flagged(d_temp, temp_size, d_idx, d_wval, d_idx_out, d_wval_out, config.n);
        return temp_size;
    }
    size_t get_size_cub_encode_faster(MSMConfig& config){
        uint32_t *d_wval = nullptr; 
        uint32_t *d_wval_out = nullptr; 
        uint32_t *d_idx = nullptr; 
        uint32_t *d_idx_out = nullptr;
        void *d_temp = NULL;
        size_t temp_size = 0;
        cub::DeviceRunLengthEncode::Encode(d_temp, temp_size, d_wval, d_wval_out, d_idx, d_idx_out, config.n);
        return temp_size;
    }
    size_t get_size_cub_sum_faster(){
        uint32_t *d_wval = nullptr; 
        uint32_t *d_wval_out = nullptr; 
        uint32_t *d_idx = nullptr; 
        uint32_t *d_idx_out = nullptr;
        void *d_temp = NULL;
        size_t temp_size = 0;
        cub::DeviceScan::InclusiveSum(d_temp, temp_size, d_wval, d_wval_out, 1 << WBITS);
        return temp_size;
    }

    result_container_t_faster get_result_container_faster() {
        result_container_t_faster res(1);
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

    size_t allocate_d_wval_faster(MSMConfig& config) {
        return d_wval_ptrs.allocate(get_size_wval_faster(config));
    }
    size_t allocate_d_idx_faster(MSMConfig& config) {
        return d_idx_ptrs.allocate(get_size_idx_faster(config));
    }

    size_t allocate_d_wval_count_faster() {
        return d_wval_ptrs.allocate(get_size_wval_count_faster());
    }
    size_t allocate_d_wval_ptr_faster() {
        return d_wval_ptrs.allocate(get_size_wval_ptr_faster());
    }
    size_t allocate_d_wval_unique_faster() {
        return d_wval_ptrs.allocate(get_size_wval_unique_faster());
    }
    size_t allocate_d_wval_unique_count_faster() {
        return d_wval_ptrs.allocate(get_size_wval_unique_count_faster());
    }

    size_t allocate_d_buckets_faster() {
        return d_bucket_ptrs.allocate(get_size_buckets_faster());
    }
    size_t allocate_d_tres_faster(MSMConfig& config) {
        return d_bucket_ptrs.allocate(get_size_tres_faster(config));
    }
    size_t allocate_d_res_faster() {
        return d_bucket_ptrs.allocate(get_size_res_faster());
    }

    size_t allocate_d_cub_sort_faster(MSMConfig& config) {
        return d_cub_ptrs.allocate(get_size_cub_sort_faster(config));
    }
    size_t allocate_d_cub_flag_faster(MSMConfig& config) {
        return d_cub_ptrs.allocate(get_size_cub_flag_faster(config));
    }
    size_t allocate_d_cub_encode_faster(MSMConfig& config) {
        return d_cub_ptrs.allocate(get_size_cub_encode_faster(config));
    }
    size_t allocate_d_cub_sum_faster() {
        return d_cub_ptrs.allocate(get_size_cub_sum_faster());
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


    void transfer_res_to_host_faster(result_container_t_faster &res, size_t d_res_idx,
                                  cudaStream_t s = nullptr) {
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        bucket_t *d_res = d_bucket_ptrs[d_res_idx];
        CUDA_OK(cudaSetDevice(device));
        CUDA_OK(cudaMemcpyAsync(res[0], d_res, sizeof(res[0]),
                                cudaMemcpyDeviceToHost, stream));
    }

    void synchronize_stream() {
        CUDA_OK(cudaSetDevice(device));
        CUDA_OK(cudaStreamSynchronize(default_stream));
    }


    void launch_kernel_faster_1(MSMConfig& config,
                    size_t d_scalars_idx, size_t d_buckets_idx,
                    size_t d_wval_idx, size_t d_idx_idx,
                    cudaStream_t s = nullptr)
    {
        assert(WBITS > NTHRBITS);
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        scalar_t *d_scalars = d_scalar_ptrs[d_scalars_idx];
        bucket_t (*d_buckets)[1<<WBITS] =
            reinterpret_cast<decltype(d_buckets)>(d_bucket_ptrs[d_buckets_idx]);
        uint32_t *d_wval = d_wval_ptrs[d_wval_idx];
        uint32_t *d_idx = d_idx_ptrs[d_idx_idx];

        CUDA_OK(cudaSetDevice(device));
        launch_coop(pippenger_faster_1, (NWINS - 1) * config.N, NTHREADS, stream,
                    (const scalar_t*)d_scalars, config.npoints, d_buckets, 
                    d_wval, d_idx);
    }

    void launch_kernel_faster_2(MSMConfig& config,
                       size_t d_bases_idx, size_t d_buckets_idx,
                       size_t d_wval_idx, size_t d_idx_idx, 
                       size_t d_wval_count_idx, size_t d_wval_unique_idx, 
                       size_t d_wval_ptr_idx, size_t d_wval_unique_count_idx, 
                       cudaStream_t s = nullptr)
    {
        assert(WBITS > NTHRBITS);
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        affine_t *d_points = d_base_ptrs[d_bases_idx];
        bucket_t (*d_buckets)[1<<WBITS] =
            reinterpret_cast<decltype(d_buckets)>(d_bucket_ptrs[d_buckets_idx]);
        uint32_t *d_wval = d_wval_ptrs[d_wval_idx];
        uint32_t *d_idx = d_idx_ptrs[d_idx_idx];
        uint32_t *d_wval_count = d_wval_ptrs[d_wval_count_idx];
        uint32_t *d_wval_ptr = d_wval_ptrs[d_wval_ptr_idx];
        uint32_t *d_wval_unique = d_wval_ptrs[d_wval_unique_idx];
        uint32_t *d_wval_unique_count = d_wval_ptrs[d_wval_unique_count_idx];

        CUDA_OK(cudaSetDevice(device));
        pippenger_faster_2<<<dim3((NWINS - 1), config.N), NTHREADS, 0, stream>>>(d_points, config.npoints,
                     d_buckets, d_wval, d_idx, d_wval_count, d_wval_unique, d_wval_ptr, d_wval_unique_count);
    }


    void launch_kernel_faster_3(MSMConfig& config, size_t d_buckets_idx, size_t d_tres_idx,
                    cudaStream_t s = nullptr)
    {
        assert(WBITS > NTHRBITS);
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        bucket_t (*d_buckets)[1<<WBITS] =
            reinterpret_cast<decltype(d_buckets)>(d_bucket_ptrs[d_buckets_idx]);
        bucket_t *d_tres = d_bucket_ptrs[d_tres_idx];

        CUDA_OK(cudaSetDevice(device));
        pippenger_faster_3<<<dim3((NWINS - 1), config.N), NTHREADS, 0, stream>>>(d_buckets, d_tres);
    }

    void launch_kernel_faster_4(MSMConfig& config,
                       size_t d_bases_idx, size_t d_wval_idx, 
                       size_t d_idx_idx, size_t d_tres_idx,
                    cudaStream_t s = nullptr)
    {
        assert(WBITS > NTHRBITS);
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        affine_t *d_points = d_base_ptrs[d_bases_idx];
        uint32_t *d_s_count = d_wval_ptrs[d_wval_idx] + (NWINS - 1) * config.npoints;
        uint32_t *d_idx = d_idx_ptrs[d_idx_idx];
        bucket_t *d_tres = d_bucket_ptrs[d_tres_idx];

        CUDA_OK(cudaSetDevice(device));
        pippenger_faster_4<<<dim3((NWINS - 1), config.N), NTHREADS, 0, stream>>>(d_points, config.npoints,
                            d_s_count, d_idx, d_tres);
    }


    void launch_kernel_faster_5(MSMConfig& config,
                    size_t d_tres_idx, size_t d_out_idx,
                    cudaStream_t s = nullptr)
    {
        assert(WBITS > NTHRBITS);
        cudaStream_t stream = (s == nullptr) ? default_stream : s;

        bucket_t *d_tres = d_bucket_ptrs[d_tres_idx];
        bucket_t *d_out = d_bucket_ptrs[d_out_idx];

        CUDA_OK(cudaSetDevice(device));

        launch_coop(CSum, dim3((NWINS - 1), config.N / 2), NTHREADS, stream,
                    d_out, d_tres);
    }

    void launch_kernel_faster_6(MSMConfig& config,
                    size_t d_tres_idx, size_t d_res_idx,
                    cudaStream_t s = nullptr)
    {
        assert(WBITS > NTHRBITS);
        cudaStream_t stream = (s == nullptr) ? default_stream : s;

        bucket_t *d_tres = d_bucket_ptrs[d_tres_idx];
        bucket_t *d_res = d_bucket_ptrs[d_res_idx];

        CUDA_OK(cudaSetDevice(device));

        launch_coop(LSum, dim3((NWINS - 1), config.N / 2), NTHREADS, stream,
                    d_res, d_tres);
    }

    // Perform final accumulation on CPU.
    void accumulate_faster(point_t &out, result_container_t_faster &res) {
        out.inf();

        for(int32_t k = NWINS - 1; k >= 0; k--)
        {
            for (int32_t i = 0; i < WBITS; i++)
            {
                out.dbl();
            }
            point_t p = (res[0])[k];
            out.add(p);
        }
    }
};

#endif
