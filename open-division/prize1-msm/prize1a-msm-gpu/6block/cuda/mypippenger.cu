//
// Created by hulei on 22-9-24.
//
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <cuda_profiler_api.h>
#include <util/exception.cuh>
#include <util/rusterror.h>
#include <ff/bls12-377.hpp>
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>
#include <algorithm>
#include <vector>
#include <numeric>
#include "CudaDeviceMemWrapper.h"
#include "CudaHostMemWrapper.h"
#include "CudaStreamWrapper.h"
#include "CudaEventWrapper.h"
#include "util/thread_pool_t.hpp"

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_inf_t affine_t;
typedef fr_t scalar_t;

#ifndef cudaCheckError
#define cudaCheckError(expr) {                                                               \
    cudaError e;                                                                             \
    if ((e = expr) != cudaSuccess) {                                                         \
        const char* error_str = cudaGetErrorString(e);                                       \
        throw std::runtime_error(error_str);                                                 \
    }                                                                                        \
}
#endif

/*
 * Extracts bits of scalars in bit window and generate original index of each base in base array.
 * @param [out] extracted_bits Extracted bits of each scalar in current bit window.
 * @param [out] point_index Original index of each base in base array.
 * @param [in] bits_start. Starting bit of bit window.
 * @param [in] bits_num. Length of bit window.
 * @param [in] scalars.
 * @param [in] point_num. Number of points.
 */
__global__ void d_extract_bits(int *extracted_bits,
                               int *point_index,
                               int bits_start,
                               int bits_num,
                               const scalar_t *scalars,
                               int point_num);

/*
 * Each cuda thread is assigned a range of bases and calculate msm on that range. Then store result in out array.
 * As points have been sorted, each thread calculate msm from back to front and use the following technique:
 * For example, the scalars and the base may be as follows:
 * 5P1 7P2 7P3 9P4 9P5 9P6
 * We use two bucket_t t and s to store intermediate result:
 * t stores sum of all scanned points.
 * Pseudocode:
 * for each element from back to front {
 *     if the scalar is different from the previous scalar{
 *        s += (previous scalar - current scalar) * t;
 *     }
 *     t += current base;
 * }
 * So after each loop:
 *               t                              s
 * loop 1:       P6                             inf
 * loop 2:       P5 + P6                        inf
 * loop 3:       P4 + P5 + P6                   inf
 * loop 4:       P3 + P4 + P5 + P6              2(P4 + P5 + P6)
 * loop 5:       P2 + P3 + P4 + P5 + P6         2(P4 + P5 + P6)
 * loop 6:       P1 + P2 + P3 + P4 + P5 + P6    2(P2 + P3 + P4 + P5 + P6) + 2(P4 + P5 + P6)
 *
 * The final result is equal to: the first scalar * t + s
 * So 5t + s = 5(P1 + P2 + P3 + P4 + P5 + P6) + 2(P2 + P3 + P4 + P5 + P6) + 2(P4 + P5 + P6) = 5P1 + 7P2 + 7P3 + 9P4 + 9P5 + 9P6
 *
 * @param [out] out. Msm result of each thread.
 * @param [in] tmp_store_s Used to store s.
 * @param [in] tmp_store_bucket. Array. Allocated for each thread to store t when scalar is different from previous scalar.
 * @param [in] tmp_store_scalar. Array. Allocated for each thread to store scalar when scalar is different from previous scalar.
 * @param [in] tmp_store_size. Length of tmp_store_bucket and tmp_store_scalar
 * @param [in] affine_points. Bases.
 * @param [in] point_original_index. Sorted bases' original index in original bases array.
 * @param [in] bucket_index_of_point. Each base's scalar.
 * @param [in] points_per_thread. Number of bases should be calculated by each thread.
 * @param [in] point_num. Number of bases.
 * */
__global__ void d_pippenger(bucket_t *out,
                            bucket_t *tmp_store_s,
                            bucket_t *tmp_store_bucket,
                            int *tmp_store_scalar,
                            const int tmp_store_size,
                            const affine_t *affine_points,
                            const int *point_original_index,
                            const int *bucket_index_of_point,
                            const int points_per_thread,
                            const int point_num);

/*
 * Use cub to sort pair (bucket index, point index).
 */
void sort_by_bucket_id(void *d_tmp_storage, size_t &tmp_storage_bytes,
                       const int *d_bucket_index_in, int *d_bucket_index_out,
                       const int *d_point_index_in, int *d_point_index_out,
                       const int point_num,
                       const int begin_bit,
                       const int end_bit){
    cudaCheckError(cub::DeviceRadixSort::SortPairs(d_tmp_storage, tmp_storage_bytes,
                                                   d_bucket_index_in, d_bucket_index_out,
                                                   d_point_index_in, d_point_index_out,
                                                   point_num,
                                                   begin_bit,
                                                   end_bit));
}

#ifdef __CUDA_ARCH__

__device__ int get_wval(const scalar_t& d, uint32_t off, uint32_t bits)
{
    uint32_t top = off + bits - 1;
    uint64_t ret = ((uint64_t)d[top/32] << 32) | d[off/32];

    return (int)(ret >> (off%32)) & ((1<<bits) - 1);
}

__global__ void d_extract_bits( int *extracted_bits,
                                int *point_index,
                                int bits_start,
                                int bits_num,
                                const scalar_t *scalars,
                                int point_num) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= point_num) {
        return;
    }
    point_index[tid] = tid;
    extracted_bits[tid] = get_wval(scalars[tid], bits_start, bits_num);
}

/*
 * Return a * b.
 */
__device__ bucket_t d_mul(bucket_t &a, int b){
    if (b == 1) {
        return a;
    }
    int first_bit = 0;
    int t = b;
    while (t>>=1) first_bit++;
    bucket_t ret;
    ret.inf();
    for (int i = first_bit; i >= 1; i--) {
        if (b & (1 << i)) {
            ret.add(a);
        }
        ret.add(ret);
    }
    if (b & 1) {
        ret.add(a);
    }
    return ret;
}

__global__ void d_pippenger(bucket_t *out,
                            bucket_t *tmp_store_s,
                            bucket_t *tmp_store_bucket,
                            int *tmp_store_scalar,
                            const int tmp_store_size,
                            const affine_t *affine_points,
                            const int *point_original_index,
                            const int *bucket_index_of_point,
                            const int points_per_thread,
                            const int point_num) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    tmp_store_bucket += tmp_store_size * tid;
    tmp_store_scalar += tmp_store_size * tid;
    //Number of points stored in tmp_store_bucket
    int stored_bucket_num = 0;
    int begin = tid * points_per_thread;
    int index = begin + points_per_thread;
    if (index > point_num) {
        index = point_num;
    }
    if (index <= begin) {
        out[tid].inf();
        return;
    }
    index--;
    bucket_t t;
    t = affine_points[point_original_index[index]];
    tmp_store_s[tid].inf();
    uint32_t prev_bucket_index = bucket_index_of_point[index];
    index--;
    while(true){
        //Check if all bases are calculated.
        bool should_exe = index >= begin;
        //If no thread in warp has more base to calculate, then break.
        if(__any_sync(0xffffffff, should_exe) == 0) {
            break;
        }
        if (should_exe){
            int cur_bucket_index = bucket_index_of_point[index];
            if (cur_bucket_index != prev_bucket_index) {
                //Store t and scalr in temporary space and calculate later together to avoid divergency.
                tmp_store_bucket[stored_bucket_num] = t;
                tmp_store_scalar[stored_bucket_num] = prev_bucket_index - cur_bucket_index;
                stored_bucket_num++;
            }
            prev_bucket_index = cur_bucket_index;
        }
        //Check if tempory storage of any thread in warp is full.
        if (__any_sync(0xffffffff, stored_bucket_num >= tmp_store_size) != 0) {
            if(stored_bucket_num > 0) {
                stored_bucket_num--;
                //s += t * scalar.
                tmp_store_s[tid].add(d_mul(tmp_store_bucket[stored_bucket_num], tmp_store_scalar[stored_bucket_num]));
            }
        }

        //Update t.
        if (should_exe){
            uint32_t addend_offset = point_original_index[index];
            t.add(affine_points[addend_offset]);
            index--;
        }
    }
    if (prev_bucket_index > 0) {
        t = d_mul(t, prev_bucket_index);
    } else {
        t.inf();
    }
    t.add(tmp_store_s[tid]);
    for (int i = 0; i < stored_bucket_num; i++) {
        t.add(d_mul(tmp_store_bucket[i], tmp_store_scalar[i]));
    }

    out[tid] = t;
}

#else

//Return a * b
bucket_t h_mul(bucket_t &a, int b){
    if (b == 1) {
        return a;
    }
    int first_bit = 0;
    int t = b;
    while (t>>=1) first_bit++;
    bucket_t ret;
    ret.inf();
    for (int i = first_bit; i >= 1; i--) {
        if (b & (1 << i)) {
            ret.add(a);
        }
        ret.add(ret);
    }
    if (b & 1) {
        ret.add(a);
    }
    return ret;
}

struct Context{
    cuda_helper::CudaDeviceMemWrapper<bucket_t> d_out;
    cuda_helper::CudaDeviceMemWrapper<bucket_t> d_tmp_store_s;
    //Bases
    cuda_helper::CudaDeviceMemWrapper<affine_t> d_points;
    //Scalars of one batch
    cuda_helper::CudaDeviceMemWrapper<scalar_t> d_scalars;
    //Temporary memory to help avoid divergence
    cuda_helper::CudaDeviceMemWrapper<bucket_t> d_tmp_store_bucket;
    cuda_helper::CudaDeviceMemWrapper<int> d_tmp_store_scalar;
    int tmp_store_size;


    cuda_helper::CudaDeviceMemWrapper<int> d_bucket_index_tmp;
    cuda_helper::CudaDeviceMemWrapper<int> d_bucket_index;
    cuda_helper::CudaDeviceMemWrapper<int> d_point_index_tmp;
    cuda_helper::CudaDeviceMemWrapper<int> d_point_index;

    //Memory used by cub to sort.
    cuda_helper::CudaDeviceMemWrapper<uint8_t> d_sort_temp_storage;
    size_t temp_storage_bytes;

    //Pinned memory to speedup transfer from host to device.
    std::vector<cuda_helper::CudaHostMemWrapper<scalar_t>> h_pinned_scalars;

    //Configuration fo kernel
    int extract_bits_block_dim;

    //Kernel configuration of global function d_pippenger.
    int pippenger_block_num;
    int pippenger_block_dim;

    //Number of bases.
    size_t point_num;

    //Host memory used to store msm results of each cuda threads.
    cuda_helper::CudaHostMemWrapper<bucket_t> h_t;
    cuda_helper::CudaHostMemWrapper<bucket_t> h_s;

    //Msm result of each round.
    std::vector<bucket_t> sums;
};

void init(Context *context,
          const affine_t h_points[],
          const size_t npoints,
          const int max_window_size_in_bit,
          size_t ffi_affine_sz = sizeof(affine_t)) {
    context->point_num = npoints;
    cudaCheckError(cudaOccupancyMaxPotentialBlockSize(&context->pippenger_block_num, &context->pippenger_block_dim, d_pippenger, 0, 128));
    int thread_num = context->pippenger_block_num * context->pippenger_block_dim;
    context->d_out.resize(thread_num);
    context->d_tmp_store_s.resize(thread_num);
    context->d_points.resize(npoints);
    if (ffi_affine_sz == sizeof(affine_t)){
        cudaCheckError(cudaMemcpyAsync(context->d_points.ptr(), h_points, npoints * sizeof(affine_t),
                                       cudaMemcpyHostToDevice, 0));
    } else {
        cudaCheckError(cudaMemcpy2DAsync(context->d_points.ptr(), sizeof(affine_t), h_points, ffi_affine_sz, ffi_affine_sz, npoints,
                                         cudaMemcpyHostToDevice, 0));
    }

    context->d_scalars.resize(npoints);
    context->d_bucket_index.resize(npoints);
    context->d_bucket_index_tmp.resize(npoints);
    context->d_point_index.resize(npoints);
    context->d_point_index_tmp.resize(npoints);
    sort_by_bucket_id(NULL, context->temp_storage_bytes,
                      context->d_bucket_index_tmp.ptr(), context->d_bucket_index.ptr(),
                      context->d_point_index_tmp.ptr(), context->d_point_index_tmp.ptr(),
                      context->point_num,
                      0,
                      max_window_size_in_bit);
    context->d_sort_temp_storage.resize(context->temp_storage_bytes);
    context->h_pinned_scalars.resize(2);
    for (auto &h_scalars : context->h_pinned_scalars) {
        h_scalars.resize(npoints);
    }
    int block_num;
    context->h_t.resize(thread_num);
    context->d_out.resize(thread_num);
    context->d_tmp_store_s.resize(thread_num);
    cudaCheckError(cudaOccupancyMaxPotentialBlockSize(&block_num, &context->extract_bits_block_dim, d_extract_bits, 0, 128));
    context->tmp_store_size = 10;
    context->d_tmp_store_bucket.resize(thread_num * context->tmp_store_size);
    context->d_tmp_store_scalar.resize(thread_num * context->tmp_store_size);
}

Context g_ctx;

struct MemcpyInfo{
    void *dst;
    const void *src;
    size_t bytes;
};

void h_memcpy(void *user_data) {
    MemcpyInfo *pinfo = (MemcpyInfo*)user_data;
    memcpy(pinfo->dst, pinfo->src, pinfo->bytes);
    delete pinfo;
}

struct PippengerInfo{
    int bit0;
};

void h_cal_pippenger(void *userData) {
    PippengerInfo *pinfo = (PippengerInfo*)userData;
    int thread_num = g_ctx.pippenger_block_num * g_ctx.pippenger_block_dim;
    auto start = std::chrono::steady_clock::now();
    bucket_t sum;
    sum.inf();
    for (int i = 0; i < thread_num ; i++){
        sum.add(g_ctx.h_t.hptr()[i]);
    }
    for (int i = 0; i < pinfo->bit0; i++) {
        sum.add(sum);
    }
    g_ctx.sums.push_back(sum);
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    delete pinfo;
}

RustError h_init(const affine_t points[], const size_t npoints, const int max_window_size_in_bits, const size_t ffi_affine_sz = sizeof(affine_t)){
    init(&g_ctx,
         points,
         npoints,
         max_window_size_in_bits,
         ffi_affine_sz);
    std::cout << "Initialized" << std::endl;
    return RustError(0);
}

RustError h_msm(point_t *out, const scalar_t h_scalars[], const size_t scalar_num, std::vector<int> bits)
{
    if (scalar_num % g_ctx.point_num != 0) {
        std::cout << "Invalid scalar number:" << scalar_num << std::endl;
        return RustError(-1);
    }
    if (accumulate(bits.begin(), bits.end(),0) != 253){
        printf("Wrong bits configuration\n");
        return RustError(-1);
    };

    cuda_helper::CudaStreamWrapper cpy_stream(cudaStreamNonBlocking);
    cuda_helper::CudaStreamWrapper cpu_cal_stream(cudaStreamNonBlocking);

    cuda_helper::CudaEventWrapper after_pippenger(cudaEventDisableTiming), after_cpy_to_cpu(cudaEventDisableTiming),
                                  after_extract_bits(cudaEventDisableTiming), after_scalar_copied(cudaEventDisableTiming),
                                  after_dtoh(cudaEventDisableTiming);

    int batch_num = scalar_num / g_ctx.point_num;
    out->inf();
    if (batch_num <= 0) {
        return RustError(0);
    }

    g_ctx.sums.clear();
    int scalar_buf_index = 0;
    //Copy data of scalars used in first batch
    memcpy(g_ctx.h_pinned_scalars[scalar_buf_index].hptr(), h_scalars, g_ctx.point_num * sizeof(scalar_t));
    cudaCheckError(cudaMemcpyAsync(g_ctx.d_scalars.ptr(), g_ctx.h_pinned_scalars[scalar_buf_index].hptr(), g_ctx.point_num * sizeof(scalar_t), cudaMemcpyHostToDevice));
    cudaCheckError(cudaDeviceSynchronize());
    //Begin timing
    auto start_time = std::chrono::steady_clock::now();
    const auto thread_num = g_ctx.pippenger_block_num * g_ctx.pippenger_block_dim;
    for (int batch_index = 0; batch_index < batch_num; batch_index++) {
        int next_scalar_buf_index = (scalar_buf_index + 1) % g_ctx.h_pinned_scalars.size();
        if (batch_index + 1 < batch_num){
            //Begin to copy scalars in next round.
            MemcpyInfo *pinfo = new MemcpyInfo();
            pinfo->dst = g_ctx.h_pinned_scalars[next_scalar_buf_index].hptr();
            pinfo->src = &h_scalars[(batch_index + 1) * g_ctx.point_num];
            pinfo->bytes = g_ctx.point_num * sizeof(scalar_t);
            //Copy data to pinned memory
            cudaCheckError(cudaLaunchHostFunc(cpy_stream.get(), h_memcpy, pinfo));
        }
        //BLock kernel d_extract_bits until scalars are transfered to device memory
        cudaCheckError(cudaStreamWaitEvent(0, after_scalar_copied.get()));

        int bit0 = 0;
        int round = 0;
        for (auto bit_num : bits) {
            auto bucket_num = 1 << bit_num;
            int block_num = (g_ctx.point_num + g_ctx.extract_bits_block_dim - 1) / g_ctx.extract_bits_block_dim;
            d_extract_bits<<<block_num, g_ctx.extract_bits_block_dim>>>(g_ctx.d_bucket_index_tmp.ptr(),
                                                                      g_ctx.d_point_index_tmp.ptr(),
                                                                      bit0,
                                                                      bit_num,
                                                                      g_ctx.d_scalars.ptr(),
                                                                      g_ctx.point_num);
            //If it's the last bit window,
            if (batch_index + 1 < batch_num && round + 1 == (int)bits.size()){
                //Block until d_extract_bits executed so the d_scalars buffer can be reused.
                cudaCheckError(cudaEventRecord(after_extract_bits.get()));
                cudaCheckError(cudaStreamWaitEvent(cpy_stream.get(), after_extract_bits.get()));
                //Copy scalars from host side to device side.
                cudaCheckError(cudaMemcpyAsync(g_ctx.d_scalars.ptr(), g_ctx.h_pinned_scalars[next_scalar_buf_index].hptr(), g_ctx.point_num * sizeof(scalar_t), cudaMemcpyHostToDevice, cpy_stream.get()));
                cudaCheckError(cudaEventRecord(after_scalar_copied.get(), cpy_stream.get()));
            }

            sort_by_bucket_id(g_ctx.d_sort_temp_storage.ptr(), g_ctx.temp_storage_bytes,
                              g_ctx.d_bucket_index_tmp.ptr(), g_ctx.d_bucket_index.ptr(),
                              g_ctx.d_point_index_tmp.ptr(), g_ctx.d_point_index.ptr(),
                              g_ctx.point_num,
                               0,
                               bit_num);

            //Block d_pippenger until msm result of previous bit window has been copied to host side.
            cudaCheckError(cudaStreamWaitEvent(0, after_dtoh.get()));
            d_pippenger<<<g_ctx.pippenger_block_num, g_ctx.pippenger_block_dim>>>(g_ctx.d_out.ptr(),
                                                                  g_ctx.d_tmp_store_s.ptr(),
                                                                  g_ctx.d_tmp_store_bucket.ptr(),
                                                                  g_ctx.d_tmp_store_scalar.ptr(),
                                                                  g_ctx.tmp_store_size,
                                                                  g_ctx.d_points.ptr(),
                                                                  g_ctx.d_point_index.ptr(),
                                                                  g_ctx.d_bucket_index.ptr(),
                                                                  (g_ctx.point_num + thread_num - 1) / thread_num,
                                                                  g_ctx.point_num);
            cudaCheckError(cudaEventRecord(after_pippenger.get()));
            //Copy msm result to host side and use CPU to calculate the final result of current bit window.
            cudaCheckError(cudaStreamWaitEvent(cpu_cal_stream.get(), after_pippenger.get()));
            cudaCheckError(cudaMemcpyAsync(g_ctx.h_t.hptr(), g_ctx.d_out.ptr(), sizeof(bucket_t) * g_ctx.pippenger_block_num * g_ctx.pippenger_block_dim, cudaMemcpyDeviceToHost, cpu_cal_stream.get()));
            cudaCheckError(cudaEventRecord(after_dtoh.get(), cpu_cal_stream.get()));
            PippengerInfo *pinfo = new PippengerInfo();
            pinfo->bit0 = bit0;
            cudaCheckError(cudaLaunchHostFunc(cpu_cal_stream.get(), h_cal_pippenger, pinfo));
            bit0 += bit_num;
            round++;
        }
        scalar_buf_index = next_scalar_buf_index;
    }

    cudaCheckError(cudaDeviceSynchronize());
    if (g_ctx.sums.size() != batch_num * bits.size()) {
        std::cout << "Error\n" << std::endl;
        return RustError(-1);
    }

    for (int i = 0; i < batch_num; i++) {
        bucket_t sum;
        sum.inf();
        //Sum msm result of each bit window.
        for (size_t j = 0; j < bits.size(); j++) {
            sum.add(g_ctx.sums[i * bits.size() + j]);
        }
        out[i] = sum;
    }

    auto end_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    printf("total elapsed:%lf ms.\n", elapsed / 1000.0);
    return RustError{cudaSuccess};
}


extern "C"
RustError init(const affine_t points[], const size_t npoints, const int max_window_size_in_bits, const size_t ffi_affine_sz){
    try{
        std::cout << "init: npoints:" << npoints << ", max_window_size_in_bits:" << max_window_size_in_bits << ", ffi_affine_sz:" << ffi_affine_sz << std::endl ;
        return h_init(points, npoints, max_window_size_in_bits, ffi_affine_sz);
    } catch (std::exception &e) {
        printf("MSM init fail:%s\n", e.what());
        return RustError(-1);
    }
}


extern "C"
RustError msm(point_t* out, const scalar_t scalars[], const size_t scalar_num, const int *bits_window_length, const int window_num){
    try{
        std::vector<int> bits_array;
        bits_array.assign(bits_window_length, bits_window_length + window_num);
        h_msm(out, scalars, scalar_num, bits_array);
        return RustError(0);
    } catch (std::exception &e) {
        printf("MSM fail:%s\n", e.what());
        return RustError(-1);
    }
}
#endif