// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>
#include <chrono>
#include <omp.h>
#include <mutex>
#include <bit>
#include <bitset>

#define WARP_SZ 32

constexpr static int log2(int n)
{   int ret=0; while (n>>=1) ret++; return ret;   }

#define MAX_SCALAR_VAL_23_BIT 4894102

#define NBITS 253
#define WBITS 23

#define NWINS ((NBITS+WBITS-1)/WBITS) // ceil(NBITS/WBITS)

//TODO for production: may be need to unite this via templates
#define TILING   2048
#define TILING_2 4096

#define NAF 1
#define SIGNED_SCALAR 1

#define WITH_NULL 0

#if WBITS<=16
typedef unsigned short part_t;
#else
typedef unsigned int part_t;
#endif

__global__ void pippenger_group_NAF(const affine_t* __restrict__  points, bucket_t *buckets, const uint32_t* idx, const uint32_t* tableS, const uint32_t* tableE, const uint32_t* numbers, int bucketSize, bool continueAdd);
__global__ void pippenger_group_last_NAF(const affine_t* __restrict__  points, bucket_t *buckets, const uint32_t* idx, const uint32_t* tableS, const uint32_t* tableE, const uint32_t* numbers, bool continueAdd, const int lastMax, const int threadsCount);

__global__ void pippenger_final_NAF(bucket_t *buckets, bucket_t (*ret)[TILING][2], int elemsInBucket, int sizeBucket, int total);
__global__ void pippenger_final_NAF_2(bucket_t *buckets, bucket_t (*ret)[TILING_2][2], int elemsInBucket, int sizeBucket, int total);

__global__ void initGroup_NAF(const scalar_t* scalars, unsigned int* idx, part_t* keys, int npoints, int shift);
__global__ void initRows(unsigned int* idx, part_t* keys, int npoints);
__global__ void initRows(uint32_t* rows, uint32_t *sizes, const uint32_t* tableS, const uint32_t* tableE, int N);
__global__ void initTable(unsigned int* rows, unsigned int* tableS, unsigned int* tableE, int npoints);
__global__ void zero_points(affine_t* points, size_t N);

//from sort.cu
extern "C" void sort(unsigned int* idxs, part_t* part, size_t npoints, cudaStream_t& st, unsigned* getGroups);
extern "C" void sort1(unsigned int* idxs, unsigned int* sizes, size_t npoints, cudaStream_t& st, bool greater);
extern "C" void scan(unsigned int* rows, size_t npoints, cudaStream_t& st);
extern "C" void clearCache();

#ifdef __CUDA_ARCH__

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

// t3 = t1 + t2
__device__ void longAdd(unsigned* t3, const unsigned* t1, const unsigned* t2, int N)
{
	uint64_t bitNext = 0;
	for (int z = 0; z < N; ++z)
	{
		uint64_t val = (uint64_t)t1[z] + (uint64_t)t2[z] + bitNext;
		t3[z] = (int)val;
		bitNext = (val >> 32) ? 1 : 0;
	}
}

// t2 = t1 >> 1;
__device__ void longShift(unsigned* t2, const unsigned* t1, int N)
{
	t2[0] = t1[0] >> 1;
	for (int z = 1; z < N; ++z)
	{
		if (t1[z] & 1)
			t2[z - 1] |= ((unsigned)1 << 31);
		t2[z] = t1[z] >> 1;
	}
}

// t3 = t2 ^ t1
__device__ void longXor(unsigned* t3, const unsigned* t1, const unsigned* t2, int N)
{
	for (int z = 0; z < N; ++z)
		t3[z] = t2[z] ^ t1[z];
}

// t3 = t2 & t1
__device__ void longAnd(unsigned* t3, const unsigned* t1, const unsigned* t2, int N)
{
	for (int z = 0; z < N; ++z)
		t3[z] = t2[z] & t1[z];
}

// zero point if points[id].is_inf() is equal true
__global__ void zero_points(affine_t* points, size_t N)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= N)
		return;

	if (points[id].is_inf())
	{
		points[id].X.zero();
		points[id].Y.zero();

		points[id + N].X = points[id].X;
		points[id + N].Y = points[id].Y;
	}
	else
	{
		points[id + N].X = points[id].X;
		fp_t Y = points[id].Y;
		Y.cneg(true);
		points[id + N].Y = Y;
	}
}

//TODO for production: move this function back to affine_t class method

//parameters:
//IN: point with coordinates X2, Y2,
//    isSecond - flag which indicates the second add operation to empry bucket
//INOUT: resInf - flag which indicates empty bucket p31
//       bucket p31
__device__ void madd_base(const fp_t& X2, const fp_t& Y2, bucket_t& p31, int& resInf, bool isSecond)
{
#if WITH_NULL
	if (X2.is_zero())
		if (Y2.is_zero())
			return;
#endif

	if (resInf == 1)
	{
		resInf = 0;
		p31.X = X2;
		p31.Y = Y2;
		p31.ZZZ = p31.ZZ = fp_t::one();
	}
	else
	{
		fp_t P, R;

		if (isSecond)
			R = Y2;
		else
			R = Y2 * p31.ZZZ;         /* S2 = Y2*ZZZ1 */
		R -= p31.Y;                 /* R = S2-Y1 */

		if (isSecond)
			P = X2;
		else
			P = X2 * p31.ZZ;          /* U2 = X2*ZZ1 */
		P -= p31.X;                 /* P = U2-X1 */

		if (!P.is_zero()) {         /* X1!=X2 */
			fp_t PP;             /* add |p2| to |p1| */

			PP = P^2;               /* PP = P^2 */
#define PPP P
			PPP = P * PP;           /* PPP = P*PP */
			if (isSecond)
			{
				p31.ZZ = PP;
				p31.ZZZ = PPP;
			}
			else
			{
				p31.ZZ *= PP;           /* ZZ3 = ZZ1*PP */
				p31.ZZZ *= PPP;         /* ZZZ3 = ZZZ1*PPP */
			}
#define Q PP
			Q = PP * p31.X;         /* Q = X1*PP */
			p31.X = R^2;            /* R^2 */
			p31.X -= PPP;           /* R^2-PPP */
			p31.X -= Q;
			p31.X -= Q;             /* X3 = R^2-PPP-2*Q */
			Q -= p31.X;
			Q *= R;                 /* R*(Q-X3) */
			p31.Y *= PPP;           /* Y1*PPP */
			p31.Y = Q - p31.Y;      /* Y3 = R*(Q-X3)-Y1*PPP */
#undef Q
#undef PPP
		}
		else if (R.is_zero())    /* X1==X2 && Y1==Y2 */
		{
			fp_t M;              /* double |p2| */
#define U P
			U = Y2 + Y2;        /* U = 2*Y1 */
			p31.ZZ = U^2;           /* [ZZ3 =] V = U^2 */
			p31.ZZZ = p31.ZZ * U;   /* [ZZZ3 =] W = U*V */
#define S R
			S = X2 * p31.ZZ;      /* S = X1*V */
			M = X2^2;
			M = M + M + M;          /* M = 3*X1^2[+a] */
			p31.X = M^2;
			p31.X -= S;
			p31.X -= S;             /* X3 = M^2-2*S */
			p31.Y = p31.ZZZ * Y2; /* W*Y1 */
			S -= p31.X;
			S *= M;                 /* M*(S-X3) */
			p31.Y = S - p31.Y;      /* Y3 = M*(S-X3)-W*Y1 */
#undef S
#undef U

		} else
		{                    /* X1==X2 && Y1==-Y2 */
		    resInf = 1;
			p31.inf();              /* set |p3| to infinity */
		}
	}
}

//main kernel which distributes all points into buckets for current logical window of scalar (except last logical windows)
/*
	parameters:
	 points - fixed points
	 buckets - number of buckets for current window
	 idx - bucket id
	 tableS, tableE - markers of starts and ends of grouped indices in array 'numbers'
	 numbers - grouped indices for each bucket in current window
	 bucketSize - size of array 'buckets'
	 continueAdd - flag which indicates continuing addition to current buckets (e.g. pippenger_group_NAF was launch twice)
*/
__global__
__launch_bounds__(128, 4)
void pippenger_group_NAF(const affine_t* __restrict__  points, bucket_t *buckets,
						 const uint32_t* idx, const uint32_t* tableS, const uint32_t* tableE,
						 const uint32_t* numbers, int bucketSize, bool continueAdd)
{
	const int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	const int id = globalId;

	if (id >= bucketSize)
		return;

	int idxBucket = idx[id];
	int start = tableS[idxBucket];
	int end = tableE[idxBucket];

	bucket_t result;
	int resInf = 1;

	if (continueAdd)
	{
		if (idxBucket > 0)
		{
			result = buckets[idxBucket - 1];
			resInf = result.is_inf();
		}
	}
	else
		result.inf();

	if (idxBucket > 0)
	{
		for (int z = start, k = 0; z < end; z++, ++k)
		{
			int idx = numbers[z];

			fp_t X = points[idx].X;
			fp_t Y = points[idx].Y;
			madd_base(X, Y, result, resInf, k == 1 && !continueAdd);
		}

		buckets[idxBucket - 1] = result;
	}
	else
		buckets[bucketSize - 1] = result;

}

//main kernel which distributes all points into buckets for last logical window of scalar
/*
	parameters:
	 points - fixed points
	 buckets - number of buckets for current window
	 idx - bucket id
	 tableS, tableE - markers of starts and ends of grouped indices in array 'numbers'
	 numbers - grouped indices for each bucket in current window
	 bucketSize - size of array 'buckets'
	 continueAdd - flag which indicates continuing addition to current buckets (e.g. pippenger_group_NAF was launch twice)
	 lastMax - the maximum elements in last logical window of scalar
	 threadsCount - number of cuda threads was launched
*/
__global__
__launch_bounds__(128, 4)
void pippenger_group_last_NAF(const affine_t* __restrict__  points, bucket_t *buckets,
							  const uint32_t* idx, const uint32_t* tableS, const uint32_t* tableE,
							  const uint32_t* numbers, bool continueAdd,
							  const int lastMax, const int threadsCount)
{
	int globalId = threadIdx.x  + blockIdx.x * blockDim.x;
	if (globalId >= threadsCount)
		return;

	const int id = globalId % lastMax;
	const int stride = globalId / lastMax;
	const int groups = threadsCount / lastMax;

	int idxBucket = idx[id];
	int start = tableS[idxBucket];
	int end = tableE[idxBucket];

	bucket_t result;
	int resInf = 1;

	if (continueAdd)
	{
		if (idxBucket > 0)
		{
			result = buckets[(idxBucket - 1) + lastMax * stride];
			resInf = result.is_inf();
		}
	}
	else
		result.inf();

	if (idxBucket > 0)
	{
		for (int z = start + stride, k = 0; z < end; z += groups, ++k)
		{
			int idx = numbers[z];

			auto X = points[idx].X;
			auto Y = points[idx].Y;
			madd_base(X, Y, result, resInf, k == 1 && !continueAdd);
		}

		buckets[(idxBucket - 1) + lastMax * stride] = result;
	}
}

//TODO for production: unite two kernels below with C++ template
/// Sums buckets in groups of TILING and TILING_2 correspondingly
///
///	parameters:
///		buckets - number of buckets for set of logical windows
///		ret - partly result of TILING summation
///		elemsInBucket - elements in each bucket
///		sizeBucket - real size of bucket
///		total - total threads was launched
///
__global__
void pippenger_final_NAF_2(bucket_t *buckets, bucket_t (*ret)[TILING_2][2], int elemsInBucket, int sizeBucket, int total)
{
    const int NTHRBITS = dlog2(TILING_2);
	const uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= total)
		return;

	int idBucket = id / elemsInBucket;
	int idInBucket = id % elemsInBucket;

    bucket_t* row = buckets + idBucket * sizeBucket;
    uint32_t i = (1<<(WBITS-1-NTHRBITS));

    row += idInBucket * i;

	__shared__ bucket_t scratch[32];
	int tid = threadIdx.x;

	bucket_t acc = row[--i];
	scratch[tid] = acc;

    while (i--)
	{
		bucket_t p = row[i];

		for (int pc = 0; pc < 2; pc++)
		{
            acc.add(p);
			p = scratch[tid];
            scratch[tid] = acc;
        }
        acc = p;
    }

	ret[idBucket][idInBucket][0] = scratch[tid];
    ret[idBucket][idInBucket][1] = acc;
}

__global__
void pippenger_final_NAF(bucket_t *buckets, bucket_t (*ret)[TILING][2], int elemsInBucket, int sizeBucket, int total)
{
    const int NTHRBITS = dlog2(TILING);
	const uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= total)
		return;

	int idBucket = id / elemsInBucket;
	int idInBucket = id % elemsInBucket;

    bucket_t* row = buckets + idBucket * sizeBucket;
    uint32_t i = (1<<(WBITS-NTHRBITS));

    row += idInBucket * i;

	__shared__ bucket_t scratch[32];
	int tid = threadIdx.x;

	bucket_t acc = row[--i];
	scratch[tid] = acc;

    while (i--)
	{
		bucket_t p = row[i];

		for (int pc = 0; pc < 2; pc++)
		{
            acc.add(p);
			p = scratch[tid];
            scratch[tid] = acc;
        }
        acc = p;
    }

	ret[idBucket][idInBucket][0] = scratch[tid];
    ret[idBucket][idInBucket][1] = acc;
}

/// Distributes scalars into windows of WBITS size. Depending on SIGNED_SCALAR NAF or the singned scalars method is used.
__global__ void initGroup_NAF(const scalar_t* scalars, unsigned int* idx, part_t* keys, int npoints, int shift)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= npoints)
		return;

#if SIGNED_SCALAR
	int plus = 0;
	const int maxVal = (1 << (WBITS - 1));
	const int substruct = (1 << WBITS);
#pragma unroll
	for (int group = 0; group < NWINS; ++group)
	{
		const int wbits = ((group * WBITS) > NBITS-WBITS) ? NBITS-(group * WBITS) : WBITS;
		int val = get_wval(scalars[id + shift], group * WBITS, wbits) + plus;
		plus = 0;
		if (val >= maxVal && group != NWINS - 1)
		{
			val -= substruct;
			plus = 1;
		}

		if (val < 0)
		{
			keys[id + npoints * group] = -val;
			idx[id + npoints * group] = id + shift + npoints;
		}
		else
		{
			keys[id + npoints * group] = val;
			idx[id + npoints * group] = id + shift;
		}
	}

#else
	unsigned X[8], XH[8], X3[8], C[8], NP[8], NM[8];
	for (int z = 0; z < 8; ++z)
		X[z] = scalars[id + shift][z];

	longShift(XH, X, 8);
	longAdd(X3, X, XH, 8);
	longXor(C, XH, X3, 8);
	longAnd(NP, X3, C, 8);
	longAnd(NM, XH, C, 8);

#pragma unroll
	for (int group = 0; group < NWINS; ++group)
	{
		const int wbits = ((group * WBITS) > (NBITS + 1) - WBITS) ? (NBITS + 1) - (group * WBITS) : WBITS;

		int valP = get_wval(NP, group * WBITS, wbits);
		int valN = get_wval(NM, group * WBITS, wbits);
		int diff = valP - valN;

		keys[id + npoints * group] = diff >= 0 ? diff : -diff;
		idx[id + npoints * group] = diff >= 0 ? (id + shift) : (id + shift + npoints);
	}
#endif
}

// заполнение массива rows по ключевым значениям keys
__global__ void initRows(unsigned int* rows, part_t* keys, int npoints)
{
	int id = threadIdx.x  + blockIdx.x * blockDim.x;
	if (id >= npoints)
		return;

	rows[id] = 0;
	if (id > 0)
	{
		if (keys[id] != keys[id - 1])
			rows[id] = keys[id] - keys[id - 1];
	}
	else
		rows[id] = keys[id];
}

// заполнение массивов rows и sizes по индексным массивам tableS и tableE
__global__ void initRows(uint32_t* rows, uint32_t *sizes, const uint32_t* tableS, const uint32_t* tableE, int N)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= N)
		return;

	rows[id] = id;
	sizes[id] = tableE[id] - tableS[id];
}

// заполнение массивов tableS и tableE по массиву rows
__global__ void initTable(unsigned int* rows, unsigned int* tableS, unsigned int* tableE, int npoints)
{
	int id = threadIdx.x  + blockIdx.x * blockDim.x;
	if (id >= npoints)
		return;

	if (id > 0)
	{
		if (rows[id] != rows[id - 1])
		{
			tableS[rows[id]] = id;
			tableE[rows[id - 1]] = id;
		}
	}

	if (id == npoints - 1)
		tableE[rows[id]] = npoints;
}

//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
#else  // ------------------------ HOST PART ------------------------------- //
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//

#include <cassert>
#include <vector>
#include <thread>
#include <map>
#include <set>
#include <algorithm>
#include <iostream>

using namespace std;

#include <util/exception.cuh>
#include <util/rusterror.h>
#include <util/thread_pool_t.hpp>
#include <util/host_pinned_allocator_t.hpp>

class stream_t
{
  cudaStream_t stream;
 public:
  stream_t(int device)
  {
    CUDA_OK(cudaSetDevice(device));
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  }

  ~stream_t() { cudaStreamDestroy(stream); }
  inline operator decltype(stream)() { return stream; }
};

template<class T>
class device_ptr_list_t
{
  vector<T*> d_ptrs;
  vector<bool> isHost;
 public:
  device_ptr_list_t() {}
  ~device_ptr_list_t()
  {
    for (int z = 0; z < d_ptrs.size(); ++z)
    {
      if (isHost[z])
        cudaFreeHost(d_ptrs[z]);
      else
        cudaFree(d_ptrs[z]);
    }
  }

  //TODO: need to take account of type T !!
  size_t allocate(size_t bytes, bool onHost = false)
  {
    T *d_ptr;
    if (onHost)
      CUDA_OK(cudaMallocHost(&d_ptr, bytes));
    else
      CUDA_OK(cudaMalloc(&d_ptr, bytes));
    d_ptrs.push_back(d_ptr);
    isHost.push_back(onHost);
    return d_ptrs.size() - 1;
  }

  size_t size() { return d_ptrs.size(); }

  T* operator[](size_t i)
  {
    if (i > d_ptrs.size() - 1)
      CUDA_OK(cudaErrorInvalidDevicePointer);
    return d_ptrs[i];
  }
};

// Pippenger MSM class

template<class bucket_t, class point_t, class affine_t, class scalar_t>
class pippenger_t {

 private:
  size_t sm_count;
  bool init_done = false;
  device_ptr_list_t<affine_t> d_base_ptrs;
  device_ptr_list_t<scalar_t> d_scalar_ptrs;
  device_ptr_list_t<bucket_t> d_bucket_ptrs;

  //TODO for production: only one pointer per each variable, need to improve this!
  device_ptr_list_t<part_t> part_prts;
  device_ptr_list_t<unsigned> idxs_prts;
  device_ptr_list_t<unsigned> rows_prts;
  device_ptr_list_t<unsigned> tableS_prts;
  device_ptr_list_t<unsigned> sizes_prts;
  device_ptr_list_t<bucket_t> d_buckets_ptrs;
  device_ptr_list_t<bucket_t> d_none_ptrs;
  device_ptr_list_t<unsigned> pointNotNull;

  device_ptr_list_t<bucket_t> c_none_ptrs;
  device_ptr_list_t<bucket_t> cpuBucket_ptrs;
  int copied;
  point_t integrated_prt[2][NWINS];
  point_t integrated_batched[4 * NWINS];
  point_t integrated_cpu[4];

  int numCPUAsync[2];
  vector<cudaEvent_t> events;

  // GPU device number
  int device;

  // TODO: Move to device class eventually
  thread_pool_t *da_pool = nullptr;

  vector<cudaStream_t> streams;
  vector<cudaStream_t> copy_streams;
 public:
  // Default stream for operations
  stream_t default_stream;

  // Parameters for an MSM operation
  class MSMConfig
  {
    friend pippenger_t;
   public:
    size_t npoints;

    int getOneWave() const { return oneWaveBuckets; }
   private:
    size_t n;
    int numAsync;
    int numBlocks;
    int thF;
    int oneWaveBuckets;
    int naf_elems;
    double naf_coef;
  };

  pippenger_t() : default_stream(0) { device = 0; }

  pippenger_t(int _device, thread_pool_t *pool = nullptr) : default_stream(_device)
  {
    da_pool = pool;
    device = _device;
  }

  ~pippenger_t()
  {
    for (auto& st : streams)
      cudaStreamDestroy(st);
    for (auto&st : copy_streams)
      cudaStreamDestroy(st);
    for (auto& ev : events)
      cudaEventDestroy(ev);
    clearCache();
  }

  // Initialize instance. Throws cuda_error on error.
  void init()
  {
    if (!init_done)
    {
      CUDA_OK(cudaSetDevice(device));
      cudaDeviceProp prop;
      if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess || prop.major < 7)
        CUDA_OK(cudaErrorInvalidDevice);

      sm_count = prop.multiProcessorCount;

      for (int z = 0; z < 6; ++z)
      {
        cudaEvent_t tmp;
        CUDA_OK(cudaEventCreate(&tmp));
        events.push_back(tmp);
      }

      if (da_pool == nullptr)
        da_pool = new thread_pool_t();
      init_done = true;
    }
  }

  int get_device() { return device; }

  // Initialize parameters for a specific size MSM. Throws cuda_error on error.
  MSMConfig init_msm(size_t npoints)
  {
    init();

    MSMConfig config;
    config.npoints = npoints;

    config.n = (npoints+WARP_SZ-1) & ((size_t)0-WARP_SZ);

    config.numAsync = 2;
    config.thF = 32;

    if (NAF)
    {
      if (SIGNED_SCALAR)
      {
        config.naf_coef = 0.585;
        config.naf_elems = (int)(TILING * config.naf_coef);
      }
      else
      {
        config.naf_coef = 2.0 / 3.0;
        config.naf_elems = (int)(TILING * config.naf_coef) + 1;
      }
    }
    else
      config.naf_elems = TILING;

    CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor (&config.numBlocks, pippenger_final_NAF, config.thF, 0));

    const int totalThreads = config.numBlocks * config.thF * sm_count;

    if (SIGNED_SCALAR)
      config.oneWaveBuckets = std::max(NAF ? (int)(NWINS - 1) : (int)NWINS, (int)totalThreads / (TILING / 2));
    else
      config.oneWaveBuckets = std::max(NAF ? (int)(NWINS - 1) : (int)NWINS, (int)totalThreads / config.naf_elems);

    for (int z = 0; z < config.numAsync + 1; ++z)
      streams.push_back(0);
    for (int z = 0; z < NWINS; ++z)
      copy_streams.push_back(0);

    for (int z = 0; z < config.numAsync + 1; ++z)
      CUDA_OK(cudaStreamCreate(&streams[z]));

    for (int z = 0; z < NWINS; ++z)
      CUDA_OK(cudaStreamCreate(&copy_streams[z]));

    clearCache();
    return config;
  }

  size_t get_size_bases(MSMConfig& config)   { return config.n * sizeof(affine_t) * 2; }
  size_t get_size_scalars(MSMConfig& config) { return config.n * sizeof(scalar_t); }

  // Allocate storage for bases on device. Throws cuda_error on error.
  // Returns index of the allocated base storage.
  size_t allocate_d_bases(MSMConfig& config) { return d_base_ptrs.allocate(get_size_bases(config)); }

  // Allocate storage for scalars on device. Throws cuda_error on error.
  // Returns index of the allocated base storage.
  size_t allocate_d_scalars(MSMConfig& config) { return d_scalar_ptrs.allocate(get_size_scalars(config)); }

  void allocate_tables(MSMConfig& config)
  {
    const unsigned tN = (1 << WBITS);
    //TODO: move it to config
    const int batches = 4;

    part_prts.allocate(sizeof(part_t) * config.npoints * NWINS);
    idxs_prts.allocate(sizeof(unsigned) * config.npoints * NWINS);
    rows_prts.allocate(sizeof(unsigned) * std::max(config.npoints, (size_t)tN) * config.numAsync);
    tableS_prts.allocate(sizeof(unsigned) * tN * (2 * config.numAsync));
    sizes_prts.allocate(sizeof(unsigned) * tN * config.numAsync);

    d_buckets_ptrs.allocate(sizeof(bucket_t) * (config.naf_elems * (tN / TILING)) * config.oneWaveBuckets);
    d_buckets_ptrs.allocate(sizeof(bucket_t) * (config.naf_elems * (tN / TILING)) * 2);

    CUDA_OK(cudaMemset(d_buckets_ptrs[0], 0, sizeof(bucket_t) * (config.naf_elems * (tN / TILING)) * config.oneWaveBuckets));
    CUDA_OK(cudaMemset(d_buckets_ptrs[1], 0, sizeof(bucket_t) * (config.naf_elems * (tN / TILING)) * 2));

    cpuBucket_ptrs.allocate(sizeof(bucket_t) * (config.naf_elems * (tN / TILING)) * 2, true);
    memset(cpuBucket_ptrs[0], 0, sizeof(bucket_t) * (config.naf_elems * (tN / TILING)) * 2);
    copied = 0;

    d_none_ptrs.allocate(sizeof(bucket_t) * NWINS * TILING * 2 * batches);

    c_none_ptrs.allocate(sizeof(bucket_t) * (NWINS * TILING * 2 * batches + TILING_2 * 2 * 2), true);
    c_none_ptrs.allocate(sizeof(bucket_t) * NWINS * TILING * 2 * batches, true);

    size_t free, total;
    cudaMemGetInfo (&free, &total);
    //printf("total mem %lld, free mem %lld\n", total, free);
  }

  size_t get_num_base_ptrs()   { return d_base_ptrs.size();   }
  size_t get_num_scalar_ptrs() { return d_scalar_ptrs.size(); }
  size_t get_num_bucket_ptrs() { return d_bucket_ptrs.size(); }

  // Transfer bases to device. Throws cuda_error on error.
  void transfer_bases_to_device(MSMConfig& config, size_t d_bases_idx, const affine_t points[], size_t ffi_affine_sz = sizeof(affine_t), cudaStream_t s = nullptr)
  {
    cudaStream_t stream = (s == nullptr) ? default_stream : s;
    affine_t *d_points = d_base_ptrs[d_bases_idx];
    CUDA_OK(cudaSetDevice(device));

    if (ffi_affine_sz != sizeof(*d_points))
    {
      CUDA_OK(cudaMemcpy2DAsync(d_points, sizeof(*d_points), points, ffi_affine_sz, ffi_affine_sz, config.npoints, cudaMemcpyHostToDevice, stream));
      zero_points<<<config.npoints / 256 + 1, 256, 0, stream>>>(d_points, config.npoints);
      CUDA_OK(cudaStreamSynchronize(stream));
    }
    else
      CUDA_OK(cudaMemcpyAsync(d_points, points, config.npoints*sizeof(*d_points), cudaMemcpyHostToDevice, stream));
  }

  // Transfer scalars to device. Throws cuda_error on error.
  void transfer_scalars_to_device(size_t d_scalars_idx, const scalar_t scalars[], size_t size, cudaStream_t s = nullptr)
  {
    cudaStream_t stream = (s == nullptr) ? default_stream : s;
    scalar_t *d_scalars = d_scalar_ptrs[d_scalars_idx];
    CUDA_OK(cudaSetDevice(device));
    CUDA_OK(cudaMemcpyAsync(d_scalars, scalars, size * sizeof(*d_scalars), cudaMemcpyHostToDevice, stream));
  }

  static point_t integrate_row_host_2(const bucket_t row[TILING_2][2], int wbits, bool isLastBucket)
  {
    const int NTHRBITS = log2(TILING_2);
    size_t i = TILING_2-1;

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
      for (size_t j = 0; j < WBITS-1-NTHRBITS; j++)
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

  static point_t integrate_row_host(const bucket_t row[TILING][2], int wbits, bool isLastBucket)
  {
    const int NTHRBITS = log2(TILING);
    size_t i = TILING-1;
    if (NAF)
    {
      if (SIGNED_SCALAR)
      {
        if (isLastBucket)
          i = (int)(TILING / 3.0 * 2.0) + 1;
        else
          i = TILING / 2;
      }
      else
        i = (int)(TILING / 3.0 * 2.0) + 1;
    }

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

  // threads for async summing partial sums in a bucket on CPU
  vector<thread*> asyncIntegrate;
  // threads for async summing bucket parts on CPU
  vector<thread*> asyncLastPart;

  /// Sums points for the latest windows in case WBITS == 23
  void sumPartLastBucketAsync(int elems, int numBucket, int numIter)
  {
    CUDA_OK(cudaStreamSynchronize(copy_streams[1]));

    const int inRow = (1 << WBITS) / TILING;
    bucket_t* base = cpuBucket_ptrs[0] + inRow * elems * numBucket;
    bucket_t (*c_none_1)[TILING][2] = reinterpret_cast<decltype(c_none_1)>(c_none_ptrs[1]);

#pragma omp parallel for schedule(dynamic)
    for (int z = 0; z < elems; ++z)
    {
      bucket_t* row = base + z * inRow;

      bucket_t res, sum;
      int t = inRow - 1;
      sum = row[t];
      res = row[t];
      --t;
      while (t >= 0)
      {
        sum.add(row[t]);
        res.add(sum);
        --t;
      }

      c_none_1[numIter][TILING / 2 + z][0] = res;
      c_none_1[numIter][TILING / 2 + z][1] = sum;
    }
  }

  /// Sums points for the latest windows in case WBITS != 23
  void sumLastBucketAsync(int elemsInBucket, int groups, int numIter)
  {
    CUDA_OK(cudaStreamSynchronize(copy_streams[1]));

#pragma omp parallel for
    for (int z = 0; z < elemsInBucket - 1; ++z)
      for (int k = 1; k < groups; ++k)
        cpuBucket_ptrs[0][z].add(cpuBucket_ptrs[0][z + k * elemsInBucket]);

    bucket_t res, sum;
    int t = elemsInBucket - 2;
    sum = cpuBucket_ptrs[0][t];
    res = cpuBucket_ptrs[0][t];
    --t;
    while (t >= 0)
    {
      sum.add(cpuBucket_ptrs[0][t]);
      res.add(sum);
      --t;
    }
    integrated_batched[(NWINS - 1) + numIter * NWINS] = res;
  }

  /// Do async copying of the next scalar group
  static void moveNext(scalar_t *d_scalars, const scalar_t* scalars, size_t shift, size_t points,	cudaStream_t& st)
  {
    CUDA_OK(cudaMemcpyAsync(d_scalars + shift, scalars, points * sizeof(scalar_t), cudaMemcpyHostToDevice, st));
    CUDA_OK(cudaStreamSynchronize(st));
  }

  /// The main hanler of the algorithm.
  /// arguments:
  ///   numIter - number of a scalar group
  ///   bucketsDone - number of computed windows. F.e. for one scalar group it equals 253/ WBITS
  ///   allDone - the whole number of computed windows
  ///   config - run configuration
  void launch_best(int numIter,
                   int& bucketsDone, int& allDone, point_t* out,
                   MSMConfig& config, const size_t d_bases_idx, const size_t d_scalars_idx, const scalar_t* scalars)
  {
    const int idx = numIter % 2;
    const unsigned tN = (1 << WBITS);
    const size_t npoints = config.npoints;
    const int numStrs = config.numAsync;

    affine_t *d_points = d_base_ptrs[d_bases_idx];
    scalar_t *d_scalars = d_scalar_ptrs[d_scalars_idx];

    part_t* part = part_prts[0];
    unsigned* idxs = idxs_prts[0];
    unsigned* rows = rows_prts[0];
    unsigned* tableS = tableS_prts[0];
    unsigned* tableE = tableS + tN;
    unsigned* sizes = sizes_prts[0];

    bucket_t* d_buckets = d_buckets_ptrs[0];
    bucket_t (*d_none)[TILING][2] = reinterpret_cast<decltype(d_none)>(d_none_ptrs[0]);
    bucket_t (*c_none)[TILING][2] = reinterpret_cast<decltype(c_none)>(c_none_ptrs[0]);

    unsigned* rowsA[numStrs];
    unsigned* tableS_A[numStrs];
    unsigned* tableE_A[numStrs];
    unsigned* sizesA[numStrs];

    for (int z = 0; z < numStrs; ++z)
    {
      rowsA[z] = rows + z * std::max(npoints, (size_t)tN);
      sizesA[z] = sizes + z * tN;
      tableS_A[z] = tableS + 2 * z * tN;
      tableE_A[z] = tableS + tN + 2 * z * tN;
    }

    const int xBlock = npoints / 256 + ((npoints % 256) != 0);
    const int xBlockT = tN / 256 + ((tN % 256) != 0);

    const int thF = config.thF;
    const int xBlockF = (TILING * NWINS) / thF + (((TILING * NWINS) % thF) != 0);

    const int thP = 32;
    const int finalTh = 1 << (NBITS % NWINS);

    int realTab = (int)(tN * config.naf_coef) + 1;
    int realTabAligned = config.naf_elems * (tN / TILING);

    const int xBlockP = realTab / thP + ((realTab % thP) != 0);
    const int xBlockP_half = (1 << (WBITS - 1)) / thP;

    int numBlocks = config.numBlocks;
    numCPUAsync[idx] = 0;

    int numStr = 0;
    initGroup_NAF<<<xBlock, 256, 0, streams[numStr]>>>(d_scalars, idxs, part, npoints, 0); // TODO -> delete last arg

    // first init
    if (bucketsDone == 0 && numIter == 0)
    {
      //TODO: batches = 4, move to config
      // init partial sums
      memset(c_none_ptrs[0], 0, sizeof(bucket_t) * TILING * 2 * NWINS * 4);
    }
    CUDA_OK(cudaStreamSynchronize(streams[numStr]));

    if (bucketsDone == config.oneWaveBuckets)
      doFinalSum(-2, numStrs, bucketsDone, allDone, config, realTabAligned, false);

    const bool continueAdd = false; // TODO -> delete

    thread* copyNextAsync = NULL;
    thread* lastSumAsync = NULL;
    bucket_t* d_bucket_tmp = d_buckets_ptrs[1];

    // the main cycle for windows in scalars
    // this cycle runs in numStrs CUDA streams
    for (int group = NWINS - 1, k = 0; group >= 0; --group, ++k)
    {
      unsigned elemsInBucket = 0;

      if (k == 2 && scalars) // run async copying of the next scalar group
        copyNextAsync = new thread(moveNext, ref(d_scalars), ref(scalars), 0, config.npoints, ref(streams.back())); // TODO-> delete second param

      numStr = group % numStrs;

      // current CUDA stream
      cudaStream_t currSt = streams[numStr];

      if (!(group != NWINS - 1 || (NBITS % WBITS) == 0)) // last
        currSt = copy_streams[1];

      sort(idxs + npoints * group, part + npoints * group, npoints, currSt, ((group == NWINS - 1) && ((NBITS % WBITS) != 0)) ? &elemsInBucket : NULL);
      initRows<<<xBlock, 256, 0, currSt>>>(rowsA[numStr], part + npoints * group, npoints);
      scan(rowsA[numStr], npoints, currSt);
      CUDA_OK(cudaMemsetAsync(tableS_A[numStr], 0, sizeof(int) * tN * 2, currSt));
      initTable<<<xBlock, 256, 0, currSt>>>(rowsA[numStr], tableS_A[numStr], tableE_A[numStr], npoints);
      initRows<<<xBlockT, 256, 0, currSt>>>(rowsA[numStr], sizesA[numStr], tableS_A[numStr], tableE_A[numStr], tN);
      // distribute indices by number of elements
      sort1(rowsA[numStr], sizesA[numStr], tN, currSt, true);

      // it's true for all window except tha latest in case (NBITS % WBITS) != 0
      // or for all windows in case (NBITS % WBITS) == 0
      if (group != NWINS - 1 || (NBITS % WBITS) == 0)
      {
        int blockX = xBlockP;
        int bucketSize = realTab;

        // ddp sums of points in a bucket
        pippenger_group_NAF<<<blockX, thP, 0, currSt>>>(d_points, d_buckets + realTabAligned * bucketsDone, rowsA[numStr], tableS_A[numStr], tableE_A[numStr], idxs + npoints * group, bucketSize, continueAdd);

        if (group == NWINS - 1 && (NBITS % WBITS) == 0)
        {
          // for the latest window in case WBITS == 23 the count of elements bugger then then (1 << (WBITS - 1)),
          // so for balancing sums for scalars in interval [1 << (WBITS - 1), MAX_SCALAR_VAL_23_BIT - (1 << (WBITS - 1))] done on CPU in parallel
          CUDA_OK(cudaStreamSynchronize(currSt));
          const int shift = (1 << (WBITS - 1));
          const int maxElems = MAX_SCALAR_VAL_23_BIT; // max for 253bit scalar and only selected MOD
          const int numReduceElems = (1 << WBITS) / TILING;
          const int count = ((maxElems - shift) / numReduceElems + 1) * numReduceElems; // align to numReduceElems chunks

          CUDA_OK(cudaMemcpyAsync(cpuBucket_ptrs[0] + copied * count, d_buckets + realTabAligned * bucketsDone + shift, sizeof(bucket_t) * count, cudaMemcpyDeviceToHost, copy_streams[1]));

          thread* accAsyncJob = new thread(&pippenger_t::sumPartLastBucketAsync, this, count / numReduceElems, copied, numIter);
          asyncLastPart.push_back(accAsyncJob);
          copied++;
        }
      }
      else
      {
        bucket_t* d_bucket_tmp = d_buckets_ptrs[1];
        int threadCount = ((384 * sm_count * 8) / elemsInBucket) * elemsInBucket;
        int xBlockP = threadCount / thP + ((threadCount % thP) != 0);
        int groups = threadCount / elemsInBucket;

        // do sums of points in a bucket
        pippenger_group_last_NAF<<<xBlockP, thP, 0, currSt>>>(d_points, d_bucket_tmp, rowsA[numStr], tableS_A[numStr], tableE_A[numStr], idxs + npoints * group, false, elemsInBucket, threadCount);
        bucketsDone--;

        // for the latest window in case WBITS != 23 the count of elements much less then maximum (1 << (WBITS - 1)),
        // so for balancing it's done on CPU in parallel
        CUDA_OK(cudaMemcpyAsync(cpuBucket_ptrs[0], d_bucket_tmp, threadCount * sizeof(bucket_t), cudaMemcpyDeviceToHost, currSt));
        lastSumAsync = new thread(&pippenger_t::sumLastBucketAsync, this, elemsInBucket, groups, numIter);
      }

      bucketsDone++;

      // do the final sum if a number of calculated sum equals to threshold
      if (bucketsDone == config.oneWaveBuckets)
        doFinalSum(k, numStrs, bucketsDone, allDone, config, realTabAligned, (k == NWINS - 1) && scalars == NULL);
    }

    // wait for a func that executes in parallel on CPU
    if (lastSumAsync)
    {
      lastSumAsync->join();
      delete lastSumAsync;
      lastSumAsync = NULL;
    }

    //last buckets summation
    if (bucketsDone && scalars == NULL)
      doFinalSum(-1, numStrs, bucketsDone, allDone, config, realTabAligned, true);

    // wait for copying of next scalar group in GPU
    if (copyNextAsync)
    {
      copyNextAsync->join();
      delete copyNextAsync;
    }
  }

  // Sums partial sums with grouping TILING/TILING_2 on GPU
  void doFinalSum(int k, int numStrs, int& bucketsDone, int& allDone, MSMConfig& config, int bucketSize, bool isLast)
  {
    copied = 0;
    bucket_t* d_buckets = d_buckets_ptrs[0];
    bucket_t (*d_none)[TILING][2] = reinterpret_cast<decltype(d_none)>(d_none_ptrs[0]);

    cudaStream_t currS = streams[0];

    for (int z = 0; z < numStrs; ++z)
      CUDA_OK(cudaStreamSynchronize(streams[z]));

    for (int z = 0; z < asyncIntegrate.size(); ++z)
    {
      asyncIntegrate[z]->join();
      delete asyncIntegrate[z];
    }
    asyncIntegrate.clear();

    int blocks = (config.naf_elems * bucketsDone) / config.thF + 1;
    CUDA_OK(cudaMemsetAsync(d_none, 0, bucketsDone * sizeof(d_none[0]), currS));

    if (SIGNED_SCALAR)
    {
      if (bucketsDone == 2 && (NBITS % WBITS) == 0)
      {
        const int realElems = TILING_2;
        blocks = (realElems * bucketsDone) / config.thF;

        bucket_t (*d_none_2)[TILING_2][2] = reinterpret_cast<decltype(d_none_2)>(d_none_ptrs[0]);
        pippenger_final_NAF_2<<<blocks, config.thF, 0, currS>>>(d_buckets, d_none_2, realElems, bucketSize, realElems * bucketsDone);
      }
      else
      {
        const int realElems = TILING / 2;
        blocks = (realElems * bucketsDone) / config.thF;

        pippenger_final_NAF<<<blocks, config.thF, 0, currS>>>(d_buckets, d_none, realElems, bucketSize, realElems * bucketsDone);
      }
    }
    else
      pippenger_final_NAF<<<blocks, config.thF, 0, currS>>>(d_buckets, d_none, config.naf_elems, bucketSize, config.naf_elems * bucketsDone);

    CUDA_OK(cudaStreamSynchronize(currS));

    if (SIGNED_SCALAR)
    {
      for (int z = 0; z < asyncLastPart.size(); ++z)
      {
        asyncLastPart[z]->join();
        delete asyncLastPart[z];
      }
      asyncLastPart.clear();
    }

    thread* accAsyncJob = new thread(&pippenger_t::intergateAsync, this, allDone, allDone + bucketsDone, TILING * 2 * allDone, bucketsDone);
    asyncIntegrate.push_back(accAsyncJob);

    allDone += bucketsDone;
    bucketsDone = 0;
  }

  // Sums partial sums with grouping TILING/TILING_2 on CPU
  void intergateAsync(int start, int end, int shiftCopy, int bucketsDone)
  {
    bucket_t (*d_none)[TILING][2] = reinterpret_cast<decltype(d_none)>(d_none_ptrs[0]);
    bucket_t (*d_none1)[TILING_2][2] = reinterpret_cast<decltype(d_none1)>(d_none_ptrs[0]);

    if ((NBITS % WBITS) == 0 && bucketsDone == 2)
      CUDA_OK(cudaMemcpyAsync(c_none_ptrs[0] + shiftCopy, d_none1, bucketsDone * sizeof(d_none1[0]), cudaMemcpyDeviceToHost, copy_streams[0]));
    else
      CUDA_OK(cudaMemcpyAsync(c_none_ptrs[0] + shiftCopy, d_none, bucketsDone * sizeof(d_none[0]), cudaMemcpyDeviceToHost, copy_streams[0]));
    CUDA_OK(cudaStreamSynchronize(copy_streams[0]));

    bucket_t (*c_none)[TILING][2] = reinterpret_cast<decltype(c_none)>(c_none_ptrs[0]);
    bucket_t (*c_none_1)[TILING][2] = reinterpret_cast<decltype(c_none_1)>(c_none_ptrs[1]);
    bucket_t (*c_none_2)[TILING_2][2] = reinterpret_cast<decltype(c_none_2)>(c_none_ptrs[0] + shiftCopy);

    int wins = NWINS;
    if ((NBITS%WBITS) != 0)
      wins = NWINS - 1;

#pragma omp parallel for schedule(dynamic)
    for (int p = start; p < end; ++p)
    {
      int z = (wins - 1) - (p % wins);
      int shift = (p / wins) * NWINS;

      if (z == NWINS - 1) // last bucket
      {
        if ((NBITS%WBITS) == 0)
        {
          if (SIGNED_SCALAR)
          {
            int err = 0;

            const int shift = (1 << (WBITS - 1));
            const int maxElems = MAX_SCALAR_VAL_23_BIT;
            const int numReduceElems = (1 << WBITS) / TILING;
            const int count = (maxElems - shift) / numReduceElems + 1; // align to numReduceElems chunks

            for (int z = TILING / 2; z < TILING / 2 + count; ++z)
            {
              c_none[p][z][0] = c_none_1[p / NWINS][z][0];
              c_none[p][z][1] = c_none_1[p / NWINS][z][1];
            }
          }
          integrated_batched[z + shift] = integrate_row_host(c_none[p], WBITS, true);

        }
        else
          ;//integrated_batched[z + shift] = integrated_cpu[batch];
      }
      else
      {
        if (SIGNED_SCALAR)
        {
          if ((NBITS % WBITS) == 0 && bucketsDone == 2)
            integrated_batched[z + shift] = integrate_row_host_2(c_none_2[p - start], WBITS, false);
          else
            integrated_batched[z + shift] = integrate_row_host(c_none[p], WBITS, false);
        }
        else
          integrated_batched[z + shift] = integrate_row_host(c_none[p], WBITS, false);
      }
    }
  }

  // Wait all async operations
  void waitAll()
  {
    if (asyncIntegrate.size() == 0)
      printf("nothing to wait\n");

    for (auto& elem : asyncIntegrate)
    {
      elem->join();
      delete elem;
    }
    asyncIntegrate.clear();
  }

  static point_t pippenger_final_host(point_t* integrated)
  {
    size_t i = NWINS - 1;
    point_t res	= integrated[i];
    while (i--)
    {
      for (size_t j = 0; j < WBITS; j++)
        res.dbl();
      res.add(integrated[i]);
    }
    return res;
  }

  // Perform final accumulation on CPU
  void accumulateAsync(point_t *out, int start, int end)
  {
    for (int p = start; p < end; ++p)
      out[p] = pippenger_final_host(integrated_batched + p * NWINS);
  }
};



#endif
