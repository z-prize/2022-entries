// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>
#include <chrono>
#include <omp.h>
#include <mutex>
#include <bit>
#include <bitset>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

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
/* sizeof(B[0]) = 192b
W=16 -> N 16  T->256 KB   B->0.01 GB / WIN
W=17 -> N 15  T->512 KB   B->0.02 GB / WIN
W=19 -> N 14  T->2048KB   B->0.08 GB / WIN
W=20 -> N 13  T->4096 KB  B->0.18 GB / WIN
W=22 -> N 12  T->16 MB    B->0.75 GB / WIN
W=23 -> N 11  T->32 MB    B->1.5  GB / WIN 

W=26 -> N 10  T->256 MB   B->12   GB / WIN
W=29 -> N 9   T->2 GB     B->96   GB / WIN
W=32 -> N 8   T->16 GB    B->768  GB / WIN

NOTE! for W > 25 get_wval does not work!
*/

#define MAX_SCALAR_VAL_23_BIT 4894102

#ifndef NBITS
# define NBITS 253
#endif
#ifndef WBITS
# define WBITS 23
#endif
#define NWINS ((NBITS+WBITS-1)/WBITS)   // ceil(NBITS/WBITS)

/*#ifndef LARGE_L1_CODE_CACHE
# define LARGE_L1_CODE_CACHE 0
#endif*/

#ifndef TILING 
#define TILING   2048
#define TILING_2 4096
#endif

#define CPU_CMP 0
#define WITH_TABLES 0
#define BUCKET_TEST 0
#define REF 0

#define PRINT_ASYNC 1
#define PRINT 1
#define PROF 1

#define CPU_CHECK 0
#define FROM_CPU 0

#define NAF 1
#define SIGNED_SCALAR 1

#define SIZE 2
#define WITH_NULL 0

static const int NTH = 1;

#if WBITS<=16
typedef unsigned short part_t;
#else
typedef unsigned int part_t;
#endif

//
// To be launched as 'pippenger<<<dim3(NWINS, N), NTHREADS>>>(...)'  with
// |npoints| being around N*2**20 and N*NWINS*NTHREADS not exceeding the
// occupancy limit.
//

struct AffinePoint
{
	fp_t X, Y;
};

__global__
void pippenger(const affine_t* points, size_t npoints,
               const scalar_t* scalars, bool mont,
               bucket_t (*buckets)[NWINS][1<<WBITS],
               bucket_t (*ret)[NWINS][NTHREADS][2] = nullptr);

__global__ void pippenger_group_NAF(const affine_t* __restrict__  points, bucket_t *buckets, const uint32_t* idx, const uint32_t* tableS, const uint32_t* tableE, const uint32_t* numbers, int bucketSize, bool continueAdd);
__global__ void pippenger_group_last_NAF(const affine_t* __restrict__  points, bucket_t *buckets, const uint32_t* idx, const uint32_t* tableS, const uint32_t* tableE, const uint32_t* numbers, bool continueAdd, const int lastMax, const int threadsCount);
__global__ void pippenger_group_last_NAF(bucket_t *buckets, const bucket_t *bucket_tmp, const int lastMax, const int groups);
								
__global__ void pippenger_group(const affine_t* points, bucket_t *buckets, const uint32_t* idx, const uint32_t* tableS, const uint32_t* tableE, const uint32_t* numbers, int wnum, bool continueAdd);
__global__ void pippenger_group_last(const affine_t* points, bucket_t *buckets, const uint32_t* idx, const uint32_t* tableS, const uint32_t* tableE, const uint32_t* numbers, bool continueAdd);

__global__ void pippenger_final(bucket_t *buckets, bucket_t (*ret)[TILING][2], int shift);
__global__ void pippenger_final_NAF(bucket_t *buckets, bucket_t (*ret)[TILING][2], int elemsInBucket, int sizeBucket, int total);
__global__ void pippenger_final_NAF_2(bucket_t *buckets, bucket_t (*ret)[TILING_2][2], int elemsInBucket, int sizeBucket, int total);

__global__ void zero_bucket(bucket_t *buckets, int elemsInBucket, int sizeBucket, int total);


__global__ void initGroup_full(const scalar_t* scalars, unsigned int* idx, part_t* keys, int npoints);
__global__ void initGroup_full2(const scalar_t* scalars, unsigned int* idx, part_t* keys, int npoints, int shift);
__global__ void initGroup_NAF(const scalar_t* scalars, unsigned int* idx, part_t* keys, int npoints, int shift);
__global__ void initIdx(unsigned int* idx, int npoints);
__global__ void initRows(unsigned int* idx, part_t* keys, int npoints);
__global__ void initRows(uint32_t* rows, uint32_t *sizes, const uint32_t* tableS, const uint32_t* tableE, int N);
__global__ void initTable(unsigned int* rows, unsigned int* tableS, unsigned int* tableE, int npoints);
__global__ void copyIdxs(const unsigned int* idxIn, unsigned int* idxOut, int npoints);
__global__ void zero_points(affine_t* points, size_t N);
__global__ void fill_not_null(const affine_t* points, unsigned* out, size_t N);
__global__ void copyToHost(const affine_t* points, fp_t* X, fp_t* Y, int npoints);

__global__ void countTable(const part_t* parts, const unsigned* idx, unsigned int* counter, unsigned* place, int N);

extern "C" void sort(unsigned int* idxs, part_t* part, size_t npoints, cudaStream_t& st, unsigned* getGroups);
extern "C" void sort1(unsigned int* idxs, unsigned int* sizes, size_t npoints, cudaStream_t& st, bool greater);
extern "C" void scan(unsigned int* rows, size_t npoints, cudaStream_t& st);
extern "C" void clearCache();

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
				/*if (bid == 1)
				{
					int* t = (int*)&row[wval-1];
					int z = 0;
					printf("- %d %d = %u %u %u %u %u %u %u %u %u %u %u %u\n", wval-1, i, t[z++], t[z++], t[z++], t[z++], t[z++], t[z++], t[z++], t[z++], t[z++], t[z++], t[z++], t[z++]);
				}*/
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

__global__ void countTable(const part_t* parts, const unsigned* idx, unsigned int* counter, unsigned* place, int N)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= N)
		return;
	
	part_t partN = parts[tid];
	auto val = idx[tid];
	
	auto oldIdx = atomicInc(counter + partN, N);
	place[partN * 64 + oldIdx] = val;
}


struct affinePair
{
	fp_t X, Y;
};

__device__ void addAffine(affinePair& t1, const fp_t& X, const fp_t& Y, const fp_t& d, bool doubling)
{
	fp_t x2, L;
	x2 = X + t1.X;
	if (doubling)
	{
		L = X * X;
		L = L + L + L;
	}
	else
		L = Y - t1.Y;
	
	L *= d;			
	t1.X = L * L;
	t1.X -= x2;
	
	t1.Y = X - t1.X;
	t1.Y *= L;
	t1.Y -= Y;
}
	
__device__ void longAdd(unsigned* t3, const unsigned* t1, const unsigned* t2, int N) // t3 = t1 + t2
{
	uint64_t bitNext = 0;
	for (int z = 0; z < N; ++z)
	{
		uint64_t val = (uint64_t)t1[z] + (uint64_t)t2[z] + bitNext;
		t3[z] = (int)val;
		bitNext = (val >> 32) ? 1 : 0;
	}
}

__device__ void longShift(unsigned* t2, const unsigned* t1, int N) // t2 = t1 >> 1;
{
	t2[0] = t1[0] >> 1;
	for (int z = 1; z < N; ++z)
	{
		if (t1[z] & 1)
			t2[z - 1] |= ((unsigned)1 << 31);
		t2[z] = t1[z] >> 1;
	}
}

__device__ void longXor(unsigned* t3, const unsigned* t1, const unsigned* t2, int N) // t3 = t2 ^ t1
{
	for (int z = 0; z < N; ++z)
		t3[z] = t2[z] ^ t1[z];
}

__device__ void longAnd(unsigned* t3, const unsigned* t1, const unsigned* t2, int N) // t3 = t2 & t1
{
	for (int z = 0; z < N; ++z)
		t3[z] = t2[z] & t1[z];
}

__device__ void longOr(unsigned* t3, const unsigned* t1, const unsigned* t2, int N) // t3 = t2 | t1
{
	for (int z = 0; z < N; ++z)
		t3[z] = t2[z] | t1[z];
}
	
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

__global__ void fill_not_null(const affine_t* points, unsigned* out, size_t N)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= N)
		return;
	
	unsigned res = 0;
	int id = tid * 32;
	for (int i = id, k = 0; i < id + 32; ++i, ++k)
	{
		bool isZero = points[i].X.is_zero() && points[i].Y.is_zero();
		if (!isZero)
			res |= (1 << k);
	}
	out[tid] = res;
}

__global__ void copyToHost(const affine_t* points, fp_t* X, fp_t* Y, int npoints)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= npoints)
		return;
	
	X[id] = points[id].X;
	Y[id] = points[id].Y;
}

__device__ __inline__ void add(bucket_t& p31, bucket_t& p2)
{    		
	if (p2.is_inf()) 
		return;
	else if (p31.is_inf()) 
	{
		p31 = p2;
		return;
	}

	fp_t P, R;

	P = p2.X * p31.ZZ;
	R = p2.Y * p31.ZZZ;
	
	p2.X = p31.X * p2.ZZ;  //U
	p2.Y = p31.Y * p2.ZZZ; //S

	P -= p2.X;
	R -= p2.Y;

	if (!P.is_zero()) 
	{
		p31.X = P*P; // PP
		P = P*p31.X;
		
		p31.ZZ *= p31.X;           
		p31.ZZ *= p2.ZZ;
		
		p31.ZZZ *= P;		
		p31.ZZZ *= p2.ZZZ;
		
		p2.ZZ = p2.X * p31.X;
		p31.X = R*R;            
		p31.X -= P;           
		p31.X -= p2.ZZ;
		p31.X -= p2.ZZ;             
		p2.ZZ -= p31.X;
		p2.ZZ *= R;                 
		p31.Y = p2.Y * P;        
		p31.Y = p2.ZZ - p31.Y;      
		
	}
	else if (R.is_zero()) 
	{
		p2.X = p31.Y+p31.Y;      

		P = p2.X*p2.X;                
		R = p2.X*P;              
		p2.Y = p31.X * P;        
		p2.X = p31.X*p31.X;
		p2.X = p2.X + p2.X + p2.X;        
		p31.X = p2.X*p2.X;
		p31.X -= p2.Y;
		p31.X -= p2.Y;           
		p31.Y *= R;           
		p2.Y -= p31.X;
		p2.Y *= p2.X;               
		p31.Y = p2.Y - p31.Y;    
		p31.ZZ *= P;          
		p31.ZZZ *= R;
	} 
	else
		p31.inf();	
}

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

#define CHECK_ON 0

__global__
//__launch_bounds__(64, 12)
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
	
	//if (id == 0)
	//printf("[%d] %d %d %d %d\n", id, idxBucket, start, end, end - start);
	
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
			//printf("%d %d\n", idxBucket, idx);
			
			/*bool isNeg = idx < 0;
			if (isNeg) 
				idx = -idx;
			idx = idx - 1;

			fp_t X = points[idx].X;
			fp_t Y = points[idx].Y;
			if (isNeg)
				Y.cneg(true);*/
			
			fp_t X = points[idx].X;
			fp_t Y = points[idx].Y;
			madd_base(X, Y, result, resInf, k == 1 && !continueAdd);
		}
		
		buckets[idxBucket - 1] = result;
	}
	else
		buckets[bucketSize - 1] = result;
	
} 

__global__
//__launch_bounds__(128, 4)
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
	/*if (stride == 0)
		printf("%d %d %d %d\n", id, idxBucket, start, end);*/
	
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
			//printf("%d %d\n", idxBucket, idx);
			
			/*bool isNeg = idx < 0;
			if (isNeg) 
				idx = -idx;
			idx = idx - 1;
			
			//result.add(points[idx], isNeg);
			
			auto X = points[idx].X;
			auto Y = points[idx].Y;
			if (isNeg)
				Y.cneg(true);*/
			
			auto X = points[idx].X;
			auto Y = points[idx].Y;
			madd_base(X, Y, result, resInf, k == 1 && !continueAdd);
		}

		buckets[(idxBucket - 1) + lastMax * stride] = result;
	}
}

__global__
void pippenger_group_last_NAF(bucket_t *buckets, const bucket_t *bucket_tmp, const int lastMax, const int groups)
{
	int globalId = threadIdx.x  + blockIdx.x * blockDim.x;
	if (globalId >= lastMax - 1)
		return;

	bucket_t result;
	result.inf();
	
	for (int z = globalId, k = 0; k < groups; k++, z += lastMax)
		result.add(bucket_tmp[z]);
	
	buckets[globalId] = result;
}

__global__
//__launch_bounds__(64, 12)
void pippenger_group(const affine_t* points, bucket_t *buckets, 
				     const uint32_t* idx, const uint32_t* tableS, const uint32_t* tableE,
					 const uint32_t* numbers, int wnum, bool continueAdd)
{	
	const int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	const int id = globalId;	
	
	if (id >= (1 << WBITS))
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
			//result.add(points[numbers[z]]);
			
			const int idx = numbers[z];
			auto X = points[idx].X;
			auto Y = points[idx].Y;
			madd_base(X, Y, result, resInf, k == 1 && !continueAdd);
			
			//printf("%d %d, %d %d\n", idxBucket - 1, idx, continueAdd, k == 1 && !continueAdd);
		}

#if BUCKET_TEST
		buckets[threadIdx.x] = result;
#else
		buckets[idxBucket - 1] = result;
#endif
	}
} 

__global__
//__launch_bounds__(128, 4)
void pippenger_group_last(const affine_t* points, bucket_t *buckets, 
						  const uint32_t* idx, const uint32_t* tableS, const uint32_t* tableE,
						  const uint32_t* numbers, bool continueAdd)
{
	int globalId = threadIdx.x  + blockIdx.x * blockDim.x;
	if (globalId >= (1 << WBITS))
		return;
	
	const int lastMax = (1 << (NBITS % WBITS));
	const int id = globalId % lastMax;
	const int stride = globalId / lastMax;
	const int groups = (1 << WBITS) / lastMax;

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
			//result.add(points[numbers[z]]);
			
			const int pp = numbers[z];
			fp_t X2 = points[pp].X;
			fp_t Y2 = points[pp].Y;
			madd_base(X2, Y2, result, resInf, k == 1 && !continueAdd);
			
			//printf("%d %d, %d %d\n", idxBucket - 1, pp, continueAdd, k == 1 && !continueAdd);
		}
	}

	if (idxBucket > 0)
#if BUCKET_TEST
		buckets[threadIdx.x] = result;
#else		
		buckets[(idxBucket - 1) + lastMax * stride] = result;
#endif
}

__global__ 
//__launch_bounds__(32, 10)
void pippenger_final(bucket_t *buckets, bucket_t (*ret)[TILING][2], int shift)
{
    const int NTHRBITS = dlog2(TILING);	
	const uint32_t id = threadIdx.x + blockIdx.x * blockDim.x + shift;

	const uint32_t bucket_size = (1 << WBITS);

#if BUCKET_TEST	
	bucket_t* row = buckets;
#else
    bucket_t* row = buckets + (id / TILING) * bucket_size;
#endif	
    uint32_t i = (1<<(WBITS-NTHRBITS));

#if BUCKET_TEST	
	
#else
    row += (id % TILING) * i;
#endif

	__shared__ bucket_t scratch[32];	
	int tid = threadIdx.x;
	
	bucket_t acc = row[--i];
	scratch[tid] = acc;
	//bucket_t tmp = acc;

    while (i--) 
	{
#if BUCKET_TEST			
        bucket_t p = row[i % 1024];
#else
		bucket_t p = row[i];
#endif

		for (int pc = 0; pc < 2; pc++) 
		{
            acc.add(p);//add(acc, p);
			p = scratch[tid];
            scratch[tid] = acc;			
        }
        acc = p;
    }

	ret[id / TILING][id % TILING][0] = scratch[tid];
    ret[id / TILING][id % TILING][1] = acc;
}

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

__global__ void zero_bucket(bucket_t *buckets, int elemsInBucket, int sizeBucket, int total)
{
	const uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= total)
		return;
	
	int idBucket = id / elemsInBucket;
	int idInBucket = id % elemsInBucket;
	
	bucket_t* row = buckets + idBucket * sizeBucket;
	
	row[idInBucket].ZZ.zero();
	row[idInBucket].ZZZ.zero();
}

__global__ void initGroup_full(const scalar_t* scalars, unsigned int* idx, part_t* keys, int npoints)
{
	__shared__ int scalars_cache[8][256]; // 8 x 32 scalars
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= npoints)
		return;
	idx[id] = id;
	
	int warpLane = (id % blockDim.x) / 32;
	int warpId = (id % blockDim.x) % 32;
	
	int* source = (int*)(scalars + id - warpId);
	
	scalars_cache[warpLane][warpId] = source[warpId];
	scalars_cache[warpLane][32 + warpId] = source[32 + warpId];
	scalars_cache[warpLane][64 + warpId] = source[64 + warpId];
	scalars_cache[warpLane][96 + warpId] = source[96 + warpId];
	scalars_cache[warpLane][128 + warpId] = source[128 + warpId];
	scalars_cache[warpLane][160 + warpId] = source[160 + warpId];
	scalars_cache[warpLane][192 + warpId] = source[192 + warpId];
	scalars_cache[warpLane][224 + warpId] = source[224 + warpId];
	
	__syncwarp();
	
	scalar_t* inShared = (scalar_t*) scalars_cache[warpLane];
	
#pragma unroll	
	for (int group = 0; group < NWINS; ++group)
	{
		const int wbits = ((group * WBITS) > NBITS-WBITS) ? NBITS-(group * WBITS) : WBITS;
		uint32_t val = get_wval(inShared[warpId], group * WBITS, wbits);
		keys[id + npoints * group] = val;
	}
}

__global__ void initGroup_full2(const scalar_t* scalars, unsigned int* idx, part_t* keys, int npoints, int shift)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= npoints)
		return;

#pragma unroll	
	for (int group = 0; group < NWINS; ++group)
	{
		const int wbits = ((group * WBITS) > NBITS-WBITS) ? NBITS-(group * WBITS) : WBITS;
		uint32_t val = get_wval(scalars[id + shift], group * WBITS, wbits);
		keys[id + npoints * group] = val;
		idx[id + npoints * group] = id + shift;
	}
}

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
		//idx[id + npoints * group] = diff >= 0 ? (id + shift + 1) : -(id + shift + 1);
		idx[id + npoints * group] = diff >= 0 ? (id + shift) : (id + shift + npoints);
	}
#endif	
}

__global__ void initIdx(unsigned int* idx, int npoints)
{
	int id = threadIdx.x  + blockIdx.x * blockDim.x;
	if (id >= npoints)
		return;
	idx[id] = id;
}

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

__global__ void initRows(uint32_t* rows, uint32_t *sizes, const uint32_t* tableS, const uint32_t* tableE, int N)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= N)
		return;
	
	rows[id] = id;
	sizes[id] = tableE[id] - tableS[id];
}

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

__global__ void copyIdxs(const unsigned int* idxIn, unsigned int* idxOut, int npoints)
{
	int id = threadIdx.x  + blockIdx.x * blockDim.x;
	if (id >= npoints)
		return;
	
	idxOut[id] = idxIn[id];
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

#if WBITS==16
static int get_wval(const unsigned int* d, uint32_t off, uint32_t bits)
{
    uint32_t ret = d[off/32];
    return (ret >> (off%32)) & ((1<<bits) - 1);
}
#else
static int get_wval(const unsigned int* d, uint32_t off, uint32_t bits)
{
    uint32_t top = off + bits - 1;
    uint64_t ret = ((uint64_t)d[top/32] << 32) | d[off/32];

    return (int)(ret >> (off%32)) & ((1<<bits) - 1);
}
#endif


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
    CUDA_OK(cudaLaunchCooperativeKernel((const void*)f, gridDim, blockDim, va_args, 0, 0));
}

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

template<class bucket_t> class result_t 
{
    bucket_t ret[NWINS][NTHREADS][2];
public:
    result_t() { }
    inline operator decltype(ret)&() { return ret; }
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
public:
    typedef vector<result_t<bucket_t>, host_pinned_allocator_t<result_t<bucket_t>>> result_container_t;

private:
    size_t sm_count;
    bool init_done = false;
    device_ptr_list_t<affine_t> d_base_ptrs;
    device_ptr_list_t<scalar_t> d_scalar_ptrs;
    device_ptr_list_t<bucket_t> d_bucket_ptrs;

	// only one pointer per each variable, TODO: rewrite this!
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
        size_t N;
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
#if REF
        config.N = 1;
#else
		config.N = (sm_count*256) / (NTHREADS*NWINS);
#endif	
        size_t delta = ((npoints+(config.N)-1)/(config.N)+WARP_SZ-1) & (0U-WARP_SZ);
        config.N = (npoints+delta-1) / delta;
		
		config.numAsync = 2;
		config.thF = 32;
		
		if (NAF)
		{
			if (SIGNED_SCALAR)
			{
				config.naf_coef = 0.585;//2.0 / 3.0;
				config.naf_elems = (int)(TILING * config.naf_coef);
			}
			else
			{
				config.naf_coef = 2.0 / 3.0;
				config.naf_elems = (int)(TILING * config.naf_coef) + 1;
			}
			
#if PRINT			
			printf("NAF elems %d\n", config.naf_elems);
#endif			
		}
		else
			config.naf_elems = TILING;
		
#if NAF		
		CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor (&config.numBlocks, pippenger_final_NAF, config.thF, 0));
#else		
		CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor (&config.numBlocks, pippenger_final, config.thF, 0));
#endif

		int totalThreads = config.numBlocks * config.thF * sm_count;
#if CPU_CHECK		
		config.oneWaveBuckets = NWINS;
#else		
		if (SIGNED_SCALAR)
			config.oneWaveBuckets = std::max(NAF ? (int)(NWINS - 1) : (int)NWINS, (int)totalThreads / (TILING / 2));
		else
			config.oneWaveBuckets = std::max(NAF ? (int)(NWINS - 1) : (int)NWINS, (int)totalThreads / config.naf_elems);
#endif	
		
		for (int z = 0; z < config.numAsync + 1; ++z)
			streams.push_back(0);
		for (int z = 0; z < NWINS; ++z)
			copy_streams.push_back(0);
#if 1
		for (int z = 0; z < config.numAsync + 1; ++z)
			CUDA_OK(cudaStreamCreate(&streams[z]));
		
		for (int z = 0; z < NWINS; ++z)
			CUDA_OK(cudaStreamCreate(&copy_streams[z]));
#endif
		clearCache();
        return config;
    }

    size_t get_size_bases(MSMConfig& config)   { return config.n * sizeof(affine_t) * 2; }
    size_t get_size_scalars(MSMConfig& config) { return config.n * sizeof(scalar_t); }
    size_t get_size_buckets(MSMConfig& config) { return config.N * sizeof(bucket_t) * NWINS * (1 << WBITS); }

    result_container_t get_result_container(MSMConfig& config) 
	{
        result_container_t res(config.N);
        return res;
    }

    // Allocate storage for bases on device. Throws cuda_error on error.
    // Returns index of the allocated base storage.
    size_t allocate_d_bases(MSMConfig& config) { return d_base_ptrs.allocate(get_size_bases(config)); }

    // Allocate storage for scalars on device. Throws cuda_error on error.
    // Returns index of the allocated base storage.
    size_t allocate_d_scalars(MSMConfig& config) { return d_scalar_ptrs.allocate(get_size_scalars(config)); }

    // Allocate storage for buckets on device. Throws cuda_error on error.
    // Returns index of the allocated base storage.
    size_t allocate_d_buckets(MSMConfig& config) { return d_bucket_ptrs.allocate(get_size_buckets(config)); }
	
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
#if BUCKET_TEST == 0
#if NAF
		d_buckets_ptrs.allocate(sizeof(bucket_t) * (config.naf_elems * (tN / TILING)) * config.oneWaveBuckets);
		d_buckets_ptrs.allocate(sizeof(bucket_t) * (config.naf_elems * (tN / TILING)) * 2);
		
		CUDA_OK(cudaMemset(d_buckets_ptrs[0], 0, sizeof(bucket_t) * (config.naf_elems * (tN / TILING)) * config.oneWaveBuckets));
		CUDA_OK(cudaMemset(d_buckets_ptrs[1], 0, sizeof(bucket_t) * (config.naf_elems * (tN / TILING)) * 2));
		
		cpuBucket_ptrs.allocate(sizeof(bucket_t) * (config.naf_elems * (tN / TILING)) * 2, true);		
		memset(cpuBucket_ptrs[0], 0, sizeof(bucket_t) * (config.naf_elems * (tN / TILING)) * 2);
		copied = 0;
#else
		d_buckets_ptrs.allocate(sizeof(bucket_t) * tN * config.oneWaveBuckets);
		CUDA_OK(cudaMemset(d_buckets_ptrs[0], 0, sizeof(bucket_t) * tN * config.oneWaveBuckets));
#endif	
#else
		d_buckets_ptrs.allocate(sizeof(bucket_t) * NWINS * 1024);
#endif	
		d_none_ptrs.allocate(sizeof(bucket_t) * NWINS * TILING * 2 * batches);
		
		c_none_ptrs.allocate(sizeof(bucket_t) * (NWINS * TILING * 2 * batches + TILING_2 * 2 * 2), true);
		c_none_ptrs.allocate(sizeof(bucket_t) * NWINS * TILING * 2 * batches, true);
		//cpuBucket_ptrs.allocate(sizeof(bucket_t) * 5 * tN, true);

		size_t free, total;
		cudaMemGetInfo (&free, &total);
		printf("total mem %lld, free mem %lld\n", total, free);
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

		/*if ((config.npoints % 32) != 0 || config.npoints / 32 == 0)
			;//throw -1;
		else
		{
			if (pointNotNull.size() == 0)
				pointNotNull.allocate(sizeof(unsigned) * (config.npoints / 32));
			fill_not_null<<<(config.npoints / 32) / 256 + 1, 256, 0, stream>>>(d_points, pointNotNull[0], config.npoints / 32);			
			CUDA_OK(cudaStreamSynchronize(stream));
		}*/
    }

    // Transfer scalars to device. Throws cuda_error on error.
    void transfer_scalars_to_device(size_t d_scalars_idx, const scalar_t scalars[], size_t size, cudaStream_t s = nullptr) 
	{
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        scalar_t *d_scalars = d_scalar_ptrs[d_scalars_idx];
        CUDA_OK(cudaSetDevice(device));
        CUDA_OK(cudaMemcpy(d_scalars, scalars, size * sizeof(*d_scalars), cudaMemcpyHostToDevice));
    }
	
	static void copy(scalar_t* to, const scalar_t* from, size_t size)
	{
#if PROF		
		double T = omp_get_wtime();
#endif		
		memcpy(to, from, size);
/*#pragma omp parallel for
		for (int z = 0; z < size / sizeof(scalar_t); ++z)
			to[z] = from[z];*/
#if PROF		
		T = omp_get_wtime() - T;
		printf("  --- copy cpu-cpu %f ms, %f gb/s\n", T * 1000., size * (1.0 / T) / 1024. / 1024. / 1024.);
#endif
	}
	
    // Transfer scalars to device. Throws cuda_error on error.
    void transfer_scalars_to_device_first(MSMConfig& config, size_t d_scalars_idx, scalar_t *h_scalars, const scalar_t* scalars) 
	{
		CUDA_OK(cudaSetDevice(device));

        scalar_t *d_scalars = d_scalar_ptrs[d_scalars_idx];
		size_t total = config.npoints * sizeof(*d_scalars);
		size_t shift = config.npoints / 2;
		
		/*{
		for (int k = 0; k < 5; ++k)
		for (int z = 5; z >= 0; --z)
		{
#if PROF		
		double T = omp_get_wtime();
#endif
		
		CUDA_OK(cudaMemcpy(d_scalars, scalars, total / (1 << z), cudaMemcpyHostToDevice));
#if PROF		
		T = omp_get_wtime() - T;
        
		printf("[%d]  --- copy cpu-gpu #1 %f ms, %f gb/s\n", T * 1000., z, (total / (1 << z)) * (1.0 / T) / 1024. / 1024. / 1024.);
#endif
        }
		}
		exit(0);*/
		
		scalar_t* ptr1 = h_scalars;
		const scalar_t* ptr2 = scalars + shift;
		thread job_copy(copy, ptr1, ptr2, total / 2);
		//memcpy(h_scalars, scalars + shift, total / 2);
		
#if PROF		
		double T = omp_get_wtime();
#endif
		CUDA_OK(cudaMemcpy(d_scalars, scalars, total / 2, cudaMemcpyHostToDevice));		
#if PROF		
		T = omp_get_wtime() - T;
        printf("  --- copy cpu-gpu #1 %f ms, %f gb/s\n", T * 1000., total/2 * (1.0 / T) / 1024. / 1024. / 1024.);

#endif
		
		job_copy.join();
#if PROF
		T = omp_get_wtime();
#endif		
		CUDA_OK(cudaMemcpy(d_scalars + shift, h_scalars, total / 2, cudaMemcpyHostToDevice));
#if PROF		
		T = omp_get_wtime() - T;
        printf("  --- copy gpu-gpu #2 %f ms, %f gb/s\n", T * 1000., total/2 * (1.0 / T) / 1024. / 1024. / 1024.);
#endif
    }	
	
    // Transfer buckets from device. Throws cuda_error on error.
    void transfer_buckets_to_host(MSMConfig& config, result_container_t &res, size_t d_buckets_idx, cudaStream_t s = nullptr) 
	{
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        bucket_t *d_buckets = d_bucket_ptrs[d_buckets_idx];
        CUDA_OK(cudaSetDevice(device));
        CUDA_OK(cudaMemcpyAsync(res[0], d_buckets, config.N*sizeof(res[0]), cudaMemcpyDeviceToHost, stream));
    }

    void synchronize_stream() 
	{
        //CUDA_OK(cudaSetDevice(device));
        CUDA_OK(cudaStreamSynchronize(default_stream));
    }

#if REF
	bucket_t (*cmp_buckets)[NWINS][1<<WBITS] = nullptr;
#endif
	
	void printCMP(int* elem1, int* elem2, int num)
	{
		int* t = elem1;
		for (int z = 0; z < num; ++z)
		{
			printf("%u ", t[z]);
			if (((z+1) % 12) == 0)
				printf("\n");
		}
		printf("\n\n");

		if (elem2 == NULL)
			return;

		t = elem2;
		for (int z = 0; z < num; ++z)
		{
			printf("%u ", t[z]);
			if (((z+1) % 12) == 0)
				printf("\n");
		}
		printf("\n");
	}

    // Perform accumulation into buckets on GPU. Throws cuda_error on error.
    void launch_kernel(MSMConfig& config, size_t d_bases_idx, size_t d_scalars_idx, size_t d_buckets_idx, bool mont = true, cudaStream_t s = nullptr)
    {
        assert(WBITS > NTHRBITS);
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        affine_t *d_points = d_base_ptrs[d_bases_idx];
        scalar_t *d_scalars = d_scalar_ptrs[d_scalars_idx];
        bucket_t (*d_buckets)[NWINS][1<<WBITS] = reinterpret_cast<decltype(d_buckets)>(d_bucket_ptrs[d_buckets_idx]);

		bucket_t (*d_none)[NWINS][NTHREADS][2] = nullptr;
#if REF
		//CUDA_OK(cudaMemset((bucket_t*)d_buckets, 0, sizeof(bucket_t) * (1<<WBITS) * NWINS));
		//CUDA_OK(cudaMalloc(&d_none, sizeof(d_none[0])));
#endif
        CUDA_OK(cudaSetDevice(device));
        launch_coop(pippenger, dim3(NWINS, config.N), NTHREADS, stream, (const affine_t*)d_points, config.npoints, (const scalar_t*)d_scalars, mont, d_buckets, d_none);
		
#if REF
		bucket_t (*c_buckets)[NWINS][(1<<WBITS)] = nullptr;
		CUDA_OK(cudaMallocHost(&c_buckets, sizeof(c_buckets[0])));		
		CUDA_OK(cudaMemcpy(c_buckets, d_buckets, sizeof(bucket_t) * NWINS * (1 << WBITS), cudaMemcpyDeviceToHost));
			
		printf("check start\n");		
		for (int z = 0; z < NWINS; ++z)
		{
			int count = 0;
#pragma omp parallel for reduction(+: count)			
			for (int k = 0; k < (1 << WBITS); ++k)
			{
				if (!cmp_buckets[0][z][k].cmp(c_buckets[0][z][k]))
				{
					if (z == 1)
					{
						printf(" NOT at %d win, %d position\n", z, k);
						//printCMP((int*)&cmp_buckets[0][z][k], (int*)&c_buckets[0][z][k], 12*4);
					}
					//break;					
					count++;
				}			
			}
			if (count)
				printf("NE at %d win, %d positions\n", z + 1, count);
		}
		printf("check done\n");
		CUDA_OK(cudaFreeHost(c_buckets));
#endif		
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

	static point_t integrate_row_host_t(const bucket_t row[TILING][2], const int TILES, int wbits = WBITS)
	{
		const int NTHRBITS = log2(TILES);
		size_t i = TILES-1;
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
		while (i--) 
		{
			point_t raise = acc;
			for (size_t j = 0; j < WBITS-NTHRBITS; j++)
				raise.dbl();
			res.add(raise);
			res.add(row[i][0]);
			if (i & mask) 
				acc.add(row[i][1]);
			else 
			{
				ret.add(res);
				if (i-- == 0)
					break;
				res = row[i][0];
				acc = row[i][1];
			}
		}
		return ret;
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

	void finalyze(int numStrs, int thF, int idx)
	{
		bucket_t* d_buckets = d_buckets_ptrs[0];
		bucket_t (*d_none)[TILING][2] = reinterpret_cast<decltype(d_none)>(d_none_ptrs[0]);
		bucket_t (*c_none)[TILING][2] = reinterpret_cast<decltype(c_none)>(c_none_ptrs[idx]);
		
		for (int z = 0; z < numStrs; ++z)
			CUDA_OK(cudaStreamSynchronize(streams[z]));
#if BUCKET_TEST == 0		
		pippenger_final<<<(TILING * (NWINS - numCPUAsync[idx])) / thF, thF, 0, streams[0]>>>(d_buckets, d_none, 0);
#endif	
		CUDA_OK(cudaMemcpy(c_none, d_none, (NWINS - numCPUAsync[idx]) * sizeof(d_none[0]), cudaMemcpyDeviceToHost));
	}
	
	void longAdd(unsigned* t3, const unsigned* t1, const unsigned* t2, int N) // t3 = t1 + t2
	{
		uint64_t bitNext = 0;
		for (int z = 0; z < N; ++z)
		{
			uint64_t val = (uint64_t)t1[z] + (uint64_t)t2[z] + bitNext;
			t3[z] = (int)val;
			bitNext = (val >> 32) ? 1 : 0;
		}
	}

	void longAdd(unsigned* t3, const unsigned* t1, const unsigned t2, int N) // t3 = t1 + t2
	{
		uint64_t bitNext = 0;
		uint64_t val = (uint64_t)t1[0] + (uint64_t)t2;
		t3[0] = (int)val;
		bitNext = (val >> 32) ? 1 : 0;

		for (int z = 1; z < N && bitNext; ++z)
		{
			uint64_t val = (uint64_t)t1[z] + bitNext;
			t3[z] = (int)val;
			bitNext = (val >> 32) ? 1 : 0;
		}
	}

	void longSub(unsigned* t3, const unsigned* t1, const unsigned t2, int N) // t3 = t1 - t2
	{
		uint64_t bitNext = 0;
		uint64_t val = (uint64_t)t1[0] - (uint64_t)t2;
		t3[0] = (int)val;
		bitNext = (val >> 32) ? 1 : 0;

		for (int z = 1; z < N && bitNext; ++z)
		{
			uint64_t val = (uint64_t)t1[z] - bitNext;
			t3[z] = (int)val;
			bitNext = (val >> 32) ? 1 : 0;
		}
	}

	void longShift(unsigned* t2, const unsigned* t1, int N) // t2 = t1 >> 1;
	{
		t2[0] = t1[0] >> 1;
		for (int z = 1; z < N; ++z)
		{
			if (t1[z] & 1)
				t2[z - 1] |= ((unsigned)1 << 31);
			t2[z] = t1[z] >> 1;
		}
	}

	void longShift(unsigned* t2, const unsigned* t1, int N, int shift) // t2 = t1 >> shift;
	{
		if (shift < 32)
		{
			t2[0] = t1[0] >> shift;
			for (int z = 1; z < N; ++z)
			{
				int odd = t1[z] & ((1 << shift) - 1);
				t2[z - 1] |= ((unsigned)odd << (32 - shift));
				t2[z] = t1[z] >> shift;
			}
		}
		else
		{
			for (int z = 0; z < N - 1; ++z)
				t2[z] = t1[z + 1];
		}
	}

	void longXor(unsigned* t3, const unsigned* t1, const unsigned* t2, int N) // t3 = t2 ^ t1
	{
		for (int z = 0; z < N; ++z)
			t3[z] = t2[z] ^ t1[z];
	}

	void longAnd(unsigned* t3, const unsigned* t1, const unsigned* t2, int N) // t3 = t2 & t1
	{
		for (int z = 0; z < N; ++z)
			t3[z] = t2[z] & t1[z];
	}

	void longOr(unsigned* t3, const unsigned* t1, const unsigned* t2, int N) // t3 = t2 | t1
	{
		for (int z = 0; z < N; ++z)
			t3[z] = t2[z] | t1[z];
	}
	
	bool notZero(unsigned *t, int N) 
	{
		for (int z = 0; z < N; ++z)
			if (t[z])
				return true;
		return false;
	}

	static void madd_base(const fp_t& X2, const fp_t& Y2, bucket_t& p31, int& resInf, bool isSecond = false)
	{
		if (X2.is_zero())
			if (Y2.is_zero())
				return;
		
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

	vector<thread*> asyncIntegrate;
	vector<thread*> asyncLastPart;
	
	void sumPartLastBucketAsync(int elems, int numBucket, int numIter)
	{
#if PRINT_ASYNC		
		double ompT = omp_get_wtime();
#endif	
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
		
#if PRINT_ASYNC		
		ompT = omp_get_wtime() - ompT;
		printf("  -->job sum part last finished in %f ms\n", ompT * 1000.);
#endif
	}
	
	void sumLastBucketAsync(int elemsInBucket, int groups, int numIter, bool continueAdd)
	{
#if PRINT_ASYNC		
		double ompT = omp_get_wtime();
#endif		
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
		//integrated_cpu[numIter] = res;
		if (continueAdd)
		{
			point_t tmp = res;
			integrated_batched[(NWINS - 1) + numIter * NWINS].add(tmp);
		}
		else
			integrated_batched[(NWINS - 1) + numIter * NWINS] = res;
#if PRINT_ASYNC		
		ompT = omp_get_wtime() - ompT;
		printf("  -->job sum last finished in %f ms\n", ompT * 1000.);
#endif
	}
	
	void doFinalSum(int k, int numStrs, int& bucketsDone, int& allDone, MSMConfig& config, int bucketSize, bool isLast)
	{
		copied = 0;
		bucket_t* d_buckets = d_buckets_ptrs[0];
		bucket_t (*d_none)[TILING][2] = reinterpret_cast<decltype(d_none)>(d_none_ptrs[0]);
		
		cudaStream_t currS = streams[0];
		
		for (int z = 0; z < numStrs; ++z)
			CUDA_OK(cudaStreamSynchronize(streams[z]));				
#if PROF
		CUDA_OK(cudaEventRecord (events[4]));
#endif

#if FROM_CPU && CPU_CHECK
		CUDA_OK(cudaMemcpy(d_buckets, cpuRes, sizeof(bucket_t) * NWINS * bucketSize, cudaMemcpyHostToDevice));
#endif

#if NAF
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
#else
		int blocks = (TILING * bucketsDone) / config.thF;
		pippenger_final<<<blocks, config.thF, 0, currS>>>(d_buckets, d_none, 0);
#endif

		CUDA_OK(cudaStreamSynchronize(currS));
		//CUDA_OK(cudaMemcpyAsync(c_none_ptrs[0] + TILING * 2 * allDone, d_none, bucketsDone * sizeof(d_none[0]), cudaMemcpyDeviceToHost, streams[0]));

#if CPU_CHECK
		CUDA_OK(cudaMemcpy(cpuBucket, d_buckets, sizeof(bucket_t) * NWINS * bucketSize, cudaMemcpyDeviceToHost));
#endif

#if NAF == 0
		if (!isLast)
			CUDA_OK(cudaMemsetAsync(d_buckets, 0, sizeof(bucket_t) * bucketSize * config.oneWaveBuckets, streams[1]));
		CUDA_OK(cudaStreamSynchronize(streams[1]));
#endif
		//CUDA_OK(cudaStreamSynchronize(streams[0]));		
		
		/*int err = 0;
		for (int z = 0; z < bucketsDone * TILING * 2; ++z)
		{
			if (!c_none_ptrs[1][TILING * 2 * allDone + z].cmp(c_none_ptrs[0][TILING * 2 * allDone + z]))
			{
				printf(" ne at %d\n", z);
				err++;
			}
		}
		printf("SUMS not eq %d / %d\n", err, TILING * 2 * bucketsDone);*/
		
#if PROF
		CUDA_OK(cudaEventRecord (events[5]));
		CUDA_OK(cudaEventSynchronize ( events[5] ));
		{
			float time = 0;
			CUDA_OK(cudaEventElapsedTime (&time, events[4], events[5]));
			if (isLast)
				printf(" ** [X] final part %f ms, bDone %d, allDone %d\n", time, bucketsDone, allDone);
			else
				printf(" ** [%d] final part %f ms, bDone %d, allDone %d\n", k, time, bucketsDone, allDone);
		}
#endif	

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
	
	static void moveNextScalar(scalar_t* h_scalars, const scalar_t* scalars, scalar_t *d_scalars, size_t npoints, cudaStream_t& st)
	{
#if PRINT_ASYNC		
		double ompT = omp_get_wtime();
#endif		
		//memcpy(h_scalars, scalars, npoints * sizeof(scalar_t));
#pragma omp parallel for num_threads(2)
		for (int z = 0; z < npoints; ++z)
			h_scalars[z] = scalars[z];
		
		CUDA_OK(cudaMemcpyAsync(d_scalars, h_scalars, npoints * sizeof(scalar_t), cudaMemcpyHostToDevice, st));
		CUDA_OK(cudaStreamSynchronize(st));
#if PRINT_ASYNC		
		ompT = omp_get_wtime() - ompT;
		printf("  -->job move next finished in %f ms\n", ompT * 1000.);
#endif
	}
		
	static void moveNext(scalar_t *d_scalars, 
						 const scalar_t* scalars, size_t shift, size_t points,
						 const scalar_t *scalars_prev, size_t shift_prev, size_t points_prev, 
						 cudaStream_t& st)
	{
#if PRINT_ASYNC		
		double ompT = omp_get_wtime();
#endif	
		CUDA_OK(cudaMemcpyAsync(d_scalars + shift, scalars, points * sizeof(scalar_t), cudaMemcpyHostToDevice, st));
	
		if (scalars_prev)
			CUDA_OK(cudaMemcpyAsync(d_scalars + shift_prev, scalars_prev, points_prev * sizeof(scalar_t), cudaMemcpyHostToDevice, st));
		
		CUDA_OK(cudaStreamSynchronize(st));
#if PRINT_ASYNC		
		ompT = omp_get_wtime() - ompT;
		printf("  -->job move next finished in %f ms\n", ompT * 1000.);
#endif
	}
	
	void addAffine(affine_t& t1, const fp_t& X, const fp_t& Y, const fp_t& d, bool doubling)
	{
		fp_t x2, L;
		x2 = X + t1.X;
		if (doubling)
		{
			L = X * X;
			L = L + L + L;
		}
		else
			L = Y - t1.Y;
		
		L *= d;			
		t1.X = L * L;
		t1.X -= x2;
		
		t1.Y = X - t1.X;
		t1.Y *= L;
		t1.Y -= Y;
	}
	
	//bucket_t* cpuRes = NULL;
	void launch_best(int numIter, int& bucketsDone, int& allDone, point_t* out, 
					 MSMConfig& config, size_t d_bases_idx, size_t d_scalars_idx, size_t d_scalars_xfer, 
					 scalar_t* h_scalars, const scalar_t* scalars, const scalar_t* scalars_prev, bool isFull, 
					 int shiftPartly, size_t sizePartly, bool isLastIfNotFull = false)
	{
#if 0
		{		
			affine_t *d_points = d_base_ptrs[d_bases_idx];
			scalar_t *d_scalars = d_scalar_ptrs[d_scalars_idx];
		
			const size_t npoints = config.npoints;
			unsigned* hSc = new unsigned[npoints * 8];
			fp_t* hPointsX, *hPointsY;
			CUDA_OK(cudaMallocHost(&hPointsX, sizeof(fp_t) * npoints));
			CUDA_OK(cudaMallocHost(&hPointsY, sizeof(fp_t) * npoints));
			
			copyToHost<<<npoints / 256 + 1, 256>>>(d_points, hPointsX, hPointsY, npoints);
			CUDA_OK(cudaMemcpy(hSc, d_scalars, sizeof(unsigned) * 8 * npoints, cudaMemcpyDeviceToHost));

			int tN = 1 << WBITS;
			bucket_t* cpuRes = new bucket_t[NWINS * tN];
			fp_t *points = new fp_t[2 * npoints];
			
			for (int z = 0; z < npoints; ++z)
			{
				points[2 * z] = hPointsX[z];
				points[2 * z + 1] = hPointsY[z];
			}
			
			double t = omp_get_wtime();
			for (int z = 0; z < npoints; ++z)
			{
				points[2 * z + 1] = 1 / points[2 * z];
			}
			t = omp_get_wtime() - t;
			printf("%f\n", t);
			exit(0);
			
			printf("copy points done\n");
			memset(cpuRes, 0, sizeof(bucket_t) * NWINS * tN);
			
			printf("memset buckets done\n");
			vector<vector<vector<int>>> table(NWINS);
			
			for (int k = 0; k < NWINS; ++k)
				table[k] = vector<vector<int>>(tN);
			
			printf("tables allocate done\n");
			for (int z = 0; z < npoints; ++z)
			{
				unsigned* s = hSc + z * 8;
				for (int k = 0; k < NWINS; ++k)
				{
					const int wbits = ((k * WBITS) > NBITS - WBITS) ? NBITS - (k * WBITS) : WBITS;
					int valP = get_wval(s, k * WBITS, wbits);
					if (valP)
						table[k][valP].push_back(z);
				}
			}
			
			printf("distribute done\n");
			for (int k = 0; k < NWINS; ++k)
				std::sort(table[k].begin(), table[k].end(), [](vector<int>& a, vector<int>& b) { return a.size() > b.size(); });
			
			printf("sort  done\n");
			
			double T = omp_get_wtime();
			int addCount = 0;
			for (int k = 0; k < NWINS; ++k)
			{
				bucket_t* currB = cpuRes + tN * k;
				for (int p = 0; p < table[k].size(); ++p)
				{
					bucket_t tmp;
					tmp.inf();
					
					for (int z = 0; z < table[k][p].size(); ++z)
					{
						int idx = table[k][p][z];
						
						affine_t point;
						point.X = points[2 * idx];
						point.Y = points[2 * idx + 1];

						if (point.X.is_zero() && point.Y.is_zero())	
						{
							point.inf[0] = 1;
							continue;
						}
						else
							point.inf[0] = 0;
						tmp.add(point);
						addCount++;
					}
					currB[p] = tmp;
				}
			}
			T = omp_get_wtime() - T;
			printf("done with time %f sec, addC %d\n", T, addCount);
			
			affine_t* res1 = new affine_t[NWINS * tN];
			affine_t* res2 = new affine_t[NWINS * tN];
			memset(res1, 0, sizeof(affine_t) * NWINS * tN);
			memset(res2, 0, sizeof(affine_t) * NWINS * tN);

			T = omp_get_wtime();
			addCount = 0;
			int assignCount = 0;
			int doubleC = 0;
			for (int k = NWINS + 1; k < NWINS; ++k)
			{
				affine_t* currB = res1 + tN * k;
				
				for (int p = 0; p < table[k].size(); ++p)
				{
					affine_t tmp;
					tmp.X.zero();
					tmp.Y.zero();
					bool tmpInf = true;
					
					bucket_t tmpB;
					tmpB.inf();
					for (int z = 0; z < table[k][p].size(); ++z)
					{
						int idx = table[k][p][z];
						
						affine_t point;
						point.X = points[2 * idx];
						point.Y = points[2 * idx + 1];

						bool inf = false;
						if (point.X.is_zero() && point.Y.is_zero())			
							inf = true;
						
						if (inf)
							continue;
						
						if (tmpInf)
						{
							tmpInf = false;
							tmp.X = point.X;
							tmp.Y = point.Y;
							assignCount++;
							tmpB.add(point);
						}
						else
						{							
							fp_t d = point.X - tmp.X;
							bool doubling = false;
							if (d.is_zero())
							{
								//printf("need double\n");
								doubleC++;
								doubling = true;
								d = point.Y + point.Y;								
							}
							d = 1 / d;
							
							addAffine(tmp, point.X, point.Y, d, doubling);
							addCount++;
							tmpB.add(point);
						}
						
						/*if (!tmpB.cmp(tmp.X, tmp.Y))
						{
							printf("NE inf %d, tmpB inf %d, point num %d\n", tmpInf, tmpB.is_inf(), z);
							exit(-1);
						}*/
					}
					currB[p] = tmp;
				}
			}
			T = omp_get_wtime() - T;
			printf("done with time %f sec, addC = %d, assC %d, total %d\n", T, addCount, assignCount, assignCount + addCount);
			 
			int err = 0;
#pragma omp parallel for reduction(+: err)
			for (int z = 0; z < NWINS * tN; ++z)
			{
				if (!cpuRes[z].cmp(res1[z].X, res1[z].Y))
					err++;
			}
			printf("err1 cmp %d / %d, diff %d, doubleC = %d\n", err, NWINS * tN,  NWINS * tN - err, doubleC);
			
			T = omp_get_wtime();
			addCount = 0;
			assignCount = 0;
			for (int k = 0; k < NWINS; ++k)
			{
				affine_t* currB = res2 + tN * k;
								
				//bucket_t tmpB;
				//tmpB.inf();
#define SIZE 128	
#define CHECK_ON 0
			
				int sizes[SIZE];
				int currS[SIZE];
				int lastGot[SIZE];
				
				affine_t tmpS[SIZE];
				//bool tmpInfS[SIZE];
				unsigned char dd[SIZE];
				
				fp_t X[SIZE];
				fp_t nextP_diff[SIZE];
				fp_t prod[SIZE];
				
				int SS = SIZE;
				for (int p = 0; p < table[k].size(); p += SS)
				{
					//printf("p=%d\n", p);
					for (int t = 0; t < SS; ++t)
					{
						sizes[t] = table[k][p + t].size();
						//printf("%d => %d\n", p + t, sizes[t]);

						currS[t] = 0;
						dd[t] = 3;						
						tmpS[t].X.zero();
						tmpS[t].Y.zero();
					}
					
					//printf(" init done\n");
					int gotCount = 1;					
					while (gotCount)
					{
						gotCount = 0;
						int lastGotIdx = -1;
						fp_t partlyProd;
						for (int t = 0; t < SS; ++t)
						{
							bool got = false;
							while (!got)
							{
								if (currS[t] >= sizes[t])
								{
									dd[t] = 2;
									//nextP_diff[t].zero();
									break;
								}

								int idx = table[k][p + t][currS[t]];
								currS[t]++;

								X[t] = points[2 * idx]; //X
								
								if (X[t].is_zero() && points[2 * idx + 1].is_zero())			
									continue;
								
								if (dd[t] == 3)
								{
									dd[t] = 4;
									tmpS[t].X = X[t];
									tmpS[t].Y = points[2 * idx + 1];
									assignCount++;
									continue;
								}
								
								nextP_diff[t] = X[t] - tmpS[t].X;
								if (nextP_diff[t].is_zero())
								{
									fp_t Y = points[2 * idx + 1];
									nextP_diff[t] = Y + Y;
									dd[t] = 1;
#if CHECK_ON									
									printf("need double\n");
#endif
								}
								else
									dd[t] = 0;
								
								got = true;
								gotCount++;
							}
							
							if (got)
							{
								if (gotCount == 1)
									partlyProd = nextP_diff[t];
								else if (gotCount > 1)
									partlyProd *= nextP_diff[t];
								
								if (gotCount >= 1)
								{								
									prod[t] = partlyProd;
									lastGot[t] = lastGotIdx;
									//printf(" add to prod %d\n", t);
								}
								lastGotIdx = t;
							}
						}
						
						//printf(" got done %d\n", gotCount);
						if (gotCount == 0)
						{
							for (int t = 0; t < SS; ++t)
								currB[p + t] = tmpS[t];
							continue;
						}
						/*for (int t = 0; t < SS; ++t)
							printf("%d => %d, %d\n", p + t, sizes[t], currS[t]);*/
#if CHECK_ON															
						fp_t checkProd;					
						for (int t = 0, k = 0; t < SS; ++t)
						{
							if (currS[t] >= sizes[t])
								continue;
							
							if (k == 0)
								checkProd = nextP_diff[t];
							else
								checkProd *= nextP_diff[t];
							++k;
							if (checkProd != prod[t])
								printf("prod NE %d\n", t);
						}
#endif
								
						fp_t inv = 1 / partlyProd;//prod[lastGotIdx];
						for (int t = SS - 1; t >= 0; --t)
						{
							if (dd[t] == 2)
								continue;
							/*if (nextP_diff[t].is_zero())
								continue;*/

							addCount++;
							int idx = table[k][p + t][currS[t] - 1];
							
							fp_t Y = points[2 * idx + 1];							
							
							fp_t d;
							
							bool doubling = dd[t] == 1;//false;
							/*fp_t diff = X[t] - tmpS[t].X;
							if (diff.is_zero())
								doubling = true;*/
							
							if (gotCount > 1)
							{
								d = inv * prod[lastGot[t]]; //prod[t - 1];
								inv *= nextP_diff[t];
							}
							else
								d = inv;
#if CHECK_ON																
							fp_t XX = points[2 * idx];
							fp_t check;
							if (doubling)
								check = Y + Y;
							else
								check = XX - tmpS[t].X;
							
							if (check != nextP_diff[t])
								printf("diff NE %d\n", t);
							
							check = 1 / check;
							if (check != d)
								printf("inv NE %d\n", t);
#endif
							addAffine(tmpS[t], X[t], Y, d, doubling);
							gotCount--;
						}
						gotCount = 1; // next iteration
					}
					
					/*if (p > 12)
						exit(0);*/
				}
			}
			T = omp_get_wtime() - T;
			printf("done with time %f sec, addC = %d, assC = %d, total %d\n", T, addCount, assignCount, addCount + assignCount);

			
			err = 0;
#pragma omp parallel for reduction(+: err)			
			for (int z = 0; z < NWINS * tN; ++z)
			{
				if (!cpuRes[z].cmp(res2[z].X, res2[z].Y))
					err++;
			}
			printf("err cmp %d / %d, diff %d\n", err, NWINS * tN,  NWINS * tN - err);
			exit(0);
		}
#endif		
#if 0		
		{
			unsigned* s = (unsigned*)(&scalars[0]);
									
			unsigned tt = 0;
			for (int p = 0; p < 8; ++p)
			{				
  			    unsigned res = 0;
				unsigned n = s[p];
				while (n) {
				  res ++;
				  n &= n-1;  // Забираем младшую единичку.
				}				
				tt += res;				
			}	
			printf("tt = %d\n", tt);
			
			fp_t XY[4];
			
			CUDA_OK(cudaMemcpy(XY, d_base_ptrs[d_bases_idx], sizeof(fp_t) * 4, cudaMemcpyDeviceToHost));

			bucket_t tmp, res;
			tmp.X = XY[0];
			tmp.Y = XY[1];
			tmp.ZZ = tmp.ZZZ = fp_t::one();
			
			res.inf();
			
			int lastBit = 0;
			for (int z = 0; z < 256; ++z)
			{
				int bit = get_wval(s, z, 1);
				if (bit)
				{
					res.add(tmp);
					lastBit = z;
					printf(" add %d\n", z);
				}
				tmp.add(tmp);
			}
			printf(" ================ \n");
			unsigned X[8], XH[8], X3[8], C[8], NP[8], NM[8], TOTAL[8];
			for (int p = 0; p < 8; ++p)
				X[p] = s[p];
			
			for (int p = 7; p >= 0; --p)
			{
				auto t = std::bitset<32>(X[p]);
				std::cout<< t;
			}
			printf("\n");
			longShift(XH, X, 8);
		    longAdd(X3, X, XH, 8);
			longXor(C, XH, X3, 8);
			longAnd(NP, X3, C, 8);
			longAnd(NM, XH, C, 8);
			
			for (int p = 7; p >= 0; --p)
			{
				auto t = std::bitset<32>(NP[p]);
				std::cout<< t;
			}
			printf("\n");			
			
			for (int p = 7; p >= 0; --p)
			{
				auto t = std::bitset<32>(NM[p]);
				std::cout<< t;
			}
			printf("\n");
			bucket_t res1;
			tmp.X = XY[0];
			tmp.Y = XY[1];
			tmp.ZZ = tmp.ZZZ = fp_t::one();
			
			res1.inf();
			
			int lastBit1 = 0;
			for (int z = 0; z < 256; ++z)
			{
				int bitP = get_wval(NP, z, 1);
				int bitN = get_wval(NM, z, 1);
				if (bitP)
				{
					res1.add(tmp);
					lastBit1 = std::max(lastBit1, z);
					printf(" add %d\n", z);
				}				
				else if (bitN)
				{					
					tmp.ZZZ.cneg(true);
					res1.add(tmp);
					tmp.ZZZ.cneg(true);

					//res1.add(tmp, true);					
					lastBit1 = std::max(lastBit1, z);
					printf(" subtract %d\n", z);
				}
				tmp.add(tmp);				
			}
			
			
			printf("is eq %d\n", res.cmp(res1));
			printf("is inf %d %d\n", res.is_inf(), res1.is_inf());
			printf("last %d %d\n", lastBit1, lastBit);
		}
#endif		
		
#if 0
		uint64_t bitCount = 0;
		unsigned min = 32, max = 0;
		
		vector<vector<int>> distr;		
		int wBit = 20;
		int nWin = 253 / wBit + ((253 % wBit) != 0);
		for (int z = 0; z <= nWin; ++z)
		{
			vector<int> tmp(1 << wBit);
			for (int k = 0; k < (1 << wBit); ++k)
				tmp[k] = 0;
			distr.push_back(tmp);
		}
		
//#pragma omp parallel for reduction(+: bitCount) reduction(max: max) reduction(min: min)	
		for (int z = 0; z < config.npoints; ++z)
		{
			unsigned* s = (unsigned*)&scalars[z];
			unsigned tt = 0;
			for (int p = 0; p < 8; ++p)
			{				
  			    unsigned res = 0;
				unsigned n = s[p];
				while (n) {
				  res ++;
				  n &= n-1;  // Забираем младшую единичку.
				}				
				tt += res;
				bitCount += res;
			}		
				
			min = std::min(min, tt);
			max = std::max(max, tt);
			
			for (int z = 0; z < nWin; ++z)
			{
				int val = get_wval(s, z * wBit, (z == nWin - 1) ? ((253 % wBit) ? (253 % wBit) : wBit) : wBit);
				distr[z][val]++;
			}
		}
		printf("total %lld, max %lld, ratio %f\n", bitCount, (uint64_t)config.npoints * 253,  (double)bitCount / ((double)config.npoints * 253.));
		printf("min %d, max %d\n", min, max);
		
		for (int z = 0; z < nWin; ++z)
		{
			size_t sum = 0;
			int lastNonZero = 0;
			for (int k = 1; k < (1 << wBit); ++k)
			{
				sum += distr[z][k];
				if (distr[z][k] != 0)
					lastNonZero	= k;
			}			
			printf("[%d] => %lld, nonZero idx %d\n", z, sum, lastNonZero);
		}

		bitCount = 0;
		min = 32, max = 0;

		for (int z = 0; z < nWin; ++z)
			for (int k = 0; k < (1 << wBit); ++k)
				distr[z][k] = 0;
		
		vector<set<int>> zeros(NWINS);
		vector<set<int>> countBucketsLast(1<<wBit);

//#pragma omp parallel for reduction(+: bitCount) reduction(max: max) reduction(min: min)		
		for (int z = 0; z < config.npoints; ++z)
		{
			unsigned* s = (unsigned*)&scalars[z];
			
			unsigned X[8], XH[8], X3[8], C[8], NP[8], NM[8], TOTAL[8];
			for (int p = 0; p < 8; ++p)
				X[p] = s[p];
			
			longShift(XH, X, 8);
		    longAdd(X3, X, XH, 8);
			longXor(C, XH, X3, 8);
			longAnd(NP, X3, C, 8);
			longAnd(NM, XH, C, 8);

			longOr(TOTAL, NP, NM, 8);
			unsigned tt = 0;
			for (int p = 0; p < 8; ++p)
			{				
  			    unsigned res = 0;
				unsigned n = TOTAL[p];
				while (n) {
				  res ++;
				  n &= n-1;
				}				
				tt += res;
				bitCount += res;								
			}
			min = std::min(min, tt);
			max = std::max(max, tt);
			
			int lastIdx = -1;
			for (int z = 0; z < nWin; ++z)
			{
				int valP = get_wval(NP, z * wBit, (z == nWin - 1) ? ((253 % wBit) ? (253 % wBit) : wBit) : wBit);
				int valN = get_wval(NM, z * wBit, (z == nWin - 1) ? ((253 % wBit) ? (253 % wBit) : wBit) : wBit);
				distr[z][abs(valP - valN)]++;
				
				if (abs(valP - valN) == 0)
					zeros[z].insert(z);
				
				if (z == nWin - 1 && lastIdx && abs(valP - valN))
					countBucketsLast[lastIdx].insert(abs(valP - valN));
				lastIdx = abs(valP - valN);
			}
		}
		printf("total %lld, max %lld, ratio %f\n", bitCount, (uint64_t)config.npoints * 253,  (double)bitCount / ((double)config.npoints * 253.));
		printf("min %d, max %d\n", min, max);
		
		for (int z = 0; z < nWin; ++z)
		{
			size_t sum = 0;
			int lastNonZero = 0;
			for (int k = 1; k < (1 << wBit); ++k)
			{
				sum += distr[z][k];
				if (distr[z][k] != 0)
					lastNonZero	= k;
			}			
			printf("[%d] => %lld, nonZero idx %d ", z, sum, lastNonZero);
			/*if (z == nWin - 2)
			{
				int minC = 5000;
				int maxC = 0;
				int sum = 0;
				for (int k = 0; k < lastNonZero; ++k)
				{
					minC = std::min((int)countBucketsLast[k].size(), minC);
					maxC = std::max((int)countBucketsLast[k].size(), maxC);
					sum += (int)countBucketsLast[k].size();					
				}
				printf(" min %d, max %d, %f, total sums %d", minC, maxC, sum / (double)lastNonZero, sum);
			}
			if (z != nWin - 1)
			{
				set<int> intersect;
				set_intersection(zeros[z].begin(), zeros[z].end(), zeros[nWin - 1].begin(), zeros[nWin - 1].end(), std::inserter(intersect, intersect.begin()));
				printf(" intersects zeros with last %d", intersect.size());
			}*/
			printf("\n");
		}

		bitCount = 0;
		min = 32, max = 0;

		for (int z = 0; z <= nWin; ++z)
			for (int k = 0; k < (1 << wBit); ++k)
				distr[z][k] = 0;
		
		int maxChunk = (1 << (wBit - 1));
		int substruct = (1 << wBit);
		for (int z = 0; z < config.npoints; ++z)
		{
			unsigned* s = (unsigned*)&scalars[z];
			unsigned tt = 0;
			for (int p = 0; p < 8; ++p)
			{				
  			    unsigned res = 0;
				unsigned n = s[p];
				while (n) {
				  res ++;
				  n &= n-1;  // Забираем младшую единичку.
				}				
				tt += res;
				bitCount += res;
			}		
				
			min = std::min(min, tt);
			max = std::max(max, tt);
			
			int nextPlus = 0;
			for (int z = 0; z < nWin; ++z)
			{
				int val = get_wval(s, z * wBit, (z == nWin - 1) ? ((253 % wBit) ? (253 % wBit) : wBit) : wBit) + nextPlus;
				nextPlus = 0;
				
				if (val >= maxChunk)
				{
					val -= substruct;
					nextPlus = 1;
				}
				
				val = (val < 0) ? -val : val;
				distr[z][val]++;
			}
			if (nextPlus == 1)
				distr[nWin][1]++;
		}
		printf("total %lld, max %lld, ratio %f\n", bitCount, (uint64_t)config.npoints * 253,  (double)bitCount / ((double)config.npoints * 253.));
		printf("min %d, max %d\n", min, max);
		
		for (int z = 0; z <= nWin; ++z)
		{
			size_t sum = 0;
			int lastNonZero = 0;
			for (int k = 1; k < (1 << wBit); ++k)
			{
				sum += distr[z][k];
				if (distr[z][k] != 0)
					lastNonZero	= k;
			}			
			printf("[%d] => %lld, nonZero idx %d ", z, sum, lastNonZero);
			printf("\n");
		}
		
		exit(0);
#endif

		thread* accAsyncJob = NULL;

		const int idx = numIter % 2;
		const unsigned tN = (1 << WBITS);
		const size_t npoints = isFull ? config.npoints : sizePartly;
		//printf(" SIZE = %u, shiftPartly = %d, sizePartly = %d\n", npoints, shiftPartly, sizePartly);
		const int numStrs = config.numAsync;

        affine_t *d_points = d_base_ptrs[d_bases_idx];
        scalar_t *d_scalars = d_scalar_ptrs[d_scalars_idx];
		scalar_t *d_scalars_next = d_scalar_ptrs[d_scalars_xfer];
				
#if PRINT
		printf("table size = %u, wbits = %d, NWINS = %d\n", tN, WBITS, NWINS);
#endif

		/*if (cpuRes == NULL)
			cpuRes = new bucket_t[NWINS * tN];*/

#if CPU_CHECK
		unsigned* hSc = new unsigned[npoints * 8];
		fp_t* hPointsX, *hPointsY;
		CUDA_OK(cudaMallocHost(&hPointsX, sizeof(fp_t) * npoints));
		CUDA_OK(cudaMallocHost(&hPointsY, sizeof(fp_t) * npoints));
		
		copyToHost<<<npoints / 256 + 1, 256>>>(d_points, hPointsX, hPointsY, npoints);
		CUDA_OK(cudaMemcpy(hSc, d_scalars, sizeof(unsigned) * 8 * npoints, cudaMemcpyDeviceToHost));

		bucket_t* cpuRes = new bucket_t[NWINS * tN];
		
		printf("add start, %d\n", sizeof(affine_t));
		memset(cpuRes, 0, sizeof(bucket_t) * NWINS * tN);
		
		int pointsNotInf = 0;
		
		int lastP = 0;
		int lastM = 0;
		
#if PRINT
		printf("npoints = %d\n", npoints);
#endif		
		for (int z = 0; z < npoints; ++z)
		{
			//printf("%d\n", z);
			unsigned *s = hSc + z * 8;
			affine_t point;
			point.X = hPointsX[z];
			point.Y = hPointsY[z];
			if (point.X.is_zero() && point.Y.is_zero())			
				point.inf[0] = 1;
			else
			{
				pointsNotInf++;
				point.inf[0] = 0;
			}

#if NAF
			unsigned X[8], XH[8], X3[8], C[8], NP[8], NM[8];
			for (int k = 0; k < 8; ++k)
				X[k] = s[k];
			
			longShift(XH, X, 8);
			longAdd(X3, X, XH, 8);
			longXor(C, XH, X3, 8);
			longAnd(NP, X3, C, 8);
			longAnd(NM, XH, C, 8);
			
			
			unsigned ss1 = NP[7];
			unsigned ss2 = NM[7];
			for (int t = 0; t < 32; ++t)
			{
				if (ss1 & 1)
					lastP = std::max(lastP, t + 1);
				if (ss2 & 1)
					lastM = std::max(lastM, t + 1);
				ss1 = ss1 >> 1;
				ss2 = ss2 >> 1;
			}			
#endif
			for (int p = 0; p < NWINS; ++p)
			{
				//printf(" -- %d\n", p);
				const int wbits = ((p * WBITS) > NBITS - WBITS) ? NBITS - (p * WBITS) : WBITS;
#if NAF
				int valP = get_wval(NP, p * WBITS, wbits);
				int valN = get_wval(NM, p * WBITS, wbits);
				int diff = valP - valN;
				
				if (diff)
				{
					if (diff < 0)
					{
						affine_t pointNeg = point;
						if (!pointNeg.is_inf())
							pointNeg.Y.cneg(true);
						cpuRes[(NWINS - 1 - p) * tN + abs(diff) - 1].add(pointNeg);
					}
					else
						cpuRes[(NWINS - 1 - p) * tN + abs(diff) - 1].add(point);
					
					//printf("%d %d %d\n", p, diff, z);
				}
#else
				unsigned val = get_wval(s, p * WBITS, wbits);
				if (val)
					cpuRes[(NWINS - 1 - p) * tN + val - 1].add(point);
#endif				
			}
		}
		
#if NAF		
		printf("lastP %d lastM %d\n", lastP, lastM);
#endif		
		printf("add end, non zero %d\n", pointsNotInf);
#endif

		part_t* part = part_prts[0];
		unsigned* idxs = idxs_prts[0];
		unsigned* rows = rows_prts[0];		
		unsigned* tableS = tableS_prts[0];
		unsigned* tableE = tableS + tN;
		unsigned* sizes = sizes_prts[0]; 
		
		bucket_t* d_buckets = d_buckets_ptrs[0];
		bucket_t (*d_none)[TILING][2] = reinterpret_cast<decltype(d_none)>(d_none_ptrs[0]);
		bucket_t (*c_none)[TILING][2] = reinterpret_cast<decltype(c_none)>(c_none_ptrs[0]);
#if CPU_CHECK
		bucket_t* cpuBucket = new bucket_t[NWINS * tN];
#else		
		bucket_t* cpuBucket = NULL;
#endif	
		
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
#if NAF
		
		int realTab = (int)(tN * config.naf_coef) + 1;
		int realTabAligned = config.naf_elems * (tN / TILING);
#if PRINT
		printf("naf real tab %d, aligned %d\n", realTab, realTabAligned);
#endif		
		const int xBlockP = realTab / thP + ((realTab % thP) != 0);
		const int xBlockP_half = (1 << (WBITS - 1)) / thP;
#else
		int realTab = tN;
		int realTabAligned = tN;
		const int xBlockP = tN / thP + ((tN % thP) != 0);
#endif	

		int numBlocks = config.numBlocks;
		numCPUAsync[idx] = 0;

		int numStr = 0;
#if PROF
		CUDA_OK(cudaEventRecord (events[0], streams[numStr]));
#endif

#if NAF
		initGroup_NAF<<<xBlock, 256, 0, streams[numStr]>>>(d_scalars, idxs, part, npoints, shiftPartly);
#else
		initGroup_full2<<<xBlock, 256, 0, streams[numStr]>>>(d_scalars, idxs, part, npoints, shiftPartly);
#endif

		if (bucketsDone == 0 && numIter == 0 && isFull || numIter == 0 && !isFull && shiftPartly == 0) // first init
		{
#if NAF == 0			
			CUDA_OK(cudaMemsetAsync(d_buckets, 0, sizeof(bucket_t) * realTabAligned * config.oneWaveBuckets, streams[numStr]));
#endif
			//TODO: batches = 4, move to config
			memset(c_none_ptrs[0], 0, sizeof(bucket_t) * TILING * 2 * NWINS * 4);
		}
		CUDA_OK(cudaStreamSynchronize(streams[numStr]));

#if PROF
		CUDA_OK(cudaEventRecord (events[1], streams[numStr]));
		CUDA_OK(cudaEventSynchronize ( events[1] ));
		{
			float time = 0;
			CUDA_OK(cudaEventElapsedTime (&time, events[0], events[1]));
			printf(" ** init %f ms\n", time);
		}		
#endif

#if PROF
		CUDA_OK(cudaEventRecord (events[0]));
#endif	
		if (bucketsDone == config.oneWaveBuckets && isFull)
			doFinalSum(-2, numStrs, bucketsDone, allDone, config, realTabAligned, false);
		
		thread* copyNextAsync = NULL;		
		if (isFull)
		{
			//MOVED
			//if (scalars) // if not last
			//	copyNextAsync = new thread(moveNext, ref(d_scalars_next), ref(scalars), 0, config.npoints, ref(scalars_prev), 0, config.npoints, ref(streams.back()));			
		}
		else
		{
			if (scalars) // if not last
				copyNextAsync = new thread(moveNext, ref(d_scalars_next), ref(scalars), shiftPartly, npoints, ref(scalars_prev), shiftPartly + npoints, npoints, ref(streams.back()));
		}
		
		const bool continueAdd = !isFull && shiftPartly > 0;
		unsigned elemsInBucket = 0;
		
		thread* lastSumAsync = NULL;
		bucket_t* d_bucket_tmp = d_buckets_ptrs[1];
		
		bool firstAsyncSumDone = false;
		for (int group = NWINS - 1, k = 0; group >= 0; --group, ++k)
		{
			if (k == 2 && isFull && scalars)			
				copyNextAsync = new thread(moveNext, ref(d_scalars_next), ref(scalars), 0, config.npoints, ref(scalars_prev), 0, config.npoints, ref(streams.back()));
			
			numStr = group % numStrs;
			
			cudaStream_t currSt = streams[numStr];
#if NAF
			if (!(group != NWINS - 1 || (NBITS % WBITS) == 0)) // last
				currSt = copy_streams[1];
#endif			
			sort(idxs + npoints * group, part + npoints * group, npoints, currSt, ((group == NWINS - 1) && ((NBITS % WBITS) != 0)) ? &elemsInBucket : NULL);
			initRows<<<xBlock, 256, 0, currSt>>>(rowsA[numStr], part + npoints * group, npoints);
			scan(rowsA[numStr], npoints, currSt);
			CUDA_OK(cudaMemsetAsync(tableS_A[numStr], 0, sizeof(int) * tN * 2, currSt));
			initTable<<<xBlock, 256, 0, currSt>>>(rowsA[numStr], tableS_A[numStr], tableE_A[numStr], npoints);
			initRows<<<xBlockT, 256, 0, currSt>>>(rowsA[numStr], sizesA[numStr], tableS_A[numStr], tableE_A[numStr], tN);
			sort1(rowsA[numStr], sizesA[numStr], tN, currSt, true);
			//printf(" ==== %d\n", group);
#if NAF
			if (group != NWINS - 1 || (NBITS % WBITS) == 0)
			{
				int blockX = xBlockP;
				int bucketSize = realTab;
				/*if (SIGNED_SCALAR) //TODO
				{
					if (group != NWINS - 1)
					{
						blockX = xBlockP_half;
						bucketSize = (1 << (WBITS - 1));
					}
				}*/

				pippenger_group_NAF<<<blockX, thP, 0, currSt>>>(d_points, d_buckets + realTabAligned * bucketsDone, rowsA[numStr], tableS_A[numStr], tableE_A[numStr], idxs + npoints * group, bucketSize, continueAdd);
				
				if (group == NWINS - 1 && (NBITS % WBITS) == 0 && (isFull || (!isFull && isLastIfNotFull)))
				{
					//printf("elemsInBucket = %d\n", elemsInBucket);
					
					CUDA_OK(cudaStreamSynchronize(currSt));
					const int shift = (1 << (WBITS - 1));
					const int maxElems = MAX_SCALAR_VAL_23_BIT; // max for 253bit scalar and only selected MOD
					const int numReduceElems = (1 << WBITS) / TILING;
					const int count = ((maxElems - shift) / numReduceElems + 1) * numReduceElems; // align to numReduceElems chunks
					//printf(" --- copy chunks %d\n", count / numReduceElems);
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
				
				pippenger_group_last_NAF<<<xBlockP, thP, 0, currSt>>>(d_points, d_bucket_tmp, rowsA[numStr], tableS_A[numStr], tableE_A[numStr], idxs + npoints * group, false, elemsInBucket, threadCount);
				bucketsDone--;
				
				CUDA_OK(cudaMemcpyAsync(cpuBucket_ptrs[0], d_bucket_tmp, threadCount * sizeof(bucket_t), cudaMemcpyDeviceToHost, currSt));
				lastSumAsync = new thread(&pippenger_t::sumLastBucketAsync, this, elemsInBucket, groups, numIter, continueAdd);
			}
#else
			
#if CPU_CHECK	
			pippenger_group<<<xBlockP, thP, 0, streams[numStr]>>>(d_points, d_buckets + realTabAligned * bucketsDone, rowsA[numStr], tableS_A[numStr], tableE_A[numStr], idxs + npoints * group, group, continueAdd);
#else
			if (group != NWINS - 1 || (NBITS % WBITS) == 0)
				pippenger_group<<<xBlockP, thP, 0, streams[numStr]>>>(d_points, d_buckets + realTabAligned * bucketsDone, rowsA[numStr], tableS_A[numStr], tableE_A[numStr], idxs + npoints * group, group, continueAdd);
			else
				pippenger_group_last<<<xBlockP, thP, 0, streams[numStr]>>>(d_points, d_buckets + realTabAligned * bucketsDone, rowsA[numStr], tableS_A[numStr], tableE_A[numStr], idxs + npoints * group, continueAdd);
#endif
			
#endif
			bucketsDone++;
			
			if (bucketsDone == config.oneWaveBuckets && (isFull || (!isFull && isLastIfNotFull)))
				doFinalSum(k, numStrs, bucketsDone, allDone, config, realTabAligned, (k == NWINS - 1) && scalars == NULL);
		}
		
		if (lastSumAsync)
		{
			lastSumAsync->join();
			delete lastSumAsync;
			lastSumAsync = NULL;			
		}

		//LAST
		if (bucketsDone && scalars == NULL && isFull)
			doFinalSum(-1, numStrs, bucketsDone, allDone, config, realTabAligned, true);

#if PROF		
		for (int z = 0; z < numStrs; ++z)
			CUDA_OK(cudaStreamSynchronize(streams[z]));

		CUDA_OK(cudaEventRecord (events[1]));
		CUDA_OK(cudaEventSynchronize ( events[1] ));
		{
			float time = 0;
			CUDA_OK(cudaEventElapsedTime (&time, events[0], events[1]));
			printf(" ** main part %f ms, avg %f ms\n", time, time / NWINS);
		}
#endif		

#if CPU_CHECK
		for (int z = 0; z < NWINS; ++z)
		{
			int err = 0;
			int err1 = 0;
			int notInf = 0;
			bucket_t* bb = cpuRes + z * tN;
			
			bucket_t* curr = cpuBucket + z * tN;
			int lastNotInf = 0;
#pragma omp parallel for reduction(+: err) reduction(+: notInf) reduction(max: lastNotInf)
			for (int k = 0; k < tN; ++k)
			{
				if (!curr[k].cmp(bb[k]))
				{
					err++;
					if (curr[k].is_inf())
						printf("  [%d] inf on CPU\n", k);
					else if (bb[k].is_inf())
						printf("  [%d] inf on GPU\n", k);
					else 
						printf("  [%d] diff\n", k);
				}

				if (!bb[k].is_inf())
				{
					notInf++;
					lastNotInf = std::max(lastNotInf, k);
				}
			}
			printf("[%d] err %d / %d, notInf %d, lastNotInf %d\n", NWINS - 1 - z, err, tN, notInf, lastNotInf);
		}
		
		delete[] cpuBucket;
#endif		

		double joinTime = omp_get_wtime();
		if (accAsyncJob)
		{
			accAsyncJob->join();
			delete accAsyncJob;
		}
		
		if (copyNextAsync)
		{
			copyNextAsync->join();
			delete copyNextAsync;
		}
		joinTime = omp_get_wtime() - joinTime;
#if PRINT		
		//printf("  join time %f\n", joinTime * 1000.);
#endif		
		//CUDA_OK(cudaStreamSynchronize(streams[numStr]));
		//printf(" --> DONE for idx%d\n", idx);		

	}
	
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
#if NAF
		if ((NBITS%WBITS) != 0)
			wins = NWINS - 1;
#endif

#pragma omp parallel for schedule(dynamic)
		for (int p = start; p < end; ++p)
		{
			int z = (wins - 1) - (p % wins);
			int shift = (p / wins) * NWINS;

			if (z == NWINS - 1)
			{
#if NAF
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
							/*if (!c_none[p][z][0].cmp(c_none_1[p / NWINS][z][0]))
								err++;
							if (!c_none[p][z][1].cmp(c_none_1[p / NWINS][z][1]))
								err++;*/
							
							c_none[p][z][0] = c_none_1[p / NWINS][z][0];
							c_none[p][z][1] = c_none_1[p / NWINS][z][1];
						}
						//printf("error count %d, p = %d, count = %d\n", err, p,  count);
					}
					integrated_batched[z + shift] = integrate_row_host(c_none[p], WBITS, true);
					
				}
				else
					;//integrated_batched[z + shift] = integrated_cpu[batch];
				
#else
				integrated_batched[z + shift] = integrate_row_host(c_none[p], NBITS%WBITS ? NBITS%WBITS : WBITS, true);
#endif	
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
	
	void accumulateAsync(point_t *out, int start, int end) 
	{
		
		for (int p = start; p < end; ++p)
			out[p] = pippenger_final_host(integrated_batched + p * NWINS);
	}
	
    // Perform final accumulation on CPU.
    void accumulate(MSMConfig& config, point_t &out, result_container_t &res) 
	{
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
