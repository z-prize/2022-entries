// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>
#include <chrono>
#include <omp.h>
#include <mutex>

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
/*
W=16 -> N 16  T->256 KB   B->0.01 GB / WIN
W=17 -> N 15  T->512 KB   B->0.02 GB / WIN
W=19 -> N 14  T->2048KB   B->0.08 GB / WIN
W=20 -> N 13  T->4096 KB  B->0.18 GB / WIN   <---- on 2080Ti
W=22 -> N 12  T->16 MB    B->0.75 GB / WIN
W=23 -> N 11  T->32 MB    B->1.5  GB / WIN   <---- on A40

W=26 -> N 10  T->256 MB   B->12   GB / WIN
W=29 -> N 9   T->2 GB     B->96   GB / WIN
W=32 -> N 8   T->16 GB    B->768  GB / WIN

NOTE! for W > 25 get_wval does not work!
*/

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
#define TILING 2048
#endif

#define CPU_CMP 0
#define WITH_TABLES 0
#define BUCKET_TEST 0
#define REF 0

#define PRINT_ASYNC 0
#define PRINT 1
#define PROF 1

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
__global__
void pippenger(const affine_t* points, size_t npoints,
               const scalar_t* scalars, bool mont,
               bucket_t (*buckets)[NWINS][1<<WBITS],
               bucket_t (*ret)[NWINS][NTHREADS][2] = nullptr);

__global__ void pippenger_group(const affine_t* points, bucket_t *buckets, const uint32_t* idx, const uint32_t* tableS, const uint32_t* tableE, const uint32_t* numbers, int wnum);
__global__ void pippenger_group_last(const affine_t* points, bucket_t *buckets, const uint32_t* idx, const uint32_t* tableS, const uint32_t* tableE, const uint32_t* numbers);
__global__ void pippenger_final(bucket_t *buckets, bucket_t (*ret)[TILING][2], int shift);

__global__ void initGroup_full(const scalar_t* scalars, unsigned int* idx, part_t* keys, int npoints);
__global__ void initGroup_full2(const scalar_t* scalars, unsigned int* idx, part_t* keys, int npoints, const unsigned *notNull);
__global__ void initIdx(unsigned int* idx, int npoints);
__global__ void initGroup(const scalar_t* scalars, unsigned int* idx, part_t* keys, int group, int npoints);
__global__ void initRows(unsigned int* idx, part_t* keys, int npoints);
__global__ void initRows(uint32_t* rows, uint32_t *sizes, const uint32_t* tableS, const uint32_t* tableE, int N);
__global__ void initTable(unsigned int* rows, unsigned int* tableS, unsigned int* tableE, int npoints);
__global__ void copyIdxs(const unsigned int* idxIn, unsigned int* idxOut, int npoints);
__global__ void zero_points(affine_t* points, size_t N);
__global__ void fill_not_null(const affine_t* points, unsigned* out, size_t N);

extern "C" void sort(unsigned int* idxs, part_t* part, size_t npoints, cudaStream_t& st);
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

__global__ void zero_points(affine_t* points, size_t N)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= N)
		return;
	
	if (points[id].is_inf())
	{
		points[id].X.zero();
		points[id].Y.zero();
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

__device__ void madd_base(const fp_t& X2, const fp_t& Y2, bucket_t& p31, int& resInf)
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
#if 1		
		fp_t P, R;

		R = Y2 * p31.ZZZ;         /* S2 = Y2*ZZZ1 */

		R -= p31.Y;                 /* R = S2-Y1 */
		P = X2 * p31.ZZ;          /* U2 = X2*ZZ1 */
		P -= p31.X;                 /* P = U2-X1 */

		if (!P.is_zero()) {         /* X1!=X2 */
			fp_t PP;             /* add |p2| to |p1| */

			PP = P^2;               /* PP = P^2 */
#define PPP P
			PPP = P * PP;           /* PPP = P*PP */
			p31.ZZ *= PP;           /* ZZ3 = ZZ1*PP */
			p31.ZZZ *= PPP;         /* ZZZ3 = ZZZ1*PPP */
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
		} else if (R.is_zero()) {   /* X1==X2 && Y1==Y2 */
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
#endif		
	}
}
	
__device__ __inline__ void madd(fp_t& X2, fp_t& Y2, bucket_t& result, int& resInf)
{
	if (X2.is_zero())
		if (Y2.is_zero())
			return;
	
	if (resInf == 1) 
	{
		resInf = 0;
		result.X = X2;
		result.Y = Y2;
		result.ZZZ = result.ZZ = fp_t::one();
	} 
	else 
	{
		fp_t P = X2 * result.ZZ;          
		P -= result.X;
		
		// X1!=X2
		if (!P.is_zero()) 
		{
			X2 = P*P;
			result.ZZ *= X2;			
			Y2 *= result.ZZZ;			
			P *= X2;
			result.ZZZ *= P;			
			Y2 -= result.Y;
			X2 *= result.X;       
			result.X = Y2*Y2;
			result.X -= P;         
			result.X -= X2;
			result.X -= X2;           
			X2 -= result.X;
			X2 *= Y2;                 
			result.Y *= P;         
			result.Y = X2 - result.Y;
		}
		else 
		{
			P = Y2 * result.ZZZ;
			P -= result.Y;
			
			// X1==X2 && Y1==Y2 
			if (P.is_zero())   
			{						
				P = Y2 + Y2;        
				result.ZZ = P*P;          
				result.ZZZ = result.ZZ * P;						
				result.Y = result.ZZZ * Y2;
				
				P = X2 * result.ZZ;     
				X2 = X2*X2;
				result.X = X2 + X2;
				X2 += result.X;
				result.X = X2*X2;
				result.X -= P;
				result.X -= P;           
				
				P -= result.X;
				P *= X2;
				result.Y = P - result.Y;
			} 
			else // X1==X2 && Y1==-Y2 
			{
				resInf = 1;
				result.inf();
			}
		}
	}
}

__global__
//__launch_bounds__(128, 4)
void pippenger_group(const affine_t* points, bucket_t *buckets, 
				     const uint32_t* idx, const uint32_t* tableS, const uint32_t* tableE,
					 const uint32_t* numbers, int wnum)
{	
	const int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	const int id = globalId;	
	
	if (id >= (1 << WBITS))
		return;

	bucket_t result;
	result.inf();
	int resInf = 1;
	
#if CPU_CMP
	memset(&result, 0, sizeof(bucket_t));
#endif

#if 1
	int idxBucket = idx[id];
	int start = tableS[idxBucket];
	int end = tableE[idxBucket];

	if (idxBucket > 0)
	{
		for (int z = start; z < end; z++)
		{			
			//result.add(points[numbers[z]]);

			const int idx = numbers[z];
			fp_t X2 = points[idx].X;
			fp_t Y2 = points[idx].Y;
		
			madd_base(X2, Y2, result, resInf);
		}

#if BUCKET_TEST
		buckets[threadIdx.x] = result;
#else
		buckets[idxBucket - 1] = result;
#endif
	}
#endif
} 

__global__
//__launch_bounds__(128, 4)
void pippenger_group_last(const affine_t* points, bucket_t *buckets, 
						  const uint32_t* idx, const uint32_t* tableS, const uint32_t* tableE,
						  const uint32_t* numbers)
{
	int globalId = threadIdx.x  + blockIdx.x * blockDim.x;
	if (globalId >= (1 << WBITS))
		return;
	
	const int lastMax = (1 << (NBITS % WBITS));
	const int id = globalId % lastMax;
	const int stride = globalId / lastMax;
	const int groups = (1 << WBITS) / lastMax;

	bucket_t result;
	result.inf();
	int resInf = 1;
#if CPU_CMP
	memset(&result, 0, sizeof(bucket_t));
#endif

	int idxBucket = idx[id];
	int start = tableS[idxBucket];
	int end = tableE[idxBucket];
	
	if (idxBucket > 0)
	{
		for (int z = start + stride; z < end; z += groups)
		{			
			//result.add(points[numbers[z]]);
			
			const int pp = numbers[z];
			fp_t X2 = points[pp].X;
			fp_t Y2 = points[pp].Y;
			madd_base(X2, Y2, result, resInf);
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
	
	
/*     bucket_t acc = row[--i];	
	bucket_t tmp = acc;

    while (i--) 
	{
#if BUCKET_TEST			
        bucket_t p = row[i % 1024];
#else
		bucket_t p = row[i];
#endif
		acc.add(p);
		tmp.add(acc);		
    }

	ret[id / TILING][id % TILING][0] = tmp;
    ret[id / TILING][id % TILING][1] = acc;*/	
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

__global__ void initGroup_full2(const scalar_t* scalars, unsigned int* idx, part_t* keys, int npoints, const unsigned* notNull)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= npoints)
		return;
	idx[id] = id;
	bool notNullFlag = notNull ? ((notNull[id / 32] & (1 << (id % 32))) != 0) : true;
	
#pragma unroll	
	for (int group = 0; group < NWINS; ++group)
	{
		const int wbits = ((group * WBITS) > NBITS-WBITS) ? NBITS-(group * WBITS) : WBITS;
		uint32_t val = get_wval(scalars[id], group * WBITS, wbits);
		keys[id + npoints * group] = notNullFlag ? val : 0;
	}
}

__global__ void initIdx(unsigned int* idx, int npoints)
{
	int id = threadIdx.x  + blockIdx.x * blockDim.x;
	if (id >= npoints)
		return;
	idx[id] = id;
}

__global__ void initGroup(const scalar_t* scalars, unsigned int* idx, part_t* keys, int group, int npoints)
{
	int id = threadIdx.x  + blockIdx.x * blockDim.x;
	if (id >= npoints)
		return;
	idx[id] = id;

	uint32_t val = get_wval(scalars[id], group * WBITS, WBITS);	
	keys[id] = val;
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
	point_t integrated_prt[2][NWINS];
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
    private:
        size_t N;
        size_t n;
		int numAsync;
		int numBlocks;
		int thF;
		bool filedNotNull;
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
		config.filedNotNull = false;
		CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor (&config.numBlocks, pippenger_final, config.thF, 0));
		
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

    size_t get_size_bases(MSMConfig& config)   { return config.n * sizeof(affine_t); }
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

		part_prts.allocate(sizeof(part_t) * config.npoints * NWINS); 
		idxs_prts.allocate(sizeof(unsigned) * config.npoints * (config.numAsync + 1));
		rows_prts.allocate(sizeof(unsigned) * std::max(config.npoints, (size_t)tN) * config.numAsync);
		tableS_prts.allocate(sizeof(unsigned) * tN * (2 * config.numAsync));
		sizes_prts.allocate(sizeof(unsigned) * tN * config.numAsync);
#if BUCKET_TEST == 0
		d_buckets_ptrs.allocate(sizeof(bucket_t) * NWINS * tN);
#else
		d_buckets_ptrs.allocate(sizeof(bucket_t) * NWINS * 1024);
#endif	
		d_none_ptrs.allocate(sizeof(bucket_t) * NWINS * TILING * 2);
		
		c_none_ptrs.allocate(sizeof(bucket_t) * NWINS * TILING * 2, true);
		c_none_ptrs.allocate(sizeof(bucket_t) * NWINS * TILING * 2, true);
		cpuBucket_ptrs.allocate(sizeof(bucket_t) * 5 * tN, true);

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
			config.filedNotNull = true;
			CUDA_OK(cudaStreamSynchronize(stream));
		}*/
    }

    // Transfer scalars to device. Throws cuda_error on error.
    void transfer_scalars_to_device(MSMConfig& config, size_t d_scalars_idx, const scalar_t scalars[], cudaStream_t s = nullptr) 
	{
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        scalar_t *d_scalars = d_scalar_ptrs[d_scalars_idx];
        CUDA_OK(cudaSetDevice(device));
        CUDA_OK(cudaMemcpy(d_scalars, scalars, config.npoints*sizeof(*d_scalars), cudaMemcpyHostToDevice));
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
		

    		size_t cpuCopy = (total / 2);
		size_t gpuCopy = total - cpuCopy;
                size_t shift = gpuCopy / sizeof(*d_scalars);

		scalar_t* ptr1 = h_scalars;
		const scalar_t* ptr2 = scalars + shift;
		
		thread job_copy(copy, ptr1, ptr2, cpuCopy);
		//memcpy(h_scalars, scalars + shift, total / 2);
#if PROF		
		double T = omp_get_wtime();
#endif
		CUDA_OK(cudaMemcpy(d_scalars, scalars, gpuCopy, cudaMemcpyHostToDevice));
#if PROF		
		T = omp_get_wtime() - T;
                printf("  --- copy cpu-gpu #1 %f ms, %f gb/s\n", T * 1000., gpuCopy * (1.0 / T) / 1024. / 1024. / 1024.);
#endif

		job_copy.join();
#if PROF
		T = omp_get_wtime();
#endif		
		CUDA_OK(cudaMemcpy(d_scalars + shift, h_scalars, cpuCopy, cudaMemcpyHostToDevice));
#if PROF		
		T = omp_get_wtime() - T;
                printf("  --- copy gpu-gpu #2 %f ms, %f gb/s\n", T * 1000., cpuCopy * (1.0 / T) / 1024. / 1024. / 1024.);
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

	static point_t integrate_row_host(const bucket_t row[TILING][2], int wbits = WBITS)
	{
		const int NTHRBITS = log2(TILING);
		size_t i = TILING-1;
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

	static point_t pippenger_final_host(const bucket_t ret[NWINS][TILING][2], point_t* integrated)
	{
		size_t i = NWINS-1;
		//point_t res = integrate_row_host(ret[i], NBITS%WBITS ? NBITS%WBITS : WBITS);

		point_t res	= integrated[i];
		while (i--) {
			for (size_t j = 0; j < WBITS; j++)
				res.dbl();
			res.add(integrated[i]);
			//res.add(integrate_row_host(ret[i], WBITS));
		}

		return res;
	}

	std::mutex g_lock;
	
	void unite(int idx, int TILES, int GR, int str)
	{
		bucket_t (*c_none)[TILING][2] = reinterpret_cast<decltype(c_none)>(c_none_ptrs[idx]);
		bucket_t* cpuBucket = cpuBucket_ptrs[0];
		point_t* integrated = integrated_prt[idx];
		bucket_t* d_buckets = d_buckets_ptrs[0];
		
		const int NTHRBITS = log2(TILES);	
		const uint32_t bucket_size = (1 << WBITS);
		
		CUDA_OK(cudaMemcpyAsync(cpuBucket + bucket_size * (NWINS - 1 - GR), d_buckets + bucket_size * GR, (1<<WBITS) * sizeof(bucket_t), cudaMemcpyDeviceToHost, copy_streams[str]));
		CUDA_OK(cudaStreamSynchronize(copy_streams[str]));
		
		//const int TILES = TILING;
		//printf("job started for %d\n", GR);
		
		//g_lock.lock();
#if PRINT_ASYNC
		double ompT = omp_get_wtime();
#endif 		
		// total = NWINS * TILES
#pragma omp parallel for
		for (int id = 0; id < TILES; ++id)
		{	
			bucket_t* row = cpuBucket + (NWINS - 1 - GR) * bucket_size;
			uint32_t i = (1<<(WBITS-NTHRBITS));

			row += (id % TILES) * i;						
			
			bucket_t acc = row[--i];
			bucket_t tmp = acc;
			while (i--) 
			{
				bucket_t p = row[i];
				acc.add(p);
				tmp.add(acc);
			}

			c_none[GR][id % TILES][0] = tmp;			
			c_none[GR][id % TILES][1] = acc;
		}
		
		if (GR == NWINS - 1)
			integrated[GR] = integrate_row_host_t(c_none[GR], TILES, NBITS%WBITS ? NBITS%WBITS : WBITS);
		else
			integrated[GR] = integrate_row_host_t(c_none[GR], TILES, WBITS);
#if PRINT_ASYNC		
		ompT = omp_get_wtime() - ompT;
		printf("  -->job unite finished for %d win in %f ms\n", GR, ompT * 1000.);
#endif		
		//g_lock.unlock();
	}
	
	static void moveNextScalar(scalar_t* h_scalars, const scalar_t* scalars, scalar_t *d_scalars, size_t npoints, cudaStream_t& st)
	{
#if PRINT_ASYNC		
		double ompT = omp_get_wtime();
#endif		
		memcpy(h_scalars, scalars, npoints * sizeof(scalar_t));
/*#pragma omp parallel for 
		for (int z = 0; z < npoints; ++z)
			h_scalars[z] = scalars[z];*/
		
		CUDA_OK(cudaMemcpyAsync(d_scalars, h_scalars, npoints * sizeof(scalar_t), cudaMemcpyHostToDevice, st));
		CUDA_OK(cudaStreamSynchronize(st));
#if PRINT_ASYNC		
		ompT = omp_get_wtime() - ompT;
		printf("  -->job move next finished in %f ms\n", ompT * 1000.);
#endif
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
	
	void launch_best(int numIter, point_t* out, 
					 MSMConfig& config, size_t d_bases_idx, size_t d_scalars_idx, size_t d_scalars_xfer, 
					 scalar_t* h_scalars, const scalar_t* scalars, 
					 bool mont = true, cudaStream_t s = nullptr)
	{
		thread* accAsyncJob = NULL;
		point_t integrated_copy[NWINS];
		if (numIter > 0)
		{
			//accumulate(numIter - 1, out[numIter - 1]);
			accAsyncJob = new thread(&pippenger_t::accumulate_, this, numIter - 1, ref(out[numIter - 1]));
		}

		const int idx = numIter % 2;
#if BUCKET_TEST		
		const uint32_t bucket_size = 1024;
#else
		const uint32_t bucket_size = (1<<WBITS);
#endif			
		const unsigned tN = (1 << WBITS);
		const size_t npoints = config.npoints;		
		const int numStrs = config.numAsync;
		
		assert(WBITS > NTHRBITS);
        cudaStream_t stream = (s == nullptr) ? default_stream : s;
        affine_t *d_points = d_base_ptrs[d_bases_idx];
        scalar_t *d_scalars = d_scalar_ptrs[d_scalars_idx];
		scalar_t *d_scalars_next = d_scalar_ptrs[d_scalars_xfer];
				
#if PRINT
		printf("table size = %u, wbits = %d, NWINS = %d\n", tN, WBITS, NWINS);
#endif
		part_t* part = part_prts[0];
		unsigned* idxs = idxs_prts[0];
		unsigned* rows = rows_prts[0];		
		unsigned* tableS = tableS_prts[0];
		unsigned* tableE = tableS + tN;
		unsigned* sizes = sizes_prts[0]; 
		
		bucket_t* d_buckets = d_buckets_ptrs[0];
		bucket_t (*d_none)[TILING][2] = reinterpret_cast<decltype(d_none)>(d_none_ptrs[0]);
		bucket_t (*c_none)[TILING][2] = reinterpret_cast<decltype(c_none)>(c_none_ptrs[idx]);
		bucket_t* cpuBucket = cpuBucket_ptrs[0];

		unsigned* idxsA[numStrs + 1];
		for (int z = 0; z < numStrs + 1; ++z)
			idxsA[z] = idxs + z * npoints;
		
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
		const int xBlockP = tN / thP + ((tN % thP) != 0);

		int numBlocks = config.numBlocks;
		numCPUAsync[idx] = 0;
				
		const int totalThreads = numBlocks * thF * sm_count;
#if PRINT		
		printf("  max active blocks %d, threads %d, total %d\n", numBlocks, numBlocks * thF, totalThreads);
		printf("  tiling size %d, total needs for all wins %d threads\n", TILING, TILING * NWINS);
#endif		
		//numCPUAsync[idx] = 3;
		/*if (TILING * NWINS > totalThreads)
		{
			numCPUAsync[idx] = NWINS - totalThreads / TILING;
#if PRINT			
			printf("  need to count on cpu %d wins, on gpu threads %d\n", numCPUAsync[idx], TILING * (NWINS - numCPUAsync[idx]));
#endif			
		}*/

		int numStr = 0;
		int prevStr = 0;		
		vector<thread*> jobs;
#if PROF
		CUDA_OK(cudaEventRecord (events[0], streams[numStr]));
#endif
		CUDA_OK(cudaMemsetAsync(d_buckets, 0, sizeof(bucket_t) * bucket_size * NWINS, streams[numStr]));
		initGroup_full2<<<xBlock, 256, 0, streams[numStr]>>>(d_scalars, idxsA[numStrs], part, npoints, config.filedNotNull ? pointNotNull[0] : NULL);		
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
		
		thread* copyNextAsync = NULL;
		if (scalars)
			copyNextAsync = new thread(moveNextScalar, ref(h_scalars), ref(scalars), ref(d_scalars_next), npoints, ref(streams.back()));

		int usedCopyStrs = 0;
#if PROF
		CUDA_OK(cudaEventRecord (events[0]));
#endif		
		for (int group = NWINS - 1, k = 0; group >= 0; --group, ++k)
		{
			numStr = group % numStrs;
			if (k != 0)
				prevStr	= (group + 1) % numStrs;
			
			if (group == 2)
			{
#if PROF
				CUDA_OK(cudaEventRecord (events[2], streams[numStr]));
#endif		
			}				
			//initIdx<<<xBlock, 256, 0, streams[numStr]>>>(idxsA[numStr], npoints);
			CUDA_OK(cudaMemcpyAsync(idxsA[numStr], idxsA[numStrs], sizeof(int) * npoints, cudaMemcpyDeviceToDevice, streams[numStr]));
						
			sort(idxsA[numStr], part + npoints * group, npoints, streams[numStr]);
			initRows<<<xBlock, 256, 0, streams[numStr]>>>(rowsA[numStr], part + npoints * group, npoints);
			scan(rowsA[numStr], npoints, streams[numStr]);
			CUDA_OK(cudaMemsetAsync(tableS_A[numStr], 0, sizeof(int) * tN * 2, streams[numStr]));
			initTable<<<xBlock, 256, 0, streams[numStr]>>>(rowsA[numStr], tableS_A[numStr], tableE_A[numStr], npoints);
			initRows<<<xBlockT, 256, 0, streams[numStr]>>>(rowsA[numStr], sizesA[numStr], tableS_A[numStr], tableE_A[numStr], tN);
			sort1(rowsA[numStr], sizesA[numStr], tN, streams[numStr], true);
			if (group == 2)
			{
#if PROF
				CUDA_OK(cudaEventRecord (events[3], streams[numStr]));
				CUDA_OK(cudaStreamSynchronize(streams[numStr]));
				CUDA_OK(cudaEventSynchronize ( events[3] ));
				{
					float time = 0;
					CUDA_OK(cudaEventElapsedTime (&time, events[2], events[3]));
					printf(" **** sort of #2 group %f ms\n", time);
				}
				
				CUDA_OK(cudaEventRecord (events[4], streams[numStr]));				
#endif		
			}
			if (group != NWINS - 1 || (NBITS % WBITS) == 0)
				pippenger_group<<<xBlockP, thP, 0, streams[numStr]>>>(d_points, d_buckets + bucket_size * group, rowsA[numStr], tableS_A[numStr], tableE_A[numStr], idxsA[numStr], group);
			else
				pippenger_group_last<<<xBlockP, thP, 0, streams[numStr]>>>(d_points, d_buckets + bucket_size * group, rowsA[numStr], tableS_A[numStr], tableE_A[numStr], idxsA[numStr]);

			if (group == 2)
			{
#if PROF
				CUDA_OK(cudaEventRecord (events[5], streams[numStr]));
				CUDA_OK(cudaEventSynchronize ( events[5] ));
				{
					float time = 0;
					CUDA_OK(cudaEventElapsedTime (&time, events[4], events[5]));
					printf(" **** part of #2 group %f ms\n", time);
				}
#endif		
			}				
			
			if (k <= numCPUAsync[idx] && k != 0)
			{
				//CUDA_OK(cudaStreamSynchronize(streams[prevStr]));
				//jobs.push_back(new std::thread(&pippenger_t::unite, this, idx, 256, group + 1, usedCopyStrs++));
			}
		}
		//finalyze(numStrs, thF, idx);
		
		for (int z = 0; z < numStrs; ++z)
			CUDA_OK(cudaStreamSynchronize(streams[z]));
#if PROF
		CUDA_OK(cudaEventRecord (events[1]));
		CUDA_OK(cudaEventSynchronize ( events[1] ));
		{
			float time = 0;
			CUDA_OK(cudaEventElapsedTime (&time, events[0], events[1]));
			printf(" ** main part %f ms, avg %f ms\n", time, time / NWINS);
		}
#endif		

#if PROF
		CUDA_OK(cudaEventRecord (events[0]));
#endif		
#if BUCKET_TEST == 0		
		pippenger_final<<<(TILING * (NWINS - numCPUAsync[idx]))/ thF, thF, 0, streams[numStrs > 1 ? 1 : 0]>>>(d_buckets, d_none, 0);
#endif	
		CUDA_OK(cudaMemcpy(c_none, d_none, (NWINS - numCPUAsync[idx]) * sizeof(d_none[0]), cudaMemcpyDeviceToHost));
#if PROF
		CUDA_OK(cudaEventRecord (events[1]));
		CUDA_OK(cudaEventSynchronize ( events[1] ));
		{
			float time = 0;
			CUDA_OK(cudaEventElapsedTime (&time, events[0], events[1]));
			printf(" ** final part %f ms\n", time);
		}
#endif	
		double joinTime = omp_get_wtime();
		for (auto& elem : jobs)
		{
			elem->join();
			delete elem;
		}

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
		printf("  join time %f\n", joinTime * 1000.);
#endif		
		//CUDA_OK(cudaStreamSynchronize(streams[numStr]));
		//printf(" --> DONE for idx%d\n", idx);		
	}

	void accumulate_(int numIter, point_t &out) 
	{
		const int idx = numIter % 2;
		
		//printf(" --> INTEGRATE for idx%d\n", idx);		
		bucket_t (*c_none)[TILING][2] = reinterpret_cast<decltype(c_none)>(c_none_ptrs[idx]);
		
#pragma omp parallel for 	
		for (int z = 0; z < NWINS - numCPUAsync[idx]; ++z)
		{
			if (z == NWINS - 1)
				integrated_prt[idx][z] = integrate_row_host(c_none[z], NBITS%WBITS ? NBITS%WBITS : WBITS);
			else
				integrated_prt[idx][z] = integrate_row_host(c_none[z], WBITS);
		}
		out = pippenger_final_host(c_none, integrated_prt[idx]);
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
