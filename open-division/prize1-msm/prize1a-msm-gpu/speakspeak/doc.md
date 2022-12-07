Following [the ZPrize specification](https://assets.website-files.com/625a083eef681031e135cc99/6314b255fac8ee1e63c4bde3_gpu-fpga-msm.pdf),
Our MSM implementation is specifically designed for the fixed-base point MSM with 2^26 randomly sampled scalars from the BLS12-377 scalar field. 
Due to the deadline of the competition, GPU implemetations for MSM on other elliptic curves will be released later.

Below we describe our MSM implementation in detail.

Similar to [the reference](https://github.com/z-prize/test-msm-gpu), our MSM implementation is also Pippenger-based. 

1. We divide each randomly sampled BLS12-377 scalar (253 bits) into 13 parts (13 sub-scalars). The first 12 sub-scalars have 21 bits each, 
and the last sub-scalar has only 1 bit. (12 * 21 + 1 = 253). Since there are a total of 2^26 scalars, there will be 13 * (2^26) sub-scalars.
2. We store these sub-scalars and their corresponding scalar indexes into the GPU global memory. 
Note that each group of 2^26 sub-scalars obtained from the same bit position of BLS12-377 scalars are stored consecutively.

Specifically,
```
# define NBITS 253
# define WBITS 21
# define NWINS 13

/* "get_wval" function is used to get sub-scalar from the BLS12-377 scalar's "off" bit to "top" bit */
static __device__ int get_wval(const scalar_t& d, uint32_t off, uint32_t bits)
{
    uint32_t top = off + bits - 1;
    uint64_t ret = ((uint64_t)d[top/32] << 32) | d[off/32];
    return (int)(ret >> (off%32)) & ((1<<bits) - 1);
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
void pippenger_faster_1(const scalar_t* scalars_, size_t nscalars, uint32_t *wval, uint32_t *idx) 
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
}
```

After 2^26 scalars are divided into 13 * (2^26) sub-scalars, we actually convert the original computation to 13 smaller subtasks, 
where each subtask performs inner product between 2^26 BLS12-377 points and 2^26 sub-scalars.

Next, we process the first 12 subtasks and the last subtask in two different methods.

For the first 12 subtasks, 

3. We sort 2^26 sub-scalars of each subtasks into ascending order. Here, we use "key"-"value" pair sort method from [CUB](https://nvlabs.github.io/cub/). 
The "key" is the sub-scalar and the "value" is its corresponding scalar index.

Specifically, 
```
/*
[in]   wval: 'NWINS - 1' groups of sub-scalars. 
[in]   idx:  'NWINS - 1' groups of scalar indexes.
[out]  wval_out: 'NWINS - 1' groups of sorted sub-scalars. 
    (allocated in the GPU global memory, and each group has 'nscalars' elements)
[out]  idx_out: 'NWINS - 1' groups of sorted scalar indexes. 
    (allocated in the GPU global memory, and each group has 'nscalars' elements)
*/
for(size_t k=0; k < NWINS - 1; k++)
{
    size_t ptr = k * nscalars;
    cub::DeviceRadixSort::SortPairs(tmp, tmp_size, wval + ptr, wval_out + ptr, idx + ptr, idx_out + ptr, nscalars, 0 ,24);
}
```

4. For each subtasks, we count the number of sub-scalars with the same value. Because sub-scalars have been sorted
in ascending order, we can use run-length encoding to accomplish this counting step.

Specifically, 
```
/*
[in]  wval_out: 'NWINS - 1' groups of sorted sub-scalars. 
[out] wval_count: the number of sub-scalars with the same value in each group of sorted sub-scalars. 
                  (allocated in the GPU global memory with '(NWINS - 1) * (1 << WBITS)' elements)
[out] wval_unique: 'NWINS - 1' groups of unique sub-scalars.  
                  (allocated in the GPU global memory, and each group has (1 << WBITS) elements)
[out] wval_unique_count: the number of elements in each group of unique sub-scalars. 
                  (allocated in the GPU global memory with 'NWINS - 1' elements)
*/
for(uint32_t k=0; k < NWINS - 1; k++)
{
    uint32_t ptr = k * nscalars;
    uint32_t cptr = k * (1 << WBITS);
    cub::DeviceRunLengthEncode::Encode(tmp, tmp_size, wval_out + ptr, wval_unique + cptr, wval_count + cptr, wval_unique_count + k, nscalars);
}
```

5. We calculate the prefix sum of the "wval_count".

Specifically, 
```
/*
[in]  wval_count: the number of sub-scalars with the same value in each group of sorted sub-scalars.
[out] wval_ptr: the prefix sum of "wval_count".  
                (allocated in the GPU global memory with '(NWINS - 1) * (1 << WBITS)' elements)
*/
for(uint32_t k=0; k < NWINS - 1; k++)
{
    uint32_t ptr = k * (1 << WBITS);
    cub::DeviceScan::InclusiveSum(tmp, tmp_size, wval_count + ptr, wval_ptr + ptr, 1 << WBITS);
}
```

6. Similar to the Pippenger algorithm and [the reference](https://github.com/z-prize/test-msm-gpu),
we put BLS12-377 points into buckets according to their corresponding sub-scalars. 
For each group of sub-scalars, we launch N * NTHREADS threads to process this step, 
where the total number of threads '(NWINS - 1) * N * NTHREADS' is twice the total number of GPU cores, N = 2 * Cores / ((NWINS - 1) *  NTHREADS).
'NTHREADS' represent the number of threads in a block. 

Speficically, 
```
/* the number of threads in a block. */
#define NTHREADS 32
static __shared__ affine_t spoints[NTHREADS]; 

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
pippenger_faster_2<<<dim3(NWINS - 1, N), NTHREADS>>>(points, npoints, wval_out, idx_out, 
                    wval_count, wval_ptr, wval_unique, wval_unique_count, buckets);

__global__
void pippenger_faster_2(affine_t* points, size_t npoints, uint32_t *wval_out, uint32_t *idx_out,
        uint32_t *wval_count, uint32_t *wval_ptr, uint32_t *wval_unique, uint32_t *wval_unique_count,
        bucket_t (*buckets)[1<<WBITS])
{
    const uint32_t tnum = blockDim.x * gridDim.y;
    const uint32_t tid = blockIdx.y * blockDim.x + threadIdx.x;
    const uint32_t bid = blockIdx.x;
    bucket_t* row = buckets[bid];

    uint32_t ptr = bid * npoints;
    uint32_t cptr = bid * (1 << WBITS);

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
    uint32_t wcount = wval_count[cptr + i];
    uint32_t wptr = wval_ptr[pptr + i];
    uint32_t w = wval_out[ptr + wptr - wcount];
    bucket_t mid;
    mid.inf();
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
        uint32_t idx = idx_out[ptr + j];
        spoints[threadIdx.x] = points[idx];
        mid.add(spoints[threadIdx.x]);
        wcount--;
        if(wcount == 0)
        {
            row[w] = mid;
            i++;
        }
    }
}
```

7. Similar to the Pippenger algorithm and [the reference](https://github.com/z-prize/test-msm-gpu),
we add up points in each group of buckets weighted by the corresponding index to that bucket in parallel.

Specifically, 
```
/* the number of threads in a block. */
#define NTHREADS 32
static __shared__ bucket_t ress[NTHREADS]; 
static __shared__ bucket_t accs[NTHREADS]; 
static __shared__ bucket_t smid[NTHREADS]; 

/*
[in]  buckets: (NWINS - 1) groups of buckets used to store BLS12-377 points. 
[out] tres: the calculation results of all threads individually
    (allocated in the GPU global memory with '(NWINS - 1) * N * NTHREADS' elements)
*/
pippenger_faster_3<<<dim3(NWINS - 1, N), NTHREADS>>>(buckets, tres);

__global__
void pippenger_faster_3(bucket_t (*buckets)[1<<WBITS], bucket_t* tres)
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
        spoints[threadIdx.x] = row[i];
        accs[threadIdx.x].add(spoints[threadIdx.x]);
        ress[threadIdx.x].add(accs[threadIdx.x]);
    }

    /*  after the following steps, 
        smid[threadIdx.x] = (s - 1) * accs[threadIdx.x]
                          = (s - 1) * row[e - 1] + (s - 1) * row[e - 2] + ... + (s - 1) * row[s + 1] + (s - 1) * row[s]
        ress[threadIdx.x] = (e - 1) * row[e - 1] + (e - 2)row[e - 2] + ... + (s + 1) * row[s + 1] + s * row[s]
    */
    if(s != 0)
    {
        mul(smid[threadIdx.x], accs[threadIdx.x], (s - 1));
        ress[threadIdx.x].add(smid[threadIdx.x]);
    }

    uint32_t bptr = bid * tnum;
    tres[bptr + tid] = ress[threadIdx.x];
}
```

8. We reduce the calculation results of all threads to the subtask results (the first 12 subtasks). 

Specifically, 
```
/*
[in]  tres: the calculation results of all threads individually.
[out] res: results of each subtasks.  
    (allocated in the GPU global memory with length: NWINS)
*/
launch_coop(CSum, dim3(NWINS-1, N / 2), NTHREADS, stream, res, fres); 

template<typename... Types>
inline void launch_coop(void(*f)(Types...), dim3 gridDim, dim3 blockDim, cudaStream_t stream, Types... args)
{
    void* va_args[sizeof...(args)] = { &args... };
    CUDA_OK(cudaLaunchCooperativeKernel((const void*)f, gridDim, blockDim, va_args, 0, stream));
}

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
```

For the last subtask,

9. We select sub-scalars whose values are equal to 1. (Sub-scalars whose values are equal to 0 have no effect on the final result)

Specifically, 
```
/*
[in]   idx:  the last group of scalar indexes.
[in]   wval: the last group of sub-scalars. 
[out]  idx_out:  the corresponding scalar indexes to the selected sub-scalars.
    (allocated in the GPU global memory with 'nscalars' elements)
[out]  s_count: the number of selected sub-scalars.
*/
uint32_t ptr = (NWINS - 1)  * nscalars;
cub::DeviceSelect::Flagged(tmp, tmp_size, idx + ptr, wval + ptr, idx_out + ptr, s_count, nscalars);
```

10. We launch (NWINS - 1) * N * NTHREADS threads to add BLS12-377 points whose corresponding sub-scalars are equal to 1, 
where '(NWINS - 1) * N * NTHREADS' is twice the total number of GPU cores. N = 2 * Cores / ((NWINS - 1) *  NTHREADS).
'NTHREADS' represent the number of threads in a block. 

Specifically, 
```
#define NTHREADS 32
static __shared__ affine_t spoints[NTHREADS]; 

/*
[in]  points: BLS12-377 points.
[in]  npoints: the number of BLS12-377 points.
[in]  s_count: the number of selected sub-scalars.
[in]  idx_out: the corresponding scalar indexes to the selected sub-scalars.
[out] tres:  the calculation results of all threads individually.
    (allocated in the GPU global memory with '(NWINS - 1) * N * NTHREADS' elements)
*/
pippenger_faster_4<<<dim3(NWINS - 1, N), NTHREADS>>>(points, npoints, s_count, idx_out, tres);

__global__
void pippenger_faster_4(affine_t* points, size_t npoints, uint32_t *s_count, uint32_t *idx_out, bucket_t* tres)
{
    /* We divide threads into (NWINS - 1) groups, with (N * NTHREADS) threads in each group. */
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
        uint32_t idx = idx_out[ptr + i];
        spoints[threadIdx.x] = points[idx];
        mid.add(spoints[threadIdx.x]);
    }
    uint32_t bptr = bid * tnum;
    tres[bptr + tid] = mid;
}
```

11. We reduce the calculation results of all threads to the subtask result (the last subtask). 

Specifically, 
```
/*
[in]  tres: the calculation results of all threads individually.
[out] res: the result of the last subtask.
*/
launch_coop(LSum, dim3(NWINS-1, N / 2), NTHREADS, stream, res, fres);

template<typename... Types>
inline void launch_coop(void(*f)(Types...), dim3 gridDim, dim3 blockDim, cudaStream_t stream, Types... args)
{
    void* va_args[sizeof...(args)] = { &args... };
    CUDA_OK(cudaLaunchCooperativeKernel((const void*)f, gridDim, blockDim, va_args, 0, stream));
}

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
        for(uint32_t i=0; i < NWINS - 1; i++) res[NWINS - 1].add(tres[i * tnum * 2]);
    }
}
```

After all subtasks have been performed,

12. We accumulate results of subtasks to the final result. Note that this step can be performed in CPU serially.

Specifically, 
```
/*
[in]  res: results of all subtasks.
[out] out: the final result.
*/
void accumulate_faster(point_t &out, result_container_t_faster &res) {
    out.inf();
    for(int32_t k = NWINS - 1; k >= 0; k--)
    {
        for (int32_t i = 0; i < WBITS; i++) out.dbl();
        point_t p = res[k];
        out.add(p);
    }
}
```


If there are any questions, please contact Tao Lu: lutaocc2020@gmail.com