# ZPrize MSM on GPU Mike and Alex submission

We can consider implemented Pippenger algo as a five-step algo:  
0. Pre-computation
1. Distribution of points and scalars into buckets
2. Compute partial sums into each buckets
3. Compute sum of partial sums
4. Compute the final sum (on CPU)

Comparing to the baseline the major changes are in 1st and 2nd steps.    

The 1st step is a usual one when each scalar is divided into group of `wb` bits and all `2^26` scalars are distributed into buckets. In our approach we distribute points into buckets before running the 2nd step (comparing to the baseline where concurrent updatable table is used). The distribution algo is based on a histogram computing (similar to what usually used in Radix sort) for offsets and then distribute points indices into such buckets.  

On this step we also exploit ability to cheap negation of points in elliptic curves. This allows use to shift a scalar part into `[-2^(wb-1), 2^(wb-1)]` range by subtraction `2^wb` from a scalar part bugger then `2^(wb-1)-1`. In this case we store negative point index into corresponding bucket and propagate `1` into the next scalar part. Potentially a scalar could overflow, but since scalar is 253 bits and are stored into 256 bit limbs this overflow is painless. The result of this is that a count of buckets for one window is reduced in two times (`2^(wb-1)-1` instead of `2^wb - 1`) allowing to reduce number of sums on the 3rd stage.  

On the 2nd step we compute partial sums in parallel with by tiles almost as well as the baseline. The main difference is more careful algorithm of waves separation. Also, we compute the latest window of each scalar group on CPU simultaneously with GPU, because the latest group of scalars contains much less elements then others.  

The 3rd step is always done on CPU in parallel with the next wave on the 2nd step (except the last wave).  

The 4th step is the same as in the baseline.

Tested ideas:
 - NAF - signed scalars are obviously better, because NAF reduces number of buckets into 2/3 times, but singned scalars in 2 times
 - NAF + signed scalars - the main drawback of this variant is that a count of sums on the 2nd step twice more
 - Karatsuba multiplication + Barrett reduction - turned out that the CIOS baseline Montgomery is better
 - Affine summation + the Montgomery trick - turned out to be slower than the baseline summation
 
(code for the most of these trials could found in sppark/msm/_pippengerN.cuh files)
