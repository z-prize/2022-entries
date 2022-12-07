# Z-Prize MSM on the GPU Submission

## Introduction

The following is Yrrid Software's GPU MSM submission to the Z-Prize.   The submission is available from
our [GitHub repository](https://github.com/yrrid/submission-msm-gpu).

## Full Run Performance

We generally observe the running time to be between ***2500 milliseconds*** and ***2600 milliseconds*** for a full run 
of 4 MSMs of size 2^26.  Performance is quite dependent on the other workloads running on the same physical machine.

## Building and running the submission

Install [CUDA 11.7](https://developer.nvidia.com/cuda-downloads).   Install rust, for example, `rustup install stable`. 
Next clone the repository, and run the benchmark.
```
git clone https://github.com/yrrid/submission-msm-gpu
cd submission-msm-gpu
cargo bench
```

To run the correctness test, use the following command.   Note, the util.rs routines generate a small number of points 
and copy them many times to generate the 2^26 points needed for a run, so that benchmarking and correctness testing can 
be performed in a reasonable amount of time.
```
cargo test --release
```

## GPU requiments

Since this is a competition and every millisecond counts, the software has been tuned to run a batch of 4 x 2^26 MSMs on the target GPU,
an NVIDIA A40.  Our solution requires Compute Capability 8.0 (Ampere) and roughly 46 GB (46 x 2^30 bytes) of memory.

## Optimizations in our solution

In this section, we give a high level overview of the optimizations we have used to accelerate the computation:

-  Pippenger bucket algorithm with a 23-bit window.
-  Signed digits.  Since 11 windows of 23-bits is exactly 253 bits, we employ the following trick.  If the MSB of 
   the scalar is set, we negate the scalar and point, and continue processing normally.   This works since:  
    &nbsp;&nbsp; *(M - s) (-P)* = *-s (-P)* = *s P*
-  Pre-process all of the scalars to generate lists of points to add to each bucket.
-  The buckets are then sorted, such that the buckets with the most points are run first.  This allows the GPU warps to run convergent workloads and 
   minimizes the tail effect.
-  For an input point Pi, we precompute 6 points: 2<sup>46</sup> Pi, 2<sup>92</sup> Pi, 2<sup>138</sup>, ..., 2<sup>230</sup> Pi.   This allows us to
   compress our 11 windows down to 2 windows, since, for example, adding Pi to window 4 is the same as adding 2<sup>92</sup> Pi to window 0.
-  The pre-processing sorting routines are custom written and are very fast and efficient.  Much faster than CUB based solutions.
-  The FF and EC routines have been carefully optimized:  
   - Based on Montgomery multiplication 
   - Minimize correction steps in the FF operations
   - Use an XYZZ representation for the EC point accumulators
   - Use fast squaring

## Questions

For technical questions about this submission, please contact `nemmart at yrrid.com`.
