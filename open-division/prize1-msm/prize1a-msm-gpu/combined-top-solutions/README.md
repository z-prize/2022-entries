# Z-Prize MSM on the GPU 

## Introduction

Matter Labs and Yrrid have joined forces to produce an improved version of MSM for the GPU that takes the best ideas
and implementations from our separate MSM for the GPU submissions to the ZPrize.   Our improved solution is about 10% 
(240 ms) faster than either of our ZPrize submissions.   The improvements break down as follows:
- Yrrid's custom scalar/point sorting routines (which are faster than CUB)
- Yrrid's improved scalar preprocessing/windowing algorithms
- Matter Lab's significantly faster FF and EC routines in the bucket accumulation phase (saves 150 ms)
- Matter Lab's idea to break the first set of 2^26 scalars into pieces, which allows for copy and compute overlap (saves 90 ms)
- We use the main driver routines from Yrrid's solution 

The combined, improved solution is available from two github locations: [Matter Labs](https://github.com/matter-labs/z-prize-msm-gpu-combined)
and [Yrrid Software](https://github.com/yrrid/combined-msm-gpu).

## Source Directory Structure

| Directory                 | Description                   |
|---------------------------|-------------------------------|
| combined-msm              | Top level kernel sources      |
| combined-msm/ml-ff-ec     | Matter Lab's FF & EC routines |
| combined-msm/yrrid-ff-ec  | Yrrid's FF & EC routines      |

## Full Run Performance

We generally observe our running time to be between ***2200 milliseconds*** and ***2300 milliseconds*** for a full run
of 4 MSMs of size 2^26. Performance is somewhat dependent on other workloads running on the same physical machine and
there is some GPU to GPU variation.

## Building and running the submission

Install [CUDA 11.7](https://developer.nvidia.com/cuda-downloads). Install rust, for example, `rustup install stable`.
Next clone the repository, and run the benchmark.

```
git clone https://github.com/matter-labs/z-prize-msm-gpu-combined
cd z-prize-msm-gpu-combined
cargo bench
```

To run the correctness test, use the following command. Note, it can take several hours to generate the input points for a full 2^26 run.

```
cargo test --release
```

## GPU requiments

We adopt the same GPU requirements as the ZPrize competition.  The software has been tuned to run a batch a batch of 4 x 2^26 MSMs on the 
target GPU, an NVIDIA A40. The solution requires Compute Capability 8.0 (Ampere) and roughly 46 GB (46 x 2^30 bytes) of memory.

## Optimizations in our solution

In this section, we give a high level overview of the optimizations we have used to accelerate the computation:

- Pippenger bucket algorithm with a 23-bit window.
- Signed digits. Since 11 windows of 23-bits is exactly 253 bits, we employ the following trick. If the MSB of
  the scalar is set, we negate the scalar and point, and continue processing normally. This works since:  
  &nbsp;&nbsp; *(M - s) (-P)* = *-s (-P)* = *s P*
- Pre-process all of the scalars to generate lists of points to add to each bucket.
- The lists are further sorted by the number of points in each bucket. This allows the GPU warps to run convergent workloads.
- For an input point Pi, we precompute 6 points: 2<sup>46</sup> Pi, 2<sup>92</sup> Pi, 2<sup>138</sup>, ..., 2<sup>230</sup> Pi. This allows us to
  compress our 11 windows down to 2 windows, since, for example, adding Pi to window 4 is the same as adding 2<sup>92</sup> Pi to window 0.
- The pre-processing sorting routines are custom written and are very fast and efficient. Much faster than CUB based solutions.
- The FF and EC routines have been heavily optimized:
    - Based on Montgomery multiplication
    - Minimize correction steps in the FF operations
    - Use an XYZZ representation for the EC point accumulators
    - Use fast squaring
    - Take advantage of `reduce(a*b) +/- reduce(c*d)` = `reduce(a*b +/- c*d)` to save reduction steps.
- We break the first MSM of size 2^26 into two MSMs of size 2^24 and 3*2^24. This allows us to hide much of the PCIe copy time for the
  first set of scalars.

## Questions

For technical questions about our submission, please contact `rr at matterlabs.dev` and `nemmart at yrrid.com`.
