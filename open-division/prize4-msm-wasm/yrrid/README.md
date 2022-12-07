# WASM MSM Z-prize challenge

## Introduction

The following is Yrrid Software's WASM MSM submission to the Z-Prize.  The submission is available from our
[GitHub repository](https://github.com/yrrid/submission-wasm-msm).

## The Goal

Improve the performance of WASM based MSM on the BLS12-381-G1 Curve.

We have tested our solution on the provided CoreWeave instance, an EPYC 7443P with 24 cores.  Our solution
uses only a single core, as required by the competition specifications.  Performance seems to vary somewhat 
from instance to instance, but on our instance we observe the following run times:

|Input Vector Length | Our WASM Submission (milliseconds) | 
| --- | --- |
| 2^14 | 326 |
| 2^15 | 613 |
| 2^16 | 1180 | 
| 2^17 | 2308 | 
| 2^18 | 4257 | 

NOTE: The solution has been tuned for an EPYC 7443P processor, and would need to be re-tuned to achieve optimum 
performance on other processors.

## Submission Authors

This solution was developed by Niall Emmart, Sougata Bhattacharya, and Antony Suresh.  In addition, we would like 
to thank Kushal Neralakatte for his insights and some algorithmic suggestions.

## Dependencies

1. Rust ark-bls12-381 and dependent libraries for generating the input vectors.
2. Wasm binary generated for MSM using C code base with clang toolchain.

## Installation of Rust Toolchain
* [Rust toolchain](https://www.rust-lang.org/tools/install):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```
* [npm](https://www.npmjs.com/get-npm)
* `wasm-pack` package:
    ```bash
    cargo install wasm-pack
    ```

## Running the benchmark

Open a shell on the instance and run the following command:

```
./evaluate.sh
```

Next, open a browser with the URL http://localhost:8080.  Alternatively, on a headless machine (such as the
CoreWeave instances), you can try the following:

```
cd bench
./bench.sh
```

This uses nodejs to emulate the browser and run the MSM code.  

The default run is quite small, with just 64 scalars and points.  The sizes can be changed by editing `www/index.js`.
Note, you must re-run `./evaluate.sh` after changing the index.js file.

## Optimizations in our Solution

In this section, we give a high level overview of the optimizations we have used to accelerate the MSM computation:

-  Pippenger bucket algorithm, supporting window lengths 10, 11, 12, 13, 15, and 16 bits.
-  We use the cube root of one endomorphism to decompose each scalar into two 128-bit scalars.
-  We use signed digits.  Note, for the window length 16, we evenly divide the 128-bit scalar
   values into 8 windows.  Scalars with the MSB set exploit the following trick: we negate the
   scalar and point and continue processing normally.  This works since:  
    &nbsp;&nbsp; *(M - s) (-P)* = *-s (-P)* = *s P*
-  We use Aztec's trick of affine point addition with a batch inversion for both the bucket
   summation and bucket reduction (the sum of sums algorithm).
-  We do not do any preprocessing or sorting of the scalars, instead, we use a simple bit
   array to determine if a bucket is already the target of a point addition.  In the event of 
   a collision, we add the point and bucket pair to a linked list for later processing.  When we 
   have processed 1024 unique buckets, or reached 128 collisions, we do the batch invert and 
   second phase of the point adds.   We repeat this process until all the scalars have been 
   processed.
-  The FF and EC routines have been carefully optimized:
   - Based on Montgomery multiplication.
   - We use a redundant representations to minimize the number of correction steps in
     the FF and EC routines.   
   - Use 30 bits limbs.  This is important since we can accumulate sixteen 30-bit * 30-bit 
     values in a 64-bit register without having to worry about overflow.
   - Special multiplication tricks:  use fast squaring.
   - Use one level of Karatsuba.
   - Highly optimized field inverse algorithm and implementation.

## Compiling from the C Source Code

The compiled MSM binary has been provided in the repository but can be built from the source using Clang version 14.0.0-1ubuntu1.
The build can be initiated by deleting `submission/submission.wasm` and running `./evaluate.sh`.   The compilation process verifies
the checksum of the resulting binary before proceeding, since compiling with the wrong version of clang could potentially impact
performance in unexpected ways.

## Questions

For technical questions about this submission, please contact `nemmart at yrrid.com`.
