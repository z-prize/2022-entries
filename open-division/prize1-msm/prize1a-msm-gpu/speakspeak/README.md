# A GPU Implemetation of MSM for ZPrize Competition

This library is a GPU Implemetation of MSM for ZPrize Competition. 

Following [the ZPrize specification](https://assets.website-files.com/625a083eef681031e135cc99/6314b255fac8ee1e63c4bde3_gpu-fpga-msm.pdf),
this library is specifically designed for the fixed-base point MSM with 2^26 randomly sampled scalars from the BLS12-377 scalar field. 
Due to the deadline of the competition, GPU implemetations for MSM on other elliptic curves will be released later.

## Author and License

This GPU implemetation of MSM is developed by Tao Lu with the template from [the reference](https://github.com/z-prize/test-msm-gpu) provided by [ZPrize](https://www.zprize.io/).

Copyright (c) 2022 Tao Lu and contributors to [the reference](https://github.com/z-prize/test-msm-gpu).

This library is licensed under the Apache License Version 2.0 and MIT licenses.

## Contact

If there are any questions about this library, please contact Tao Lu: lutaocc2020@gmail.com

## Make MSM faster

Here, we use a new faster parallel Pippenger-based MSM algorithm, which is equivalent to the one from our paper "cuZK: Accelerating Zero-Knowledge Proof with A Faster Parallel Multi-Scalar Multiplication Algorithm on GPUs". The full version of this paper will be published soon.

Details of our MSM implementation are described [here](./doc.md) (doc.md).  

## Result

We evaluate our implementation on a dedicated instance of baseline image consisting of an AMD Epyc Milan CPU (8 cores), an A40
NVIDIA GPU with 48 GB of RAM.

Given input vectors consisting of 2^26 fixed elliptic curve points (bases), we compute a
scalar multiplication of those bases with a set of four vectors of randomly sampled scalar elements from the associated
BLS 12-377 G1 field in succession.

| Ours            | [Reference](https://github.com/z-prize/test-msm-gpu)  |Speedup |
| :---:           | :-----:                                               |:-----:  |
| 3.38 s          | 5.65 s                                                | 1.67x |

## Build
Our GPU Implemetation of MSM relies on [`CUB`](https://nvlabs.github.io/cub/), which provides state-of-the-art, reusable software components for every layer of the CUDA programming model.

By default, CUB is included in the CUDA Toolkit. 
If there is no CUB after installing the CUDA Toolkit, it is no need to build CUB separately. CUB is implemented as a C++ header library. To use CUB primitives in your code, simply:
1. Download and unzip the latest CUB distribution
2. #include the <cub/cub.cuh> header file in your CUDA C++ sources. 
3. Compile your program with NVIDIA's nvcc CUDA compiler, specifying a -I<path-to-CUB> include-path flag to reference the location of the CUB header library.

The CUDA [`sppark`](https://github.com/supranational/sppark) from Supranational has a Rust binding. To install the latest version of Rust, first install rustup. Once rustup is installed, install the Rust toolchain by invoking:

```
rustup install stable
```
After that, use cargo, the standard Rust build tool, to build the libraries:

```
git clone https://github.com/speakspeak/zprize-msm-gpu.git
cd zprize-msm-gpu
cargo build --release
```

## Test
To run a test of an MSM of `2^26` random points and scalars on the BLS12-377 curve, run:

```
cargo test --release
```

## Bench
To run the fixed-base point MSM (`2^26` random points on BLS12-377) with four randomly selected test scalar vectors across ten trials in total, run:

```
cargo bench
```