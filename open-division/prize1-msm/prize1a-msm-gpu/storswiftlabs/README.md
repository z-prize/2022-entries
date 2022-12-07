# ZPrize MSM on GPU Reference Test Harness

Example and reference testing harness for the ["Accelerating MSM on GPU"](https://github.com/z-prize/prize-gpu-fpga-msm) challenge in ZPrize 2022.

## Build
The CUDA [`sppark`](https://github.com/supranational/sppark) from Supranational has a Rust binding. To install the latest version of Rust, first install rustup. Once rustup is installed, install the Rust toolchain by invoking:

```
rustup install stable
```
After that, use cargo, the standard Rust build tool, to build the libraries:

```
git clone https://github.com/z-prize/test-msm-gpu.git
cd test-msm-gpu
cargo build --release
```

## Test
To run a test of an MSM of `2^15` random points and scalars on the BLS12-377 curve, run:

```
cargo test --release
```

## Bench
This repository specifies default features corresponding to the ZPrize competition. To run the target benchmark of the competition (`2^26` random points on BLS12-377), run:

```
cargo bench
```
To specify the window size in the Pippenger algorithm, pass the flag `CXXFLAGS="-D WBITS=17"`. For BLS12-377, a window of size 17 seems to give the best results but it is up to competitors to specify it.
