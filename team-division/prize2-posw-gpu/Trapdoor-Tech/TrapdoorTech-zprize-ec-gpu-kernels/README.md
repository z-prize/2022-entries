# Elliptic Curve GPU Kernel

## Usage

specify generating cuda/opencl kernel:

``` bash
# for cuda kernel
$ cargo run -- --cuda

# for opencl kernel
$ cargo run -- --opencl
```

## Running a full TPS test

1. generate cuda kernel for the test, using above commands
2. prepare all the files needed and copy them to snarkVM folder: `lagrange-g-calced.dat`, `gpu-kernel.bin`
3. generate precalc table by specifying window size: `cargo run --bin generator --features "table_generator" --release -- --window-size 8`
4. start ws_server process in another window: `cargo run --bin test_ws_server --release`
5. test TPS: `WORKER_NUM=16 cargo run --bin snarkvm_worker --release --features "worker_debug" 2>/dev/null | grep "height = "`
