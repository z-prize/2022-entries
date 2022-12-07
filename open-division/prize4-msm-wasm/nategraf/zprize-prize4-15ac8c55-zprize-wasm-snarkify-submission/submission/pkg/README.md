# `WASM` Z-prize challenge proposal (Draft)

## Introduction
WASM (WebAssembly) is the de-facto standard for smart contact VM like Polkadot, Definity, Cosmos. And also critical for wallet adoption. However, making ZKP works efficiently on WASM is still a challenge today. In Manta’s internal benchmark, we can observe 10x - 15x performance penalty on WASM compared with X86 native binary. This WASM ZKP challenge is bring breakthroughs in compilation to make ZKP on WASM (both prover and verifier)

Goal: Test the correctness and performance of WASM binaries on some operations that are common in ZKP systems.

(Note: Bls12-381 can be replaced with any ZK friendly curves)

In particular, we consider three types of test suites:
* Low-Degree Extension: Measure the performance of (I)FFT
* Product of Pairings: Measure the performance of billinear pairing
* Multi-Scalar Multiplication: Measure the performance of scalar multiplication

Please check detailed documents at our [proposal](https://hackmd.io/@tsunrise/rJ5yqr4Z5/edit).

## Dependencies:
* [Rust toolchain](https://www.rust-lang.org/tools/install)
* [npm](https://www.npmjs.com/get-npm)
* `wasm-pack` package:
    ```bash
    cargo install wasm-pack
    ```

## Run the benchmark

* WASM time:
    ```bash
    ./serve.sh
    ```
    You can view the result at `localhost:8080`.
    Please update [this line](https://github.com/Manta-Network/wasm-zkp-challenge/blob/main/www/index.js#L79-L86) to benchmark different test suites.

* Native time:
    ```bash
    cargo bench
    ```

### Benchmarking notes

* On Linux, more stable benchmarking results can be achieved by increasing the execution pririty of
    the benchmark via nice. Here is an example command to do that for native CPU profiling.
    ```bash
    nice -n -15 cargo bench
    ```
* It's possible to get pprof profile outputs by running the benchmarks with the following command.
    ```bash
    # Run 30 seconds of profiling for each benchmark target. Use nice to increase priority.
    nice -n -15 cargo bench --bench bench_pippenger_msm -- --profile-time 30
    ```
    Profile outputs will be stored at `./target/criterion/msm/$FUNCTION/$INPUT_SIZE/profile/profile.pprof`
    Profiles can be opened with the pprof tool, which can be installed and run with
    ```bash
    go tool pprof -http localhost:8080 target/criterion/msm/$FUNCTION/$INPUT_SIZE/profile.pprof
    ```
* Especially when trying to meausre smaller effects, 100 samples may not be enough. Number of
    samples can be increased using the `--sample-size` flag.
    ```bash
    nice -n -15 cargo bench --bench bench_pippenger_msm -- --sample-size 1000
    ```
Note: SUPERCOP, which is a cryptography benchmarking suite, has useful recommendations on reducing
randomness in benchmarking results, mostly focused on randomness due to clock rate variability.

https://bench.cr.yp.to/supercop.html
