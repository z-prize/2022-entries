# Snarkify WASM MSM Submission Rust Crate

## Build

WASM package suitable for use with Webpack can be built with:

```bash
wasm-pack build --release
```

Resulting package will be available in the `pkg` directory including a Wasm module and JavaScript
and TypeScript bindings. This package can be imported into Webpack applications.

## Dependencies:

* [Rust toolchain](https://www.rust-lang.org/tools/install)
* [npm](https://www.npmjs.com/get-npm)
* `wasm-pack` package:
    ```bash
    cargo install wasm-pack
    ```

## Benchmark

### Wasm in Browser

```bash
./serve.sh
```

Opening https://localhost:8080 in your browser will give a page where you can provide input test
vector files to run. You can generate input files with:

```bash
cargo run --bin generate-input-files
```

### Native

```bash
cargo bench
```

Note: Default features in this crate are intended for Wasm targets. If looking to run optimally on
native, turn off the default features for compilation.

### Benchmarking notes

* On x86 Linux, more stable benchmarking results can be achieved by increasing the execution pririty of
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
