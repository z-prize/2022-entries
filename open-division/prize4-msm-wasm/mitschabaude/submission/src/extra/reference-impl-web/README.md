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

## Initial Results

### Platform
Intel i7-6560U CPU @ 2.2GHz, 8GB Memory, Linux 16.04 LTS.

### FFT Results

|Input Vecotr Length | Output Vector Length | WASM (ms) | Native (ms) | Ratio |
| --- | --- | --- | --- | --- |
| 2^14 | 2^15 | 111 | 18 | 6.2 |
| 2^16 | 2^17 | 496 | 86 | 5.8 |
| 2^18 | 2^19 | 2373 | 450 | 5.3 |
| 2^20 | 2^21 | 9912 | 2118 | 4.7 |

We note that, for the same input vector length, the output vector length also has impact over the latency. Please see below:

|Input Vecotr Length | Output Vector Length | WASM (ms) | Native (ms) | Ratio |
| --- | --- | --- | --- | --- |
| 2^14 | 2^15 | 111 | 18 | 6.2 |
| 2^14 | 2^16 | 190 | 32 | 5.9 |
| 2^14 | 2^17 | 358 | 62 | 5.8 |
| 2^14 | 2^18 | 773 | 129 | 6.0|

### MSM Results

|Input Vecotr Length | WASM (ms) | Native (ms) | Ratio |
| --- | --- | --- | --- |
| 2^8 | 478 | 14 | 34.1 |
| 2^10 | 1627 | 38 | 42.8 |
| 2^12 | 6191 | 125 | 49.5 |
| 2^14 | 24243 | 393 | 61.7 |

### Pairing Results

|Input Vecotr Length | WASM (ms) | Native (ms) | Ratio |
| --- | --- | --- | --- |
| 2^2 | 97 | 7 | 13.9 |
| 2^4 | 368 | 30 | 12.3 |
| 2^6 | 1479 | 121 | 12.3 |
| 2^8 | 5916 | 485 | 12.2 |
