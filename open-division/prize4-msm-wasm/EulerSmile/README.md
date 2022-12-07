# Accelerating Elliptic Curve Operation and Finite Field Arithmetic (WASM)

  

Prize Sponsor: Polkadot Pioneer Prize

Prize Architect: Manta Network

## Prize Description

### Summary

Multi-Scalar multiplication (MSM) operations are essential building blocks for zk computations. This prize will focus on minimizing latency of these operations on client-type devices and blockchain-based VMs, specifically the WebAssembly (WASM) runtime.

### Optimization Objective

Achieve the lowest combined latency of MSM over a range of input vector lengths in WASM runtimes.

### Constraints

We focus on variable-base MSM [1], which is widely used in zkSNARKs. Formally, variable-base MSM takes an input vector of elliptic curve points

![](https://lh5.googleusercontent.com/JYhSGXPDFH-_2Gf-T9aivPpeq0C8uuiCKhjRg8zYxie0in0RjIht8OZp6zq09V0xvq9i4cUWDqBHAKRiQu-XtevebIE7TyRPgzhrz-LApfuPpdVj6dRjFYUkSmOHT2u2sdXz5vUV_BmuIasSWg)

and an input vector of finite field elements from the associated scalar field

![](https://lh5.googleusercontent.com/HtnrKsnZkLwU2FUXn_AOkweUS2f4w1ZR65gy_RfYjlaZScZUpwO4oN2Jps3NGcB7t6cYVDTK7KLcXSB6AWRvoYjEdwveuNNDnx8FEQrpUrj3gx-uDPwLfDLJDqrVj8e67TLGt9rI0f87uN75CQ)

Here, n is the input vector length (i.e., the number of elements in a vector).

The output is an elliptic curve point Q:

![](https://lh4.googleusercontent.com/bHfO9zAMT5fiNqykRuWecAz2c-Fwa81AXRcIIMKLtr9-nVBCPHCrMY3ZIj2QztrsutBS8t5vawZ857VkCBpY_Qtm4fIPAW_EGmsEk5QS5TNGpLtfFI8zbcCgccRqJBZHYQpzKLYRuEFmcToogA)

-   The implementation must provide the following interface in JavaScript [2]: `compute_msm(point_vec, scalar_vec)`
    

1.  The function name is `compute_msm`
    
2.  There are two input vectors: point_vec for a vector of elliptic curve points and scalar_vec for a vector of finite field elements.
    
3.  The output is a single elliptic curve point.
    

-   The implementation can be constructed either in high-level language (Rust, C, C++, Javascript) or manually written WebAssembly.
    
-   The submitted WASM module can be run in Chrome 96, Firefox 90, Safari 15.2, Wasmtime 0.33.
    
-   Only the following WASM features are allowed: “JS Bigint to Wasm i64 integration”, “Bulk memory operations”, “Multi-value”, “Import & export of mutable globals”, “Reference types”, “Non-trapping float-to-int conversions”, and “sign-extension operations”. We allow these features since they are currently implemented in popular engines [3].
    

  

![](https://lh6.googleusercontent.com/tewO232Raz5bODrFASIztBaflz41bWGiH0ILl8W82SJSmgotMb2VmJUsQguUACGvj6ej3FdwLGQ-KcAkPAPCsQ8yPDt3hRcCI6Yr0jU6FyAPQ7fB0vAdLLkWYaBNR1wxAZKR2vsrMni-22kZVw)

  

-   The MSM must be over the BLS12-381 G1 curve.
    
-   The submission should produce correct outputs on input vectors with length up to 2^18. The evaluation will be using input randomly sampled from size 2^14 ~ 2^18, which is the range that we find covers most use cases.
    
-   The submissions will be evaluated in single-threaded runtime without allowing OpenCL feature. This makes the submission more applicable for targeted execution environments, such as a wasm module in metamask-snap.
    
-   All submissions must include documentation (in English) sufficient to understand the approach being taken.
    

## Timeline

June 10 - Competition begins

July 25 - Mid-competition submission due

September 10 - Final submission due

## Judging

Submissions will be analyzed for both correctness and performance.

### Correctness

We will provide a set of test input/output pairs so that the competitors can sanity check the correctness of their code.

  

The final correctness of the submission will be tested using randomly sampled test inputs/outputs that are not disclosed to the competitors during the competition in addition to the test input/output distributed to the competitor. All these test cases will be generated using Arkworks reference implementation. Submissions failed in any test cases will be judged as incorrect and lose the opportunity to win the prize.

### Performance

To evaluate the performance of each submission, the prize committee will sample a large number (N > 200) of input vectors at random, in terms of both the input vector length and the point values. The input vector length will be sampled between 2^14 and 2^18. Then, we compute the multi-scalar multiplication using the submission. The submitted score will be the relative speedup from arkworks implementation, measured across 100 trials. More specifically, we compute the submitted score as follows:

  

Given N randomly selected input vectors, we measure the latency of baseline (i.e., Arkworks) as

![](https://lh5.googleusercontent.com/1xdVDKUSlg9FzyXJ5gEpp4FhnndIHA3QHqbDP8_pl3V8umz_TB6vy5zTAWq5kcCtpmRI316GnGrKHMlsmozx_zKZRWBpR26Ny96JAJBiv-KZoM73YSA98iDSVW213dsGLGPbVyaNOAVZNzcuKg)

We measure the latency of submission as

  

![](https://lh4.googleusercontent.com/ZCMpEaePN6ropQsIOjOohohht7KKkz7jLNpJ0fQ_Si6zRJjIlPQUflYmd3a_y6RwAp1LhRtiuqk3mmIf0I5f9cFNmfVNVQT8q0oAfXLl5R15YrHNRe_x4IejZFWBYNJJ6rb0FQW17BpPWAqGVg)

  

We compute the submitted score as

  

![](https://lh5.googleusercontent.com/A4zlOZtwgZm-RNPCo8khke1t4k_dRsnMhRRiUNkjq71Lzj_vBGhY0CzaDHDiiDb8U7spPmJsVPINJK_ip3eF5r_zRm9AkB6BQdAr-WtORnLu5eT_nWCX6Mi_H_ECmYmltD-3EmuqNNB8-EPv3A)

  

Intuitively, the submitted score represents the relative speedup from baseline over a range of input vector lengths.

  

In addition, all submissions will be manually reviewed by the prize committee.

### Hardware & Benchmarks

-   Competitors will be provided with access to a Coreweave-provided virtual workstation: consisting of an NVIDIA Quadro RTX 4000 with 8GB of GDDR6 RAM and 5 vCPU cores with 30 GB of CPU RAM.
    

  

-   The baseline will be the Arkworks MSM implementation over BLS12-381 G1. This baseline is originally implemented in Rust. We compile the rust implementation to WASM runtime (through wasm-pack 0.10.2) as the baseline. Submissions must beat this baseline by at least 10% in order to be eligible for the prize.
    

## Prize Allocation

The prize amount will be divided among the top three finishers according to the following proportions: 65% to winning implementation, 25% to second place, and 10% to third place.

  

In the event that there are only two qualifying submissions, first place will receive 70% of the prize pool and second place 30%. In the event there is only one qualifying submission, they will receive 100% of the prize pool.

  

Prizes will be given out in good faith and in the sole discretion of the prize committee.

## Notes

  

All submission code must be open-sourced at the time of submission. Code and documentation must be dual-licensed under both the MIT and Apache-2.0 licenses

  

Why BLS-12-381?

1.  BLS12-381 curve is used in BLS signature [4]
    
2.  For Poseidon hash, BLS12-381 is more efficient than BLS12-377. Poseidon hash is widely used in many industry products such as Filecoin and Zcash. The performance of Poseidon hash is critical. To make Poseidon hash secure, BLS12-377 needs around 66% more computation and latency than BLS12-381.
    

## Submission Instruction

Please include your implementation under the `submission` folder and update `evaluate.sh` correspondingly. You may also adapt `submission_compute_msm` function in the `www/index.js` file.

## Questions

If there are any questions about this prize, please contact boyuan@manta.network

## Reference

[1] Scalar-multiplication algorithms. [https://cryptojedi.org/peter/data/eccss-20130911b.pdf](https://cryptojedi.org/peter/data/eccss-20130911b.pdf)

[2] compute_msm interface in Javascript. [https://github.com/Manta-Network/wasm-zkp-challenge/blob/main/www/index.js#L22](https://github.com/Manta-Network/wasm-zkp-challenge/blob/main/www/index.js#L22)

[3] Web Assembly Roadmap. [https://webassembly.org/roadmap/](https://webassembly.org/roadmap/)

[4] BLS Signature. [https://datatracker.ietf.org/doc/draft-irtf-cfrg-bls-signature/04/](https://datatracker.ietf.org/doc/draft-irtf-cfrg-bls-signature/04/)
