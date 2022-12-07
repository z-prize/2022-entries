# Accelerating Elliptic Curve Operation and Finite Field Arithmetic (WASM)

## Prize Description

### Summary

Multi-Scalar multiplication (MSM) operations are essential building blocks for zk computations.
This prize will focus on minimizing latency of these operations on client-type devices and
blockchain-based VMs, specifically the WebAssembly (WASM) runtime.

## Snarkify Submission

This repository is a fork of the official MSM on WASM test harness.

A compiled submission package is provided in the `submission` directory.
Source code for the submission, along with build instructions and a detailed description of the
approach, can be found in the [snarkify-zprize repository](https://github.com/nategraf/snarkify-zprize)
(of which this repository in a submodule).

### Evaluation Instructions

As with the origonal test harness, running `./evaluate.sh` will start a web server for a page
including both reference and submission. Pointing a browser to this server will automatically run
the benchamrks on a randomized inputs.
