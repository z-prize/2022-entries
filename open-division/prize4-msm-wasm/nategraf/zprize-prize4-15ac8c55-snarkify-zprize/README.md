# Snarkify WASM MSM Submission

This is the root repository for the Snarkify submission to the MSM on Wasm ZPrize.

Note: This repository should be cloned with `git clone --recursive` or use `git update --init
--recursive` after the repository has been cloned, as the relevant code is contained in submodules
of this repository.

## Build

Building the optimized WASM module, and installing it into the official test harness can be
accomplished with the following tests. Note that the test harness repository already has a built
Wasm module checked in, so these steps are only to reproduce the same results.

```bash
cd wasm
wasm-pack build --release
cd -
cp -R wasm/pkg official-test-harness/submission/pkg
```

## Run

Running a benchmark of the Wasm module can be accomplished in either the `official-test-harness`
submodule or the `wasm` submodule. See the instructions in those repositories for more information.

## Optimizations

Our optimization approach was to start with the latest version of Arkworks, taken from `master` when
starting this project, and to optimize it with a number of techniques, in order of importance.

* We implemented in Rust a bucketized Pippenger's algorithm inspired by Aztec's implementation of
    this technique. In particular, we utilized the batch-affine formula in an addition tree for
    accumulation of the buckets.
* We identified the LLVM compilation of u128 multiplication, used in Arkworks for full-width u64
    multiplication, in Wasm to be problematic. We wrote a full-width u64 multiplication routine
    avoiding usage of the u128 data type.
* We tuned the bucket sizes to optimize for performance in Chrome on our testing machine, which is
    an 8-core Intel 11-gen i7.
* We eliminated the need for a conditional subtraction after Montgomery multiplication in the
    finite-field implementation by allowing the values to be in the range [0, 2q) instead of [0, q).
    Combined with a change to simplify the is-zero check for group elements, this improved
    performance.
* We implemented the no-carry optimization for finite field squaring as described by the gnark team
    https://hackmd.io/@gnark/modular_multiplication.

By combining these changes, we see a speedup of 2.4x in Chrome and 1.8x in Firefox measured on a
2018 x86 MacBook Pro.

Additional information about some of these techniques can be found in our notes at
https://www.notion.so/victorgraf/Known-Optimizations-for-MSM-13042f29196544938d98e83a19934305
