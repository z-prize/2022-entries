Zprize WASM competition submission
==================================

The submission is based on the [provided test harness](https://github.com/z-prize/prize-wasm-msm.git).

It is written in Go, and derived from the [`gnark-crypto`](https://github.com/consensys/gnark-crypto) project. Even though the `bls12-381` package is audited, the submission in this repository is manually optimised and uses different algorithmic tricks compared to `gnark-crypto`.

Results
-------

The following benchmarks were run on the given CoreWeave instance using puppeteer with the locally available Google Chrome instance (version 103.0.5060.53). Chrome was run with the arguments:

```
google-chrome --headless --disable-dev-shm-usage --remote-debugging-port=7777 --enable-logging --v=1
```

The output results were:

```
Correctness check passed.

Submission performance.
Input vector length: 2^14, latency: 1416.5999999046326 ms
Input vector length: 2^15, latency: 2625.5 ms
Input vector length: 2^16, latency: 4810.300000190735 ms
Input vector length: 2^17, latency: 9102.099999904633 ms
Input vector length: 2^18, latency: 16775.199999809265 ms

Reference performance.
Input vector length: 2^14, latency: 3238.9000000953674 ms
Input vector length: 2^15, latency: 5949.900000095367 ms
Input vector length: 2^16, latency: 10859.300000190735 ms
Input vector length: 2^17, latency: 19694.099999904633 ms
Input vector length: 2^18, latency: 37178.800000190735 ms
```

Relatively, the speedup is between 53-57% depending on the test instance size.

Pre-requisites for building
---------------------------

You need to have Tinygo. See [Quick install guide](https://tinygo.org/getting-started/install/) for installing on your system.

To run the Node.js server, you also need to have a local installation of Node.js which can be obtained from [here](https://nodejs.org/dist/v19.0.0/node-v19.0.0-linux-x64.tar.xz).

Install puppeteer for remotely controlling Chrome browser: `npm i puppeteer`.

Building
--------

To manually build only the WASM binary, run `make submission.wasm` in the `submission/` directory. The submission already includes a compiled WASM binary in case setting up pre-requisites fail.

Testing
-------

Run the `evaluate.sh` script in the root folder and then in another terminal run `node headless-test.js` in the `submission/` folder. You may have to modify the benchmark parameters in `www/index.js` to include larger MSM instances. Note that generating the test vectors is quite time-consuming for `n = 2**18` (approximately 10 minutes).

Optimisations
-------------

* First, `gnark-crypto` is already very well optimised. Out of the box, it performs very well compared to the baseline implementation (arkworks).

* We use the bucket method in `gnark-crypto` (as described [here](https://eprint.iacr.org/2022/1400.pdf)), but instead of using the extended Jacobian (mixed) addition to accumulate the points in the buckets we use the batch affine addition. This is derived from https://github.com/ConsenSys/gnark-crypto/pull/249.

* We focus on MSM of sizes between `2**14` and `2**18`. For these sizes, we choose a window `c` of size that varies from 11 to 15, which determines the number of buckets to use to process each chunk. All the chunks are processed using batch affine addition except the last chuck which uses extended Jacobians, as in `gnark-crypto`. This is because the last chunk has smaller number of buckets when `c` does not divide the scalars' bitsize (`b=256` and `c=11,12,13,14,15`).

* For this to work efficiently, we need a performant scheduler to reorder the operations. Our algorithm is derived from CycloneMSM ([code](https://github.com/ConsenSys/gnark-crypto/pull/249), [paper](https://ia.cr/2022/1396)). In essence, there are 2 phases. In the first one, we loop through all the scalars (in our chunk) and fill the "batch" of a fixed size as we go. If there is a conflict, (i.e. if in the current batch, the bucketID already appears in an addition), we add the current point/scalar to a queue. Each time the batch is "full" (i.e. we reached the desired length to maximize the benfits of performing a batch affine addition with a single inverse), we perform the batch addition. In the second step, we loop through the queue and fill the batches until the queue is empty. At the end, we reduce the weighted sum of the buckets into the result of the MSM in extended Jacobian coordinates.

* We stripped out all non-essential code to prevent them being linked into the WASM binary.

* We tried using Go compiler to compile the library into a WASM binary, but it gave unsatisfactory results. We turned to TinyGo which uses LLVM for compiling the constructed intermediate representation into a WASM binary. This yielded first results which were faster than the reference.

* We checked where the runtime is invoked due to garbage collector. TinyGo escape analysis is not as efficient as plain Go and we rewrote local variable initialisations using ring-buffer. This removed all dynamic memory allocations in the library and allowed us to use `leaking` garbage collector provided by TinyGo (as we do not allocate nor free, then we won't have any memory leaks).

* We checked the optimal parameters for TinyGo. We used `-opt=2` for optimising for speed (which inlines small functions), disabled scheduler `-scheduler=none` as MSM runs in a single coroutine, use trap opcode when panic occurs `-panic=trap` to avoid linking in Go standard library (for string parsing) and stripped debug symbols `no-debug`.

* We also stripped all possible I/O calls which required importing runtime environment in WASM binary.
