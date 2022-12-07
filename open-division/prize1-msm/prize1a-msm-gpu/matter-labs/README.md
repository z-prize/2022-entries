# Accelerating MSM Operations on GPU/FPGA

This is a submission for **Z-Prize** category **Accelerating MSM Operations on GPU/FPGA** as
defined [here](https://www.zprize.io/prizes/accelerating-msm-operations-on-gpu-fpga).

This submission was developed by Matter Labs and is available at the following github repository: https://github.com/matter-labs/z-prize-msm-gpu

## Performance

The runtime for a batch of 4 MSMs of size 2^26 is usually in the area of ***2500 milliseconds*** executing on a single Nvidia A40 GPU.

Performance can vary quite a bit depending on the physical machine on which the particular VM instance gets provisioned and varies also with a particular GPU in
a machine and other factors like for example temperature.

## Building

The submission requires the [cuda toolkit](https://developer.nvidia.com/cuda-downloads) and [rustup](https://rustup.rs/)

Clone the repository and build using the following commands:

```
git clone https://github.com/matter-labs/z-prize-msm-gpu
cd z-prize-msm-gpu
cargo build --release
```

## Running benchmark

Run the benchmark using the following command:

```
cargo bench
```

## Running the correctness test

Run the correctness test using the following command:

```
cargo test --release
```

## Algorithm outline and description of implemented optimizations

- Workflow is based on the Pippenger algorithm with 23-bit windows
- One-time pre-computation of the powers of the bases by a factor of 4 is performed allowing the reduction of the number of windows from 11 to 3
- Initial scalars transfer is split into smaller chunks to allow processing while other scalar parts are still in transfer from host to device memory
- The scalars array on the host is opportunisticaly pinned on first use, if this behaviour is not desired, it can be disabled by changing the value of the [`REGISTER_SCALARS`](https://github.com/matter-labs/z-prize-msm-gpu/blob/main/src/lib.rs#L19) constant to false
- Scalars are processed, generating tuples of base-indexes window/bucket indexes
- The above list is sorted and run length encoded, then a list of offsets is generated based on the list of runs
- The lists are further sorted based on the run length to enable efficient usage of the GPU hardware
- Once all the above pre-processing is done, the bucket aggregation is executed, adding bases into their respective buckets
- After all the buckets have been processed, we employ a reduction algorithm that performs a series of window-splitting steps that leads to single-bit windows
- The single-bit windows are then gathered and moved to the host where a final single-threaded double-and-add reduction is performed
- The FF and EC routines have been heavily optimized:
    - Based on Montgomery multiplication
    - Minimized correction steps in the FF operations
    - Use of XYZZ representation for the EC point accumulators
    - Use of fast squaring
    - Taking advantage of `reduce(a*b) +/- reduce(c*d)` = `reduce(a*b +/- c*d)` to save reduction steps

## Further possible improvements

This solution is using CUB routines for sorting, run length encoding and offset calculations.
We are aware that CUB is not ideal for this task.
It would be beneficial to replace the CUB routines by custom implementations.
Custom routines should lead to reduction in sorting/encoding runtime and also be better in terms of memory requirements which would in turn lead to further
possible performance improvements.  

