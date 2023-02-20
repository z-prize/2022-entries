# Griffinfly - ZPRIZE FPGA NTT

Griffinfly is COSIC's submission to the [ZPRIZE](https://www.zprize.io/) competition under the category, [Accelerating NTT Operations on an FPGA](https://www.zprize.io/prizes/accelerating-ntt-operations-on-an-fpga) by Michiel Van Beirendonck, Jonas Bertels, Furkan Turan, and Ingrid Verbauwhede.

## Overview

Griffinfly accelerates Number Theoretic Transforms (NTTs) in zero-knowledge proof protocols on hbm-based FPGAs. For the ZPRIZE competition, Griffinfly implements size-$2^{24}$ NTTs over the Goldilocks field (modulo $2^{64} - 2^{32} + 1$). 

Griffinfly is developed fully in HLS with minimal performance penalty. We hope that the C++-based design helps with readability and enables developers to quickly jump in and improve upon our work. The key design characteristics of Griffinfly are:

* Two recursive steps of the four-step NTT algorithm:
    * We view the $2^{24}$-point NTT as a 2D NTT of $2^{12}$ by $2^{12}$
    * We view each $2^{12}$-point NTT as a 2D NTT of $2^{6}$ by $2^{6}$
    * Four-step matrix-transpositions are implemented at full throughput using XOR permutations
* Efficient twiddle-factor generation using on-the-fly twiddling 
* Efficient goldilocks arithmetic, including the [goldilocks root-of-unity trick](./doc/omega.ipynb)

The low-level computational unit of Griffinfly is a fully-unrolled [$2^{6}$-point NTT core](./src/hls/ntt_2_6.cpp) with an Initiation Interval of II=1, i.e. the core can accept a vector of $2^{6}$ input points *every clock cycle*. Using the goldilocks root-of-unity trick, all twiddle factor multiplications in this unit are realized with logical shifts only.

One level up, the $2^{6}$ NTT core is used to construct a [$2^{12}$-point NTT unit](./src/hls/ntt_2_12.cpp) using the four-step algorithm[^1]. In this NTT algorithm, the $2^{12}$-point input is viewed as a 2D array of $2^{6} \times 2^{6}$. Then, the four steps are:

1. Compute $2^{6}$-point NTTs on each row
2. Multiply the resulting matrix with twiddles
3. Transpose the matrix
4. Compute $2^{6}$-point NTTs on each column

Griffinfly allows two options for the $2^{12}$-point NTT unit:

1. `ntt_2_12_dataflow`: Here, the four steps are connected in a dataflow pipeline. This pipeline includes two $2^{6}$-point cores, $2^{6}$ parallel twiddle multipliers, and a full-throughput matrix transpose using spatial and temporal XOR permutations[^2]. This dataflow pipeline can sustain the throughput of the $2^{6}$-point core, processing $2^{6}$ input points every clock cycle. As a result, the $2^{12}$-point achieves an II=64.

2. `ntt_2_12_serial` : Here, the four steps are executed serially. This uses a single $2^{6}$-point core, a single twiddle multiplier, and a serial matrix transpose. This $2^{12}$-point unit has significant area savings, but a smaller II=~4096.

The [$2^{24}$-point NTT unit](./src/hls/ntt_2_24.cpp) is completed using another application the four-step algorithm. This requires two passes through the $2^{12}$-point unit. Between the passes is again a twiddle multiplication --- with an II matched to the $2^{12}$-point core --- and a large $2^{12} \times 2^{12}$ XOR-based matrix transpose.

For this large matrix transpose, data is too large to fit onto the FPGA and must be written back to HBM. Temporal XOR permutations are executed inside the FPGA HBM, using an addressing scheme that is programmed into our `mm2s` and `s2mm` DMA engines.

For a more in-depth overview of our design, we refer to our paper [XYZ](./README.md), which will be available shortly. 

[^1]: Bailey, David H. "FFTs in external of hierarchical memory." Proceedings of the 1989 ACM/IEEE conference on Supercomputing. 1989.

[^2]: Garrido, Mario, and Peter Pirsch. "Continuous-flow matrix transposition using memories." IEEE TCAS-I: Regular Papers 67.9. 2020.

## Usage

:warning: **The project was built and tested using Xilinx Vitis tools 2022.2. We have encountered several problems with earlier versions of the tools.**

This project comes with a [Makefile](./Makefile) built after [Vitis Accel Examples](https://github.com/Xilinx/Vitis_Accel_Examples). 

Griffinfly can be configured by changing variables in [mk/config.mk](./mk/config.mk), e.g. setting the target platform and point size, and selection between the `dataflow` and `serial` versions of `ntt_2_12`. This version of the repository only includes a $2^{24}$-point NTT implementation. 

:warning: Make sure to set the `DEVICE_ID` to the correct ID for the chosen platform.

After sourcing your Xilinx tools setup and configuring `config.mk`, the project can be built and run on the FPGA in either configuration using:

```bash
make run TARGET=hw FCLK=200 NTT_2_12_FLAG=NTT_2_12_SERIAL
```


```bash
make run TARGET=hw FCLK=100 NTT_2_12_FLAG=NTT_2_12_DATAFLOW
```

or emulated using:


```bash
make run TARGET=sw_emu
```

First-time users will need to generate testvectors for the design using our software reference. This can take up to ~10 minutes:

```bash
make testvectors
```

Additional make targets are available for individual build steps, use `make help` for details. 

A successful run should show the following output:

```
[INFO] Open device[1]
[INFO] Load xclbin: COSIC_NTT_2_24_100_MHz_NTT_2_12_DATAFLOW.xclbin
[INFO] Fetch compute kernel: ntt_2_24
[INFO] Allocate Buffer of size 8388608B in each HBM
[INFO] Loading testvectors
[INFO] Synchronize input buffer data to device HBM (timing: 40936 microseconds)
[INFO] NTT IP Run (timing: 253457 microseconds)
[INFO] Synchronize output buffer data from device HBM (timing: 56507 microseconds)
[INFO] Validating results
[INFO] TEST PASSED
```

## Results

The following results are for the `xilinx_u55n_gen3x4_xdma_2_202110_1` platform. We will refer to the $2^{24}$-point NTT design using `ntt_2_12_dataflow` as `ntt_2_24_dataflow`, and the design using `ntt_2_12_serial` as `ntt_2_24_serial`.

### Achievable Clock Frequency

Frequency is limited by routing congestion. The `ntt_2_24_serial` can achieve higher clock frequency due to the heavily reduced area cost.

| Name              | Frequency |
| ----------------- | --------- |
| ntt_2_24_serial   | 200 MHz   |
| ntt_2_24_dataflow | 100 MHz   |
| hbm               | 450 MHz   |

### Utilization

`ntt_2_24_serial.impl_1_kernel_util_routed.rpt`

```
+---------------------+------------------+------------------+-------------------+----------------+---------------+----------------+
| Name                | LUT              | LUTAsMem         | REG               | BRAM           | URAM          | DSP            |
+---------------------+------------------+------------------+-------------------+----------------+---------------+----------------+
| Platform            |  96960 [ 11.14%] |  12130 [  3.01%] |  144105 [  8.27%] |  133 [  9.90%] |   0 [  0.00%] |    4 [  0.07%] |
| User Budget         | 773760 [100.00%] | 390590 [100.00%] | 1599255 [100.00%] | 1211 [100.00%] | 640 [100.00%] | 5948 [100.00%] |
|    Used Resources   | 170364 [ 22.02%] |   7681 [  1.97%] |  192009 [ 12.01%] |   63 [  5.20%] |   0 [  0.00%] |  160 [  2.69%] |
|    Unused Resources | 603396 [ 77.98%] | 382909 [ 98.03%] | 1407246 [ 87.99%] | 1148 [ 94.80%] | 640 [100.00%] | 5788 [ 97.31%] |
| ntt_2_24            | 170364 [ 22.02%] |   7681 [  1.97%] |  192009 [ 12.01%] |   63 [  5.20%] |   0 [  0.00%] |  160 [  2.69%] |
|    ntt_2_24_1       | 170364 [ 22.02%] |   7681 [  1.97%] |  192009 [ 12.01%] |   63 [  5.20%] |   0 [  0.00%] |  160 [  2.69%] |
+---------------------+------------------+------------------+-------------------+----------------+---------------+----------------+
```

`ntt_2_24_dataflow.impl_1_kernel_util_routed.rpt`

```
+---------------------+------------------+------------------+-------------------+----------------+---------------+----------------+
| Name                | LUT              | LUTAsMem         | REG               | BRAM           | URAM          | DSP            |
+---------------------+------------------+------------------+-------------------+----------------+---------------+----------------+
| Platform            |  97024 [ 11.14%] |  12130 [  3.01%] |  144106 [  8.27%] |  133 [  9.90%] |   0 [  0.00%] |    4 [  0.07%] |
| User Budget         | 773696 [100.00%] | 390590 [100.00%] | 1599254 [100.00%] | 1211 [100.00%] | 640 [100.00%] | 5948 [100.00%] |
|    Used Resources   | 401622 [ 51.91%] |  17153 [  4.39%] |  238409 [ 14.91%] |  688 [ 56.81%] |   0 [  0.00%] | 3120 [ 52.45%] |
|    Unused Resources | 372074 [ 48.09%] | 373437 [ 95.61%] | 1360845 [ 85.09%] |  523 [ 43.19%] | 640 [100.00%] | 2828 [ 47.55%] |
| ntt_2_24            | 401622 [ 51.91%] |  17153 [  4.39%] |  238409 [ 14.91%] |  688 [ 56.81%] |   0 [  0.00%] | 3120 [ 52.45%] |
|    ntt_2_24_1       | 401622 [ 51.91%] |  17153 [  4.39%] |  238409 [ 14.91%] |  688 [ 56.81%] |   0 [  0.00%] | 3120 [ 52.45%] |
+---------------------+------------------+------------------+-------------------+----------------+---------------+----------------+
```

### Performance

In the instantiation in this repository, Griffinfly is bottlenecked by the irregular memory pattern of the `mm2s` and `s2mm` DMA engines. As a result, the design with `ntt_2_24_serial` at 200 MHz and `ntt_2_24_dataflow` at 100 MHz feature identical latency. 

| Name              | Time       |
| --------------    | ---------- |
| ntt_2_24_serial   | 253 mSec   |
| ntt_2_24_dataflow | 253 mSec   |

:hourglass: Stay tuned for a design that fully exploits the massive throughput of `ntt_2_24_dataflow`. :hourglass:

## Directory Layout

    .
    ├── build                   # Kernel compilation artifacts
    ├── cfg                     # Vitis configuration files
    ├── common                  # Host code include dirs
    ├── doc                     # Documentation
    ├── mk                      # Makefile utilities
        └── config.mk           # Project configuration (NTT size, target device...)
    ├── run                     # Kernel run artifacts
    ├── scripts                 # Tcl tools
    ├── src                     # Source files
        ├── hls                 # Kernel HLS source files
        └── host                # Host run source files
    ├── testvectors             # Testvectors
        └── testvectors.py      # NTT software reference 
    └── Makefile                # Main project Makefile


---
Michiel Van Beirendonck, Jonas Bertels, Furkan Turan, and Ingrid Verbauwhede.
