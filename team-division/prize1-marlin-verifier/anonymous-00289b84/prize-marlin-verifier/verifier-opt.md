# Marlin Verifier Engineering Optimization

[TOC]

## 1. Source Code Repos

+ snarkVM:

  ../zprize-verifier-snarkVM

+ ZPrize Test Harness:

  ./


## 2. Hardware Configuration & Performance Summary

The Marlin verifier's performance is improved by ***~3 times***. The hardware configurations and test harness result are listed as follows.

| Machine | CPU | core | oririgal rounds | our rounds | original duration(s) | our duration(s) | rounds ratio(ours/original) | price($/hour) |
| :----: | :----: | :----: | :-----: | :----: | :----: | :----: | :-----: | :----: |
| AMD Epyc Rome | AMD EPYC 7402P 24-Core Processor| 1 | 2 | 7 | 10.78257612 | 10.689897122 | **3.5** | 0.03 |
| Intel Xeon Scalable | Intel(R) Xeon(R) Gold 6134 CPU @ 3.20GHz | 1 | 3 | 8 | 11.95363704 | 10.793846235 | **2.67** | 0.03 |



## 3. Optimizations

### 3.1. CPU optimizations
1. Enable BLS12_377 Fq assembly. This will accelerate the Fq multiplication, square and inverse. We support BLS12_377 curve based on the [blst](https://github.com/supranational/blst) library.
2. Initialize **FIXED** parameters once. There are 2 kinds of related parameters:
    + Poseidon Parameters: These parameters aim to calculate the poseidon hash, including rounds, ark, power and mds. There are 2 Poseidon sponges in one "verification", originally, every sponge, every "verification" will initialize its own poseidon parameters. But this is unnecessary because the all poseidon parameters are the same once the poseidon sponge is fixed. So the poseidon parameters can be prepared just once.
    + Circuit Verification Key: Obviously, the VK is the circuit's own attribute. Namely, it is fixed once the circuit is fixed. But original implementation will prepare it in every "verification". This is also unnecessary. Even considering the real scenes, the VK also can be prepared outside of each single "verify".
3. Use pippenger algorithm to do MSM with a larger scale. The original implementation just uses the naive method to do MSM in "verification". This is not adapt to every case espacially for a larger scale MSM.
4. Fix the calculation for the specific poseidon instance. In fact, the optimization utilizes the fact that the poseidon instance could be fixed before "prove" and "verification". Concretely, the alpha in apply_s_box, and state length and MDS all are fixed. So the general implementation of apply_s_box and apply_mds could be replaced by the specific calculation.
    + apply_s_box: According to the poseidon instance the alpha is 5. So s_box can be done by only two Fq squares and one Fq multiplication. This avoids the extra overhead in general implementation.
    + apply_mds: Unrolling loop. This contributes a tiny performance improvement.

### 3.2. GPU Experiments

We found that Spong (Poseidon) calculation is time-costy. We enabled GPU implementatation trying to accelerate Poseidon Hash calculation. The following featurs are enabled by us trying to utilize GPU fully.
1. Use 7 cuda cores to do 1 round.
2. All calculation logic is done in shared memory.

Due to the competition rule - parallelization is impossible between different rounds or groups, only less than one hundred of cores are active.

The performance of our GPU implementation is not good for this scenario. Worse than, In order to make GPU full utilized, we have to use lots of CPU threads to produce sufficient parallel woking jobs. According to the zprize competition rule, if the performance/cost is concerned, the tuned GPU solution is NOT good as expected.



## 4. Benchmark Modification Explaination

To be adapt to the above optimizations, there are some tiny modifications to the benchmark itself.

1. Make poseidon parameters static, and initialize them only once.
2. Prepare "prepared verification key" outside of each single verification.

The above tiny modifications are reasonable because it is a general optimization, not just for the zprize itself.

## 5. How to run

Just run the following command:

cargo run --release --features "blst_asm"
