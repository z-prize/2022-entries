# ZPrize POSW

## Overview

This is Supranational's entry for [ZPrize’s PoSW Acceleration (GPU) track](https://www.zprize.io/prizes/proof-of-succinct-work-acceleration-gpu). The goal of this track is to produce as many proofs as possible in 20 seconds at the lowest cost per proof. The minimum requirement is 400 proofs and available system configurations and their associated prices are from [Coreweave](http://coreweave.com). We utilize a combination of [CPU and GPU based optimizations](#Optimizations), to improve throughput beyond the minimum requirement while minimizing cost. 

## Building and Running

The Supranational entry should be built and run on Coreweave. We provide a [Kubernetes YAML file](posw.yaml) alongside this README to easily deploy our chosen system configuration on Coreweave. It can be deployed as follows:
```
kubectl apply -f posw.yaml
```

Next to log into the deployed system you must first obtain the node ID. It will look something like ```rtx4000-7954799cfd-9rmt8``` and can be obtained by running the following command:
```
kubectl get pods
```

Once the node is in the "Running" state copy this repository to the new instance, replacing the node_id in the following command with the ID resulting from the previous command:
```
git clone https://github.com/simonatsn/zprize-posw-gpu
kubectl cp zprize-posw-gpu <node_id>:/
```

Log in:
```
kubectl exec --stdin --tty <node_id> -- /bin/bash
```

Prior to building, the test harness must be cloned into the submission repository and the Cargo.toml must be modified to point to the provided snarkVM. To facilitate easier testing a [Cargo.toml](Cargo.toml) has been provided that is modified to work with a harness that is cloned into our submission’s directory. 

The executable can then be built and run as follows:
```
# Update the system
apt update
apt install -y curl git openssl libssl-dev g++ pkg-config clang libgmp-dev

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustup toolchain install nightly
rustup default nightly

# Set up the harness
cd zprize-posw-gpu
git clone https://github.com/z-prize/prize-posw-gpu.git
cd prize-posw-gpu/
git checkout 83b649f
mv ../Cargo.toml .

# Build and run
cargo run --release
```

NOTE: version 83b649f of the test harness is the latest version as of 10/20/22 and what we tested with.

## Performance

For ZPrize we focused on a cost optimized configuration and decided to utilize a Quadro RTX 4000 GPU. We expect to see approximately 745 to 755 proofs generated in twenty seconds using a single GPU. During pre-submission testing we witnessed run to run variation most likely based on factors such as the particular host instantiated, multi-tenancy, the temperature of the GPU, etc. 

We also tested our PoSW software on larger and multi-GPU systems, including the A40 and systems with multiple Quadro RTX 4000 GPUs. In these configurations we witnessed a single system producing in excess of 4000 proofs over 20s. However, alternative GPUs provided inferior performance per dollar and as a result we focused on optimizing our software for the Quadro RTX 4000 GPU. This cost optimized system can be easily scaled by increasing the number of ‘replicas’ in the [YAML file](posw.yaml).

## Cost

The system configuration deployed leverages the following compute and memory resources:

16 Xeon Scalable vCPU at $0.01/vCPU

5GB of memory at $0.005/GB

1 NVIDIA Quadro RTX 4000 at $0.24/GPU 


For a total cost of $0.425/hr.

More information on Coreweave’s resource based pricing can be found [here](https://docs.coreweave.com/resources/resource-based-pricing).

## Expected Score

The “dollar-normalized” score for the entry is estimated at approximately 1776 proofs over the twenty second testing period (755 proofs / $0.425 = 1776 proofs). Alternatively, extrapolating the cost and performance out to one hour gives approximately 320,000 proofs/$/hour (755 proofs/testing period * 180 testing periods/hour / $0.425/hour = 319,765).

## Optimizations

We implemented a number of optimizations to PoSW on both CPU and GPU. When running the baseline configuration we observed around 10 proofs in 20s on a 64-core system, a long way from the minimum 400 proofs required. The two largest bottlenecks to address were the performance of multi-scalar multiplication (MSM) and the approach to parallelism. 

### Parallelism and Threading

SnarkVM utilizes Rust parallelism (rayon, etc.) to distribute work among cores at a function level (e.g. MSM). While this works well to reduce latency, it's often not the most efficient approach for throughput. We remove all Rust level parallelism so that a single thread generates a single proof in a loop. We then instantiate multiple threads to fill the host machine. The generated proofs are stored in a queue and delivered to the harness with each call to ```prove_once_unchecked```. We also use core affinity to lock threads to cores in order to improve cache locality. 
  
### GPU Offload

GPUs offer significantly better performance per dollar for many highly parallel arithmetic operations. To take advantage of this we identified the largest and most parallelizable functions and migrated them to the GPU. Such functions include MSM and NTT, core focuses of many zPrize competition tracks, as well as some of the polynomial operations from the second and third rounds of the Marlin proof. We further improve efficiency by caching large, fixed data buffers on the GPU, including MSM bases, NTT twiddle factors, and the arithmetization matrix. These caches are shared by all threads and serve to reduce the amount of data sent over PCIe, saving transfer time and reducing system overhead.

### Marlin First Round Reuse

We observed that it is possible in this formulation of PoSW to compute and reuse the first round of the Marlin proof for all further PoSW proofs. To generate unique proofs from a shared first round we introduce randomness in the second round, thus reducing the overall amount of work per proof. In addition, we also remove the computation required for hiding as this is not enabled in PoSW.
  
### CPU Assembly

In addition to the above optimizations, we also added more performant assembly code for the 256 and 384 bit field operations.

