# ZPrize Marlin Verifier

## Overview

This is Supranational's entry for [ZPrize’s Fast Verifier for Marlin Proof System track](https://www.zprize.io/prizes/fast-verifier-for-marlin-proof-system). The goal of this track is to verify as many proofs as possible in 10 seconds at the lowest cost per verification. The minimum requirement is one complete round of batch verifications and available system configurations and their associated prices are from [Coreweave](http://coreweave.com). We utilize [CPU optimizations](#Optimizations) only to improve throughput beyond the minimum requirement while minimizing cost. 

## Building and Running

The Supranational entry should be built and run on Coreweave. We provide a [Kubernetes YAML file](verif.yaml) alongside this README to easily deploy our chosen system configuration on Coreweave. It can be deployed as follows:
```
kubectl apply -f verif.yaml
```

Next to log into the deployed system you must first obtain the node ID. It will look something like `verify-5bc9db4b8c-42wcb` and can be obtained by running the following command:
```
kubectl get pods
```

Once the node is in the "Running" state copy this repository to the new instance, replacing the node_id in the following command with the ID resulting from the previous command:
```
git clone https://github.com/sean-sn/zprize-marlin-verifier
kubectl cp zprize-marlin-verifier <node_id>:/
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
apt install curl git openssl libssl-dev pkg-config g++

# Install Rust nightly
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustup toolchain install nightly
rustup default nightly

# Set up the harness
cd zprize-marlin-verifier
git clone https://github.com/z-prize/prize-marlin-verifier.git
cd prize-marlin-verifier
mv ../Cargo.toml .

# Build and run
cargo run --release
```

NOTE: version 02051ae of the test harness is the latest version as of 10/31/22 and what we tested with.

## Performance

For ZPrize we focused on a cost optimized configuration and decided to utilize only a single vCPU core. The submission is expected to complete 14 rounds during the 10 second testing period on the chosen hardware. 

## Cost

The system configuration deployed leverages the following compute and memory resources:

1 Intel Xeon v4 vCPU and 4GB of memory at $0.02/vCPU and 4GB of memory.

For a total cost of $0.02/hr.

More information on Coreweave’s resource based pricing can be found [here](https://www.coreweave.com/gpu-cloud-pricing).

## Expected Score

The “dollar-normalized” score for the entry is estimated at **700** rounds (14 rounds / 10 seconds / $0.02 = **700** rounds/10 seconds/$).

## Optimizations

We implemented a number of optimizations to Marlin Verify on CPU. The primary improvement came from utilizing an optimized version of Poseidon with precalculated matrices for the Fiat-Shamir transformation. Other improvements came from caching the circuit verifying key, modifying the MSM parameters to  use improved algorithms and bucket sizes, and leveraging assembly versions of field and pairing operations.
