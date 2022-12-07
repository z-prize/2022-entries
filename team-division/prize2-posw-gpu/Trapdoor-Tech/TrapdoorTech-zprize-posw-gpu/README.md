# Accelerating PoSW on GPU

Prize Sponsor: Aleo

Prize Architect: Aleo  
Prize Reward: 3M Aleo Credits

## Run the benchmark

Our best performance/cost setup would be a Coreweave Server with following hardware ($0.87/hour), on which we achieved around **56.5** TPS, or equivalently **1130** proofs generated in 20 seconds:

* 6 vCPUs on AMD EPYC 7413 Processor
* 6G memory
* 40G storage space
* a single Nvidia RTX A5000 GPU card
* Ubuntu 20.04 LTS

***For the details of our optimizations, please check our
[ZPrize - POSW GPU optimization.pdf](./ZPrize%20-%20POSW%20GPU%20optimization.pdf)***

To run the benchmark, please follow these steps:

1. Install Nvidia Cuda 11-7 by running `sudo apt install cuda-11-7`
2. Generate all GPU lookup tables by running `cargo run --bin generator --release -- --window-size 9`, for Nvidia A5000 GPU, we recommend setting `window size = 9`. This may take some time. On a 30 cores server it will take 5-10 minutes. We suggest generating them on a high-performance server then copy it to our featured test server.
3. Run `THREAD_COUNT=10 cargo run --release` to test TPS, `THREAD_COUNT` specifies the number of prover threads and `10` is an empirical value. 

Note:

1. if you want to regenerate GPU kernel `gpu-kernel.bin`, checkout `TrapdoorTech-zprize-ec-gpu-kernels` repo, then follow the instructions inside it.
2. if `lagrange-g-calced.data` file is corrupted, you can regenerate it along with lookup tables, by running `cargo run --bin generator --release -- --window-size 9 --gen-shifted-lagrange-basis`

## Prize Description

### Summary

### Proof of Succinct Work (PoSW) is a novel work-based consensus algorithm for blockchains where the work that is proven is the generation of a SNARK proof. Miners compete to provide a valid solution to the PoSW puzzle by repeatedly generating SNARK proofs until they satisfy a given difficulty level provided by the protocol.


	PoSW_Miner(difficulty, state_root, witness)
		proof_preprocess = Marlin_Prover_Preprocess(witness)

		while(Hash(coinbase_proof) > difficulty)
			seed = Hash(state_root || nonce)
			coinbase_proof = Marlin_RandProver(seed, proof_preprocess)

		return coinbase_proof


We define each of the above variables and methods below:

  

`difficulty` : updated based on the chain update rule according to the consensus algorithm

`state_root` : root of the state tree of the latest block

`witness` :  user  data required by the [output predicate](https://github.com/AleoHQ/snarkVM/blob/testnet3/dpc/src/circuits/output/output_circuit.rs) in order to credit the current miner

  

`proof_preprocess` :  output of the [first Marlin round](https://github.com/AleoHQ/snarkVM/blob/a48919d2bf0e644904f83bab83ceffbcd0e18708/algorithms/src/snark/marlin/ahp/prover/round_functions/first.rs) over the output predicate, consisting of:

-   `com(w)` : KZG commitment to field elements constituting the witness
    
-   `com(z_A), com(z_B)` : KZG commitments to the z_A, z_B polynomials
    
-   `MM::ZK flag set to FALSE`
    

  

`seed` : commitment to the current state and nonce

`coinbase_proof` : Marlin proof of the output circuit certifying the generation of a new coinbase output crediting the miner address provided in witness

`Hash`

A cryptographic hash function, herein initialized as SHA256.

  

`Marlin_Prover_Preprocess`

This method is defined as the compilation of the first round of Marlin over the output predicate circuit and depends only on the witness elements provided by the miner.

  

`Marlin_RandProver`

This method takes the output of Marlin_Prover_Preprocess and computes the [rest](https://github.com/AleoHQ/snarkVM/blob/a48919d2bf0e644904f83bab83ceffbcd0e18708/algorithms/src/snark/marlin/ahp/prover/round_functions/second.rs)  [of](https://github.com/AleoHQ/snarkVM/blob/a48919d2bf0e644904f83bab83ceffbcd0e18708/algorithms/src/snark/marlin/ahp/prover/round_functions/third.rs)  [the rounds](https://github.com/AleoHQ/snarkVM/blob/a48919d2bf0e644904f83bab83ceffbcd0e18708/algorithms/src/snark/marlin/ahp/prover/round_functions/fourth.rs) of the Marlin protocol in order to complete the generation of a valid proof for the given output circuit. Before the round computations begin, seed is added to the RNG so that all of the computation performed commits to seed  from the beginning.

### Optimization Objective

  

The objective of this contest is to produce the highest amount of PoSW proofs within a set duration of time. The target block time is 20 seconds, so the objective function will be the total number of proofs that are computed within a fixed number of 20 second time intervals.

  

For each block interval, the `PoSW_Miner` method will be executed with the same difficulty and witness values, while `state_root` will be randomly resampled. Thus, optimizations to both `Marlin_Prover_Preprocess` and `Marlin_RandProver`  will contribute to the final objective.

### Constraints

  

The input variables (`difficulty, state_root, witness`) and `nonce` will be chosen by the testing method, and cannot be hardcoded or chosen by participating parties.

  

Although the optimization objective will be based on a fixed set of inputs, all valid solutions should provide correct output proofs for any set of input variables.

## Timeline

  

June 1 - Final competitor selection

June 10 - Competition begins

July 25 - Mid-competition IPR

September 10 - Deadline for submissions

October 1 - Winners announced

## Judging

Competitors for the prize will be selected based on their prior documented experience and academic achievement. Submissions will be analyzed for both correctness and performance.

### Correctness

All proofs that are generated must be properly formed and pass verification. Submissions that fail any test cases will be judged as incorrect and lose the opportunity to win the prize.

### Performance

Scores will be tallied as the total number of PoSW proofs within a twenty-second time period, divided by the total cost of the hardware setup.

### Hardware & Benchmarks

Competitors can use any hardware available via HPC cloud-provider [CoreWeave](https://www.coreweave.com/).

## Prize Allocation

  

Prizes will be distributed from the total Prize Reward according to a Rank-Weighted Performance-based scoring system. The baseline score of each competitor's solution will be measured as the total number of proofs generated over a 20-second period, with an additional Rank-Based multiplier applied to each competitor's total to determine their final score and resulting Prize Payout (see below). Additionally, there will be a minimum benchmark of 500 proofs in a twenty-second period that competitors must achieve in order to be eligible for prize rewards. Prizes will be given out in good faith and in the sole discretion of the prize committee.


**Total Prize Reward: 3M Aleo Credits**

**Benchmark Minimum: 500 Proofs/20-seconds**

|  | RANK | Proofs | Rank Multiplier | Weighted Score | Prize Share | Prize Payout |
|--|--|--|--|--|--|--|
| Alice | 1st | 1500 | 4x | 6000 | .55 | 1,650,000 |
| Bob | 2nd | 1450 | 2x | 2900 | .26 | 780,000 |
| Carrol | 3rd | 900 | 1.5x | 1350 | .12 | 360,000 |
| Dave | 4th | 600 | 1x | 600 | .05 | 150,000 |
| Eve | 5th | 450 | 1x | 0 (below min) | 0 | 0 |
| SUM |  | 4900 |  | 10850 | 1.00 | 3,000,000 |


- Weighted Score =  Total Proofs  *  Rank Multiplier[1]

- Prize Share  =  Weighted Score / SUM of  all Weighted Scores

- Prize Payout  =  Prize Share  * Total Prize Reward

  

----------

[1] The Rank Multipliers to be used will not be finalized until the number of competitors is finalized, but will be chosen to enable a distribution curve roughly proportional to the example shown given the solutions provided.

## Notes

  

All submission code must be open-sourced at the time of submission. Code and documentation must be made available under the GPL 3.0 license.

## Questions

  

If there are any questions about this prize, please contact Alex at Aleo: [zprize@aleo.org](mailto:zprize@aleo.org)
