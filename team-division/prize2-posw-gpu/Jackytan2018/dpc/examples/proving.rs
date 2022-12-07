// Copyright (C) 2019-2022 Aleo Systems Inc.
// This file is part of the snarkVM library.

// The snarkVM library is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// The snarkVM library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with the snarkVM library. If not, see <https://www.gnu.org/licenses/>.

use core::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use snarkvm_dpc::{
    posw::PoSWCircuit, testnet2::Testnet2, BlockTemplate, Network, PoSWError, PoSWScheme,
};
use snarkvm_utilities::Uniform;

use rand::thread_rng;

/// Run the PoSW prover for 20 seconds, attempting to generate as many proofs as possible.
pub fn proving_test<N: Network>() -> Result<usize, PoSWError> {
    println!("Running initial setup...");
    let mut rng = thread_rng();

    // Construct the block template.
    let block = Testnet2::genesis_block();
    let block_template = BlockTemplate::new(
        block.previous_block_hash(),
        block.height(),
        block.timestamp(),
        block.difficulty_target(),
        block.cumulative_weight(),
        block.previous_ledger_root(),
        block.transactions().clone(),
        block
            .to_coinbase_transaction()
            .unwrap()
            .to_records()
            .next()
            .unwrap(),
    );

    const MAXIMUM_MINING_DURATION: u64 = 20;

    let terminator = AtomicBool::new(false);

    // Instantiate the circuit.
    let mut circuit = PoSWCircuit::<Testnet2>::new(&block_template, Uniform::rand(&mut rng))?;

    println!("Done! Running proving challenge...");
    let mut proofs_generated = 0;
    let now = Instant::now();
    loop {
        // Run one iteration of PoSW.
        let proof = Testnet2::posw().prove_once_unchecked(&mut circuit, &terminator, &mut rng)?;
        // Check if the updated proof is valid.
        if !Testnet2::posw().verify(
            block_template.difficulty_target(),
            &circuit.to_public_inputs(),
            &proof,
        ) {
            // Construct a block header.
            return Err(PoSWError::Message(
                "proof verification failed, contestant disqualified".to_string(),
            ));
        }

        proofs_generated += 1;

        // Break if time has elapsed
        if now.elapsed() > Duration::from_secs(MAXIMUM_MINING_DURATION) {
            break;
        }
    }

    println!("Finished!");
    Ok(proofs_generated)
}

fn main() {
    println!("Starting the ZPrize proving challenge.");
    println!(
        "Generated {} proofs within allotted time. ",
        proving_test::<Testnet2>().unwrap()
    );
}
