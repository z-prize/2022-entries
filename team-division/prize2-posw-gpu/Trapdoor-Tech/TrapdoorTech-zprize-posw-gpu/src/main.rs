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

use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use snarkvm_curves::PairingEngine;
use snarkvm_dpc::PoSWProof;
use snarkvm_dpc::{posw::PoSWCircuit, testnet2::Testnet2, BlockTemplate, Network, PoSWScheme};

use snarkvm_utilities::Uniform;

use rand::thread_rng;

use std::sync::Arc;
use std::thread::spawn;

/// Run the PoSW prover for 20 seconds, attempting to generate as many proofs as possible.
fn main() {
    let _ = env_logger::builder().try_init();

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

    // in order to measure our maximum TPS, set the target difficulty to a very high level so that our loops inside prover won't break too early
    let target_difficulty = 1;
    // let target_difficulty = block.difficulty_target();

    let block_height = block.height();

    const MAXIMUM_MINING_DURATION: u64 = 20;

    let starter = Arc::new(AtomicBool::new(false));
    let terminator = Arc::new(AtomicBool::new(false));

    // Instantiate the circuit.
    let circuit = PoSWCircuit::<Testnet2>::new(&block_template, Uniform::rand(&mut rng)).unwrap();

    snarkvm_dpc::posw::reload_all_bases::<
        <<Testnet2 as Network>::InnerCurve as PairingEngine>::G1Affine,
    >();

    println!("Done! Running proving challenge...");

    let proofs_generated = Arc::new(AtomicUsize::new(0));

    let thread_count = std::env::var("THREAD_COUNT")
        .and_then(|v| match v.parse() {
            Ok(val) => Ok(val),
            Err(_) => {
                eprintln!("Invalid env THREAD_COUNT! Defaulting to 1...");
                Ok(1)
            }
        })
        .unwrap_or(1);

    let mut thread_handlers = Vec::new();

    for i in 0..thread_count {
        let mut circuit1 = circuit.clone();
        let block_template1 = block_template.clone();
        let starter = starter.clone();
        let terminator = terminator.clone();
        let proof_counter = proofs_generated.clone();
        println!("spawning #{i} threads");

        let thread = spawn(move || {
            let mut rng = thread_rng();

            loop {
                if starter.load(Ordering::SeqCst) {
                    break;
                }
            }

            println!("#{i} thread start computing");

            let mut tmp_proof = None;

            // Run a loop of PoSW.
            let _ = Testnet2::posw().prove_once_unchecked_ctr(
                &mut circuit1,
                &terminator,
                &proof_counter,
                &mut rng,
                target_difficulty,
                block_height,
                &mut tmp_proof,
            );

            // Either the loop was ended by terminator or difficulty target was met, we have a valid proof inside `tmp_proof`
            let final_proof = PoSWProof::<Testnet2>::new(tmp_proof.unwrap().into());

            // Check if the updated proof is valid.
            if !Testnet2::posw().verify(
                block_template1.difficulty_target(),
                &circuit1.to_public_inputs(),
                &final_proof,
            ) {
                panic!("proof verification failed, contestant disqualified");
            }
        });

        thread_handlers.push(thread);
    }

    let time_counting_thread = spawn(move || {
        println!("waiting 5 seconds to start computation...");
        thread::sleep(Duration::new(5, 0));

        println!("running 20 seconds benchmark!");
        starter.swap(true, Ordering::SeqCst);

        // wait 20 seconds to end computation
        thread::sleep(Duration::new(MAXIMUM_MINING_DURATION, 0));

        terminator.swap(true, Ordering::SeqCst);

        println!("Finished!");
        println!(
            "proofs_generated = {}",
            proofs_generated.load(Ordering::SeqCst)
        );
    });

    for handler in thread_handlers {
        handler.join().unwrap();
    }

    time_counting_thread.join().unwrap();
}
