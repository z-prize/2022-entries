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

use std::sync::atomic::AtomicBool;
use std::time::Instant;
use std::thread;

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;

use snarkvm_dpc::{testnet2::Testnet2, BlockTemplate, Network, PoSWScheme, Block};

use rand::SeedableRng;
use rand_chacha::ChaChaRng;

fn run_posw(block: Block<Testnet2>, num: i32) {
    let rng = &mut ChaChaRng::seed_from_u64(1234567);
    let now = Instant::now();
    let block_template = BlockTemplate::new(
        block.previous_block_hash(),
        block.height(),
        block.timestamp(),
        block.difficulty_target(),
        block.cumulative_weight(),
        block.previous_ledger_root(),
        block.transactions().clone(),
        block.to_coinbase_transaction().unwrap().to_records().next().unwrap(),
    );
    let elapsed_time = now.elapsed();
    
    env::set_var("PARALLEL_NUMS", num.to_string());
    for _ in 0..num {
        let now = Instant::now();
        Testnet2::posw()
            .mine(&block_template, &AtomicBool::new(false), rng)
            .unwrap();
        let elapsed_time = now.elapsed();
    }
    
    
}


fn get_git_version() -> String {
    let version = env::var("CARGO_PKG_VERSION").unwrap().to_string();

    let child = Command::new("git")
        .args(&["describe", "--always", "--dirty"])
        .output();
    match child {
        Ok(child) => {
            let buf = String::from_utf8(child.stdout).expect("failed to read stdout");
            return version + "-" + &buf;
        },
        Err(err) => {
            eprintln!("`git describe` err: {}", err);
            return version;
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        println!("msm total threads");
        return;
    }

    let total: i32 = args[1].parse().unwrap();
    let threads: i32 = args[2].parse().unwrap();
    if total < threads || threads < 1 {
        println!("bad args");
        return;
    }
    
    let now = Instant::now();
    // Construct the block template.
    let block = Testnet2::genesis_block();
    let elapsed_time = now.elapsed();

    let now = Instant::now();

    let mut thread_handles: Vec<thread::JoinHandle<()>> = Vec::new();
    for _i in 0..threads {
        thread_handles.push(thread::spawn(move || {
            run_posw(block.clone(), total / threads);
        }));
    }

    while !thread_handles.is_empty() {
        let handle = thread_handles.remove(0);
        handle.join().unwrap();
    }

    let elapsed_time = now.elapsed();
    println!("{} ms.", elapsed_time.as_millis() as f64 / total as f64);
}
