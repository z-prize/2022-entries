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

use snarkvm_dpc::{testnet2::Testnet2, BlockTemplate, Network, PoSWScheme};

use criterion::{criterion_group, criterion_main, Criterion};
use rand::SeedableRng;
use rand_chacha::ChaChaRng;

fn marlin_posw2(n: usize) {
    let mut cc = Criterion::default().sample_size(10);
    let mut group = cc.benchmark_group("Proof of Succinct Work: Marlin");
    group.sample_size(10);

    println!("start mine {}", n);

    let rng = &mut ChaChaRng::seed_from_u64(1234567*((n+1) as u64));

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
        block.to_coinbase_transaction().unwrap().to_records().next().unwrap(),
    );

    group.bench_function("mine", |b| {
        b.iter(|| {
            Testnet2::posw().mine(&block_template, &AtomicBool::new(false), rng).unwrap();
        });
    });

    if n == 0 {
    group.bench_function("verify", |b| {
        b.iter(|| {
            let _is_valid = Testnet2::posw().verify_from_block_header(Testnet2::genesis_block().header());
        });
    });
    }
}

fn marlin_posw(_c: &mut Criterion) {

    let mut n = 1;
    if let Ok(t) = std::env::var("THREAD") {
        let m = t.parse().unwrap();
        n = if m > 0 {m} else {1};
    }

    let mut h = Vec::with_capacity(n);
    for i in 0..n {
        h.push(std::thread::spawn(move || marlin_posw2(i)));
    }

   for _ in 0..n {
       if let Some(jh) = h.pop() {
           jh.join().unwrap();
       }
       
   }
}

criterion_group! {
    name = posw;
    config = Criterion::default().sample_size(10);
    targets = marlin_posw
}

criterion_main!(posw);
