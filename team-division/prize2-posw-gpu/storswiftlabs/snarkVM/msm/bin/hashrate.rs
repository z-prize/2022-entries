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

use snarkvm_dpc::{testnet2::Testnet2, BlockHeader, BlockTemplate, Network, PoSWScheme};
use std::{
    collections::VecDeque,
    env,
    process,
    sync::{
        atomic::{AtomicBool, AtomicI64, AtomicU32, Ordering},
        Arc,
    },
    time::Duration,
};

use chrono::Local;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use structopt::StructOpt;

use rayon::{ThreadPool, ThreadPoolBuilder};
use tokio::task;

use tracing::*;
use tracing_subscriber::{fmt, EnvFilter};

pub const VERSION: &'static str = include_str!(concat!(env!("OUT_DIR"), "/VERSION"));
const MAXIMUM_MINING_DURATION: i64 = 20000;

#[derive(Debug, StructOpt)]
#[structopt(name = "prover", about = "Standalone prover.", setting = structopt::clap::AppSettings::ColoredHelp)]
struct Opt {
    /// Enable debug logging
    #[structopt(short = "d", long = "debug")]
    debug: bool,

    /// Number of threads, GPU defaults is 2, CPU defaults is 7
    #[structopt(short = "t", long = "threads")]
    threads: Option<usize>,

    #[structopt(verbatim_doc_comment)]
    /// Indexes of GPUs to use (starts from 0)
    /// Specify multiple times to use multiple GPUs
    /// Example: -g 0 -g 0,1
    /// Note: Pure CPU proving will be disabled as each GPU job requires one CPU thread as well
    #[structopt(short = "g", long = "cuda")]
    cuda: Option<String>,

    #[structopt(verbatim_doc_comment)]
    /// Parallel jobs per GPU, defaults to 26
    /// Example: -j 4
    /// The above example will result in 8 jobs in total
    #[structopt(short = "j", long = "cuda-jobs")]
    jobs: Option<u8>,

    #[structopt(verbatim_doc_comment)]
    /// 默认不开启20s测试
    /// Example: -s
    /// 是否开启20s测试
    #[structopt(short = "s", long = "second")]
    second: bool,

    #[structopt(verbatim_doc_comment)]
    /// verify prover
    /// Example: -v
    #[structopt(short = "v", long = "verify")]
    verify: bool,

    #[structopt(verbatim_doc_comment)]
    /// once prover
    /// Example: -o
    #[structopt(short = "o", long = "once")]
    once: bool,

    #[structopt(verbatim_doc_comment)]
    /// 在-s设置后生效，表示时间周期末端不起的新任务
    /// Example: -l 500
    #[structopt(short = "l", long = "loss-time")]
    loss_time: Option<i64>,
}

#[tokio::main]
async fn main() {
    let opt = Opt::from_args();

    let debug = opt.debug;
    if debug {
        init_logger(2);
    } else {
        init_logger(0);
    }

    let second = opt.second;
    let verify = opt.verify;
    let once = opt.once;

    let loss_time = opt.loss_time.unwrap_or(0);
    // 单任务线程数
    let mut threads = opt.threads.unwrap_or(7);
    // 并发任务数
    let jobs: Option<u8>;
    // GPU编号
    let cuda: Option<String>;

    if cfg!(feature = "cuda") || cfg!(feature = "opencl") {
        cuda = opt.cuda;
        if cuda.is_none() {
            if opt.jobs.is_none() {
                jobs = Some(2)
            } else {
                jobs = opt.jobs;
            }
        } else {
            threads = 2;
            if opt.jobs.is_none() {
                jobs = Some(26)
            } else {
                jobs = opt.jobs;
            }
        }
    } else {
        cuda = None;
        if opt.jobs.is_none() {
            jobs = Some(2)
        } else {
            jobs = opt.jobs;
        }
    }

    if let Some(cuda) = cuda.clone() {
        env::set_var("BELLMAN_WORKING_GPUS", cuda.to_string());
        info!("{} Starting prover with GPU {}", VERSION, cuda.to_string());
    } else {
        info!("{} Starting prover with CPU", VERSION);
    }

    let mut threadpools: Vec<Arc<ThreadPool>> = Vec::new();

    for _ in 0..jobs.unwrap() {
        let p = ThreadPoolBuilder::new().stack_size(8 << 20).num_threads(threads).build().unwrap();
        threadpools.push(Arc::new(p));
    }

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

    let threadpools = Arc::new(threadpools);
    let terminator = Arc::new(AtomicBool::new(false));
    let total_proofs: Arc<AtomicU32> = Arc::new(AtomicU32::new(0));
    let invalid_proofs: Arc<AtomicU32> = Arc::new(AtomicU32::new(0));
    let usetime: Arc<AtomicI64> = Arc::new(AtomicI64::new(0));
    let last_thread_time: Arc<AtomicI64> = Arc::new(AtomicI64::new(0));

    let threadpools = threadpools.clone();
    let totals = total_proofs.clone();

    let mut rng = ChaChaRng::seed_from_u64(1234567);
    BlockHeader::mine_once_unchecked(&block_template, &terminator, &mut rng).unwrap();

    task::spawn(async move {
        log_rate(totals).await;
    });
    let start_time = Local::now();
    let mut handlers = Vec::new();
    let mut pool_i: i64 = 0;
    for pool in threadpools.iter() {
        let terminator = terminator.clone();
        let block_template = block_template.clone();
        let pool = pool.clone();
        let total_proofs = total_proofs.clone();
        let invalid_proofs = invalid_proofs.clone();
        let usetime = usetime.clone();
        let last_thread_time = last_thread_time.clone();
        pool_i = pool_i + 1;
        if pool_i % 5 == 0{
            let mut min_time: i64 = 0;
            if let Ok(t) = std::env::var("INIT_WAIT_TIME") {
                let m = t.parse::<u32>().unwrap();
                min_time = if m <= 800 { m as i64 } else { 800 };
            }

            let thread_last_time = last_thread_time.load(Ordering::SeqCst);
            let thread_start_time = Local::now().timestamp_millis();
            let diff_time = thread_start_time - thread_last_time;
            if min_time - diff_time > 0 {
                last_thread_time.store(thread_last_time + min_time, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis((min_time - diff_time).try_into().unwrap())).await;
            } else {
                last_thread_time.store(thread_start_time, Ordering::SeqCst);
            }
        }
        handlers.push(task::spawn(async move {
            loop {
                let terminator = terminator.clone();
                let block_template = block_template.clone();
                // let block_height = block_template.block_height();
                let difficulty_target = block_template.difficulty_target();
                let header_root = block_template.to_header_root().unwrap();
                let pool = pool.clone();
                let mut rng = ChaChaRng::seed_from_u64(1234567);
                if let Ok(Ok(block_header)) = task::spawn_blocking(move || {
                    pool.install(|| BlockHeader::mine_once_unchecked(&block_template, &terminator, &mut rng))
                })
                .await
                {
                    let proof = block_header.proof().clone();
                    if verify {
                        let difficulty = proof.to_proof_difficulty().unwrap_or(u64::MAX);
                        debug!("cacl proof difficulty {}, block template target {}", difficulty, difficulty_target);
                        if !Testnet2::posw().verify(difficulty_target, &[*header_root, *block_header.nonce()], &proof) {
                            invalid_proofs.fetch_add(1, Ordering::SeqCst);
                            let invalid = invalid_proofs.load(Ordering::SeqCst);
                            error!("proof invalid {}", invalid);
                            continue;
                        }
                    }

                    if second {
                        let start_t = start_time.timestamp_millis();
                        let cur_t = Local::now().timestamp_millis();
                        let diff_time = cur_t - start_t;
                        if diff_time < MAXIMUM_MINING_DURATION {
                            usetime.store(diff_time, Ordering::SeqCst);
                            total_proofs.fetch_add(1, Ordering::SeqCst);
                        }
                    } else {
                        total_proofs.fetch_add(1, Ordering::SeqCst);
                    }
                }

                if second {
                    let start_t = start_time.timestamp_millis();
                    let cur_t = Local::now().timestamp_millis();
                    let diff_time = cur_t - start_t;
                    if diff_time >= (MAXIMUM_MINING_DURATION - loss_time) {
                        break;
                    }
                }

                if once {
                    break;
                }
            }
        }));

        if once {
            break;
        }
    }
    futures::future::join_all(handlers).await;
    let proofs = total_proofs.load(Ordering::SeqCst);
    let use_t = usetime.load(Ordering::SeqCst);
    info!(
        "{}",
        format!("Test end ! 20 second create proofs {}, lost time {})", proofs, (MAXIMUM_MINING_DURATION - use_t))
    );
}

async fn log_rate(total_proofs: Arc<AtomicU32>) {
    fn calculate_proof_rate(now: u32, past: u32, interval: u32) -> Box<str> {
        if interval < 1 {
            return Box::from("---");
        }
        if now <= past || past == 0 {
            return Box::from("---");
        }
        let rate = (now - past) as f64 / (interval * 60) as f64;
        Box::from(format!("{:.2}", rate))
    }
    let mut log = VecDeque::<u32>::from(vec![0; 60]);
    loop {
        tokio::time::sleep(Duration::from_secs(20)).await;
        let proofs = total_proofs.load(Ordering::SeqCst);
        log.push_back(proofs);
        let s20 = *log.get(59).unwrap_or(&0);
        let m1 = *log.get(57).unwrap_or(&0);
        let m5 = *log.get(45).unwrap_or(&0);
        let m10 = *log.get(30).unwrap_or(&0);
        let m20 = log.pop_front().unwrap_or_default();

        info!(
            "{}",
            (format!(
                "{} {} perf: {} (last 20s:{})(1m: {} P/s, 5m: {} P/s, 10m: {} P/s, 20m: {} P/s)",
                VERSION,
                process::id(),
                proofs,
                proofs - s20,
                calculate_proof_rate(proofs, m1, 1),
                calculate_proof_rate(proofs, m5, 5),
                calculate_proof_rate(proofs, m10, 10),
                calculate_proof_rate(proofs, m20, 20),
            ))
        );
    }
}

fn init_logger(v: u8) {
    match v {
        0 => std::env::set_var("RUST_LOG", "info"),
        1 => std::env::set_var("RUST_LOG", "debug"),
        2 | 3 => std::env::set_var("RUST_LOG", "trace"),
        _ => std::env::set_var("RUST_LOG", "info"),
    };

    let filter = EnvFilter::from_default_env()
        .add_directive("mio=off".parse().unwrap())
        .add_directive("tokio_util=off".parse().unwrap())
        .add_directive("rust_gpu_tools=off".parse().unwrap());

    let _ = fmt().with_env_filter(filter).try_init();
}
