use core::{fmt, iter, slice};
// use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use fpga::{Fpga, SendBuffer64 as SendBuffer};

const PAGE_SIZE: u32 = 256;
// const PAGE_SIZE: u32 = 1;
const SRAM_SIZE: u32 = 2 * PAGE_SIZE;
const SRAM_MASK: usize = SRAM_SIZE as usize - 1;
const COLUMN_BITS: u8 = 16;
// const NUM_BITS: usize = 377;
const COLUMNS: usize = 16; //(NUM_BITS / COLUMN_BITS as usize) as u8;
const NUM_BUCKETS: usize = 1 << (COLUMN_BITS - 1);
const NUM_BUCKETS_MASK: usize = NUM_BUCKETS - 1;

const CYCLES: usize = 200;

fn diff(t0: SystemTime, t1: SystemTime) -> f64 {
    t1.duration_since(t0).unwrap().as_secs_f64()
}

pub type Scalar = [u64; 4];
// pub type Digit = i32;
pub type Digit = u16;
// pub type SDigit = i32;

#[derive(Copy, Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct Instruction {
    point: u32,
    process: bool,
    sram: u16,
    negate: bool,
    bucket: u16,
}

pub struct Command(u64);

impl fmt::Debug for Command {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        f.write_fmt(format_args!(
            "ins {} point {:04} process {} sram {:03X} bucket {:04X}",
            self.0 & 0b111,
            self.point(),
            (self.0 >> 29) & 1,
            self.sram(),
            self.bucket()
        ))
    }
}

impl Command {
    #[inline]
    pub fn bucket(&self) -> u16 {
        ((self.0 >> 14) as u16) & ((1 << 15) - 1)
    }

    #[inline]
    pub fn point(&self) -> u32 {
        (self.0 >> 30) as _
    }

    #[inline]
    pub fn sram(&self) -> u16 {
        ((self.0 >> 5) as u16) & ((1 << 9) - 1)
    }

    #[inline]
    pub fn process(&self) -> bool {
        (self.0 >> 29) & 1 != 0
    }

    #[inline]
    pub fn negate(&self) -> bool {
        (self.0 >> 4) & 1 != 0
    }

    #[inline]
    pub fn ins(&self) -> u8 {
        self.0 as u8 & 0b1111
    }
}

impl From<u64> for Instruction {
    fn from(cmd: u64) -> Instruction {
        let cmd = Command(cmd);
        Instruction {
            point: cmd.point(),
            process: cmd.process(),
            sram: cmd.sram(),
            negate: cmd.negate(),
            bucket: cmd.bucket(),
        }
    }
}

impl Instruction {
    const WASTE_CYCLE: u64 = 4;

    #[inline]
    pub fn serialize(self) -> u64 {
        let Instruction {
            negate,
            sram,
            bucket,
            process,
            point,
        } = self;

        // bits   len content             model
        // ----------------------------------------
        // 0:1    2   3
        // 2:3    2   RFU
        // 4      1   negate              bool
        // 5:13   9   SRAM index, 0..512  u16
        // 14:28  15  bucket index        u16
        // 29     1   process             bool
        // 30:63  34  point index         usize

        let mut entry = 3;

        entry |= (negate as u64) << 4;
        debug_assert!(sram < 1 << 9);
        // entry |= (sram as u64 & ((1 << 9) - 1)) << 5;
        entry |= (sram as u64) << 5;
        debug_assert!(sram < 1 << 15);
        entry |= (bucket as u64 & ((1 << 15) - 1)) << 14;
        // entry |= (bucket as u64) << 14;
        entry |= (process as u64) << 29;
        entry |= (point as u64) << 30;

        entry
    }
}

pub struct App {
    fpga: fpga::F1,
    len: usize,
    cmd_addr: usize,
}

impl App {
    pub fn new(fpga: fpga::F1, size: u8) -> Self {
        assert!(size < 32);
        let app = App {
            fpga,
            len: 1 << size,
            cmd_addr: 0,
        };
        app.set_size();
        app.set_page_size(PAGE_SIZE);
        #[cfg(feature = "read-1")]
        app.set_page_size(1);
        #[cfg(feature = "read-2")]
        app.set_page_size(2);

        app
    }

    fn stats(&self) {
        self.fpga.write_register(0x10, 0);
        // sys::write_32_f1(0x10<<2, 0);
        println!("missed  = {}", self.fpga.read_register(0x20));
        // sys::write_32_f1(0x10<<2, 1);
        self.fpga.write_register(0x10, 1);
        println!("bubbles = {}", self.fpga.read_register(0x20));
    }

    fn set_size(&self) {
        self.fpga.write_register(0x20, self.len as _);
    }

    fn set_page_size(&self, page_size: u32) {
        self.fpga.write_register(0x11, page_size);
    }

    pub fn set_points(&self, points: &[()]) {
        assert!(self.len == points.len());
        todo!();
    }

    pub fn backoff(&self) {
        while self.fpga.read_register(0x21) > 256 {
            continue;
        }
    }

    pub fn start(&mut self) {
        self.cmd_addr = 4 << 26;
        let mut cmds = SendBuffer::default();
        cmds[0] = 1;
        self.fpga.send64(self.cmd_addr, &cmds);
        self.fpga.flush();
        self.cmd_addr += 1;
    }

    pub fn update(&mut self, commands: &SendBuffer) {
        self.fpga.send64(self.cmd_addr, &commands);
        self.cmd_addr += 1;

        if (self.cmd_addr & 0x3f) == 0 {
            self.fpga.flush();
            self.backoff();
        }
    }

    pub fn finish(&self) {
        println!("finish");

        self.fpga.flush();

        let mut cmds = SendBuffer::default();
        let cmd = 2 | (NUM_BUCKETS as u64 - 1) << 14;
        cmds[0] = cmd;
        self.fpga.send64(self.cmd_addr, &cmds);
        self.fpga.flush();

        println!("receive");
        for _ in 0..4 {
            self.fpga.receive_alloc();
        }
        println!("onward");
    }

    pub fn column_msm(&mut self, processor: impl FnOnce(&mut App) -> ()) {
        self.start();
        processor(self);
        self.finish();
    }

    pub fn msm(&mut self, mut scheduler: impl Scheduler) {
        let t0 = SystemTime::now();
        let mut t1 = SystemTime::now();

        let mut cmds = SendBuffer::default();
        for column in 0..COLUMNS {
            // dbg!(column);

            #[cfg(feature = "log-cmd")]
            use std::io::Write;
            #[cfg(feature = "log-cmd")]
            let suffix = std::env::args().nth(2).unwrap_or("".to_string());
            #[cfg(feature = "log-cmd")]
            let mut file =
                std::fs::File::create(format!("column-{:02}{}.log", column, suffix)).unwrap();

            scheduler.set_column(column as _);

            self.start();

            // send cmd_point
            let mut more_to_come = true;
            // let mut prev_skip = false;
            #[allow(unused_variables)]
            let mut cycle = 0;
            while more_to_come {
                cmds.fill(0);
                let cmds_iterator = cmds.iter_mut();
                #[cfg(feature = "single-cmd")]
                let cmds_iterator = cmds_iterator.take(1);

                for cmd in cmds_iterator {
                    *cmd = scheduler.schedule().unwrap_or_else(|| {
                        more_to_come = false;
                        0
                    });
                    #[cfg(feature = "log-cmd")]
                    if Command(*cmd).point() > (1 << 26) - 1_000_000 {
                        writeln!(
                            &mut file,
                            "{:06},{:016X},{},{}",
                            cycle,
                            *cmd,
                            Command(*cmd).ins(),
                            serde_json::to_string(&Instruction::from(*cmd)).unwrap()
                        )
                        .unwrap();
                    }
                    #[cfg(feature = "trace-cmd")]
                    println!(
                        "{:06},{:016X},{},{}",
                        cycle,
                        *cmd,
                        Command(*cmd).ins(),
                        serde_json::to_string(&Instruction::from(*cmd)).unwrap()
                    );
                    cycle += 1;
                }
                self.update(&cmds);
            }
            #[cfg(feature = "log-cmd")]
            file.flush().unwrap();

            t1 = SystemTime::now();
            self.finish();
            println!("skips: {}", scheduler.skips());
        }

        let t2 = SystemTime::now();
        println!("time finish: {}", diff(t1, t2));
        println!("time total: {}", diff(t0, t2));
    }

    // pub fn msm_par(&mut self, digits: Digits<'_>, mut scheduler0: impl Scheduler, mut scheduler1: impl Scheduler) {
    //     let t0 = SystemTime::now();
    //     let mut t1 = SystemTime::now();
    //
    //     crossbeam::thread::scope(|scope| {
    //
    //         // let s1 = Vec::with_capacity(1 << 26);
    //         // let s0 = Arc::new(Mutex::new(Vec::with_capacity(1 << 26)));//s1);
    //         let s1 = Arc::new(Mutex::new(Vec::with_capacity(1 << 26)));//s1);
    //         // let s1: Arc<Mutex<Vec<SendBuffer>>> = Arc::new(s1);
    //
    //         let s1_clone = s1.clone();
    //
    //         let mut h1;
    //
    //         h1 = scope.spawn(move |_| {
    //             let handle = &mut s1_clone.lock().expect("can lock");
    //
    //             let mut scheduler1 = Greedy::new(digits);
    //
    //             let mut cmds = SendBuffer::default();
    //             scheduler1.set_column(1);
    //
    //             #[allow(unused_variables)]
    //             let mut more_to_come = true;
    //             println!("in thread");
    //             while more_to_come {
    //                 cmds.fill(0);
    //                 let cmds_iterator = cmds.iter_mut();
    //
    //                 for cmd in cmds_iterator {
    //                     *cmd = scheduler1.schedule()
    //                         .unwrap_or_else(|| { more_to_come = false; 0 });
    //                 }
    //                 handle.push(cmds);
    //             }
    //         });
    //
    //         // column 0
    //
    //         let time0 = SystemTime::now();
    //         let mut cmds = SendBuffer::default();
    //         let mut scheduler0 = Greedy::new(digits);
    //         scheduler0.set_column(0);
    //         self.start();
    //
    //         #[allow(unused_variables)]
    //         let mut more_to_come = true;
    //         while more_to_come {
    //             cmds.fill(0);
    //             let cmds_iterator = cmds.iter_mut();
    //
    //             for cmd in cmds_iterator {
    //                 *cmd = scheduler0.schedule()
    //                     .unwrap_or_else(|| { more_to_come = false; 0 });
    //             }
    //             self.update(&cmds);
    //         }
    //
    //         self.finish();
    //         println!("column 0 = {:?}", time0.elapsed().unwrap());
    //
    //         // column 1
    //
    //         let time1 = SystemTime::now();
    //         println!("waiting for threaded scheduler to finish");
    //         h1.join().unwrap();
    //         println!("column 1 join = {:?}", time1.elapsed().unwrap());
    //         let time1b = SystemTime::now();
    //
    //         self.start();
    //
    //         #[allow(unused_variables)]
    //         for cmds in s1.lock().unwrap().iter() {
    //             self.update(&cmds);
    //         }
    //
    //         self.finish();
    //         println!("column 1 = {:?}", time1.elapsed().unwrap());
    //         println!("column 1b = {:?}", time1b.elapsed().unwrap());
    //
    //         // columns 2..16
    //
    //         let mut cmds = SendBuffer::default();
    //
    //         println!("other columns");
    //
    //         for column in 2..COLUMNS {
    //             scheduler0.set_column(column as _);
    //
    //             self.start();
    //
    //             #[allow(unused_variables)]
    //             let mut cycle = 0;
    //             let mut more_to_come = true;
    //             while more_to_come {
    //                 cmds.fill(0);
    //                 let cmds_iterator = cmds.iter_mut();
    //
    //                 for cmd in cmds_iterator {
    //                     *cmd = scheduler0.schedule()
    //                         .unwrap_or_else(|| { more_to_come = false; 0 });
    //                     cycle += 1;
    //                 }
    //                 self.update(&cmds);
    //             }
    //
    //             t1 = SystemTime::now();
    //             self.finish();
    //         }
    //     }).expect("success");
    //
    //     let t2 = SystemTime::now();
    //     println!("time finish: {}", diff(t1, t2));
    //     println!("time total: {}", diff(t0, t2));
    // }
}

// pub trait Schedule: Iterator<Item=u64> {}

pub trait Scheduler: Iterator<Item = u64> + Send {
    // type Schedule: Iterator<Item=u64>;

    fn set_column(&mut self, column: u8);
    // fn schedule(&mut self) -> Self::Schedule {
    fn schedule(&mut self) -> Option<u64> {
        self.next()
    }
    fn skips(&self) -> usize {
        0
    }
}

#[derive(Default)]
pub struct ConflictFree {
    i: usize,
    len: usize,
}

impl ConflictFree {
    pub fn new(size: u8) -> Self {
        Self {
            i: 0,
            len: 1 << size,
        }
    }
}

impl Scheduler for ConflictFree {
    fn set_column(&mut self, _: u8) {
        self.i = 0;
    }
}

impl Iterator for ConflictFree {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        if self.i < self.len {
            let instruction = Instruction {
                point: self.i as _,
                process: true,
                sram: (self.i & SRAM_MASK) as _,
                negate: false,
                bucket: (self.i & NUM_BUCKETS_MASK) as _,
            };

            self.i += 1;

            Some(instruction.serialize())
        } else {
            None
        }
    }
}

// #[derive(Debug, Eq, PartialEq)]
// pub struct PostponedInstruction {
//     cycle: usize,
//     cmd: u64,
// }
//
// impl PartialOrd for PostponedInstruction {
//     fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
//         Some(self.cmp(&other))
//     }
// }
//
// impl Ord for PostponedInstruction {
//     fn cmp(&self, other: &Self) -> cmp::Ordering {
//         self.cycle.cmp(&other.cycle).reverse()
//     }
// }

/// A greedy scheduler.
///
/// It loops over the digits of a column, and consumes each digit
/// as soon as possible.
///
/// If the FPGA is currently processing the bucket corresponding to
/// the digit, it is kept in a queue with scheduling information.
#[allow(dead_code)]
pub struct Greedy<'a> {
    len: u32,

    // scalars: &'a [Scalar],
    digits: Digits<'a>,
    column: Column<'a>,
    // columns: &'a [Column],

    // column: u8,
    digit: Digit,
    /// next scalar to process
    point: u32,
    /// next scalar to move to SRAM
    next_sram_point: u32,
    /// presumed cycle of the FPGA
    cycle: usize,
    /// how many points have we sent actual commands for
    processed: u32,

    /// cycle at which bucket is available
    // available: Map<u16, usize>,
    available: [usize; NUM_BUCKETS],
    /// cycle at which to pop instruction
    /// NB: push in Reverse order (!)
    postponed: Vec<Option<u64>>,
    /// location of point in SRAM, if known and not yet processed
    sram: Vec<u16>,

    skips: usize,
}

impl Scheduler for Greedy<'_> {
    fn set_column(&mut self, column: u8) {
        println!(":: column {:02}", column);

        self.column = self.digits.column(column);
        self.point = 0;
        self.digit = *self.column.next().expect("non-empty scalars");
        self.processed = 0;
        self.cycle = 0;
        self.skips = 0;

        self.sram.fill(0xFFFF);
        for point in 0..SRAM_SIZE.min(self.len) {
            self.sram[point as usize] = point as u16;
        }
        self.next_sram_point = SRAM_SIZE;

        self.available.fill(0);
        self.postponed = vec![None; self.len as usize * 2];
    }
    fn skips(&self) -> usize {
        self.skips
    }
}

/// A column of digits, ready for scheduling
pub type Column<'a> = iter::StepBy<iter::Skip<slice::Iter<'a, Digit>>>;

#[derive(Copy, Clone)]
/// Decomposition of scalars into column digits.
pub struct Digits<'a> {
    digits: &'a [Digit],
    len: usize,
}

impl<'a> Digits<'a> {
    /// For u16 digits, this has zero overhead.
    pub fn new(scalars: &'a [Scalar]) -> Self {
        let digits: &'a [Digit] = unsafe {
            slice::from_raw_parts(&scalars[0] as *const u64 as *const u16, scalars.len() * 16)
        };
        Self {
            digits,
            len: scalars.len(),
        }
    }

    pub fn column(&self, column: u8) -> Column<'a> {
        self.digits.iter().skip(column as _).step_by(16)
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl<'a> Greedy<'a> {
    pub fn new(digits: Digits<'a>) -> Self {
        let len = digits.len();
        let mut sram = vec![0xFFFF; len];
        for point in 0..SRAM_SIZE.min(len as u32) {
            sram[point as usize] = point as u16;
        }

        let mut column = digits.column(0);
        let digit = *column.next().expect("non-empty scalars");
        // println!("sram {:?}", &sram[..8]);
        Greedy {
            // inputs
            len: len as u32,
            digits,

            column,
            digit,
            skips: 0,

            // counters
            // column: 0,
            point: 0,
            next_sram_point: SRAM_SIZE as _,
            cycle: 0,
            processed: 0,

            // data structures
            available: [0; NUM_BUCKETS],
            // TODO: length of this?! use NonZeroU64 so the Option fits in one long word?
            postponed: vec![None; 2 * len],
            sram,
        }
    }

    fn increment_point(&mut self) {
        self.point += 1;
        self.digit = self.column.next().copied().unwrap_or(0); //.expect("have digit");
    }

    fn increment_cycle(&mut self) {
        self.cycle += 1;
    }

    fn refresh_sram(&mut self, cmd: u64) {
        let cmd = Command(cmd);
        if self.next_sram_point < self.len {
            self.sram[self.next_sram_point as usize] = cmd.sram();
            self.next_sram_point += 1;
        }
    }

    fn point_unavailable(&self) -> bool {
        self.sram[self.point as usize] == 0xFFFF
    }

    fn sram(&self, point: u32) -> u16 {
        let sram = self.sram[point as usize];
        assert_ne!(sram, 0xFFFF);
        sram
    }

    /// returns digit for current point (in current column)
    fn digit(&self) -> Digit {
        self.digit
        // self.columns[self.column as usize][self.point as usize]
    }
}

impl Iterator for Greedy<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        // A) termination condition:
        // if self.done() { return None }
        if self.processed >= self.len {
            //) && self.point >= self.len {
            return None;
        }

        // B) greedy condition: cmd waiting at this cycle
        // dbg!(self.cycle, self.postponed.len());
        if let Some(&Some(cmd)) = self.postponed.get(self.cycle) {
            #[cfg(feature = "trace-queue")]
            {
                println!("<- {:?}", Command(cmd));
            }
            // TODO: revisit

            self.refresh_sram(cmd);
            // self.block_bucket(cmd);
            self.increment_cycle();
            self.processed += 1;

            return Some(cmd);
        }

        //** loop here (?!) **//

        loop {
            // println!("checking if {} is available, queue {}, len{}", self.point, self.postponed.len(), self.len);
            if self.point >= self.len {
                self.increment_cycle();
                return Some(Instruction::WASTE_CYCLE);
            }

            if self.point_unavailable() {
                self.increment_cycle();
                return Some(Instruction::WASTE_CYCLE);
            }

            // distinguish cases, based on digit for next scalar
            let digit = self.digit();
            // println!("dealing with {:02} {:04X}", self.point, self.digit);

            // C0) zero => handle straight away
            if digit == 0 {
                // println!("digit is zero");
                let cmd = Instruction {
                    point: self.point,
                    process: false,
                    sram: self.sram(self.point),
                    negate: false,
                    // there is no zero bucket - this is the +/- 1 bucket
                    bucket: 0,
                }
                .serialize();

                self.refresh_sram(cmd);
                self.increment_point();
                self.increment_cycle();
                self.processed += 1;

                // println!("zero");
                return Some(cmd);
            }

            // Cn) non-zero => depends on whether bucket is available or not
            let bucket = digit;
            let cmd = Instruction {
                point: self.point,
                process: true,
                sram: self.sram(self.point),
                negate: false,
                bucket,
            }
            .serialize();

            let blocked_until = self.available[bucket as usize];
            // dbg!(self.cycle, blocked_until);
            if self.cycle >= blocked_until {
                // println!("avail");

                self.refresh_sram(cmd);
                self.available[bucket as usize] = self.cycle + CYCLES;
                self.increment_point();
                self.increment_cycle();
                self.processed += 1;

                return Some(cmd);
            } else {
                let cmd = Instruction {
                    point: self.point,
                    process: false,
                    sram: self.sram(self.point),
                    negate: false,
                    // there is no zero bucket - this is the +/- 1 bucket
                    bucket: 0,
                }
                .serialize();

                self.refresh_sram(cmd);
                self.increment_point();
                self.increment_cycle();
                self.processed += 1;
                self.skips += 1;

                // println!("zero");
                return Some(cmd);

                // // println!("unavail");
                //
                // let l = self.postponed.len();
                // // dbg!(blocked_until, l);
                // if blocked_until >= l {
                //     println!("need to extend postponed array from {} to {}", l, blocked_until * 2);
                //     self.postponed.resize(blocked_until * 2, None);
                // }
                // assert!(blocked_until < self.postponed.len());
                // self.postponed[blocked_until] = Some(cmd);
                // #[cfg(feature = "trace-queue")] {
                //     println!("-> {:05} {:?}", blocked_until, Command(cmd));
                // }
                // self.available[bucket as usize] += CYCLES;
                // self.increment_point();
            }
        }
    }
}

pub fn random_scalars(size: u8) -> Vec<Scalar> {
    use rand_core::{RngCore, SeedableRng};
    let mut rng = rand::prelude::StdRng::from_entropy();

    (0..(1 << size))
        .map(|_| {
            let mut scalar = Scalar::default();
            for limb in scalar.iter_mut() {
                // TODO: remove this again - it's for testing u16 digits with an i16 FPGA impl
                *limb = rng.next_u64() & 0x7FFF_7FFF_7FFF_7FFF;
            }
            scalar
        })
        .collect()
}

pub fn conflict_free_scalars(size: u8) -> Vec<Scalar> {
    (0..(1 << size))
        .map(|i| {
            let digit = i % 1024;
            [digit; 4]
        })
        .collect()
}

fn time<R>(name: &str, f: impl FnOnce() -> R) -> R {
    println!("{} ...", name);
    let t = SystemTime::now();
    let r = f();
    println!("... {:?}", t.elapsed());
    r
}

// /// Wrapper around a base scheduler which
// /// pre-schedules.
// pub struct Prescheduler<'a> {
//     // digits: Digits<'a>,
//     greedy_0: Greedy<'a>,
//     greedy_1: Greedy<'a>,
//     schedule_0: Vec<u64>,
//     schedule_1: Vec<u64>,
// }
//
// impl<'a> Prescheduler<'a> {
//     // pub fn new(digits: Digits<'a>) -> Self {
//     //     let prescheduler = Prescheduler {
//     //
//     //     }
//     // }
// }
//
// impl Scheduler for Prescheduler<'_> {
//     fn set_column(&mut self, _column: u8) {
//         todo!();
//     }
// }
//
// impl Iterator for Prescheduler<'_> {
//     type Item = u64;
//
//     fn next(&mut self) -> Option<u64> {
//         todo!();
//     }
// }

fn main() {
    let size = std::env::args()
        .nth(1)
        .expect("pass with SIZE argument")
        .parse()
        .expect("SIZE invalid as u8");

    let f1 = fpga::F1::new(0, 0x500).unwrap();
    let mut app = App::new(f1, size);
    app.stats();

    let scalars = time("randoming", || random_scalars(size));
    let digits = Digits::new(&scalars);

    // time("cloning scalars", || scalars.clone());
    // // let scalars = time("conflict-free scalars", || conflict_free_scalars(size));
    // let d0: Vec<Digit> = digits.column(3).copied().collect();
    // let mut d1 = vec![0; d0.len()];
    // time("cloning digits", || d1.copy_from_slice(&d0));
    // let scheduler0 = Greedy::new(digits);
    // let scheduler1 = Greedy::new(digits);

    let scheduler = Greedy::new(digits);

    // let scheduler = ConflictFree::new(size);

    println!("start");
    // time("msm", || app.msm_par(digits, scheduler0, scheduler1));
    time("msm", || app.msm(scheduler));
    app.stats();
}
