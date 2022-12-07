use core::fmt;

use ark_std::Zero;
use our_bls12_377::{Fq, G1Affine, G1TEProjective};
use our_ff::{FromBytes, ToBytes};

use fpga::{Fpga, ReceiveBuffer, SendBuffer as SendBuffer8, SendBuffer64 as SendBuffer};

use crate::{
    digits::single_digit_carry, limb_carries, timed, twisted::into_weierstrass, G1PTEAffine,
    G1Projective, Scalar,
};

const DDR_READ_LEN: u32 = 64;

const NUM_BUCKETS: u32 = 1 << 15;
#[allow(dead_code)]
const FIRST_BUCKET: u32 = 0;
const LAST_BUCKET: u32 = NUM_BUCKETS - 1;

const BACKOFF_THRESHOLD: u32 = 64;
// const FLUSH_BACKOFF_EVERY: usize = 512;
const FLUSH_BACKOFF_EVERY: usize = 512;

#[repr(usize)]
#[derive(Copy, Clone, Debug)]
/// Top-level commands of the FPGA App's interface.
/// Of more interest are the subcommands of `Cmd::Msm` in [`Command`].
pub enum Cmd {
    SetX = 1 << 26,
    SetY = 2 << 26,
    SetZ = 3 << 26,
    // family of subcommands, packed + encoded in the payload
    Msm = 4 << 26,
    SetZero = 5 << 26,
}

impl Cmd {
    pub fn addr(self, addr: usize) -> usize {
        self as usize | addr
    }
}

pub struct App {
    pub fpga: fpga::F1,
    len: usize,
    cmd_addr: usize,
    pool: Option<rayon::ThreadPool>,
    carried: Option<Vec<Scalar>>,
}

impl App {
    pub fn new(fpga: fpga::F1, size: u8) -> Self {
        assert!(size < 32);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let mut app = App {
            fpga,
            len: 1 << size,
            cmd_addr: 0,
            pool: Some(pool),
            carried: Some(vec![Scalar::default(); 1 << size]),
        };
        app.set_size();
        app.set_first_bucket();
        app.set_last_bucket();
        app.set_ddr_read_len();
        app.set_zero();

        app
    }

    /// Perform full MSM.
    #[inline]
    pub fn msm(&mut self, scalars: &[Scalar]) -> G1Projective {
        assert_eq!(scalars.len(), self.len);

        let pool = self.pool.take().unwrap_or_else(|| unreachable!());
        let mut carried = self.carried.take().unwrap_or_else(|| unreachable!());

        let mut cmds = SendBuffer::default();
        let mut total = G1TEProjective::zero();
        let mut total0 = G1TEProjective::zero();
        pool.scope(|s| {
            s.spawn(|_| {
                timed("limb carries", || limb_carries(scalars, &mut carried));
            });

            s.spawn(|_| {
                for j in (0..4).rev() {
                    timed(&format!("\n:: column {}", j as usize), || {
                        self.start();

                        for chunk in scalars.chunks(8) {
                            for (cmd, scalar) in cmds.iter_mut().zip(chunk) {
                                let digit = single_digit_carry(scalar, 0, j);
                                *cmd = Instruction::new(digit);
                            }
                            self.update(&cmds);
                        }

                        self.flush();

                        total0 += timed("fetching point", || self.get_point());
                        if j != 0 {
                            total0 <<= 16;
                        }
                    });
                }
            });
        });

        for i in (1..4).rev() {
            for j in (0..4).rev() {
                timed(&format!("\n:: column {}", i * 4 + j as usize), || {
                    self.start();

                    for chunk in carried.chunks(8) {
                        for (cmd, scalar) in cmds.iter_mut().zip(chunk) {
                            let digit = single_digit_carry(scalar, i, j);
                            *cmd = Instruction::new(digit);
                        }
                        self.update(&cmds);
                    }

                    self.flush();

                    total += timed("fetching point", || self.get_point());
                    total <<= 16;
                });
            }
        }

        total <<= 48;
        total += total0;

        let total = into_weierstrass(&total);
        self.pool = Some(pool);
        self.carried = Some(carried);
        total
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn set_zero(&mut self) {
        let zero = G1TEProjective::zero();
        let mut buffer = SendBuffer8::default();

        zero.x.0.write(&mut buffer[..48]).unwrap();
        self.fpga.send(Cmd::SetZero.addr(0), &buffer);
        zero.y.0.write(&mut buffer[..48]).unwrap();
        self.fpga.send(Cmd::SetZero.addr(1), &buffer);
        zero.z.0.write(&mut buffer[..48]).unwrap();
        self.fpga.send(Cmd::SetZero.addr(2), &buffer);
        zero.t.0.write(&mut buffer[..48]).unwrap();
        self.fpga.send(Cmd::SetZero.addr(3), &buffer);

        self.fpga.flush();
    }

    fn set_size(&mut self) {
        self.fpga.write_register(0x20, self.len as _);
    }

    fn set_last_bucket(&mut self) {
        self.fpga.write_register(0x21, LAST_BUCKET);
    }

    fn set_first_bucket(&mut self) {
        self.fpga.write_register(0x22, FIRST_BUCKET);
    }

    fn set_ddr_read_len(&mut self) {
        self.fpga.write_register(0x11, DDR_READ_LEN);
    }

    #[inline]
    fn set_point(&mut self, point: &G1PTEAffine, index: usize) {
        let mut buffer = SendBuffer8::default();

        // NB: Use `point.x.0.write` for the Montgomery representation,
        // as `point.x.write` converts to BigInt/non-Montgomery first.

        point.x.0.write(&mut buffer[..48]).unwrap();
        self.fpga.send(Cmd::SetX.addr(index), &buffer);

        point.y.0.write(&mut buffer[..48]).unwrap();
        self.fpga.send(Cmd::SetY.addr(index), &buffer);

        point.kt.0.write(&mut buffer[..48]).unwrap();
        self.fpga.send(Cmd::SetZ.addr(index), &buffer);
    }

    #[inline]
    pub fn set_preprocessed_points(&mut self, points: &[G1PTEAffine]) {
        assert!(self.len == points.len());

        let mut index = 0;
        let bar = indicatif::ProgressBar::new(self.len as _);
        const CHUNK: usize = 1024;
        for chunk in points.chunks(CHUNK) {
            for point in chunk.iter() {
                self.set_point(point, index);
                index += 1;
            }
            bar.inc(CHUNK as _);

            // flush from time to time
            self.fpga.flush();
        }
    }

    pub fn set_points(&mut self, points: &[G1Affine]) {
        assert!(self.len == points.len());

        let mut index = 0;
        let bar = indicatif::ProgressBar::new(self.len as _);
        const CHUNK: usize = 1024;
        for chunk in points.chunks(CHUNK) {
            for point in chunk.iter() {
                self.set_point(&crate::twisted::into_preprocessed(point), index);
                index += 1;
            }
            bar.inc(CHUNK as _);

            // flush from time to time
            self.fpga.flush();
        }
    }

    pub fn set_preprocessed_point_repeatedly(&mut self, point: &G1PTEAffine) {
        let mut index = 0;
        let bar = indicatif::ProgressBar::new(self.len as _);
        const CHUNK: usize = 256;
        for _ in (0..self.len).step_by(CHUNK) {
            for _ in 0..CHUNK {
                self.set_point(point, index);
                index += 1;
            }
            bar.inc(CHUNK as u64);

            // flush from time to time
            self.fpga.flush();
        }
    }

    pub fn get_point(&mut self) -> G1TEProjective {
        let mut buffer = ReceiveBuffer::default();
        #[allow(unused_mut)]
        let mut point = G1TEProjective::zero();

        #[cfg(feature = "hw")]
        {
            self.fpga.receive(&mut buffer);
            point.x = Fq::read(&buffer[..48]).unwrap();

            self.fpga.receive(&mut buffer);
            point.y = Fq::read(&buffer[..48]).unwrap();

            self.fpga.receive(&mut buffer);
            point.z = Fq::read(&buffer[..48]).unwrap();

            self.fpga.receive(&mut buffer);
            point.t = Fq::read(&buffer[..48]).unwrap();
        }
        #[cfg(not(feature = "hw"))]
        {
            self.fpga.receive(&mut buffer);
            self.fpga.receive(&mut buffer);
            self.fpga.receive(&mut buffer);
            self.fpga.receive(&mut buffer);
        }

        point
    }

    pub fn flush(&self) {
        self.fpga.flush()
    }

    pub fn backoff(&mut self) {
        let mut show = true;
        let mut back = self.fpga.read_register(0x21);
        while back > BACKOFF_THRESHOLD {
            if show {
                // println!("backing off at {} cmd {}", back, self.cmd_addr & ((1 << 26) - 1));
                // self.print_stats();
            }
            show = false;
            back = self.fpga.read_register(0x21);
            continue;
        }
    }

    pub fn print_stats(&mut self) {
        println!("dropped cmds    = {}", self.missed());
        println!("DDR read miss   = {}", self.bubbles());
        println!("DDR write miss  = {}", self.register(2));
        // DDR push count
        println!("reg[3]          = {}", self.register(3));
        // DDR read count (all channels)
        println!("reg[4]          = {}", self.register(4));
        println!("reg[5]          = {}", self.register(5));
        println!("reg[6]          = {}", self.register(6));
        // println!("reg[7]          = {}", self.register(7));
        // println!("reg[8]          = {}", self.register(8));
    }

    pub fn register(&mut self, reg: u32) -> u32 {
        self.fpga.write_register(0x10, reg);
        self.fpga.read_register(0x20)
    }

    // DDR not taking fast enough
    pub fn something(&mut self) -> u32 {
        self.fpga.write_register(0x10, 2);
        self.fpga.read_register(0x20)
    }

    // DDR not responding fast enough
    pub fn bubbles(&mut self) -> u32 {
        self.fpga.write_register(0x10, 1);
        self.fpga.read_register(0x20)
    }

    // dropped commands
    pub fn missed(&mut self) -> u32 {
        self.fpga.write_register(0x10, 0);
        self.fpga.read_register(0x20)
    }

    pub fn start(&mut self) {
        self.cmd_addr = 4 << 26;
        let mut cmds = SendBuffer::default();
        cmds[0] = 1;
        self.fpga.send64(self.cmd_addr, &cmds);
        self.fpga.flush();
        self.cmd_addr += 1;
    }

    #[inline]
    pub fn update(&mut self, commands: &SendBuffer) {
        self.fpga.send64(self.cmd_addr, commands);
        self.cmd_addr += 1;

        if (self.cmd_addr & (FLUSH_BACKOFF_EVERY - 1)) == 0 {
            self.fpga.flush();
            self.backoff();
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Instruction {
    pub point: u32,
    pub process: bool,
    pub sram: u16,
    pub negate: bool,
    pub digit: i16,
    // pub bucket: u16,
}

pub struct Command(pub u64);

impl fmt::Debug for Command {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        f.write_fmt(format_args!(
            "ins {} point {:04} process {} sram {:03X} digit {:04X}",
            self.0 & 0b111,
            self.point(),
            (self.0 >> 29) & 1,
            self.sram(),
            self.digit()
        ))
    }
}

impl Command {
    #[inline]
    pub fn digit(&self) -> i16 {
        ((self.0 >> 14) as u16) as i16
    }

    #[inline]
    pub fn point(&self) -> u32 {
        (self.0 >> 31) as _
    }

    #[inline]
    pub fn sram(&self) -> u16 {
        ((self.0 >> 5) as u16) & ((1 << 9) - 1)
    }

    #[inline]
    pub fn process(&self) -> bool {
        (self.0 >> 30) & 1 != 0
    }

    #[inline]
    pub fn negate(&self) -> bool {
        (self.0 >> 4) & 1 != 0
    }

    #[inline]
    pub fn ins(&self) -> u8 {
        self.0 as u8 & ((1 << 4) - 1)
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
            digit: cmd.digit(),
        }
    }
}

impl Instruction {
    pub const WASTE_CYCLE: u64 = 4;

    #[inline]
    #[allow(clippy::new_ret_no_self)]
    pub fn new(digit: i16) -> u64 {
        3u64 | (digit as u16 as u64) << 14
    }

    #[inline]
    pub fn serialize(self) -> u64 {
        let Instruction {
            negate,
            sram,
            digit,
            process,
            point,
        } = self;

        // bits   len content             model
        // ----------------------------------------
        // 0:3    4   command, = 3
        // 4      1   negate              bool
        // 5:13   9   SRAM index, 0..512  u16
        // 14:29  16  signed digit        i16
        // 30     1   process             bool
        // 31:63  33  point index         usize

        let mut entry = 3;

        entry |= (negate as u64) << 4;

        debug_assert!(sram < (1 << 9));
        entry |= (sram as u64) << 5;

        entry |= ((digit as u64) & 0xFFFF_FFFF) << 14;

        entry |= (process as u64) << 30;

        entry |= (point as u64) << 31;

        entry
    }
}
