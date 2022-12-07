use fpga::Fpga;
use std::time::SystemTime;

fn diff(t0: SystemTime, t1: SystemTime) -> f64 {
    t1.duration_since(t0).unwrap().as_secs_f64()
}

fn stats(f1: &fpga::F1) {
    f1.write_register(0x10, 0);
    println!("missed  = {}", f1.read_register(0x20));
    f1.write_register(0x10, 1);
    println!("bubbles = {}", f1.read_register(0x20));
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Instruction {
    point_index: u32,
    skip: bool,
    sram_index: u16,
    negate: bool,
    bucket_index: u16,
}

impl Instruction {
    #[inline]
    pub fn serialize(self) -> u64 {
        let Instruction {
            negate,
            sram_index,
            bucket_index,
            skip,
            point_index,
        } = self;

        // bits   len content             model
        // ----------------------------------------
        // 0:3    4   command, = 3
        // 4      1   negate              bool
        // 5:13   9   SRAM index, 0..512  u16
        // 14:29  16  bucket index        u16
        // 30     1   process             bool
        // 31:63  33  point index         usize

        let mut entry = 3;

        entry |= (negate as u64) << 4;

        debug_assert!(sram_index < 1 << 9);
        // entry |= (sram_index as u64 & ((1 << 9) - 1)) << 5;
        entry |= (sram_index as u64) << 5;

        debug_assert!(bucket_index < 1 << 16);
        // entry |= (bucket_index as u64 & ((1 << 16) - 1)) << 14;
        entry |= (bucket_index as u64) << 14;

        entry |= (!skip as u64) << 30;
        entry |= (point_index as u64) << 31;

        entry
    }
}

fn main() {
    let f1 = fpga::F1::new(0, 0x500).unwrap();
    stats(&f1);

    let s: u32 = std::env::args().nth(1).unwrap().parse().unwrap();
    #[allow(non_snake_case)]
    let S: u32 = 1 << s;
    let w_s: u32 = 16;
    let buffer_sz: u32 = 512;
    let buffer_m = buffer_sz as u64 - 1;
    let no_buckets = 1 << (w_s - 1);
    let no_buckets_m = no_buckets - 1;

    let mut cmds = fpga::SendBuffer::default();

    f1.write_register(0x20, S);
    f1.write_register(0x11, 256);

    println!("start");

    // clock starts
    let t0 = SystemTime::now();
    let mut t1 = SystemTime::now();

    let mut cmd_addr: usize;

    for column in 0..16 {
        dbg!(column);

        // send cmd_start
        cmd_addr = 4 << 26;
        cmds.fill(0);
        cmds[0] = 1;
        f1.send(cmd_addr, &cmds);
        f1.flush();
        cmd_addr += 1;

        // send cmd_point
        let mut pi: u64 = 0;
        for i in (0..S).step_by(8) {
            for j in 0..8 {
                let instruction = Instruction {
                    point_index: pi as _,
                    skip: false,
                    sram_index: (pi & buffer_m) as _,
                    negate: false,
                    bucket_index: (pi & no_buckets_m) as _,
                };

                let cmd = instruction.serialize();
                cmds[j * 8..][..8].copy_from_slice(&cmd.to_le_bytes());

                pi += 1;
            }

            f1.send(cmd_addr, &cmds);

            if (cmd_addr & 0x3f) == 0 {
                f1.flush();
            }

            cmd_addr += 1;

            if (i & 0x03ff) == 0 {
                while f1.read_register(0x21) > 256 {
                    continue;
                }
            }
        }

        f1.flush();

        t1 = SystemTime::now();

        // send cmd_finish
        cmds.fill(0);
        let mut cmd = 2;
        cmd |= (no_buckets - 1) << (0 + 4 + 1 + 9);
        cmds[..8].copy_from_slice(&cmd.to_le_bytes());
        f1.send(cmd_addr, &cmds);
        f1.flush();

        for _ in 0..4 {
            f1.receive_alloc();
        }
    }

    let t2 = SystemTime::now();
    println!("time 0-1: {}", diff(t0, t1));
    println!("time 1-2: {}", diff(t1, t2));
    println!("time 0-2: {}", diff(t0, t2));
    stats(&f1);
}
