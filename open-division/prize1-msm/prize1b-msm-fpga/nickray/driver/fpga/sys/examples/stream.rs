use fpga_sys as sys;
use std::time::SystemTime;

fn diff(t0: SystemTime, t1: SystemTime) -> f64 {
    t1.duration_since(t0).unwrap().as_secs_f64()
}

#[repr(align(64))]
struct Aligner;

pub struct Buffer {
    __: [Aligner; 0],
    buf: [u64; 8],
}

fn stats() {
    unsafe {
        sys::write_32_f1(0x10 << 2, 0);
        println!("missed  = {}", sys::read_32_f1(0x20 << 2));
        sys::write_32_f1(0x10 << 2, 1);
        println!("bubbles = {}", sys::read_32_f1(0x20 << 2));
    }
}

fn main() {
    unsafe {
        sys::init_f1(0, 0x500);
        stats();

        // seems to break for s > 28
        let s: u32 = std::env::args().nth(1).unwrap().parse().unwrap();
        #[allow(non_snake_case)]
        let S: u32 = 1 << s;
        let w_s: u32 = 16;
        let buffer_sz: u32 = 512;
        let buffer_m = buffer_sz as u64 - 1;
        let no_buckets = 1 << (w_s - 1);
        let no_buckets_m = no_buckets - 1;

        let mut cmds_aligned = Buffer {
            __: [],
            buf: [0u64; 8],
        };
        let cmds = &mut cmds_aligned.buf;
        println!("&cmds: {:p}", cmds);

        sys::write_32_f1(0x20 << 2, S); // no_points
        sys::write_32_f1(0x11 << 2, 256); // ddr_rd_len

        println!("start");

        // clock starts
        let t0 = SystemTime::now();
        let mut t1 = SystemTime::now();

        let mut cmd_a: u64;

        for column in 0..16 {
            dbg!(column);

            // send cmd_start
            cmd_a = 4 << 32;
            cmds.fill(0);
            // memset((void*)cmds, 0, 64);
            cmds[0] = 1;
            sys::write_512_f1(cmd_a, &mut cmds[0] as *mut u64 as *mut cty::c_void);
            sys::write_flush();
            cmd_a += 64;

            // send cmd_point
            let mut pi: u64 = 0;
            for i in (0..S).step_by(8) {
                for j in 0..8 {
                    cmds[j] = 0;
                    cmds[j] |= 3 << (0);
                    cmds[j] |= 0 << (0 + 4);
                    cmds[j] |= (pi & buffer_m) << (0 + 4 + 1);
                    cmds[j] |= (pi & no_buckets_m) << (0 + 4 + 1 + 9);
                    cmds[j] |= (1) << (0 + 4 + 1 + 9 + 16);
                    cmds[j] |= (pi) << (0 + 4 + 1 + 9 + 16 + 1);

                    pi += 1;
                }
                sys::write_512_f1(cmd_a, &mut cmds[0] as *mut u64 as *mut cty::c_void);
                if ((cmd_a >> 6) & 0x3f) == 0 {
                    sys::write_flush();
                }
                cmd_a += 64;

                if (i & 0x03ff) == 0 {
                    while sys::read_32_f1(0x21 << 2) > 256 {
                        continue;
                    }
                }
            }

            sys::write_flush();

            t1 = SystemTime::now();

            // send cmd_finish
            cmds.fill(0);
            cmds[0] = 0;
            cmds[0] |= 2 << (0);
            cmds[0] |= (no_buckets - 1) << (0 + 4 + 1 + 9);
            sys::write_512_f1(cmd_a, &mut cmds[0] as *mut u64 as *mut cty::c_void);
            sys::write_flush();

            for _ in 0..4 {
                sys::dma_wait_512();
            }
        }

        let t2 = SystemTime::now();
        println!("time 0-1: {}", diff(t0, t1));
        println!("time 1-2: {}", diff(t1, t2));
        println!("time 0-2: {}", diff(t0, t2));
        stats();
    }
}
