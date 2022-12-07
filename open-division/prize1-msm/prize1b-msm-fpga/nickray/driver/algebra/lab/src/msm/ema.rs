use ark_ec::AffineCurve;
use std::os::raw::c_void;

use ark_ec::msm::Msm;
use ark_ff::{BigInteger as _, PrimeField, ToConstraintField};

use itertools::Itertools;
use std::{
    fmt::Debug,
    mem,
    ops::Neg,
    time::{Duration, Instant},
};


#[repr(align(64))]
#[derive(Copy, Clone, Debug)]
struct Aligner;

pub struct Aligned<T> {
    // this 0-sized, 64-byte aligned entry aligns the entire struct
    __: [Aligner; 0],
    value: T,
}

pub fn convert_scalar<Input: Copy + Neg, T: Msm<Input>>(
    bigint: &<T::Scalar as PrimeField>::BigInt,
    window_start: usize,
) -> u64 {
    let c = 16;
    let msb_window = 1 << (c - 1);

    let mut carry = 0i64;
    let mut new_limb = 0u64;

    // processing with a for-loop as in add_with_carry
    for w_start in (0..window_start + 1).step_by(c) {
        let w_word = w_start / 64;
        let w_shift = w_start % 64;
        let mut limb = carry + ((bigint.as_ref()[w_word] >> w_shift) % (1 << c)) as i64;

        // if limb >= 2^(c-1), make it "negative" borrowing from the next limb
        carry = 0;
        if limb >= msb_window {
            carry = 1;
            // note that in some (edge) cases limb can still be positive
            limb -= 1 << c;
        }

        // create new_limb encoding positive and negative values
        new_limb = if limb >= 0 {
            limb
        } else {
            (-limb - 1) | msb_window
        } as u64;
    }
    new_limb
}



struct FpgaSimulator {
    timer: Instant,
    time_first_op: Duration,

    // batch_size: usize,
    n_ops: usize,
}

impl FpgaSimulator {
    fn default() -> FpgaSimulator {
        FpgaSimulator {
            timer: Instant::now(),
            time_first_op: Duration::new(0, 0),
            // batch_size: 0,
            n_ops: 0,
        }
    }

    fn op(&mut self, _scheduled_op: u64) {
        if self.n_ops == 0 {
            self.time_first_op = self.timer.elapsed();
        }
        self.n_ops += 1;
    }
}

const USE_HW: bool = true;
use libloading::{Library, Symbol};
type WriteFlushFunc = fn() -> ();
type Write512F1Func = fn(u64, &[u64; 8]) -> ();
type Write32F1Func = fn(u32, u32) -> ();
type Read32F1Func = fn(u32) -> u32;
type DMAWait512Func = fn() -> *const u32;



#[repr(C)]
pub struct MultiScalarMultContext {
    context: *mut c_void,
}

pub fn multi_scalar_mult_init<G: AffineCurve>(_points: &[G]) -> MultiScalarMultContext {
    MultiScalarMultContext {
        context: std::ptr::null_mut(),
    }
}

pub fn multi_scalar_mult<G: AffineCurve>(
    _context: &mut MultiScalarMultContext,
    points: &[G],
    scalars: &[<G::ScalarField as PrimeField>::BigInt],
) -> Vec<G::Projective> {
    let npoints = points.len();
    if scalars.len() % npoints != 0 {
        panic!("length mismatch")
    }

    scalars
        .chunks_exact(npoints)
        .map(|batch| points[0].into())
        .collect()
}




/// Optimized implementation of multi-scalar multiplication.
pub fn msm_bigint_ema<Input: Copy + Neg<Output = Input> + Debug, T: Msm<Input> + Debug>(
    bases: &[Input],
    bigints: &[<T::Scalar as PrimeField>::BigInt],
) -> T {
    // hw init
    let lib = unsafe { Library::new("./appxfer.so").unwrap() };
    let write_flush: Symbol<WriteFlushFunc> = unsafe { lib.get(b"write_flush").unwrap() };
    let write_512_f1: Symbol<Write512F1Func> = unsafe { lib.get(b"write_512_f1").unwrap() };
    let write_32_f1: Symbol<Write32F1Func> = unsafe { lib.get(b"write_32_f1").unwrap() };
    let read_32_f1: Symbol<Read32F1Func> = unsafe { lib.get(b"read_32_f1").unwrap() };
    let dma_wait_512: Symbol<DMAWait512Func> = unsafe { lib.get(b"dma_wait_512").unwrap() };    

    let zero = T::zero();
    let mut _cmds = Aligned{ __: [], value: [0u64; 8] };
    let mut cmds = _cmds.value;

    if USE_HW {
        let init_f1: Symbol<fn(isize, isize) -> isize> = unsafe { lib.get(b"init_f1").unwrap() };
        init_f1(0, 0x500);

        write_32_f1(0x20<<2, bases.len() as u32); // no_points
        write_32_f1(0x11<<2, 256); // ddr_rd_len


        let ptr: *const u64 = unsafe { mem::transmute(&zero) };
        for i in 0..6 {
            unsafe { cmds[i] = *ptr.offset(i as isize) }
        }
        write_512_f1((5u64<<32)+(0<<6) as u64, &cmds);
        for i in 0..6 {
            unsafe { cmds[i] = *ptr.offset(6 + i as isize) }
        }
        write_512_f1((5u64<<32)+(1<<6) as u64, &cmds);
        for i in 0..6 {
            unsafe { cmds[i] = *ptr.offset(18 + i as isize) }
        }
        write_512_f1((5u64<<32)+(2<<6) as u64, &cmds);
        for i in 0..6 {
            unsafe { cmds[i] = *ptr.offset(12 + i as isize) }
        }
        write_512_f1((5u64<<32)+(3<<6) as u64, &cmds);

        println!{"sending points..."}
        for pi in 0..bases.len() {
            let ptr: *const u64 = unsafe { mem::transmute(&bases[pi]) };
            let mut buf = Aligned{ __: [], value: [0u64; 8] };
            
            // send X
            for i in 0..6 {
                unsafe { buf.value[i] = *ptr.offset(i as isize) }
            }
            write_512_f1((1u64<<32)+(pi<<6) as u64, &buf.value);

            // send Y
            for i in 0..6 {
                unsafe { buf.value[i] = *ptr.offset(6 + i as isize) }
            }
            write_512_f1((2u64<<32)+(pi<<6) as u64, &buf.value);

            // send kT
            for i in 0..6 {
                unsafe { buf.value[i] = *ptr.offset(12 + i as isize) }
            }
            write_512_f1((3u64<<32)+(pi<<6) as u64, &buf.value);

            if pi & 0xf == 0 {
                write_flush();
            }
        }
        write_flush();
    }
    println!{"finished sending points"}

    let mut time_fpga_end: Duration = Duration::new(0, 0);
    let timer = Instant::now();
    let mut fpga = FpgaSimulator {
        timer,
        ..FpgaSimulator::default()
    };

    let c = 16;
    let num_bits = T::Scalar::MODULUS_BIT_SIZE as usize;
    let msb_window = 1 << (c - 1);
    let window_starts = (0..num_bits).step_by(c).rev();
    let mut cmd_addr = 4u64 << 32;
    let mut i = 0usize;
    let mut j = 0usize;

    println!{"start... {}", num_bits}
    let scalars_and_bases_iter = bigints.iter().zip(bases); //.filter(|(s, _)| !s.is_zero());

    // Each window is of size `c`.
    // We divide up the bits 0..num_bits into windows of size `c`, and
    // in parallel process each such window.
    // let window_sums: Vec<_> = ark_std::cfg_into_iter!(window_starts)
    let result = window_starts.fold(zero, |mut total, w_start| {
        // .map(|w_start| {
        // let w_word = w_start / 64;
        // let w_shift = w_start % 64;
        // Use 2^(c-1) buckets, i.e. half
        println!{"column {}...", w_start}
        if USE_HW {
            cmds = [0u64; 8];
            cmds[0] = 1;
            write_512_f1(cmd_addr, &cmds);
            write_flush();
            cmd_addr += 64;
        }
        i = 0;
        j = 0;

        let mut buckets = vec![zero; 1 << (c - 1)];
        // This clone is cheap, because the iterator contains just a
        // pointer and an index into the original vectors.
        scalars_and_bases_iter
            .clone()
            .filter_map(|(scalar, base)| {
                // get the current window scalar in (-w/2,w/2) form.
                // internally, it propagates carry.
                // we're doing a little repeated work for each window, but negligible.
                // by doing it here vs a precompute phase, we can immediately begin the heavy
                // computation.
                let scalar = convert_scalar::<Input, T>(scalar, w_start);
                // let scalar = (scalar.as_ref()[w_word] >> w_shift) % (1 << c)) as u64;

                // if scalar == 0 {
                //     None
                // } else {
                //     Some((scalar, base))
                // }
                Some((scalar, base))
            })
            .for_each(|(scalar, base)| {
                fpga.op(scalar);

                if USE_HW {

                    let mut write = 1u64;
                    let mut sign = 0u64;
                    let mut bucket = 0u64;
                    if scalar == 0 {
                        bucket = 0;
                        write = 0;
                    } else {
                        if scalar & msb_window == 0 {
                            sign = 0;
                            bucket = scalar - 1;
                        } else {
                            sign = 1;
                            bucket = scalar & !msb_window;
                        }
                    }

                    cmds[j] = 3 << (0);
                    cmds[j] |= sign << (0+4);
                    cmds[j] |= (i as u64 & 511) << (0+4+1);
                    cmds[j] |= bucket << (0+4+1+9);
                    cmds[j] |= write << (0+4+1+9+15);
                    cmds[j] |= (i as u64) << (0+4+1+9+15+1);
                    if w_start == 0 {
                        println!{"bucket[{}] {}= sram[{}] (point[{}]) - {}", bucket, if sign==0 {"+"} else {"-"}, (i as u64 & 511), i, write}
                    }

                    i += 1;
                    j += 1;
                    if j == 8 {
                        // println!{"sendings cmds {:?}...", i}

                        write_512_f1(cmd_addr, &cmds);
                        cmd_addr += 64;
                        if ((cmd_addr >> 6) & 0x3f) == 0x3f {
                            println!{"flush at i={}", i}
                            write_flush();
                        }
    
                        if (i & 0x03ff) == 0x03ff {
                            println!{"read_32_f1 at i={}", i}
                            while read_32_f1(0x21<<2) > 256 {
                                // pass
                            }
                        }

                        j = 0;
                        cmds = [0u64; 8];
                    }

                } else {
                    if scalar & msb_window == 0 {
                        buckets[(scalar - 1) as usize] += base;
                    } else {
                        buckets[(scalar & !msb_window) as usize] -= base;
                    }
                }
            });

        let mut res = zero;

        if USE_HW {
            // println!{"aggregating buckets..."}
            // flush any remaining cmds
            if j > 0 {
                write_512_f1(cmd_addr, &cmds);
                cmd_addr += 64;
            }
            write_flush();
            // aggregate
            cmds = [0u64; 8];
            cmds[0] = 0;
            cmds[0] |= 2 << (0);
            // cmds[0] |= 0 << (0+4);
            // cmds[0] |= (pi & buffer_m) << (0+4+1);
            cmds[0] |= ((buckets.len()-1) as u64) << (0+4+1+9);
            // cmds[0] |= (0) << (0+4+1+9+1+15);
            write_512_f1(cmd_addr, &cmds);
            cmd_addr += 64;
            write_flush();

            // println!{"reading result..."}
            let mut res2 = zero;
            unsafe {
                let ptr: *mut u32 = mem::transmute(&res2);

                // set X
                // println!{"reading x..."}
                let x = dma_wait_512();
                std::ptr::copy::<u32>(x.offset(1), ptr, 7);
                std::ptr::copy::<u32>(x.offset(9), ptr.offset(7), 5);

                // set Y
                // println!{"reading y..."}
                let x = dma_wait_512();
                std::ptr::copy::<u32>(x.offset(1), ptr.offset(12), 7);
                std::ptr::copy::<u32>(x.offset(9), ptr.offset(12+7), 5);

                // set Z
                let x = dma_wait_512();
                std::ptr::copy::<u32>(x.offset(1), ptr.offset(36), 7);
                std::ptr::copy::<u32>(x.offset(9), ptr.offset(36+7), 5);

                // set T
                let x = dma_wait_512();
                std::ptr::copy::<u32>(x.offset(1), ptr.offset(24), 7);
                std::ptr::copy::<u32>(x.offset(9), ptr.offset(24+7), 5);

                let ptr: *const u32 = mem::transmute(&res2);
                let mut all_zeros = true;
                for i in 0..12*4 {
                    if *ptr.offset(i) != 0 {
                        println!{"value {} is != 0 at pos {}", *ptr.offset(i), i}
                        all_zeros = false;
                        break;
                    }
                }
                if all_zeros {
                    println!{"all zeros!!!"}
                } else {
                    res = res2;
                }
            }

        } else {
            // aggregate buckets
            let mut running_sum = zero;
            buckets.into_iter().rev().for_each(|b| {
                running_sum += &b;
                res += &running_sum;
            });
            // for i in (0..buckets.len()).rev() {
            //     running_sum += &buckets[i];
            //     res += &running_sum;
            // }
        }
        println!{"res: {:?}", res}
        // println!{"res: {}", res}

        total += &res;
        if w_start > 0 {
            total = total << c;
        }
        // println!{"total: {:?}", total}
        // println!{"total: {}", total}
        total
    });

    let time_end = timer.elapsed();
    println! {"time_first_op: {}", fpga.time_first_op.as_nanos()}
    // 45ms with 32k buckets (c=16) / 25ms with 16k buckets (c=15 as last round from
    // c=17)
    println! {"time_last_round: {}", time_end.as_nanos() - time_fpga_end.as_nanos()}
    println! {"total ops: {}", fpga.n_ops}

    if USE_HW {
        for i in 0..16 {
            write_32_f1(0x10<<2, i);
            println!("cnt[{}] = {}", i, read_32_f1(0x20<<2));
        }
    }

    // // We store the sum for the lowest window.
    // let lowest = *window_sums.first().unwrap();

    // // We're traversing windows from high to low.
    // lowest
    //     + &window_sums[1..]
    //         .iter()
    //         // .rev()
    //         .fold(zero, |mut total, sum_i| {
    //             total += sum_i;
    //             total << c
    //         })

    // window_sums[window_sums.len()-1]
    //     + &window_sums[0..window_sums.len()-1]
    //         .iter()
    //         // .rev()
    //         .fold(zero, |mut total, sum_i| {
    //             total += sum_i;
    //             total << c
    //         })
    result
}
