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

use rayon::range;
use snarkvm_curves::{
    bls12_377::{Fr, G1Affine, G1Projective},
    traits::{AffineCurve}, ProjectiveCurve,
};
use snarkvm_fields::{Zero, PrimeField, FftField, Field};

use rust_gpu_tools::{Device, GPUError};
use rust_gpu_tools::{opencl, opencl::Program, opencl::Buffer};
use std::{sync::{RwLock, Arc, Mutex}, any::TypeId};

pub struct GPUData {
    g1: (usize, usize),
    fr_0: (usize, usize),
    fr_1: (usize, usize),
}

enum GPUResult {
    Ok,
    G1(G1Projective),
    Fr(Vec<Fr>),
}

pub struct GPURequest {
    call_fn: usize,
    response: crossbeam_channel::Sender<Result<GPUResult, GPUError>>,
    data: GPUData,
}

pub struct GPUCache {
    powers: (Vec<Fr>, Vec<Buffer<Fr>>),

    g1_buff: Buffer<G1Affine>,
    fr_buff: Buffer<Fr>,
    g1_bucket: Buffer<G1Projective>,

    o1_result: Buffer<G1Projective>,
    g1_result: Buffer<G1Projective>,
    lc_result: Buffer<Fr>,
    lc_sum:    Buffer<Fr>,
}

struct GPUContext {
    cache: GPUCache,
    program: Program,

    num_windows: usize,
    num_groups: usize,
    global_size: usize,
}


const WINDOW_BITS: usize = 11;
const SCALAR_BITS: usize = 253;
const LOCAL_WORK_SIZE: usize = 256;
const MAX_INPUT_SIZE: usize = 65536;
const POWERS_SIZE: usize = 65536;
const POWERS_CACHES: usize = 300;


fn load_program(device: &Device) -> Result<Program, GPUError> {
    let opencl_kernel = include_str!("blst_377_opencl/msm.cl");
    let opencl_device = match device.opencl_device() {
        Some(opencl_device) => opencl_device,
        None => return Err(GPUError::DeviceNotFound),
    };
    // println!("Using OpenCL device '{}' with {} bytes of memory", device.name(), device.memory());
    let opencl_program = opencl::Program::from_opencl(opencl_device, opencl_kernel)?;
    Ok(opencl_program)
}


/// Run the MSM operation for a given request.
fn gpu_scalar_mul(context: &mut GPUContext, data: GPUData) -> Result<GPUResult, GPUError> {
    let closures = |program: &Program, _arg| -> Result<G1Projective, GPUError> {
        let g1_in = unsafe {std::slice::from_raw_parts(data.g1.0 as *const G1Affine, data.g1.1)};
        let fr_in = unsafe {std::slice::from_raw_parts(data.fr_0.0 as *const Fr, data.fr_0.1)};
        let data_size = fr_in.len();

        program.write_from_buffer(&mut context.cache.g1_buff, g1_in)?;
        program.write_from_buffer(&mut context.cache.fr_buff, fr_in)?;

        std::mem::forget(g1_in);
        std::mem::forget(fr_in);

        if data_size <32000 {
            let block_size = (data_size + 7) / 8;
            let kernel_0 = program.create_kernel(
                "per_scalars_one",
                8,
                LOCAL_WORK_SIZE)?;
    
            kernel_0
                .arg(&context.cache.o1_result)
                .arg(&context.cache.g1_buff)
                .arg(&context.cache.fr_buff)
                .arg(&(block_size as u32))
                .arg(&(data_size as u32))
                .run()?;
        } else {
            let block_size = (data_size + context.global_size -1) / context.global_size;
            let kernel_0 = program.create_kernel(
                "per_scalars",
                context.global_size,
                LOCAL_WORK_SIZE)?;
    
            kernel_0
                .arg(&context.cache.fr_buff)
                .arg(&(block_size as u32))
                .arg(&(data_size as u32))
                .run()?;
        }


        let num_windows = context.num_windows * context.num_groups;
        let kernel_1 = program.create_kernel(
            "msm_cl",
            num_windows,
            LOCAL_WORK_SIZE)?;

        kernel_1
            .arg(&context.cache.g1_bucket)
            .arg(&context.cache.g1_result)
            .arg(&context.cache.g1_buff)
            .arg(&context.cache.fr_buff)
            .arg(&(data_size as u32))
            .arg(&(WINDOW_BITS as u32))
            .arg(&(context.num_groups as u32))
            .run()?;

        let mut results = vec![G1Projective::zero(); num_windows];
        program.read_into_buffer(&context.cache.g1_result, &mut results)?;
        
        let mut acc = G1Projective::zero();
        for idx in (0..context.num_windows).rev() {
            for _ in 0..WINDOW_BITS {
                acc.double_in_place();
            }
            
            for gid in 0..context.num_groups {
                acc += results[idx+ gid*context.num_windows];
            }
        }

        // let mut acc = results.iter().rev().fold(G1Projective::zero(), |mut total, sum_i| {
        //     for _ in 0..WINDOW_BITS {
        //         total.double_in_place();
        //     }
        //     total += sum_i;
        //     total
        // });

        if data_size <32000 {
            let mut ones = vec![G1Projective::zero(); 8];
            program.read_into_buffer(&context.cache.o1_result, &mut ones)?;
            for one in ones {
                acc += one;
            }
        }

        Ok(acc)
    };

    Ok(GPUResult::G1(context.program.run(closures, ())?))
}

fn gpu_mul_assign(context: &mut GPUContext, data: GPUData) -> Result<GPUResult, GPUError> {
    let closures = |program: &Program, _| -> Result<_, GPUError> {
        let fr0 = unsafe { std::slice::from_raw_parts_mut(data.fr_0.0 as *mut Fr, data.fr_0.1) };
        let fr1 = unsafe { std::slice::from_raw_parts(data.fr_1.0 as *const Fr, data.fr_1.1) };

        let size = data.fr_0.1;
        let mut group = size >> 10;
        if size & 0x3ff > 0 {
            group += 1;
        }

        let buff_base = program.create_buffer_from_slice(fr0)?;
        let buff_other = program.create_buffer_from_slice(fr1)?;
        let kernel = program.create_kernel(
            "msm4_mul_assign",
            group,
            128,
        )?;

        kernel
            .arg(&buff_base)
            .arg(&buff_other)
            .arg(&(size as u32))
            .run()?;

        program.read_into_buffer(&buff_base, fr0)?;
        Ok(())
    };
    context.program.run(closures, ())?;
    Ok(GPUResult::Ok)
}

fn gpu_mul2_assign(context: &mut GPUContext, data: GPUData) -> Result<GPUResult, GPUError> {
    let closures = |program: &Program, _| -> Result<_, GPUError> {
        let fr0 = unsafe { std::slice::from_raw_parts_mut(data.fr_0.0 as *mut Fr, data.fr_0.1) };
        let fr1 = unsafe { std::slice::from_raw_parts(data.fr_1.0 as *const Fr, data.fr_1.1) };

        let mut size = data.fr_0.1;
        if size < data.fr_1.1 {
            size = data.fr_1.1;
        }

        let mut group = size >> 10;
        if size & 0x3ff > 0 {
            group += 1;
        }

        let buff_base = program.create_buffer_from_slice(fr0)?;
        let buff_other = program.create_buffer_from_slice(fr1)?;
        let kernel = program.create_kernel(
            "msm4_mul2_assign",
            group,
            128,
        )?;

        kernel
            .arg(&buff_base)
            .arg(&buff_other)
            .arg(&(size as u32))
            .run()?;

        program.read_into_buffer(&buff_base, fr0)?;
        Ok(())
    };

    context.program.run(closures, ())?;
    Ok(GPUResult::Ok)
}

fn gpu_io_helper(context: &mut GPUContext, data: GPUData) -> Result<GPUResult, GPUError> {
    let closures = |program: &Program, _| -> Result<_, GPUError> {
        let bsize = data.fr_0.1;
        let dsize = bsize >> 1;
        let mut group = dsize >> 10;
        if dsize & 0x3ff > 0 {
            group += 1;
        }

        let fr0 = unsafe { std::slice::from_raw_parts_mut(data.fr_0.0 as *mut Fr, data.fr_0.1) };
        let fr1 = unsafe { std::slice::from_raw_parts(data.fr_1.0 as *const Fr, data.fr_1.1) };

        let mut cache_id = 99999;
        for (index, root) in context.cache.powers.0.iter().enumerate() {
            if *root == fr1[0] {
                cache_id = index;
            }
        }
        let buff_powers = if cache_id < context.cache.powers.1.len() {
            // println!("gpu_io_helper powers cache in {}, len:{}",cache_id, context.cache.powers.1.len());
            &context.cache.powers.1[cache_id]
        } else {
            // println!("powers_buff new, len:{}", context.cache.powers.1.len());
            let buff_roots = unsafe { program.create_buffer::<Fr>(POWERS_SIZE) }?;
            let buff_root = program.create_buffer_from_slice(fr1)?;
            let kernel_0 = program.create_kernel("msm4_powers_serial", 64, 128)?;
            kernel_0.arg(&buff_roots).arg(&buff_root).arg(&0u32).arg(&(POWERS_SIZE as u32)).run()?;
            if context.cache.powers.1.len() >= POWERS_CACHES {
                context.cache.powers.0.remove(0);
                context.cache.powers.1.remove(0);
            }
            context.cache.powers.1.push(buff_roots);
            context.cache.powers.0.push(fr1[0]);
            context.cache.powers.1.last().unwrap()
        };

        let buff_base = program.create_buffer_from_slice(fr0)?;
        let mut gap = bsize >> 1;
        while gap > 0 {
            let chunk_size = gap << 1;
            let chunk_num = bsize / chunk_size;
            let kernel_1 = program.create_kernel(
                "msm4_io_helper",
                group,
                128,
            )?;
            kernel_1
                .arg(&buff_base)
                .arg(buff_powers)
                .arg(&(chunk_size as u32))
                .arg(&(chunk_num as u32))
                .arg(&(gap as u32))
                .arg(&(dsize as u32))
                .run()?;
            gap >>= 1;
        }

        program.read_into_buffer(&buff_base, fr0)?;
        std::mem::forget(fr0);
        std::mem::forget(fr1);
        Ok(())
    };
    context.program.run(closures, ())?;
    Ok(GPUResult::Ok)
}

fn gpu_oi_helper(context: &mut GPUContext, data: GPUData) -> Result<GPUResult, GPUError> {
    let closures = |program: &Program, _| -> Result<_, GPUError> {
        let bsize = data.fr_0.1;
        let dsize = bsize >> 1;
        let mut group = dsize >> 10;
        if dsize & 0x3ff > 0 {
            group += 1;
        }

        let fr0 = unsafe { std::slice::from_raw_parts_mut(data.fr_0.0 as *mut Fr, data.fr_0.1) };
        let fr1 = unsafe { std::slice::from_raw_parts(data.fr_1.0 as *const Fr, data.fr_1.1) };

        let mut cache_id = 99999;
        for (index, root) in context.cache.powers.0.iter().enumerate() {
            if *root == fr1[0] {
                cache_id = index;
            }
        }
        let buff_powers = if cache_id < context.cache.powers.1.len() {
            // println!("gpu_oi_helper cache in {}, len:{}",cache_id, context.cache.powers.1.len());
            &context.cache.powers.1[cache_id]
        } else {
            // println!("gpu_oi_helper new powers, len:{}", context.cache.powers.1.len());
            let buff_roots = unsafe { program.create_buffer::<Fr>(POWERS_SIZE) }?;
            let buff_root = program.create_buffer_from_slice(fr1)?;
            let kernel_0 = program.create_kernel("msm4_powers_serial", 64, 128)?;
            kernel_0.arg(&buff_roots).arg(&buff_root).arg(&0u32).arg(&(POWERS_SIZE as u32)).run()?;
            if context.cache.powers.1.len() >= POWERS_CACHES {
                context.cache.powers.0.remove(0);
                context.cache.powers.1.remove(0);
            }
            context.cache.powers.1.push(buff_roots);
            context.cache.powers.0.push(fr1[0]);
            context.cache.powers.1.last().unwrap()
        };

        let buff_base = program.create_buffer_from_slice(fr0)?;
        let mut gap = 1;
        while gap < bsize {
            let chunk_size = gap << 1;
            let chunk_num = bsize / chunk_size;
            let kernel_1 = program.create_kernel(
                "msm4_oi_helper",
                group,
                128,
            )?;
            kernel_1
                .arg(&buff_base)
                .arg(buff_powers)
                .arg(&(chunk_size as u32))
                .arg(&(chunk_num as u32))
                .arg(&(gap as u32))
                .arg(&(dsize as u32))
                .run()?;
            gap <<= 1;
        }

        program.read_into_buffer(&buff_base, fr0)?;
        std::mem::forget(fr0);
        std::mem::forget(fr1);
        Ok(())
    };

    context.program.run(closures, ())?;
    Ok(GPUResult::Ok)
}

fn gpu_powers_serial(context: &mut GPUContext, data: GPUData) -> Result<GPUResult, GPUError> {
    let closures = |program: &Program, _arg| -> Result<Vec<Fr>, GPUError> {
        let fr0 = unsafe { std::slice::from_raw_parts(data.fr_0.0 as *const Fr, 1) };
        let mut group = data.fr_0.1 >> 10;
        if data.fr_0.1 & 0x3ff > 0 {
            group += 1;
        }

        let mut results = vec![Fr::zero(); data.fr_0.1];
        let base_buff = unsafe { program.create_buffer::<Fr>(data.fr_0.1) }?;
        let root_buff = program.create_buffer_from_slice(fr0)?;
        let kernel = program.create_kernel(
            "msm4_powers_serial",
            group,
            128,
        )?;

        kernel
            .arg(&base_buff)
            .arg(&root_buff)
            .arg(&0u32)
            .arg(&(data.fr_0.1 as u32))
            .run()?;

        program.read_into_buffer(&base_buff, &mut results)?;
        Ok(results)
    };

    let results = context.program.run(closures, ())?;
    Ok(GPUResult::Fr(results))
}

fn gpu_evaluate(context: &mut GPUContext, data: GPUData) -> Result<GPUResult, GPUError> {
    let closures = |program: &Program, _arg| -> Result<Vec<Fr>, GPUError> {
        let fr0 = unsafe { std::slice::from_raw_parts(data.fr_0.0 as *const Fr, data.fr_0.1) };
        let fr1 = unsafe { std::slice::from_raw_parts(data.fr_1.0 as *const Fr, data.fr_1.1) };
        let buff_beta = unsafe { program.create_buffer::<Fr>(32768) }?;
        let buff_gamma = unsafe { program.create_buffer::<Fr>(65536) }?;
        let buff_coeffs = program.create_buffer_from_slice(fr0)?;
        let buff_roots = program.create_buffer_from_slice(fr1)?;

        let kernel_0 = program.create_kernel("msm4_powers_serial", 32, 128)?;
        kernel_0.arg(&buff_beta).arg(&buff_roots).arg(&0u32).arg(&32768u32).run()?;

        let kernel_1 = program.create_kernel("msm4_powers_serial", 64, 128)?;
        kernel_1.arg(&buff_gamma).arg(&buff_roots).arg(&1u32).arg(&65536u32).run()?;

        std::mem::forget(fr0);
        std::mem::forget(fr1);

        let kernel_2 = program.create_kernel(
            "msm4_evaluate",
            8,
            256)?;

        kernel_2
            .arg(&context.cache.lc_result)
            .arg(&buff_coeffs)
            .arg(&buff_beta)
            .arg(&buff_gamma)
            .run()?;

        let kernel_3 = program.create_kernel(
            "msm4_sum",
            1,
            8,
        )?;

        kernel_3
            .arg(&context.cache.lc_sum)
            .arg(&context.cache.lc_result)
            .arg(&256u32)
            .arg(&2048u32)
            .run()?;

        let mut results = vec![Fr::zero(); 8];
        program.read_into_buffer(&context.cache.lc_sum, &mut results)?;

        let res = vec![results[0], results[1], results[2] + results[3],
                       results[4] + results[5], results[6] + results[7]];

        Ok(res)
    };

    let res = context.program.run(closures, ())?;
    Ok(GPUResult::Fr(res))
}


/// Initialize the gpu request handler.
fn initialize_gpu_request_handler(input: crossbeam_channel::Receiver<GPURequest>, device: &Device) {
    match load_program(device) {
        Ok(program) => {
            let global_sizes = [
                ("3090",    82),
                ("3080 ti", 80),
                ("3080",    68),
                ("3070 ti", 48),
                ("3070",    46),
                ("3060 ti", 38),
                ("3060",    42),
                ("2080 ti", 68),
                ("2080",    46),
                ("2070",    36),
                ("170hx",   70),
                ("50hx",    84),
                ("gfx1031", 40)
            ];
            let num_windows = (SCALAR_BITS + WINDOW_BITS - 1) / WINDOW_BITS;
            let mut global_size = num_windows;
            for (name, size) in global_sizes {
                if device.name().to_ascii_lowercase().find(name).is_some() {
                    global_size = size;
                    break;
                }
            }

            if let Ok(cudas) = std::env::var("GPU_WORKER") {
                // let cuda_num:usize = cudas.parse().unwrap();
                // global_size = (cuda_num + 63)/64;
                // while global_size <100 { global_size *= 2; }
                global_size = cudas.parse().unwrap();
            };

            let num_groups = if global_size>num_windows {global_size/num_windows} else {1};
            let num_buckets = num_groups * num_windows * (1 << WINDOW_BITS);
            let cache = program.run(|p: &Program, _| -> Result<GPUCache, GPUError>{
                Ok(GPUCache{
                    powers: (Vec::new(),Vec::new()),
                    g1_buff:   unsafe {p.create_buffer::<G1Affine>(MAX_INPUT_SIZE)}.unwrap(),
                    fr_buff:   unsafe {p.create_buffer::<Fr>(MAX_INPUT_SIZE)}.unwrap(),
                    g1_bucket: unsafe {p.create_buffer::<G1Projective>(num_buckets)}.unwrap(),
                    g1_result: unsafe {p.create_buffer::<G1Projective>(num_windows * num_groups)}.unwrap(),
                    o1_result: unsafe {p.create_buffer::<G1Projective>(8)}.unwrap(),
                    lc_result: unsafe {p.create_buffer::<Fr>(2048)}.unwrap(),
                    lc_sum:    unsafe {p.create_buffer::<Fr>(8)}.unwrap(),
                })
            }, ()).unwrap();

            let mut context = GPUContext {
                cache,
                program,
                num_windows,
                num_groups,
                global_size,
            };
            // Handle each cuda request received from the channel.
            while let Ok(request) = input.recv() {
                // let time = start_timer!(|| format!("process start fn {}",request.call_fn));
                let out = match request.call_fn {
                    1 => gpu_scalar_mul(&mut context, request.data),
                    2 => gpu_mul2_assign(&mut context, request.data),
                    3 => gpu_mul_assign(&mut context, request.data),
                    4 => gpu_io_helper(&mut context, request.data),
                    5 => gpu_oi_helper(&mut context, request.data),
                    6 => gpu_powers_serial(&mut context, request.data),
                    7 => gpu_evaluate(&mut context, request.data),
                    __ => Err(GPUError::InvalidId("invalid func".to_string())),
                };
                // end_timer!(time);

                if let Err(e) = out {
                    eprintln!("gpu error {}", e.to_string());
                    request.response.send(Err(e)).ok();
                } else {
                    request.response.send(out).ok();
                }
            }
        }
        Err(err) => {
            eprintln!("Error loading program: {:?}", err);
            // If the program fails to load, notify the request dispatcher.
            while let Ok(request) = input.recv() {
                request.response.send(Err(GPUError::DeviceNotFound)).ok();
            }
        }
    }
}

fn initialize_request_dispatcher() {
    if let Ok(mut dispatchers) = GPU_DISPATCH.write() {
        if dispatchers.len() > 0 {
            return;
        }

        for _ in 0..3 {
            for device in Device::specific_list() {
                let (sender, receiver) = crossbeam_channel::bounded(4096);
                std::thread::spawn(move || initialize_gpu_request_handler(receiver, device));
                dispatchers.push(sender);
            }
        }
    }
}

lazy_static::lazy_static! {
    static ref GPU_DISPATCH: RwLock<Vec<crossbeam_channel::Sender<GPURequest>>> = RwLock::new(Vec::new());
    static ref GPU_DISPATCH_INDEX: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(vec![0usize; 100]));
}

fn fetch_gpu() -> usize {
    let mut min_i = 0usize;
    if let Ok(dispatchers) = GPU_DISPATCH.read() {
        let len = dispatchers.len();
        let mut min_q = 10000usize;
        let mut gpu_queues = GPU_DISPATCH_INDEX.lock().unwrap();
        for i in (0..len).rev() {
            if gpu_queues[i] <= min_q {
                min_q = gpu_queues[i];
                min_i = i;
            }
        }
        gpu_queues[min_i] += 1;
    }
    min_i
}

fn release_gpu(index: usize) {
    if let Ok(dispatchers) = GPU_DISPATCH.read() {
        if index < dispatchers.len() {
            let mut gpu_queues = GPU_DISPATCH_INDEX.lock().unwrap();
            gpu_queues[index] -= 1;
        }
    }
}

fn initialize_dispatcher() {
    let len;
    if let Ok(dispatchers) = GPU_DISPATCH.read() {
        len = dispatchers.len();
    } else {
        len = 0;
    }
    if len == 0 {
        initialize_request_dispatcher();
    }
}

#[allow(clippy::transmute_undefined_repr)]
pub(super) fn msm_opencl<G: AffineCurve, F: PrimeField>(
    mut bases: &[G],
    mut scalars: &[F],
) -> Result<G::Projective, GPUError> {
    if TypeId::of::<G>() != TypeId::of::<G1Affine>() {
        unimplemented!("trying to use gpu for unsupported curve");
    }

    initialize_dispatcher();

    match bases.len() < scalars.len() {
        true => scalars = &scalars[..bases.len()],
        false => bases = &bases[..scalars.len()],
    }

    let index = fetch_gpu();
    // println!("send gpu:{},  size: {}", index, scalars.len());       
    if let Ok(dispatchers) = GPU_DISPATCH.read() {
        if let Some(dispatcher) = dispatchers.get(index) {
            let (sender, receiver) = crossbeam_channel::bounded(1);
            dispatcher
                .send(GPURequest {
                    call_fn: 1,
                    response: sender,
                    data: GPUData {
                        g1: (bases.as_ptr() as usize, bases.len()),
                        fr_0: (scalars.as_ptr() as usize, scalars.len()),
                        fr_1: (0, 0),
                    },
                })
                .map_err(|_| GPUError::DeviceNotFound)?;
            if let Ok(Ok(GPUResult::G1(data))) = receiver.recv() {
                release_gpu(index);
                return Ok(unsafe { std::mem::transmute_copy(&data) });
            }
        }
    }

    release_gpu(index);
    Err(GPUError::DeviceNotFound)
}

pub(super) fn msm4_mul2_assign<F: FftField>(bases: &mut [F], others: &[F]) -> Result<(), GPUError> {
    initialize_dispatcher();
    let index = fetch_gpu();
    // println!("msm4_mul2_assign gpu:{}, bsize:{}", index, bases.len());       
    if let Ok(dispatchers) = GPU_DISPATCH.read() {
        if let Some(dispatcher) = dispatchers.get(index) {
            let (sender, receiver) = crossbeam_channel::bounded(1);
            dispatcher
                .send(GPURequest {
                    call_fn: 2,
                    response: sender,
                    data: GPUData {
                        g1: (0, 0),
                        fr_0: (bases.as_mut_ptr() as usize, bases.len()),
                        fr_1: (others.as_ptr() as usize, others.len()),
                    },
                })
                .map_err(|_| GPUError::DeviceNotFound)?;
            if let Ok(Ok(GPUResult::Ok)) = receiver.recv() {
                release_gpu(index);
                return Ok(());
            }
        }
    }
    release_gpu(index);
    Err(GPUError::DeviceNotFound)
}

pub(super) fn msm4_mul_assign<F: FftField, T: crate::fft::DomainCoeff<F>>(bases: &mut [T], inv: F) -> Result<(), GPUError> {
    initialize_dispatcher();
    let index = fetch_gpu();
    // println!("msm4_mul_assign gpu:{}, bsize:{}", index, bases.len());       
    if let Ok(dispatchers) = GPU_DISPATCH.read() {
        if let Some(dispatcher) = dispatchers.get(index) {
            let (sender, receiver) = crossbeam_channel::bounded(1);
            dispatcher
                .send(GPURequest {
                    call_fn: 3,
                    response: sender,
                    data: GPUData {
                        g1: (0, 0),
                        fr_0: (bases.as_mut_ptr() as usize, bases.len()),
                        fr_1: ([inv].as_ptr() as usize, 1),
                    },
                })
                .map_err(|_| GPUError::DeviceNotFound)?;
            if let Ok(Ok(GPUResult::Ok)) = receiver.recv() {
                release_gpu(index);
                return Ok(());
            }
        }
    }
    release_gpu(index);
    Err(GPUError::DeviceNotFound)
}

pub(super) fn msm4_io_helper<F: FftField, T: crate::fft::DomainCoeff<F>>(bases: &mut [T], root: F) -> Result<(), GPUError> {
    initialize_dispatcher();

    let index = fetch_gpu();
    // println!("msm4_io_helper gpu:{}, bsize:{}", index, bases.len());       
    if let Ok(dispatchers) = GPU_DISPATCH.read() {
        if let Some(dispatcher) = dispatchers.get(index) {
            let (sender, receiver) = crossbeam_channel::bounded(1);
            dispatcher
                .send(GPURequest {
                    call_fn: 4,
                    response: sender,
                    data: GPUData {
                        g1: (0, 0),
                        fr_0: (bases.as_mut_ptr() as usize, bases.len()),
                        fr_1: ([root].as_ptr() as usize, 1),
                    },
                })
                .map_err(|_| GPUError::DeviceNotFound)?;
            if let Ok(Ok(GPUResult::Ok)) = receiver.recv() {
                release_gpu(index);
                return Ok(());
            }
        }
    }
    release_gpu(index);
    Err(GPUError::DeviceNotFound)
}

pub(super) fn msm4_oi_helper<F: FftField, T: crate::fft::DomainCoeff<F>>(bases: &mut [T], root: F) -> Result<(), GPUError> {
    initialize_dispatcher();

    let index = fetch_gpu();
    // println!("msm4_oi_helper gpu:{}, bsize:{}", index, bases.len());        
    if let Ok(dispatchers) = GPU_DISPATCH.read() {
        if let Some(dispatcher) = dispatchers.get(index) {
            let (sender, receiver) = crossbeam_channel::bounded(1);
            dispatcher
                .send(GPURequest {
                    call_fn: 5,
                    response: sender,
                    data: GPUData {
                        g1: (0, 0),
                        fr_0: (bases.as_mut_ptr() as usize, bases.len()),
                        fr_1: ([root].as_ptr() as usize, 1),
                    },
                })
                .map_err(|_| GPUError::DeviceNotFound)?;
            if let Ok(Ok(GPUResult::Ok)) = receiver.recv() {
                release_gpu(index);
                return Ok(());
            }
        }
    }
    release_gpu(index);
    Err(GPUError::DeviceNotFound)
}

pub(super) fn msm4_powers_serial<F: Field>(bases: F, size: usize) -> Result<Vec<Fr>, GPUError> {
    initialize_dispatcher();

    let index = fetch_gpu();
    // println!("msm4_powers_serial gpu:{},  size: {}", index, size);       
    if let Ok(dispatchers) = GPU_DISPATCH.read() {
        if let Some(dispatcher) = dispatchers.get(index) {
            let (sender, receiver) = crossbeam_channel::bounded(1);
            dispatcher
                .send(GPURequest {
                    call_fn: 6,
                    response: sender,
                    data: GPUData {
                        g1: (0, 0),
                        fr_0: ([bases].as_ptr() as usize, size),
                        fr_1: (0, 0),
                    },
                })
                .map_err(|_| GPUError::DeviceNotFound)?;
            if let Ok(Ok(GPUResult::Fr(data))) = receiver.recv() {
                release_gpu(index);
                return Ok(data);
            }
        }
    }
    release_gpu(index);
    Err(GPUError::DeviceNotFound)
}

pub(super) fn msm4_evaluate<F: PrimeField>(bases: &[F], roots: &[F]) -> Result<Vec<Fr>, GPUError> {
    initialize_dispatcher();
    let index = fetch_gpu();
    // println!("msm4_evaluate gpu:{}, bsize:{}", index, bases.len());
    if let Ok(dispatchers) = GPU_DISPATCH.read() {
        if let Some(dispatcher) = dispatchers.get(index) {
            let (sender, receiver) = crossbeam_channel::bounded(1);
            dispatcher
                .send(GPURequest {
                    call_fn: 7,
                    response: sender,
                    data: GPUData {
                        g1: (0, 0),
                        fr_0: (bases.as_ptr() as usize, bases.len()),
                        fr_1: (roots.as_ptr() as usize, roots.len()),
                    },
                })
                .map_err(|_| GPUError::DeviceNotFound)?;
            if let Ok(Ok(GPUResult::Fr(data))) = receiver.recv() {
                release_gpu(index);
                return Ok(data);
            }
        }
    }
    release_gpu(index);
    Err(GPUError::DeviceNotFound)
}
