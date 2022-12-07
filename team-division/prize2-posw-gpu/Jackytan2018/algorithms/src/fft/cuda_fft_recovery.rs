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
use snarkvm_fields::FftField;
use rust_gpu_tools::{program_closures, GPUError};

use std::cmp;
use std::time::Instant;
use crate::fft::DomainCoeff;
use crate::ars_gpu::*;
use crate::fft::cuda_fft::{ars_load_cuda_program, CudaContext};
use crate::fft::cuda_fft::LOG2_MAX_ELEMENTS;
use crate::fft::cuda_fft::MAX_LOG2_RADIX;
use crate::fft::cuda_fft::MAX_LOG2_LOCAL_WORK_SIZE;
use crate::fft::cuda_fft::LOCAL_WORK_SIZE;

static OMGA_TYPE_FFT_COSET: usize = 1;
static OMGA_TYPE_IFFT_COSET: usize = 2;

pub fn ars_radix_fft_recovery<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    context: &mut CudaContext,
    input: &mut Vec<T>,
    omega: &F,
    log_n: u32) -> Result<(), GPUError> {
    let closures = program_closures!(|program, args: (usize, &mut Vec<T>)| -> Result<(), GPUError> {
        let (gpu_index, input) = args;

        let n = 1 << log_n;
        let mut max_log2_radix_n = MAX_LOG2_RADIX;
        let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
        if n <= 80000 {
            max_log2_radix_n -= 1;
            max_log2_local_work_size_n -= 1;
        }
        let max_deg = cmp::min(max_log2_radix_n, log_n);

        let mut pq = vec![F::zero(); 1 << max_deg >> 1];//T还是F?
        let twiddle = omega.pow([(n >> max_deg) as u64]);
        pq[0] =F::one();
        if max_deg > 1 {
            pq[1] = twiddle;
            for i in 2..(1 << max_deg >> 1) {
                pq[i] = pq[i - 1];
                pq[i] *= &twiddle // 乘法
            }
        }
        let mut pq_buffer = program.create_buffer_pool::<F>(gpu_index, pq.len())? ;
        program.write_from_buffer(&mut pq_buffer, &pq)?;


        // Precalculate [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]
        let mut omegas = vec![F::zero(); 32];
        omegas[0] = *omega;
        for i in 1..LOG2_MAX_ELEMENTS {
            omegas[i] = omegas[i - 1].pow([2u64]);
        }
        let mut omegas_buffer = program.create_buffer_pool::<F>(gpu_index, omegas.len())? ;
        program.write_from_buffer(&mut omegas_buffer, &omegas)?;

        let mut src_buffer = program.create_buffer_pool::<T>(gpu_index, n)?;
        let mut dst_buffer = program.create_buffer_pool::<T>(gpu_index, n)?;
        program.write_from_buffer(&mut src_buffer, &*input)?;

        // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
        let mut log_p = 0u32;
        // Each iteration performs a FFT round¬

        while log_p < log_n {
            // 1=>radix2, 2=>radix4, 3=>radix8, ...
            let deg = cmp::min(max_deg, log_n - log_p);
            let n = 1u32 << log_n;
            let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
            let global_work_size = n >> deg;
            let kernel = program.create_kernel(
                "radix_fft",
                global_work_size as usize,
                local_work_size as usize,
            )?;

            let time_kernel = start_timer!(|| format!("核函数 radix_fft log_n = {}",log_n));
            kernel
                .arg(&src_buffer)
                .arg(&dst_buffer)
                .arg(&pq_buffer)
                .arg(&omegas_buffer)
                .arg(&n)
                .arg(&log_p)
                .arg(&deg)
                .arg(&max_deg)
                .run()?;
            end_timer!(time_kernel);

            log_p += deg;
            std::mem::swap(&mut src_buffer, &mut dst_buffer);
        }


        program.read_into_buffer(&src_buffer, input)?;

        program.recovery_buffer_pool(gpu_index, input.len(), src_buffer)?;
        program.recovery_buffer_pool(gpu_index, input.len(), dst_buffer)?;
        program.recovery_buffer_pool(gpu_index, pq.len(), pq_buffer)?;
        program.recovery_buffer_pool(gpu_index, omegas.len(), omegas_buffer)?;
        Ok(())
    });
    let time = start_timer!(|| format!("ars_radix_fft_recovery log_n = {}",log_n));
    let res = context.program.p.run(closures, (context.program.id, input));
    end_timer!(time);
    res
}


pub fn ars_radix_fft_recovery_cache<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    input: &mut Vec<T>,
    omega: &F,
    log_n: u32,
    precomputation_roots: &Vec<F>,
    domain_size: u64,
) -> Result<(), GPUError> {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };
            let closures = program_closures!(|program, args: (usize, &mut Vec<T>)| -> Result<(), GPUError> {
                let (gpu_index, input) = args;
                let n = 1 << log_n;
                let max_deg = cmp::min(MAX_LOG2_RADIX, log_n);

                let pq_len = 1 << max_deg >> 1;
                let (is_create, mut pq_buffer) = program.create_pq_buffer::<F>(gpu_index, pq_len, OMGA_TYPE_FFT_COSET)? ;
                if is_create {
                    let mut pq = vec![F::zero(); pq_len];//T还是F?
                    let twiddle = omega.pow([(n >> max_deg) as u64]);
                    pq[0] =F::one();
                    if max_deg > 1 {
                        pq[1] = twiddle;
                        for i in 2..pq_len {
                            pq[i] = pq[i - 1];
                            pq[i] *= &twiddle
                        }
                    }
                    program.write_from_buffer(&mut pq_buffer, &pq)?;
                }

                let (is_create, mut omegas_buffer) = program.create_omega_buffer::<F>(gpu_index, n,OMGA_TYPE_FFT_COSET)? ;
                if is_create {
                     let time_writer = start_timer!(|| format!("写omega FFT gpu_index = {:?} log_n = {} F个数 = {:?}", gpu_index ,log_n, n));
                     let mut omegas = vec![F::one(); n];
                     let ratio = (domain_size / n as u64) as usize;
                     for i in 0..n/2 {
                        omegas[i] = precomputation_roots[i * ratio];
                        omegas[i + n/2] =-precomputation_roots[i * ratio];
                     }
                    program.write_from_buffer(&mut omegas_buffer, &omegas)?;
                    end_timer!(time_writer);
                }


                let mut src_buffer = program.create_buffer_pool::<T>(gpu_index, n)?;
                let mut dst_buffer = program.create_buffer_pool::<T>(gpu_index, n)?;
                program.write_from_buffer(&mut src_buffer, &*input)?;


                let mut log_p = 0u32;
                // Each iteration performs a FFT round¬
                while log_p < log_n {
                    // 1=>radix2, 2=>radix4, 3=>radix8, ...
                    let deg = cmp::min(max_deg, log_n - log_p);
                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE);
                    let global_work_size = n >> deg;
                    let kernel = program.create_kernel(
                        "radix_fft_full",
                        global_work_size as usize,
                        local_work_size as usize,
                    )?;

                    let time_kernel = start_timer!(|| format!("核函数 radix_fft_full log_n = {}",log_n));
                    kernel
                        .arg(&src_buffer)
                        .arg(&dst_buffer)
                        .arg(&pq_buffer)
                        .arg(&omegas_buffer)
                        .arg(&n)
                        .arg(&log_p)
                        .arg(&deg)
                        .arg(&max_deg)
                        .run()?;
                   end_timer!(time_kernel);

                    log_p += deg;
                    std::mem::swap(&mut src_buffer, &mut dst_buffer);
                }

                program.read_into_buffer(&src_buffer, input)?;
                // recovery
                program.recovery_buffer_pool(gpu_index, input.len(), src_buffer)?;
                program.recovery_buffer_pool(gpu_index, input.len(), dst_buffer)?;
                program.recovery_pq_buffer(gpu_index, pq_len, OMGA_TYPE_FFT_COSET, pq_buffer)?;
                program.recovery_omega_buffer(gpu_index, n, OMGA_TYPE_FFT_COSET, omegas_buffer)?;

                Ok(())
            });
            let time = start_timer!(|| format!("ars_radix_fft_recovery_cache log_n = {}",log_n));
            context.program.p.run(closures, (context.program.id, input))?;
            end_timer!(time);
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}


pub fn ars_radix_fft_2_recovery<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>,
    a2: &mut Vec<T>,
    omega: &F,
    log_n: u32,
) -> Result<(), GPUError> {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };
            let closures = program_closures!(|program, args: (usize, &mut Vec<T>, &mut Vec<T>)| -> Result<(), GPUError> {
                let (gpu_index, input1, input2) = args;

                let n = 1 << log_n;
                let mut max_log2_radix_n = MAX_LOG2_RADIX;
                let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
                if n <= 80000 {
                    max_log2_radix_n -= 1;
                    max_log2_local_work_size_n -= 1;
                }
                let max_deg = cmp::min(max_log2_radix_n, log_n);

                let mut pq = vec![F::zero(); 1 << max_deg >> 1];
                let twiddle = omega.pow([(n >> max_deg) as u64]);
                pq[0] = F::one();
                if max_deg > 1 {
                    pq[1] = twiddle;
                    for i in 2..(1 << max_deg >> 1) {
                        pq[i] = pq[i - 1];
                        pq[i] *= &twiddle
                    }
                }
                let mut pq_buffer = program.create_buffer_pool::<F>(gpu_index, pq.len())? ;
                program.write_from_buffer(&mut pq_buffer, &pq)?;


                let mut omegas = vec![F::zero(); 32];
                omegas[0] = *omega;
                for i in 1..LOG2_MAX_ELEMENTS {
                    omegas[i] = omegas[i - 1].pow([2u64]);
                }

                let mut omegas_buffer = program.create_buffer_pool::<F>(gpu_index, omegas.len())? ;
                program.write_from_buffer(&mut omegas_buffer, &omegas)?;


                let mut src_buffer1 = program.create_buffer_pool::<T>(gpu_index,input1.len())?;
                let mut src_buffer2 = program.create_buffer_pool::<T>(gpu_index,input2.len())?;
                let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index,n)?;
                let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index,n)?;
                //写缓存
                program.write_from_buffer(&mut src_buffer1, &input1)?;
                program.write_from_buffer(&mut src_buffer2, &input2)?;

                let mut log_p = 0u32;
                while log_p < log_n {
                        let deg = cmp::min(max_deg, log_n - log_p);

                        let n = 1u32 << log_n;
                        let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                        let global_work_size = n >> deg;
                        let kernel = program.create_kernel("radix_fft_2", global_work_size as usize, local_work_size as usize)?;


                        let time_kernel = start_timer!(|| format!("核函数 radix_fft_2 log_n = {}",log_n));
                        kernel
                            .arg(&src_buffer1)
                            .arg(&src_buffer2)
                            .arg(&dst_buffer1)
                            .arg(&dst_buffer2)
                            .arg(&pq_buffer)
                            .arg(&omegas_buffer)
                            .arg(&n)
                            .arg(&log_p)
                            .arg(&deg)
                            .arg(&max_deg)
                            .run()?;
                        end_timer!(time_kernel);

                        log_p += deg;
                        std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                        std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                    }


                program.read_into_buffer(&src_buffer1, input1)?;
                program.read_into_buffer(&src_buffer2, input2)?;

                     //回收gpu缓存
                program.recovery_buffer_pool(gpu_index, input1.len(), src_buffer1)?;
                program.recovery_buffer_pool(gpu_index, input2.len(), src_buffer2)?;
                program.recovery_buffer_pool(gpu_index, n, dst_buffer1)?;
                program.recovery_buffer_pool(gpu_index, n, dst_buffer2)?;
                program.recovery_buffer_pool(gpu_index, omegas.len(), omegas_buffer)?;
                program.recovery_buffer_pool(gpu_index, pq.len(), pq_buffer)?;

                Ok(())
            });

            let time = start_timer!(|| format!("ars_radix_fft_2_recovery log_n = {}",log_n));
            context.program.p.run(closures, (context.program.id, a1, a2)).unwrap();
            end_timer!(time);
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}


pub fn ars_radix_fft_2_recovery_cache<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>,
    a2: &mut Vec<T>,
    omega: &F,
    log_n: u32,
    precomputation_roots: &Vec<F>,
    domain_size: u64,
) -> Result<(), GPUError> {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };
            let closures = program_closures!(|program, args: (usize, &mut Vec<T>, &mut Vec<T>)| -> Result<(), GPUError> {
                let (gpu_index, input1, input2) = args;
                let n = 1 << log_n;

                let mut max_log2_radix_n = MAX_LOG2_RADIX;
                let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
                if n <= 80000 {
                    max_log2_radix_n -= 1;
                    max_log2_local_work_size_n -= 1;
                }
                let max_deg = cmp::min(max_log2_radix_n, log_n);

                let pq_len = 1 << max_deg >> 1 ;
                let (is_create, mut pq_buffer) = program.create_pq_buffer::<F>(gpu_index, pq_len, OMGA_TYPE_FFT_COSET)? ;
                if is_create {
                    let mut pq = vec![F::zero(); pq_len];
                    let twiddle = omega.pow([(n >> max_deg) as u64]);
                    pq[0] = F::one();
                    if max_deg > 1 {
                        pq[1] = twiddle;
                        for i in 2..pq_len {
                            pq[i] = pq[i - 1];
                            pq[i] *= &twiddle
                        }
                    }
                    program.write_from_buffer(&mut pq_buffer, &pq)?;
                }

                let (is_create, mut omegas_buffer) = program.create_omega_buffer::<F>(gpu_index, n,OMGA_TYPE_FFT_COSET)? ;
                if is_create {
                     let time_writer = start_timer!(|| format!("写omega FFT gpu_index = {:?} log_n = {} F个数 = {:?}", gpu_index ,log_n, n));
                     let mut omegas = vec![F::one(); n];
                     let ratio = (domain_size / n as u64) as usize;
                     for i in 0..n/2 {
                        omegas[i] = precomputation_roots[i * ratio];
                        omegas[i + n/2] = -precomputation_roots[i * ratio];
                     }
                    program.write_from_buffer(&mut omegas_buffer, &omegas)?;
                    end_timer!(time_writer);
                }


                let mut src_buffer1 = program.create_buffer_pool::<T>(gpu_index,input1.len())?;
                let mut src_buffer2 = program.create_buffer_pool::<T>(gpu_index,input2.len())?;
                let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index,n)?;
                let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index,n)?;
                //写缓存
                program.write_from_buffer(&mut src_buffer1, &input1)?;
                program.write_from_buffer(&mut src_buffer2, &input2)?;


                let mut log_p = 0u32;
                while log_p < log_n {
                        let deg = cmp::min(max_deg, log_n - log_p);

                        let n = 1u32 << log_n;
                        let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                        // let global_work_size = n >> deg;
                        let global_work_size = (n >> deg) * 2;
                        let kernel = program.create_kernel("radix_fft_2_full", global_work_size as usize, local_work_size as usize)?;

                        let time_kernel = start_timer!(|| format!("核函数 radix_fft_2_full log_n = {}",log_n));
                        kernel
                            .arg(&src_buffer1)
                            .arg(&src_buffer2)
                            .arg(&dst_buffer1)
                            .arg(&dst_buffer2)
                            .arg(&pq_buffer)
                            .arg(&omegas_buffer)
                            .arg(&n)
                            .arg(&log_p)
                            .arg(&deg)
                            .arg(&max_deg)
                            .run()?;
                        end_timer!(time_kernel);

                        log_p += deg;
                        std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                        std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                    }


                program.read_into_buffer(&src_buffer1, input1)?;
                program.read_into_buffer(&src_buffer2, input2)?;

                     //回收gpu缓存
                program.recovery_buffer_pool(gpu_index, input1.len(), src_buffer1)?;
                program.recovery_buffer_pool(gpu_index, input2.len(), src_buffer2)?;
                program.recovery_buffer_pool(gpu_index, n, dst_buffer1)?;
                program.recovery_buffer_pool(gpu_index, n, dst_buffer2)?;

                program.recovery_pq_buffer(gpu_index, pq_len, OMGA_TYPE_FFT_COSET, pq_buffer)?;
                program.recovery_omega_buffer(gpu_index, n, OMGA_TYPE_FFT_COSET, omegas_buffer)?;
                Ok(())
            });

            let time = start_timer!(|| format!("ars_radix_fft_2_recovery_cache log_n = {}",log_n));
            context.program.p.run(closures, (context.program.id, a1, a2))?;
            end_timer!(time);
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}


pub fn ars_radix_fft_3_recovery<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>,
    a2: &mut Vec<T>,
    a3: &mut Vec<T>,
    omega: &F,
    log_n: u32,
) -> Result<(), GPUError> {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };
            let closures = program_closures!(|program, input: (usize,&mut Vec<T>, &mut Vec<T>, &mut Vec<T>)|-> Result<(), GPUError> {
                let (gpu_index, input1, input2, input3) = input;

                    let n = 1 << log_n;                    
                    let mut max_log2_radix_n = MAX_LOG2_RADIX;
                    let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
                    if n <= 80000 {
                        max_log2_radix_n -= 1;
                        max_log2_local_work_size_n -= 1;
                    }
                    let max_deg = cmp::min(max_log2_radix_n, log_n);
                    //处理pq
                    let mut pq = vec![F::zero(); 1 << max_deg >> 1];
                    let twiddle = omega.pow([(n >> max_deg) as u64]);
                    pq[0] = F::one();
                    if max_deg > 1 {
                        pq[1] = twiddle;
                        for i in 2..(1 << max_deg >> 1) {
                            pq[i] = pq[i - 1];
                            pq[i] *= &twiddle
                        }
                    }

                    let mut pq_buffer = program.create_buffer_pool::<F>(gpu_index, pq.len())?;
                    program.write_from_buffer(&mut pq_buffer, &pq)?;


                    // 处理omegas
                    let mut omegas = vec![F::zero(); 32];
                    omegas[0] = *omega;
                    for i in 1..LOG2_MAX_ELEMENTS {
                        omegas[i] = omegas[i - 1].pow([2u64]);
                    }
                    let mut omegas_buffer = program.create_buffer_pool::<F>(gpu_index, omegas.len())?;
                    program.write_from_buffer(&mut omegas_buffer, &omegas)?;


                    //创建可回收缓存
                    let mut src_buffer1 = program.create_buffer_pool::<T>(gpu_index,input1.len())?;
                    let mut src_buffer2 = program.create_buffer_pool::<T>(gpu_index,input2.len())?;
                    let mut src_buffer3 = program.create_buffer_pool::<T>(gpu_index,input3.len())?;
                    let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index,n)?;
                    let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index,n)?;
                    let mut dst_buffer3 = program.create_buffer_pool::<T>(gpu_index,n)?;
                    //写缓存
                    program.write_from_buffer(&mut src_buffer1, &input1)?;
                    program.write_from_buffer(&mut src_buffer2, &input2)?;
                    program.write_from_buffer(&mut src_buffer3, &input3)?;


                    let mut log_p = 0u32;
                    while log_p < log_n {
                        let deg = cmp::min(max_deg, log_n - log_p);
                        let n = 1u32 << log_n;
                        let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                        let global_work_size = n >> deg;

                        let kernel = program.create_kernel("radix_fft_3", global_work_size as usize, local_work_size as usize)?;

                        let time_kernel = start_timer!(|| format!("核函数 radix_fft_3 log_n = {}",log_n));
                        kernel
                            .arg(&src_buffer1)
                            .arg(&src_buffer2)
                            .arg(&src_buffer3)
                            .arg(&dst_buffer1)
                            .arg(&dst_buffer2)
                            .arg(&dst_buffer3)
                            .arg(&pq_buffer)
                            .arg(&omegas_buffer)
                            .arg(&n)
                            .arg(&log_p)
                            .arg(&deg)
                            .arg(&max_deg)
                            .run()?;
                        end_timer!(time_kernel);

                        log_p += deg;

                        std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                        std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                        std::mem::swap(&mut src_buffer3, &mut dst_buffer3);
                    }

                program.read_into_buffer(&src_buffer1, input1)?;
                program.read_into_buffer(&src_buffer2, input2)?;
                program.read_into_buffer(&src_buffer3, input3)?;


                //回收gpu缓存
                program.recovery_buffer_pool(gpu_index,input1.len(),src_buffer1)?;
                program.recovery_buffer_pool(gpu_index,input2.len(),src_buffer2)?;
                program.recovery_buffer_pool(gpu_index,input3.len(),src_buffer3)?;
                program.recovery_buffer_pool(gpu_index,n,dst_buffer1)?;
                program.recovery_buffer_pool(gpu_index,n,dst_buffer2)?;
                program.recovery_buffer_pool(gpu_index,n,dst_buffer3)?;
                program.recovery_buffer_pool(gpu_index,omegas.len(),omegas_buffer)?;
                program.recovery_buffer_pool(gpu_index,pq.len(),pq_buffer)?;
                Ok(())
            });
            let time = start_timer!(|| format!("ars_radix_fft_3_recovery log_n = {}",log_n));
            context.program.p.run(closures, (context.program.id, a1, a2, a3)).unwrap();
            end_timer!(time);
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}


pub fn ars_radix_fft_3_recovery_cache<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>,
    a2: &mut Vec<T>,
    a3: &mut Vec<T>,
    omega: &F,
    log_n: u32,
    precomputation_roots: &Vec<F>,
    domain_size: u64,
) -> Result<(), GPUError> {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };
            let closures = program_closures!(|program, input: (usize,&mut Vec<T>, &mut Vec<T>, &mut Vec<T>)|-> Result<(), GPUError> {
                let (gpu_index, input1, input2, input3) = input;
                let n = 1 << log_n;
                let mut max_log2_radix_n = MAX_LOG2_RADIX;
                let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
                if n <= 80000 {
                    max_log2_radix_n -= 1;
                    max_log2_local_work_size_n -= 1;
                }
                let max_deg = cmp::min(max_log2_radix_n, log_n);
                //处理pq
                let pq_len = 1 << max_deg >> 1;
                let (is_create, mut pq_buffer) = program.create_pq_buffer::<F>(gpu_index, pq_len, OMGA_TYPE_FFT_COSET)? ;
                if is_create {
                    let mut pq = vec![F::zero(); pq_len];
                    let twiddle = omega.pow([(n >> max_deg) as u64]);
                    pq[0] = F::one();
                    if max_deg > 1 {
                        pq[1] = twiddle;
                        for i in 2..pq_len {
                            pq[i] = pq[i - 1];
                            pq[i] *= &twiddle
                        }
                    }
                    program.write_from_buffer(&mut pq_buffer, &pq)?;
                }


                // 处理omegas
                let (is_create, mut omegas_buffer) = program.create_omega_buffer::<F>(gpu_index, n,OMGA_TYPE_FFT_COSET)? ;
                if is_create {
                    let time_writer = start_timer!(|| format!("写omega FFT gpu_index = {:?} log_n = {} F个数 = {:?}", gpu_index ,log_n, n));
                    let mut omegas = vec![F::one(); n];
                    let ratio = (domain_size / n as u64) as usize;
                     for i in 0..n/2 {
                        omegas[i] = precomputation_roots[i * ratio];
                        omegas[i + n/2] =-precomputation_roots[i * ratio];
                     }
                    program.write_from_buffer(&mut omegas_buffer, &omegas)?;
                    end_timer!(time_writer);
                }


                //创建可回收缓存
                let mut src_buffer1 = program.create_buffer_pool::<T>(gpu_index,input1.len())?;
                let mut src_buffer2 = program.create_buffer_pool::<T>(gpu_index,input2.len())?;
                let mut src_buffer3 = program.create_buffer_pool::<T>(gpu_index,input3.len())?;
                let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index,n)?;
                let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index,n)?;
                let mut dst_buffer3 = program.create_buffer_pool::<T>(gpu_index,n)?;
                //写缓存
                program.write_from_buffer(&mut src_buffer1, &input1)?;
                program.write_from_buffer(&mut src_buffer2, &input2)?;
                program.write_from_buffer(&mut src_buffer3, &input3)?;



                let mut log_p = 0u32;
                while log_p < log_n {
                    let deg = cmp::min(max_deg, log_n - log_p);
                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                    // let global_work_size = n >> deg;
                    let global_work_size = (n >> deg) * 3;

                    let kernel = program.create_kernel("radix_fft_3_full", global_work_size as usize, local_work_size as usize)?;

                    let time_kernel = start_timer!(|| format!("核函数 radix_fft_3_full log_n = {}",log_n));
                    kernel
                        .arg(&src_buffer1)
                        .arg(&src_buffer2)
                        .arg(&src_buffer3)
                        .arg(&dst_buffer1)
                        .arg(&dst_buffer2)
                        .arg(&dst_buffer3)
                        .arg(&pq_buffer)
                        .arg(&omegas_buffer)
                        .arg(&n)
                        .arg(&log_p)
                        .arg(&deg)
                        .arg(&max_deg)
                        .run()?;
                    end_timer!(time_kernel);

                    log_p += deg;

                    std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                    std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                    std::mem::swap(&mut src_buffer3, &mut dst_buffer3);
                }

                program.read_into_buffer(&src_buffer1, input1)?;
                program.read_into_buffer(&src_buffer2, input2)?;
                program.read_into_buffer(&src_buffer3, input3)?;


                //回收gpu缓存
                program.recovery_buffer_pool(gpu_index,input1.len(),src_buffer1)?;
                program.recovery_buffer_pool(gpu_index,input2.len(),src_buffer2)?;
                program.recovery_buffer_pool(gpu_index,input3.len(),src_buffer3)?;
                program.recovery_buffer_pool(gpu_index,n,dst_buffer1)?;
                program.recovery_buffer_pool(gpu_index,n,dst_buffer2)?;
                program.recovery_buffer_pool(gpu_index,n,dst_buffer3)?;

                program.recovery_pq_buffer(gpu_index, pq_len, OMGA_TYPE_FFT_COSET, pq_buffer)?;
                program.recovery_omega_buffer(gpu_index, n, OMGA_TYPE_FFT_COSET, omegas_buffer)?;
                Ok(())
            });
            let time = start_timer!(|| format!("ars_radix_fft_3_recovery_cache log_n = {}",log_n));
            context.program.p.run(closures, (context.program.id, a1, a2, a3))?;
            end_timer!(time);
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}


pub fn ars_radix_ifft_recovery<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    context: &mut CudaContext,
    input: &mut Vec<T>,
    omega: &F,
    log_n: u32,
    size_inv: &F) -> Result<(), GPUError> {
    let closures = program_closures!(|program, args:(usize,&mut Vec<T>) | -> Result<(), GPUError> {
        let (gpu_index,input) = args;

        let n = 1 << log_n;      
        let mut max_log2_radix_n = MAX_LOG2_RADIX;
        let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
        if n <= 80000 {
            max_log2_radix_n -= 1;
            max_log2_local_work_size_n -= 1;
        }
        let max_deg = cmp::min(max_log2_radix_n, log_n);

        // Precalculate:
        // [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]
        let mut pq = vec![F::zero(); 1 << max_deg >> 1];//T还是F?
        let twiddle = omega.pow([(n >> max_deg) as u64]);
        pq[0] =F::one();
        if max_deg > 1 {
            pq[1] = twiddle;
            for i in 2..(1 << max_deg >> 1) {
                pq[i] = pq[i - 1];
                pq[i] *= &twiddle // 乘法
                //pq[i].mul_assign(&twiddle);
            }
        }
        let mut pq_buffer = program.create_buffer_pool::<F>(gpu_index, pq.len())?;
        program.write_from_buffer(&mut pq_buffer, &pq)?;


        // Precalculate [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]
        let mut omegas = vec![F::zero(); 32];
        omegas[0] = *omega;
        for i in 1..LOG2_MAX_ELEMENTS {
            omegas[i] = omegas[i - 1].pow([2u64]);
        }
        let mut omegas_buffer = program.create_buffer_pool::<F>(gpu_index, omegas.len())?;
        program.write_from_buffer(&mut omegas_buffer, &omegas)?;


        let mut src_buffer =  program.create_buffer_pool::<T>(gpu_index, n)?;
        let mut dst_buffer = program.create_buffer_pool::<T>(gpu_index, n)?;
        program.write_from_buffer(&mut src_buffer, &*input)?;


        // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
        let mut log_p = 0u32;
        // Each iteration performs a FFT round
        while log_p < log_n {
            // 1=>radix2, 2=>radix4, 3=>radix8, ...
            let deg = cmp::min(max_deg, log_n - log_p);
            let n = 1u32 << log_n;
            let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
            let global_work_size = n >> deg;

            let kernel = program.create_kernel(
                "radix_fft",
                global_work_size as usize,
                local_work_size as usize,
            )?;

            let time_kernel = start_timer!(|| format!("核函数 radix_fft log_n = {}",log_n));
            kernel
                .arg(&src_buffer)
                .arg(&dst_buffer)
                .arg(&pq_buffer)
                .arg(&omegas_buffer)
                .arg(&n)
                .arg(&log_p)
                .arg(&deg)
                .arg(&max_deg)
                .run()?;
            end_timer!(time_kernel);

            log_p += deg;
            std::mem::swap(&mut src_buffer, &mut dst_buffer);
        }

		let mut size_invs = vec![F::zero(); 1];
		size_invs[0] = *size_inv;
		let size_inv_buffer = program.create_buffer_from_slice(&size_invs)?;

        let local_work_size = LOCAL_WORK_SIZE;
        let global_work_size = n / LOCAL_WORK_SIZE;
		let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
        let kernel1 = program.create_kernel(
            "compute_mul_size_inv",
            global_work_size as usize,
            local_work_size as usize,
        )?;


        kernel1
            .arg(&src_buffer)
            .arg(&dst_buffer)
            .arg(&size_inv_buffer)
            .arg(&(n as u32))
            .arg(&(num_width as u32))
            .run()?;
        program.read_into_buffer(&dst_buffer, input)?;

        //回收缓存
        program.recovery_buffer_pool(gpu_index,n,src_buffer)?;
        program.recovery_buffer_pool(gpu_index,n,dst_buffer)?;
        program.recovery_buffer_pool(gpu_index,pq.len(),pq_buffer)?;
        program.recovery_buffer_pool(gpu_index,omegas.len(),omegas_buffer)?;
        Ok(())
    });
    let time = start_timer!(|| format!("ars_radix_ifft_recovery log_n = {}",log_n));
    let res = context.program.p.run(closures, (context.program.id, input));
    end_timer!(time);
    res
}


pub fn ars_radix_ifft_recovery_cache<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    input: &mut Vec<T>,
    omega: &F,
    log_n: u32,
    size_inv: &F,
    precomputation_roots: &Vec<F>,
    domain_size: u64,
) -> Result<(), GPUError> {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };

            let closures = program_closures!(|program, args:(usize,&mut Vec<T>) | -> Result<(), GPUError> {
                let (gpu_index,input) = args;
                let n = 1 << log_n;
                // The precalculated values pq` and `omegas` are valid for radix degrees up to `max_deg`
                // let max_deg = cmp::min(MAX_LOG2_RADIX, log_n);
                let mut max_log2_radix_n = MAX_LOG2_RADIX;
                let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
                if n <= 80000 {
                    max_log2_radix_n -= 1;
                    max_log2_local_work_size_n -= 1;
                }
                let max_deg = cmp::min(max_log2_radix_n, log_n);

                let pq_len = 1 << max_deg >> 1;
                let (is_create, mut pq_buffer) = program.create_pq_buffer::<F>(gpu_index, pq_len, OMGA_TYPE_IFFT_COSET)? ;
                if is_create {
                    let mut pq = vec![F::zero(); pq_len];//T还是F?
                    let twiddle = omega.pow([(n >> max_deg) as u64]);
                    pq[0] =F::one();
                    if max_deg > 1 {
                        pq[1] = twiddle;
                        for i in 2..pq_len {
                            pq[i] = pq[i - 1];
                            pq[i] *= &twiddle // 乘法
                        }
                    }
                    program.write_from_buffer(&mut pq_buffer, &pq)?;
                }

                let (is_create, mut omegas_buffer) = program.create_omega_buffer::<F>(gpu_index, n,OMGA_TYPE_IFFT_COSET)? ;
                if is_create {
                     let time_writer = start_timer!(|| format!("写omega IFFT gpu_index = {:?} log_n = {} F个数 = {:?}", gpu_index ,log_n, n));
                     let mut omegas = vec![F::one(); n];
                     let ratio = (domain_size / n as u64) as usize;
                     for i in 0..n/2 {
                        omegas[i] = precomputation_roots[i * ratio];
                        omegas[i + n/2] =-precomputation_roots[i * ratio];
                     }
                    program.write_from_buffer(&mut omegas_buffer, &omegas)?;
                    end_timer!(time_writer);
                }

                let mut src_buffer =  program.create_buffer_pool::<T>(gpu_index, n)?;
                let mut dst_buffer = program.create_buffer_pool::<T>(gpu_index, n)?;
                program.write_from_buffer(&mut src_buffer, &*input)?;

                // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
                let mut log_p = 0u32;
                // Each iteration performs a FFT round
                while log_p < log_n {
                    // 1=>radix2, 2=>radix4, 3=>radix8, ...
                    let deg = cmp::min(max_deg, log_n - log_p);
                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE);
                    let global_work_size = n >> deg;

                    let kernel = program.create_kernel(
                        "radix_fft_full",
                        global_work_size as usize,
                        local_work_size as usize,
                    )?;

                    let time_kernel = start_timer!(|| format!("核函数 radix_fft_full log_n = {}",log_n));
                    kernel
                        .arg(&src_buffer)
                        .arg(&dst_buffer)
                        .arg(&pq_buffer)
                        .arg(&omegas_buffer)
                        .arg(&n)
                        .arg(&log_p)
                        .arg(&deg)
                        .arg(&max_deg)
                        .run()?;
                    end_timer!(time_kernel);

                    log_p += deg;
                    std::mem::swap(&mut src_buffer, &mut dst_buffer);
                }

                let time_kernel = start_timer!(|| format!("ars_radix_ifft_recovery_cache  size_inv_buffer gpu内存all log_n = {}",log_n));
                let (is_create, mut size_inv_buffer) = program.create_size_inv_buffer::<F>(gpu_index, log_n as usize)? ;
                if is_create {
                    let time_c = start_timer!(|| format!("ars_radix_ifft_recovery_cache  size_inv_buffer 写内存 log_n = {}",log_n));
                    let size_invs = vec![*size_inv; 1];
                    program.write_from_buffer(&mut size_inv_buffer, &size_invs)?;
                    end_timer!(time_c);
                }
                end_timer!(time_kernel);

                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = 64;
                let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                let kernel1 = program.create_kernel(
                    "compute_mul_size_inv",
                    global_work_size as usize,
                    local_work_size as usize,
                )?;

                kernel1
                    .arg(&src_buffer)
                    .arg(&dst_buffer)
                    .arg(&size_inv_buffer)
                    .arg(&(n as u32))
                    .arg(&(num_width as u32))
                    .run()?;
                program.read_into_buffer(&dst_buffer, input)?;

                //回收缓存
                program.recovery_buffer_pool(gpu_index,n,src_buffer)?;
                program.recovery_buffer_pool(gpu_index,n,dst_buffer)?;

                program.recovery_pq_buffer(gpu_index, pq_len, OMGA_TYPE_IFFT_COSET, pq_buffer)?;
                program.recovery_omega_buffer(gpu_index, n, OMGA_TYPE_IFFT_COSET, omegas_buffer)?;
                program.recovery_size_inv_buffer(gpu_index, log_n as usize, size_inv_buffer)?;
                Ok(())
            });
            let time = start_timer!(|| format!("ars_radix_ifft_recovery_cache log_n = {}",log_n));
            context.program.p.run(closures, (context.program.id, input))?;
            end_timer!(time);
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}


pub fn ars_radix_ifft_2_recovery<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>, a2: &mut Vec<T>,
    omega: &F, log_n: u32, size_inv: &F,
) -> Result<(), GPUError> {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };
            let closures = program_closures!(|program, args: (usize, &mut Vec<T>, &mut Vec<T>)| -> Result<(), GPUError> {
                let (gpu_index, input1, input2) = args;

                let n = 1 << log_n;
                let mut max_log2_radix_n = MAX_LOG2_RADIX;
                let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
                if n <= 80000 {
                    max_log2_radix_n -= 1;
                    max_log2_local_work_size_n -= 1;
                }
                let max_deg = cmp::min(max_log2_radix_n, log_n);

                let mut pq = vec![F::zero(); 1 << max_deg >> 1];
                let twiddle = omega.pow([(n >> max_deg) as u64]);
                pq[0] = F::one();
                if max_deg > 1 {
                    pq[1] = twiddle;
                    for i in 2..(1 << max_deg >> 1) {
                        pq[i] = pq[i - 1];
                        pq[i] *= &twiddle
                    }
                }
                let mut pq_buffer = program.create_buffer_pool::<F>(gpu_index, pq.len())? ;
                program.write_from_buffer(&mut pq_buffer, &pq)?;


                let mut omegas = vec![F::zero(); 32];
                omegas[0] = *omega;
                for i in 1..LOG2_MAX_ELEMENTS {
                    omegas[i] = omegas[i - 1].pow([2u64]);
                }
                let mut omegas_buffer = program.create_buffer_pool::<F>(gpu_index, omegas.len())? ;
                program.write_from_buffer(&mut omegas_buffer, &omegas)?;

                let mut src_buffer1 = program.create_buffer_pool::<T>(gpu_index,input1.len())?;
                let mut src_buffer2 = program.create_buffer_pool::<T>(gpu_index,input2.len())?;
                let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index,n)?;
                let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index,n)?;
                //写缓存
                program.write_from_buffer(&mut src_buffer1, &input1)?;
                program.write_from_buffer(&mut src_buffer2, &input2)?;

                let mut log_p = 0u32;
                while log_p < log_n {
                    let deg = cmp::min(max_deg, log_n - log_p);

                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                    let global_work_size = n >> deg;
                    let kernel = program.create_kernel("radix_fft_2", global_work_size as usize, local_work_size as usize)?;

                    let time_kernel = start_timer!(|| format!("核函数 radix_fft_2 log_n = {}",log_n));
                    kernel
                        .arg(&src_buffer1)
                        .arg(&src_buffer2)
                        .arg(&dst_buffer1)
                        .arg(&dst_buffer2)
                        .arg(&pq_buffer)
                        .arg(&omegas_buffer)
                        .arg(&n)
                        .arg(&log_p)
                        .arg(&deg)
                        .arg(&max_deg)
                        .run()?;
                    end_timer!(time_kernel);


                    log_p += deg;
                    std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                    std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                }

                let mut size_invs = vec![F::zero(); 1];
                size_invs[0] = *size_inv;
                let size_inv_buffer = program.create_buffer_from_slice(&size_invs)?;

                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = n / LOCAL_WORK_SIZE;
                let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                let kernel1 = program.create_kernel("compute_mul_size_inv_2", global_work_size as usize, local_work_size as usize)?;

                kernel1
                    .arg(&src_buffer1)
                    .arg(&src_buffer2)
                    .arg(&dst_buffer1)
                    .arg(&dst_buffer2)
                    .arg(&size_inv_buffer)
                    .arg(&(n as u32))
                    .arg(&(num_width as u32))
                    .run()?;
                program.read_into_buffer(&dst_buffer1, input1)?;
                program.read_into_buffer(&dst_buffer2, input2)?;

                //回收gpu缓存
                program.recovery_buffer_pool(gpu_index, input1.len(), src_buffer1)?;
                program.recovery_buffer_pool(gpu_index, input2.len(), src_buffer2)?;
                program.recovery_buffer_pool(gpu_index, n, dst_buffer1)?;
                program.recovery_buffer_pool(gpu_index, n, dst_buffer2)?;
                program.recovery_buffer_pool(gpu_index, omegas.len(), omegas_buffer)?;
                program.recovery_buffer_pool(gpu_index, pq.len(), pq_buffer)?;

                Ok(())
            });
            let time = start_timer!(|| format!("ars_radix_ifft_2_recovery log_n = {}",log_n));
            context.program.p.run(closures, (context.program.id, a1, a2)).unwrap();
            end_timer!(time);
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}


pub fn ars_radix_ifft_2_recovery_cache<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>,
    a2: &mut Vec<T>,
    omega: &F,
    log_n: u32,
    size_inv: &F,
    precomputation_roots: &Vec<F>,
    domain_size: u64,
) -> Result<(), GPUError> {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };
            let closures = program_closures!(|program, args: (usize, &mut Vec<T>, &mut Vec<T>)| -> Result<(), GPUError> {
                let (gpu_index, input1, input2) = args;
                let n = 1 << log_n;
                let mut max_log2_radix_n = MAX_LOG2_RADIX;
                let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
                if n <= 80000 {
                    max_log2_radix_n -= 1;
                    max_log2_local_work_size_n -= 1;
                }
                let max_deg = cmp::min(max_log2_radix_n, log_n);

                let pq_len = 1 << max_deg >> 1;
                let (is_create, mut pq_buffer) = program.create_pq_buffer::<F>(gpu_index, pq_len, OMGA_TYPE_IFFT_COSET)? ;
                if is_create {
                    let mut pq = vec![F::zero(); pq_len];
                    let twiddle = omega.pow([(n >> max_deg) as u64]);
                    pq[0] = F::one();
                    if max_deg > 1 {
                        pq[1] = twiddle;
                        for i in 2..pq_len {
                            pq[i] = pq[i - 1];
                            pq[i] *= &twiddle
                        }
                    }
                    program.write_from_buffer(&mut pq_buffer, &pq)?;
                }

                let (is_create, mut omegas_buffer) = program.create_omega_buffer::<F>(gpu_index, n,OMGA_TYPE_IFFT_COSET)? ;
                if is_create {
                    let mut omegas = vec![F::one(); n];
                    let time_writer = start_timer!(|| format!("写omega IFFT gpu_index = {:?} log_n = {} F个数 = {:?}", gpu_index ,log_n, n));
                     let ratio = (domain_size / n as u64) as usize;
                     for i in 0..n/2 {
                        omegas[i] = precomputation_roots[i * ratio];
                        omegas[i + n/2] =-precomputation_roots[i * ratio];
                     }
                    program.write_from_buffer(&mut omegas_buffer, &omegas)?;
                    end_timer!(time_writer);
                }

                let mut src_buffer1 = program.create_buffer_pool::<T>(gpu_index,input1.len())?;
                let mut src_buffer2 = program.create_buffer_pool::<T>(gpu_index,input2.len())?;
                let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index,n)?;
                let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index,n)?;
                //写缓存
                program.write_from_buffer(&mut src_buffer1, &input1)?;
                program.write_from_buffer(&mut src_buffer2, &input2)?;

                let mut log_p = 0u32;
                while log_p < log_n {
                    let deg = cmp::min(max_deg, log_n - log_p);

                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                    // let global_work_size = n >> deg;
                    let global_work_size = (n >> deg) * 2;
                    let kernel = program.create_kernel("radix_fft_2_full", global_work_size as usize, local_work_size as usize)?;


                    let time_kernel = start_timer!(|| format!("核函数 radix_fft_2_full log_n = {}",log_n));
                    kernel
                        .arg(&src_buffer1)
                        .arg(&src_buffer2)
                        .arg(&dst_buffer1)
                        .arg(&dst_buffer2)
                        .arg(&pq_buffer)
                        .arg(&omegas_buffer)
                        .arg(&n)
                        .arg(&log_p)
                        .arg(&deg)
                        .arg(&max_deg)
                        .run()?;
                    end_timer!(time_kernel);

                    log_p += deg;
                    std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                    std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                }

                let time_kernel = start_timer!(|| format!("ars_radix_ifft_2_recovery_cache  size_inv_buffer gpu内存all log_n = {}",log_n));
                let (is_create, mut size_inv_buffer) = program.create_size_inv_buffer::<F>(gpu_index, log_n as usize)? ;
                if is_create {
                    let time_c = start_timer!(|| format!("ars_radix_ifft_2_recovery_cache  size_inv_buffer 写内存 log_n = {}",log_n));
                    let size_invs = vec![*size_inv; 1];
                    program.write_from_buffer(&mut size_inv_buffer, &size_invs)?;
                    end_timer!(time_c);
                }
                end_timer!(time_kernel);

                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = 16;
                let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                let kernel1 = program.create_kernel("compute_mul_size_inv_2", global_work_size as usize, local_work_size as usize)?;

                kernel1
                    .arg(&src_buffer1)
                    .arg(&src_buffer2)
                    .arg(&dst_buffer1)
                    .arg(&dst_buffer2)
                    .arg(&size_inv_buffer)
                    .arg(&(n as u32))
                    .arg(&(num_width as u32))
                    .run()?;
                program.read_into_buffer(&dst_buffer1, input1)?;
                program.read_into_buffer(&dst_buffer2, input2)?;

                //回收gpu缓存
                program.recovery_buffer_pool(gpu_index, input1.len(), src_buffer1)?;
                program.recovery_buffer_pool(gpu_index, input2.len(), src_buffer2)?;
                program.recovery_buffer_pool(gpu_index, n, dst_buffer1)?;
                program.recovery_buffer_pool(gpu_index, n, dst_buffer2)?;

                program.recovery_pq_buffer(gpu_index, pq_len, OMGA_TYPE_IFFT_COSET, pq_buffer)?;
                program.recovery_omega_buffer(gpu_index, n, OMGA_TYPE_IFFT_COSET, omegas_buffer)?;
                program.recovery_size_inv_buffer(gpu_index, log_n as usize, size_inv_buffer)?;

                Ok(())
            });
            let time = start_timer!(|| format!("ars_radix_ifft_2_recovery_cache log_n = {}",log_n));
            context.program.p.run(closures, (context.program.id, a1, a2))?;
            end_timer!(time);
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}


pub fn ars_radix_ifft_3_recovery<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>, a2: &mut Vec<T>, a3: &mut Vec<T>,
    omega: &F, log_n: u32, size_inv: &F,
) -> Result<(), GPUError> {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };
            let closures = program_closures!(|program, args: (usize, &mut Vec<T>, &mut Vec<T>, &mut Vec<T>)|-> Result<(), GPUError> {
              let (gpu_index, input1, input2, input3) = args;

                let n = 1 << log_n;
                let mut max_log2_radix_n = MAX_LOG2_RADIX;
                let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
                if n <= 80000 {
                    max_log2_radix_n -= 1;
                    max_log2_local_work_size_n -= 1;
                }
                let max_deg = cmp::min(max_log2_radix_n, log_n);

                let mut pq = vec![F::zero(); 1 << max_deg >> 1];
                let twiddle = omega.pow([(n >> max_deg) as u64]);
                pq[0] = F::one();
                if max_deg > 1 {
                    pq[1] = twiddle;
                    for i in 2..(1 << max_deg >> 1) {
                        pq[i] = pq[i - 1];
                        pq[i] *= &twiddle
                    }
                }
                let mut pq_buffer = program.create_buffer_pool::<F>(gpu_index, pq.len())? ;
                program.write_from_buffer(&mut pq_buffer, &pq)?;


                let mut omegas = vec![F::zero(); 32];
                omegas[0] = *omega;
                for i in 1..LOG2_MAX_ELEMENTS {
                    omegas[i] = omegas[i - 1].pow([2u64]);
                }
                let mut omegas_buffer = program.create_buffer_pool::<F>(gpu_index, omegas.len())?;
                program.write_from_buffer(&mut omegas_buffer, &omegas)?;


                //创建可回收缓存
                let mut src_buffer1 = program.create_buffer_pool::<T>(gpu_index,input1.len())?;
                let mut src_buffer2 = program.create_buffer_pool::<T>(gpu_index,input2.len())?;
                let mut src_buffer3 = program.create_buffer_pool::<T>(gpu_index,input3.len())?;
                let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index,n)?;
                let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index,n)?;
                let mut dst_buffer3 = program.create_buffer_pool::<T>(gpu_index,n)?;
                //写缓存
                program.write_from_buffer(&mut src_buffer1, &input1)?;
                program.write_from_buffer(&mut src_buffer2, &input2)?;
                program.write_from_buffer(&mut src_buffer3, &input3)?;

                let mut log_p = 0u32;
                while log_p < log_n {
                    let deg = cmp::min(max_deg, log_n - log_p);

                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                    let global_work_size = n >> deg;
                    let kernel = program.create_kernel("radix_fft_3", global_work_size as usize, local_work_size as usize)?;

                    let time_kernel = start_timer!(|| format!("核函数 radix_fft_3 log_n = {}",log_n));
                    kernel
                        .arg(&src_buffer1)
                        .arg(&src_buffer2)
                        .arg(&src_buffer3)
                        .arg(&dst_buffer1)
                        .arg(&dst_buffer2)
                        .arg(&dst_buffer3)
                        .arg(&pq_buffer)
                        .arg(&omegas_buffer)
                        .arg(&n)
                        .arg(&log_p)
                        .arg(&deg)
                        .arg(&max_deg)
                        .run()?;
                    end_timer!(time_kernel);

                    log_p += deg;
                    std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                    std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                    std::mem::swap(&mut src_buffer3, &mut dst_buffer3);
                }

                let mut size_invs = vec![F::zero(); 1];
                size_invs[0] = *size_inv;
                let size_inv_buffer = program.create_buffer_from_slice(&size_invs)?;

                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = n / LOCAL_WORK_SIZE;
                let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                let kernel1 = program.create_kernel("compute_mul_size_inv_3", global_work_size as usize, local_work_size as usize)?;

                kernel1
                    .arg(&src_buffer1)
                    .arg(&src_buffer2)
                    .arg(&src_buffer3)
                    .arg(&dst_buffer1)
                    .arg(&dst_buffer2)
                    .arg(&dst_buffer3)
                    .arg(&size_inv_buffer)
                    .arg(&(n as u32))
                    .arg(&(num_width as u32))
                    .run()?;
                program.read_into_buffer(&dst_buffer1, input1)?;
                program.read_into_buffer(&dst_buffer2, input2)?;
                program.read_into_buffer(&dst_buffer3, input3)?;

                //回收gpu缓存
                program.recovery_buffer_pool(gpu_index,input1.len(),src_buffer1)?;
                program.recovery_buffer_pool(gpu_index,input2.len(),src_buffer2)?;
                program.recovery_buffer_pool(gpu_index,input3.len(),src_buffer3)?;
                program.recovery_buffer_pool(gpu_index,n,dst_buffer1)?;
                program.recovery_buffer_pool(gpu_index,n,dst_buffer2)?;
                program.recovery_buffer_pool(gpu_index,n,dst_buffer3)?;
                program.recovery_buffer_pool(gpu_index,omegas.len(),omegas_buffer)?;
                program.recovery_buffer_pool(gpu_index,pq.len(),pq_buffer)?;
                Ok(())
            });
            let time = start_timer!(|| format!("ars_radix_ifft_3_recovery log_n = {}",log_n));
            context.program.p.run(closures, (context.program.id, a1, a2, a3)).unwrap();
            end_timer!(time);
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}


pub fn ars_radix_ifft_3_recovery_cache<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>,
    a2: &mut Vec<T>,
    a3: &mut Vec<T>,
    omega: &F,
    log_n: u32,
    size_inv: &F,
    precomputation_roots: &Vec<F>,
    domain_size: u64,
) -> Result<(), GPUError> {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };
            let closures = program_closures!(|program, args: (usize, &mut Vec<T>, &mut Vec<T>, &mut Vec<T>)|-> Result<(), GPUError> {
                let (gpu_index, input1, input2, input3) = args;
                let n = 1 << log_n;

                let mut max_log2_radix_n = MAX_LOG2_RADIX;
                let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
                if n <= 80000 {
                    max_log2_radix_n -= 1;
                    max_log2_local_work_size_n -= 1;
                }

                let max_deg = cmp::min(max_log2_radix_n, log_n);

                let pq_len = 1 << max_deg >> 1;
                let (is_create, mut pq_buffer) = program.create_pq_buffer::<F>(gpu_index, pq_len, OMGA_TYPE_IFFT_COSET)? ;
                if is_create {
                    let mut pq = vec![F::zero(); pq_len];
                    let twiddle = omega.pow([(n >> max_deg) as u64]);
                    pq[0] = F::one();
                    if max_deg > 1 {
                        pq[1] = twiddle;
                        for i in 2..pq_len {
                            pq[i] = pq[i - 1];
                            pq[i] *= &twiddle
                        }
                    }
                    program.write_from_buffer(&mut pq_buffer, &pq)?;
                }

                let (is_create, mut omegas_buffer) = program.create_omega_buffer::<F>(gpu_index, n,OMGA_TYPE_IFFT_COSET)? ;
                if is_create {
                     let time_writer = start_timer!(|| format!("写omega IFFT gpu_index = {:?} log_n = {} F个数 = {:?}", gpu_index ,log_n, n));
                     let mut omegas = vec![F::one(); n];
                     let ratio = (domain_size / n as u64) as usize;
                     for i in 0..n/2 {
                        omegas[i] = precomputation_roots[i * ratio];
                        omegas[i + n/2] =-precomputation_roots[i * ratio];
                     }
                    program.write_from_buffer(&mut omegas_buffer, &omegas)?;
                    end_timer!(time_writer);
                }


                //创建可回收缓存
                let mut src_buffer1 = program.create_buffer_pool::<T>(gpu_index,input1.len())?;
                let mut src_buffer2 = program.create_buffer_pool::<T>(gpu_index,input2.len())?;
                let mut src_buffer3 = program.create_buffer_pool::<T>(gpu_index,input3.len())?;
                let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index,n)?;
                let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index,n)?;
                let mut dst_buffer3 = program.create_buffer_pool::<T>(gpu_index,n)?;
                //写缓存
                program.write_from_buffer(&mut src_buffer1, &input1)?;
                program.write_from_buffer(&mut src_buffer2, &input2)?;
                program.write_from_buffer(&mut src_buffer3, &input3)?;

                let mut log_p = 0u32;
                while log_p < log_n {
                    let deg = cmp::min(max_deg, log_n - log_p);

                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                    // let global_work_size = n >> deg;
                    let global_work_size = (n >> deg) * 3;
                    let kernel = program.create_kernel("radix_fft_3_full", global_work_size as usize, local_work_size as usize)?;

                    let time_kernel = start_timer!(|| format!("核函数 radix_fft_3_full log_n = {}",log_n));
                    kernel
                        .arg(&src_buffer1)
                        .arg(&src_buffer2)
                        .arg(&src_buffer3)
                        .arg(&dst_buffer1)
                        .arg(&dst_buffer2)
                        .arg(&dst_buffer3)
                        .arg(&pq_buffer)
                        .arg(&omegas_buffer)
                        .arg(&n)
                        .arg(&log_p)
                        .arg(&deg)
                        .arg(&max_deg)
                        .run()?;
                    end_timer!(time_kernel);

                    log_p += deg;
                    std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                    std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                    std::mem::swap(&mut src_buffer3, &mut dst_buffer3);
                }

                let time_kernel = start_timer!(|| format!("ars_radix_ifft_3_recovery_cache  size_inv_buffer gpu内存all log_n = {}",log_n));
                let (is_create, mut size_inv_buffer) = program.create_size_inv_buffer::<F>(gpu_index, log_n as usize)? ;
                if is_create {
                    let time_c = start_timer!(|| format!("ars_radix_ifft_3_recovery_cache  size_inv_buffer 写内存 log_n = {}",log_n));
                    let size_invs = vec![*size_inv; 1];
                    program.write_from_buffer(&mut size_inv_buffer, &size_invs)?;
                    end_timer!(time_c);
                }
                end_timer!(time_kernel);



                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = 16;
                let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                let kernel1 = program.create_kernel("compute_mul_size_inv_3", global_work_size as usize, local_work_size as usize)?;

                kernel1
                    .arg(&src_buffer1)
                    .arg(&src_buffer2)
                    .arg(&src_buffer3)
                    .arg(&dst_buffer1)
                    .arg(&dst_buffer2)
                    .arg(&dst_buffer3)
                    .arg(&size_inv_buffer)
                    .arg(&(n as u32))
                    .arg(&(num_width as u32))
                    .run()?;
                program.read_into_buffer(&dst_buffer1, input1)?;
                program.read_into_buffer(&dst_buffer2, input2)?;
                program.read_into_buffer(&dst_buffer3, input3)?;

                //回收gpu缓存
                program.recovery_buffer_pool(gpu_index,input1.len(),src_buffer1)?;
                program.recovery_buffer_pool(gpu_index,input2.len(),src_buffer2)?;
                program.recovery_buffer_pool(gpu_index,input3.len(),src_buffer3)?;
                program.recovery_buffer_pool(gpu_index,n,dst_buffer1)?;
                program.recovery_buffer_pool(gpu_index,n,dst_buffer2)?;
                program.recovery_buffer_pool(gpu_index,n,dst_buffer3)?;

                program.recovery_pq_buffer(gpu_index, pq_len, OMGA_TYPE_IFFT_COSET, pq_buffer)?;
                program.recovery_omega_buffer(gpu_index, n, OMGA_TYPE_IFFT_COSET, omegas_buffer)?;
                program.recovery_size_inv_buffer(gpu_index, log_n as usize, size_inv_buffer)?;

                Ok(())
            });
            let time = start_timer!(|| format!("ars_radix_ifft_3_recovery_cache log_n = {}",log_n));
            context.program.p.run(closures, (context.program.id, a1, a2, a3))?;
            end_timer!(time);
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}


pub fn ars_radix_ifft_6_recovery<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>, a2: &mut Vec<T>, a3: &mut Vec<T>, a4: &mut Vec<T>, a5: &mut Vec<T>, a6: &mut Vec<T>,
    omega: &F, log_n: u32, size_inv: &F,
) -> Result<(), GPUError> {
    // println!("ars_radix_ifft_6_recovery log_n= {:?},  omega = {:?} ", log_n, omega);
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };

            let closures = program_closures!(|program, args: (usize, &mut Vec<T>, &mut Vec<T>, &mut Vec<T>, &mut Vec<T>, &mut Vec<T>, &mut Vec<T>)|
             -> Result<(), GPUError> {
                    let (gpu_index, input1, input2, input3, input4, input5, input6) = args;

                    let n = 1 << log_n;
                    let mut max_log2_radix_n = MAX_LOG2_RADIX;
                    let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
                    if n <= 80000 {
                        max_log2_radix_n -= 1;
                        max_log2_local_work_size_n -= 1;
                    }
                    let max_deg = cmp::min(max_log2_radix_n, log_n);

                    let mut pq = vec![F::zero(); 1 << max_deg >> 1];
                    let twiddle = omega.pow([(n >> max_deg) as u64]);
                    pq[0] = F::one();
                    if max_deg > 1 {
                        pq[1] = twiddle;
                        for i in 2..(1 << max_deg >> 1) {
                            pq[i] = pq[i - 1];
                            pq[i] *= &twiddle
                        }
                    }
                    let mut pq_buffer = program.create_buffer_pool::<F>(gpu_index, pq.len())? ;
                    program.write_from_buffer(&mut pq_buffer, &pq)?;


                    let mut omegas = vec![F::zero(); 32];
                    omegas[0] = *omega;
                    for i in 1..LOG2_MAX_ELEMENTS {
                        omegas[i] = omegas[i - 1].pow([2u64]);
                    }
                    let mut omegas_buffer = program.create_buffer_pool::<F>(gpu_index, omegas.len())?;
                    program.write_from_buffer(&mut omegas_buffer, &omegas)?;


                     //创建可回收缓存
                    let mut src_buffer1 = program.create_buffer_pool::<T>(gpu_index,input1.len())?;
                    let mut src_buffer2 = program.create_buffer_pool::<T>(gpu_index,input2.len())?;
                    let mut src_buffer3 = program.create_buffer_pool::<T>(gpu_index,input3.len())?;
                    let mut src_buffer4 = program.create_buffer_pool::<T>(gpu_index,input4.len())?;
                    let mut src_buffer5 = program.create_buffer_pool::<T>(gpu_index,input5.len())?;
                    let mut src_buffer6 = program.create_buffer_pool::<T>(gpu_index,input6.len())?;

                    let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index,n)?;
                    let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index,n)?;
                    let mut dst_buffer3 = program.create_buffer_pool::<T>(gpu_index,n)?;
                    let mut dst_buffer4 = program.create_buffer_pool::<T>(gpu_index,n)?;
                    let mut dst_buffer5 = program.create_buffer_pool::<T>(gpu_index,n)?;
                    let mut dst_buffer6 = program.create_buffer_pool::<T>(gpu_index,n)?;

                    //写缓存
                    program.write_from_buffer(&mut src_buffer1, &input1)?;
                    program.write_from_buffer(&mut src_buffer2, &input2)?;
                    program.write_from_buffer(&mut src_buffer3, &input3)?;
                    program.write_from_buffer(&mut src_buffer4, &input4)?;
                    program.write_from_buffer(&mut src_buffer5, &input5)?;
                    program.write_from_buffer(&mut src_buffer6, &input6)?;

                    let mut log_p = 0u32;
                    while log_p < log_n {
                        let deg = cmp::min(max_deg, log_n - log_p);

                        let n = 1u32 << log_n;
                        let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                        let global_work_size = n >> deg;
                        let kernel = program.create_kernel("radix_fft_6", global_work_size as usize, local_work_size as usize)?;

                        kernel
                            .arg(&src_buffer1)
                            .arg(&src_buffer2)
                            .arg(&src_buffer3)
                            .arg(&src_buffer4)
                            .arg(&src_buffer5)
                            .arg(&src_buffer6)
                            .arg(&dst_buffer1)
                            .arg(&dst_buffer2)
                            .arg(&dst_buffer3)
                            .arg(&dst_buffer4)
                            .arg(&dst_buffer5)
                            .arg(&dst_buffer6)
                            .arg(&pq_buffer)
                            .arg(&omegas_buffer)
                            .arg(&n)
                            .arg(&log_p)
                            .arg(&deg)
                            .arg(&max_deg)
                            .run()?;

                        log_p += deg;
                        std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                        std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                        std::mem::swap(&mut src_buffer3, &mut dst_buffer3);
                        std::mem::swap(&mut src_buffer4, &mut dst_buffer4);
                        std::mem::swap(&mut src_buffer5, &mut dst_buffer5);
                        std::mem::swap(&mut src_buffer6, &mut dst_buffer6);
                    }

                    let mut size_invs = vec![F::zero(); 1];
                    size_invs[0] = *size_inv;
                    let size_inv_buffer = program.create_buffer_from_slice(&size_invs)?;

                    let local_work_size = LOCAL_WORK_SIZE;
                    let global_work_size = n / LOCAL_WORK_SIZE;
                    let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                    let kernel1 = program.create_kernel("compute_mul_size_inv_6", global_work_size as usize, local_work_size as usize)?;

                    kernel1
                        .arg(&src_buffer1)
                        .arg(&src_buffer2)
                        .arg(&src_buffer3)
                        .arg(&src_buffer4)
                        .arg(&src_buffer5)
                        .arg(&src_buffer6)
                        .arg(&dst_buffer1)
                        .arg(&dst_buffer2)
                        .arg(&dst_buffer3)
                        .arg(&dst_buffer4)
                        .arg(&dst_buffer5)
                        .arg(&dst_buffer6)
                        .arg(&size_inv_buffer)
                        .arg(&(n as u32))
                        .arg(&(num_width as u32))
                        .run()?;
                    program.read_into_buffer(&dst_buffer1, input1)?;
                    program.read_into_buffer(&dst_buffer2, input2)?;
                    program.read_into_buffer(&dst_buffer3, input3)?;
                    program.read_into_buffer(&dst_buffer4, input4)?;
                    program.read_into_buffer(&dst_buffer5, input5)?;
                    program.read_into_buffer(&dst_buffer6, input6)?;


                    //回收gpu缓存
                    program.recovery_buffer_pool(gpu_index,input1.len(),src_buffer1)?;
                    program.recovery_buffer_pool(gpu_index,input2.len(),src_buffer2)?;
                    program.recovery_buffer_pool(gpu_index,input3.len(),src_buffer3)?;
                    program.recovery_buffer_pool(gpu_index,input4.len(),src_buffer4)?;
                    program.recovery_buffer_pool(gpu_index,input5.len(),src_buffer5)?;
                    program.recovery_buffer_pool(gpu_index,input6.len(),src_buffer6)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer1)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer2)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer3)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer4)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer5)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer6)?;
                    program.recovery_buffer_pool(gpu_index,omegas.len(),omegas_buffer)?;
                    program.recovery_buffer_pool(gpu_index,pq.len(),pq_buffer)?;

                    Ok(())
                });
            let time = start_timer!(|| format!("ars_radix_ifft_6_recovery log_n = {}",log_n));
            context.program.p.run(closures, (context.program.id, a1, a2, a3, a4, a5, a6)).unwrap();
            end_timer!(time);
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}

pub fn ars_radix_ifft_6_recovery_cache<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>, a2: &mut Vec<T>, a3: &mut Vec<T>, a4: &mut Vec<T>, a5: &mut Vec<T>, a6: &mut Vec<T>,
    omega: &F, log_n: u32, size_inv: &F,
    precomputation_roots: &Vec<F>,
    domain_size: u64,
) -> Result<(), GPUError> {
    // println!("ars_radix_ifft_6_recovery log_n= {:?},  omega = {:?} ", log_n, omega);
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };

            let closures = program_closures!(|program, args: (usize, &mut Vec<T>, &mut Vec<T>, &mut Vec<T>, &mut Vec<T>, &mut Vec<T>, &mut Vec<T>)|
             -> Result<(), GPUError> {
                    let (gpu_index, input1, input2, input3, input4, input5, input6) = args;
                    let n = 1 << log_n;


                    let mut max_log2_radix_n = MAX_LOG2_RADIX;
                    let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
                    if n <= 80000 {
                        max_log2_radix_n -= 1;
                        max_log2_local_work_size_n -= 1;
                    }

                    let max_deg = cmp::min(max_log2_radix_n, log_n);


                    let pq_len = 1 << max_deg >> 1;
                    let (is_create, mut pq_buffer) = program.create_pq_buffer::<F>(gpu_index, pq_len, OMGA_TYPE_IFFT_COSET)? ;
                    if is_create {
                        let mut pq = vec![F::zero(); pq_len];
                        let twiddle = omega.pow([(n >> max_deg) as u64]);
                        pq[0] = F::one();
                        if max_deg > 1 {
                            pq[1] = twiddle;
                            for i in 2..pq_len {
                                pq[i] = pq[i - 1];
                                pq[i] *= &twiddle
                            }
                        }
                        program.write_from_buffer(&mut pq_buffer, &pq)?;
                    }

                    // let mut pq = vec![F::zero(); 1 << max_deg >> 1];
                    // let twiddle = omega.pow([(n >> max_deg) as u64]);
                    // pq[0] = F::one();
                    // if max_deg > 1 {
                    //     pq[1] = twiddle;
                    //     for i in 2..(1 << max_deg >> 1) {
                    //         pq[i] = pq[i - 1];
                    //         pq[i] *= &twiddle
                    //     }
                    // }
                    // let mut pq_buffer = program.create_buffer_pool::<F>(gpu_index, pq.len())? ;
                    // program.write_from_buffer(&mut pq_buffer, &pq)?;


                    let (is_create, mut omegas_buffer) = program.create_omega_buffer::<F>(gpu_index, n,OMGA_TYPE_IFFT_COSET)? ;
                    if is_create {
                         let time_writer = start_timer!(|| format!("写omega IFFT gpu_index = {:?} log_n = {} F个数 = {:?}", gpu_index ,log_n, n));
                         let mut omegas = vec![F::one(); n];
                         let ratio = (domain_size / n as u64) as usize;
                         for i in 0..n/2 {
                            omegas[i] = precomputation_roots[i * ratio];
                            omegas[i + n/2] =-precomputation_roots[i * ratio];
                         }
                        program.write_from_buffer(&mut omegas_buffer, &omegas)?;
                        end_timer!(time_writer);
                    }

                    // let mut omegas = vec![F::zero(); 32];
                    // omegas[0] = *omega;
                    // for i in 1..LOG2_MAX_ELEMENTS {
                    //     omegas[i] = omegas[i - 1].pow([2u64]);
                    // }
                    // let mut omegas_buffer = program.create_buffer_pool::<F>(gpu_index, omegas.len())?;
                    // program.write_from_buffer(&mut omegas_buffer, &omegas)?;


                     //创建可回收缓存
                    let mut src_buffer1 = program.create_buffer_pool::<T>(gpu_index,input1.len())?;
                    let mut src_buffer2 = program.create_buffer_pool::<T>(gpu_index,input2.len())?;
                    let mut src_buffer3 = program.create_buffer_pool::<T>(gpu_index,input3.len())?;
                    let mut src_buffer4 = program.create_buffer_pool::<T>(gpu_index,input4.len())?;
                    let mut src_buffer5 = program.create_buffer_pool::<T>(gpu_index,input5.len())?;
                    let mut src_buffer6 = program.create_buffer_pool::<T>(gpu_index,input6.len())?;

                    let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index,n)?;
                    let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index,n)?;
                    let mut dst_buffer3 = program.create_buffer_pool::<T>(gpu_index,n)?;
                    let mut dst_buffer4 = program.create_buffer_pool::<T>(gpu_index,n)?;
                    let mut dst_buffer5 = program.create_buffer_pool::<T>(gpu_index,n)?;
                    let mut dst_buffer6 = program.create_buffer_pool::<T>(gpu_index,n)?;

                    //写缓存
                    program.write_from_buffer(&mut src_buffer1, &input1)?;
                    program.write_from_buffer(&mut src_buffer2, &input2)?;
                    program.write_from_buffer(&mut src_buffer3, &input3)?;
                    program.write_from_buffer(&mut src_buffer4, &input4)?;
                    program.write_from_buffer(&mut src_buffer5, &input5)?;
                    program.write_from_buffer(&mut src_buffer6, &input6)?;

                    let mut log_p = 0u32;
                    while log_p < log_n {
                        let deg = cmp::min(max_deg, log_n - log_p);

                        let n = 1u32 << log_n;
                        let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                        // let global_work_size = n >> deg;
                        let global_work_size = (n >> deg) * 6;
                        let kernel = program.create_kernel("radix_fft_6_full", global_work_size as usize, local_work_size as usize)?;

                        let time_kernel = start_timer!(|| format!("核函数 radix_fft_6_full log_n = {}",log_n));
                        kernel
                            .arg(&src_buffer1)
                            .arg(&src_buffer2)
                            .arg(&src_buffer3)
                            .arg(&src_buffer4)
                            .arg(&src_buffer5)
                            .arg(&src_buffer6)
                            .arg(&dst_buffer1)
                            .arg(&dst_buffer2)
                            .arg(&dst_buffer3)
                            .arg(&dst_buffer4)
                            .arg(&dst_buffer5)
                            .arg(&dst_buffer6)
                            .arg(&pq_buffer)
                            .arg(&omegas_buffer)
                            .arg(&n)
                            .arg(&log_p)
                            .arg(&deg)
                            .arg(&max_deg)
                            .run()?;
                         end_timer!(time_kernel);

                        log_p += deg;
                        std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                        std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                        std::mem::swap(&mut src_buffer3, &mut dst_buffer3);
                        std::mem::swap(&mut src_buffer4, &mut dst_buffer4);
                        std::mem::swap(&mut src_buffer5, &mut dst_buffer5);
                        std::mem::swap(&mut src_buffer6, &mut dst_buffer6);
                    }

                    // let mut size_invs = vec![F::zero(); 1];
                    // size_invs[0] = *size_inv;
                    // let size_inv_buffer = program.create_buffer_from_slice(&size_invs)?;

                    let time_kernel = start_timer!(|| format!("ars_radix_ifft_6_recovery_cache  size_inv_buffer gpu内存all log_n = {}",log_n));
                    let (is_create, mut size_inv_buffer) = program.create_size_inv_buffer::<F>(gpu_index, log_n as usize)? ;
                    if is_create {
                        let time_c = start_timer!(|| format!("ars_radix_ifft_6_recovery_cache  size_inv_buffer 写内存 log_n = {}",log_n));
                        let size_invs = vec![*size_inv; 1];
                        program.write_from_buffer(&mut size_inv_buffer, &size_invs)?;
                        end_timer!(time_c);
                    }
                    end_timer!(time_kernel);


                    let local_work_size = LOCAL_WORK_SIZE;
                    let global_work_size = 16;
                    let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                    let kernel1 = program.create_kernel("compute_mul_size_inv_6", global_work_size as usize, local_work_size as usize)?;

                    kernel1
                        .arg(&src_buffer1)
                        .arg(&src_buffer2)
                        .arg(&src_buffer3)
                        .arg(&src_buffer4)
                        .arg(&src_buffer5)
                        .arg(&src_buffer6)
                        .arg(&dst_buffer1)
                        .arg(&dst_buffer2)
                        .arg(&dst_buffer3)
                        .arg(&dst_buffer4)
                        .arg(&dst_buffer5)
                        .arg(&dst_buffer6)
                        .arg(&size_inv_buffer)
                        .arg(&(n as u32))
                        .arg(&(num_width as u32))
                        .run()?;
                    program.read_into_buffer(&dst_buffer1, input1)?;
                    program.read_into_buffer(&dst_buffer2, input2)?;
                    program.read_into_buffer(&dst_buffer3, input3)?;
                    program.read_into_buffer(&dst_buffer4, input4)?;
                    program.read_into_buffer(&dst_buffer5, input5)?;
                    program.read_into_buffer(&dst_buffer6, input6)?;


                    //回收gpu缓存
                    program.recovery_buffer_pool(gpu_index,input1.len(),src_buffer1)?;
                    program.recovery_buffer_pool(gpu_index,input2.len(),src_buffer2)?;
                    program.recovery_buffer_pool(gpu_index,input3.len(),src_buffer3)?;
                    program.recovery_buffer_pool(gpu_index,input4.len(),src_buffer4)?;
                    program.recovery_buffer_pool(gpu_index,input5.len(),src_buffer5)?;
                    program.recovery_buffer_pool(gpu_index,input6.len(),src_buffer6)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer1)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer2)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer3)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer4)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer5)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer6)?;

                    program.recovery_pq_buffer(gpu_index, pq_len, OMGA_TYPE_IFFT_COSET, pq_buffer)?;
                    program.recovery_omega_buffer(gpu_index, n, OMGA_TYPE_IFFT_COSET, omegas_buffer)?;
                    program.recovery_size_inv_buffer(gpu_index, log_n as usize, size_inv_buffer)?;

                    Ok(())
                });
            let time = start_timer!(|| format!("ars_radix_ifft_6_recovery_cache log_n = {}",log_n));
            context.program.p.run(closures, (context.program.id, a1, a2, a3, a4, a5, a6)).unwrap();
            end_timer!(time);
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}

pub fn ars_radix_coset_fft_recovery<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    context: &mut CudaContext,
    input: &mut Vec<T>,
    omega: &F,
    log_n: u32,
    g: F) -> Result<(), GPUError> {
    let closures = program_closures!(|program, args:(usize, &mut Vec<T>)| -> Result<(), GPUError> {
        let (gpu_index, input) = args;

        let n = 1 << log_n;
        let mut max_log2_radix_n = MAX_LOG2_RADIX;
        let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
        if n <= 80000 {
            max_log2_radix_n -= 1;
            max_log2_local_work_size_n -= 1;
        }
        let max_deg = cmp::min(max_log2_radix_n, log_n);

        let mut src_buffer = program.create_buffer_pool::<T>(gpu_index, n)?;
        let mut dst_buffer = program.create_buffer_pool::<T>(gpu_index, n)?;
        program.write_from_buffer(&mut src_buffer, &*input)?;


		let mut gen = vec![F::zero(); 1];
		gen[0] = g;
		let gen_buffer = program.create_buffer_from_slice(&gen)?;

        let local_work_size = LOCAL_WORK_SIZE;
        let global_work_size = 64;
		let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
        let kernel1 = program.create_kernel(
            "distribute_powers",
            global_work_size as usize,
            local_work_size as usize,
        )?;

        kernel1
            .arg(&src_buffer)
            .arg(&dst_buffer)
            .arg(&gen_buffer)
            .arg(&(n as u32))
            .arg(&(num_width as u32))
            .run()?;
		std::mem::swap(&mut src_buffer, &mut dst_buffer);
        
        // Precalculate:
        // [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]
        let mut pq = vec![F::zero(); 1 << max_deg >> 1];//T还是F?
        let twiddle = omega.pow([(n >> max_deg) as u64]);
        pq[0] =F::one();
        if max_deg > 1 {
            pq[1] = twiddle;
            for i in 2..(1 << max_deg >> 1) {
                pq[i] = pq[i - 1];
                pq[i] *= &twiddle // 乘法
            }
        }
        let mut pq_buffer = program.create_buffer_pool::<F>(gpu_index, pq.len())? ;
        program.write_from_buffer(&mut pq_buffer, &pq)?;

        // Precalculate [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]
        let mut omegas = vec![F::zero(); 32];
        omegas[0] = *omega;
        for i in 1..LOG2_MAX_ELEMENTS {
            omegas[i] = omegas[i - 1].pow([2u64]);
        }
        let mut omegas_buffer = program.create_buffer_pool::<F>(gpu_index, omegas.len())?;
        program.write_from_buffer(&mut omegas_buffer, &omegas)?;


        // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
        let mut log_p = 0u32;
        //let mut u = unsafe { program.create_buffer::<T>(1 << max_deg)? };
        // Each iteration performs a FFT round
        while log_p < log_n {
            // 1=>radix2, 2=>radix4, 3=>radix8, ...
            let deg = cmp::min(max_deg, log_n - log_p);

            let n = 1u32 << log_n;
            let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
            let global_work_size = n >> deg;
            let kernel = program.create_kernel(
                "radix_fft",
                global_work_size as usize,
                local_work_size as usize,
            )?;

            let time_kernel = start_timer!(|| format!("核函数 radix_fft log_n = {}",log_n));
            kernel
                .arg(&src_buffer)
                .arg(&dst_buffer)
                .arg(&pq_buffer)
                .arg(&omegas_buffer)
                .arg(&n)
                .arg(&log_p)
                .arg(&deg)
                .arg(&max_deg)
                .run()?;
            end_timer!(time_kernel);


            log_p += deg;
            std::mem::swap(&mut src_buffer, &mut dst_buffer);
        }

        program.read_into_buffer(&src_buffer, input)?;

        program.recovery_buffer_pool(gpu_index, input.len(), src_buffer)?;
        program.recovery_buffer_pool(gpu_index, input.len(), dst_buffer)?;
        program.recovery_buffer_pool(gpu_index, pq.len(), pq_buffer)?;
        program.recovery_buffer_pool(gpu_index, omegas.len(), omegas_buffer)?;

        Ok(())
    });
    let time = start_timer!(|| format!("ars_radix_coset_fft_recovery log_n = {}",log_n));
    let res = context.program.p.run(closures, (context.program.id, input));
    end_timer!(time);
    res
}


pub fn ars_radix_coset_fft_recovery_cache<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    input: &mut Vec<T>,
    omega: &F,
    log_n: u32,
    g: F,
    precomputation_roots: &Vec<F>,
    domain_size: u64,
) -> Result<(), GPUError> {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };
            let closures = program_closures!(|program, args:(usize, &mut Vec<T>)| -> Result<(), GPUError> {
                let (gpu_index, input) = args;
                let n = 1 << log_n;
                // All usages are safe as the buffers are initialized from either the host or the GPU
                // before they are read.
                let mut src_buffer = program.create_buffer_pool::<T>(gpu_index, n)?;
                let mut dst_buffer = program.create_buffer_pool::<T>(gpu_index, n)?;
                program.write_from_buffer(&mut src_buffer, &*input)?;

                let time_kernel = start_timer!(|| format!("ars_radix_coset_fft_recovery_cache  gen_buffer gpu内存all log_n = {}",log_n));
                let (is_create, mut gen_buffer) = program.create_gen_buffer::<F>(gpu_index, OMGA_TYPE_FFT_COSET)? ;
                if is_create {
                    let time_c = start_timer!(|| format!("ars_radix_coset_fft_recovery_cache  gen_buffer 写内存 log_n = {}",log_n));
                    let gen = vec![g; 1];
                    program.write_from_buffer(&mut gen_buffer,&gen)?;
                    end_timer!(time_c);
                }
                end_timer!(time_kernel);


                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = 64;
                let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                let kernel1 = program.create_kernel(
                    "distribute_powers",
                    global_work_size as usize,
                    local_work_size as usize,
                )?;

                kernel1
                    .arg(&src_buffer)
                    .arg(&dst_buffer)
                    .arg(&gen_buffer)
                    .arg(&(n as u32))
                    .arg(&(num_width as u32))
                    .run()?;
                std::mem::swap(&mut src_buffer, &mut dst_buffer);

                // The precalculated values pq` and `omegas` are valid for radix degrees up to `max_deg`
                let max_deg = cmp::min(MAX_LOG2_RADIX, log_n);

                let pq_len =  1 << max_deg >> 1;
                let (is_create, mut pq_buffer) = program.create_pq_buffer::<F>(gpu_index, pq_len, OMGA_TYPE_FFT_COSET)? ;
                if is_create {
                    let mut pq = vec![F::zero(); pq_len];//T还是F?
                    let twiddle = omega.pow([(n >> max_deg) as u64]);
                    pq[0] = F::one();
                    if max_deg > 1 {
                        pq[1] = twiddle;
                        for i in 2..pq_len {
                            pq[i] = pq[i - 1];
                            pq[i] *= &twiddle
                        }
                    }
                    program.write_from_buffer(&mut pq_buffer, &pq)?;
                }

                let (is_create, mut omegas_buffer) = program.create_omega_buffer::<F>(gpu_index, n,OMGA_TYPE_FFT_COSET)? ;
                if is_create {
                    let time_writer = start_timer!(|| format!("写omega FFT gpu_index = {:?} log_n = {} F个数 = {:?}", gpu_index ,log_n, n));
                    let mut omegas = vec![F::one(); n];
                    let ratio = (domain_size / n as u64) as usize;
                    for i in 0..n/2 {
                        omegas[i] = precomputation_roots[i * ratio];
                        omegas[i + n/2] =-precomputation_roots[i * ratio];
                    }
                    program.write_from_buffer(&mut omegas_buffer, &omegas)?;
                    end_timer!(time_writer);
                }


                // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
                let mut log_p = 0u32;
                //let mut u = unsafe { program.create_buffer::<T>(1 << max_deg)? };
                // Each iteration performs a FFT round
                while log_p < log_n {
                    // 1=>radix2, 2=>radix4, 3=>radix8, ...
                    let deg = cmp::min(max_deg, log_n - log_p);

                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE);
                    let global_work_size = n >> deg;
                    let kernel = program.create_kernel(
                        "radix_fft_full",
                        global_work_size as usize,
                        local_work_size as usize,
                    )?;

                    let time_kernel = start_timer!(|| format!("核函数 radix_fft log_n = {}",log_n));
                    kernel
                        .arg(&src_buffer)
                        .arg(&dst_buffer)
                        .arg(&pq_buffer)
                        .arg(&omegas_buffer)
                        .arg(&n)
                        .arg(&log_p)
                        .arg(&deg)
                        .arg(&max_deg)
                        .run()?;
                    end_timer!(time_kernel);

                    log_p += deg;
                    std::mem::swap(&mut src_buffer, &mut dst_buffer);
                }

                program.read_into_buffer(&src_buffer, input)?;

                program.recovery_buffer_pool(gpu_index, input.len(), src_buffer)?;
                program.recovery_buffer_pool(gpu_index, input.len(), dst_buffer)?;

                program.recovery_pq_buffer(gpu_index, pq_len, OMGA_TYPE_FFT_COSET, pq_buffer)?;
                program.recovery_omega_buffer(gpu_index, n, OMGA_TYPE_FFT_COSET, omegas_buffer)?;
                program.recovery_gen_buffer(gpu_index, OMGA_TYPE_FFT_COSET, gen_buffer)?;
                Ok(())
            });

            let time = start_timer!(|| format!("ars_radix_coset_fft_recovery_cache log_n = {}",log_n));
            context.program.p.run(closures, (context.program.id, input))?;
            end_timer!(time);

            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}


pub fn ars_radix_coset_fft_2_recovery<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>, a2: &mut Vec<T>,
    omega: &F, log_n: u32, g: F,
) -> Result<(), GPUError> {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };

            let closures = program_closures!(|program, args: (usize, &mut Vec<T>, &mut Vec<T>)| -> Result<(), GPUError> {
                let (gpu_index, input1, input2) = args;

                let n = 1 << log_n;
                let mut max_log2_radix_n = MAX_LOG2_RADIX;
                let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
                if n <= 80000 {
                    max_log2_radix_n -= 1;
                    max_log2_local_work_size_n -= 1;
                }
                let max_deg = cmp::min(max_log2_radix_n, log_n);

                let mut src_buffer1 = program.create_buffer_pool::<T>(gpu_index,input1.len())?;
                let mut src_buffer2 = program.create_buffer_pool::<T>(gpu_index,input2.len())?;
                let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index,n)?;
                let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index,n)?;
                //写缓存
                program.write_from_buffer(&mut src_buffer1, &input1)?;
                program.write_from_buffer(&mut src_buffer2, &input2)?;

                let mut gen = vec![F::zero(); 1];
                gen[0] = g;
                let gen_buffer = program.create_buffer_from_slice(&gen)?;

                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = 64;
                let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                let kernel1 = program.create_kernel("distribute_powers_2", global_work_size as usize, local_work_size as usize)?;

                kernel1
                    .arg(&src_buffer1)
                    .arg(&src_buffer2)
                    .arg(&dst_buffer1)
                    .arg(&dst_buffer2)
                    .arg(&gen_buffer)
                    .arg(&(n as u32))
                    .arg(&(num_width as u32))
                    .run()?;
                std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                std::mem::swap(&mut src_buffer2, &mut dst_buffer2);                

                let mut pq = vec![F::zero(); 1 << max_deg >> 1];
                let twiddle = omega.pow([(n >> max_deg) as u64]);
                pq[0] = F::one();
                if max_deg > 1 {
                    pq[1] = twiddle;
                    for i in 2..(1 << max_deg >> 1) {
                        pq[i] = pq[i - 1];
                        pq[i] *= &twiddle
                    }
                }
                let mut pq_buffer = program.create_buffer_pool::<F>(gpu_index, pq.len())? ;
                program.write_from_buffer(&mut pq_buffer, &pq)?;

                let mut omegas = vec![F::zero(); 32];
                omegas[0] = *omega;
                for i in 1..LOG2_MAX_ELEMENTS {
                    omegas[i] = omegas[i - 1].pow([2u64]);
                }
                let mut omegas_buffer = program.create_buffer_pool::<F>(gpu_index, omegas.len())? ;
                program.write_from_buffer(&mut omegas_buffer, &omegas)?;


                let mut log_p = 0u32;
                while log_p < log_n {
                    let deg = cmp::min(max_deg, log_n - log_p);

                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                    let global_work_size = n >> deg;
                    let kernel =
                        program.create_kernel("radix_fft_2", global_work_size as usize, local_work_size as usize)?;

                    let time_kernel = start_timer!(|| format!("核函数 radix_fft_2 log_n = {}",log_n));
                    kernel
                        .arg(&src_buffer1)
                        .arg(&src_buffer2)
                        .arg(&dst_buffer1)
                        .arg(&dst_buffer2)
                        .arg(&pq_buffer)
                        .arg(&omegas_buffer)
                        .arg(&n)
                        .arg(&log_p)
                        .arg(&deg)
                        .arg(&max_deg)
                        .run()?;
                    end_timer!(time_kernel);

                    log_p += deg;
                    std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                    std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                }

                program.read_into_buffer(&src_buffer1, input1)?;
                program.read_into_buffer(&src_buffer2, input2)?;

                //回收gpu缓存
                program.recovery_buffer_pool(gpu_index, input1.len(), src_buffer1)?;
                program.recovery_buffer_pool(gpu_index, input2.len(), src_buffer2)?;
                program.recovery_buffer_pool(gpu_index, n, dst_buffer1)?;
                program.recovery_buffer_pool(gpu_index, n, dst_buffer2)?;
                program.recovery_buffer_pool(gpu_index, omegas.len(), omegas_buffer)?;
                program.recovery_buffer_pool(gpu_index, pq.len(), pq_buffer)?;

                Ok(())
            });
            let time = start_timer!(|| format!("ars_radix_coset_fft_2_recovery log_n = {}",log_n));
            context.program.p.run(closures, (context.program.id, a1, a2)).unwrap();
            end_timer!(time);
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}

pub fn ars_radix_coset_fft_2_recovery_cache<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>,
    a2: &mut Vec<T>,
    omega: &F,
    log_n: u32,
    g: F,
    precomputation_roots: &Vec<F>,
    domain_size: u64,
) -> Result<(), GPUError> {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };

            let closures = program_closures!(|program, args: (usize, &mut Vec<T>, &mut Vec<T>)| -> Result<(), GPUError> {
                let (gpu_index, input1, input2) = args;
                let n = 1 << log_n;

                let mut src_buffer1 = program.create_buffer_pool::<T>(gpu_index,input1.len())?;
                let mut src_buffer2 = program.create_buffer_pool::<T>(gpu_index,input2.len())?;
                let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index,n)?;
                let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index,n)?;
                //写缓存
                program.write_from_buffer(&mut src_buffer1, &input1)?;
                program.write_from_buffer(&mut src_buffer2, &input2)?;

                let time_kernel = start_timer!(|| format!("ars_radix_coset_fft_2_recovery_cache  gen_buffer gpu内存all log_n = {}",log_n));
                let (is_create, mut gen_buffer) = program.create_gen_buffer::<F>(gpu_index, OMGA_TYPE_FFT_COSET)? ;
                if is_create {
                    let time_c = start_timer!(|| format!("ars_radix_coset_fft_2_recovery_cache  gen_buffer 写内存 log_n = {}",log_n));
                    let gen = vec![g; 1];
                    let gen_buffer = program.write_from_buffer(&mut gen_buffer,&gen)?;
                    end_timer!(time_c);
                }
                end_timer!(time_kernel);

                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = 64;
                let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                let kernel1 = program.create_kernel("distribute_powers_2", global_work_size as usize, local_work_size as usize)?;

                kernel1
                    .arg(&src_buffer1)
                    .arg(&src_buffer2)
                    .arg(&dst_buffer1)
                    .arg(&dst_buffer2)
                    .arg(&gen_buffer)
                    .arg(&(n as u32))
                    .arg(&(num_width as u32))
                    .run()?;
                std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                std::mem::swap(&mut src_buffer2, &mut dst_buffer2);

                let mut max_log2_radix_n = MAX_LOG2_RADIX;
                let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
                if n <= 80000 {
                    max_log2_radix_n -= 1;
                    max_log2_local_work_size_n -= 1;
                }
                let max_deg = cmp::min(max_log2_radix_n, log_n);

                let pq_len = 1 << max_deg >> 1;
                let (is_create, mut pq_buffer) = program.create_pq_buffer::<F>(gpu_index, pq_len, OMGA_TYPE_FFT_COSET)? ;
                if is_create {
                    let mut pq = vec![F::zero(); pq_len];
                    let twiddle = omega.pow([(n >> max_deg) as u64]);
                    pq[0] = F::one();
                    if max_deg > 1 {
                        pq[1] = twiddle;
                        for i in 2..pq_len {
                            pq[i] = pq[i - 1];
                            pq[i] *= &twiddle
                        }
                    }
                    program.write_from_buffer(&mut pq_buffer, &pq)?;
                }

                let (is_create, mut omegas_buffer) = program.create_omega_buffer::<F>(gpu_index, n,OMGA_TYPE_FFT_COSET)? ;
                if is_create {
                    let time_writer = start_timer!(|| format!("写omega FFT gpu_index = {:?} log_n = {} F个数 = {:?}", gpu_index ,log_n, n));
                    let mut omegas = vec![F::one(); n];
                    let ratio = (domain_size / n as u64) as usize;
                    for i in 0..n/2 {
                        omegas[i] = precomputation_roots[i * ratio];
                        omegas[i + n/2] =-precomputation_roots[i * ratio];
                    }
                    program.write_from_buffer(&mut omegas_buffer, &omegas)?;
                    end_timer!(time_writer);
                }

                let mut log_p = 0u32;
                while log_p < log_n {
                    let deg = cmp::min(max_deg, log_n - log_p);

                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                    // let global_work_size = n >> deg;
                    let global_work_size = (n >> deg) * 2;
                    let kernel =
                        program.create_kernel("radix_fft_2_full", global_work_size as usize, local_work_size as usize)?;

                    let time_kernel = start_timer!(|| format!("核函数 radix_fft_2_full log_n = {}",log_n));
                    kernel
                        .arg(&src_buffer1)
                        .arg(&src_buffer2)
                        .arg(&dst_buffer1)
                        .arg(&dst_buffer2)
                        .arg(&pq_buffer)
                        .arg(&omegas_buffer)
                        .arg(&n)
                        .arg(&log_p)
                        .arg(&deg)
                        .arg(&max_deg)
                        .run()?;
                    end_timer!(time_kernel);

                    log_p += deg;
                    std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                    std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                }

                program.read_into_buffer(&src_buffer1, input1)?;
                program.read_into_buffer(&src_buffer2, input2)?;

                //回收gpu缓存
                program.recovery_buffer_pool(gpu_index, input1.len(), src_buffer1)?;
                program.recovery_buffer_pool(gpu_index, input2.len(), src_buffer2)?;
                program.recovery_buffer_pool(gpu_index, n, dst_buffer1)?;
                program.recovery_buffer_pool(gpu_index, n, dst_buffer2)?;

                program.recovery_pq_buffer(gpu_index, pq_len, OMGA_TYPE_FFT_COSET, pq_buffer)?;
                program.recovery_omega_buffer(gpu_index, n, OMGA_TYPE_FFT_COSET, omegas_buffer)?;
                program.recovery_gen_buffer(gpu_index, OMGA_TYPE_FFT_COSET, gen_buffer)?;

                Ok(())
            });
            let time = start_timer!(|| format!("ars_radix_coset_fft_2_recovery_cache log_n = {}",log_n));
            context.program.p.run(closures, (context.program.id, a1, a2))?;
            end_timer!(time);
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}


pub fn ars_radix_coset_fft_3_recovery<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>, a2: &mut Vec<T>, a3: &mut Vec<T>,
    omega: &F, log_n: u32, g: F,
) -> Result<(), GPUError> {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };

            let closures = program_closures!(|program, args: (usize, &mut Vec<T>, &mut Vec<T>, &mut Vec<T>)|
             -> Result<(), GPUError> {
                    let (gpu_index, input1, input2, input3) = args;

                    let n = 1 << log_n;
                    let mut max_log2_radix_n = MAX_LOG2_RADIX;
                    let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
                    if n <= 80000 {
                        max_log2_radix_n -= 1;
                        max_log2_local_work_size_n -= 1;
                    }
                    let max_deg = cmp::min(max_log2_radix_n, log_n);

                    //创建可回收缓存
                    let mut src_buffer1 = program.create_buffer_pool::<T>(gpu_index, input1.len())?;
                    let mut src_buffer2 = program.create_buffer_pool::<T>(gpu_index, input2.len())?;
                    let mut src_buffer3 = program.create_buffer_pool::<T>(gpu_index, input3.len())?;
                    let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer3 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    //写缓存
                    program.write_from_buffer(&mut src_buffer1, &input1)?;
                    program.write_from_buffer(&mut src_buffer2, &input2)?;
                    program.write_from_buffer(&mut src_buffer3, &input3)?;

                    let mut gen = vec![F::zero(); 1];
                    gen[0] = g;
                    let gen_buffer = program.create_buffer_from_slice(&gen)?;

                    let local_work_size = LOCAL_WORK_SIZE;
                    let global_work_size = 64;
                    let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                    let kernel1 = program.create_kernel("distribute_powers_3", global_work_size as usize, local_work_size as usize)?;

                    kernel1
                        .arg(&src_buffer1)
                        .arg(&src_buffer2)
                        .arg(&src_buffer3)
                        .arg(&dst_buffer1)
                        .arg(&dst_buffer2)
                        .arg(&dst_buffer3)
                        .arg(&gen_buffer)
                        .arg(&(n as u32))
                        .arg(&(num_width as u32))
                        .run()?;
                    std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                    std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                    std::mem::swap(&mut src_buffer3, &mut dst_buffer3);                    

                    let mut pq = vec![F::zero(); 1 << max_deg >> 1];
                    let twiddle = omega.pow([(n >> max_deg) as u64]);
                    pq[0] = F::one();
                    if max_deg > 1 {
                        pq[1] = twiddle;
                        for i in 2..(1 << max_deg >> 1) {
                            pq[i] = pq[i - 1];
                            pq[i] *= &twiddle
                        }
                    }
                    let mut pq_buffer = program.create_buffer_pool::<F>(gpu_index, pq.len())? ;
                    program.write_from_buffer(&mut pq_buffer, &pq)?;


                    let mut omegas = vec![F::zero(); 32];
                    omegas[0] = *omega;
                    for i in 1..LOG2_MAX_ELEMENTS {
                        omegas[i] = omegas[i - 1].pow([2u64]);
                    }
                    let mut omegas_buffer = program.create_buffer_pool::<F>(gpu_index, omegas.len())? ;
                    program.write_from_buffer(&mut omegas_buffer, &omegas)?;

                    let mut log_p = 0u32;
                    while log_p < log_n {
                        let deg = cmp::min(max_deg, log_n - log_p);

                        let n = 1u32 << log_n;
                        let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                        let global_work_size = n >> deg;
                        let kernel =
                            program.create_kernel("radix_fft_3", global_work_size as usize, local_work_size as usize)?;

                        let time_kernel = start_timer!(|| format!("核函数 radix_fft_3 log_n = {}",log_n));
                        kernel
                            .arg(&src_buffer1)
                            .arg(&src_buffer2)
                            .arg(&src_buffer3)
                            .arg(&dst_buffer1)
                            .arg(&dst_buffer2)
                            .arg(&dst_buffer3)
                            .arg(&pq_buffer)
                            .arg(&omegas_buffer)
                            .arg(&n)
                            .arg(&log_p)
                            .arg(&deg)
                            .arg(&max_deg)
                            .run()?;
                       end_timer!(time_kernel);

                        log_p += deg;
                        std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                        std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                        std::mem::swap(&mut src_buffer3, &mut dst_buffer3);
                    }

                    program.read_into_buffer(&src_buffer1, input1)?;
                    program.read_into_buffer(&src_buffer2, input2)?;
                    program.read_into_buffer(&src_buffer3, input3)?;

                    //回收gpu缓存
                    program.recovery_buffer_pool(gpu_index,input1.len(),src_buffer1)?;
                    program.recovery_buffer_pool(gpu_index,input2.len(),src_buffer2)?;
                    program.recovery_buffer_pool(gpu_index,input3.len(),src_buffer3)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer1)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer2)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer3)?;
                    program.recovery_buffer_pool(gpu_index,omegas.len(),omegas_buffer)?;
                    program.recovery_buffer_pool(gpu_index,pq.len(),pq_buffer)?;

                    Ok(())
                });

            let time = start_timer!(|| format!("ars_radix_coset_fft_3_recovery log_n = {}",log_n));
            context.program.p.run(closures, (context.program.id, a1, a2, a3)).unwrap();
            end_timer!(time);
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}

pub fn ars_radix_coset_fft_3_recovery_cache<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>,
    a2: &mut Vec<T>,
    a3: &mut Vec<T>,
    omega: &F,
    log_n: u32,
    g: F,
    precomputation_roots: &Vec<F>,
    domain_size: u64,
) -> Result<(), GPUError> {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };

            let closures = program_closures!(|program, args: (usize, &mut Vec<T>, &mut Vec<T>, &mut Vec<T>)|
             -> Result<(), GPUError> {
                    let (gpu_index, input1, input2, input3) = args;
                    let n = 1 << log_n;

                    //创建可回收缓存
                    let mut src_buffer1 = program.create_buffer_pool::<T>(gpu_index, input1.len())?;
                    let mut src_buffer2 = program.create_buffer_pool::<T>(gpu_index, input2.len())?;
                    let mut src_buffer3 = program.create_buffer_pool::<T>(gpu_index, input3.len())?;
                    let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer3 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    //写缓存
                    program.write_from_buffer(&mut src_buffer1, &input1)?;
                    program.write_from_buffer(&mut src_buffer2, &input2)?;
                    program.write_from_buffer(&mut src_buffer3, &input3)?;


                    let time_kernel = start_timer!(|| format!("ars_radix_coset_fft_3_recovery_cache  gen_buffer gpu内存all log_n = {}",log_n));
                    let (is_create, mut gen_buffer) = program.create_gen_buffer::<F>(gpu_index, OMGA_TYPE_FFT_COSET)? ;
                    if is_create {
                        let time_c = start_timer!(|| format!("ars_radix_coset_fft_3_recovery_cache  gen_buffer 写内存 log_n = {}",log_n));
                        let gen = vec![g; 1];
                        let gen_buffer = program.write_from_buffer(&mut gen_buffer,&gen)?;
                        end_timer!(time_c);
                    }
                    end_timer!(time_kernel);


                    let local_work_size = LOCAL_WORK_SIZE;
                    let global_work_size = 64;
                    let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                    let kernel1 = program.create_kernel("distribute_powers_3", global_work_size as usize, local_work_size as usize)?;

                    kernel1
                        .arg(&src_buffer1)
                        .arg(&src_buffer2)
                        .arg(&src_buffer3)
                        .arg(&dst_buffer1)
                        .arg(&dst_buffer2)
                        .arg(&dst_buffer3)
                        .arg(&gen_buffer)
                        .arg(&(n as u32))
                        .arg(&(num_width as u32))
                        .run()?;
                    std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                    std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                    std::mem::swap(&mut src_buffer3, &mut dst_buffer3);

                    let mut max_log2_radix_n = MAX_LOG2_RADIX;
                    let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
                    if n <= 80000 {
                        max_log2_radix_n -= 1;
                        max_log2_local_work_size_n -= 1;
                    }
                    let max_deg = cmp::min(max_log2_radix_n, log_n);

                    let pq_len = 1 << max_deg >> 1;
                    let (is_create, mut pq_buffer) = program.create_pq_buffer::<F>(gpu_index, pq_len, OMGA_TYPE_FFT_COSET)? ;
                    if is_create {
                        let mut pq = vec![F::zero(); pq_len];
                        let twiddle = omega.pow([(n >> max_deg) as u64]);
                        pq[0] = F::one();
                        if max_deg > 1 {
                            pq[1] = twiddle;
                            for i in 2..pq_len {
                                pq[i] = pq[i - 1];
                                pq[i] *= &twiddle
                            }
                        }
                        program.write_from_buffer(&mut pq_buffer, &pq)?;
                    }

                    let (is_create, mut omegas_buffer) = program.create_omega_buffer::<F>(gpu_index, n,OMGA_TYPE_FFT_COSET)? ;
                    if is_create {
                        let time_writer = start_timer!(|| format!("写omega FFT gpu_index = {:?} log_n = {} F个数 = {:?}", gpu_index ,log_n, n));
                        let mut omegas = vec![F::one(); n];
                        let ratio = (domain_size / n as u64) as usize;
                        for i in 0..n/2 {
                            omegas[i] = precomputation_roots[i * ratio];
                            omegas[i + n/2] =-precomputation_roots[i * ratio];
                        }
                        program.write_from_buffer(&mut omegas_buffer, &omegas)?;
                        end_timer!(time_writer);
                    }

                    let mut log_p = 0u32;
                    while log_p < log_n {
                        let deg = cmp::min(max_deg, log_n - log_p);

                        let n = 1u32 << log_n;
                        let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                        // let global_work_size = n >> deg;
                        let global_work_size = (n >> deg) * 3;
                        let kernel =
                            program.create_kernel("radix_fft_3_full", global_work_size as usize, local_work_size as usize)?;

                        let time_kernel = start_timer!(|| format!("核函数 radix_fft_3_full log_n = {}",log_n));
                        kernel
                            .arg(&src_buffer1)
                            .arg(&src_buffer2)
                            .arg(&src_buffer3)
                            .arg(&dst_buffer1)
                            .arg(&dst_buffer2)
                            .arg(&dst_buffer3)
                            .arg(&pq_buffer)
                            .arg(&omegas_buffer)
                            .arg(&n)
                            .arg(&log_p)
                            .arg(&deg)
                            .arg(&max_deg)
                            .run()?;
                        end_timer!(time_kernel);

                        log_p += deg;
                        std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                        std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                        std::mem::swap(&mut src_buffer3, &mut dst_buffer3);
                    }

                    program.read_into_buffer(&src_buffer1, input1)?;
                    program.read_into_buffer(&src_buffer2, input2)?;
                    program.read_into_buffer(&src_buffer3, input3)?;

                    //回收gpu缓存
                    program.recovery_buffer_pool(gpu_index,input1.len(),src_buffer1)?;
                    program.recovery_buffer_pool(gpu_index,input2.len(),src_buffer2)?;
                    program.recovery_buffer_pool(gpu_index,input3.len(),src_buffer3)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer1)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer2)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer3)?;

                    program.recovery_pq_buffer(gpu_index, pq_len, OMGA_TYPE_FFT_COSET, pq_buffer)?;
                    program.recovery_omega_buffer(gpu_index, n, OMGA_TYPE_FFT_COSET, omegas_buffer)?;
                    program.recovery_gen_buffer(gpu_index, OMGA_TYPE_FFT_COSET, gen_buffer)?;

                    Ok(())
                });

            let time = start_timer!(|| format!("ars_radix_coset_fft_3_recovery_cache log_n = {}",log_n));
            context.program.p.run(closures, (context.program.id, a1, a2, a3))?;
            end_timer!(time);
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}


pub fn ars_radix_coset_fft_9_recovery<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>, a2: &mut Vec<T>, a3: &mut Vec<T>, a4: &mut Vec<T>, a5: &mut Vec<T>, a6: &mut Vec<T>,
    a7: &mut Vec<T>, a8: &mut Vec<T>, a9: &mut Vec<T>,
    omega: &F, log_n: u32, g: F,
) -> Result<(), GPUError> {
    // println!("ars_radix_coset_fft_9_recovery log_n= {:?},  omega = {:?} ", log_n, omega);
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };

            let closures = program_closures!(|program, args: (usize, &mut Vec<T>, &mut Vec<T>, &mut Vec<T>,
                &mut Vec<T>, &mut Vec<T>, &mut Vec<T>, &mut Vec<T>, &mut Vec<T>, &mut Vec<T>)|
             -> Result<(), GPUError> {
                    let (gpu_index,
                        input1, input2, input3,
                        input4, input5, input6,
                        input7, input8, input9,
                    ) = args;

                    let n = 1 << log_n;
                    let mut max_log2_radix_n = MAX_LOG2_RADIX;
                    let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
                    if n <= 80000 {
                        max_log2_radix_n -= 1;
                        max_log2_local_work_size_n -= 1;
                    }
                    let max_deg = cmp::min(max_log2_radix_n, log_n);

                    //创建可回收缓存
                    let mut src_buffer1 = program.create_buffer_pool::<T>(gpu_index, input1.len())?;
                    let mut src_buffer2 = program.create_buffer_pool::<T>(gpu_index, input2.len())?;
                    let mut src_buffer3 = program.create_buffer_pool::<T>(gpu_index, input3.len())?;
                    let mut src_buffer4 = program.create_buffer_pool::<T>(gpu_index, input4.len())?;
                    let mut src_buffer5 = program.create_buffer_pool::<T>(gpu_index, input5.len())?;
                    let mut src_buffer6 = program.create_buffer_pool::<T>(gpu_index, input6.len())?;
                    let mut src_buffer7 = program.create_buffer_pool::<T>(gpu_index, input7.len())?;
                    let mut src_buffer8 = program.create_buffer_pool::<T>(gpu_index, input8.len())?;
                    let mut src_buffer9 = program.create_buffer_pool::<T>(gpu_index, input9.len())?;

                    let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer3 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer4 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer5 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer6 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer7 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer8 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer9 = program.create_buffer_pool::<T>(gpu_index, n)?;

                    //写缓存
                    program.write_from_buffer(&mut src_buffer1, &input1)?;
                    program.write_from_buffer(&mut src_buffer2, &input2)?;
                    program.write_from_buffer(&mut src_buffer3, &input3)?;
                    program.write_from_buffer(&mut src_buffer4, &input4)?;
                    program.write_from_buffer(&mut src_buffer5, &input5)?;
                    program.write_from_buffer(&mut src_buffer6, &input6)?;
                    program.write_from_buffer(&mut src_buffer7, &input7)?;
                    program.write_from_buffer(&mut src_buffer8, &input8)?;
                    program.write_from_buffer(&mut src_buffer9, &input9)?;




                    let mut gen = vec![F::zero(); 1];
                    gen[0] = g;
                    let gen_buffer = program.create_buffer_from_slice(&gen)?;

                    let local_work_size = LOCAL_WORK_SIZE;
                    let global_work_size = 64;
                    let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                    let kernel1 = program.create_kernel("distribute_powers_9", global_work_size as usize, local_work_size as usize)?;

                    kernel1
                        .arg(&src_buffer1)
                        .arg(&src_buffer2)
                        .arg(&src_buffer3)
                        .arg(&src_buffer4)
                        .arg(&src_buffer5)
                        .arg(&src_buffer6)
                        .arg(&src_buffer7)
                        .arg(&src_buffer8)
                        .arg(&src_buffer9)
                        .arg(&dst_buffer1)
                        .arg(&dst_buffer2)
                        .arg(&dst_buffer3)
                        .arg(&dst_buffer4)
                        .arg(&dst_buffer5)
                        .arg(&dst_buffer6)
                        .arg(&dst_buffer7)
                        .arg(&dst_buffer8)
                        .arg(&dst_buffer9)
                        .arg(&gen_buffer)
                        .arg(&(n as u32))
                        .arg(&(num_width as u32))
                        .run()?;
                    std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                    std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                    std::mem::swap(&mut src_buffer3, &mut dst_buffer3);
                    std::mem::swap(&mut src_buffer4, &mut dst_buffer4);
                    std::mem::swap(&mut src_buffer5, &mut dst_buffer5);
                    std::mem::swap(&mut src_buffer6, &mut dst_buffer6);
                    std::mem::swap(&mut src_buffer7, &mut dst_buffer7);
                    std::mem::swap(&mut src_buffer8, &mut dst_buffer8);
                    std::mem::swap(&mut src_buffer9, &mut dst_buffer9);

                    let mut pq = vec![F::zero(); 1 << max_deg >> 1];
                    let twiddle = omega.pow([(n >> max_deg) as u64]);
                    pq[0] = F::one();
                    if max_deg > 1 {
                        pq[1] = twiddle;
                        for i in 2..(1 << max_deg >> 1) {
                            pq[i] = pq[i - 1];
                            pq[i] *= &twiddle
                        }
                    }
                    let mut pq_buffer = program.create_buffer_pool::<F>(gpu_index, pq.len())? ;
                    program.write_from_buffer(&mut pq_buffer, &pq)?;

                    let mut omegas = vec![F::zero(); 32];
                    omegas[0] = *omega;
                    for i in 1..LOG2_MAX_ELEMENTS {
                        omegas[i] = omegas[i - 1].pow([2u64]);
                    }
                    let mut omegas_buffer = program.create_buffer_pool::<F>(gpu_index, omegas.len())? ;
                    program.write_from_buffer(&mut omegas_buffer, &omegas)?;

                    let mut log_p = 0u32;
                    while log_p < log_n {
                        let deg = cmp::min(max_deg, log_n - log_p);

                        let n = 1u32 << log_n;
                        let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                        let global_work_size = n >> deg;
                        let kernel = program.create_kernel("radix_fft_9", global_work_size as usize, local_work_size as usize)?;

                        kernel
                            .arg(&src_buffer1)
                            .arg(&src_buffer2)
                            .arg(&src_buffer3)
                            .arg(&src_buffer4)
                            .arg(&src_buffer5)
                            .arg(&src_buffer6)
                            .arg(&src_buffer7)
                            .arg(&src_buffer8)
                            .arg(&src_buffer9)
                            .arg(&dst_buffer1)
                            .arg(&dst_buffer2)
                            .arg(&dst_buffer3)
                            .arg(&dst_buffer4)
                            .arg(&dst_buffer5)
                            .arg(&dst_buffer6)
                            .arg(&dst_buffer7)
                            .arg(&dst_buffer8)
                            .arg(&dst_buffer9)
                            .arg(&pq_buffer)
                            .arg(&omegas_buffer)
                            .arg(&n)
                            .arg(&log_p)
                            .arg(&deg)
                            .arg(&max_deg)
                            .run()?;

                        log_p += deg;
                        std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                        std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                        std::mem::swap(&mut src_buffer3, &mut dst_buffer3);
                        std::mem::swap(&mut src_buffer4, &mut dst_buffer4);
                        std::mem::swap(&mut src_buffer5, &mut dst_buffer5);
                        std::mem::swap(&mut src_buffer6, &mut dst_buffer6);
                        std::mem::swap(&mut src_buffer7, &mut dst_buffer7);
                        std::mem::swap(&mut src_buffer8, &mut dst_buffer8);
                        std::mem::swap(&mut src_buffer9, &mut dst_buffer9);
                    }

                    program.read_into_buffer(&src_buffer1, input1)?;
                    program.read_into_buffer(&src_buffer2, input2)?;
                    program.read_into_buffer(&src_buffer3, input3)?;
                    program.read_into_buffer(&src_buffer4, input4)?;
                    program.read_into_buffer(&src_buffer5, input5)?;
                    program.read_into_buffer(&src_buffer6, input6)?;
                    program.read_into_buffer(&src_buffer7, input7)?;
                    program.read_into_buffer(&src_buffer8, input8)?;
                    program.read_into_buffer(&src_buffer9, input9)?;

                     //回收gpu缓存
                    program.recovery_buffer_pool(gpu_index,input1.len(),src_buffer1)?;
                    program.recovery_buffer_pool(gpu_index,input2.len(),src_buffer2)?;
                    program.recovery_buffer_pool(gpu_index,input3.len(),src_buffer3)?;
                    program.recovery_buffer_pool(gpu_index,input4.len(),src_buffer4)?;
                    program.recovery_buffer_pool(gpu_index,input5.len(),src_buffer5)?;
                    program.recovery_buffer_pool(gpu_index,input6.len(),src_buffer6)?;
                    program.recovery_buffer_pool(gpu_index,input7.len(),src_buffer7)?;
                    program.recovery_buffer_pool(gpu_index,input8.len(),src_buffer8)?;
                    program.recovery_buffer_pool(gpu_index,input9.len(),src_buffer9)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer1)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer2)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer3)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer4)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer5)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer6)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer7)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer8)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer9)?;
                    program.recovery_buffer_pool(gpu_index,omegas.len(),omegas_buffer)?;
                    program.recovery_buffer_pool(gpu_index,pq.len(),pq_buffer)?;

                    Ok(())
                });

            let time = start_timer!(|| format!("ars_radix_coset_fft_9_recovery log_n = {}",log_n));
            context.program.p.run(closures, (context.program.id, a1, a2, a3, a4, a5, a6, a7, a8, a9)).unwrap();
            end_timer!(time);
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}


pub fn ars_radix_coset_fft_9_recovery_cache<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>, a2: &mut Vec<T>, a3: &mut Vec<T>, a4: &mut Vec<T>, a5: &mut Vec<T>, a6: &mut Vec<T>,
    a7: &mut Vec<T>, a8: &mut Vec<T>, a9: &mut Vec<T>,
    omega: &F, log_n: u32, g: F,
    precomputation_roots: &Vec<F>,
    domain_size: u64,
) -> Result<(), GPUError> {
    // println!("ars_radix_coset_fft_9_recovery log_n= {:?},  omega = {:?} ", log_n, omega);
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };

            let closures = program_closures!(|program, args: (usize, &mut Vec<T>, &mut Vec<T>, &mut Vec<T>,
                &mut Vec<T>, &mut Vec<T>, &mut Vec<T>, &mut Vec<T>, &mut Vec<T>, &mut Vec<T>)|
             -> Result<(), GPUError> {
                    let (gpu_index,
                        input1, input2, input3,
                        input4, input5, input6,
                        input7, input8, input9,
                    ) = args;
                    let n = 1 << log_n;
                     //创建可回收缓存
                    let mut src_buffer1 = program.create_buffer_pool::<T>(gpu_index, input1.len())?;
                    let mut src_buffer2 = program.create_buffer_pool::<T>(gpu_index, input2.len())?;
                    let mut src_buffer3 = program.create_buffer_pool::<T>(gpu_index, input3.len())?;
                    let mut src_buffer4 = program.create_buffer_pool::<T>(gpu_index, input4.len())?;
                    let mut src_buffer5 = program.create_buffer_pool::<T>(gpu_index, input5.len())?;
                    let mut src_buffer6 = program.create_buffer_pool::<T>(gpu_index, input6.len())?;
                    let mut src_buffer7 = program.create_buffer_pool::<T>(gpu_index, input7.len())?;
                    let mut src_buffer8 = program.create_buffer_pool::<T>(gpu_index, input8.len())?;
                    let mut src_buffer9 = program.create_buffer_pool::<T>(gpu_index, input9.len())?;

                    let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer3 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer4 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer5 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer6 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer7 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer8 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer9 = program.create_buffer_pool::<T>(gpu_index, n)?;

                    //写缓存
                    program.write_from_buffer(&mut src_buffer1, &input1)?;
                    program.write_from_buffer(&mut src_buffer2, &input2)?;
                    program.write_from_buffer(&mut src_buffer3, &input3)?;
                    program.write_from_buffer(&mut src_buffer4, &input4)?;
                    program.write_from_buffer(&mut src_buffer5, &input5)?;
                    program.write_from_buffer(&mut src_buffer6, &input6)?;
                    program.write_from_buffer(&mut src_buffer7, &input7)?;
                    program.write_from_buffer(&mut src_buffer8, &input8)?;
                    program.write_from_buffer(&mut src_buffer9, &input9)?;


                    // let mut gen = vec![F::zero(); 1];
                    // gen[0] = g;
                    // let gen_buffer = program.create_buffer_from_slice(&gen)?;

                    let time_kernel = start_timer!(|| format!("ars_radix_coset_fft_9_recovery_cache  gen_buffer gpu内存all log_n = {}",log_n));
                    let (is_create, mut gen_buffer) = program.create_gen_buffer::<F>(gpu_index, OMGA_TYPE_FFT_COSET)? ;
                    if is_create {
                        let time_c = start_timer!(|| format!("ars_radix_coset_fft_9_recovery_cache  gen_buffer 写内存 log_n = {}",log_n));
                        let gen = vec![g; 1];
                        program.write_from_buffer(&mut gen_buffer,&gen)?;
                        end_timer!(time_c);
                    }
                    end_timer!(time_kernel);


                    let local_work_size = LOCAL_WORK_SIZE;
                    let global_work_size = 64;
                    let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                    let kernel1 = program.create_kernel("distribute_powers_9", global_work_size as usize, local_work_size as usize)?;

                    kernel1
                        .arg(&src_buffer1)
                        .arg(&src_buffer2)
                        .arg(&src_buffer3)
                        .arg(&src_buffer4)
                        .arg(&src_buffer5)
                        .arg(&src_buffer6)
                        .arg(&src_buffer7)
                        .arg(&src_buffer8)
                        .arg(&src_buffer9)
                        .arg(&dst_buffer1)
                        .arg(&dst_buffer2)
                        .arg(&dst_buffer3)
                        .arg(&dst_buffer4)
                        .arg(&dst_buffer5)
                        .arg(&dst_buffer6)
                        .arg(&dst_buffer7)
                        .arg(&dst_buffer8)
                        .arg(&dst_buffer9)
                        .arg(&gen_buffer)
                        .arg(&(n as u32))
                        .arg(&(num_width as u32))
                        .run()?;
                    std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                    std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                    std::mem::swap(&mut src_buffer3, &mut dst_buffer3);
                    std::mem::swap(&mut src_buffer4, &mut dst_buffer4);
                    std::mem::swap(&mut src_buffer5, &mut dst_buffer5);
                    std::mem::swap(&mut src_buffer6, &mut dst_buffer6);
                    std::mem::swap(&mut src_buffer7, &mut dst_buffer7);
                    std::mem::swap(&mut src_buffer8, &mut dst_buffer8);
                    std::mem::swap(&mut src_buffer9, &mut dst_buffer9);

                    let mut max_log2_radix_n = MAX_LOG2_RADIX;
                    let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
                    if n <= 80000 {
                        max_log2_radix_n -= 1;
                        max_log2_local_work_size_n -= 1;
                    }
                    let max_deg = cmp::min(max_log2_radix_n, log_n);

                    // let mut pq = vec![F::zero(); 1 << max_deg >> 1];
                    // let twiddle = omega.pow([(n >> max_deg) as u64]);
                    // pq[0] = F::one();
                    // if max_deg > 1 {
                    //     pq[1] = twiddle;
                    //     for i in 2..(1 << max_deg >> 1) {
                    //         pq[i] = pq[i - 1];
                    //         pq[i] *= &twiddle
                    //     }
                    // }
                    // let mut pq_buffer = program.create_buffer_pool::<F>(gpu_index, pq.len())? ;
                    // program.write_from_buffer(&mut pq_buffer, &pq)?;

                    let pq_len = 1 << max_deg >> 1;
                    let (is_create, mut pq_buffer) = program.create_pq_buffer::<F>(gpu_index, pq_len, OMGA_TYPE_FFT_COSET)? ;
                    if is_create {
                        let mut pq = vec![F::zero(); pq_len];
                        let twiddle = omega.pow([(n >> max_deg) as u64]);
                        pq[0] = F::one();
                        if max_deg > 1 {
                            pq[1] = twiddle;
                            for i in 2..pq_len {
                                pq[i] = pq[i - 1];
                                pq[i] *= &twiddle
                            }
                        }
                        program.write_from_buffer(&mut pq_buffer, &pq)?;
                    }

                    // let mut omegas = vec![F::zero(); 32];
                    // omegas[0] = *omega;
                    // for i in 1..LOG2_MAX_ELEMENTS {
                    //     omegas[i] = omegas[i - 1].pow([2u64]);
                    // }
                    // let mut omegas_buffer = program.create_buffer_pool::<F>(gpu_index, omegas.len())? ;
                    // program.write_from_buffer(&mut omegas_buffer, &omegas)?;

                    let (is_create, mut omegas_buffer) = program.create_omega_buffer::<F>(gpu_index, n,OMGA_TYPE_FFT_COSET)? ;
                    if is_create {
                        let time_writer = start_timer!(|| format!("写omega FFT gpu_index = {:?} log_n = {} F个数 = {:?}", gpu_index ,log_n, n));
                        let mut omegas = vec![F::one(); n];
                        let ratio = (domain_size / n as u64) as usize;
                        for i in 0..n/2 {
                            omegas[i] = precomputation_roots[i * ratio];
                            omegas[i + n/2] =-precomputation_roots[i * ratio];
                        }
                        program.write_from_buffer(&mut omegas_buffer, &omegas)?;
                        end_timer!(time_writer);
                    }

                    let mut log_p = 0u32;
                    while log_p < log_n {
                        let deg = cmp::min(max_deg, log_n - log_p);

                        let n = 1u32 << log_n;
                        let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                        // let global_work_size = n >> deg;
                        let global_work_size = (n >> deg) * 9;
                        let kernel = program.create_kernel("radix_fft_9_full", global_work_size as usize, local_work_size as usize)?;
                        let time_kernel = start_timer!(|| format!("核函数 radix_fft_9_full log_n = {}",log_n));
                        kernel
                            .arg(&src_buffer1)
                            .arg(&src_buffer2)
                            .arg(&src_buffer3)
                            .arg(&src_buffer4)
                            .arg(&src_buffer5)
                            .arg(&src_buffer6)
                            .arg(&src_buffer7)
                            .arg(&src_buffer8)
                            .arg(&src_buffer9)
                            .arg(&dst_buffer1)
                            .arg(&dst_buffer2)
                            .arg(&dst_buffer3)
                            .arg(&dst_buffer4)
                            .arg(&dst_buffer5)
                            .arg(&dst_buffer6)
                            .arg(&dst_buffer7)
                            .arg(&dst_buffer8)
                            .arg(&dst_buffer9)
                            .arg(&pq_buffer)
                            .arg(&omegas_buffer)
                            .arg(&n)
                            .arg(&log_p)
                            .arg(&deg)
                            .arg(&max_deg)
                            .run()?;
                        end_timer!(time_kernel);

                        log_p += deg;
                        std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                        std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                        std::mem::swap(&mut src_buffer3, &mut dst_buffer3);
                        std::mem::swap(&mut src_buffer4, &mut dst_buffer4);
                        std::mem::swap(&mut src_buffer5, &mut dst_buffer5);
                        std::mem::swap(&mut src_buffer6, &mut dst_buffer6);
                        std::mem::swap(&mut src_buffer7, &mut dst_buffer7);
                        std::mem::swap(&mut src_buffer8, &mut dst_buffer8);
                        std::mem::swap(&mut src_buffer9, &mut dst_buffer9);
                    }

                    program.read_into_buffer(&src_buffer1, input1)?;
                    program.read_into_buffer(&src_buffer2, input2)?;
                    program.read_into_buffer(&src_buffer3, input3)?;
                    program.read_into_buffer(&src_buffer4, input4)?;
                    program.read_into_buffer(&src_buffer5, input5)?;
                    program.read_into_buffer(&src_buffer6, input6)?;
                    program.read_into_buffer(&src_buffer7, input7)?;
                    program.read_into_buffer(&src_buffer8, input8)?;
                    program.read_into_buffer(&src_buffer9, input9)?;

                     //回收gpu缓存
                    program.recovery_buffer_pool(gpu_index,input1.len(),src_buffer1)?;
                    program.recovery_buffer_pool(gpu_index,input2.len(),src_buffer2)?;
                    program.recovery_buffer_pool(gpu_index,input3.len(),src_buffer3)?;
                    program.recovery_buffer_pool(gpu_index,input4.len(),src_buffer4)?;
                    program.recovery_buffer_pool(gpu_index,input5.len(),src_buffer5)?;
                    program.recovery_buffer_pool(gpu_index,input6.len(),src_buffer6)?;
                    program.recovery_buffer_pool(gpu_index,input7.len(),src_buffer7)?;
                    program.recovery_buffer_pool(gpu_index,input8.len(),src_buffer8)?;
                    program.recovery_buffer_pool(gpu_index,input9.len(),src_buffer9)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer1)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer2)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer3)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer4)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer5)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer6)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer7)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer8)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer9)?;

                    program.recovery_pq_buffer(gpu_index, pq_len, OMGA_TYPE_FFT_COSET, pq_buffer)?;
                    program.recovery_omega_buffer(gpu_index, n, OMGA_TYPE_FFT_COSET, omegas_buffer)?;
                    program.recovery_gen_buffer(gpu_index, OMGA_TYPE_FFT_COSET, gen_buffer)?;

                    // program.recovery_buffer_pool(gpu_index,omegas.len(),omegas_buffer)?;
                    // program.recovery_buffer_pool(gpu_index,pq.len(),pq_buffer)?;

                    Ok(())
                });

            let time = start_timer!(|| format!("ars_radix_coset_fft_9_recovery log_n = {}",log_n));
            context.program.p.run(closures, (context.program.id, a1, a2, a3, a4, a5, a6, a7, a8, a9)).unwrap();
            end_timer!(time);
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}


pub fn ars_radix_coset_ifft_recovery<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    context: &mut CudaContext,
    input: &mut Vec<T>,
    omega: &F,
    log_n: u32,
    size_inv: &F,
    g: F) -> Result<(), GPUError> {
    let closures = program_closures!(|program, args:(usize, &mut Vec<T>) | -> Result<(), GPUError> {
        let (gpu_index, input) = args;

        let n = 1 << log_n;
        let mut max_log2_radix_n = MAX_LOG2_RADIX;
        let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
        if n <= 80000 {
            max_log2_radix_n -= 1;
            max_log2_local_work_size_n -= 1;
        }
        let max_deg = cmp::min(max_log2_radix_n, log_n);

        // All usages are safe as the buffers are initialized from either the host or the GPU
        let mut src_buffer = program.create_buffer_pool::<T>(gpu_index, n)?;
        let mut dst_buffer = program.create_buffer_pool::<T>(gpu_index, n)?;
        program.write_from_buffer(&mut src_buffer, &*input)?;

        // The precalculated values pq` and `omegas` are valid for radix degrees up to `max_deg`
        

        // Precalculate:
        // [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]
        let mut pq = vec![F::zero(); 1 << max_deg >> 1];//T还是F?
        let twiddle = omega.pow([(n >> max_deg) as u64]);
        pq[0] =F::one();
        if max_deg > 1 {
            pq[1] = twiddle;
            for i in 2..(1 << max_deg >> 1) {
                pq[i] = pq[i - 1];
                pq[i] *= &twiddle // 乘法
                //pq[i].mul_assign(&twiddle);
            }
        }
        let mut pq_buffer = program.create_buffer_pool::<F>(gpu_index, pq.len())? ;
        program.write_from_buffer(&mut pq_buffer, &pq)?;

        // Precalculate [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]
        let mut omegas = vec![F::zero(); 32];
        omegas[0] = *omega;
        for i in 1..LOG2_MAX_ELEMENTS {
            omegas[i] = omegas[i - 1].pow([2u64]);
        }
        let mut omegas_buffer = program.create_buffer_pool::<F>(gpu_index, omegas.len())? ;
        program.write_from_buffer(&mut omegas_buffer, &omegas)?;

        // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
        let mut log_p = 0u32;
        //let mut u = unsafe { program.create_buffer::<T>(1 << max_deg)? };
        // Each iteration performs a FFT round
        while log_p < log_n {
            // 1=>radix2, 2=>radix4, 3=>radix8, ...
            let deg = cmp::min(max_deg, log_n - log_p);

            let n = 1u32 << log_n;
            let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
            let global_work_size = n >> deg;
            let kernel = program.create_kernel(
                "radix_fft",
                global_work_size as usize,
                local_work_size as usize,
            )?;

            let time_kernel = start_timer!(|| format!("核函数 radix_fft log_n = {}",log_n));
            kernel
                .arg(&src_buffer)
                .arg(&dst_buffer)
                .arg(&pq_buffer)
                .arg(&omegas_buffer)
                .arg(&n)
                .arg(&log_p)
                .arg(&deg)
                .arg(&max_deg)
                .run()?;
            end_timer!(time_kernel);

            log_p += deg;
            std::mem::swap(&mut src_buffer, &mut dst_buffer);
        }

		let mut size_invs = vec![F::zero(); 1];
		size_invs[0] = *size_inv;
		let size_inv_buffer = program.create_buffer_from_slice(&size_invs)?;

        let local_work_size = LOCAL_WORK_SIZE;
        let global_work_size = 64;
		let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
        let kernel1 = program.create_kernel(
            "compute_mul_size_inv",
            global_work_size as usize,
            local_work_size as usize,
        )?;

        kernel1
            .arg(&src_buffer)
            .arg(&dst_buffer)
            .arg(&size_inv_buffer)
            .arg(&(n as u32))
            .arg(&(num_width as u32))
            .run()?;

        std::mem::swap(&mut src_buffer, &mut dst_buffer);

		let mut gen = vec![F::zero(); 1];
		gen[0] = g;
		let gen_buffer = program.create_buffer_from_slice(&gen)?;

        let local_work_size = LOCAL_WORK_SIZE;
        let global_work_size = 64;
		let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
        let kernel1 = program.create_kernel(
            "distribute_powers",
            global_work_size as usize,
            local_work_size as usize,
        )?;

        kernel1
            .arg(&src_buffer)
            .arg(&dst_buffer)
            .arg(&gen_buffer)
            .arg(&(n as u32))
            .arg(&(num_width as u32))
            .run()?;

        program.read_into_buffer(&dst_buffer, input)?;


        program.recovery_buffer_pool(gpu_index, input.len(), src_buffer)?;
        program.recovery_buffer_pool(gpu_index, input.len(), dst_buffer)?;
        program.recovery_buffer_pool(gpu_index, pq.len(), pq_buffer)?;
        program.recovery_buffer_pool(gpu_index, omegas.len(), omegas_buffer)?;


        Ok(())
    });
    let time = start_timer!(|| format!("ars_radix_coset_ifft_recovery log_n = {}",log_n));
    let res = context.program.p.run(closures, (context.program.id, input));
    end_timer!(time);
    res
}


pub fn ars_radix_coset_ifft_recovery_cache<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    input: &mut Vec<T>,
    omega: &F,
    log_n: u32,
    size_inv: &F,
    g: F,
    precomputation_roots: &Vec<F>,
    domain_size: u64,
) -> Result<(), GPUError> {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };

            let closures = program_closures!(|program, args:(usize, &mut Vec<T>) | -> Result<(), GPUError> {
                let (gpu_index, input) = args;
                let n = 1 << log_n;

                // All usages are safe as the buffers are initialized from either the host or the GPU
                let mut src_buffer = program.create_buffer_pool::<T>(gpu_index, n)?;
                let mut dst_buffer = program.create_buffer_pool::<T>(gpu_index, n)?;
                program.write_from_buffer(&mut src_buffer, &*input)?;

                // The precalculated values pq` and `omegas` are valid for radix degrees up to `max_deg`
                let max_deg = cmp::min(MAX_LOG2_RADIX, log_n);

                let pq_len = 1 << max_deg >> 1;
                let (is_create, mut pq_buffer) = program.create_pq_buffer::<F>(gpu_index, pq_len, OMGA_TYPE_IFFT_COSET)? ;
                if is_create {
                    let mut pq = vec![F::zero(); pq_len];
                    let twiddle = omega.pow([(n >> max_deg) as u64]);
                    pq[0] = F::one();
                    if max_deg > 1 {
                        pq[1] = twiddle;
                        for i in 2..pq_len {
                            pq[i] = pq[i - 1];
                            pq[i] *= &twiddle
                        }
                    }
                    program.write_from_buffer(&mut pq_buffer, &pq)?;
                }

                let (is_create, mut omegas_buffer) = program.create_omega_buffer::<F>(gpu_index, n,OMGA_TYPE_IFFT_COSET)? ;
                if is_create {
                    let time_writer = start_timer!(|| format!("写omega IFFT gpu_index = {:?} log_n = {} F个数 = {:?}", gpu_index ,log_n, n));
                    let mut omegas = vec![F::one(); n];
                    let ratio = (domain_size / n as u64) as usize;
                    for i in 0..n/2 {
                        omegas[i] = precomputation_roots[i * ratio];
                        omegas[i + n/2] =-precomputation_roots[i * ratio];
                    }
                    program.write_from_buffer(&mut omegas_buffer, &omegas)?;
                    end_timer!(time_writer);
                }

                // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
                let mut log_p = 0u32;
                //let mut u = unsafe { program.create_buffer::<T>(1 << max_deg)? };
                // Each iteration performs a FFT round
                while log_p < log_n {
                    // 1=>radix2, 2=>radix4, 3=>radix8, ...
                    let deg = cmp::min(max_deg, log_n - log_p);

                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE);
                    let global_work_size = n >> deg;
                    let kernel = program.create_kernel(
                        "radix_fft_full",
                        global_work_size as usize,
                        local_work_size as usize,
                    )?;

                     let time_kernel = start_timer!(|| format!("核函数 radix_fft_full log_n = {}",log_n));
                    kernel
                        .arg(&src_buffer)
                        .arg(&dst_buffer)
                        .arg(&pq_buffer)
                        .arg(&omegas_buffer)
                        .arg(&n)
                        .arg(&log_p)
                        .arg(&deg)
                        .arg(&max_deg)
                        .run()?;
                    end_timer!(time_kernel);

                    log_p += deg;
                    std::mem::swap(&mut src_buffer, &mut dst_buffer);
                }

                let time_kernel = start_timer!(|| format!("ars_radix_coset_ifft_recovery_cache  size_inv_buffer gpu内存all log_n = {}",log_n));
                let (is_create, mut size_inv_buffer) = program.create_size_inv_buffer::<F>(gpu_index, log_n as usize)? ;
                if is_create {
                    let time_c = start_timer!(|| format!("ars_radix_coset_ifft_recovery_cache  size_inv_buffer 写内存 log_n = {}",log_n));
                    let size_invs = vec![*size_inv; 1];
                    let size_inv_buffer = program.write_from_buffer(&mut size_inv_buffer, &size_invs)?;
                    end_timer!(time_c);
                }
                end_timer!(time_kernel);


                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = 64;
                let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                let kernel1 = program.create_kernel(
                    "compute_mul_size_inv",
                    global_work_size as usize,
                    local_work_size as usize,
                )?;

                kernel1
                    .arg(&src_buffer)
                    .arg(&dst_buffer)
                    .arg(&size_inv_buffer)
                    .arg(&(n as u32))
                    .arg(&(num_width as u32))
                    .run()?;

                std::mem::swap(&mut src_buffer, &mut dst_buffer);

                let time_kernel = start_timer!(|| format!("ars_radix_coset_ifft_recovery_cache  gen_buffer gpu内存all log_n = {}",log_n));
                let (is_create, mut gen_buffer) = program.create_gen_buffer::<F>(gpu_index, OMGA_TYPE_IFFT_COSET)? ;
                if is_create {
                    let time_c = start_timer!(|| format!("ars_radix_coset_ifft_recovery_cache  gen_buffer 写内存 log_n = {}",log_n));
                    let gen = vec![g; 1];
                    let gen_buffer = program.write_from_buffer(&mut gen_buffer,&gen)?;
                    end_timer!(time_c);
                }
                end_timer!(time_kernel);


                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = 64;
                let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                let kernel1 = program.create_kernel(
                    "distribute_powers",
                    global_work_size as usize,
                    local_work_size as usize,
                )?;

                kernel1
                    .arg(&src_buffer)
                    .arg(&dst_buffer)
                    .arg(&gen_buffer)
                    .arg(&(n as u32))
                    .arg(&(num_width as u32))
                    .run()?;

                program.read_into_buffer(&dst_buffer, input)?;


                program.recovery_buffer_pool(gpu_index, input.len(), src_buffer)?;
                program.recovery_buffer_pool(gpu_index, input.len(), dst_buffer)?;

                program.recovery_pq_buffer(gpu_index, pq_len, OMGA_TYPE_IFFT_COSET, pq_buffer)?;
                program.recovery_omega_buffer(gpu_index, n, OMGA_TYPE_IFFT_COSET, omegas_buffer)?;
                program.recovery_size_inv_buffer(gpu_index, log_n as usize, size_inv_buffer)?;
                program.recovery_gen_buffer(gpu_index, OMGA_TYPE_IFFT_COSET, gen_buffer)?;

                Ok(())
            });
            let time = start_timer!(|| format!("ars_radix_coset_ifft_recovery_cache log_n = {}",log_n));
            context.program.p.run(closures, (context.program.id, input))?;
            end_timer!(time);
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}


pub fn ars_radix_coset_ifft_2_recovery<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>, a2: &mut Vec<T>,
    omega: &F, log_n: u32, size_inv: &F, g: F,
) -> Result<(), GPUError> {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };

            let closures = program_closures!(|program, args: (usize, &mut Vec<T>, &mut Vec<T>)| -> Result<(), GPUError> {
                let (gpu_index, input1, input2) = args;

                let n = 1 << log_n;
                let mut max_log2_radix_n = MAX_LOG2_RADIX;
                let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
                if n <= 80000 {
                    max_log2_radix_n -= 1;
                    max_log2_local_work_size_n -= 1;
                }
                let max_deg = cmp::min(max_log2_radix_n, log_n);

                let mut src_buffer1 = program.create_buffer_pool::<T>(gpu_index,input1.len())?;
                let mut src_buffer2 = program.create_buffer_pool::<T>(gpu_index,input2.len())?;
                let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index,n)?;
                let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index,n)?;
                //写缓存
                program.write_from_buffer(&mut src_buffer1, &input1)?;
                program.write_from_buffer(&mut src_buffer2, &input2)?;

                let mut pq = vec![F::zero(); 1 << max_deg >> 1];
                let twiddle = omega.pow([(n >> max_deg) as u64]);
                pq[0] = F::one();
                if max_deg > 1 {
                    pq[1] = twiddle;
                    for i in 2..(1 << max_deg >> 1) {
                        pq[i] = pq[i - 1];
                        pq[i] *= &twiddle
                    }
                }
                let mut pq_buffer = program.create_buffer_pool::<F>(gpu_index, pq.len())? ;
                program.write_from_buffer(&mut pq_buffer, &pq)?;

                let mut omegas = vec![F::zero(); 32];
                omegas[0] = *omega;
                for i in 1..LOG2_MAX_ELEMENTS {
                    omegas[i] = omegas[i - 1].pow([2u64]);
                }
                let mut omegas_buffer = program.create_buffer_pool::<F>(gpu_index, omegas.len())? ;
                program.write_from_buffer(&mut omegas_buffer, &omegas)?;

                let mut log_p = 0u32;
                while log_p < log_n {
                    let deg = cmp::min(max_deg, log_n - log_p);

                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                    let global_work_size = n >> deg;
                    let kernel = program.create_kernel("radix_fft_2", global_work_size as usize, local_work_size as usize)?;

                    let time_kernel = start_timer!(|| format!("核函数 radix_fft_2 log_n = {}",log_n));
                    kernel
                        .arg(&src_buffer1)
                        .arg(&src_buffer2)
                        .arg(&dst_buffer1)
                        .arg(&dst_buffer2)
                        .arg(&pq_buffer)
                        .arg(&omegas_buffer)
                        .arg(&n)
                        .arg(&log_p)
                        .arg(&deg)
                        .arg(&max_deg)
                        .run()?;
                   end_timer!(time_kernel);

                    log_p += deg;
                    std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                    std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                }

                let mut size_invs = vec![F::zero(); 1];
                size_invs[0] = *size_inv;
                let size_inv_buffer = program.create_buffer_from_slice(&size_invs)?;

                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = 64;
                let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                let kernel1 = program.create_kernel("compute_mul_size_inv_2", global_work_size as usize, local_work_size as usize)?;

                kernel1
                    .arg(&src_buffer1)
                    .arg(&src_buffer2)
                    .arg(&dst_buffer1)
                    .arg(&dst_buffer2)
                    .arg(&size_inv_buffer)
                    .arg(&(n as u32))
                    .arg(&(num_width as u32))
                    .run()?;

                std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                std::mem::swap(&mut src_buffer2, &mut dst_buffer2);

                let mut gen = vec![F::zero(); 1];
                gen[0] = g;
                let gen_buffer = program.create_buffer_from_slice(&gen)?;

                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = 64;
                let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                let kernel1 = program.create_kernel("distribute_powers_2", global_work_size as usize, local_work_size as usize)?;

                kernel1
                    .arg(&src_buffer1)
                    .arg(&src_buffer2)
                    .arg(&dst_buffer1)
                    .arg(&dst_buffer2)
                    .arg(&gen_buffer)
                    .arg(&(n as u32))
                    .arg(&(num_width as u32))
                    .run()?;

                program.read_into_buffer(&dst_buffer1, input1)?;
                program.read_into_buffer(&dst_buffer2, input2)?;

                 //回收gpu缓存
                program.recovery_buffer_pool(gpu_index, input1.len(), src_buffer1)?;
                program.recovery_buffer_pool(gpu_index, input2.len(), src_buffer2)?;
                program.recovery_buffer_pool(gpu_index, n, dst_buffer1)?;
                program.recovery_buffer_pool(gpu_index, n, dst_buffer2)?;
                program.recovery_buffer_pool(gpu_index, omegas.len(), omegas_buffer)?;
                program.recovery_buffer_pool(gpu_index, pq.len(), pq_buffer)?;

                Ok(())
            });

            let time = start_timer!(|| format!("ars_radix_coset_ifft_2_recovery log_n = {}",log_n));
            context.program.p.run(closures, (context.program.id, a1, a2)).unwrap();
            end_timer!(time);
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}


pub fn ars_radix_coset_ifft_2_recovery_cache<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>,
    a2: &mut Vec<T>,
    omega: &F,
    log_n: u32,
    size_inv: &F,
    g: F,
    precomputation_roots: &Vec<F>,
    domain_size: u64,
) -> Result<(), GPUError> {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };

            let closures = program_closures!(|program, args: (usize, &mut Vec<T>, &mut Vec<T>)| -> Result<(), GPUError> {
                let (gpu_index, input1, input2) = args;
                let n = 1 << log_n;

                let mut src_buffer1 = program.create_buffer_pool::<T>(gpu_index,input1.len())?;
                let mut src_buffer2 = program.create_buffer_pool::<T>(gpu_index,input2.len())?;
                let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index,n)?;
                let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index,n)?;
                //写缓存
                program.write_from_buffer(&mut src_buffer1, &input1)?;
                program.write_from_buffer(&mut src_buffer2, &input2)?;


                let mut max_log2_radix_n = MAX_LOG2_RADIX;
                let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
                if n <= 80000 {
                    max_log2_radix_n -= 1;
                    max_log2_local_work_size_n -= 1;
                }
                let max_deg = cmp::min(max_log2_radix_n, log_n);

                let pq_len = 1 << max_deg >> 1;
                let mut pq = vec![F::zero(); pq_len];
                let (is_create, mut pq_buffer) = program.create_pq_buffer::<F>(gpu_index, pq_len, OMGA_TYPE_IFFT_COSET)? ;
                if is_create {
                    let twiddle = omega.pow([(n >> max_deg) as u64]);
                    pq[0] = F::one();
                    if max_deg > 1 {
                        pq[1] = twiddle;
                        for i in 2..pq_len {
                            pq[i] = pq[i - 1];
                            pq[i] *= &twiddle
                        }
                    }
                    program.write_from_buffer(&mut pq_buffer, &pq)?;
                }

                let (is_create, mut omegas_buffer) = program.create_omega_buffer::<F>(gpu_index, n, OMGA_TYPE_IFFT_COSET)? ;
                if is_create {
                    let time_writer = start_timer!(|| format!("写omega IFFT gpu_index = {:?} log_n = {} F个数 = {:?}", gpu_index ,log_n, n));
                    let mut omegas = vec![F::one(); n];
                    let ratio = (domain_size / n as u64) as usize;
                    for i in 0..n/2 {
                        omegas[i] = precomputation_roots[i * ratio];
                        omegas[i + n/2] =-precomputation_roots[i * ratio];
                    }
                    program.write_from_buffer(&mut omegas_buffer, &omegas)?;
                    end_timer!(time_writer);
                }

                let mut log_p = 0u32;
                while log_p < log_n {
                    let deg = cmp::min(max_deg, log_n - log_p);

                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                    // let global_work_size = n >> deg;
                    let global_work_size = (n >> deg) * 2;
                    let kernel = program.create_kernel("radix_fft_2_full", global_work_size as usize, local_work_size as usize)?;

                    let time_kernel = start_timer!(|| format!("核函数 radix_fft_2_full log_n = {}",log_n));
                    kernel
                        .arg(&src_buffer1)
                        .arg(&src_buffer2)
                        .arg(&dst_buffer1)
                        .arg(&dst_buffer2)
                        .arg(&pq_buffer)
                        .arg(&omegas_buffer)
                        .arg(&n)
                        .arg(&log_p)
                        .arg(&deg)
                        .arg(&max_deg)
                        .run()?;
                    end_timer!(time_kernel);

                    log_p += deg;
                    std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                    std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                }

                let time_kernel = start_timer!(|| format!("ars_radix_coset_ifft_2_recovery_cache  size_inv_buffer gpu内存all log_n = {}",log_n));
                let (is_create, mut size_inv_buffer) = program.create_size_inv_buffer::<F>(gpu_index, log_n as usize)? ;
                if is_create {
                    let time_c = start_timer!(|| format!("ars_radix_coset_ifft_2_recovery_cache  size_inv_buffer 写内存 log_n = {}",log_n));
                    let size_invs = vec![*size_inv; 1];
                    let size_inv_buffer = program.write_from_buffer(&mut size_inv_buffer, &size_invs)?;
                    end_timer!(time_c);
                }
                end_timer!(time_kernel);



                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = 64;
                let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                let kernel1 = program.create_kernel("compute_mul_size_inv_2", global_work_size as usize, local_work_size as usize)?;

                kernel1
                    .arg(&src_buffer1)
                    .arg(&src_buffer2)
                    .arg(&dst_buffer1)
                    .arg(&dst_buffer2)
                    .arg(&size_inv_buffer)
                    .arg(&(n as u32))
                    .arg(&(num_width as u32))
                    .run()?;

                std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                std::mem::swap(&mut src_buffer2, &mut dst_buffer2);

                let time_kernel = start_timer!(|| format!("ars_radix_coset_ifft_2_recovery_cache  gen_buffer gpu内存all log_n = {}",log_n));
                let (is_create, mut gen_buffer) = program.create_gen_buffer::<F>(gpu_index, OMGA_TYPE_IFFT_COSET)? ;
                if is_create {
                    let time_c = start_timer!(|| format!("ars_radix_coset_ifft_2_recovery_cache  gen_buffer 写内存 log_n = {}",log_n));
                    let gen = vec![g; 1];
                    let gen_buffer = program.write_from_buffer(&mut gen_buffer,&gen)?;
                    end_timer!(time_c);
                }
                end_timer!(time_kernel);


                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = 64;
                let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                let kernel1 = program.create_kernel("distribute_powers_2", global_work_size as usize, local_work_size as usize)?;

                kernel1
                    .arg(&src_buffer1)
                    .arg(&src_buffer2)
                    .arg(&dst_buffer1)
                    .arg(&dst_buffer2)
                    .arg(&gen_buffer)
                    .arg(&(n as u32))
                    .arg(&(num_width as u32))
                    .run()?;

                program.read_into_buffer(&dst_buffer1, input1)?;
                program.read_into_buffer(&dst_buffer2, input2)?;

                 //回收gpu缓存
                program.recovery_buffer_pool(gpu_index, input1.len(), src_buffer1)?;
                program.recovery_buffer_pool(gpu_index, input2.len(), src_buffer2)?;
                program.recovery_buffer_pool(gpu_index, n, dst_buffer1)?;
                program.recovery_buffer_pool(gpu_index, n, dst_buffer2)?;

                program.recovery_pq_buffer(gpu_index, pq_len, OMGA_TYPE_IFFT_COSET, pq_buffer)?;
                program.recovery_omega_buffer(gpu_index, n, OMGA_TYPE_IFFT_COSET, omegas_buffer)?;
                program.recovery_size_inv_buffer(gpu_index, log_n as usize, size_inv_buffer)?;
                program.recovery_gen_buffer(gpu_index, OMGA_TYPE_IFFT_COSET, gen_buffer)?;

                Ok(())
            });

            let time = start_timer!(|| format!("ars_radix_coset_ifft_2_recovery_cache log_n = {}",log_n));
            context.program.p.run(closures, (context.program.id, a1, a2))?;
            end_timer!(time);
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}


pub fn ars_radix_coset_ifft_3_recovery<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>, a2: &mut Vec<T>, a3: &mut Vec<T>,
    omega: &F, log_n: u32, size_inv: &F, g: F,
) -> Result<(), GPUError> {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };

            let closures = program_closures!(|program, args: (usize, &mut Vec<T>, &mut Vec<T>, &mut Vec<T>)|
             -> Result<(), GPUError> {
                    let (gpu_index, input1, input2, input3) = args;

                    let n = 1 << log_n;
                    let mut max_log2_radix_n = MAX_LOG2_RADIX;
                    let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
                    if n <= 80000 {
                        max_log2_radix_n -= 1;
                        max_log2_local_work_size_n -= 1;
                    }
                    let max_deg = cmp::min(max_log2_radix_n, log_n);

                    //创建可回收缓存
                    let mut src_buffer1 = program.create_buffer_pool::<T>(gpu_index, input1.len())?;
                    let mut src_buffer2 = program.create_buffer_pool::<T>(gpu_index, input2.len())?;
                    let mut src_buffer3 = program.create_buffer_pool::<T>(gpu_index, input3.len())?;
                    let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer3 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    //写缓存
                    program.write_from_buffer(&mut src_buffer1, &input1)?;
                    program.write_from_buffer(&mut src_buffer2, &input2)?;
                    program.write_from_buffer(&mut src_buffer3, &input3)?; 

                    let mut pq = vec![F::zero(); 1 << max_deg >> 1];
                    let twiddle = omega.pow([(n >> max_deg) as u64]);
                    pq[0] = F::one();
                    if max_deg > 1 {
                        pq[1] = twiddle;
                        for i in 2..(1 << max_deg >> 1) {
                            pq[i] = pq[i - 1];
                            pq[i] *= &twiddle
                        }
                    }
                    let mut pq_buffer = program.create_buffer_pool::<F>(gpu_index, pq.len())? ;
                    program.write_from_buffer(&mut pq_buffer, &pq)?;

                    let mut omegas = vec![F::zero(); 32];
                    omegas[0] = *omega;
                    for i in 1..LOG2_MAX_ELEMENTS {
                        omegas[i] = omegas[i - 1].pow([2u64]);
                    }
                    let mut omegas_buffer = program.create_buffer_pool::<F>(gpu_index, omegas.len())? ;
                    program.write_from_buffer(&mut omegas_buffer, &omegas)?;

                    let mut log_p = 0u32;
                    while log_p < log_n {
                        let deg = cmp::min(max_deg, log_n - log_p);

                        let n = 1u32 << log_n;
                        let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                        let global_work_size = n >> deg;
                        let kernel = program.create_kernel("radix_fft_3", global_work_size as usize, local_work_size as usize)?;

                        let time_kernel = start_timer!(|| format!("核函数 radix_fft_3 log_n = {}",log_n));
                        kernel
                            .arg(&src_buffer1)
                            .arg(&src_buffer2)
                            .arg(&src_buffer3)
                            .arg(&dst_buffer1)
                            .arg(&dst_buffer2)
                            .arg(&dst_buffer3)
                            .arg(&pq_buffer)
                            .arg(&omegas_buffer)
                            .arg(&n)
                            .arg(&log_p)
                            .arg(&deg)
                            .arg(&max_deg)
                            .run()?;
                        end_timer!(time_kernel);

                        log_p += deg;
                        std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                        std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                        std::mem::swap(&mut src_buffer3, &mut dst_buffer3);
                    }

                    let mut size_invs = vec![F::zero(); 1];
                    size_invs[0] = *size_inv;
                    let size_inv_buffer = program.create_buffer_from_slice(&size_invs)?;

                    let local_work_size = LOCAL_WORK_SIZE;
                    let global_work_size = 64;
                    let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                    let kernel1 = program.create_kernel("compute_mul_size_inv_3", global_work_size as usize, local_work_size as usize)?;

                    kernel1
                        .arg(&src_buffer1)
                        .arg(&src_buffer2)
                        .arg(&src_buffer3)
                        .arg(&dst_buffer1)
                        .arg(&dst_buffer2)
                        .arg(&dst_buffer3)
                        .arg(&size_inv_buffer)
                        .arg(&(n as u32))
                        .arg(&(num_width as u32))
                        .run()?;

                    std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                    std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                    std::mem::swap(&mut src_buffer3, &mut dst_buffer3);

                    let mut gen = vec![F::zero(); 1];
                    gen[0] = g;
                    let gen_buffer = program.create_buffer_from_slice(&gen)?;

                    let local_work_size = LOCAL_WORK_SIZE;
                    let global_work_size = 64;
                    let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                    let kernel1 = program.create_kernel("distribute_powers_3", global_work_size as usize, local_work_size as usize)?;

                    kernel1
                        .arg(&src_buffer1)
                        .arg(&src_buffer2)
                        .arg(&src_buffer3)
                        .arg(&dst_buffer1)
                        .arg(&dst_buffer2)
                        .arg(&dst_buffer3)
                        .arg(&gen_buffer)
                        .arg(&(n as u32))
                        .arg(&(num_width as u32))
                        .run()?;

                    program.read_into_buffer(&dst_buffer1, input1)?;
                    program.read_into_buffer(&dst_buffer2, input2)?;
                    program.read_into_buffer(&dst_buffer3, input3)?;

                    //回收gpu缓存
                    program.recovery_buffer_pool(gpu_index,input1.len(),src_buffer1)?;
                    program.recovery_buffer_pool(gpu_index,input2.len(),src_buffer2)?;
                    program.recovery_buffer_pool(gpu_index,input3.len(),src_buffer3)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer1)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer2)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer3)?;
                    program.recovery_buffer_pool(gpu_index,omegas.len(),omegas_buffer)?;
                    program.recovery_buffer_pool(gpu_index,pq.len(),pq_buffer)?;

                    Ok(())
                });
            let time = start_timer!(|| format!("ars_radix_coset_ifft_3_recovery log_n = {}",log_n));
            context.program.p.run(closures, (context.program.id, a1, a2, a3)).unwrap();
            end_timer!(time);
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}

pub fn ars_radix_coset_ifft_3_recovery_cache<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>,
    a2: &mut Vec<T>,
    a3: &mut Vec<T>,
    omega: &F,
    log_n: u32,
    size_inv: &F,
    g: F,
    precomputation_roots: &Vec<F>,
    domain_size: u64,
) -> Result<(), GPUError> {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };

            let closures = program_closures!(|program, args: (usize, &mut Vec<T>, &mut Vec<T>, &mut Vec<T>)|
             -> Result<(), GPUError> {
                    let time_kernel_in = start_timer!(|| format!("ars_radix_coset_ifft_3_recovery_cache GPU内部计算 log_n = {}",log_n));

                    let time_kernel = start_timer!(|| format!("ars_radix_coset_ifft_3_recovery_cache 取参数 log_n = {}",log_n));
                    let (gpu_index, input1, input2, input3) = args;
                    let n = 1 << log_n;
                    end_timer!(time_kernel);

                    //创建可回收缓存
                    let time_kernel = start_timer!(|| format!("ars_radix_coset_ifft_3_recovery_cache create_buffer_pool * 6 log_n = {}",log_n));

                    let mut src_buffer1 = program.create_buffer_pool::<T>(gpu_index, input1.len())?;
                    let mut src_buffer2 = program.create_buffer_pool::<T>(gpu_index, input2.len())?;
                    let mut src_buffer3 = program.create_buffer_pool::<T>(gpu_index, input3.len())?;
                    let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer3 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    end_timer!(time_kernel);

                    //写缓存
                    let time_kernel = start_timer!(|| format!("ars_radix_coset_ifft_3_recovery_cache write_from_buffer * 3 log_n = {}",log_n));
                    program.write_from_buffer(&mut src_buffer1, &input1)?;
                    program.write_from_buffer(&mut src_buffer2, &input2)?;
                    program.write_from_buffer(&mut src_buffer3, &input3)?;
                    end_timer!(time_kernel);


                    let time_kernel_2 = start_timer!(|| format!("ars_radix_coset_ifft_3_recovery_cache pq   log_n = {}",log_n));
                    let mut max_log2_radix_n = MAX_LOG2_RADIX;
                    let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
                    if n <= 80000 {
                        max_log2_radix_n -= 1;
                        max_log2_local_work_size_n -= 1;
                    }
                    let max_deg = cmp::min(max_log2_radix_n, log_n);
                    let pq_len = 1 << max_deg >> 1;
                    let (is_create, mut pq_buffer) = program.create_pq_buffer::<F>(gpu_index, pq_len, OMGA_TYPE_IFFT_COSET)? ;
                    if is_create {
                        let mut pq = vec![F::zero(); pq_len];
                        let twiddle = omega.pow([(n >> max_deg) as u64]);
                        pq[0] = F::one();
                        if max_deg > 1 {
                            pq[1] = twiddle;
                            for i in 2..pq_len {
                                pq[i] = pq[i - 1];
                                pq[i] *= &twiddle
                            }
                        }
                        program.write_from_buffer(&mut pq_buffer, &pq)?;
                    }
                    end_timer!(time_kernel_2);


                    let time_kernel_1 = start_timer!(|| format!("ars_radix_coset_ifft_3_recovery_cache omega  log_n = {}",log_n));
                    let (is_create, mut omegas_buffer) = program.create_omega_buffer::<F>(gpu_index, n, OMGA_TYPE_IFFT_COSET)? ;
                    if is_create {
                        let mut omegas = vec![F::one(); n];
                        let time_writer = start_timer!(|| format!("写omega IFFT gpu_index = {:?} log_n = {} F个数 = {:?}", gpu_index ,log_n, omegas.len()));
                        let ratio = (domain_size / n as u64) as usize;
                        for i in 0..n/2 {
                            omegas[i] = precomputation_roots[i * ratio];
                            omegas[i + n/2] =-precomputation_roots[i * ratio];
                        }
                        program.write_from_buffer(&mut omegas_buffer, &omegas)?;
                        end_timer!(time_writer);
                    }
                    end_timer!(time_kernel_1);

                    let time_kernel_while = start_timer!(|| format!("核函数 radix_fft_3_full 两次循环 log_n = {}",log_n));
                    let mut log_p = 0u32;
                    while log_p < log_n {
                        let deg = cmp::min(max_deg, log_n - log_p);

                        let n = 1u32 << log_n;
                        let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                        // let global_work_size = n >> deg;
                        let global_work_size = (n >> deg) * 3;
                        let kernel = program.create_kernel("radix_fft_3_full", global_work_size as usize, local_work_size as usize)?;

                        let time_kernel = start_timer!(|| format!("核函数 radix_fft_3_full log_n = {}",log_n));
                        kernel
                            .arg(&src_buffer1)
                            .arg(&src_buffer2)
                            .arg(&src_buffer3)
                            .arg(&dst_buffer1)
                            .arg(&dst_buffer2)
                            .arg(&dst_buffer3)
                            .arg(&pq_buffer)
                            .arg(&omegas_buffer)
                            .arg(&n)
                            .arg(&log_p)
                            .arg(&deg)
                            .arg(&max_deg)
                            .run()?;
                        end_timer!(time_kernel);

                        log_p += deg;
                        std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                        std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                        std::mem::swap(&mut src_buffer3, &mut dst_buffer3);
                    }
                    end_timer!(time_kernel_while);


                    let time_kernel = start_timer!(|| format!("ars_radix_coset_ifft_3_recovery_cache  size_inv_buffer gpu内存all log_n = {}",log_n));
                    let (is_create, mut size_inv_buffer) = program.create_size_inv_buffer::<F>(gpu_index, log_n as usize)? ;
                    if is_create {
                        let time_c = start_timer!(|| format!("ars_radix_coset_ifft_3_recovery_cache  size_inv_buffer 写内存 log_n = {}",log_n));
                        let size_invs = vec![*size_inv; 1];
                        let size_inv_buffer = program.write_from_buffer(&mut size_inv_buffer, &size_invs)?;
                        end_timer!(time_c);
                    }
                    end_timer!(time_kernel);

                    let time_kernel = start_timer!(|| format!("ars_radix_coset_ifft_3_recovery_cache compute_mul_size_inv_3  log_n = {}",log_n));
                    let local_work_size = LOCAL_WORK_SIZE;
                    let global_work_size = 64;
                    let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                    let kernel1 = program.create_kernel("compute_mul_size_inv_3", global_work_size as usize, local_work_size as usize)?;

                    kernel1
                        .arg(&src_buffer1)
                        .arg(&src_buffer2)
                        .arg(&src_buffer3)
                        .arg(&dst_buffer1)
                        .arg(&dst_buffer2)
                        .arg(&dst_buffer3)
                        .arg(&size_inv_buffer)
                        .arg(&(n as u32))
                        .arg(&(num_width as u32))
                        .run()?;

                    std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                    std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                    std::mem::swap(&mut src_buffer3, &mut dst_buffer3);
                    end_timer!(time_kernel);

                    let time_kernel = start_timer!(|| format!("ars_radix_coset_ifft_3_recovery_cache  gen_buffer gpu内存all log_n = {}",log_n));
                    let (is_create, mut gen_buffer) = program.create_gen_buffer::<F>(gpu_index, OMGA_TYPE_IFFT_COSET)? ;
                    if is_create {
                        let time_c = start_timer!(|| format!("ars_radix_coset_ifft_3_recovery_cache  gen_buffer 写内存 log_n = {}",log_n));
                        let gen = vec![g; 1];
                        let gen_buffer = program.write_from_buffer(&mut gen_buffer,&gen)?;
                        end_timer!(time_c);
                    }
                    end_timer!(time_kernel);

                    let time_kernel = start_timer!(|| format!("ars_radix_coset_ifft_3_recovery_cache distribute_powers_3 run  log_n = {}",log_n));
                    let local_work_size = LOCAL_WORK_SIZE;
                    let global_work_size = 64;
                    let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);

                    let kernel1 = program.create_kernel("distribute_powers_3", global_work_size as usize, local_work_size as usize)?;
                    kernel1
                        .arg(&src_buffer1)
                        .arg(&src_buffer2)
                        .arg(&src_buffer3)
                        .arg(&dst_buffer1)
                        .arg(&dst_buffer2)
                        .arg(&dst_buffer3)
                        .arg(&gen_buffer)
                        .arg(&(n as u32))
                        .arg(&(num_width as u32))
                        .run()?;
                    end_timer!(time_kernel);


                    let time_kernel = start_timer!(|| format!("ars_radix_coset_ifft_3_recovery_cache read_into_buffer log_n = {}",log_n));
                    program.read_into_buffer(&dst_buffer1, input1)?;
                    program.read_into_buffer(&dst_buffer2, input2)?;
                    program.read_into_buffer(&dst_buffer3, input3)?;
                    end_timer!(time_kernel);



                    let time_kernel = start_timer!(|| format!("ars_radix_coset_ifft_3_recovery_cache recovery_buffer_pool log_n = {}",log_n));
                    //回收gpu缓存
                    program.recovery_buffer_pool(gpu_index,input1.len(),src_buffer1)?;
                    program.recovery_buffer_pool(gpu_index,input2.len(),src_buffer2)?;
                    program.recovery_buffer_pool(gpu_index,input3.len(),src_buffer3)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer1)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer2)?;
                    program.recovery_buffer_pool(gpu_index,n,dst_buffer3)?;

                    program.recovery_pq_buffer(gpu_index, pq_len, OMGA_TYPE_IFFT_COSET, pq_buffer)?;
                    program.recovery_omega_buffer(gpu_index, n, OMGA_TYPE_IFFT_COSET, omegas_buffer)?;
                    program.recovery_size_inv_buffer(gpu_index, log_n as usize, size_inv_buffer)?;
                    program.recovery_gen_buffer(gpu_index, OMGA_TYPE_IFFT_COSET, gen_buffer)?;
                    end_timer!(time_kernel);
                    end_timer!(time_kernel_in);

                    Ok(())
                });
            let time = start_timer!(|| format!("ars_radix_coset_ifft_3_recovery_cache log_n = {}",log_n));
            context.program.p.run(closures, (context.program.id, a1, a2, a3))?;
            end_timer!(time);
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}

pub fn ars_batch_inversion_recovery<F: FftField>(input: &mut [F]) {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let closures = program_closures!(|program, args: (usize, &mut [F])| -> Result<(), GPUError> {
                let (gpu_index, v) = args;
                let f_poly_time0 = start_timer!(|| "batch_inversion closures");
                let fft_time = start_timer!(|| "[[ars_batch_inversion]] v_buffer create_buffer + write_from_buffer");
                let mut v_buffer =program.create_buffer_pool::<F>(gpu_index, v.len())?;
                program.write_from_buffer(&mut v_buffer, &*v)?;
                end_timer!(fft_time);

                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = 64;
                let num_width = (v.len()+ (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                let f_poly_time = start_timer!(|| "batch_inversion closures kernel");
                let kernel = program.create_kernel(
                    "batch_inversion",
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                .arg(&v_buffer)
                .arg(&(v.len() as u32))
                .arg(&(num_width as u32))
                .run()?;
                end_timer!(f_poly_time);
                program.read_into_buffer(&v_buffer, v)?;
                end_timer!(f_poly_time0);

                program.recovery_buffer_pool(gpu_index, v.len(), v_buffer)?;

                Ok(())
            });
            let f_poly_time = start_timer!(|| "batch_inversion");
            program.p.run(closures, (program.id, input)).unwrap();
            end_timer!(f_poly_time);

            ars_recycle_fft_program(program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
}

pub fn ars_batch_inversion_and_mul_gpu_recovery<F: FftField>(v: &mut [F], coeff: &F) {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let closures = program_closures!(|program, args:(usize, &mut [F]) | -> Result<(), GPUError> {
                let (gpu_index, v) = args;
                let f_poly_time = start_timer!(|| "batch_inversion_and_mul_gpu closures create_buffer");
                // let mut v_buffer = unsafe { program.create_buffer::<F>( v.len())? };
                let mut v_buffer =program.create_buffer_pool::<F>(gpu_index, v.len())? ;
                program.write_from_buffer(&mut v_buffer, &*v)?;

                let mut coeffs  = vec![F::zero(); 1];
                coeffs[0] = *coeff;
                // let coeff_buffer = program.create_buffer_from_slice(&coeffs)?;
                let mut coeff_buffer =program.create_buffer_pool::<F>(gpu_index, coeffs.len())?;
                program.write_from_buffer(&mut coeff_buffer, &coeffs)?;

                end_timer!(f_poly_time);
                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = 64;
                let num_width = (v.len()+ (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                let f_poly_time = start_timer!(|| "batch_inversion_and_mul_gpu closures kernel");
                let kernel = program.create_kernel(
                    "batch_inversion_and_mul",
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                .arg(&v_buffer)
                //.arg(&result_buffer)
                .arg(&coeff_buffer)
                .arg(&(v.len() as u32))
                .arg(&(num_width as u32))
                .run()?;
                end_timer!(f_poly_time);
                program.read_into_buffer(&v_buffer, v)?;

                program.recovery_buffer_pool(gpu_index, v.len(), v_buffer)?;
                program.recovery_buffer_pool(gpu_index, coeffs.len(), coeff_buffer)?;

                Ok(())
            });
            let f_poly_time = start_timer!(|| "batch_inversion_and_mul_gpu closures");
            program.p.run(closures, (program.id, v)).unwrap();
            end_timer!(f_poly_time);

            ars_recycle_fft_program(program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
}


pub fn ars_batch_inversion_and_prod_gpu_recovery<F: FftField>(v1: &mut [F], v2: &mut [F]) {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let closures = program_closures!(|program, args: (usize, &mut [F], &mut[F]) | -> Result<(), GPUError> {
                let (gpu_index, v1, v2) = args;
                let f_poly_time0 = start_timer!(|| "batch_inversion_and_and_prod_gpu closures");
                let fft_time = start_timer!(|| "[[ars_batch_inversion_and_mul_gpu]] v_buffer create_buffer + write_from_buffer");

                let mut v1_buffer =program.create_buffer_pool::<F>(gpu_index, v1.len())?;
                program.write_from_buffer(&mut v1_buffer, &*v1)?;
                end_timer!(fft_time);

                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = 64;
                let num_width = (v1.len()+ (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                let f_poly_time = start_timer!(|| "batch_inversion closures kernel");
                let kernel = program.create_kernel(
                    "batch_inversion",
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                .arg(&v1_buffer)
                .arg(&(v1.len() as u32))
                .arg(&(num_width as u32))
                .run()?;
                end_timer!(f_poly_time);


                let fft_time = start_timer!(|| "[[ars_batch_inversion_and_mul_gpu]] v_buffer create_buffer + write_from_buffer");
                 let mut v2_buffer =program.create_buffer_pool::<F>(gpu_index, v2.len())?;
                program.write_from_buffer(&mut v2_buffer, &*v2)?;
                end_timer!(fft_time);
                let f_poly_time = start_timer!(|| "compute_deri_vanishing_poly   product");
                    let kernel1 = program.create_kernel(
                        "product",
                        global_work_size as usize,
                        local_work_size as usize,
                    )?;

                    kernel1
                    .arg(&v1_buffer)
                    .arg(&v2_buffer)
                    //.arg(&result_buffer)
                    .arg(&(v1.len() as u32))
                    .arg(&(num_width as u32))
                    .run()?;
                    end_timer!(f_poly_time);

                    program.read_into_buffer(&v1_buffer, v1)?;

                    program.recovery_buffer_pool(gpu_index,v1.len(),v1_buffer)?;
                    program.recovery_buffer_pool(gpu_index,v2.len(),v2_buffer)?;

                    end_timer!(f_poly_time0);

                Ok(())
            });
            let f_poly_time = start_timer!(|| "batch_inversion_and_mul_gpu closures");
            program.p.run(closures, (program.id, v1, v2)).unwrap();
            end_timer!(f_poly_time);

            ars_recycle_fft_program(program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
}
