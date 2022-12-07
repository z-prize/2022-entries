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
use rust_gpu_tools::{cuda, program_closures, Device, GPUError, Program};

use std::{path::Path, process::Command};
use std::cmp;

use crate::fft::DomainCoeff;

use crate::fft::EvaluationDomain;

use crate::ars_gpu::*;

use snarkvm_utilities::cfg_iter;

pub const LOG2_MAX_ELEMENTS: usize = 32;
pub const MAX_LOG2_RADIX: u32 = 9;
pub const MAX_LOG2_LOCAL_WORK_SIZE: u32 = 8;

pub const LOCAL_WORK_SIZE: usize = 512;

static OMGA_TYPE_1: usize = 1;
static OMGA_TYPE_2: usize = 2;

static OMGA_TYPE_FFT_COSET: usize = 1;
static OMGA_TYPE_IFFT_COSET: usize = 2;

pub struct CudaContext {
    pub program: ArsProgram,
}

/// Generates the cuda msm binary.
pub fn ars_generate_cuda_binary<P: AsRef<Path>>(file_path: P, debug: bool) -> Result<(), GPUError> {
    // Find the latest compute code values.
    let nvcc_help = Command::new("nvcc").arg("-h").output()?.stdout;
    let nvcc_output =
        std::str::from_utf8(&nvcc_help).map_err(|_| GPUError::Generic("Missing nvcc command".to_string()))?;

    // Generate the parent directory.
    let mut resource_path = aleo_std::aleo_dir();
    resource_path.push("resources/fft_cuda/");
    std::fs::create_dir_all(resource_path)?;

    // TODO (raychu86): Fix this approach to generating files. Should just read all files in the `blst_377_cuda` directory.
    // Store the `.cu` and `.h` files temporarily for fatbin generation
    let mut asm_fft_cuda_path = aleo_std::aleo_dir();
    let mut asm_fft_cuda_h_path = aleo_std::aleo_dir();
    asm_fft_cuda_path.push("resources/fft_cuda/asm_fft_cuda.cu");
    asm_fft_cuda_h_path.push("resources/fft_cuda/asm_fft_cuda.h");

    let mut blst_ops_path = aleo_std::aleo_dir();
    let mut blst_ops_h_path = aleo_std::aleo_dir();
    blst_ops_path.push("resources/fft_cuda/blst_ops.cu");
    blst_ops_h_path.push("resources/fft_cuda/blst_ops.h");

    let mut fft_path = aleo_std::aleo_dir();
    fft_path.push("resources/fft_cuda/fft.cu");

    let mut types_fft_path = aleo_std::aleo_dir();
    types_fft_path.push("resources/fft_cuda/types_fft.h");

    let mut tests_fft_path = aleo_std::aleo_dir();
    tests_fft_path.push("resources/fft_cuda/tests_fft.cu");

    // Write all the files to the relative path.
    {
        let asm_fft_cuda = include_bytes!("./fft_cuda/asm_fft_cuda.cu");
        let asm_fft_cuda_h = include_bytes!("./fft_cuda/asm_fft_cuda.h");
        std::fs::write(&asm_fft_cuda_path, asm_fft_cuda)?;
        std::fs::write(&asm_fft_cuda_h_path, asm_fft_cuda_h)?;

        let blst_ops = include_bytes!("./fft_cuda/blst_ops.cu");
        let blst_ops_h = include_bytes!("./fft_cuda/blst_ops.h");
        std::fs::write(&blst_ops_path, blst_ops)?;
        std::fs::write(&blst_ops_h_path, blst_ops_h)?;

        let fft = include_bytes!("./fft_cuda/fft.cu");
        std::fs::write(&fft_path, fft)?;

        let types_fft = include_bytes!("./fft_cuda/types_fft.h");
        std::fs::write(&types_fft_path, types_fft)?;
    }

    // Generate the cuda fatbin.
    let mut command = Command::new("nvcc");
    command.arg(asm_fft_cuda_path.as_os_str()).arg(fft_path.as_os_str());

    // Add the debug feature for tests.
    if debug {
        let tests_fft = include_bytes!("./fft_cuda/test_fft.cu");
        std::fs::write(&tests_fft_path, tests_fft)?;

        command.arg(tests_fft_path.as_os_str()).arg("--device-debug");
    }

    // Add supported gencodes
    command
        .arg("--generate-code=arch=compute_60,code=sm_60")
        .arg("--generate-code=arch=compute_70,code=sm_70")
        .arg("--generate-code=arch=compute_75,code=sm_75");

    if nvcc_output.contains("compute_80") {
        command.arg("--generate-code=arch=compute_80,code=sm_80");
    }

    if nvcc_output.contains("compute_86") {
        command.arg("--generate-code=arch=compute_86,code=sm_86");
    }

    command.arg("-fatbin").arg("-dlink").arg("-o").arg(file_path.as_ref().as_os_str());

    eprintln!("\nRunning command: {:?}", command);

    let status = command.status()?;

    // Delete all the temporary .cu and .h files.
    {
        let _ = std::fs::remove_file(asm_fft_cuda_path);
        let _ = std::fs::remove_file(asm_fft_cuda_h_path);
        let _ = std::fs::remove_file(blst_ops_path);
        let _ = std::fs::remove_file(blst_ops_h_path);
        let _ = std::fs::remove_file(fft_path);
        let _ = std::fs::remove_file(types_fft_path);
        let _ = std::fs::remove_file(tests_fft_path);
    }

    // Execute the command.
    if !status.success() {
        gpu_msg_dispatch_notify("GPUError: Could not generate a new msm kernel".to_string());
        return Err(GPUError::KernelNotFound("Could not generate a new fft kernel".to_string()));
    }

    let version_command = Command::new("git").args(["log", "-1", "--pretty=format:%h"]).output().expect("failed to execute process");

    let mut command1 = Command::new("ln");
    command1.arg("-s").arg(file_path.as_ref().as_os_str()).arg(format!("{}-{}", file_path.as_ref().to_str().unwrap(), String::from_utf8(version_command.stdout).unwrap()));

    eprintln!("\nRunning command1: {:?}", command1);

    let status1 = command1.status()?;
    if !status1.success() {
        return Err(GPUError::KernelNotFound("Could not create msm kernel".to_string()));
    }

    Ok(())
}

/// Loads the fft.fatbin into an executable CUDA program.
pub fn ars_load_cuda_program(cur_dev_idx: usize) -> Result<Program, GPUError> {
    let devices: Vec<_> = Device::all();
    let device = match devices.get(cur_dev_idx) {
        Some(device) => device,
        None => {
            gpu_msg_dispatch_notify("GPUError: Device Not Found".to_string());
            return Err(GPUError::DeviceNotFound);
        }
    };

    // Find the path to the fft fatbin kernel
    let mut file_path = aleo_std::aleo_dir();
    let mut cuda_kernel = vec![];

    cuda_kernel = include_bytes!("./fft_fatbin/fft.fatbin").to_vec();

    let cuda_device = match device.cuda_device() {
        Some(device) => device,
        None => {
            gpu_msg_dispatch_notify("GPUError: Device Not Found".to_string());
            return Err(GPUError::DeviceNotFound);
        }
    };

    eprintln!("\nUsing '{}' as CUDA device with {} bytes of memory, for fft", device.name(), device.memory());

    // Load the cuda program from the kernel bytes.
    let cuda_program = match cuda::Program::from_bytes(cuda_device, &cuda_kernel) {
        Ok(program) => program,
        Err(err) => {
            // Delete the failing cuda kernel.
            std::fs::remove_file(file_path)?;
            return Err(err);
        }
    };

    Ok(Program::Cuda(cuda_program))
}

/// Use data caching to reduce data transmission
/// Using memory pools
pub fn ars_radix_fft<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    input: &mut Vec<T>, omega: &F, log_n: u32, 
    precomputation_roots: &Vec<F>, domain_size: u64,) -> Result<(), GPUError>
{
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
                    let mut pq = vec![F::zero(); pq_len];
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
                    let mut omegas = vec![F::one(); n];
                    let ratio = (domain_size / n as u64) as usize;
                    for i in 0..n/2 {
                    omegas[i] = precomputation_roots[i * ratio];
                    omegas[i + n/2] =-precomputation_roots[i * ratio];
                    }
                    program.write_from_buffer(&mut omegas_buffer, &omegas)?;                    
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

                    log_p += deg;
                    std::mem::swap(&mut src_buffer, &mut dst_buffer);
                }

                program.read_into_buffer(&src_buffer, input)?;
                program.recovery_buffer_pool(gpu_index, input.len(), src_buffer)?;
                program.recovery_buffer_pool(gpu_index, input.len(), dst_buffer)?;
                program.recovery_pq_buffer(gpu_index, pq_len, OMGA_TYPE_FFT_COSET, pq_buffer)?;
                program.recovery_omega_buffer(gpu_index, n, OMGA_TYPE_FFT_COSET, omegas_buffer)?;

                Ok(())
            });            
            context.program.p.run(closures, (context.program.id, input))?;            
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}

/// Consolidate multiple ffts into one to reduce the interaction between CPU memory and GPU memory
pub fn ars_radix_fft_2<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>, a2: &mut Vec<T>,
    omega: &F, log_n: u32,
    precomputation_roots: &Vec<F>,  domain_size: u64,
) -> Result<(), GPUError> 
{
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
                    let mut omegas = vec![F::one(); n];
                    let ratio = (domain_size / n as u64) as usize;
                    for i in 0..n/2 {
                        omegas[i] = precomputation_roots[i * ratio];
                        omegas[i + n/2] = -precomputation_roots[i * ratio];
                    }
                    program.write_from_buffer(&mut omegas_buffer, &omegas)?;                   
                }

                let mut src_buffer1 = program.create_buffer_pool::<T>(gpu_index,input1.len())?;
                let mut src_buffer2 = program.create_buffer_pool::<T>(gpu_index,input2.len())?;
                let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index,n)?;
                let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index,n)?;
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

                        log_p += deg;
                        std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                        std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                    }


                program.read_into_buffer(&src_buffer1, input1)?;
                program.read_into_buffer(&src_buffer2, input2)?;
                program.recovery_buffer_pool(gpu_index, input1.len(), src_buffer1)?;
                program.recovery_buffer_pool(gpu_index, input2.len(), src_buffer2)?;
                program.recovery_buffer_pool(gpu_index, n, dst_buffer1)?;
                program.recovery_buffer_pool(gpu_index, n, dst_buffer2)?;

                program.recovery_pq_buffer(gpu_index, pq_len, OMGA_TYPE_FFT_COSET, pq_buffer)?;
                program.recovery_omega_buffer(gpu_index, n, OMGA_TYPE_FFT_COSET, omegas_buffer)?;
                Ok(())
            });

            context.program.p.run(closures, (context.program.id, a1, a2))?;
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}

pub fn ars_radix_fft_3<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>, a2: &mut Vec<T>, a3: &mut Vec<T>,
    omega: &F, log_n: u32,
    precomputation_roots: &Vec<F>,  domain_size: u64,
) -> Result<(), GPUError> 
{
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
                    let mut omegas = vec![F::one(); n];
                    let ratio = (domain_size / n as u64) as usize;
                    for i in 0..n/2 {
                        omegas[i] = precomputation_roots[i * ratio];
                        omegas[i + n/2] =-precomputation_roots[i * ratio];
                    }
                    program.write_from_buffer(&mut omegas_buffer, &omegas)?;                    
                }

                let mut src_buffer1 = program.create_buffer_pool::<T>(gpu_index,input1.len())?;
                let mut src_buffer2 = program.create_buffer_pool::<T>(gpu_index,input2.len())?;
                let mut src_buffer3 = program.create_buffer_pool::<T>(gpu_index,input3.len())?;
                let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index,n)?;
                let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index,n)?;
                let mut dst_buffer3 = program.create_buffer_pool::<T>(gpu_index,n)?;
                program.write_from_buffer(&mut src_buffer1, &input1)?;
                program.write_from_buffer(&mut src_buffer2, &input2)?;
                program.write_from_buffer(&mut src_buffer3, &input3)?;

                let mut log_p = 0u32;
                while log_p < log_n {
                    let deg = cmp::min(max_deg, log_n - log_p);
                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                    let global_work_size = (n >> deg) * 3;

                    let kernel = program.create_kernel("radix_fft_3_full", global_work_size as usize, local_work_size as usize)?;

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

                    log_p += deg;

                    std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                    std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                    std::mem::swap(&mut src_buffer3, &mut dst_buffer3);
                }

                program.read_into_buffer(&src_buffer1, input1)?;
                program.read_into_buffer(&src_buffer2, input2)?;
                program.read_into_buffer(&src_buffer3, input3)?;
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
            context.program.p.run(closures, (context.program.id, a1, a2, a3))?;
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}


pub fn ars_radix_ifft<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    input: &mut Vec<T>, omega: &F,
    log_n: u32, size_inv: &F,
    precomputation_roots: &Vec<F>, domain_size: u64,) -> Result<(), GPUError> 
{
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };

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

                let pq_len = 1 << max_deg >> 1;
                let (is_create, mut pq_buffer) = program.create_pq_buffer::<F>(gpu_index, pq_len, OMGA_TYPE_IFFT_COSET)? ;
                if is_create {
                    let mut pq = vec![F::zero(); pq_len];
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

                let (is_create, mut omegas_buffer) = program.create_omega_buffer::<F>(gpu_index, n,OMGA_TYPE_IFFT_COSET)? ;
                if is_create {
                    let mut omegas = vec![F::one(); n];
                    let ratio = (domain_size / n as u64) as usize;
                    for i in 0..n/2 {
                    omegas[i] = precomputation_roots[i * ratio];
                    omegas[i + n/2] =-precomputation_roots[i * ratio];
                    }
                    program.write_from_buffer(&mut omegas_buffer, &omegas)?;                    
                }

                let mut src_buffer =  program.create_buffer_pool::<T>(gpu_index, n)?;
                let mut dst_buffer = program.create_buffer_pool::<T>(gpu_index, n)?;
                program.write_from_buffer(&mut src_buffer, &*input)?;

                // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
                let mut log_p = 0u32;
                while log_p < log_n {
                    let deg = cmp::min(max_deg, log_n - log_p);
                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE);
                    let global_work_size = n >> deg;

                    let kernel = program.create_kernel(
                        "radix_fft_full",
                        global_work_size as usize,
                        local_work_size as usize,
                    )?;

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

                    log_p += deg;
                    std::mem::swap(&mut src_buffer, &mut dst_buffer);
                }

                let (is_create, mut size_inv_buffer) = program.create_size_inv_buffer::<F>(gpu_index, log_n as usize)? ;
                if is_create {
                    let size_invs = vec![*size_inv; 1];
                    program.write_from_buffer(&mut size_inv_buffer, &size_invs)?;                   
                }

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
                program.recovery_buffer_pool(gpu_index,n,src_buffer)?;
                program.recovery_buffer_pool(gpu_index,n,dst_buffer)?;
                program.recovery_pq_buffer(gpu_index, pq_len, OMGA_TYPE_IFFT_COSET, pq_buffer)?;
                program.recovery_omega_buffer(gpu_index, n, OMGA_TYPE_IFFT_COSET, omegas_buffer)?;
                program.recovery_size_inv_buffer(gpu_index, log_n as usize, size_inv_buffer)?;
                Ok(())
            });            
            context.program.p.run(closures, (context.program.id, input))?;
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}

pub fn ars_radix_ifft_2<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>, a2: &mut Vec<T>,
    omega: &F, log_n: u32, size_inv: &F,
    precomputation_roots: &Vec<F>, domain_size: u64,) -> Result<(), GPUError> 
{
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
                    let ratio = (domain_size / n as u64) as usize;
                    for i in 0..n/2 {
                        omegas[i] = precomputation_roots[i * ratio];
                        omegas[i + n/2] =-precomputation_roots[i * ratio];
                    }
                    program.write_from_buffer(&mut omegas_buffer, &omegas)?;                    
                }

                let mut src_buffer1 = program.create_buffer_pool::<T>(gpu_index,input1.len())?;
                let mut src_buffer2 = program.create_buffer_pool::<T>(gpu_index,input2.len())?;
                let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index,n)?;
                let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index,n)?;
                program.write_from_buffer(&mut src_buffer1, &input1)?;
                program.write_from_buffer(&mut src_buffer2, &input2)?;

                let mut log_p = 0u32;
                while log_p < log_n {
                    let deg = cmp::min(max_deg, log_n - log_p);

                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                    let global_work_size = (n >> deg) * 2;
                    let kernel = program.create_kernel("radix_fft_2_full", global_work_size as usize, local_work_size as usize)?;
                    
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

                    log_p += deg;
                    std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                    std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                }

                let (is_create, mut size_inv_buffer) = program.create_size_inv_buffer::<F>(gpu_index, log_n as usize)? ;
                if is_create {                    
                    let size_invs = vec![*size_inv; 1];
                    program.write_from_buffer(&mut size_inv_buffer, &size_invs)?;;
                }

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
                program.recovery_buffer_pool(gpu_index, input1.len(), src_buffer1)?;
                program.recovery_buffer_pool(gpu_index, input2.len(), src_buffer2)?;
                program.recovery_buffer_pool(gpu_index, n, dst_buffer1)?;
                program.recovery_buffer_pool(gpu_index, n, dst_buffer2)?;
                program.recovery_pq_buffer(gpu_index, pq_len, OMGA_TYPE_IFFT_COSET, pq_buffer)?;
                program.recovery_omega_buffer(gpu_index, n, OMGA_TYPE_IFFT_COSET, omegas_buffer)?;
                program.recovery_size_inv_buffer(gpu_index, log_n as usize, size_inv_buffer)?;

                Ok(())
            });
            context.program.p.run(closures, (context.program.id, a1, a2))?;
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}

pub fn ars_radix_ifft_3<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>, a2: &mut Vec<T>, a3: &mut Vec<T>,
    omega: &F, log_n: u32, size_inv: &F,
    precomputation_roots: &Vec<F>, domain_size: u64,) -> Result<(), GPUError> 
{
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
                    let mut omegas = vec![F::one(); n];
                    let ratio = (domain_size / n as u64) as usize;
                    for i in 0..n/2 {
                        omegas[i] = precomputation_roots[i * ratio];
                        omegas[i + n/2] =-precomputation_roots[i * ratio];
                    }
                    program.write_from_buffer(&mut omegas_buffer, &omegas)?;
                }

                let mut src_buffer1 = program.create_buffer_pool::<T>(gpu_index,input1.len())?;
                let mut src_buffer2 = program.create_buffer_pool::<T>(gpu_index,input2.len())?;
                let mut src_buffer3 = program.create_buffer_pool::<T>(gpu_index,input3.len())?;
                let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index,n)?;
                let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index,n)?;
                let mut dst_buffer3 = program.create_buffer_pool::<T>(gpu_index,n)?;
                program.write_from_buffer(&mut src_buffer1, &input1)?;
                program.write_from_buffer(&mut src_buffer2, &input2)?;
                program.write_from_buffer(&mut src_buffer3, &input3)?;

                let mut log_p = 0u32;
                while log_p < log_n {
                    let deg = cmp::min(max_deg, log_n - log_p);

                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                    let global_work_size = (n >> deg) * 3;
                    let kernel = program.create_kernel("radix_fft_3_full", global_work_size as usize, local_work_size as usize)?;

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

                    log_p += deg;
                    std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                    std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                    std::mem::swap(&mut src_buffer3, &mut dst_buffer3);
                }

                let (is_create, mut size_inv_buffer) = program.create_size_inv_buffer::<F>(gpu_index, log_n as usize)? ;
                if is_create {
                    let size_invs = vec![*size_inv; 1];
                    program.write_from_buffer(&mut size_inv_buffer, &size_invs)?;
                }

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
            context.program.p.run(closures, (context.program.id, a1, a2, a3))?;
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}

pub fn ars_radix_ifft_6<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>, a2: &mut Vec<T>, a3: &mut Vec<T>, a4: &mut Vec<T>, a5: &mut Vec<T>, a6: &mut Vec<T>,
    omega: &F, log_n: u32, size_inv: &F,
    precomputation_roots: &Vec<F>, domain_size: u64,) -> Result<(), GPUError> 
{
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

                    let (is_create, mut omegas_buffer) = program.create_omega_buffer::<F>(gpu_index, n,OMGA_TYPE_IFFT_COSET)? ;
                    if is_create {
                        let mut omegas = vec![F::one(); n];
                        let ratio = (domain_size / n as u64) as usize;
                        for i in 0..n/2 {
                            omegas[i] = precomputation_roots[i * ratio];
                            omegas[i + n/2] =-precomputation_roots[i * ratio];
                        }
                        program.write_from_buffer(&mut omegas_buffer, &omegas)?;
                    }

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
                        let global_work_size = (n >> deg) * 6;
                        let kernel = program.create_kernel("radix_fft_6_full", global_work_size as usize, local_work_size as usize)?;

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

                    let (is_create, mut size_inv_buffer) = program.create_size_inv_buffer::<F>(gpu_index, log_n as usize)? ;
                    if is_create {
                        let size_invs = vec![*size_inv; 1];
                        program.write_from_buffer(&mut size_inv_buffer, &size_invs)?;
                    }

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
            context.program.p.run(closures, (context.program.id, a1, a2, a3, a4, a5, a6)).unwrap();
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}


pub fn ars_radix_coset_fft<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    input: &mut Vec<T>, omega: &F,
    log_n: u32, g: F,
    precomputation_roots: &Vec<F>, domain_size: u64,) -> Result<(), GPUError> 
{
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };
            let closures = program_closures!(|program, args:(usize, &mut Vec<T>)| -> Result<(), GPUError> {
                let (gpu_index, input) = args;
                let n = 1 << log_n;
                let mut src_buffer = program.create_buffer_pool::<T>(gpu_index, n)?;
                let mut dst_buffer = program.create_buffer_pool::<T>(gpu_index, n)?;
                program.write_from_buffer(&mut src_buffer, &*input)?;

                let (is_create, mut gen_buffer) = program.create_gen_buffer::<F>(gpu_index, OMGA_TYPE_FFT_COSET)? ;
                if is_create {
                    let gen = vec![g; 1];
                    program.write_from_buffer(&mut gen_buffer,&gen)?;
                }

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

                let max_deg = cmp::min(MAX_LOG2_RADIX, log_n);

                let pq_len =  1 << max_deg >> 1;
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
                    let mut omegas = vec![F::one(); n];
                    let ratio = (domain_size / n as u64) as usize;
                    for i in 0..n/2 {
                        omegas[i] = precomputation_roots[i * ratio];
                        omegas[i + n/2] =-precomputation_roots[i * ratio];
                    }
                    program.write_from_buffer(&mut omegas_buffer, &omegas)?;                    
                }

                let mut log_p = 0u32;
                while log_p < log_n { 
                    let deg = cmp::min(max_deg, log_n - log_p);

                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE);
                    let global_work_size = n >> deg;
                    let kernel = program.create_kernel(
                        "radix_fft_full",
                        global_work_size as usize,
                        local_work_size as usize,
                    )?;

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

            context.program.p.run(closures, (context.program.id, input))?;
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}

pub fn ars_radix_coset_fft_2<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>, a2: &mut Vec<T>,
    omega: &F, log_n: u32, g: F,
    precomputation_roots: &Vec<F>, domain_size: u64,) -> Result<(), GPUError> 
{
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

                program.write_from_buffer(&mut src_buffer1, &input1)?;
                program.write_from_buffer(&mut src_buffer2, &input2)?;

                let (is_create, mut gen_buffer) = program.create_gen_buffer::<F>(gpu_index, OMGA_TYPE_FFT_COSET)? ;
                if is_create {
                    let gen = vec![g; 1];
                    let gen_buffer = program.write_from_buffer(&mut gen_buffer,&gen)?;
                }

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
                    let mut omegas = vec![F::one(); n];
                    let ratio = (domain_size / n as u64) as usize;
                    for i in 0..n/2 {
                        omegas[i] = precomputation_roots[i * ratio];
                        omegas[i + n/2] =-precomputation_roots[i * ratio];
                    }
                    program.write_from_buffer(&mut omegas_buffer, &omegas)?;
                }

                let mut log_p = 0u32;
                while log_p < log_n {
                    let deg = cmp::min(max_deg, log_n - log_p);

                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                    let global_work_size = (n >> deg) * 2;
                    let kernel =
                        program.create_kernel("radix_fft_2_full", global_work_size as usize, local_work_size as usize)?;

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

                    log_p += deg;
                    std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                    std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                }

                program.read_into_buffer(&src_buffer1, input1)?;
                program.read_into_buffer(&src_buffer2, input2)?;
                program.recovery_buffer_pool(gpu_index, input1.len(), src_buffer1)?;
                program.recovery_buffer_pool(gpu_index, input2.len(), src_buffer2)?;
                program.recovery_buffer_pool(gpu_index, n, dst_buffer1)?;
                program.recovery_buffer_pool(gpu_index, n, dst_buffer2)?;
                program.recovery_pq_buffer(gpu_index, pq_len, OMGA_TYPE_FFT_COSET, pq_buffer)?;
                program.recovery_omega_buffer(gpu_index, n, OMGA_TYPE_FFT_COSET, omegas_buffer)?;
                program.recovery_gen_buffer(gpu_index, OMGA_TYPE_FFT_COSET, gen_buffer)?;

                Ok(())
            });
            context.program.p.run(closures, (context.program.id, a1, a2))?;
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}

pub fn ars_radix_coset_fft_3<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>, a2: &mut Vec<T>, a3: &mut Vec<T>,
    omega: &F, log_n: u32, g: F,
    precomputation_roots: &Vec<F>, domain_size: u64,) -> Result<(), GPUError> 
{
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };

            let closures = program_closures!(|program, args: (usize, &mut Vec<T>, &mut Vec<T>, &mut Vec<T>)|
            -> Result<(), GPUError> {
                    let (gpu_index, input1, input2, input3) = args;
                    let n = 1 << log_n;

                    let mut src_buffer1 = program.create_buffer_pool::<T>(gpu_index, input1.len())?;
                    let mut src_buffer2 = program.create_buffer_pool::<T>(gpu_index, input2.len())?;
                    let mut src_buffer3 = program.create_buffer_pool::<T>(gpu_index, input3.len())?;
                    let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer3 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    program.write_from_buffer(&mut src_buffer1, &input1)?;
                    program.write_from_buffer(&mut src_buffer2, &input2)?;
                    program.write_from_buffer(&mut src_buffer3, &input3)?;


                    let (is_create, mut gen_buffer) = program.create_gen_buffer::<F>(gpu_index, OMGA_TYPE_FFT_COSET)? ;
                    if is_create {
                        let gen = vec![g; 1];
                        let gen_buffer = program.write_from_buffer(&mut gen_buffer,&gen)?;
                    }

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
                        let mut omegas = vec![F::one(); n];
                        let ratio = (domain_size / n as u64) as usize;
                        for i in 0..n/2 {
                            omegas[i] = precomputation_roots[i * ratio];
                            omegas[i + n/2] =-precomputation_roots[i * ratio];
                        }
                        program.write_from_buffer(&mut omegas_buffer, &omegas)?;
                    }

                    let mut log_p = 0u32;
                    while log_p < log_n {
                        let deg = cmp::min(max_deg, log_n - log_p);

                        let n = 1u32 << log_n;
                        let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                        let global_work_size = (n >> deg) * 3;
                        let kernel =
                            program.create_kernel("radix_fft_3_full", global_work_size as usize, local_work_size as usize)?;

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

                        log_p += deg;
                        std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                        std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                        std::mem::swap(&mut src_buffer3, &mut dst_buffer3);
                    }

                    program.read_into_buffer(&src_buffer1, input1)?;
                    program.read_into_buffer(&src_buffer2, input2)?;
                    program.read_into_buffer(&src_buffer3, input3)?;
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

            context.program.p.run(closures, (context.program.id, a1, a2, a3))?;
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}

pub fn ars_radix_coset_fft_9<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>, a2: &mut Vec<T>, a3: &mut Vec<T>, a4: &mut Vec<T>, a5: &mut Vec<T>, a6: &mut Vec<T>,
    a7: &mut Vec<T>, a8: &mut Vec<T>, a9: &mut Vec<T>,
    omega: &F, log_n: u32, g: F,
    precomputation_roots: &Vec<F>, domain_size: u64,) -> Result<(), GPUError> 
{
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

                    program.write_from_buffer(&mut src_buffer1, &input1)?;
                    program.write_from_buffer(&mut src_buffer2, &input2)?;
                    program.write_from_buffer(&mut src_buffer3, &input3)?;
                    program.write_from_buffer(&mut src_buffer4, &input4)?;
                    program.write_from_buffer(&mut src_buffer5, &input5)?;
                    program.write_from_buffer(&mut src_buffer6, &input6)?;
                    program.write_from_buffer(&mut src_buffer7, &input7)?;
                    program.write_from_buffer(&mut src_buffer8, &input8)?;
                    program.write_from_buffer(&mut src_buffer9, &input9)?;

                    let (is_create, mut gen_buffer) = program.create_gen_buffer::<F>(gpu_index, OMGA_TYPE_FFT_COSET)? ;
                    if is_create {
                        let gen = vec![g; 1];
                        program.write_from_buffer(&mut gen_buffer,&gen)?;
                    }

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
                        let mut omegas = vec![F::one(); n];
                        let ratio = (domain_size / n as u64) as usize;
                        for i in 0..n/2 {
                            omegas[i] = precomputation_roots[i * ratio];
                            omegas[i + n/2] =-precomputation_roots[i * ratio];
                        }
                        program.write_from_buffer(&mut omegas_buffer, &omegas)?;
                    }

                    let mut log_p = 0u32;
                    while log_p < log_n {
                        let deg = cmp::min(max_deg, log_n - log_p);
                        let n = 1u32 << log_n;
                        let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                        // let global_work_size = n >> deg;
                        let global_work_size = (n >> deg) * 9;
                        let kernel = program.create_kernel("radix_fft_9_full", global_work_size as usize, local_work_size as usize)?;
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
                    Ok(())
                });

            context.program.p.run(closures, (context.program.id, a1, a2, a3, a4, a5, a6, a7, a8, a9)).unwrap();
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}


pub fn ars_radix_coset_ifft<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    input: &mut Vec<T>, omega: &F, log_n: u32,
    size_inv: &F, g: F,
    precomputation_roots: &Vec<F>, domain_size: u64,) -> Result<(), GPUError> 
{    
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };

            let closures = program_closures!(|program, args:(usize, &mut Vec<T>) | -> Result<(), GPUError> {
                let (gpu_index, input) = args;
                let n = 1 << log_n;
                let mut src_buffer = program.create_buffer_pool::<T>(gpu_index, n)?;
                let mut dst_buffer = program.create_buffer_pool::<T>(gpu_index, n)?;
                program.write_from_buffer(&mut src_buffer, &*input)?;

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
                    let mut omegas = vec![F::one(); n];
                    let ratio = (domain_size / n as u64) as usize;
                    for i in 0..n/2 {
                        omegas[i] = precomputation_roots[i * ratio];
                        omegas[i + n/2] =-precomputation_roots[i * ratio];
                    }
                    program.write_from_buffer(&mut omegas_buffer, &omegas)?;
                }

                let mut log_p = 0u32;
                while log_p < log_n {
                    let deg = cmp::min(max_deg, log_n - log_p);

                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE);
                    let global_work_size = n >> deg;
                    let kernel = program.create_kernel(
                        "radix_fft_full",
                        global_work_size as usize,
                        local_work_size as usize,
                    )?;

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

                    log_p += deg;
                    std::mem::swap(&mut src_buffer, &mut dst_buffer);
                }

                let (is_create, mut size_inv_buffer) = program.create_size_inv_buffer::<F>(gpu_index, log_n as usize)? ;
                if is_create {
                    let size_invs = vec![*size_inv; 1];
                    let size_inv_buffer = program.write_from_buffer(&mut size_inv_buffer, &size_invs)?;
                }

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

                let (is_create, mut gen_buffer) = program.create_gen_buffer::<F>(gpu_index, OMGA_TYPE_IFFT_COSET)? ;
                if is_create {
                    let gen = vec![g; 1];
                    let gen_buffer = program.write_from_buffer(&mut gen_buffer,&gen)?;
                }

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
            context.program.p.run(closures, (context.program.id, input))?;
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}

pub fn ars_radix_coset_ifft_2<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>, a2: &mut Vec<T>,
    omega: &F, log_n: u32, size_inv: &F, g: F,
    precomputation_roots: &Vec<F>, domain_size: u64,) -> Result<(), GPUError> 
{
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
                    let mut omegas = vec![F::one(); n];
                    let ratio = (domain_size / n as u64) as usize;
                    for i in 0..n/2 {
                        omegas[i] = precomputation_roots[i * ratio];
                        omegas[i + n/2] =-precomputation_roots[i * ratio];
                    }
                    program.write_from_buffer(&mut omegas_buffer, &omegas)?;                    
                }

                let mut log_p = 0u32;
                while log_p < log_n {
                    let deg = cmp::min(max_deg, log_n - log_p);

                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                    let global_work_size = (n >> deg) * 2;
                    let kernel = program.create_kernel("radix_fft_2_full", global_work_size as usize, local_work_size as usize)?;

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

                    log_p += deg;
                    std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                    std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                }

                let (is_create, mut size_inv_buffer) = program.create_size_inv_buffer::<F>(gpu_index, log_n as usize)? ;
                if is_create {
                    let size_invs = vec![*size_inv; 1];
                    let size_inv_buffer = program.write_from_buffer(&mut size_inv_buffer, &size_invs)?;
                }

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

                let (is_create, mut gen_buffer) = program.create_gen_buffer::<F>(gpu_index, OMGA_TYPE_IFFT_COSET)? ;
                if is_create {
                    let gen = vec![g; 1];
                    let gen_buffer = program.write_from_buffer(&mut gen_buffer,&gen)?;
                }

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

            context.program.p.run(closures, (context.program.id, a1, a2))?;
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}

pub fn ars_radix_coset_ifft_3<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>, a2: &mut Vec<T>, a3: &mut Vec<T>,
    omega: &F, log_n: u32, size_inv: &F, g: F,
    precomputation_roots: &Vec<F>, domain_size: u64,) -> Result<(), GPUError> 
{
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };

            let closures = program_closures!(|program, args: (usize, &mut Vec<T>, &mut Vec<T>, &mut Vec<T>)|
             -> Result<(), GPUError> {
                    let (gpu_index, input1, input2, input3) = args;
                    let n = 1 << log_n;

                    let mut src_buffer1 = program.create_buffer_pool::<T>(gpu_index, input1.len())?;
                    let mut src_buffer2 = program.create_buffer_pool::<T>(gpu_index, input2.len())?;
                    let mut src_buffer3 = program.create_buffer_pool::<T>(gpu_index, input3.len())?;
                    let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index, n)?;
                    let mut dst_buffer3 = program.create_buffer_pool::<T>(gpu_index, n)?;

                    program.write_from_buffer(&mut src_buffer1, &input1)?;
                    program.write_from_buffer(&mut src_buffer2, &input2)?;
                    program.write_from_buffer(&mut src_buffer3, &input3)?;

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

                    let (is_create, mut omegas_buffer) = program.create_omega_buffer::<F>(gpu_index, n, OMGA_TYPE_IFFT_COSET)? ;
                    if is_create {
                        let mut omegas = vec![F::one(); n];
                        let ratio = (domain_size / n as u64) as usize;
                        for i in 0..n/2 {
                            omegas[i] = precomputation_roots[i * ratio];
                            omegas[i + n/2] =-precomputation_roots[i * ratio];
                        }
                        program.write_from_buffer(&mut omegas_buffer, &omegas)?;
                    }

                    let mut log_p = 0u32;
                    while log_p < log_n {
                        let deg = cmp::min(max_deg, log_n - log_p);
                        let n = 1u32 << log_n;
                        let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                        let global_work_size = (n >> deg) * 3;
                        let kernel = program.create_kernel("radix_fft_3_full", global_work_size as usize, local_work_size as usize)?;

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

                        log_p += deg;
                        std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                        std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                        std::mem::swap(&mut src_buffer3, &mut dst_buffer3);
                    }

                    let (is_create, mut size_inv_buffer) = program.create_size_inv_buffer::<F>(gpu_index, log_n as usize)? ;
                    if is_create {
                        let size_invs = vec![*size_inv; 1];
                        let size_inv_buffer = program.write_from_buffer(&mut size_inv_buffer, &size_invs)?;
                    }

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

                    let (is_create, mut gen_buffer) = program.create_gen_buffer::<F>(gpu_index, OMGA_TYPE_IFFT_COSET)? ;
                    if is_create {
                        let gen = vec![g; 1];
                        let gen_buffer = program.write_from_buffer(&mut gen_buffer,&gen)?;
                    }

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
                    Ok(())
                });
            context.program.p.run(closures, (context.program.id, a1, a2, a3))?;
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}


pub fn ars_batch_inversion_and_mul_gpu<F: FftField>(v: &mut [F], coeff: &F) {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let closures = program_closures!(|program, args:(usize, &mut [F]) | -> Result<(), GPUError> {
                let (gpu_index, v) = args;                
                let mut v_buffer =program.create_buffer_pool::<F>(gpu_index, v.len())? ;
                program.write_from_buffer(&mut v_buffer, &*v)?;

                let mut coeffs  = vec![F::zero(); 1];
                coeffs[0] = *coeff;
                let mut coeff_buffer =program.create_buffer_pool::<F>(gpu_index, coeffs.len())?;
                program.write_from_buffer(&mut coeff_buffer, &coeffs)?;

                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = 64;
                let num_width = (v.len()+ (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                let kernel = program.create_kernel(
                    "batch_inversion_and_mul",
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                .arg(&v_buffer)
                .arg(&coeff_buffer)
                .arg(&(v.len() as u32))
                .arg(&(num_width as u32))
                .run()?;

                program.read_into_buffer(&v_buffer, v)?;
                program.recovery_buffer_pool(gpu_index, v.len(), v_buffer)?;
                program.recovery_buffer_pool(gpu_index, coeffs.len(), coeff_buffer)?;

                Ok(())
            });
            program.p.run(closures, (program.id, v)).unwrap();
            ars_recycle_fft_program(program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
}

pub fn ars_batch_inversion<F: FftField>(v: &mut [F]) {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let closures = program_closures!(|program, args: (usize, &mut [F])| -> Result<(), GPUError> {
                let (gpu_index, v) = args;                
                let mut v_buffer =program.create_buffer_pool::<F>(gpu_index, v.len())?;
                program.write_from_buffer(&mut v_buffer, &*v)?;

                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = 64;
                let num_width = (v.len()+ (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
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

                program.read_into_buffer(&v_buffer, v)?;
                program.recovery_buffer_pool(gpu_index, v.len(), v_buffer)?;
                Ok(())
            });
            program.p.run(closures, (program.id, v)).unwrap();
            ars_recycle_fft_program(program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
}

pub fn ars_batch_inversion_and_prod_gpu<F: FftField>(v1: &mut [F], v2: &mut [F]) {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let closures = program_closures!(|program, args: (usize, &mut [F], &mut[F]) | -> Result<(), GPUError> {
                let (gpu_index, v1, v2) = args;
                let mut v1_buffer =program.create_buffer_pool::<F>(gpu_index, v1.len())?;
                program.write_from_buffer(&mut v1_buffer, &*v1)?;

                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = 64;
                let num_width = (v1.len()+ (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
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

                let mut v2_buffer =program.create_buffer_pool::<F>(gpu_index, v2.len())?;
                program.write_from_buffer(&mut v2_buffer, &*v2)?;
 
                let kernel1 = program.create_kernel(
                    "product",
                    global_work_size as usize,
                    local_work_size as usize,
                )?;

                kernel1
                .arg(&v1_buffer)
                .arg(&v2_buffer)
                .arg(&(v1.len() as u32))
                .arg(&(num_width as u32))
                .run()?;

                program.read_into_buffer(&v1_buffer, v1)?;
                program.recovery_buffer_pool(gpu_index,v1.len(),v1_buffer)?;
                program.recovery_buffer_pool(gpu_index,v2.len(),v2_buffer)?;
                Ok(())
            });
            program.p.run(closures, (program.id, v1, v2)).unwrap();
            ars_recycle_fft_program(program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
}


pub fn calculate_lhs_gpu<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    result: &mut Vec<T>, a1: &Vec<T>, a2: &Vec<T>, a3: &Vec<T>,
    omega: &F, omega2: &F, log_n: u32, size_inv: &F,
    precomputation_roots: &Vec<F>, domain_size: u64,
    precomputation_roots_2: &Vec<F>, domain_size_2: u64,
) -> Result<(), GPUError> {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };
            let closures = program_closures!(|program, input: (usize, &mut Vec<T>, &Vec<T>, &Vec<T>, &Vec<T>)|-> Result<(), GPUError> {
                let (gpu_index, re, input1, input2, input3) = input;
                let n = 1 << log_n;
                let mut max_log2_radix_n = MAX_LOG2_RADIX;
                let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
                if n <= 80000 {
                    max_log2_radix_n -= 1;
                    max_log2_local_work_size_n -= 1;
                }
                let max_deg = cmp::min(max_log2_radix_n, log_n);

                let pq_len = 1 << max_deg >> 1;
                let (is_create, mut pq_buffer_1) = program.create_pq_buffer::<F>(gpu_index, pq_len, OMGA_TYPE_1)? ;
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
                    program.write_from_buffer(&mut pq_buffer_1, &pq)?;
                }

                let (is_create, mut omegas_buffer_1) = program.create_omega_buffer::<F>(gpu_index, n,OMGA_TYPE_1)? ;
                if is_create {
                    let mut omegas = vec![F::one(); n];
                    let ratio = (domain_size / n as u64) as usize;
                     for i in 0..n/2 {
                        omegas[i] = precomputation_roots[i * ratio];
                        omegas[i + n/2] =-precomputation_roots[i * ratio];
                     }
                    program.write_from_buffer(&mut omegas_buffer_1, &omegas)?;                    
                }
                let mut result_buffer = program.create_buffer_pool::<T>(gpu_index,re.len())?;
                let mut src_buffer1 = program.create_buffer_pool::<T>(gpu_index,input1.len())?;
                let mut src_buffer2 = program.create_buffer_pool::<T>(gpu_index,input2.len())?;
                let mut src_buffer3 = program.create_buffer_pool::<T>(gpu_index,input3.len())?;
                let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index,n)?;
                let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index,n)?;
                let mut dst_buffer3 = program.create_buffer_pool::<T>(gpu_index,n)?;

                program.write_from_buffer(&mut src_buffer1, &input1)?;
                program.write_from_buffer(&mut src_buffer2, &input2)?;
                program.write_from_buffer(&mut src_buffer3, &input3)?;
                program.write_from_buffer(&mut result_buffer, &re)?;

                let mut log_p = 0u32;
                while log_p < log_n {
                    let deg = cmp::min(max_deg, log_n - log_p);
                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                    let global_work_size = (n >> deg)*3;

                    let kernel = program.create_kernel("radix_fft_3_full", global_work_size as usize, local_work_size as usize)?;
                    kernel
                        .arg(&src_buffer1)
                        .arg(&src_buffer2)
                        .arg(&src_buffer3)
                        .arg(&dst_buffer1)
                        .arg(&dst_buffer2)
                        .arg(&dst_buffer3)
                        .arg(&pq_buffer_1)
                        .arg(&omegas_buffer_1)
                        .arg(&n)
                        .arg(&log_p)
                        .arg(&deg)
                        .arg(&max_deg)
                        .run()?;
                    log_p += deg;

                    std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                    std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                    std::mem::swap(&mut src_buffer3, &mut dst_buffer3);
                }

                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = n / LOCAL_WORK_SIZE;
                let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);

                let kernel = program.create_kernel(
                    "calculate_lhs",
                    global_work_size as usize,
                    local_work_size as usize,
                )?;

                kernel
                .arg(&result_buffer)
                .arg(&src_buffer1)
                .arg(&src_buffer2)
                .arg(&src_buffer3)
                .arg(&(n as u32))
                .arg(&(num_width as u32))
                .run()?;

                let pq_len = 1 << max_deg >> 1;
                let (is_create, mut pq_buffer_2) = program.create_pq_buffer::<F>(gpu_index, pq_len, OMGA_TYPE_2)? ;
                if is_create {
                    let mut pq = vec![F::zero(); pq_len];
                    let twiddle = omega2.pow([(n >> max_deg) as u64]);
                    pq[0] =F::one();
                    if max_deg > 1 {
                        pq[1] = twiddle;
                        for i in 2..pq_len {
                            pq[i] = pq[i - 1];
                            pq[i] *= &twiddle
                        }
                    }
                    program.write_from_buffer(&mut pq_buffer_2, &pq)?;
                }

                let (is_create, mut omegas_buffer_2) = program.create_omega_buffer::<F>(gpu_index, n,OMGA_TYPE_2)? ;
                if is_create {
                    let mut omegas = vec![F::one(); n];
                    let ratio = (domain_size_2 / n as u64) as usize;
                    for i in 0..n/2 {
                        omegas[i] = precomputation_roots_2[i * ratio];
                        omegas[i + n/2] =-precomputation_roots_2[i * ratio];
                    }
                    program.write_from_buffer(&mut omegas_buffer_2, &omegas)?;
                }

                log_p = 0;
                while log_p < log_n {
                    let deg = cmp::min(max_deg, log_n - log_p);
                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE);
                    let global_work_size = n >> deg;

                    let kernel = program.create_kernel(
                        "radix_fft_full",
                        global_work_size as usize,
                        local_work_size as usize,
                    )?;
                    kernel
                        .arg(&result_buffer)
                        .arg(&dst_buffer1)
                        .arg(&pq_buffer_2)
                        .arg(&omegas_buffer_2)
                        .arg(&n)
                        .arg(&log_p)
                        .arg(&deg)
                        .arg(&max_deg)
                        .run()?;
                    log_p += deg;
                    std::mem::swap(&mut result_buffer, &mut dst_buffer1);
                }

                let (is_create, mut size_inv_buffer) = program.create_size_inv_buffer::<F>(gpu_index, log_n as usize)? ;
                if is_create {
                    let size_invs = vec![*size_inv; 1];
                    let size_inv_buffer = program.write_from_buffer(&mut size_inv_buffer, &size_invs)?;
                }

                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = n / LOCAL_WORK_SIZE;
                let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);

                let kernel = program.create_kernel(
                    "compute_mul_size_inv",
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                    .arg(&result_buffer)
                    .arg(&dst_buffer1)
                    .arg(&size_inv_buffer)
                    .arg(&(n as u32))
                    .arg(&(num_width as u32))
                    .run()?;

                program.read_into_buffer(&dst_buffer1, re)?;

                program.recovery_buffer_pool(gpu_index,re.len(),result_buffer)?;
                program.recovery_buffer_pool(gpu_index,input1.len(),src_buffer1)?;
                program.recovery_buffer_pool(gpu_index,input2.len(),src_buffer2)?;
                program.recovery_buffer_pool(gpu_index,input3.len(),src_buffer3)?;
                program.recovery_buffer_pool(gpu_index,n,dst_buffer1)?;
                program.recovery_buffer_pool(gpu_index,n,dst_buffer2)?;
                program.recovery_buffer_pool(gpu_index,n,dst_buffer3)?;
                program.recovery_pq_buffer(gpu_index, pq_len, OMGA_TYPE_1, pq_buffer_1)?;
                program.recovery_omega_buffer(gpu_index, n, OMGA_TYPE_1, omegas_buffer_1)?;
                program.recovery_pq_buffer(gpu_index, pq_len, OMGA_TYPE_2, pq_buffer_2)?;
                program.recovery_omega_buffer(gpu_index, n, OMGA_TYPE_2, omegas_buffer_2)?;
                program.recovery_size_inv_buffer(gpu_index, log_n as usize, size_inv_buffer)?;
                Ok(())
            });
            context.program.p.run(closures, (context.program.id, result, a1, a2, a3)).unwrap();
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}

pub fn calculate_summed_z_m_gpu<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    z_a: &mut Vec<T>, z_b: &mut Vec<T>,
    omega: &F, omega2: &F, log_n: u32, size_inv: &F,
    precomputation_roots: &Vec<F>, domain_size: u64,
    precomputation_roots_2: &Vec<F>, domain_size_2: u64,
) -> Result<(), GPUError> {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let context = CudaContext { program };
            let closures = program_closures!(|program, args: (usize, &mut Vec<T>, &mut Vec<T>)| -> Result<(), GPUError> {
                let (gpu_index, input_a, input_b) = args;
                let n = 1 << log_n;

                let mut src_buffer_a = program.create_buffer_pool::<T>(gpu_index, n)?;
                program.write_from_buffer(&mut src_buffer_a, &input_a)?;

                let mut src_buffer_b = program.create_buffer_pool::<T>(gpu_index, n)?;
                program.write_from_buffer(&mut src_buffer_b, &input_b)?;

                let mut max_log2_radix_n = MAX_LOG2_RADIX;
                let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
                if n <= 80000 {
                    max_log2_radix_n -= 1;
                    max_log2_local_work_size_n -= 1;
                }
                let max_deg = cmp::min(max_log2_radix_n, log_n);

                let pq_len = 1 << max_deg >> 1 ;
                let (is_create, mut pq_buffer_1) = program.create_pq_buffer::<F>(gpu_index, pq_len, OMGA_TYPE_1)? ;
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
                    program.write_from_buffer(&mut pq_buffer_1, &pq)?;
                }

                let (is_create, mut omegas_buffer_1) = program.create_omega_buffer::<F>(gpu_index, n,OMGA_TYPE_1)? ;
                if is_create {
                    let mut omegas = vec![F::one(); n];
                    let ratio = (domain_size / n as u64) as usize;
                    for i in 0..n/2 {
                        omegas[i] = precomputation_roots[i * ratio];
                        omegas[i + n/2] = -precomputation_roots[i * ratio];
                    }
                    program.write_from_buffer(&mut omegas_buffer_1, &omegas)?;
                }

                let mut dst_buffer1 = program.create_buffer_pool::<T>(gpu_index,n)?;
                let mut dst_buffer2 = program.create_buffer_pool::<T>(gpu_index,n)?;

                let mut log_p = 0u32;
                while log_p < log_n {
                    let deg = cmp::min(max_deg, log_n - log_p);

                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                    let global_work_size = (n >> deg)*2;

                    let kernel = program.create_kernel("radix_fft_2_full", global_work_size as usize, local_work_size as usize)?;
                    kernel
                        .arg(&src_buffer_a)
                        .arg(&src_buffer_b)
                        .arg(&dst_buffer1)
                        .arg(&dst_buffer2)
                        .arg(&pq_buffer_1)
                        .arg(&omegas_buffer_1)
                        .arg(&n)
                        .arg(&log_p)
                        .arg(&deg)
                        .arg(&max_deg)
                        .run()?;

                    log_p += deg;
                    std::mem::swap(&mut src_buffer_a, &mut dst_buffer1);
                    std::mem::swap(&mut src_buffer_b, &mut dst_buffer2);
                }

                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = n / LOCAL_WORK_SIZE;
                let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);

                let kernel = program.create_kernel(
                    "product",
                    global_work_size as usize,
                    local_work_size as usize,
                )?;

                kernel
                .arg(&src_buffer_a)
                .arg(&src_buffer_b)
                .arg(&(n as u32))
                .arg(&(num_width as u32))
                .run()?;

                let pq_len = 1 << max_deg >> 1;
                let (is_create, mut pq_buffer_2) = program.create_pq_buffer::<F>(gpu_index, pq_len, OMGA_TYPE_2)? ;
                if is_create {
                    let mut pq = vec![F::zero(); pq_len];
                    let twiddle = omega2.pow([(n >> max_deg) as u64]);
                    pq[0] =F::one();
                    if max_deg > 1 {
                        pq[1] = twiddle;
                        for i in 2..pq_len {
                            pq[i] = pq[i - 1];
                            pq[i] *= &twiddle
                        }
                    }
                    program.write_from_buffer(&mut pq_buffer_2, &pq)?;
                }

                let (is_create, mut omegas_buffer_2) = program.create_omega_buffer::<F>(gpu_index, n,OMGA_TYPE_2)? ;
                if is_create {
                    let mut omegas = vec![F::one(); n];
                    let ratio = (domain_size_2 / n as u64) as usize;
                    for i in 0..n/2 {
                       omegas[i] = precomputation_roots_2[i * ratio];
                       omegas[i + n/2] =-precomputation_roots_2[i * ratio];
                    }
                    program.write_from_buffer(&mut omegas_buffer_2, &omegas)?;
                }

                log_p = 0;
                while log_p < log_n {
                    let deg = cmp::min(max_deg, log_n - log_p);
                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE);
                    let global_work_size = n >> deg;

                    let kernel = program.create_kernel(
                        "radix_fft_full",
                        global_work_size as usize,
                        local_work_size as usize,
                    )?;
                    kernel
                        .arg(&src_buffer_a)
                        .arg(&dst_buffer1)
                        .arg(&pq_buffer_2)
                        .arg(&omegas_buffer_2)
                        .arg(&n)
                        .arg(&log_p)
                        .arg(&deg)
                        .arg(&max_deg)
                        .run()?;
                    log_p += deg;
                    std::mem::swap(&mut src_buffer_a, &mut dst_buffer1);
                }

                let (is_create, mut size_inv_buffer) = program.create_size_inv_buffer::<F>(gpu_index, log_n as usize)? ;
                if is_create {
                    let size_invs = vec![*size_inv; 1];
                    let size_inv_buffer = program.write_from_buffer(&mut size_inv_buffer, &size_invs)?;
                }

                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = n / LOCAL_WORK_SIZE;
                let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);

                let kernel = program.create_kernel(
                    "compute_mul_size_inv",
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                    .arg(&src_buffer_a)
                    .arg(&dst_buffer1)
                    .arg(&size_inv_buffer)
                    .arg(&(n as u32))
                    .arg(&(num_width as u32))
                    .run()?;

                program.read_into_buffer(&dst_buffer1, input_a)?;

                program.recovery_buffer_pool(gpu_index, input_a.len(), src_buffer_a)?;
                program.recovery_buffer_pool(gpu_index, input_b.len(), src_buffer_b)?;
                program.recovery_buffer_pool(gpu_index, n, dst_buffer1)?;
                program.recovery_buffer_pool(gpu_index, n, dst_buffer2)?;
                program.recovery_pq_buffer(gpu_index, pq_len, OMGA_TYPE_1, pq_buffer_1)?;
                program.recovery_omega_buffer(gpu_index, n, OMGA_TYPE_1, omegas_buffer_1)?;
                program.recovery_pq_buffer(gpu_index, pq_len, OMGA_TYPE_2, pq_buffer_2)?;
                program.recovery_omega_buffer(gpu_index, n, OMGA_TYPE_2, omegas_buffer_2)?;
                program.recovery_size_inv_buffer(gpu_index, log_n as usize, size_inv_buffer)?;
                Ok(())
            });
            context.program.p.run(closures, (context.program.id, z_a, z_b)).unwrap();
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}


pub fn third_compute_b_f_h_poly<F: FftField, T: DomainCoeff<F> + std::fmt::Debug>(
    a1: &mut Vec<T>, a2: &mut Vec<T>, a3: &mut Vec<T>, a4: &mut Vec<T>, a5: &mut Vec<T>, a6: &mut Vec<T>,
    a7: &mut Vec<T>, a8: &mut Vec<T>, a9: &mut Vec<T>,
    omega: &F, omega2: &F, log_n: u32, size_inv: &F, g: F, g2: F, f_i: F,
    precomputation_roots: &Vec<F>, domain_size: u64,
    precomputation_roots_2: &Vec<F>, domain_size_2: u64,
) -> Result<(), GPUError> {
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

                program.write_from_buffer(&mut src_buffer1, &input1)?;
                program.write_from_buffer(&mut src_buffer2, &input2)?;
                program.write_from_buffer(&mut src_buffer3, &input3)?;
                program.write_from_buffer(&mut src_buffer4, &input4)?;
                program.write_from_buffer(&mut src_buffer5, &input5)?;
                program.write_from_buffer(&mut src_buffer6, &input6)?;
                program.write_from_buffer(&mut src_buffer7, &input7)?;
                program.write_from_buffer(&mut src_buffer8, &input8)?;
                program.write_from_buffer(&mut src_buffer9, &input9)?;

                let mut max_log2_radix_n = MAX_LOG2_RADIX;
                let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
                if n <= 80000 {
                    max_log2_radix_n -= 1;
                    max_log2_local_work_size_n -= 1;
                }
                let max_deg = cmp::min(max_log2_radix_n, log_n);

                let pq_len = 1 << max_deg >> 1;
                let (is_create, mut pq_buffer_2) = program.create_pq_buffer::<F>(gpu_index, pq_len, OMGA_TYPE_2)? ;
                if is_create {
                    let mut pq = vec![F::zero(); pq_len];
                    let twiddle = omega2.pow([(n >> max_deg) as u64]);
                    pq[0] = F::one();
                    if max_deg > 1 {
                        pq[1] = twiddle;
                        for i in 2..pq_len {
                            pq[i] = pq[i - 1];
                            pq[i] *= &twiddle
                        }
                    }
                    program.write_from_buffer(&mut pq_buffer_2, &pq)?;
                }

                let (is_create, mut omegas_buffer_2) = program.create_omega_buffer::<F>(gpu_index, n, OMGA_TYPE_2)? ;
                if is_create {
                    let mut omegas = vec![F::one(); n];
                    let ratio = (domain_size / n as u64) as usize;
                    for i in 0..n/2 {
                       omegas[i] = precomputation_roots[i * ratio];
                       omegas[i + n/2] =-precomputation_roots[i * ratio];
                    }
                    program.write_from_buffer(&mut omegas_buffer_2, &omegas)?;
                }

                let mut log_p = 0u32;
                while log_p < log_n {
                    let deg = cmp::min(max_deg, log_n - log_p);

                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                    let global_work_size = (n >> deg)*6;

                    let kernel = program.create_kernel("radix_fft_6_full", global_work_size as usize, local_work_size as usize)?;
                    kernel
                        .arg(&src_buffer1)
                        .arg(&src_buffer2)
                        .arg(&src_buffer4)
                        .arg(&src_buffer5)
                        .arg(&src_buffer7)
                        .arg(&src_buffer8)
                        .arg(&dst_buffer1)
                        .arg(&dst_buffer2)
                        .arg(&dst_buffer4)
                        .arg(&dst_buffer5)
                        .arg(&dst_buffer7)
                        .arg(&dst_buffer8)
                        .arg(&pq_buffer_2)
                        .arg(&omegas_buffer_2)
                        .arg(&n)
                        .arg(&log_p)
                        .arg(&deg)
                        .arg(&max_deg)
                        .run()?;

                    log_p += deg;
                    std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                    std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                    std::mem::swap(&mut src_buffer4, &mut dst_buffer4);
                    std::mem::swap(&mut src_buffer5, &mut dst_buffer5);
                    std::mem::swap(&mut src_buffer7, &mut dst_buffer7);
                    std::mem::swap(&mut src_buffer8, &mut dst_buffer8);
                }

                let (is_create, mut size_inv_buffer) = program.create_size_inv_buffer::<F>(gpu_index, log_n as usize)? ;
                if is_create {
                    let size_invs = vec![*size_inv; 1];
                    let size_inv_buffer = program.write_from_buffer(&mut size_inv_buffer, &size_invs)?;
                }

                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = n / LOCAL_WORK_SIZE;
                let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);

                let kernel = program.create_kernel("compute_mul_size_inv_6", global_work_size as usize, local_work_size as usize)?;
                kernel
                        .arg(&src_buffer1)
                        .arg(&src_buffer2)
                        .arg(&src_buffer4)
                        .arg(&src_buffer5)
                        .arg(&src_buffer7)
                        .arg(&src_buffer8)
                        .arg(&dst_buffer1)
                        .arg(&dst_buffer2)
                        .arg(&dst_buffer4)
                        .arg(&dst_buffer5)
                        .arg(&dst_buffer7)
                        .arg(&dst_buffer8)
                        .arg(&size_inv_buffer)
                        .arg(&(n as u32))
                        .arg(&(num_width as u32))
                        .run()?;

                program.read_into_buffer(&dst_buffer2, input2)?;
                program.read_into_buffer(&dst_buffer5, input5)?;
                program.read_into_buffer(&dst_buffer8, input8)?;

                std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                std::mem::swap(&mut src_buffer2, &mut dst_buffer2);
                std::mem::swap(&mut src_buffer4, &mut dst_buffer4);
                std::mem::swap(&mut src_buffer5, &mut dst_buffer5);
                std::mem::swap(&mut src_buffer7, &mut dst_buffer7);
                std::mem::swap(&mut src_buffer8, &mut dst_buffer8);


                let (is_create, mut gen_buffer) = program.create_gen_buffer::<F>(gpu_index, OMGA_TYPE_1)? ;
                if is_create {
                    let gen = vec![g; 1];
                    let gen_buffer = program.write_from_buffer(&mut gen_buffer,&gen)?;
                }

                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = n / LOCAL_WORK_SIZE;
                let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);

                let kernel = program.create_kernel("distribute_powers_9", global_work_size as usize, local_work_size as usize)?;
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

                let pq_len = 1 << max_deg >> 1;
                let (is_create, mut pq_buffer_1) = program.create_pq_buffer::<F>(gpu_index, pq_len, OMGA_TYPE_1)? ;
                if is_create {
                    let mut pq = vec![F::zero(); pq_len];
                    let twiddle = omega.pow([(n >> max_deg) as u64]);
                    pq[0] =F::one();
                    if max_deg > 1 {
                        pq[1] = twiddle;
                        for i in 2..pq_len {
                            pq[i] = pq[i - 1];
                            pq[i] *= &twiddle
                        }
                    }
                    program.write_from_buffer(&mut pq_buffer_1, &pq)?;
                }

                let (is_create, mut omegas_buffer_1) = program.create_omega_buffer::<F>(gpu_index, n,OMGA_TYPE_1)? ;
                if is_create {
                    let mut omegas = vec![F::one(); n];
                    let ratio = (domain_size_2 / n as u64) as usize;
                    for i in 0..n/2 {
                       omegas[i] = precomputation_roots_2[i * ratio];
                       omegas[i + n/2] =-precomputation_roots_2[i * ratio];
                    }
                    program.write_from_buffer(&mut omegas_buffer_1, &omegas)?;
                }

                let mut log_p = 0u32;
                while log_p < log_n {
                    let deg = cmp::min(max_deg, log_n - log_p);

                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                    let global_work_size =( n >> deg)*9;
                    let kernel = program.create_kernel("radix_fft_9_full", global_work_size as usize, local_work_size as usize)?;
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
                        .arg(&pq_buffer_1)
                        .arg(&omegas_buffer_1)
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

                let f_i_buffer = program.create_buffer_from_slice(&[f_i])?;

                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = n / LOCAL_WORK_SIZE;
                let num_width = (input1.len() + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);

                let kernel = program.create_kernel(
                    "third_compute_h_poly",
                    global_work_size as usize,
                    local_work_size as usize,
                )?;

                kernel
                .arg(&src_buffer1)
                .arg(&src_buffer4)
                .arg(&src_buffer7)
                .arg(&src_buffer2)
                .arg(&src_buffer5)
                .arg(&src_buffer8)
                .arg(&src_buffer3)
                .arg(&src_buffer6)
                .arg(&src_buffer9)
                .arg(&f_i_buffer)
                .arg(&(input1.len() as u32))
                .arg(&(num_width as u32))
                .run()?;

                let mut max_log2_radix_n = MAX_LOG2_RADIX;
                let mut max_log2_local_work_size_n = MAX_LOG2_LOCAL_WORK_SIZE;
                if n <= 80000 {
                    max_log2_radix_n -= 1;
                    max_log2_local_work_size_n -= 1;
                }
                let max_deg = cmp::min(max_log2_radix_n, log_n);                

                let mut log_p = 0u32;
                while log_p < log_n {
                    let deg = cmp::min(max_deg, log_n - log_p);

                    let n = 1u32 << log_n;
                    let local_work_size = 1 << cmp::min(deg - 1, max_log2_local_work_size_n);
                    let global_work_size = (n >> deg)*3;

                    let kernel = program.create_kernel("radix_fft_3_full", global_work_size as usize, local_work_size as usize)?;
                    kernel
                        .arg(&src_buffer1)
                        .arg(&src_buffer4)
                        .arg(&src_buffer7)
                        .arg(&dst_buffer1)
                        .arg(&dst_buffer2)
                        .arg(&dst_buffer3)
                        .arg(&pq_buffer_2)
                        .arg(&omegas_buffer_2)
                        .arg(&n)
                        .arg(&log_p)
                        .arg(&deg)
                        .arg(&max_deg)
                        .run()?;

                    log_p += deg;
                    std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                    std::mem::swap(&mut src_buffer4, &mut dst_buffer2);
                    std::mem::swap(&mut src_buffer7, &mut dst_buffer3);
                }

                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = n / LOCAL_WORK_SIZE;
                let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);

                let kernel = program.create_kernel("compute_mul_size_inv_3", global_work_size as usize, local_work_size as usize)?;
                kernel
                    .arg(&src_buffer1)
                    .arg(&src_buffer4)
                    .arg(&src_buffer7)
                    .arg(&dst_buffer1)
                    .arg(&dst_buffer2)
                    .arg(&dst_buffer3)
                    .arg(&size_inv_buffer)
                    .arg(&(n as u32))
                    .arg(&(num_width as u32))
                    .run()?;

                std::mem::swap(&mut src_buffer1, &mut dst_buffer1);
                std::mem::swap(&mut src_buffer4, &mut dst_buffer2);
                std::mem::swap(&mut src_buffer7, &mut dst_buffer3);

                let (is_create, mut gen_buffer_2) = program.create_gen_buffer::<F>(gpu_index, OMGA_TYPE_2)? ;
                if is_create {
                    let gen = vec![g2; 1];
                    let gen_buffer_2 = program.write_from_buffer(&mut gen_buffer_2, &gen)?;
                }

                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = n / LOCAL_WORK_SIZE;
                let num_width = (n + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);

                let kernel = program.create_kernel("distribute_powers_3", global_work_size as usize, local_work_size as usize)?;
                kernel
                    .arg(&src_buffer1)
                    .arg(&src_buffer4)
                    .arg(&src_buffer7)
                    .arg(&dst_buffer1)
                    .arg(&dst_buffer2)
                    .arg(&dst_buffer3)
                    .arg(&gen_buffer_2)
                    .arg(&(n as u32))
                    .arg(&(num_width as u32))
                    .run()?;

                program.read_into_buffer(&dst_buffer1, input1)?;
                program.read_into_buffer(&dst_buffer2, input4)?;
                program.read_into_buffer(&dst_buffer3, input7)?;

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

                program.recovery_pq_buffer(gpu_index, pq_len, OMGA_TYPE_1, pq_buffer_1)?;
                program.recovery_omega_buffer(gpu_index, n, OMGA_TYPE_1, omegas_buffer_1)?;
                program.recovery_pq_buffer(gpu_index, pq_len, OMGA_TYPE_2, pq_buffer_2)?;
                program.recovery_omega_buffer(gpu_index, n, OMGA_TYPE_2, omegas_buffer_2)?;
                program.recovery_size_inv_buffer(gpu_index, log_n as usize, size_inv_buffer)?;
                program.recovery_gen_buffer(gpu_index, OMGA_TYPE_1, gen_buffer)?;
                program.recovery_gen_buffer(gpu_index, OMGA_TYPE_2, gen_buffer_2)?;
                Ok(())
            });
            context.program.p.run(closures, (context.program.id, a1, a2, a3, a4, a5, a6, a7, a8, a9)).unwrap();
            ars_recycle_fft_program(context.program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
    Ok(())
}

pub fn domain_elements_gpu<F: FftField>(domain: &EvaluationDomain<F>, v: &mut [F]) {
    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let closures = program_closures!(|program,v:&mut [F]| -> Result<(), GPUError> {
                let v_buffer = unsafe { program.create_buffer::<F>( domain.size())? };

                let mut gens  = vec![F::zero(); 1];
                gens[0] = domain.group_gen;
                let gens_buffer = program.create_buffer_from_slice(&gens)?;

                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = 64;
                let num_width = (v.len()+ (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);
                let kernel = program.create_kernel(
                    "domain_elements",
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                .arg(&v_buffer)
                .arg(&gens_buffer)
                .arg(&(domain.size() as u32))
                .arg(&(num_width as u32))
                .run()?;

                program.read_into_buffer(&v_buffer, v)?;
                Ok(())
            });

            program.p.run(closures, v).unwrap();
            ars_recycle_fft_program(program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
}

pub fn compute_deri_vanishing_poly<F: FftField>(domain_fixed: &EvaluationDomain<F>,
                                                x: F,
                                                domain: &EvaluationDomain<F>,
                                                v: &mut [F], ) {
    #[cfg(not(feature = "parallel"))]
    use itertools::Itertools;
    #[cfg(feature = "parallel")]
    use rayon::prelude::*;

    match ars_fetch_fft_program(ars_load_cuda_program) {
        Ok(program) => {
            let closures = program_closures!(|program,v:&mut [F]| -> Result<(), GPUError> {
                let vanish_x = domain_fixed.evaluate_vanishing_polynomial(x);
                let mut v_buffer = unsafe { program.create_buffer::<F>( domain.size())? };
                let mut result_buffer = unsafe { program.create_buffer::<F>(  domain.size())? };
                let mut gens  = vec![F::zero(); 1];
                gens[0] = domain.group_gen;
                let gens_buffer = program.create_buffer_from_slice(&gens)?;

                let local_work_size = LOCAL_WORK_SIZE;
                let global_work_size = 64;
                let num_width = (v.len()+ (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);

                let kernel = program.create_kernel(
                    "domain_elements",
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                .arg(&v_buffer)
                .arg(&gens_buffer)
                .arg(&(domain.size() as u32))
                .arg(&(num_width as u32))
                .run()?;

                program.read_into_buffer(&v_buffer, v)?;

                let denoms = cfg_iter!(v).map(|e| x - e).collect::<Vec<_>>();

                program.write_from_buffer(&mut v_buffer, &*denoms)?;


                if domain.size() <= domain_fixed.size() {
                    let mut vanish_x_s  = vec![F::zero(); 1];
                    vanish_x_s[0] = vanish_x;
                    let vanish_x_buffer = program.create_buffer_from_slice(&vanish_x_s)?;
                    let local_work_size = LOCAL_WORK_SIZE;
                    let global_work_size = 64;
                    let num_width = (domain.size() + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);

                    let kernel = program.create_kernel(
                        "batch_inversion_and_mul",
                        global_work_size as usize,
                        local_work_size as usize,
                    )?;
                    kernel
                    .arg(&v_buffer)
                    .arg(&vanish_x_buffer)
                    .arg(&(domain.size as u32))
                    .arg(&(num_width as u32))
                    .run()?;

                    program.read_into_buffer(&v_buffer, v)?;
                } else {
                    let local_work_size = LOCAL_WORK_SIZE;
                    let global_work_size = 64;
                    let num_width = (domain.size() + (local_work_size * global_work_size) - 1) / (local_work_size * global_work_size);

                    let kernel1= program.create_kernel(
                        "batch_inversion",
                        global_work_size as usize,
                        local_work_size as usize,
                    )?;
                    kernel1
                    .arg(&v_buffer)
                    .arg(&(domain.size as u32))
                    .arg(&(num_width as u32))
                    .run()?;

                    let mut numerators_buffer = unsafe { program.create_buffer::<F>( domain.size())? };
                    let numerators = vec![vanish_x; domain.size()];
                    program.write_from_buffer(&mut numerators_buffer, &*numerators)?;
                    program.write_from_buffer(&mut result_buffer, &*v)?;

                    let kernel2 = program.create_kernel(
                        "numerators",
                        global_work_size as usize,
                        local_work_size as usize,
                    )?;
                    kernel2
                    .arg(&numerators_buffer)
                    .arg(&result_buffer)
                    .arg(&(domain.size as u32))
                    .arg(&(num_width as u32))
                    .arg(&(domain_fixed.size  as u32))
                    .run()?;

                    let kernel3 = program.create_kernel(
                        "product",
                        global_work_size as usize,
                        local_work_size as usize,
                    )?;

                    kernel3
                    .arg(&v_buffer)
                    .arg(&numerators_buffer)
                    .arg(&result_buffer)
                    .arg(&(domain.size as u32))
                    .arg(&(num_width as u32))
                    .run()?;

                    program.read_into_buffer(&result_buffer, v)?;
                }
                Ok(())
            });

            program.p.run(closures, v).unwrap();
            ars_recycle_fft_program(program);
        }
        Err(err) => {
            gpu_msg_dispatch_notify("GPUError: Error loading cuda program".to_string());
            eprintln!("Error loading cuda program: {:?}", err);
        }
    }
}
