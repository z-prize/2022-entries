#!/bin/bash -x

nvcc ./asm_fft_cuda.cu -ptx -rdc=true -o ./asm_fft_cuda.release.ptx
nvcc ./blst_ops.cu -ptx -rdc=true -o ./blst_ops.release.ptx
nvcc ./fft.cu -ptx -rdc=true -o ./fft.release.ptx
nvcc ./asm_fft_cuda.cu ./blst_ops.cu ./fft.cu -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -dlink -fatbin -o ./fft.fatbin

nvcc ./asm_fft_cuda.cu --device-debug -ptx -rdc=true -o ./asm_fft_cuda.debug.ptx
nvcc ./blst_ops.cu --device-debug -ptx -rdc=true -o ./blst_ops.debug.ptx
nvcc ./fft.cu --device-debug -ptx -rdc=true -o ./fft.debug.ptx
nvcc ./asm_fft_cuda.cu ./blst_ops.cu ./fft.cu -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 --device-debug -dlink -fatbin -o ./fft_debug.fatbin
