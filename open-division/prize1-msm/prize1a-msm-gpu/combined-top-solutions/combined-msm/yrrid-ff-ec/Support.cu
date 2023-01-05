/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Written by Niall Emmart.

***/

__device__ __host__ void* byteOffset(void* address, uint32_t byteOffset) {
  uint8_t* ptr8=(uint8_t*)address;
  
  return (void*)(ptr8+byteOffset);
}

__device__ __forceinline__ void* byteOffset(void* address, uint32_t index, uint32_t bytes) {
  uint64_t ptr8=madwide(index, bytes, (uint64_t)address);
  
  return (void*)ptr8;
}

__device__ __forceinline__ void load_shared_u4(uint32_t& x, uint32_t& y, uint32_t& z, uint32_t& w, uint32_t sAddr) {
  uint4 load;
  
  asm volatile ("ld.shared.v4.u32 {%0,%1,%2,%3},[%4];" : "=r"(load.x), "=r"(load.y), "=r"(load.z), "=r"(load.w) : "r"(sAddr));
  x=load.x;
  y=load.y;
  z=load.z;
  w=load.w;
}

__device__ __forceinline__ uint32_t load_shared_byte(uint32_t sAddr) {
  uint32_t r;
  
  asm volatile ("ld.shared.b8 %0,[%1];" : "=r"(r) : "r"(sAddr));
  return r & 0xFF;
}

__device__ __forceinline__ void store_shared_byte(uint32_t sAddr, uint32_t value) {
  asm volatile ("st.shared.b8 [%0],%1;" : : "r"(sAddr), "r"(value));
}

__device__ __forceinline__ uint32_t warpPrefixSum(uint32_t value, uint32_t width=32) {
  uint32_t thread=threadIdx.x & width-1;
  uint32_t total=value, temp;
 
  total=value;
  
  if(width>=2) {
    temp=__shfl_up_sync(0xFFFFFFFF, value, 1);
    total=(thread>=1) ? value+temp : value;
  }
  
  if(width>=4) {
    temp=__shfl_up_sync(0xFFFFFFFF, total, 2);
    total=(thread>=2) ? total+temp : total;
  }
  
  if(width>=8) {
    temp=__shfl_up_sync(0xFFFFFFFF, total, 4);
    total=(thread>=4) ? total+temp : total;
  }
  
  if(width>=16) {
    temp=__shfl_up_sync(0xFFFFFFFF, total, 8);
    total=(thread>=8) ? total+temp : total;
  }
  
  if(width>=32) {
    temp=__shfl_up_sync(0xFFFFFFFF, total, 16);
    total=(thread>=16) ? total+temp : total;
  }
  return total;
}

__device__ __forceinline__ uint32_t multiwarpPrefixSum(uint32_t* shared, uint32_t value, uint32_t warps) {
  int32_t  warp=threadIdx.x>>5, warpThread=threadIdx.x & 0x1F;
  uint32_t localTotal;
  
  if(warp<warps) {
    localTotal=warpPrefixSum(value);
    if(warpThread==31)
      shared[warp]=localTotal;
      
    asm volatile ("bar.sync  1,%0;" : : "r"(warps*32));
    
    #pragma unroll
    for(int32_t i=0;i<warp;i++)
      localTotal+=shared[i];
  }
  return localTotal;  
}

__device__ __forceinline__ uint32_t multiwarpPrefixSum(uint32_t sAddr, uint32_t value, uint32_t warps) {
  int32_t  warp=threadIdx.x>>5, warpThread=threadIdx.x & 0x1F;
  uint32_t localTotal;
  
  if(warp<warps) {
    localTotal=warpPrefixSum(value);
    if(warpThread==31)
      store_shared_u32(sAddr + warp*4, localTotal);
      
    asm volatile ("bar.sync  1,%0;" : : "r"(warps*32));
    
    #pragma unroll
    for(int32_t i=0;i<warp;i++)
      localTotal+=load_shared_u32(sAddr + i*4);
  }
  return localTotal;  
}

__device__ __forceinline__ uint32_t udiv3(uint32_t x) {
  // IMPORTANT -- this routine will fail on values near 2^32
  return __umulhi(x, 0x55555556);
}

__device__ __forceinline__ uint32_t udiv5(uint32_t x) {
  // IMPORTANT -- this routine will fail on values near 2^32
  return __umulhi(x, 0x33333334);
}

__device__ __forceinline__ uint32_t compress(uint32_t data) {
  int32_t warpThread=threadIdx.x & 0x1F, div3=udiv3(warpThread), mod3=warpThread-3*div3, shift=mod3*8+8;
  uint32_t low, high;
  
  low=__shfl_sync(0xFFFFFFFF, data, div3*4 + mod3);
  high=__shfl_sync(0xFFFFFFFF, data, div3*4 + mod3 + 1);
  return __funnelshift_r(low<<8, high, shift);
}

__device__ __forceinline__ uint32_t uncompress(uint32_t data) {
  int32_t  warpThread=threadIdx.x & 0x1F, base=(warpThread>>2)*3, quad=(warpThread & 0x03), shift=32-quad*8;
  uint32_t low, high;
  
  low=__shfl_sync(0xFFFFFFFF, data, base + max(quad-1, 0));
  high=__shfl_sync(0xFFFFFFFF, data, base + min(quad, 2));
  return __funnelshift_r(low, high, shift) & 0xFFFFFF;
}


