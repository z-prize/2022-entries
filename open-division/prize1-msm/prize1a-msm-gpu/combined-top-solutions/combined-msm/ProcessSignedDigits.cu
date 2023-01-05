/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Written by Niall Emmart.

***/

// This code is concerned with scalar processing.  We assume the following parameters:
// Input exponentBits=256, used exponentBits=253, windowBits=23, windows=11.
// Note, with windowBits=23 and 11 windows, there are no tail bits

// Process Signed Digits:
//   1.   Let M be the group field order (253 bits)
//   2.   Let s=scalar mod m
//   3.   If s.[bit 252] is set, s=M-s, negate the corresponding point.
//        Note, s.[bit 252] will now be cleared.
//   4.   Break s into 11 windows of 23-bit window slices.  
//   5.   All 23-bit slices for the same window are stored adjacent to one another in memory.

__device__ __forceinline__ void slice23(uint32_t* sliced, uint32_t* packed) {
  sliced[0]=packed[0] & 0x7FFFFF;
  sliced[1]=__funnelshift_r(packed[0], packed[1], 23) & 0x7FFFFF;
  sliced[2]=__funnelshift_r(packed[1], packed[2], 14) & 0x7FFFFF;
  sliced[3]=(packed[2]>>5) & 0x7FFFFF;
  sliced[4]=__funnelshift_r(packed[2], packed[3], 28) & 0x7FFFFF;
  sliced[5]=__funnelshift_r(packed[3], packed[4], 19) & 0x7FFFFF;
  sliced[6]=__funnelshift_r(packed[4], packed[5], 10) & 0x7FFFFF;
  sliced[7]=(packed[5]>>1) & 0x7FFFFF;
  sliced[8]=__funnelshift_r(packed[5], packed[6], 24) & 0x7FFFFF;
  sliced[9]=__funnelshift_r(packed[6], packed[7], 15) & 0x7FFFFF;
  sliced[10]=(packed[7]>>6) & 0x7FFFFF;
  sliced[11]=0;
}

__device__ __forceinline__ bool sub(uint32_t* res, uint32_t* a, uint32_t* b) {
  return mp_sub_carry<8>(res, a, b);
}

__device__ __forceinline__ void addN(uint32_t* res, uint32_t* x) {
  uint32_t localN[8];
  
  localN[0]=0x00000001;
  localN[1]=0x0a118000;
  localN[2]=0xd0000001;
  localN[3]=0x59aa76fe;
  localN[4]=0x5c37b001;
  localN[5]=0x60b44d1e;
  localN[6]=0x9a2ca556;
  localN[7]=0x12ab655e; 

  mp_add<8>(res, localN, x);
}

__device__ __forceinline__ void negate(uint32_t* res, uint32_t *x) {
  uint32_t localN[8];
  
  localN[0]=0x00000001;
  localN[1]=0x0a118000;
  localN[2]=0xd0000001;
  localN[3]=0x59aa76fe;
  localN[4]=0x5c37b001;
  localN[5]=0x60b44d1e;
  localN[6]=0x9a2ca556;
  localN[7]=0x12ab655e; 
  
  mp_sub<8>(res, localN, x);
}

__global__ void processSignedDigitsKernel(void* processedScalarData, void* scalarData, uint32_t points) {
  uint32_t  warpThread=threadIdx.x & 0x1F, warp=threadIdx.x>>5;  
  uint32_t  current=0, distributed=0, mask=(1<<threadIdx.x)>>1, estimate;
  uint32_t  packed[8], nTerm[8], sliced[12];
  bool      carry, neg;
  void*     base;
  
  extern __shared__ uint32_t transpose[];
  
  // launch geometry is 256 threads per block, enough blocks to process everything
  
  // compare 0*N through 14*N
  if(threadIdx.x==0) distributed=0x00000001;
  if(threadIdx.x==1) distributed=0x0a118000;
  if(threadIdx.x==2) distributed=0xd0000001;
  if(threadIdx.x==3) distributed=0x59aa76fe;
  if(threadIdx.x==4) distributed=0x5c37b001;
  if(threadIdx.x==5) distributed=0x60b44d1e;
  if(threadIdx.x==6) distributed=0x9a2ca556;
  if(threadIdx.x==7) distributed=0x12ab655e; 

  if(threadIdx.x<8) 
    store_shared_u32(threadIdx.x*4 + 8448, current);
  for(int32_t i=1;i<15;i++) {
    current=uadd_cc(current, distributed);
    carry=uaddc(0, 0)!=0;
    if((__ballot_sync(0xFFFFFFFF, carry) & mask)!=0)
      current++;
    if(threadIdx.x<8)
      store_shared_u32(i*32 + threadIdx.x*4 + 8448, current);
  }
  
  scalarData=byteOffset(scalarData, blockIdx.x, 256*32);
  
  #pragma unroll
  for(int32_t i=0;i<8;i++) 
    transpose[i*256 + threadIdx.x]=*(uint32_t*)byteOffset(scalarData, (i*256 + threadIdx.x)*4);
    
  __syncthreads();
  
  load_shared_u4(packed[0], packed[1], packed[2], packed[3], threadIdx.x*32);
  load_shared_u4(packed[4], packed[5], packed[6], packed[7], threadIdx.x*32 + 16);
    
  __syncthreads();
  
  // compute packed = packed mod N
  estimate=__umulhi(packed[7], 0x0E);  //  slight over-estimate of packed / N
  load_shared_u4(nTerm[0], nTerm[1], nTerm[2], nTerm[3], estimate*32 + 8448);
  load_shared_u4(nTerm[4], nTerm[5], nTerm[6], nTerm[7], estimate*32 + 8448 + 16);
  if(!sub(packed, packed, nTerm))
    addN(packed, packed);
    
  // set neg to high bit of packed
  neg=packed[7]>=0x10000000;
  
  // if neg, packed=N-packed
  if(neg)
    negate(packed, packed);
    
  // slice the packed into 23 bit windows
  slice23(sliced, packed);
  
  // do signed digit processing
  #pragma unroll
  for(int32_t i=0;i<11;i++) {
    if(sliced[i]<=0x00400000) {
      if(neg && sliced[i]!=0)
        sliced[i]+=0x00800000;
    }
    else if(sliced[i]<0x00800000) {
      sliced[i]=(sliced[i] ^ 0x7FFFFF) + 1;
      if(!neg)
        sliced[i]+=0x00800000;
      sliced[i+1]++;
    }
    else {
      sliced[i]=0;
      sliced[i+1]++;
    }
    sliced[i]=compress(sliced[i]);
  }

  if(warpThread<24) {
    #pragma unroll
    for(int32_t i=0;i<11;i++) {
      base=byteOffset(processedScalarData, i, 3*points);
      *(uint32_t*)byteOffset(base, warpThread*4 + warp*96 + blockIdx.x*768)=sliced[i];
    }
  }
}

