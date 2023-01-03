/***

Portions of this file originated in an NVIDIA Open Source Project.  The source can be
found here:  http://github.com/NVlabs/CGBN, from: include/cgbn/arith/asm.cu.

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Written by Niall Emmart.

***/

__device__ __forceinline__ uint32_t load_shared_u32(uint32_t sAddr) {
  uint32_t r;
  
  asm volatile ("ld.shared.u32 %0,[%1];" : "=r"(r) : "r"(sAddr));
  return r;
}

__device__ __forceinline__ void store_shared_u32(uint32_t sAddr, uint32_t value) {
  asm volatile ("st.shared.u32 [%0],%1;" : : "r"(sAddr), "r"(value));
}

__device__ __forceinline__ uint32_t shared_atomic_add_u32(uint32_t sAddr, uint32_t value) {
  uint32_t r;
  
  asm volatile ("atom.shared.add.u32 %0,[%1],%2;" : "=r"(r) : "r"(sAddr), "r"(value));
  return r;
}

__device__ __forceinline__ uint32_t shared_atomic_exch_u32(uint32_t sAddr, uint32_t value) {
  uint32_t r;
  
  asm volatile ("atom.shared.exch.b32 %0,[%1],%2;" : "=r"(r) : "r"(sAddr), "r"(value));
  return r;
}

__device__ __forceinline__ void shared_reduce_add_u32(uint32_t sAddr, uint32_t value) {
  asm volatile ("red.shared.add.u32 [%0],%1;" : : "r"(sAddr), "r"(value)); 
}

__device__ __forceinline__ uint2 load_shared_u2(uint32_t sAddr) {
  uint2 r;
  
  asm volatile ("ld.shared.v2.u32 {%0,%1},[%2];" : "=r"(r.x), "=r"(r.y) : "r"(sAddr));
  return r;
}

__device__ __forceinline__ void store_shared_u2(uint32_t sAddr, uint2 value) {
  asm volatile ("st.shared.v2.u32 [%0],{%1,%2};" : : "r"(sAddr), "r"(value.x), "r"(value.y));
}

__device__ __forceinline__ uint4 load_shared_u4(uint32_t sAddr) {
  uint4 r;
  
  asm volatile ("ld.shared.v4.u32 {%0,%1,%2,%3},[%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(sAddr));
  return r;
}

__device__ __forceinline__ void store_shared_u4(uint32_t sAddr, uint4 value) {
  asm volatile ("st.shared.v4.u32 [%0],{%1,%2,%3,%4};" : : "r"(sAddr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
}

__device__ __forceinline__ void shared_async_copy_u4(uint32_t sAddr, void* ptr) {
  asm volatile ("cp.async.cg.shared.global [%0], [%1], 16;" : : "r"(sAddr), "l"(ptr));
}

__device__ __forceinline__ void shared_async_copy_commit() {
  asm volatile ("cp.async.commit_group;");
}

__device__ __forceinline__ void shared_async_copy_wait() {
  asm volatile ("cp.async.wait_all;");
}

__device__ __forceinline__ uint32_t prmt(uint32_t lo, uint32_t hi, uint32_t control) {
  uint32_t r;
  
  asm volatile ("prmt.b32 %0,%1,%2,%3;" : "=r"(r) : "r"(lo), "r"(hi), "r"(control));
  return r;
}

__device__ __forceinline__ uint32_t uadd_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("add.cc.u32 %0,%1,%2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t uaddc_cc(uint32_t a, uint32_t b) {
  uint32_t r;
  
  asm volatile ("addc.cc.u32 %0,%1,%2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t uaddc(uint32_t a, uint32_t b) {
  uint32_t r;
  
  asm volatile ("addc.u32 %0,%1,%2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t usub_cc(uint32_t a, uint32_t b) {
  uint32_t r;

  asm volatile ("sub.cc.u32 %0,%1,%2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t usubc_cc(uint32_t a, uint32_t b) {
  uint32_t r;
  
  asm volatile ("subc.cc.u32 %0,%1,%2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint32_t usubc(uint32_t a, uint32_t b) {
  uint32_t r;
  
  asm volatile ("subc.u32 %0,%1,%2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ bool getCarry() {
  return uaddc(0, 0)!=0;
}

__device__ __forceinline__ void setCarry(bool cc) {
  uadd_cc(cc ? 1 : 0, 0xFFFFFFFF);
}

__device__ __forceinline__ uint64_t mulwide(uint32_t a, uint32_t b) {
  uint64_t r;
  
  asm volatile ("mul.wide.u32 %0,%1,%2;" : "=l"(r) : "r"(a), "r"(b));
  return r;
}

__device__ __forceinline__ uint64_t madwide(uint32_t a, uint32_t b, uint64_t c) {
  uint64_t r;
  
  asm volatile ("mad.wide.u32 %0,%1,%2,%3;" : "=l"(r) : "r"(a), "r"(b), "l"(c));
  return r;
}

__device__ __forceinline__ uint64_t madwide_cc(uint32_t a, uint32_t b, uint64_t c) {
  uint64_t r;
  
  asm volatile ("{\n\t"
                ".reg .u32 lo,hi;\n\t"
                "mov.b64        {lo,hi},%3;\n\t"
                "mad.lo.cc.u32  lo,%1,%2,lo;\n\t"
                "madc.hi.cc.u32 hi,%1,%2,hi;\n\t"
                "mov.b64        %0,{lo,hi};\n\t"
                "}" : "=l"(r) : "r"(a), "r"(b), "l"(c));                
  return r;
}

__device__ __forceinline__ uint64_t madwidec_cc(uint32_t a, uint32_t b, uint64_t c) {
  uint64_t r;
  
  asm volatile ("{\n\t"
                ".reg .u32 lo,hi;\n\t"
                "mov.b64        {lo,hi},%3;\n\t"
                "madc.lo.cc.u32 lo,%1,%2,lo;\n\t"
                "madc.hi.cc.u32 hi,%1,%2,hi;\n\t"
                "mov.b64        %0,{lo,hi};\n\t"
                "}" : "=l"(r) : "r"(a), "r"(b), "l"(c));                
  return r;
}

__device__ __forceinline__ uint64_t madwidec(uint32_t a, uint32_t b, uint64_t c) {
  uint64_t r;
  
  asm volatile ("{\n\t"
                ".reg .u32 lo,hi;\n\t"
                "mov.b64        {lo,hi},%3;\n\t"
                "madc.lo.cc.u32 lo,%1,%2,lo;\n\t"
                "madc.hi.cc.u32 hi,%1,%2,hi;\n\t"
                "mov.b64        %0,{lo,hi};\n\t"
                "}" : "=l"(r) : "r"(a), "r"(b), "l"(c));                
  return r;
}

__device__ __forceinline__ uint2 u2madwidec_cc(uint32_t a, uint32_t b, uint2 c) {
  uint2 r;
  
  asm volatile ("madc.lo.cc.u32  %0,%2,%3,%4;\n\t"
                "madc.hi.cc.u32 %1,%2,%3,%5;" : "=r"(r.x), "=r"(r.y) : "r"(a), "r"(b), "r"(c.x), "r"(c.y));
  return r;
}

__device__ __forceinline__ uint32_t ulow(uint2 xy) {
  return xy.x;
}

__device__ __forceinline__ uint32_t uhigh(uint2 xy) {
  return xy.y;
}

__device__ __forceinline__ uint32_t ulow(uint64_t wide) {
  uint32_t r;

  asm volatile ("mov.b64 {%0,_},%1;" : "=r"(r) : "l"(wide));
  return r;
}

__device__ __forceinline__ uint32_t uhigh(uint64_t wide) {
  uint32_t r;

  asm volatile ("mov.b64 {_,%0},%1;" : "=r"(r) : "l"(wide));
  return r;
}

__device__ __forceinline__ uint64_t make_wide(uint32_t lo, uint32_t hi) {
  uint64_t r;
  
  asm volatile ("mov.b64 %0,{%1,%2};" : "=l"(r) : "r"(lo), "r"(hi));
  return r;
}

__device__ __forceinline__ uint64_t make_wide(uint2 xy) {
  return make_wide(xy.x, xy.y);
}
