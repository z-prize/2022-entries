/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Written by Niall Emmart.

***/

#include <stdint.h>  

class chain_t {
  public:
  bool firstOperation;
  
  __device__ __forceinline__ chain_t() {
    firstOperation=true;
  }
  
  __device__ __forceinline__ chain_t(bool carry) {
    firstOperation=false;
    uadd_cc(carry ? 1 : 0, 0xFFFFFFFF);
  }
  
  __device__ __forceinline__ void reset() {
    firstOperation=true;
    uadd_cc(0, 0);
  }
  
  __device__ __forceinline__ void reset(bool carry) {
    firstOperation=false;
    uadd_cc(carry ? 1 : 0, 0xFFFFFFFF);
  }
  
  __device__ __forceinline__ bool getCarry() {
    return uaddc(0, 0)!=0;
  }
  
  __device__ __forceinline__ uint32_t add(uint32_t a, uint32_t b) {
    if(firstOperation) 
      uadd_cc(0, 0);
    firstOperation=false;
    return uaddc_cc(a, b);
  }
  
  __device__ __forceinline__ uint32_t sub(uint32_t a, uint32_t b) {
    if(firstOperation)
      uadd_cc(1, 0xFFFFFFFF);
    firstOperation=false;
    return usubc_cc(a, b);
  }
  
  __device__ __forceinline__ uint2 madwide(uint32_t a, uint32_t b, uint2 c) {
    if(firstOperation) 
      uadd_cc(0, 0);
    firstOperation=false;    
    return u2madwidec_cc(a, b, c);
  }

  __device__ __forceinline__ uint64_t madwide(uint32_t a, uint32_t b, uint64_t c) {
    if(firstOperation) 
      uadd_cc(0, 0);
    firstOperation=false;    
    return madwidec_cc(a, b, c);
  }
};
