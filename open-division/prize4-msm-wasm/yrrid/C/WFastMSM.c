/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Author(s):  Niall Emmart
            Sougata Bhattacharya
            Anthony Suresh

***/

#include <stdint.h>
#include <stdbool.h>

#if !defined(C_BUILD) && !defined(WASM_BUILD)
  #error Please compile with -DC_BUILD or -DWASM_BUILD
#endif

#define MEMORY_BLOCK_SIZE 65536
#define MEMORY_BASE 131072

#include "Types.h"
#include "PackedField.h"
#include "Field.h"
#include "MSM.h"
#include "CollisionMethod.h"

#include "Log.c"
#include "Field.c"
#include "Curve.c"

#include "FieldInverse.c"
#include "LambdaQR.c"
#include "CollisionMethod.c"
#include "CollisionMethodReduce.c"
#include "CollisionMethodPippenger.c"

// global MSM params
uint32_t initialMemorySize=0;
CMRun    run;

#if defined(WASM_BUILD)
void memset8(uint8_t* ptr, int byte, unsigned long count) {
  for(unsigned long i=0;i<count;i++)
    ptr[i]=byte;
}

void memset32(uint32_t* ptr, int byte, unsigned long count) {
  uint32_t bytes=(byte & 0xFF) * 0x01010101;

  count=count>>2;
  for(unsigned long i=0;i<count;i++)
    ptr[i]=bytes;
}

void* memset(void* ptr, int byte, unsigned long count) {
  uint8_t* ptr8=(uint8_t*)ptr;

  if((((uint64_t)ptr) & 0x03)==0 && (count & 0x03)==0) 
    memset32((uint32_t*)ptr, byte, count);
  else
    memset8((uint8_t*)ptr, byte, count);
  return ptr;
}

uint32_t getMemorySize() {
  return __builtin_wasm_memory_size(0) * MEMORY_BLOCK_SIZE;
}

void setMemorySize(uint32_t minimumSize) { 
  uint32_t current=__builtin_wasm_memory_size(0);

  current=current*MEMORY_BLOCK_SIZE;
  if(current<minimumSize)
    __builtin_wasm_memory_grow(0, (minimumSize-current+MEMORY_BLOCK_SIZE-1)/MEMORY_BLOCK_SIZE);
}
#endif

bool msmInitialize(uint32_t pointCount, uint32_t windowBits, uint32_t maxBatch, uint32_t maxCollisions) {
  uint32_t* base;

  // We could compute the exact required sizes here, but for WASM we want our important objects at exact
  // byte offsets, seems like the compiler generates faster code.

  // Offset from the end of the "initial load memory size".
  // 0MB - 26MB     scratch space for MSM computation  (result, buckets, collision buffers, etc)
  // 26MB - 34MB    storage to hold scalars
  // 34MB - 60MB    storage to hold points

  if(initialMemorySize==0) 
    initialMemorySize=getMemorySize();

  setMemorySize(initialMemorySize + 60*1024*1024);

  base=(uint32_t*)initialMemorySize;
  run.points=base + 34*256*1024;
  run.scalars=base + 26*256*1024;
  run.logOutput=((uint8_t*)base) + 4096;

  run.pointCount=pointCount;
  run.windowBits=windowBits;
  run.maxBatch=maxBatch;
  run.maxCollisions=maxCollisions;

  return true;
}

void* msmPointsOffset() {
  return run.points;
}

void* msmScalarsOffset() {
  return run.scalars;
}

void* msmRun() {
  return cmMSM(&run);
}

void* logOutput() {
  return run.logOutput;
}

void f1m_toMontgomery(uint8_t* dest, uint8_t* src) {
  // dummy stub so utility.js doesn't complain
}

void oneMillionModMuls() {
  Field f, g;

  f.f0=0xABCD; f.f1=0x1234; f.f2=0x5555; f.f3=0x3213; f.f4=0x8781; f.f5=0xFFAB; f.f6=0x889B; f.f7=0x775F;
  f.f8=0x3214; f.f9=0xABFD; f.f10=0xCAFE; f.f11=0x4451; f.f12=0xFAC1;
  g.f0=0x8788; g.f1=0x5498; g.f2=0xFAED; g.f3=0x1234; g.f4=0x5993; g.f5=0x9874; g.f6=0xAB78; g.f7=0xBB99;
  g.f8=0x5132; g.f9=0x7666; g.f10=0x5829; g.f11=0x1216; g.f12=0xFDE0;

  for(int i=0;i<1000000;i++) 
    fieldMulResolve(&f, &f, &g);
}

void oneMillionModSqrs() {
  Field f;

  f.f0=0xABCD; f.f1=0x1234; f.f2=0x5555; f.f3=0x3213; f.f4=0x8781; f.f5=0xFFAB; f.f6=0x889B; f.f7=0x775F;
  f.f8=0x3214; f.f9=0xABFD; f.f10=0xCAFE; f.f11=0x4451; f.f12=0xFAC1;

  for(int i=0;i<1000000;i++) 
    fieldSqrResolve(&f, &f);
}

