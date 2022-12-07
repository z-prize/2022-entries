/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Author(s):  Niall Emmart
            Sougata Bhattacharya
            Antony Suresh

***/

#include <stdint.h>
#include <stdbool.h>

#if !defined(C_BUILD) && !defined(WASM_BUILD)
  #error Please compile with -DC_BUILD or -DWASM_BUILD
#endif

#define OLD_INVERSE
#define MEMORY_BLOCK_SIZE 65536

#include "Types.h"
#include "PackedField.h"
#include "Field.h"
#include "MSM.h"

#include "Log.c"
#include "Field.c"
#include "Curve.c"

#include "FieldInverse.c"
#include "LambdaQR.c"

// global MSM params
uint32_t msmNPoints;
uint32_t msmWindowBits;
uint32_t msmMaxChain;
uint32_t msmMaxCollisions;

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

void simpleMSM(uint32_t* res, uint32_t* points, uint32_t* scalars, uint32_t count) {
  AccumulatorXYZZ acc;
  PointXY         point;
  PointXYZZ       scaled;
  PointXYZZ       sum;
  PointXY         normalized;

  initializeAccumulatorXYZZ(&acc);

  // Scale points and add them
  for(int i=0;i<count;i++) {
    PointXY   point;
    PointXYZZ scaled;

    loadXY(&point, points + i*26);

    fieldMul64FullReduce(&point.x, &point.x);    // convert from Rust Montgomery to internal Montgomery
    fieldMul64FullReduce(&point.y, &point.y);

    scaleXY(&scaled, &point, scalars+i*8, 255);
    addXYZZ(&acc, &scaled);
  }    

  getAccumulator(&sum, &acc);
  normalizeXYZZ(&normalized, &sum);
  getField(res, &normalized.x, true);
  getField(res+12, &normalized.y, true);
}

bool msmInitialize(uint32_t nPoints, uint32_t windowBits, uint32_t maxChain, uint32_t maxCollisions) {
  msmNPoints=nPoints;
  msmWindowBits=windowBits;
  msmMaxChain=maxChain;
  msmMaxCollisions=maxCollisions;

  // We could compute a minimum size here, but I'm lazy.  Maximizing everything gives us 42MB .

  // 0MB - 8MB     temp storage for MSM computation  (result, buckets, collision buffers, etc)
  // 8MB - 16MB    storage to hold scalars
  // 16MB - 42MB   storage to hold points

  setMemorySize(42*1024*1024);

  return true;
}

void* msmPointsOffset() {
  uint32_t* res=0;

  // 16MB offset
  return (void*)(res + 4*1024*1024);
}

void* msmScalarsOffset() {
  uint32_t* res=0;

  // 8MB offset
  return (void*)(res + 2*1024*1024);
}

void* msmRun() {
  uint32_t* base;

  #if defined(WASM_BUILD)
    base=0;
  #endif

  simpleMSM(base, base + 4*1024*1024, base + 2*1024*1024, msmNPoints);
  return (void*)base;
}

uint32_t logTester(uint32_t passed) {
  logInitialize();
  logString("Hello world\n");
  logHex(0xCAFEBABE);
  logString("\n");
  logDec(-251);
  logString("\n");
  logString("Passed value=");
  logDec(passed);
  logString("\n");
  return passed;
}

#if defined(C_BUILD)

int main() {
  PointXY   point;
  uint32_t* res;

  msmInitialize(262144, 13, 1024, 128);
  res=msmRun();

  printf("  x=");
  for(int i=11;i>=0;i--) 
    printf("%08X", res[i]);
  printf("\n");

  printf("  y=");
  for(int i=11;i>=0;i--) 
    printf("%08X", res[i+12]);
  printf("\n");

  printf("modmul: %d\n", modmul);
  printf("modsqr: %d\n", modsqr);
  printf("tot: %d\n", modmul + modsqr);
}
#endif
