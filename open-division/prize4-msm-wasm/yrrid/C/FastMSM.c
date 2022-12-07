/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Author(s):  Niall Emmart
            Sougata Bhattacharya
            Antony Suresh

***/

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

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

#include "Reader.c"
#include <sys/time.h>

long long current_timestamp() {
    struct timeval te; 
    gettimeofday(&te, NULL); // get current time
    long long milliseconds = te.tv_sec*1000LL + te.tv_usec/1000; // calculate milliseconds
    // printf("milliseconds: %lld\n", milliseconds);
    return milliseconds;
}

void toMontgomery(uint32_t* to, uint32_t* from, uint32_t count) {
  Field f;

  for(int i=0;i<count;i++) {
    fieldLoad(&f, from+i*12);
    fieldToMontgomeryFullReduce(&f, &f);
    fieldStore(to+i*12, &f);
  }
}

int main(int argc, const char** argv) {
  uint32_t  count;
  uint32_t* points;
  uint32_t* montgomeryPoints;
  uint32_t* scalars;
  uint32_t* result;
  CMRun     run;
  PointXY   msm;

  long long start, end;

  if(argc<2) {
    fprintf(stderr, "Usage: %s <count> [<windowBits> <maxBatch> <maxCollision>]\n", argv[0]);
    fprintf(stderr, "Reads <count> XY point values from points.hex, and <count> scalars from scalars.hex\n");
    fprintf(stderr, "and computes and prints out the MSM result.\n");
    exit(1);
  }

  count=atoi(argv[1]);
  points=(uint32_t*)malloc(96*count);
  montgomeryPoints=(uint32_t*)malloc(96*count);  
  scalars=(uint32_t*)malloc(32*count);

  run.points=montgomeryPoints;
  run.scalars=scalars;
  run.pointCount=count;
  run.windowBits=13;
  run.maxBatch=2048;
  run.maxCollisions=256;

  if(argc>=3)
    run.windowBits=atoi(argv[2]);
  if(argc>=4)
    run.maxBatch=atoi(argv[3]);
  if(argc>=5)
    run.maxCollisions=atoi(argv[4]);

  FILE* pointsFile=fopen("Data/points.hex", "r");
  FILE* scalarsFile=fopen("Data/scalars.hex", "r");

  // Load the points and scalars
  for(int i=0;i<count;i++) {
    parseHex((uint8_t*)(points+i*24), pointsFile, 48);
    parseHex((uint8_t*)(points+i*24+12), pointsFile, 48);
    parseHex((uint8_t*)(scalars+i*8), scalarsFile, 32);
  }

  printf("Loading complete\n");

  // Convert the points from normal space to Montgomery space
  toMontgomery(montgomeryPoints, points, count*2);
  
  start = current_timestamp();
  result=cmMSM(&run);
  end = current_timestamp();

  loadXY(&msm, result);
  dumpXY(&msm);

  printf("Time taken to calculate MSM: %lld\n",end-start);

//  PointXY resultPoint;
//
//  loadXY(&resultPoint, resultWords);
//  dumpXY(&resultPoint);

   printf("modmul=%d modsqr=%d total=%d\n", modmul, modsqr, modmul+modsqr);
}

