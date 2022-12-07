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

#include "Log.c"
#include "Field.c"
#include "Curve.c"
#include "FieldInverse.c"
#include "LambdaQR.c"

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

void simpleMSM(uint32_t* montgomeryPoints, uint32_t* scalars, uint32_t count) {
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

    loadXY(&point, montgomeryPoints+i*24);
    scaleXY(&scaled, &point, scalars+i*8, 255);
    addXYZZ(&acc, &scaled);
  }    

  printf("Solution:\n");
  getAccumulator(&sum, &acc);
  normalizeXYZZ(&normalized, &sum);
  dumpXY(&normalized);
}

void lambdaMSM(uint32_t* montgomeryPoints, uint32_t* scalars, uint32_t count) {
  uint32_t        q[count*4], r[count*4];
  AccumulatorXYZZ acc;
  PointXY         point;
  PointXYZZ       scaled;
  PointXYZZ       sum;
  PointXY         normalized;

  // Break scalars into q and r terms
  for(int i=0;i<count;i++) 
    lambdaQR(q+i*4, r+i*4, scalars+i*8);

  initializeAccumulatorXYZZ(&acc);

  // Process r scalars
  for(int i=0;i<count;i++) {
    PointXY   point;
    PointXYZZ scaled;

    loadXY(&point, montgomeryPoints+i*24);
    scaleXY(&scaled, &point, r+i*4, 128);       // note, there are cases where both Q and R are 128 bits long
    addXYZZ(&acc, &scaled);
  }    

  // Process q scalars
  for(int i=0;i<count;i++) {
    loadXY(&point, montgomeryPoints+i*24);
    scaleByLambdaXY(&point, &point);
    scaleXY(&scaled, &point, q+i*4, 128);       // note, there are cases where both Q and R are 128 bits long
    addXYZZ(&acc, &scaled);
  }

  printf("Solution:\n");
  getAccumulator(&sum, &acc);
  normalizeXYZZ(&normalized, &sum);
  dumpXY(&normalized);
}

int main(int argc, const char** argv) {
  uint32_t  count;
  uint32_t* points;
  uint32_t* montgomeryPoints;
  uint32_t* scalars;
  long long start, end;
  if(argc!=2) {
    fprintf(stderr, "Usage: %s <count>\n", argv[0]);
    fprintf(stderr, "Reads <count> XY point values from points.hex, and <count> scalars from scalars.hex\n");
    fprintf(stderr, "and computes and prints out the MSM result.\n");
    exit(1);
  }

  count=atoi(argv[1]);
  points=(uint32_t*)malloc(96*count);
  montgomeryPoints=(uint32_t*)malloc(96*count);  
  scalars=(uint32_t*)malloc(32*count);

  FILE* pointsFile=fopen("data/points.hex", "r");
  FILE* scalarsFile=fopen("data/scalars.hex", "r");

  // Load the points and scalars
  for(int i=0;i<count;i++) {
    parseHex((uint8_t*)(points+i*24), pointsFile, 48);
    parseHex((uint8_t*)(points+i*24+12), pointsFile, 48);
    parseHex((uint8_t*)(scalars+i*8), scalarsFile, 32);
  }

  // Convert the points from normal space to Montgomery space
  toMontgomery(montgomeryPoints, points, count*2);
  
  printf("Solution using simple MSM:\n");
  simpleMSM(montgomeryPoints, scalars, count);

  printf("\n\n");
  printf("Solution using lambda MSM:\n");
  start = current_timestamp();
  lambdaMSM(montgomeryPoints, scalars, count);
  end = current_timestamp();
  printf("Time taken to calculate lambdaMSM: %lld\n",end-start);
}
