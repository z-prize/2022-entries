/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Author(s):  Niall Emmart
            Sougata Bhattacharya
            Antony Suresh
            Kushal Neralakatte 

***/

typedef struct {
  uint32_t* points;
  uint32_t* scalars;
  uint8_t*  logOutput;
  uint32_t  pointCount;
  uint32_t  windowBits;
  uint32_t  maxBatch;
  uint32_t  maxCollisions;
} CMRun;

typedef uint32_t HeadTailList;

typedef struct {
  // "instance variables"
  //PointXY      point;
  Field        inverseState;        // state used by batched affine add routine
  //int32_t      pointIndex;
  //bool         pointSign;
  //bool         pointPhi;
  uint32_t     batchCount;
  uint32_t     pointCount;
  uint32_t     collisionCount;

  HeadTailList processing;          // collision points/buckets being processed in this iteration
  HeadTailList unprocessed;         // collision points/buckets queued for next iteration
  HeadTailList available;           // list of empty slots

  // input/output pointers 
  uint32_t*    results;
  uint32_t*    points;              // point data
  uint32_t*    buckets;             // bucket data (for bucket sums)

  // scratch pointers
  uint32_t*    collisionBitMap;     // scratch space to detect collisions (size: 1 bit per bucket)
  uint32_t*    listSlots;           // scratch space - used to build linked lists (size: maxCollisions*2)
  uint32_t*    scratchPoints;       // scratch space to hold the various points we need to add (size: maxBatch*24 + maxCollisions*2*24)
  uint32_t*    pointsAndBuckets;    // scratch space - points and buckets to process in phase two (size: maxBatch)
  uint32_t*    inverses;            // scratch space needed for the batched adds (size: maxBatch*12)

  // constant parameters
  uint32_t     windows;     
  uint32_t     windowBits;
  uint32_t     maxCollisions;
  uint32_t     maxBatch;
} CMState;

void cmInitialize(CMState* state, uint32_t windowBits, uint32_t maxBatch, uint32_t maxCollisions);
void cmAddPointToBucket(CMState* state, uint32_t bucket, PointXY* point);
void cmCompleteProcessing(CMState* state);
