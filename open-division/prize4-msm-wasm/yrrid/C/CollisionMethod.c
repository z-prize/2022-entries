/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Author(s):  Niall Emmart
            Sougata Bhattacharya
            Antony Suresh
            Kushal Neralakatte 

***/

// This group of functions uses the collision method to batch point adds into buckets.
// The basic idea is that we can add any point into any bucket.  If that bucket already
// has an add pending in the batch, we have a "collision" and we queue the collision
// for processing with the next batch.

// This code is structured to avoid a lot of copying of points from one place to another.
// But as a result, the list processing is fairly complicated.

#define HEAD(listHT)                 ((listHT) & 0x3FFF)
#define TAIL(listHT)                 ((listHT)>>14)
#define NEXT(entryPN)                ((entryPN) & 0x3FFF)
#define PAYLOAD(entryPN)             ((entryPN)>>14)
#define MAKE_HEAD_TAIL(head, tail)   (((tail)<<14) | (head))

#if defined(C_BUILD)

  static inline uint32_t* cmResults(CMState* state) {
    return state->results;
  }

  static inline uint32_t* cmPoints(CMState* state) {
    return state->points;
  }

  static inline uint32_t* cmBuckets(CMState* state) {
    return state->buckets;
  }

  static inline uint32_t* cmCollisionBitMap(CMState* state) {
    return state->collisionBitMap;
  }

  static inline uint32_t* cmListSlots(CMState* state) {
    return state->listSlots;
  }

  static inline uint32_t* cmScratchPoints(CMState* state) {
    return state->scratchPoints;
  }

  static inline uint32_t* cmPointsAndBuckets(CMState* state) {
    return state->pointsAndBuckets;
  }

  static inline uint32_t* cmInverses(CMState* state) {
    return state->inverses;
  }

#endif

#if defined(WASM_BUILD) 

  // for WASM, these are all hard coded with max values
  // maybe the compiler will be smart and generate vectorized loads?

  // memory layout -- assumptions:  maxBatch<=4096, maxCollisions<=2048, (windows<<windowBits-1)<=40960
  //  Offset        Size                     Description
  //  0             96 (1 xy)                Result x and y
  //  32K           32K (256K*1 bit)         Collision bit map (windows=8, windowBits=15)
  //  64K           16K (4K*4)               List slots (2*maxCollisions)
  //  80K           16K (4K*4)               Points and Buckets (maxBatch)  
          
  //  96K           8K*96                    Scratch points (maxBatch + 2*maxCollisions)   [!!! at least 8K]
  //  864K          8K*48                    Inverses                                      [!!! at least 8K]
  //  1248K         256K*96                  Buckets

  //  26MB          256K*32                  Scalars
  //  34MB          256K*104                 Points
  //               ------
  //                60MB

  static inline uint32_t* cmResults(CMState* state) {
    // 96 bytes + some reserved space
    return (uint32_t*)MEMORY_BASE;
  }

  static inline uint32_t* cmCollisionBitMap(CMState* state) {
    // 32K (256K/8)
    return (uint32_t*)(MEMORY_BASE + 32*1024);
  }

  static inline uint32_t* cmListSlots(CMState* state) {
    // 16K (maxCollisions*2*4)
    return (uint32_t*)(MEMORY_BASE + 64*1024);
  }

  static inline uint32_t* cmPointsAndBuckets(CMState* state) {
    // 16K (maxBatch*4)
    return (uint32_t*)(MEMORY_BASE + 80*1024);
  }

  static inline uint32_t* cmScratchPoints(CMState* state) {
    return (uint32_t*)(MEMORY_BASE + 96*1024);
  }

  static inline uint32_t* cmInverses(CMState* state) {
    return (uint32_t*)(MEMORY_BASE + 864*1024);
  }

  static inline uint32_t* cmBuckets(CMState* state) {
    return (uint32_t*)(MEMORY_BASE + 1248*1024);
  }

  static inline uint32_t* cmPoints(CMState* state) {
    return (uint32_t*)(MEMORY_BASE + 34*1024*1024);
  }

#endif

static inline bool cmListEmpty(HeadTailList* list) {
  return *list==MAKE_HEAD_TAIL(0x3FFF, 0x3FFF);
}

static inline void cmListClear(HeadTailList* list) {
  *list=MAKE_HEAD_TAIL(0x3FFF, 0x3FFF);
}

static inline void cmEnqueue(CMState* state, HeadTailList* list, uint32_t entry) {
  uint32_t* listSlots=cmListSlots(state);
  uint32_t  listHT, entryPN;

  if(cmListEmpty(list)) 
    *list=MAKE_HEAD_TAIL(entry, entry);
  else {
    listHT=*list;
    *list=MAKE_HEAD_TAIL(HEAD(listHT), entry);

    entryPN=listSlots[TAIL(listHT)];
    listSlots[TAIL(listHT)]=(entryPN & 0xFFFFC000) | entry;
  }
  listSlots[entry] |= 0x3FFF;
}

static inline uint32_t cmPop(CMState* state, HeadTailList* list) {
  uint32_t* listSlots=cmListSlots(state);
  uint32_t  listHT, entryPN;

  // IMPORTANT!  Returned entry will have next pointer set

  listHT=*list;
  if(!cmListEmpty(list)) {
    if(HEAD(listHT)==TAIL(listHT))
      cmListClear(list);
    else {
      entryPN=listSlots[HEAD(listHT)];
      *list=(listHT & 0x0FFFC000) | NEXT(entryPN);
    }
  }
  return HEAD(listHT);
}

static inline void cmPushAll(CMState* state, HeadTailList* destination, HeadTailList* source) {
  uint32_t* listSlots=cmListSlots(state);
  uint32_t  sourceHT=*source, destinationHT=*destination, entryPN;

  if(cmListEmpty(source)) 
    return;

  if(cmListEmpty(destination)) 
    destinationHT=sourceHT;
  else {
    entryPN=listSlots[TAIL(sourceHT)];
    entryPN=(entryPN & 0xFFFFC000) | HEAD(destinationHT);
    listSlots[TAIL(sourceHT)]=entryPN;

    destinationHT=(destinationHT & 0x0FFFC000) | HEAD(sourceHT);
  }

  cmListClear(source);
  *destination=destinationHT;
}

static inline bool cmCollision(CMState* state, uint32_t bucket) {
  uint32_t* bitMap=cmCollisionBitMap(state);
  uint32_t  word=bucket>>5, bit=1<<(bucket & 0x1F);

  if((bitMap[word] & bit)!=0)
    return true;
  bitMap[word] |= bit;
  return false;
}

static void cmClearBitMap(CMState* state) {
  uint32_t* bitMap=cmCollisionBitMap(state);
  uint32_t  windows=state->windows;
  uint32_t  buckets=windows<<state->windowBits;
  uint32_t  words=buckets>>5;

  for(int i=0;i<words;i++)
    bitMap[i]=0;
}

void cmInitialize(CMState* state, uint32_t windowBits, uint32_t maxBatch, uint32_t maxCollisions) {
  AccumulatorXY acc;
  uint32_t*     listSlots;
  uint32_t      windows=(128 + windowBits - 1)/windowBits;

  // NOTE -- this is confusing.  Because of signed digits, the windowBits is one less.
  windowBits--;
  
  state->windows=windows;
  state->windowBits=windowBits;
  state->maxBatch=maxBatch;
  state->maxCollisions=maxCollisions;

  #if defined(C_BUILD)
     uint32_t minScratchPoints, minInverses, fieldCount, bytes;

     minScratchPoints=maxBatch + maxCollisions*2;
     if(minScratchPoints<8192)
       minScratchPoints=8192;
     minInverses=maxBatch;
     if(minInverses<8192)
       minInverses=8192;

              // res   // scratch points  // inverses      // buckets
     fieldCount=128 + minScratchPoints*2 + minInverses + (windows<<windowBits)*2;

           // scratch+inv   // collisionBitMap       // listSlots      // pointsAndBuckets
     bytes=fieldCount*48 + (windows<<windowBits)/8 + maxCollisions*2*4 + maxBatch*4;

     state->results=(uint32_t*)malloc(bytes);
     state->scratchPoints=state->results + 128*12;
     state->inverses=state->scratchPoints + minScratchPoints*2*12;
     state->buckets=state->inverses + minInverses*12;
     state->collisionBitMap=state->buckets + (windows<<windowBits)*2*12;
     state->listSlots=state->collisionBitMap + (windows<<windowBits)/32;
     state->pointsAndBuckets=state->listSlots + maxCollisions*2;
     printf("Windows=%d WindowBits=%d Bytes=%d\n", windows, windowBits+1, bytes);
  #endif

  state->batchCount=0;
  state->pointCount=0;
  state->collisionCount=0;

  initializeFieldState(&state->inverseState);

  cmClearBitMap(state);

  // link all the listSlots together
  listSlots=cmListSlots(state);
  for(int i=0;i<maxCollisions*2-1;i++) 
    listSlots[i]=i+1;
  listSlots[maxCollisions*2-1]=0x3FFF;

  cmListClear(&state->processing);
  cmListClear(&state->unprocessed);
  state->available=((maxCollisions*2-1)<<14) + 0x0000;

  initializeAccumulatorXY(&acc);
  for(int i=0;i<(windows<<windowBits);i++)
    storeXY(cmBuckets(state) + i*24, &acc.xy);
}

static void cmProcess(CMState* state) {
  uint32_t*     listSlots=cmListSlots(state);
  uint32_t*     pointsAndBuckets=cmPointsAndBuckets(state);
  uint32_t*     scratchPoints=cmScratchPoints(state);
  AccumulatorXY acc;
  PointXY       point;
  uint32_t      pointAndBucket, pointSlot, bucket;
  uint32_t      previousPointSlot, entryPN, currentSlot;
  
  // compute the inverse
  fieldInverseFullReduce(&state->inverseState, &state->inverseState);

  // run phase two
  previousPointSlot=0xFFFFFFFF;
  for(int i=state->batchCount-1;i>=0;i--) {
    pointAndBucket=pointsAndBuckets[i];
    pointSlot=pointAndBucket & 0x3FFF;
    bucket=pointAndBucket>>14;

    if(previousPointSlot!=pointSlot) {
      loadXY(&point, cmScratchPoints(state) + pointSlot*24);
      previousPointSlot=pointSlot;
    }
    loadXY(&acc.xy, cmBuckets(state) + bucket*24);
    addXYPhaseTwo(&state->inverseState, &acc, &point, cmInverses(state) + i*12);
    storeXY(cmBuckets(state) + bucket*24, &acc.xy);    
  }

  // clear bit map
  cmClearBitMap(state);
  cmPushAll(state, &state->available, &state->processing);

  initializeFieldState(&state->inverseState);

  for(int i=0;i<24;i++) 
    scratchPoints[i]=scratchPoints[state->pointCount*24 - 24 + i];

  state->batchCount=0;
  state->collisionCount=0;
  state->pointCount=1;

  // iterate over the unprocessed list, either adding the point to processing, or back to unprocessed
  currentSlot=HEAD(state->unprocessed);
  cmListClear(&state->unprocessed);
  while(currentSlot!=0x3FFF) {
    entryPN=listSlots[currentSlot];
    pointSlot=state->maxBatch + currentSlot; 
    bucket=PAYLOAD(entryPN);

    if(!cmCollision(state, bucket)) {
      uint32_t i=state->batchCount++;

      loadXY(&point, cmScratchPoints(state) + pointSlot*24);
      loadXY(&acc.xy, cmBuckets(state) + bucket*24);
      addXYPhaseOne(&state->inverseState, &acc, &point, cmInverses(state) + i*12);
      pointsAndBuckets[i]=(bucket<<14) | pointSlot;

      // no collision, put it in the processing queue
      cmEnqueue(state, &state->processing, currentSlot);
    }
    else {
      state->collisionCount++;

      // collision -- try again on the next iteration
      cmEnqueue(state, &state->unprocessed, currentSlot);
    }

    currentSlot=NEXT(entryPN);
  }
}

static inline void cmUpdatePoint(CMState* state, PointXY* point) {
  uint32_t pointIndex=state->pointCount++;

  storeXY(cmScratchPoints(state) + pointIndex*24, point);
}

void cmAddPointToBucket(CMState* state, uint32_t bucket, PointXY* point) {
  uint32_t*     listSlots=cmListSlots(state);
  uint32_t*     pointsAndBuckets=cmPointsAndBuckets(state);
  AccumulatorXY acc;
  uint32_t      freeSlot, pointSlot;

  if(!cmCollision(state, bucket)) {
    AccumulatorXY acc;
    uint32_t      i=state->batchCount++;

    loadXY(&acc.xy, cmBuckets(state) + bucket*24);
    addXYPhaseOne(&state->inverseState, &acc, point, cmInverses(state) + i*12);
    pointsAndBuckets[i]=(bucket<<14) + state->pointCount-1;
  }
  else {
    freeSlot=cmPop(state, &state->available);
    pointSlot=state->maxBatch + freeSlot; 
    storeXY(cmScratchPoints(state) + pointSlot*24, point);
    listSlots[freeSlot]=(bucket<<14) + 0x3FFF;
    cmEnqueue(state, &state->unprocessed, freeSlot);

    state->collisionCount++;
  }

  if(state->collisionCount==state->maxCollisions || state->batchCount==state->maxBatch) 
    cmProcess(state);
}

void cmCompleteProcessing(CMState* state) {
  cmProcess(state);
  while(!cmListEmpty(&state->unprocessed) || !cmListEmpty(&state->processing)) 
    cmProcess(state);
}

/*
void cmDump(CMState* state) {
  Field     f;
  uint32_t* buckets=cmBuckets(state);

  for(int i=0;i<144*1024*2;i++) {
    fieldLoad(&f, buckets + i*12);
    fieldDump(&f, true);
  }
}
*/
