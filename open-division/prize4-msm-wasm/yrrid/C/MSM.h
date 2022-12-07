#if !defined(C_BUILD) && !defined(WASM_BUILD)
  #error Please compile with -DWASM_BUILD or -DC_BUILD
#endif

// Utility APIs
void loadXY(PointXY* r, uint32_t* source);                                                // load XY from memory (24 x 32-bits)
void storeXY(uint32_t* destination, PointXY* point);                                      // store XY to memory (24 x 32-bits)
void loadXYZZ(PointXYZZ* r, uint32_t* source);                                            // load XYZZ from memory (48 x 32-bits)
void storeXYZZ(uint32_t* destination, PointXYZZ* point);                                  // store XYZZ to memory (48 x 32-bits)
void storeAccumulatorXYZZ(uint32_t* destination, AccumulatorXYZZ* accumulator);           // combination of getAccumulator and storeXYZZ

void negateXY(PointXY* affine);                                                           // affine = -affine
void negateXYZZ(PointXYZZ* point);                                                        // point = -point

void dumpXY(PointXY* point);                                                              // print out an XY point
void dumpXYZZ(PointXYZZ* point);                                                          // print out an XYZZ point

// Accumulator XYZZ APIs
void initializeAccumulatorXYZZ(AccumulatorXYZZ* accumulator);                             // acc = 0

void doubleXY(AccumulatorXYZZ* accumulator, PointXY* affine);                             // acc = 2*affine
void doubleXYZZ(AccumulatorXYZZ* accumulator, PointXYZZ* point);                          // acc = 2*point
void doubleAccumulatorXYZZ(AccumulatorXYZZ* acumulator, AccumulatorXYZZ* point);          // acc = 2*point
void addXY(AccumulatorXYZZ* accumulator, PointXY* affine);                                // acc += affine
void addXYZZ(AccumulatorXYZZ* accumulator, PointXYZZ* point);                             // acc += point

void getAccumulator(PointXYZZ* r, AccumulatorXYZZ* accumulator);                          // r (xyzz point) = acc
void normalizeXYZZ(PointXY* r, PointXYZZ* point);                                         // r (affine point) = normalized(point)

// High level APIs
void scaleXY(PointXYZZ* r, PointXY* affinePoint, uint32_t* scalar, uint32_t bits);        // r (xyzz point) = affinePoint * scalar
void scaleXYZZ(PointXYZZ* r, PointXYZZ* point, uint32_t* scalar, uint32_t bits);          // r (xyzz point) = point * scalar
void scaleByLambdaXY(PointXY* r, PointXY* point);                                         // r (affine point) = lambda * affinePoint
      
void lambdaQR(uint32_t* q, uint32_t* r, uint32_t* scalar);                                // q=scalar/lambda, r=scalar%lambda

// Batched Accumulate XY APIs
void initializeAccumulatorXY(AccumulatorXY* accumulator);                                 // acc = 0
void initializeFieldState(Field* state);                                                  // initialize field state
void addXYPhaseOne(Field* state, AccumulatorXY* acc, PointXY* affine, uint32_t* invs);    // forward pass 
void addXYPhaseTwo(Field* state, AccumulatorXY* acc, PointXY* affine, uint32_t* invs);    // backward pass -> results

