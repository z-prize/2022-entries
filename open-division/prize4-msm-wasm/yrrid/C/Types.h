/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Author(s):  Niall Emmart
            Sougata Bhattacharya
            Antony Suresh

***/

typedef struct {
  uint64_t f0, f1, f2, f3, f4, f5;
} PackedField;

typedef struct {
  uint64_t f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12;
} Field;

typedef struct {
  Field x;
  Field y;
} PointXY;

typedef struct {
  Field x;
  Field y;
  Field zz;
  Field zzz;
} PointXYZZ;

typedef struct {
  PointXY xy;
} AccumulatorXY;          // xy is private with special semantics

typedef struct {
  bool      infinity;
  bool      affine;
  PointXYZZ xyzz;         // xyzz is private, don't touch!  Access through APIs.
} AccumulatorXYZZ;

typedef struct {
  uint32_t aCoeffA;
  uint32_t aCoeffB;
  uint32_t bCoeffA;
  uint32_t bCoeffB;
} Coeffs;

typedef struct {
  uint64_t   x[6];  
  uint64_t   y[6];
  bool       infinity;
  char       ignore[7];
} RustAffinePoint;

typedef struct {
  uint64_t   x[6];        // copy x into here
  uint64_t   y[6];        // copy y into here
  uint64_t   z[6];        // set to R
} RustProjectivePoint;

