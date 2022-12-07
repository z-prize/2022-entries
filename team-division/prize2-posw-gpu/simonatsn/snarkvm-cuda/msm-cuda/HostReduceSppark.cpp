// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

namespace Host {

#ifndef __CUDA_ARCH__

#include <ff/bls12-377.hpp>

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

typedef jacobian_t<fp_t>   point_t;
typedef xyzz_t<fp_t>       bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t               scalar_t;

class HostReduceSppark {
public:
  static const size_t resultsPerBucket = 3;
    
  uint32_t windows;
  uint32_t windowBits;
  uint32_t warps;
  uint32_t warpsPerWindow;
  uint32_t bucketsPerThread;

  HostReduceSppark(MSMParams &params, uint32_t initialWarps) {
    windows    = params.precompWindows();
    windowBits = params.windowBits();
    warps      = initialWarps;

    warpsPerWindow   = warps / windows;
    uint32_t threads = warpsPerWindow * 32;
    bucketsPerThread = ((params.bucketsPerWindow() + threads - 1) /
                        threads);
  }

  void reduceWindow(uint32_t window, point_t& result, bucket_t* warpResults) {
    bucket_t sum, sos, interior;
    int      scaleAmount = bucketsPerThread;

    result.inf();
    sum.inf();
    sos.inf();
    interior.inf();
    for (int i = warpsPerWindow - 1; i >= 0; i--) {
      result.add(warpResults[i * resultsPerBucket]);
      interior.add(warpResults[i * resultsPerBucket + 2]);
      if(i>0) {
        sum.add(warpResults[i * resultsPerBucket + 1]);
        sos.add(sum);
      }
    }
    
    point_t interior_p = interior;
    point_t raise = sos;
    for (int i = 0;i < 5; i++)
      raise.dbl();
    interior_p.add(raise);
      
    while (scaleAmount != 0) {
      if ((scaleAmount & 0x01) != 0)
        result.add(interior_p);

      interior_p.dbl();
      scaleAmount = scaleAmount >> 1;
    }
}  

  void reduce(p1_xyz_t* _result, bucket_t* warpResults) {
    point_t *result = (point_t*)_result;
    point_t windowResult;

    result->inf();
    for (int window = windows - 1; window >= 0; window--) {
      for (int i = 0; i < windowBits; i++) {
        result->dbl();
      }
      reduceWindow(window, windowResult,
                   &warpResults[window * warpsPerWindow * resultsPerBucket]);
      result->add(windowResult);
    }
  }
};  

#endif // __CUDA_ARCH__
  
}  /* Namespace Host */
