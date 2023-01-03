/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Written by Niall Emmart.

***/

namespace Host {

template<class Field>
class HostReduce {
  typedef typename Host::PointXYZZ<Field>       PointXYZZ;
  typedef typename Host::AccumulatorXYZZ<Field> AccumulatorXYZZ;

  public:
  uint32_t windows, windowBits, warps, warpsPerWindow, bucketsPerThread;

  HostReduce(uint32_t initialWindows, uint32_t initialWindowBits, uint32_t initialWarps) {
    windows=initialWindows;
    windowBits=initialWindowBits;
    warps=initialWarps;

    warpsPerWindow=warps/windows;
    bucketsPerThread=((1<<windowBits-1)+warpsPerWindow*32-1)/(warpsPerWindow*32);
  }

  void reduceWindow(AccumulatorXYZZ& result, uint32_t* warpResults) {
    AccumulatorXYZZ sum, sos, interior, scaled;
    PointXYZZ       point;
    int             scaleAmount=bucketsPerThread;

    result.setZero();
    for(int i=warpsPerWindow-1;i>=0;i--) {
      point.load(warpResults + i*3*48 + 0);         // warp sum{sum of sum}
      result.add(point);
      point.load(warpResults + i*3*48 + 96);        // warp sum{warpThread * sum}
      interior.add(point);                          // must be scaled by 391
      point.load(warpResults + i*3*48 + 48);        // warp sum{sum}
      if(i>0) {
        sum.add(point);
        sos.add(sum.xyzz);                          // must be scaled by 32*391
      }
    }

    // scale sos by 32, add to interior
    for(int i=0;i<5;i++)
      sos.dbl(sos.xyzz);
    interior.add(sos.xyzz);

    // scale interior by 391, add to total
    while(scaleAmount!=0) {
      if((scaleAmount & 0x01)!=0)
        result.add(interior.xyzz);
      interior.dbl(interior.xyzz);
      scaleAmount=scaleAmount>>1;
    }
  }

  void reduce(uint64_t *msmResult, uint32_t* warpResults) {
    AccumulatorXYZZ result, windowResult;

    for(int window=windows-1;window>=0;window--) {
      for(uint32_t i=0;i<windowBits;i++)
        result.dbl(result.xyzz);
      reduceWindow(windowResult, warpResults + window*warpsPerWindow*3*48);
      result.add(windowResult.xyzz);
    }
    result.xyzz.normalize();

// FIX FIX FIX
// result.xyzz.dump();   // useful for debugging

    Field::exportField(msmResult, result.xyzz.x);
    Field::exportField(msmResult+6, result.xyzz.y);
    Field::exportField(msmResult+12, result.xyzz.zz);
  }
}; 

}  /* Namespace Host */

