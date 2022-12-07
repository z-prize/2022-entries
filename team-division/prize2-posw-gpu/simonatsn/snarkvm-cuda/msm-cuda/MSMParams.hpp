#ifndef __MSMPARAMS_HPP__
#define __MSMPARAMS_HPP__

#ifdef __CUDA_ARCH__    // device-side field types
#define DEVICE __device__
#else
#define DEVICE
#endif

typedef struct {
private:
  uint32_t _windowBits;
  // The number of effective windows, after precomputation
  uint32_t _precompWindows;
  // The stride for precomputed points. 
  uint32_t _precompStride;

public:
  uint32_t _exponentBits;
  // Number of bases/points
  uint32_t _pointCount;

  void set(uint32_t exponentBits, uint32_t windowBits,
           uint32_t pointCount, uint32_t precompWindows,
           uint32_t precompStride) {
    _exponentBits = exponentBits;
    _windowBits = windowBits;
    _pointCount = pointCount;
    _precompWindows = precompWindows;
    _precompStride = precompStride;
  }

  DEVICE inline uint32_t windowBits() {
    return _windowBits;
  }
  
  DEVICE inline uint32_t tailBits() {
    return 0;
  }

  DEVICE inline uint32_t allWindows() {
    return (_exponentBits + _windowBits - 1) / _windowBits;
  }

  DEVICE inline uint32_t precompWindows() {
    return _precompWindows;
  }

  DEVICE inline uint32_t precompStride() {
    return _precompStride;
  }
  
  // Bases times number of windows - the number of work items for the planner
  DEVICE inline uint32_t points() {
    return allWindows() * _pointCount;
  }
  
  DEVICE inline uint32_t bucketsPerWindow() {
    return 1 << _windowBits;
  }

  DEVICE inline uint32_t bucketsTail() {
    return 1 << tailBits();
  }
  
  // Total bucket count
  DEVICE inline uint32_t buckets() {
    assert (tailBits() == 0);
    return precompWindows() << _windowBits;
  }
} MSMParams;

typedef struct {
  uint32_t sumsPerBlock;
  uint32_t threadsPerSum;
  uint32_t bucketsPerThread;

  uint32_t bucketsPerSum;
  uint32_t blocks;

  uint32_t sumsPerWindow;
  uint32_t numBucketSumPoints;
} MSMReduce;

#endif
