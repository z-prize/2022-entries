#include <cub/cub.cuh>

typedef std::chrono::high_resolution_clock Clock;

class GPUPlanner {
  public:
  MSMParams    _params;
  GPUPointers  _pointers;
  uint32_t     _smCount;
  size_t       _storageSize;
  size_t       _cubStorageSize;
  void*        _storage;
  void*        _cubStorage;
  bool         _streamCreated;
  cudaStream_t _stream;

  GPUPlanner() {
    _storageSize=0;
    _cubStorageSize=0;
    _smCount=0xFFFFFFFF;
    _storage=NULL;
    _cubStorage=NULL;
    _streamCreated=false;
    $CUDA(cudaStreamCreate(&_stream));
    _streamCreated=true;
  }

  GPUPlanner(cudaStream_t stream) {
    _storageSize=0;
    _cubStorageSize=0;
    _smCount=0xFFFFFFFF;
    _storage=NULL;
    _cubStorage=NULL;
    _streamCreated=false;
    _stream=stream;
  }

  ~GPUPlanner() {
    if(_storage!=NULL)
      $CUDA(cudaFree(_storage));
    if(_cubStorage!=NULL)
      $CUDA(cudaFree(_cubStorage));
    if(_streamCreated)
      $CUDA(cudaStreamDestroy(_stream));
  }

  int32_t allocate() {
    // Bucket count is 2**window size * number of windows
    uint32_t  buckets=_params.buckets();
    // Points is the number of point indexes (N * number of windows to uniquely identify buckets)
    uint32_t  points=_params.points();
    uint32_t* overlay;
    size_t    size, overlaySize1, overlaySize2, overlaySize3;

    if(_smCount==0xFFFFFFFF) {
      cudaDeviceProp deviceProperties;

      $CUDA(cudaGetDeviceProperties(&deviceProperties, 0));
      _smCount=deviceProperties.multiProcessorCount;
    }

    overlaySize1=ROUND64(points)*2;
    overlaySize2=ROUND64(buckets) + ROUND64(buckets*2)*2;
    overlaySize3=ROUND64(buckets) + ROUND64(buckets*2) + ROUND64((buckets+31)/32)*2 + ROUND64(points + 32768);

    size=overlaySize1;
    if(overlaySize2>size)
      size=overlaySize2;
    if(overlaySize3>size)
      size=overlaySize3;

    // size in words
    size+=64;
    size+=ROUND64(points)*2;

    if(_storage==NULL) {
      $CUDA(cudaMalloc(&_storage, size*4));
      if(_storage==NULL)
        return -1;
      _storageSize=size*4;
    }
    else if(size*4>_storageSize) {
      $CUDA(cudaFree(_storage));
      $CUDA(cudaMalloc(&_storage, size*4));
      if(_storage==NULL)
        return -1;
      _storageSize=size*4;
    }

    // allocate all the storage required
    _pointers.workCounts=(uint32_t*)_storage;
    _pointers.bucketsPoints=(uint64_t *)(_pointers.workCounts + 64);

    overlay=_pointers.workCounts + ROUND64(64 + 2 * points);

    // overlay 1
    _pointers.unsortedBucketsPoints=(uint64_t*)overlay;

    // overlay 2
    _pointers.bucketOffsets=overlay;
    _pointers.countsAndBuckets=(uint64_t*)(overlay + ROUND64(buckets));
    _pointers.unsortedCountsAndBuckets=(uint64_t*)(overlay + ROUND64(buckets) + ROUND64(buckets*2));

    // overlay 3
    _pointers.taskOffsets=overlay + ROUND64(buckets) + ROUND64(buckets*2);
    _pointers.unsummedTaskOffsets=_pointers.taskOffsets + ROUND64((buckets+31)/32);
    _pointers.taskData=_pointers.taskOffsets + ROUND64((buckets+31)/32);

    // better to compute cub storage up front
    return 0;
  }

  int32_t buildBucketsAndPoints(uint32_t* aData) {
    kernelBuildBucketsAndPoints<<<_smCount*4, 128, 0, _stream>>>(_params, _pointers, aData);
    return 0;
  }

  int32_t sortBucketsAndPoints() {
    int     points=_params.points();
    void*   noStorage=NULL;
    size_t  requiredCubStorage=0;

    cub::DeviceRadixSort::SortKeys(noStorage, requiredCubStorage, _pointers.unsortedBucketsPoints, _pointers.bucketsPoints, 
                                   points, 32, 32+24, _stream);
    if(requiredCubStorage>_cubStorageSize) {
      if(_cubStorage!=NULL) {
        $CUDA(cudaFree(_cubStorage));
      }
      $CUDA(cudaMalloc(&_cubStorage, requiredCubStorage));
      _cubStorageSize=requiredCubStorage;
    }

    cub::DeviceRadixSort::SortKeys(_cubStorage, requiredCubStorage, _pointers.unsortedBucketsPoints, _pointers.bucketsPoints, 
                                   points, 32, 32+24, _stream);
    return 0;
  }

  int32_t buildCountsAndBuckets() {    
    kernelZeroCounts<<<_smCount*4, 128, 0, _stream>>>(_params, _pointers);
    kernelBuildBucketOffsets<<<_smCount*4, 128, 0, _stream>>>(_params, _pointers);    
    return 0;
  }

  int32_t sortCountsAndBuckets() {
    int     buckets=_params.buckets();
    void*   noStorage=NULL;
    size_t  requiredCubStorage=0;

    cub::DeviceRadixSort::SortKeys(noStorage, requiredCubStorage, _pointers.unsortedCountsAndBuckets, _pointers.countsAndBuckets, buckets, 0, 64, _stream);

    if(requiredCubStorage>_cubStorageSize) {
      if(_cubStorage!=NULL) {
        $CUDA(cudaFree(_cubStorage));
      }
      $CUDA(cudaMalloc(&_cubStorage, requiredCubStorage));
      _cubStorageSize=requiredCubStorage;
    }

    cub::DeviceRadixSort::SortKeys(_cubStorage, requiredCubStorage, _pointers.unsortedCountsAndBuckets, _pointers.countsAndBuckets, buckets, 0, 64, _stream);

    return 0;
  }

  int32_t buildTaskOffsets() {
    kernelBuildWorkCounts<<<_smCount*4, 128, 0, _stream>>>(_params, _pointers);
    kernelBuildTaskOffsets<<<_smCount*4, 128, 0, _stream>>>(_params, _pointers);
    return 0;
  }

  int32_t prefixScanTaskOffsets() {
    int     buckets=_params.buckets();
    void*   noStorage=NULL;
    size_t  requiredCubStorage=0;

    cub::DeviceScan::ExclusiveSum(noStorage, requiredCubStorage, _pointers.unsummedTaskOffsets, _pointers.taskOffsets, (buckets+31)/32, _stream);

    if(requiredCubStorage>_cubStorageSize) {
      if(_cubStorage!=NULL) {
        $CUDA(cudaFree(_cubStorage));
      }
      $CUDA(cudaMalloc(&_cubStorage, requiredCubStorage));
      _cubStorageSize=requiredCubStorage;
    }

    cub::DeviceScan::ExclusiveSum(_cubStorage, requiredCubStorage, _pointers.unsummedTaskOffsets, _pointers.taskOffsets,  (buckets+31)/32, _stream);
    return 0;
  }

  int32_t buildTaskData() {    
    kernelBuildTaskData<<<_smCount*4, 128, 0, _stream>>>(_params, _pointers);
    return 0;
  }

  // aData is a pointer to the scalars
  int32_t runWithDeviceData(MSMParams params, uint32_t* aData) {
    int32_t res;

    
    _params=params;
    if((res=allocate())<0)
      return res;

    if((res=buildBucketsAndPoints(aData))<0)
      return res;
    if((res=sortBucketsAndPoints())<0)
      return res;
    if((res=buildCountsAndBuckets())<0)
      return res;
    if((res=sortCountsAndBuckets())<0)
      return res;
    if((res=buildTaskOffsets())<0)
      return res;
    if((res=prefixScanTaskOffsets())<0)
      return res;
    if((res=buildTaskData())<0)
      return res;
    return 0;
  }

  int32_t runWithHostData(MSMParams params, uint32_t* aData) {
    uint32_t exponentBits=params._exponentBits, pointCount=params._pointCount;
    uint32_t bytes=(exponentBits+31)/32*pointCount*4;
    void*    copy;

    $CUDA(cudaMalloc(&copy, bytes));
    $CUDA(cudaMemcpyAsync(copy, aData, bytes, cudaMemcpyHostToDevice, _stream));
    runWithDeviceData(params, (uint32_t*)copy);
    $CUDA(cudaFree(copy));

    return 0;
  }
};

