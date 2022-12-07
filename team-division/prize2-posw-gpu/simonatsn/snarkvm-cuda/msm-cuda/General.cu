__device__ __host__ __forceinline__ void* byteOffset(void* address, uint32_t byteOffset) {
  uint8_t* ptr8=(uint8_t*)address;
  
  return (void*)(ptr8+byteOffset);
}

__device__ __forceinline__ void* byteOffset(void* address, uint32_t index, uint32_t bytes) {
  uint64_t ptr8=madwide(index, bytes, (uint64_t)address);
  
  return (void*)ptr8;
}
