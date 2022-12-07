uint32_t random_word() {
  uint32_t random;

  random=rand() & 0xFFFF;
  random=(random<<16) + (rand() & 0xFFFF);
  return random;
}

void zero_words(uint32_t *x, uint32_t count) {
  for(int index=0;index<count;index++)
    x[index]=0;
}

void copy_words(uint32_t* dst, uint32_t* src, uint32_t count) {
  for(int index=0;index<count;index++)
    dst[index]=src[index];
}

void random_words(uint32_t *x, uint32_t count) {
  int index;

  for(index=0;index<count;index++)
    x[index]=random_word();
}

int32_t cudaLastError=0;
#define $CUDA(call) \
  if((cudaLastError=call)!=0) { \
    if (!QUIET) { \
      printf("\nCall \"" #call "\" failed from %s, line %d, error=%d\n", __FILE__, __LINE__, cudaLastError); \
      exit(1); \
    } \
  }
