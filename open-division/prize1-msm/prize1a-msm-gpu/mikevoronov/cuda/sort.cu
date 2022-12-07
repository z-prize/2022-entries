#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <map>
#include <cassert>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/system/cuda/vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/pair.h>
#include <thrust/iterator/retag.h>

#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>


#define WBITS 18

#if WBITS==16
typedef unsigned short part_t;
#else
typedef unsigned int part_t;
#endif

struct not_my_pointer
{
  not_my_pointer(void* p) : message()
  {
    std::stringstream s;
    s << "Pointer `" << p << "` was not allocated by this allocator.";
    message = s.str();
  }

  virtual ~not_my_pointer() { }
  virtual const char* what() const { return message.c_str(); }

private:
  std::string message;
};

// A simple allocator for caching cudaMalloc allocations.
struct cached_allocator
{
  typedef char value_type;
  cached_allocator() { }
  ~cached_allocator() { /*free_all();*/ }

  char *allocate(std::ptrdiff_t num_bytes)
  {
	char *result = NULL;

	std::ptrdiff_t found;
	for (auto& elem : free_blocks)
	{
		if (num_bytes <= elem.first)
		{
			result = elem.second;
			found = elem.first;
			free_blocks.erase(elem.first);
			break;
		}
	}
	
	if (result == NULL)
	{
		//printf("allocate %lld\n", num_bytes);
		result = thrust::cuda::malloc<char>(num_bytes).get();
		allocated_blocks.insert(std::make_pair(result, num_bytes));
	}
	else
		allocated_blocks.insert(std::make_pair(result, found));
	
	return result;
  }

  void deallocate(char *ptr, size_t)
  {
    auto iter = allocated_blocks.find(ptr);

    if (iter == allocated_blocks.end())
      throw not_my_pointer(reinterpret_cast<void*>(ptr));

    std::ptrdiff_t num_bytes = iter->second;
    allocated_blocks.erase(iter);
    free_blocks.insert(std::make_pair(num_bytes, ptr));
  }

private:
  typedef std::multimap<std::ptrdiff_t, char*> free_blocks_type;
  typedef std::map<char*, std::ptrdiff_t>      allocated_blocks_type;

  free_blocks_type      free_blocks;
  allocated_blocks_type allocated_blocks;

public:
  void free_all()
  {
    //std::cout << "cached_allocator::free_all()" << std::endl;

	for (auto& elem : free_blocks)
	{
		//printf("free_block %lld with numSize %lld\n", elem.second, elem.first);
		thrust::cuda::free(thrust::cuda::pointer<char>(elem.second));
	}
	
	for (auto& elem : allocated_blocks)
	{
		//printf("allocated_block %lld, with numSize %lld\n", elem.first, elem.second);
		thrust::cuda::free(thrust::cuda::pointer<char>(elem.first));
	}
	
	free_blocks.clear();
	allocated_blocks.clear();
  }
};


//TODO: maybe need to call of free_all function
static std::map<cudaStream_t, cached_allocator> alloc;

#define WITH_ALLOC 1
extern "C" void sort(unsigned int* idxs, part_t* part, size_t npoints, cudaStream_t& st, unsigned* getGroups)
{
	thrust::device_ptr<unsigned int> dev_data_ptr(idxs);
	thrust::device_ptr<part_t> dev_keys_ptr(part);
#if WITH_ALLOC	
	auto exec = thrust::cuda::par(alloc[st]).on(st);
#else
	auto exec thrust::cuda::par.on(st);
#endif
	if (getGroups)
	{
		unsigned res = thrust::reduce(exec, dev_keys_ptr, dev_keys_ptr + npoints, 0, thrust::maximum<unsigned>());
		res++;
		//printf(" *** groups count %u\n", res);
		getGroups[0] = res;
	}
	thrust::sort_by_key(exec, dev_keys_ptr, dev_keys_ptr + npoints, dev_data_ptr);
}

extern "C" void sort1(unsigned int* idxs, unsigned int* sizes, size_t npoints, cudaStream_t& st, bool greater)
{
	thrust::device_ptr<unsigned int> dev_data_ptr(idxs);
	thrust::device_ptr<unsigned int> dev_keys_ptr(sizes);

#if WITH_ALLOC	
	auto exec = thrust::cuda::par(alloc[st]).on(st);
#else
	auto exec thrust::cuda::par.on(st);
#endif

	if (greater)
		thrust::sort_by_key(exec, dev_keys_ptr, dev_keys_ptr + npoints, dev_data_ptr, thrust::greater<unsigned>());
	else
		thrust::sort_by_key(exec, dev_keys_ptr, dev_keys_ptr + npoints, dev_data_ptr);
}

extern "C" void scan(unsigned int* rows, size_t npoints, cudaStream_t& st)
{	
	thrust::device_ptr<unsigned int> dev_data_ptr(rows);
#if WITH_ALLOC	
	auto exec = thrust::cuda::par(alloc[st]).on(st);
#else
	auto exec thrust::cuda::par.on(st);
#endif

    thrust::inclusive_scan(exec, dev_data_ptr, dev_data_ptr + npoints, dev_data_ptr);
}

extern "C" void clearCache()
{
	for (auto& elem : alloc)
		elem.second.free_all();
}
