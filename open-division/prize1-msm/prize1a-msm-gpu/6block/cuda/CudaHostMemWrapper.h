//
// Created by hulei on 22-8-10.
//

#ifndef CUDA_HELPER_CUDAHOSTMEMWRAPPER_H
#define CUDA_HELPER_CUDAHOSTMEMWRAPPER_H
#include <cstdint>
#include <cuda_runtime.h>
#include "cuda_common.h"

namespace cuda_helper {
    template<typename E>
    class CudaHostMemWrapper {
    public:
        CudaHostMemWrapper()
                : d_ptr(nullptr),
                  h_ptr(nullptr),
                  element_num(0),
                  flags(cudaHostAllocDefault){

        }

        CudaHostMemWrapper(uint32_t element_num)
                : d_ptr(nullptr),
                  h_ptr(nullptr),
                  element_num(element_num),
                  flags(cudaHostAllocDefault){

        }

        CudaHostMemWrapper(uint32_t element_num, unsigned int flags)
                : d_ptr(nullptr),
                  h_ptr(nullptr),
                  element_num(element_num),
                  flags(flags){

        }

        ~CudaHostMemWrapper() {
            try{
                clear();
            } catch (...) {

            }
        }

        E *hptr() {
            malloc_if_needed();
            return h_ptr;
        }

        E *dptr() {
            malloc_if_needed();
            return d_ptr;
        }

        size_t size_in_bytes() {
            return (size_t)element_num * sizeof(E);
        }

        uint32_t num() {
            return element_num;
        }

        void resize(uint32_t element_num) {
            clear();
            this->element_num = element_num;
            malloc_if_needed();
        }

        void clear() {
            if (h_ptr != nullptr) {
                cudaCheckRelease(cudaFreeHost(h_ptr));
            }
            h_ptr = d_ptr = nullptr;
            element_num = 0;
        }
    protected:
        void malloc_if_needed(){
            if (h_ptr == nullptr && element_num > 0) {
                cudaCheckError(cudaHostAlloc(&h_ptr, size_in_bytes(), flags));
                if ((cudaHostAllocMapped & flags) != 0) {
                    cudaCheckError(cudaHostGetDevicePointer(&d_ptr, h_ptr, 0));
                } else {
                    d_ptr = nullptr;
                }
            }
        }
    private:
        E *d_ptr;
        E *h_ptr;
        uint32_t element_num;
        uint32_t flags;
    };
}


#endif //CUDA_HELPER_CUDAHOSTMEMWRAPPER_H
