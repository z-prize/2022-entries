//
// Created by ps on 2021/12/2.
//

#ifndef CUDA_HELPER_CUDADEVICEMEMWRAPPER_H
#define CUDA_HELPER_CUDADEVICEMEMWRAPPER_H

#include <cstdint>
#include <cuda_runtime.h>
#include "cuda_common.h"
namespace cuda_helper {
    template<typename E>
    class CudaDeviceMemWrapper {
    public:
        CudaDeviceMemWrapper()
                : d_ptr(nullptr),
                  element_num(0) {

        }

        CudaDeviceMemWrapper(uint32_t element_num)
                : d_ptr(nullptr),
                  element_num(element_num) {

        }

        ~CudaDeviceMemWrapper() {
            try{
                clear();
            } catch (...) {

            }
        }

        E *ptr() {
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
            if (d_ptr != nullptr) {
                cudaCheckRelease(cudaFree(d_ptr));
                d_ptr = nullptr;
            }
            element_num = 0;
        }
    protected:
        void malloc_if_needed(){
            if (d_ptr == nullptr && element_num > 0) {
                cudaCheckError(cudaMalloc(&d_ptr, size_in_bytes()));
            }
        }
    private:
        E *d_ptr;
        uint32_t element_num;
    };
}

#endif //CUDA_HELPER_CUDADEVICEMEMWRAPPER_H
