//
// Created by ps on 22-7-6.
//

#ifndef CUDA_HELPER_CUDAEVENTWRAPPER_H
#define CUDA_HELPER_CUDAEVENTWRAPPER_H
#include <cuda_runtime.h>
namespace cuda_helper {
    class CudaEventWrapper {
    public:
        CudaEventWrapper();

        CudaEventWrapper(unsigned int flags);

        ~CudaEventWrapper();

        CudaEventWrapper(CudaEventWrapper &&other);

        CudaEventWrapper &operator=(CudaEventWrapper &&other);

        cudaEvent_t &get();

        /**
         * Release ownership of cuda event.
         * @return
         */
        cudaEvent_t release();

    private:
        cudaEvent_t event;
        unsigned int flags;
    };
}
#endif //CUDA_HELPER_CUDAEVENTWRAPPER_H
