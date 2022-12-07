//
// Created by hulei on 22-8-10.
//

#ifndef CUDA_HELPER_CUDASTREAMWRAPPER_H
#define CUDA_HELPER_CUDASTREAMWRAPPER_H
#include <cuda_runtime.h>
namespace cuda_helper {
    class CudaStreamWrapper {
    public:
        ~CudaStreamWrapper();

        CudaStreamWrapper();

        CudaStreamWrapper(unsigned int flags);

        CudaStreamWrapper(CudaStreamWrapper &&other);

        CudaStreamWrapper &operator=(CudaStreamWrapper &&other);

        cudaStream_t &get();

        /**
         * Release ownership of cuda stream.
         * @return
         */
        cudaStream_t release();

    private:
        cudaStream_t stream;
        unsigned int flags;
    };
}
#endif //CUDA_HELPER_CUDASTREAMWRAPPER_H
