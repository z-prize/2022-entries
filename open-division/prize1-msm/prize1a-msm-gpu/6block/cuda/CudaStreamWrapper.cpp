//
// Created by hulei on 22-8-10.
//
#include "cuda_common.h"
#include "CudaStreamWrapper.h"
namespace cuda_helper {

    CudaStreamWrapper::CudaStreamWrapper()
            : stream(nullptr),
              flags(cudaStreamDefault){

    }

    CudaStreamWrapper::CudaStreamWrapper(unsigned int flags)
            : stream(nullptr),
              flags(flags) {

    }

    CudaStreamWrapper::~CudaStreamWrapper() {
        if (stream != nullptr) {
            try {
                cudaCheckRelease(cudaStreamDestroy(stream));
            } catch (...) {
                //Do nothing.
            }
        }
    }

    CudaStreamWrapper::CudaStreamWrapper(CudaStreamWrapper &&other) {
        stream = other.stream;
        other.stream = nullptr;
    }

    CudaStreamWrapper &CudaStreamWrapper::operator=(CudaStreamWrapper &&other) {
        stream = other.stream;
        other.stream = nullptr;
        return *this;
    }

    cudaStream_t &CudaStreamWrapper::get() {
        if (stream == nullptr) {
            cudaCheckError(cudaStreamCreateWithFlags(&stream, flags));
        }
        return stream;
    }

    cudaStream_t CudaStreamWrapper::release() {
        cudaStream_t ret = stream;
        stream = nullptr;
        return ret;
    }
}