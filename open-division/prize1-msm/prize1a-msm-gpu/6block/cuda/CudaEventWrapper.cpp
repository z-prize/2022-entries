//
// Created by hulei on 9/9/21.
//

#include "cuda_common.h"
#include "CudaEventWrapper.h"
namespace cuda_helper {

    CudaEventWrapper::CudaEventWrapper()
            : event(nullptr),
              flags(cudaEventDefault){

    }

    CudaEventWrapper::CudaEventWrapper(unsigned int _flags)
            : event(nullptr),
              flags(_flags) {

    }

    CudaEventWrapper::~CudaEventWrapper() {
        if (event != nullptr) {
            try {
                cudaCheckRelease(cudaEventDestroy(event));
            } catch (...) {
                //Do nothing.
            }
        }
    }

    CudaEventWrapper::CudaEventWrapper(CudaEventWrapper &&other) {
        event = other.event;
        other.event = nullptr;
    }

    CudaEventWrapper &CudaEventWrapper::operator=(CudaEventWrapper &&other) {
        event = other.event;
        other.event = nullptr;
        return *this;
    }

    cudaEvent_t &CudaEventWrapper::get() {
        if (event == nullptr) {
            cudaCheckError(cudaEventCreate(&event, flags));
        }
        return event;
    }

    cudaEvent_t CudaEventWrapper::release() {
        cudaEvent_t ret = event;
        event = nullptr;
        return ret;
    }
}