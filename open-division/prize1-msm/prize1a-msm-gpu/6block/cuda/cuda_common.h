//
// Created by ps on 2021/9/24.
//

#ifndef CUDA_HELPER_CUDA_COMMON_H
#define CUDA_HELPER_CUDA_COMMON_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <exception>
#include <string>
#include <stdexcept>

#define THROW_EXCEPTION(exception_type, ...) {                                               \
    char msg[200];                                                                           \
    sprintf(msg, __VA_ARGS__);                                                               \
    throw exception_type(msg);                                                               \
}

#define THROW_RUNTIME_ERROR( ...) {                                                          \
    THROW_EXCEPTION(std::runtime_error, __VA_ARGS__);                                        \
}

#define THROW_INVALID_ARGS( ...) {                                                           \
    THROW_EXCEPTION(std::invalid_argument, __VA_ARGS__);                                        \
}

#define THROW_PERROR(...) {                                                                  \
    char buf[200];                                                                           \
    sprintf(buf, __VA_ARGS__);                                                               \
    sprintf(buf + strlen(buf), ". Error:%s", strerror(errno));                               \
    throw std::runtime_error(buf) ;                                                          \
}

#define cudaCheckError(expr) {                                                               \
    cudaError e;                                                                             \
    if ((e = expr) != cudaSuccess) {                                                         \
        const char* error_str = cudaGetErrorString(e);                                       \
        throw std::runtime_error(error_str);                                                 \
    }                                                                                        \
}

#define cudaCheckRelease(expr) {                                                             \
    cudaError e;                                                                             \
    if ((e = expr) != cudaSuccess && e != cudaErrorInvalidValue && e != cudaErrorContextIsDestroyed) {\
        const char* error_str = cudaGetErrorString(e);                                       \
        throw std::runtime_error(error_str);                                                 \
    }                                                                                        \
}

#define WARP_SIZE 32

#endif //CUDA_HELPER_CUDA_COMMON_H
