#include <string.h>
#include <stdio.h>
#include <stdint.h>

#include "blst_ops.h"

static const uint32_t MAX_LOG2_RADIX = 9;
static const uint32_t WINDOW_SIZE = 1 << MAX_LOG2_RADIX;

__device__ uint32_t bitreverse(uint32_t n, uint32_t bits) {
    uint32_t r = 0;
    for (int i = 0; i < bits; i++) {
        r = (r << 1) | (n & 1);
        n >>= 1;
    }
    return r;
}

extern "C" __global__ void distribute_powers(blst_fr *x,
                                             blst_fr *y,
                                             blst_fr *gen,
                                             uint32_t n,
                                             uint32_t num_width) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = num_width * index;
    uint32_t end = min(start + num_width, n);
    blst_fr offset;

    blst_fr_pow(&offset, gen[0], index * num_width);
    for (uint i = start; i < end - 1; i++) {
        blst_fr_mul(y[i], x[i], offset);
        blst_fr_mul(offset, gen[0], offset);
    }
    blst_fr_mul(y[end - 1], x[end - 1], offset);
}

extern "C" __global__ void distribute_powers_2(blst_fr *x1, blst_fr *x2,
                                               blst_fr *y1, blst_fr *y2,
                                               blst_fr *gen,
                                               uint32_t n,
                                               uint32_t num_width) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = num_width * index;
    uint32_t end = min(start + num_width, n);
    blst_fr offset;

    blst_fr_pow(&offset, gen[0], index * num_width);
    for (uint i = start; i < end - 1; i++) {
        blst_fr_mul(y1[i], x1[i], offset);
        blst_fr_mul(y2[i], x2[i], offset);
        blst_fr_mul(offset, gen[0], offset);
    }
    blst_fr_mul(y1[end - 1], x1[end - 1], offset);
    blst_fr_mul(y2[end - 1], x2[end - 1], offset);
}

extern "C" __global__ void distribute_powers_3(blst_fr *x1, blst_fr *x2, blst_fr *x3,
                                               blst_fr *y1, blst_fr *y2, blst_fr *y3,
                                               blst_fr *gen,
                                               uint32_t n,
                                               uint32_t num_width) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = num_width * index;
    uint32_t end = min(start + num_width, n);
    blst_fr offset;

    blst_fr_pow(&offset, gen[0], index * num_width);
    for (uint i = start; i < end - 1; i++) {
        blst_fr_mul(y1[i], x1[i], offset);
        blst_fr_mul(y2[i], x2[i], offset);
        blst_fr_mul(y3[i], x3[i], offset);
        blst_fr_mul(offset, gen[0], offset);
    }
    blst_fr_mul(y1[end - 1], x1[end - 1], offset);
    blst_fr_mul(y2[end - 1], x2[end - 1], offset);
    blst_fr_mul(y3[end - 1], x3[end - 1], offset);
}

extern "C" __global__ void distribute_powers_9(blst_fr *x1, blst_fr *x2, blst_fr *x3,
                                               blst_fr *x4, blst_fr *x5, blst_fr *x6,
                                               blst_fr *x7, blst_fr *x8, blst_fr *x9,
                                               blst_fr *y1, blst_fr *y2, blst_fr *y3,
                                               blst_fr *y4, blst_fr *y5, blst_fr *y6,
                                               blst_fr *y7, blst_fr *y8, blst_fr *y9,
                                               blst_fr *gen,
                                               uint32_t n,
                                               uint32_t num_width) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = num_width * index;
    uint32_t end = min(start + num_width, n);
    blst_fr offset;

    blst_fr_pow(&offset, gen[0], index * num_width);
    for (uint i = start; i < end - 1; i++) {
        blst_fr_mul(y1[i], x1[i], offset);
        blst_fr_mul(y2[i], x2[i], offset);
        blst_fr_mul(y3[i], x3[i], offset);
        blst_fr_mul(y4[i], x4[i], offset);
        blst_fr_mul(y5[i], x5[i], offset);
        blst_fr_mul(y6[i], x6[i], offset);
        blst_fr_mul(y7[i], x7[i], offset);
        blst_fr_mul(y8[i], x8[i], offset);
        blst_fr_mul(y9[i], x9[i], offset);
        blst_fr_mul(offset, gen[0], offset);
    }
    blst_fr_mul(y1[end - 1], x1[end - 1], offset);
    blst_fr_mul(y2[end - 1], x2[end - 1], offset);
    blst_fr_mul(y3[end - 1], x3[end - 1], offset);
    blst_fr_mul(y4[end - 1], x4[end - 1], offset);
    blst_fr_mul(y5[end - 1], x5[end - 1], offset);
    blst_fr_mul(y6[end - 1], x6[end - 1], offset);
    blst_fr_mul(y7[end - 1], x7[end - 1], offset);
    blst_fr_mul(y8[end - 1], x8[end - 1], offset);
    blst_fr_mul(y9[end - 1], x9[end - 1], offset);
}

extern "C" __global__ void compute_mul_size_inv(blst_fr *x,
                                                blst_fr *y,
                                                blst_fr *size_inv,
                                                uint32_t n,
                                                uint32_t num_width) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = num_width * index;
    uint32_t end = min(start + num_width, n);

    for (uint i = start; i < end; i++) {
        blst_fr_mul(y[i], x[i], size_inv[0]);
    }
}

extern "C" __global__ void compute_mul_size_inv_2(blst_fr *x1, blst_fr *x2,
                                                  blst_fr *y1, blst_fr *y2,
                                                  blst_fr *size_inv,
                                                  uint32_t n,
                                                  uint32_t num_width) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = num_width * index;
    uint32_t end = min(start + num_width, n);

    for (uint i = start; i < end; i++) {
        blst_fr_mul(y1[i], x1[i], size_inv[0]);
        blst_fr_mul(y2[i], x2[i], size_inv[0]);
    }
}

extern "C" __global__ void compute_mul_size_inv_3(blst_fr *x1, blst_fr *x2, blst_fr *x3,
                                                  blst_fr *y1, blst_fr *y2, blst_fr *y3,
                                                  blst_fr *size_inv,
                                                  uint32_t n,
                                                  uint32_t num_width) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = num_width * index;
    uint32_t end = min(start + num_width, n);

    for (uint i = start; i < end; i++) {
        blst_fr_mul(y1[i], x1[i], size_inv[0]);
        blst_fr_mul(y2[i], x2[i], size_inv[0]);
        blst_fr_mul(y3[i], x3[i], size_inv[0]);
    }
}

extern "C" __global__ void
compute_mul_size_inv_6(blst_fr *x1, blst_fr *x2, blst_fr *x3, blst_fr *x4, blst_fr *x5, blst_fr *x6,
                       blst_fr *y1, blst_fr *y2, blst_fr *y3, blst_fr *y4, blst_fr *y5, blst_fr *y6,
                       blst_fr *size_inv,
                       uint32_t n,
                       uint32_t num_width) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = num_width * index;
    uint32_t end = min(start + num_width, n);

    for (uint i = start; i < end; i++) {
        blst_fr_mul(y1[i], x1[i], size_inv[0]);
        blst_fr_mul(y2[i], x2[i], size_inv[0]);
        blst_fr_mul(y3[i], x3[i], size_inv[0]);
        blst_fr_mul(y4[i], x4[i], size_inv[0]);
        blst_fr_mul(y5[i], x5[i], size_inv[0]);
        blst_fr_mul(y6[i], x6[i], size_inv[0]);
    }
}

extern "C" __global__ void radix_fft_full(blst_fr *x,    // Source buffer
                                          blst_fr *y,    // Destination buffer
                                          blst_fr *pq,    // Precalculated twiddle factors
                                          blst_fr *omegas,
                                          uint32_t n,    // Number of elements
                                          uint32_t lgp,    // Log2 of `p` (Read more in the link above)
                                          uint32_t deg,    // 1=>radix2, 2=>radix4, 3=>radix8, ...
                                          uint32_t max_deg)    // Maximum degree supported, according to `pq` and `omegas`
{
    uint32_t lid = threadIdx.x;   // 线程id
    uint32_t lsize = blockDim.x;
    uint32_t index = blockIdx.x;
    uint32_t t = n >> deg;
    uint32_t p = 1 << lgp;
    uint32_t k = index & (p - 1);
    __shared__ blst_fr u[WINDOW_SIZE];

    x += index;
    y += ((index - k) << deg) + k;

    uint32_t count = 1 << deg;    // 2^deg
    uint32_t counth = count >> 1;    // Half of count

    uint32_t counts = count / lsize * lid;
    uint32_t counte = counts + count / lsize;

    blst_fr tmp;
    const uint32_t bit = counth;
    const uint32_t pqshift = max_deg - deg;
    for (uint32_t i = counts >> 1; i < counte >> 1; i++) {
        const uint32_t di = i & (bit - 1);
        const uint32_t i0 = (i << 1) - di;
        const uint32_t i1 = i0 + bit;
        blst_fr_mul(u[i0], omegas[(n >> lgp >> deg) * k * i0], x[i0 * t]);
        blst_fr_mul(u[i1], omegas[(n >> lgp >> deg) * k * i1], x[i1 * t]);

        memcpy(tmp, u[i0], sizeof(blst_fr));

        blst_fr_add(u[i0], u[i0], u[i1]);
        blst_fr_sub(u[i1], tmp, u[i1]);

        if (di != 0)
            blst_fr_mul(u[i1], pq[di << pqshift], u[i1]);
    }

    __syncthreads();


    for (uint32_t rnd = 1; rnd < deg - 1; rnd++) {

        const uint32_t bit = counth >> rnd;
        for (uint32_t i = counts >> 1; i < counte >> 1; i++) {
            const uint32_t di = i & (bit - 1);
            const uint32_t i0 = (i << 1) - di;
            const uint32_t i1 = i0 + bit;

            memcpy(tmp, u[i0], sizeof(blst_fr));

            blst_fr_add(u[i0], u[i0], u[i1]);
            blst_fr_sub(u[i1], tmp, u[i1]);

            if (di != 0)
                blst_fr_mul(u[i1], pq[di << rnd << pqshift], u[i1]);
        }

        __syncthreads();
    }

    const uint32_t bit0 = counth >> (deg - 1);
    for (uint32_t i = counts >> 1; i < counte >> 1; i++) {
        const uint32_t di = i & (bit0 - 1);
        const uint32_t i0 = (i << 1) - di;
        const uint32_t i1 = i0 + bit0;

        blst_fr_add(y[bitreverse(i0, deg) * p], u[i0], u[i1]);
        blst_fr_sub(y[bitreverse(i1, deg) * p], u[i0], u[i1]);
    }
}

extern "C" __global__ void radix_fft_2_full(blst_fr *x1, blst_fr *x2,
                                            blst_fr *y1, blst_fr *y2,
                                            blst_fr *pq,
                                            blst_fr *omegas,
                                            uint32_t n,
                                            uint32_t lgp,
                                            uint32_t deg,
                                            uint32_t max_deg) 
{
    uint32_t lid = threadIdx.x;   // 线程id
    uint32_t lsize = blockDim.x;
    uint32_t index = blockIdx.x / 2;
    uint32_t index0 = blockIdx.x % 2;
    uint32_t t = n >> deg;
    uint32_t p = 1 << lgp;
    uint32_t k = index & (p - 1);

    __shared__ blst_fr u[WINDOW_SIZE];

    uint32_t count = 1 << deg;    // 2^deg
    uint32_t counth = count >> 1;    // Half of count

    uint32_t counts = count / lsize * lid;
    uint32_t counte = counts + count / lsize;

    blst_fr tmp;
    const uint32_t bit = counth;
    const uint32_t pqshift = max_deg - deg;
    for (uint32_t i = counts >> 1; i < counte >> 1; i++) {
        const uint32_t di = i & (bit - 1);
        const uint32_t i0 = (i << 1) - di;
        const uint32_t i1 = i0 + bit;
        if (index0 == 0) {
            blst_fr_mul(u[i0], omegas[(n >> lgp >> deg) * k * i0], x1[i0 * t + index]);
            blst_fr_mul(u[i1], omegas[(n >> lgp >> deg) * k * i1], x1[i1 * t + index]);
        } else if (index0 == 1) {
            blst_fr_mul(u[i0], omegas[(n >> lgp >> deg) * k * i0], x2[i0 * t + index]);
            blst_fr_mul(u[i1], omegas[(n >> lgp >> deg) * k * i1], x2[i1 * t + index]);
        }
        memcpy(tmp, u[i0], sizeof(blst_fr));

        blst_fr_add(u[i0], u[i0], u[i1]);
        blst_fr_sub(u[i1], tmp, u[i1]);

        if (di != 0)
            blst_fr_mul(u[i1], pq[di << pqshift], u[i1]);
    }

    __syncthreads();

    for (uint32_t rnd = 1; rnd < deg - 1; rnd++) {
        const uint32_t bit = counth >> rnd;
        for (uint32_t i = counts >> 1; i < counte >> 1; i++) {
            const uint32_t di = i & (bit - 1);
            const uint32_t i0 = (i << 1) - di;
            const uint32_t i1 = i0 + bit;

            memcpy(tmp, u[i0], sizeof(blst_fr));

            blst_fr_add(u[i0], u[i0], u[i1]);
            blst_fr_sub(u[i1], tmp, u[i1]);

            if (di != 0)
                blst_fr_mul(u[i1], pq[di << rnd << pqshift], u[i1]);
        }
        __syncthreads();
    }
    const uint32_t bit0 = counth >> (deg - 1);
    for (uint32_t i = counts >> 1; i < counte >> 1; i++) {
        const uint32_t di = i & (bit0 - 1);
        const uint32_t i0 = (i << 1) - di;
        const uint32_t i1 = i0 + bit0;
        if (index0 == 0) {
            blst_fr_add(y1[bitreverse(i0, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
            blst_fr_sub(y1[bitreverse(i1, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
        } else if (index0 == 1) {
            blst_fr_add(y2[bitreverse(i0, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
            blst_fr_sub(y2[bitreverse(i1, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
        }
    }
}

extern "C" __global__ void radix_fft_3_full(blst_fr *x1, blst_fr *x2, blst_fr *x3,
                                            blst_fr *y1, blst_fr *y2, blst_fr *y3,
                                            blst_fr *pq,
                                            blst_fr *omegas,
                                            uint32_t n,
                                            uint32_t lgp,
                                            uint32_t deg,
                                            uint32_t max_deg) 
{
    uint32_t lid = threadIdx.x;
    uint32_t lsize = blockDim.x;
    uint32_t index = blockIdx.x / 3;
    uint32_t index0 = blockIdx.x % 3;
    uint32_t t = n >> deg;
    uint32_t p = 1 << lgp;
    uint32_t k = index & (p - 1);

    __shared__ blst_fr u[WINDOW_SIZE];

    uint32_t count = 1 << deg;    // 2^deg
    uint32_t counth = count >> 1;    // Half of count

    uint32_t counts = count / lsize * lid;
    uint32_t counte = counts + count / lsize;

    blst_fr tmp;
    const uint32_t bit = counth;
    const uint32_t pqshift = max_deg - deg;
    for (uint32_t i = counts >> 1; i < counte >> 1; i++) {
        const uint32_t di = i & (bit - 1);
        const uint32_t i0 = (i << 1) - di;
        const uint32_t i1 = i0 + bit;
        if (index0 == 0) {
            blst_fr_mul(u[i0], omegas[(n >> lgp >> deg) * k * i0], x1[i0 * t + index]);
            blst_fr_mul(u[i1], omegas[(n >> lgp >> deg) * k * i1], x1[i1 * t + index]);
        } else if (index0 == 1) {
            blst_fr_mul(u[i0], omegas[(n >> lgp >> deg) * k * i0], x2[i0 * t + index]);
            blst_fr_mul(u[i1], omegas[(n >> lgp >> deg) * k * i1], x2[i1 * t + index]);
        } else {
            blst_fr_mul(u[i0], omegas[(n >> lgp >> deg) * k * i0], x3[i0 * t + index]);
            blst_fr_mul(u[i1], omegas[(n >> lgp >> deg) * k * i1], x3[i1 * t + index]);
        }
        memcpy(tmp, u[i0], sizeof(blst_fr));

        blst_fr_add(u[i0], u[i0], u[i1]);
        blst_fr_sub(u[i1], tmp, u[i1]);

        if (di != 0)
            blst_fr_mul(u[i1], pq[di << pqshift], u[i1]);
    }
    __syncthreads();

    for (uint32_t rnd = 1; rnd < deg - 1; rnd++) {
        const uint32_t bit = counth >> rnd;
        for (uint32_t i = counts >> 1; i < counte >> 1; i++) {
            const uint32_t di = i & (bit - 1);
            const uint32_t i0 = (i << 1) - di;
            const uint32_t i1 = i0 + bit;

            memcpy(tmp, u[i0], sizeof(blst_fr));

            blst_fr_add(u[i0], u[i0], u[i1]);
            blst_fr_sub(u[i1], tmp, u[i1]);

            if (di != 0)
                blst_fr_mul(u[i1], pq[di << rnd << pqshift], u[i1]);
        }
        __syncthreads();
    }
    const uint32_t bit0 = counth >> (deg - 1);
    for (uint32_t i = counts >> 1; i < counte >> 1; i++) {
        const uint32_t di = i & (bit0 - 1);
        const uint32_t i0 = (i << 1) - di;
        const uint32_t i1 = i0 + bit0;
        if (index0 == 0) {
            blst_fr_add(y1[bitreverse(i0, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
            blst_fr_sub(y1[bitreverse(i1, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
        } else if (index0 == 1) {
            blst_fr_add(y2[bitreverse(i0, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
            blst_fr_sub(y2[bitreverse(i1, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
        } else {
            blst_fr_add(y3[bitreverse(i0, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
            blst_fr_sub(y3[bitreverse(i1, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
        }
    }
}

extern "C" __global__ void radix_fft_6_full(blst_fr *x1, blst_fr *x2, blst_fr *x3,
                                            blst_fr *x4, blst_fr *x5, blst_fr *x6,
                                            blst_fr *y1, blst_fr *y2, blst_fr *y3,
                                            blst_fr *y4, blst_fr *y5, blst_fr *y6,

                                            blst_fr *pq,
                                            blst_fr *omegas,
                                            uint32_t n,
                                            uint32_t lgp,
                                            uint32_t deg,
                                            uint32_t max_deg) 
{
    uint32_t lid = threadIdx.x;
    uint32_t lsize = blockDim.x;
    uint32_t index = blockIdx.x / 6;
    uint32_t index0 = blockIdx.x % 6;
    uint32_t t = n >> deg;
    uint32_t p = 1 << lgp;
    uint32_t k = index & (p - 1);

    __shared__ blst_fr u[WINDOW_SIZE];   

    uint32_t count = 1 << deg;    // 2^deg
    uint32_t counth = count >> 1;    // Half of count

    uint32_t counts = count / lsize * lid;
    uint32_t counte = counts + count / lsize;

    blst_fr tmp;
    const uint32_t bit = counth;
    const uint32_t pqshift = max_deg - deg;
    for (uint32_t i = counts >> 1; i < counte >> 1; i++) {
        const uint32_t di = i & (bit - 1);
        const uint32_t i0 = (i << 1) - di;
        const uint32_t i1 = i0 + bit;
        if (index0 == 0) {
            blst_fr_mul(u[i0], omegas[(n >> lgp >> deg) * k * i0], x1[i0 * t + index]);
            blst_fr_mul(u[i1], omegas[(n >> lgp >> deg) * k * i1], x1[i1 * t + index]);
        } else if (index0 == 1) {
            blst_fr_mul(u[i0], omegas[(n >> lgp >> deg) * k * i0], x2[i0 * t + index]);
            blst_fr_mul(u[i1], omegas[(n >> lgp >> deg) * k * i1], x2[i1 * t + index]);
        } else if (index0 == 2) {
            blst_fr_mul(u[i0], omegas[(n >> lgp >> deg) * k * i0], x3[i0 * t + index]);
            blst_fr_mul(u[i1], omegas[(n >> lgp >> deg) * k * i1], x3[i1 * t + index]);
        } else if (index0 == 3) {
            blst_fr_mul(u[i0], omegas[(n >> lgp >> deg) * k * i0], x4[i0 * t + index]);
            blst_fr_mul(u[i1], omegas[(n >> lgp >> deg) * k * i1], x4[i1 * t + index]);
        } else if (index0 == 4) {
            blst_fr_mul(u[i0], omegas[(n >> lgp >> deg) * k * i0], x5[i0 * t + index]);
            blst_fr_mul(u[i1], omegas[(n >> lgp >> deg) * k * i1], x5[i1 * t + index]);
        } else if (index0 == 5) {
            blst_fr_mul(u[i0], omegas[(n >> lgp >> deg) * k * i0], x6[i0 * t + index]);
            blst_fr_mul(u[i1], omegas[(n >> lgp >> deg) * k * i1], x6[i1 * t + index]);
        }

        memcpy(tmp, u[i0], sizeof(blst_fr));

        blst_fr_add(u[i0], u[i0], u[i1]);
        blst_fr_sub(u[i1], tmp, u[i1]);

        if (di != 0)
            blst_fr_mul(u[i1], pq[di << pqshift], u[i1]);
    }

    __syncthreads();

    for (uint32_t rnd = 1; rnd < deg - 1; rnd++) {
        const uint32_t bit = counth >> rnd;
        for (uint32_t i = counts >> 1; i < counte >> 1; i++) {
            const uint32_t di = i & (bit - 1);
            const uint32_t i0 = (i << 1) - di;
            const uint32_t i1 = i0 + bit;

            memcpy(tmp, u[i0], sizeof(blst_fr));

            blst_fr_add(u[i0], u[i0], u[i1]);
            blst_fr_sub(u[i1], tmp, u[i1]);

            if (di != 0)
                blst_fr_mul(u[i1], pq[di << rnd << pqshift], u[i1]);
        }

        __syncthreads();
    }
    const uint32_t bit0 = counth >> (deg - 1);
    for (uint32_t i = counts >> 1; i < counte >> 1; i++) {
        const uint32_t di = i & (bit0 - 1);
        const uint32_t i0 = (i << 1) - di;
        const uint32_t i1 = i0 + bit0;
        if (index0 == 0) {
            blst_fr_add(y1[bitreverse(i0, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
            blst_fr_sub(y1[bitreverse(i1, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
        } else if (index0 == 1) {
            blst_fr_add(y2[bitreverse(i0, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
            blst_fr_sub(y2[bitreverse(i1, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
        } else if (index0 == 2) {
            blst_fr_add(y3[bitreverse(i0, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
            blst_fr_sub(y3[bitreverse(i1, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
        } else if (index0 == 3) {
            blst_fr_add(y4[bitreverse(i0, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
            blst_fr_sub(y4[bitreverse(i1, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
        } else if (index0 == 4) {
            blst_fr_add(y5[bitreverse(i0, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
            blst_fr_sub(y5[bitreverse(i1, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
        } else if (index0 == 5) {
            blst_fr_add(y6[bitreverse(i0, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
            blst_fr_sub(y6[bitreverse(i1, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
        }
    }
}

extern "C" __global__ void radix_fft_9_full(blst_fr *x1, blst_fr *x2, blst_fr *x3,
                                            blst_fr *x4, blst_fr *x5, blst_fr *x6,
                                            blst_fr *x7, blst_fr *x8, blst_fr *x9,
                                            blst_fr *y1, blst_fr *y2, blst_fr *y3,
                                            blst_fr *y4, blst_fr *y5, blst_fr *y6,
                                            blst_fr *y7, blst_fr *y8, blst_fr *y9,
                                            blst_fr *pq,
                                            blst_fr *omegas,
                                            uint32_t n,
                                            uint32_t lgp,
                                            uint32_t deg,
                                            uint32_t max_deg) {
    uint32_t lid = threadIdx.x;   
    uint32_t lsize = blockDim.x;
    uint32_t index = blockIdx.x / 9;
    uint32_t index0 = blockIdx.x % 9;
    uint32_t t = n >> deg;
    uint32_t p = 1 << lgp;
    uint32_t k = index & (p - 1);

    __shared__ blst_fr u[WINDOW_SIZE];   

    uint32_t count = 1 << deg;    // 2^deg
    uint32_t counth = count >> 1;    // Half of count

    uint32_t counts = count / lsize * lid;
    uint32_t counte = counts + count / lsize;   

    blst_fr tmp;
    const uint32_t bit = counth;
    const uint32_t pqshift = max_deg - deg;
    for (uint32_t i = counts >> 1; i < counte >> 1; i++) {
        const uint32_t di = i & (bit - 1);
        const uint32_t i0 = (i << 1) - di;
        const uint32_t i1 = i0 + bit;
        if (index0 == 0) {
            blst_fr_mul(u[i0], omegas[(n >> lgp >> deg) * k * i0], x1[i0 * t + index]);
            blst_fr_mul(u[i1], omegas[(n >> lgp >> deg) * k * i1], x1[i1 * t + index]);
        } else if (index0 == 1) {
            blst_fr_mul(u[i0], omegas[(n >> lgp >> deg) * k * i0], x2[i0 * t + index]);
            blst_fr_mul(u[i1], omegas[(n >> lgp >> deg) * k * i1], x2[i1 * t + index]);
        } else if (index0 == 2) {
            blst_fr_mul(u[i0], omegas[(n >> lgp >> deg) * k * i0], x3[i0 * t + index]);
            blst_fr_mul(u[i1], omegas[(n >> lgp >> deg) * k * i1], x3[i1 * t + index]);
        } else if (index0 == 3) {
            blst_fr_mul(u[i0], omegas[(n >> lgp >> deg) * k * i0], x4[i0 * t + index]);
            blst_fr_mul(u[i1], omegas[(n >> lgp >> deg) * k * i1], x4[i1 * t + index]);
        } else if (index0 == 4) {
            blst_fr_mul(u[i0], omegas[(n >> lgp >> deg) * k * i0], x5[i0 * t + index]);
            blst_fr_mul(u[i1], omegas[(n >> lgp >> deg) * k * i1], x5[i1 * t + index]);
        } else if (index0 == 5) {
            blst_fr_mul(u[i0], omegas[(n >> lgp >> deg) * k * i0], x6[i0 * t + index]);
            blst_fr_mul(u[i1], omegas[(n >> lgp >> deg) * k * i1], x6[i1 * t + index]);
        } else if (index0 == 6) {
            blst_fr_mul(u[i0], omegas[(n >> lgp >> deg) * k * i0], x7[i0 * t + index]);
            blst_fr_mul(u[i1], omegas[(n >> lgp >> deg) * k * i1], x7[i1 * t + index]);
        } else if (index0 == 7) {
            blst_fr_mul(u[i0], omegas[(n >> lgp >> deg) * k * i0], x8[i0 * t + index]);
            blst_fr_mul(u[i1], omegas[(n >> lgp >> deg) * k * i1], x8[i1 * t + index]);
        } else if (index0 == 8) {
            blst_fr_mul(u[i0], omegas[(n >> lgp >> deg) * k * i0], x9[i0 * t + index]);
            blst_fr_mul(u[i1], omegas[(n >> lgp >> deg) * k * i1], x9[i1 * t + index]);
        }

        memcpy(tmp, u[i0], sizeof(blst_fr));

        blst_fr_add(u[i0], u[i0], u[i1]);
        blst_fr_sub(u[i1], tmp, u[i1]);

        if (di != 0)
            blst_fr_mul(u[i1], pq[di << pqshift], u[i1]);
    }

    __syncthreads();

    for (uint32_t rnd = 1; rnd < deg - 1; rnd++) {

        const uint32_t bit = counth >> rnd;
        for (uint32_t i = counts >> 1; i < counte >> 1; i++) {
            const uint32_t di = i & (bit - 1);
            const uint32_t i0 = (i << 1) - di;
            const uint32_t i1 = i0 + bit;

            memcpy(tmp, u[i0], sizeof(blst_fr));

            blst_fr_add(u[i0], u[i0], u[i1]);
            blst_fr_sub(u[i1], tmp, u[i1]);

            if (di != 0)
                blst_fr_mul(u[i1], pq[di << rnd << pqshift], u[i1]);
        }

        __syncthreads();
    }
    const uint32_t bit0 = counth >> (deg - 1);
    for (uint32_t i = counts >> 1; i < counte >> 1; i++) {
        const uint32_t di = i & (bit0 - 1);
        const uint32_t i0 = (i << 1) - di;
        const uint32_t i1 = i0 + bit0;
        if (index0 == 0) {
            blst_fr_add(y1[bitreverse(i0, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
            blst_fr_sub(y1[bitreverse(i1, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
        } else if (index0 == 1) {
            blst_fr_add(y2[bitreverse(i0, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
            blst_fr_sub(y2[bitreverse(i1, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
        } else if (index0 == 2) {
            blst_fr_add(y3[bitreverse(i0, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
            blst_fr_sub(y3[bitreverse(i1, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
        } else if (index0 == 3) {
            blst_fr_add(y4[bitreverse(i0, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
            blst_fr_sub(y4[bitreverse(i1, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
        } else if (index0 == 4) {
            blst_fr_add(y5[bitreverse(i0, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
            blst_fr_sub(y5[bitreverse(i1, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
        } else if (index0 == 5) {
            blst_fr_add(y6[bitreverse(i0, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
            blst_fr_sub(y6[bitreverse(i1, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
        } else if (index0 == 6) {
            blst_fr_add(y7[bitreverse(i0, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
            blst_fr_sub(y7[bitreverse(i1, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
        } else if (index0 == 7) {
            blst_fr_add(y8[bitreverse(i0, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
            blst_fr_sub(y8[bitreverse(i1, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
        } else if (index0 == 8) {
            blst_fr_add(y9[bitreverse(i0, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
            blst_fr_sub(y9[bitreverse(i1, deg) * p + ((index - k) << deg) + k], u[i0], u[i1]);
        }
    }
}


extern "C" __global__ void batch_inversion_and_mul(blst_fr *x,        
                                                   blst_fr *coeffs,
                                                   uint32_t n,
                                                   uint32_t num_width) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = num_width * index;
    uint32_t end = min(start + num_width, n);

    blst_fr tmp;
    
    blst_fr prod[WINDOW_SIZE];
    memcpy(tmp, BLS12_377_ONE, sizeof(blst_fr));
    for (uint i = start; i < end; i++) {
        blst_fr_mul(tmp, tmp, x[i]);
        memcpy(prod[i - start], tmp, sizeof(blst_fr));
    }

    blst_fr_inverse(tmp, tmp);
    blst_fr_mul(tmp, tmp, coeffs[0]);
    blst_fr new_tmp;


    for (uint32_t i0 = end - 1; i0 >= start + 1; i0--) {
        blst_fr_mul(new_tmp, tmp, x[i0]);
        blst_fr_mul(x[i0], tmp, prod[i0 - 1 - start]);
        memcpy(tmp, new_tmp, sizeof(blst_fr));

    }
    memcpy(x[start], tmp, sizeof(blst_fr));
}

extern "C" __global__ void batch_inversion(blst_fr *x,
                                           uint32_t n,
                                           uint32_t num_width) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = num_width * index;
    uint32_t end = min(start + num_width, n);

    blst_fr tmp;    
    blst_fr prod[WINDOW_SIZE];
    memcpy(tmp, BLS12_377_ONE, sizeof(blst_fr));
    for (uint i = start; i < end; i++) {
        blst_fr_mul(tmp, tmp, x[i]);
        memcpy(prod[i - start], tmp, sizeof(blst_fr));
    }

    blst_fr_inverse(tmp, tmp);    
    blst_fr new_tmp;


    for (uint32_t i0 = end - 1; i0 >= start + 1; i0--) {
        blst_fr_mul(new_tmp, tmp, x[i0]);
        blst_fr_mul(x[i0], tmp, prod[i0 - 1 - start]);
        memcpy(tmp, new_tmp, sizeof(blst_fr));

    }
    memcpy(x[start], tmp, sizeof(blst_fr));
}


extern "C" __global__ void numerators(blst_fr *x,
                                      blst_fr *y,
                                      uint32_t n,
                                      uint32_t num_width,
                                      uint32_t m) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = num_width * index;
    uint32_t end = min(start + num_width, n);


    blst_fr tmp;

    for (uint i = start; i < end; i++) {
        blst_fr_pow(&tmp, y[i], m);
        blst_fr_sub(tmp, tmp, BLS12_377_ONE);
        blst_fr_sub(x[i], x[i], tmp);
    }
}

extern "C" __global__ void product(blst_fr *x,
                                   blst_fr *y,
                                   uint32_t n,
                                   uint32_t num_width) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = num_width * index;
    uint32_t end = min(start + num_width, n);
    for (uint i = start; i < end; i++) {
        blst_fr_mul(x[i], x[i], y[i]);
    }
}

extern "C" __global__ void domain_elements(blst_fr *x,
                                           blst_fr *gens,
                                           uint32_t n,
                                           uint32_t num_width) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = num_width * index;
    uint32_t end = min(start + num_width, n);

    blst_fr_pow(&x[start], gens[0], index * num_width);
    for (uint i = start + 1; i < end; i++) {
        blst_fr_mul(x[i], x[i - 1], gens[0]);
    }
}



extern "C" __global__ void compute_a_poly(blst_fr *out_a, blst_fr *out_b, blst_fr *out_c,
                                          blst_fr *v_H,
                                          uint32_t n,
                                          uint32_t num_width) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = num_width * index;
    uint32_t end = min(start + num_width, n);

    for (uint i = start; i < end; i++) {
        blst_fr_mul(out_a[i], out_a[i], v_H[0]);
        blst_fr_mul(out_b[i], out_b[i], v_H[0]);
        blst_fr_mul(out_c[i], out_c[i], v_H[0]);
    }
}

extern "C" __global__ void blst_fr_mul_array(blst_fr *out, blst_fr *x,
                                             uint32_t n,
                                             uint32_t num_width) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = num_width * index;
    uint32_t end = min(start + num_width, n);

    for (uint i = start; i < end; i++) {
        blst_fr_mul(out[i], out[i], x[i]);
    }
}

extern "C" __global__ void calculate_lhs(blst_fr *out, blst_fr *p_a, blst_fr *p_b, blst_fr *p_c,
                                           uint32_t n,
                                           uint32_t num_width) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = num_width * index;
    uint32_t end = min(start + num_width, n);

    blst_fr tmp;
    for (uint i = start; i < end; i++) {
        blst_fr_mul(out[i], out[i], p_a[i]);
        blst_fr_mul(tmp, p_b[i], p_c[i]);
        blst_fr_sub(out[i], out[i], tmp);
    }
}

extern "C" __global__ void third_compute_h_poly(blst_fr *out_a, blst_fr *out_b, blst_fr *out_c,
                                                blst_fr *on_k_a, blst_fr *on_k_b, blst_fr *on_k_c,
                                                blst_fr *a2_a, blst_fr *a2_b, blst_fr *a2_c,
                                                blst_fr *f_i,
                                                uint32_t n,
                                                uint32_t num_width) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = num_width * index;
    uint32_t end = min(start + num_width, n);

    for (uint i = start; i < end; i++) {
        blst_fr_mul(out_a[i], out_a[i], on_k_a[i]);
        blst_fr_mul(out_b[i], out_b[i], on_k_b[i]);
        blst_fr_mul(out_c[i], out_c[i], on_k_c[i]);

        blst_fr_sub(out_a[i], a2_a[i], out_a[i]);
        blst_fr_sub(out_b[i], a2_b[i], out_b[i]);
        blst_fr_sub(out_c[i], a2_c[i], out_c[i]);

        blst_fr_mul(out_a[i], out_a[i], f_i[0]);
        blst_fr_mul(out_b[i], out_b[i], f_i[0]);
        blst_fr_mul(out_c[i], out_c[i], f_i[0]);
    }
}

extern "C" __global__ void third_compute_h_poly_parallel(blst_fr *out_a,
                                                         blst_fr *on_k_a,
                                                         blst_fr *a2_a,
                                                         blst_fr *f_i,
                                                         uint32_t n,
                                                         uint32_t num_width) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = num_width * index;
    uint32_t end = min(start + num_width, n);

    for (uint i = start; i < end; i++) {
        blst_fr_mul(out_a[i], out_a[i], on_k_a[i]);

        blst_fr_sub(out_a[i], a2_a[i], out_a[i]);

        blst_fr_mul(out_a[i], out_a[i], f_i[0]);
    }
}


int main() {
    return 0;
}