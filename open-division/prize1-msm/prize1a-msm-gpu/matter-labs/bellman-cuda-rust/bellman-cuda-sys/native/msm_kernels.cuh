#pragma once
#include "ec.cuh"
#include "ff_dispatch_st.cuh"

namespace msm {

typedef ec<fd_p> curve;
typedef curve::storage storage;
typedef curve::field field;
typedef curve::point_affine point_affine;
typedef curve::point_jacobian point_jacobian;
typedef curve::point_xyzz point_xyzz;
typedef curve::point_projective point_projective;

__host__ cudaError_t left_shift(point_affine *values, unsigned shift, unsigned count, cudaStream_t stream);

__host__ cudaError_t initialize_buckets(point_xyzz *buckets, unsigned count, cudaStream_t stream);

__host__ cudaError_t compute_bucket_indexes(const fd_q::storage *scalars, unsigned windows_count, unsigned window_bits, unsigned precomputed_windows_stride,
                                            unsigned precomputed_bases_stride, unsigned *bucket_indexes, unsigned *base_indexes, unsigned count,
                                            bool scalars_not_montgomery, cudaStream_t stream);

__host__ cudaError_t aggregate_buckets(bool is_first, const unsigned *base_indexes, const unsigned *bucket_run_offsets, const unsigned *bucket_run_lengths,
                                       const unsigned *bucket_indexes, const point_affine *bases, point_xyzz *buckets, unsigned count, cudaStream_t stream);

__host__ cudaError_t split_windows(unsigned source_window_bits_count, unsigned source_windows_count, const point_xyzz *source_buckets,
                                   point_xyzz *target_buckets, unsigned count, cudaStream_t stream);

__host__ cudaError_t reduce_buckets(point_xyzz *buckets, unsigned count, cudaStream_t stream);

__host__ cudaError_t last_pass_gather(unsigned bits_count_pass_one, const point_xyzz *source, point_jacobian *target, unsigned count, cudaStream_t stream);

__host__ cudaError_t set_kernel_attributes();

} // namespace msm