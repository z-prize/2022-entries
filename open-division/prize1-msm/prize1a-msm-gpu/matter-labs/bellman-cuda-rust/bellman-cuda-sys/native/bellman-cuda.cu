#include "bellman-cuda.h"
#include "msm.cuh"
#include <cuda_runtime_api.h>

bc_error msm_set_up() { return static_cast<bc_error>(msm::set_up()); }

bc_error msm_left_shift(void *values, const unsigned shift, const unsigned count, bc_stream stream) {
  return static_cast<bc_error>(msm::left_shift(static_cast<msm::point_affine *>(values), shift, count, static_cast<cudaStream_t>(stream.handle)));
}

bc_error msm_execute_async(const msm_configuration configuration) {
  msm::execution_configuration cfg = {static_cast<cudaMemPool_t>(configuration.mem_pool.handle),
                                      static_cast<cudaStream_t>(configuration.stream.handle),
                                      static_cast<msm::point_affine *>(configuration.bases),
                                      static_cast<fd_q::storage *>(configuration.scalars),
                                      static_cast<msm::point_jacobian *>(configuration.results),
                                      configuration.log_scalars_count,
                                      static_cast<cudaEvent_t>(configuration.h2d_copy_finished.handle),
                                      configuration.h2d_copy_finished_callback,
                                      configuration.h2d_copy_finished_callback_data,
                                      static_cast<cudaEvent_t>(configuration.d2h_copy_finished.handle),
                                      configuration.d2h_copy_finished_callback,
                                      configuration.d2h_copy_finished_callback_data,
                                      configuration.force_min_chunk_size,
                                      configuration.log_min_chunk_size,
                                      configuration.force_max_chunk_size,
                                      configuration.log_max_chunk_size,
                                      configuration.window_bits_count,
                                      configuration.precomputed_windows_stride,
                                      configuration.precomputed_bases_stride,
                                      configuration.scalars_not_montgomery};
  return static_cast<bc_error>(msm::execute_async(cfg));
}

bc_error msm_tear_down() { return static_cast<bc_error>(msm::tear_down()); };
