#pragma once

#ifndef __cplusplus
#include <stdbool.h>
#include <stddef.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// bellman-cuda API error types
typedef enum bc_error {
  bc_success = 0,                 // The API call returned with no errors. In the case of query calls,
                                  // this also means that the operation being queried is complete (see bc_event_query() and bc_stream_query()).
  bc_error_invalid_value = 1,     // This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.
  bc_error_memory_allocation = 2, // The API call failed because it was unable to allocate enough memory to perform the requested operation.
  bc_error_not_ready = 600        // This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error,
                                  // but must be indicated differently than bc_success (which indicates completion). Calls that may return this value include
                                  // bc_event_query() and bc_stream_query().
} bc_error;

// bellman-cuda stream
typedef struct bc_stream {
  void *handle;
} bc_stream;

// bellman-cuda event
typedef struct bc_event {
  void *handle;
} bc_event;

// bellman-cuda memory pool
typedef struct bc_mem_pool {
  void *handle;
} bc_mem_pool;

// bellman-cuda host function
// user_data - Argument value passed to the function
typedef void (*bc_host_fn)(void *user_data);

// Initializes the internal state for MSM computations
// Should be called once per lifetime of the process
bc_error msm_set_up();

bc_error msm_left_shift(void *values, unsigned shift, unsigned count, bc_stream stream);

// Configuration for the MSM execution
typedef struct msm_configuration {
  bc_mem_pool mem_pool;                  // The memory pool that will be used for temporary allocations needed by the execution
  bc_stream stream;                      // The stream on which the execution will be scheduled
  void *bases;                           // Device pointer to the bases that will be used for this execution
  void *scalars;                         // Pointer to the scalars used by this execution, can be either pinned or pageable host memory or device memory pointer
  void *results;                         // Pointer to an array of 254 EC points in jacobian coordinates corresponding to the 254 bits of the final MSM result,
                                         // can be either pinned or pageable host memory or device memory pointer
  unsigned log_scalars_count;            // Log2 of the number of scalars
  bc_event h2d_copy_finished;            // An optional event that should be recorded after the Host to Device memory copy  has completed
  bc_host_fn h2d_copy_finished_callback; // An optional callback that should be executed after the Host to Device memory copy has completed
  void *h2d_copy_finished_callback_data; // User-defined data for the above callback
  bc_event d2h_copy_finished;            // An optional event that should be recorded after the Device to Host memory copy has completed
  bc_host_fn d2h_copy_finished_callback; // An optional callback that should be executed after the Device to Host memory copy has completed
  void *d2h_copy_finished_callback_data; // User-defined data for the above callback
  bool force_min_chunk_size;
  unsigned log_min_chunk_size;
  bool force_max_chunk_size;
  unsigned log_max_chunk_size;
  unsigned window_bits_count;
  unsigned precomputed_windows_stride;
  unsigned precomputed_bases_stride;
  bool scalars_not_montgomery;
} msm_configuration;

// Schedule the MSM execution.
// configuration - The configuration for the execution
bc_error msm_execute_async(msm_configuration configuration);

// release all resources associated with the internal state for MSM computations
bc_error msm_tear_down();

#ifdef __cplusplus
} // extern "C"
#endif
