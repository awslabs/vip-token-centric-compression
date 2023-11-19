#include "cuda_kernel.h"
#include <stdlib.h>

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void vector_index_accumulate_cuda_kernel(
  int   *indexes,      // [batch_size]
  float *source,       // [batch_size, vector_dim]
  float *outputs,      // [output_size, vector_dim]
  int batch_size,
  int output_size,
  int vector_dim
) {

  int thread_idx = threadIdx.y * WARP_SIZE + threadIdx.x;
  int batch_idx_start = blockIdx.x * WORK_SIZE;
  // assert blockDim.x == WARP_SIZE
  // assert blockDim.y == MAX_THREADS_PER_BLOCK / WARP_SIZE

  extern __shared__ float buffer[];
  float *output_buffer = buffer;
  int *index_buffer = (int*)&buffer[output_size * vector_dim];

  for (int idx_start = 0; idx_start < output_size * vector_dim; idx_start = idx_start + MAX_THREADS_PER_BLOCK) {
    int idx = idx_start + thread_idx;
    if (idx < output_size * vector_dim) {
      output_buffer[idx] = 0;
    }
  }
  __syncthreads();

  for (int idx_start = 0; idx_start < WORK_SIZE; idx_start = idx_start + MAX_THREADS_PER_BLOCK) {
    if (true) {
      int batch_idx = batch_idx_start + idx_start + thread_idx;
      if (batch_idx < batch_size) {
        index_buffer[thread_idx] = indexes[batch_idx];
      }
    }
    __syncthreads();
    for (int buffer_idx_start = 0; buffer_idx_start < MAX_THREADS_PER_BLOCK; buffer_idx_start = buffer_idx_start + MAX_THREADS_PER_BLOCK / WARP_SIZE) {
      int buffer_idx = buffer_idx_start + threadIdx.y;
      int batch_idx = batch_idx_start + idx_start + buffer_idx;
      if (batch_idx < batch_size) {
        int index = index_buffer[buffer_idx];
        for (int j_start = 0; j_start < vector_dim; j_start = j_start + WARP_SIZE) {
          int j = j_start + threadIdx.x;
          if (j < vector_dim) {
            atomicAdd(&output_buffer[index * vector_dim + j], source[(size_t)batch_idx * (size_t)vector_dim + (size_t)j]);
          }
        }
      }
    }
    __syncthreads();
  }

  for (int idx_start = 0; idx_start < output_size * vector_dim; idx_start = idx_start + MAX_THREADS_PER_BLOCK) {
    int idx = idx_start + thread_idx;
    if (idx < output_size * vector_dim) {
      atomicAdd(&outputs[idx], output_buffer[idx]);
    }
  }
}
