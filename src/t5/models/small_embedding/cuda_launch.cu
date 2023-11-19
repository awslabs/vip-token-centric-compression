#include <torch/extension.h>
#include <ATen/ATen.h>
#include "cuda_launch.h"
#include "cuda_kernel.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

at::Tensor vector_index_accumulate_kernel(
  at::Tensor indexes,
  at::Tensor source,
  int output_size
) {

  int batch_size = source.size(0);
  int vector_dim = source.size(1);

  at::Tensor outputs = at::zeros({output_size, vector_dim}, source.options());

  int thread_x = WARP_SIZE;
  int thread_y = MAX_THREADS_PER_BLOCK / WARP_SIZE;
  int block_x = batch_size / WORK_SIZE + 1;
  dim3 threads(thread_x, thread_y);
  dim3 blocks(block_x);
  int shared_mem = (output_size * vector_dim + MAX_THREADS_PER_BLOCK) * sizeof(float);

  vector_index_accumulate_cuda_kernel<<<blocks, threads, shared_mem>>>(
    indexes.data_ptr<int>(),
    source.data_ptr<float>(),
    outputs.data_ptr<float>(),
    batch_size,
    output_size,
    vector_dim
  );

  return outputs;

}
