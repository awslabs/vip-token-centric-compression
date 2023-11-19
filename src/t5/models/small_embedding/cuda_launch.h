#include <torch/extension.h>
#include <ATen/ATen.h>

at::Tensor vector_index_accumulate_kernel(
  at::Tensor indexes,
  at::Tensor source,
  int output_size
);
