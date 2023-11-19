#include <torch/extension.h>
#include <ATen/ATen.h>
#include "cuda_launch.h"

at::Tensor vector_index_accumulate(
  at::Tensor indexes,
  at::Tensor source,
  int output_size
) {
  return vector_index_accumulate_kernel(
    indexes,
    source,
    output_size
  );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vector_index_accumulate", &vector_index_accumulate, "vector_index_accumulate (CUDA)");
}
