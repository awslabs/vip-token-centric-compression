#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 512
#define WORK_SIZE 8192
#define FULL_MASK 0xffffffff

__global__ void vector_index_accumulate_cuda_kernel(
  int   *indexes,      // [batch_size]
  float *source,       // [batch_size, vector_dim]
  float *outputs,      // [output_size, vector_dim]
  int batch_size,
  int output_size,
  int vector_dim
);
