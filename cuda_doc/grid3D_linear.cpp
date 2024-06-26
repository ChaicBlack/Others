#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__device__ int a[256][512][512];
__device__ float b[256][512][512];

__global__ void grid3D_linear(int nx, int ny, int nz, int id) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int array_size = nx * ny * nz;
  int total_threads = gridDim.x * blockDim.x;
  int tid_start = tid;
  int pass = 0;

  while (tid < array_size) {
    int x = tid % nx;
    int y = (tid / nx) % ny;
    int z = tid / (nx * ny);
    /*这样其实非常低效，建议改成位运算
     * int x = tid & 0x01ff;
     * int y = (tid >> 9) & 0x01ff;
     * int z = tid >> 18;*/
    a[z][y][x] = tid;
    b[z][y][x] = sqrtf((float)a[z][y][x]);
    if (tid == id) {
      printf("array size  %3d x %3d x %3d = %d\n", nx, ny, nz, array_size);
      printf("thread block %3d\n", blockDim.x);
      printf("thread grid  %3d\n", gridDim.x);
      printf("total number of threads in grid %d\n", total_threads);
      printf("a[%d][%d][%d] = %i and b[%d][%d][%d] = %f\n", z, y, x, a[z][y][x],
             z, y, x, b[z][y][x]);
      printf("rank_in_block = %d rank_in_grid = %d pass %d tid offset %d\n",
             threadIdx.x, tid_start, pass, tid - tid_start);
    }
    tid += total_threads;
    pass++;
  }
}

int main(int argc, char *argv[]) {
  int id = (argc > 1) ? atoi(argv[1]) : 12345;
  int blocks = (argc > 2) ? atoi(argv[2]) : 288;
  int threads = (argc > 3) ? atoi(argv[3]) : 256;

  grid3D_linear<<<blocks, threads>>>(512, 512, 256, id);
  return 0;
}
