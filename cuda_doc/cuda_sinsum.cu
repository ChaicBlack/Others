#include "cx.h"
#include "cxtimers.h"

__host__ __device__ inline float sinsum(float x, int term) {
  float x2 = x * x;
  float term = x;
  float sum = term;
  for (int n = 1; n < terms; n++) {
    term *= -x2 / (float)(2 * n * (2 * n + 1)); // get next term from previous
    sum += term;
  }
  return sum;
}

__global__ void gpu_sin(float *sums, int steps, int terms, float step_size) {
  int step = blockIdx.x * blockDim.x + threadIdx.x; // unique thread ID
  if (step < steps) {
    float x = step_size * step;
    sums[step] = sinsum(x, terms); // store sin values in array
  }
}

int main(int argc, char *argv[]) {
  int steps = (argc > 1) ? atoi(argv[1]) : 10000000; // get command
  int terms = (argc > 2) ? atoi(argv[2]) : 1000;     // line arguments
  int threads = 256;
  int blocks = (steps + threads - 1) / threads; // ensure threads*blocks ≥ steps

  double pi = 3.14159265358979323;
  double step_size = pi / (steps - 1); // NB n-1 steps between n points

  thrust::device_vector<float> dsums(steps);         // GPU buffer
  float *dptr = thrust::raw_pointer_cast(&dsums[0]); // get pointer

  cx::timer tim;
  gpu_sin<<<blocks, threads>>>(dptr, steps, terms, (float)step_size);
  double gpu_sum = thrust::reduce(dsums.begin(), dsums.end());
  double gpu_time = tim.lap_ms(); // get elapsed time

  // Trapezoidal Rule Correction
  gpu_sum -= 0.5 * (sinsum(0.0f, terms) + sinsum(pi, terms));
  gpu_sum *= step_size;
  printf("gpu sum = %.10f, steps %d terms %d time %.3f ms\n", gpu_sum, steps,
         terms, gpu_time);
  return 0;
}
