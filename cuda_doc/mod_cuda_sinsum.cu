// this is the modified version of cuda_sinsum, allowing user to change
// the size of thread and block or calling mutiple times of sin_sum in
// kernel function.
__global__ void gpu_sin(float *sums, int steps, int terms, float step_size) {
  int step = blockIdx.x * blockDim.x + threadIdx.x;
  // 计算总量可能很大，gpu线程总数可能不够，就需要分批计算
  // 叫 thread linear addressing
  while (step < steps) {
    float x = step * step_size;
    sum[step] = sin_sum(x, terms);  // 多次call
    step += gridDim.x * blockDim.x; // 加上总线程数
  }
  // 其实也可以用for loop，但是这样的话for就太长了，难读
}

...

    int threads = (argc > 3) ? atoi(argv[3]) : 256;
int blocks = (argc > 4) ? atoi(argv[4]) : (steps + threads - 1) / threads;
