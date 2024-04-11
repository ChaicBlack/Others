// Device code
__global__ void VecAdd(float *A, float *B, float *C, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}

// Host code
int main() {
  int N = ...;
  size_t size = N * sizeof(float);

  // 在主机内存中分配数组空间
  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_C = (float *)malloc(size);

  // 这里进行一些数组(向量)的初始化操作
  ...

  // 在设备内存中分配数组空间
  float *d_A;
  cudaMalloc(&d_A, size);
  float *d_B;
  cudaMalloc(&d_B, size);
  float *d_C;
  cudaMalloc(&d_C, size);

  // 从主机内存中将数据复制到设备内存中
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // 调用Kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

  // 将结果从设备内存复制到主机内存中
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  // 释放设备内存
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // 释放主机内存
  ...
}
