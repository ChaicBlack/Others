// 这是一个cuda C程序，而且是早期cuda版本的程序，可以看出很麻烦
#define N 10

int main() {
  int a[N], b[N], c[N];
  int *dev_a, *dev_b, *dev_c;

  // 我实在不知道为什么他要用handle_error来操作这些内存，为了安全？
  HANDLE_ERROR(cudaMalloc((void **)&dev_a, N * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_b, N * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_c, N * sizeof(int)));

  for (int i = 0; i < N; i++) {
    a[i] = -i;
    b[i] = i * i;
  }

  HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

  add<<<N, 1>>>(dev_a, dev_b,
                dev_c); // 作者分配了N个block而不是N个thread，我不懂

  HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

  for (int i = 0; i < N; i++) {
    printf("%d + %d = %d", a[i], b[i], c[i]);
  }

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  return 0;
}

__global__ void add(int *a, int *b, int *c) {
  int tid = blockIdx.x;
  if (tid < N)
    c[tid] = a[tid] + b[tid];
}
