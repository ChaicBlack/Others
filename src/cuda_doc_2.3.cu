// Kernel definition
// 使用__cluster_dims__参数在compile time定义cluster的三维大小
__global__ void __cluster_dims__(2, 1, 1)
    cluster_kernel(float *input, float *outout) {}

int main() {
  float *input, *output;
  // 还是像之前一样只定义thread和block
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
  cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}
