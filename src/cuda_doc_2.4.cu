// Kernel definition
// 不再在compile time定义参数了
__global__ void cluster_kernel(float *input, float *outout) {}

int main() {
  float *input, *output;
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

  // runtime时调用kernel来确定cluster size
  // 这群API我以后再看看
  {
    cudaLaunchConfig_t config = {0};

    config.gridDim = numBlocks;
    config.blockDim = threadsPerBlock;

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = 2;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.attrs = attribute;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config, cluter_kernel, input, output);
  }
}
