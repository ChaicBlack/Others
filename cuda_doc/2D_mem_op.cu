// Host code
int width = 64, height = 64;
float *devPtr;
size_t pitch; // pitch表示每行存储字节数，略大于宽度，是cuda
              // runtime根据硬件和其他参数计算出来的，用来对齐之类的
cudaMallocPitch(&devPtr, &pitch, width * sizeof(float), height);

// Device code
__global__ void MyKernel(float *devPtr, size_t pitch, int width, int height) {
  for (int r = 0; r < height; r++) {
    float *row = (float *)((char *)devPtr + r * pitch);
    for (int c = 0; c < width; c++)
      float element = row[c];
  }
}
