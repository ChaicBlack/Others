/* Below are for one block of threads
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
        int i = threadIdx.x;
        int j = threadIdx.y;
        C[i][j] = A[i][j] + B[i][j];
}

int main(){
        ...
        // Kernal invocation with one block of N * N * 1 threads
        int numBlocks = 1;
        dim3 threadsPerBlock(N, N);
        MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
        ...
}*/

// Below are for a 2-dimensional blocks of threads
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) {
  int i = blockIdx.x * blockdim.x + threadIdx.x;
  int j = blockIdx.y * blockdim.y + threadIdx.y;
  if (i < N && j < N)
    C[i][j] = A[i][j] + B[i][j];
}

int main() {
  ...	dim3 threadsPerBlock(16, 16);
  // 其实这样会让一些数据无法得到计算，如果dim3除法向下取整的话
  // dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
  //                (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
  // 这样可以算完
  dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
  MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
  ...
}
