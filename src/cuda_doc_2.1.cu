// Using pointers of pre-allocated containers to avoid data race.
__global__ void VecAdd(float *A, float *B, float *C) {
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
}

int main() {
  ...
      // Kernel invocation with N threads
      VecAdd<<<1, N>>>(A, B, C);
  ...
}
