#include <stdio.h>
#include <stdlib.h>
#include "cxtimers.h"
#include "cuda_runtime.h"
#include "thrust/device_vecetor.h"

__host__ __device__ inline float sinsum(float x, int term){
  float x2 = x * x;
  float term = x;
  float sum = term;
  for(int n = 1; n < terms; n++){

  }
}
