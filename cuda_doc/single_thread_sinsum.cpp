#include "cxtimers.h"
#include <stdio.h>
#include <stdlib.h>

inline float sinsum(float x, int terms) {
  // sin(x) = x - x^3/3 + x^5/5 ...
  float term = x;
  float sum = term;
  float x2 = x * x;
  for (int n = 1; n < terms; n++) {
    term *= -x2 / (float)(2 * n * (2 * n + 1));
    sum += term;
  }
  return sum;
}

int main(int argc, char *argv[]) {
  int steps = (argc > 1) ? atoi(argv[1]) : 10000000;
  int terms = (argc > 2) ? atoi(argv[2]) : 1000;

  double pi = 3.14159265358979323;
  double step_size = pi / (steps - 1);

  cx::timer tim;
  double cpu_sum = 0.0;
  for (int step = 0; step < steps; step++) {
    float x = step_size * step;
    cpu_sum += sinsum(x, terms);
  }
  double cpu_time = tim.lap_ms();

  cpu_sum -= 0.5 * (sinsum(0.0, terms) + sinsum(pi, terms));
  cpu_sum *= step_size;
  printf("cpu sum = %.10f,steps %d terms %d time %.3f ms\n", cpu_sum, steps,
         terms, cpu_time);
  return 0;
}
