// __device__应该就是设备端的操作
// __constant__是GPU的特殊内存，传输到GPU后无法更改，也就是运行时不变
// ，用来给所有线程读取用的
__constant__ float constData[256];
float data[256];
cudaMemcpyToSymbol(constData, data, sizeof(data));
cudaMemcpyFromSymbol(data, constData, sizeof(data));

__device__ float devData;
float value = 3.14f;
cudaMemcpyToSymbol(devData, &value, sizeof(float));

__device__ float *devPointer;
float *ptr;
cudaMalloc(&ptr, 256 * sizeof(float)); // 设备端分配内存，地址存储在主机端ptr中
cudaMemcpyToSymbol(devPointer, &ptr,
                   sizeof(ptr)); // 后来又给回了设备端devPointer

/* 在cuda中，symbol象征着全局变量、常量和函数，特点是可以在主机端和设备端共享，
   cudaGetSymbolAddress()是用来获取指向某个symbol variable的内存位置，
   cudaGetSymbolSize()用来获取其大小。*/
