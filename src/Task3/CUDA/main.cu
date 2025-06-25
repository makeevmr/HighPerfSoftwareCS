#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <string>

#include <chrono>

typedef double (*FuncPtr)(double, double);

constexpr double kMinX = 0.0;
constexpr double kMaxX = 10.0;
constexpr double kMinY = 0.0;
constexpr double kMaxY = 10.0;

double func(double x, double y) {
  return x * (sin(x) + cos(y));
}

double* generateGrid(FuncPtr func, double xmin, double xmax, int xcount,
                     double ymin, double ymax, int ycount);

__global__ void derivativeKernel(double* a, double* b, int xsize, int ysize,
                                 double dy, double dy2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < xsize * ysize) {
    if ((i % ysize) == 0) {
      b[i] = (a[i + 1] - a[i]) / dy;
    } else if (i % ysize == ysize - 1) {
      b[i] = (a[i] - a[i - 1]) / dy;
    } else {
      b[i] = (a[i + 1] - a[i - 1]) / dy2;
    }
  }
}

int main(int argc, char* argv[]) {
  int block_count = argc < 2 ? 5 : std::stoi(argv[1]);
  int x_size = argc < 3 ? 100 : std::stoi(argv[2]);
  int y_size = argc < 4 ? 100 : std::stoi(argv[3]);
  int total_size = x_size * y_size;
  if (total_size < block_count) {
    std::cout << "Matrix size must be bigger than number of blocks!"
              << std::endl;
    return 1;
  }
  double* a = generateGrid(func, kMinX, kMaxX, x_size, kMinY, kMaxY, y_size);
  double* dev_a;
  double* dev_b;
  cudaMalloc((void**)&dev_a, total_size * sizeof(double));
  cudaMalloc((void**)&dev_b, total_size * sizeof(double));
  cudaMemcpy(dev_a, a, total_size * sizeof(double), cudaMemcpyHostToDevice);
  delete[] a;
  double dy = (kMaxY - kMinY) / double(y_size - 1);
  double dy2 = dy * 2.0;
  auto time_start = std::chrono::high_resolution_clock::now();
  derivativeKernel<<<block_count, total_size / block_count + 1>>>(
      dev_a, dev_b, x_size, y_size, dy, dy2);
  cudaDeviceSynchronize();
  auto time_end = std::chrono::high_resolution_clock::now();
  double* b = new double[total_size];
  cudaMemcpy(b, dev_b, total_size * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(dev_a);
  cudaFree(dev_b);
  delete[] b;
  std::cout << "Time elapsed: "
            << std::chrono::duration<double>(time_end - time_start).count()
            << " seconds" << std::endl;
  return 0;
}

double* generateGrid(FuncPtr func, double xmin, double xmax, int xcount,
                     double ymin, double ymax, int ycount) {
  if (xmin > xmax || ymin > ymax || xcount <= 1 || ycount <= 1) {
    return nullptr;
  }
  double* arr = new double[xcount * ycount];
  if (arr == nullptr) {
    return nullptr;
  }
  double x = xmin;
  double dx = (xmax - xmin) / double(xcount - 1);
  double dy = (ymax - ymin) / double(ycount - 1);
  for (int i = 0; i < xcount; ++i) {
    double y = ymin;
    for (int j = 0; j < ycount; ++j) {
      arr[ycount * i + j] = func(x, y);
      y += dy;
    }
    x += dx;
  }
  return arr;
}
