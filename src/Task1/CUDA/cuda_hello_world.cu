#include <stdio.h>

__global__ void helloFromGPU() {
    printf("Hello, World! From thread %d in block %d\n", threadIdx.x,
           blockIdx.x);
}

int main() {
    // Launch kernel with 2 blocks and 4 threads per block
    helloFromGPU<<<2, 4>>>();

    // Wait for the GPU to finish before returning
    cudaDeviceSynchronize();

    return 0;
}
