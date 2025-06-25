#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>

#include "../generate_array/generate_array.hpp"

static constexpr std::size_t kThreadsPerBlock = 1024;
static constexpr int kIterations = 100;

__global__ void sumKernel(const int* input_arr, int* output_arr,
                          const std::size_t input_arr_size) {
    __shared__ int sdata[kThreadsPerBlock];
    // blockDim.x threads in block
    // threadId.x index of thread within the block
    // blockIdx.x index of block within the grid
    std::size_t thread_ind = threadIdx.x;
    std::size_t input_arr_ind = blockIdx.x * blockDim.x + thread_ind;
    sdata[thread_ind] =
        (input_arr_ind < input_arr_size) ? input_arr[input_arr_ind] : 0;
    __syncthreads();
    for (std::size_t add_offset = blockDim.x / 2; add_offset > 0;
         add_offset /= 2) {
        if (thread_ind < add_offset) {
            sdata[thread_ind] += sdata[thread_ind + add_offset];
        }
        __syncthreads();
    }
    if (thread_ind == 0) {
        output_arr[blockIdx.x] = sdata[0];
    }
}

[[nodiscard]] int arraySum(const int* host_arr_input, int* host_arr_ouput,
                           const std::size_t host_arr_input_size,
                           const std::size_t host_arr_output_size) {
    int* cuda_arr_input = nullptr;
    int* cuda_arr_output = nullptr;
    cudaMalloc(&cuda_arr_input, host_arr_input_size * sizeof(int));
    cudaMalloc(&cuda_arr_output, host_arr_output_size * sizeof(int));
    cudaMemcpy(cuda_arr_input, host_arr_input,
               host_arr_input_size * sizeof(int), cudaMemcpyHostToDevice);
    sumKernel<<<host_arr_output_size, kThreadsPerBlock>>>(
        cuda_arr_input, cuda_arr_output, host_arr_input_size);
    cudaMemcpy(host_arr_ouput, cuda_arr_output,
               host_arr_output_size * sizeof(int), cudaMemcpyDeviceToHost);
    int array_sum = 0;
    for (std::size_t i = 0; i < host_arr_output_size; ++i) {
        array_sum += host_arr_ouput[i];
    }
    cudaFree(cuda_arr_input);
    cudaFree(cuda_arr_output);
    return array_sum;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("./src/Task2/CUDA/cuda_array_sum <ARRAY_SIZE>\n");
        return EXIT_FAILURE;
    }
    int64_t total_sum = 0;
    std::chrono::duration<double, std::milli> total_time(0);
    constexpr int kMaxArrayElement = 100;
    const std::size_t host_arr_input_size = std::stoull(argv[1]);
    const std::size_t host_arr_output_size =
        (host_arr_input_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
    int* host_arr_input = new int[host_arr_input_size];
    int* host_arr_output = new int[host_arr_output_size];
    for (int _ = 0; _ < kIterations; ++_) {
        generateArray(host_arr_input, host_arr_input_size, kMaxArrayElement);
        auto time_start = std::chrono::high_resolution_clock::now();
        total_sum += arraySum(host_arr_input, host_arr_output,
                              host_arr_input_size, host_arr_output_size);
        auto time_end = std::chrono::high_resolution_clock::now();
        total_time += time_end - time_start;
    }
    delete[] host_arr_input;
    delete[] host_arr_output;
    printf("Average total sum: %f\n",
           static_cast<double>(total_sum) / kIterations);
    printf("Time elapsed: %f milliseconds\n",
           std::chrono::duration<double, std::milli>(total_time).count() /
               kIterations);
    return 0;
}
