#include <iostream>
#include <string>
#include <chrono>

#include <omp.h>

#include "../generate_array/generate_array.hpp"

static constexpr int kIterations = 100;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "USAGE: ./src/Task2/OpenMP/openmp_array_sum <THREADS> "
                     "<ARRAY_SIZE>\n";
        return EXIT_FAILURE;
    }
    constexpr int kMaxArrayElement = 100;
    omp_set_num_threads(std::stoi(argv[1]));
    const std::size_t arr_size = std::stoull(argv[2]);

    int64_t total_sum = 0;
    std::chrono::duration<double, std::milli> total_time(0);

    int* arr = new int[arr_size];
    for (int _ = 0; _ < kMaxArrayElement; ++_) {
        generateArray(arr, arr_size, kMaxArrayElement);
        std::size_t i;
        int64_t sum = 0;

        auto time_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for shared(arr) private(i) reduction(+ : sum) \
    schedule(static)
        for (i = 0; i < arr_size; ++i) {
            sum += arr[i];
        }
        auto time_end = std::chrono::high_resolution_clock::now();

        total_sum += sum;
        total_time += time_end - time_start;
    }
    delete[] arr;

    std::cout << "Average total sum: "
              << static_cast<double>(total_sum) / kIterations << '\n';
    std::cout << "Time elapsed: "
              << std::chrono::duration<double, std::milli>(total_time).count() /
                     kIterations
              << " milliseconds\n";
    return 0;
}
