#include "exec_thread.hpp"

#include <iostream>
#include <chrono>

#include "mpi.h"

#include "../../array_sum/array_sum.hpp"
#include "../../generate_array/generate_array.hpp"

void executeMainThread(int num_threads, int arr_size) {
    constexpr int kMaxArrayElement = 100;
    int elements_per_thread = arr_size / num_threads;
    int* arr = new int[static_cast<std::size_t>(arr_size)];
    generateArray(arr, static_cast<std::size_t>(arr_size), kMaxArrayElement);
    auto time_start = std::chrono::high_resolution_clock::now();
    if ((arr_size % num_threads) == 0) {
        for (int i = 1; i < num_threads; ++i) {
            int j = i * elements_per_thread;
            MPI_Send(&elements_per_thread, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(arr + j, elements_per_thread, MPI_INT, i, 0,
                     MPI_COMM_WORLD);
        }
    } else {
        for (int i = 1; i < num_threads - 1; ++i) {
            int j = i * elements_per_thread;

            MPI_Send(&elements_per_thread, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(arr + j, elements_per_thread, MPI_INT, i, 0,
                     MPI_COMM_WORLD);
        }
        int el_left = arr_size % num_threads;
        MPI_Send(&el_left, 1, MPI_INT, num_threads - 1, 0, MPI_COMM_WORLD);
        MPI_Send(&arr[arr_size - el_left], el_left, MPI_INT, num_threads - 1, 0,
                 MPI_COMM_WORLD);
    }
    int64_t result =
        arraySum(arr, static_cast<std::size_t>(elements_per_thread));
    for (int i = 1; i < num_threads; ++i) {
        int64_t subsum;
        MPI_Status status;
        MPI_Recv(&subsum, 1, MPI_INT64_T, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD,
                 &status);

        result += subsum;
    }
    auto time_end = std::chrono::high_resolution_clock::now();
    delete[] arr;
    std::cout << "Total sum: " << result << std::endl;
    std::cout << "Time elapsed: "
              << std::chrono::duration<double, std::milli>(time_end -
                                                           time_start)
                     .count()
              << " milliseconds" << std::endl;
}

void executeRegularThread() {
    int arr_size;
    MPI_Status status;
    MPI_Recv(&arr_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    int* arr = new int[static_cast<std::size_t>(arr_size)];
    MPI_Recv(arr, arr_size, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    int64_t result = arraySum(arr, static_cast<std::size_t>(arr_size));
    delete[] arr;
    MPI_Send(&result, 1, MPI_INT64_T, 0, 0, MPI_COMM_WORLD);
}
