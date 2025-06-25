#include "mpi.h"

#include <iostream>
#include <string>

#include "exec_thread/exec_thread.hpp"

int main(int argc, char* argv[]) {
  // Total number of processes
  int process_num = 0;
  // Current process
  int process_rank = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &process_num);
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

  if (argc != 2) {
    std::cerr << "USAGE: mpiexec -n <PROCESSES> "
                 "./src/Task2/MPI/mpi_array_sum <ARRAY_SIZE>\n";
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  int arr_size = std::stoi(argv[1]);
  if (arr_size < process_num) {
    std::cerr << "Array size must be bigger than number of threads\n";
    MPI_Finalize();
    return EXIT_FAILURE;
  }
  if (process_rank == 0) {
    executeMainThread(process_num, arr_size);
  } else {
    executeRegularThread();
  }
  MPI_Finalize();
  return EXIT_SUCCESS;
}
