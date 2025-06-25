#include <mpi.h>
#include <stdlib.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

static constexpr int kN = 10000;

double a[kN][kN], b[kN][kN];

double func(double x, double y) {
  return x * (sin(x) + cos(y));
}

double dx = 0.01;

void generateMatrix(int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      a[i][j] = func(i * dx, j * dx);
    }
  }
}

void computeDerivativeX(int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols - 1; ++j) {
      if (j == 0) {
        b[i][j] = (a[i][j + 1] - a[i][j]) / dx;
      } else {
        b[i][j] = (a[i][j + 1] - a[i][j - 1]) / (2 * dx);
      }
    }
  }
}

int main(int argc, char* argv[]) {
  // Total number of processes
  int process_num = 0;
  // Current process
  int process_rank = 0;
  int elements_per_process, n_elements_recieved, index;
  std::vector<int> matrix_sizes = {10, 100, 1000, 10000};

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &process_num);
  MPI_Comm_size(MPI_COMM_WORLD, &process_rank);

  MPI_Status status;
  for (const int& size : matrix_sizes) {
    if (process_num == 0) {
      generateMatrix(size, size);
      int i;
      elements_per_process = size / process_rank;
      auto start_time = std::chrono::high_resolution_clock::now();
      if (process_rank > 1) {
        for (i = 1; i < process_rank - 1; i++) {
          index = i * elements_per_process;
          MPI_Send(&elements_per_process, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
          MPI_Send(&a[index][0], elements_per_process * size, MPI_DOUBLE, i, 0,
                   MPI_COMM_WORLD);
          MPI_Send(&index, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        index = i * elements_per_process;
        int elements_left = size - index;
        MPI_Send(&elements_left, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        MPI_Send(&a[index][0], elements_left * size, MPI_DOUBLE, i, 0,
                 MPI_COMM_WORLD);
        MPI_Send(&index, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      }
      computeDerivativeX(elements_per_process, size);
      for (int i = 1; i < process_rank; i++) {
        MPI_Recv(&index, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&n_elements_recieved, 1, MPI_INT, i, 1, MPI_COMM_WORLD,
                 &status);
        MPI_Recv(&b[index][0], n_elements_recieved * size, MPI_DOUBLE, i, 1,
                 MPI_COMM_WORLD, &status);
      }
      auto end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end_time - start_time;

      std::cout << "Matrix size: " << size << "x" << size
                << " Time elapsed: " << (elapsed.count()) << " seconds" << '\n';
    } else {
      MPI_Recv(&n_elements_recieved, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
      MPI_Recv(&a, n_elements_recieved * size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
               &status);
      MPI_Recv(&index, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
      computeDerivativeX(n_elements_recieved, size);
      MPI_Send(&index, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
      MPI_Send(&n_elements_recieved, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
      MPI_Send(&b, n_elements_recieved * size, MPI_DOUBLE, 0, 1,
               MPI_COMM_WORLD);
    }
  }
  MPI_Finalize();
  return 0;
}
