#include <omp.h>

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

std::vector<std::vector<int>> generateRandomMatrix(const int rows,
                                                   const int columns) {
  std::vector<std::vector<int>> matrix(rows, std::vector<int>(columns));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < columns; ++j) {
      matrix[i][j] = rand() % 9 + 1;
    }
  }
  return matrix;
}

std::vector<std::vector<int>> multiplyMatrices(
    const std::vector<std::vector<int>>& matrix1,
    const std::vector<std::vector<int>>& matrix2) {
  int rows1 = matrix1.size();
  int cols1 = matrix1[0].size();
  int rows2 = matrix2.size();
  int cols2 = matrix2[0].size();
  if (cols1 != rows2) {
    std::cerr << "Unable to multiply matrix\n";
    exit(1);
  }
  std::vector<std::vector<int>> result(rows1, std::vector<int>(cols2));

#pragma omp parallel for collapse(2)
  for (int i = 0; i < rows1; ++i) {
    for (int j = 0; j < cols2; ++j) {
      result[i][j] = 0;
      for (int k = 0; k < cols1; ++k) {
        result[i][j] += matrix1[i][k] * matrix2[k][j];
      }
    }
  }

  return result;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr
        << "USAGE: ./src/Task4/OpenMP/openmp_matrix_composition <THREADS>\n";
    return EXIT_FAILURE;
  }
  omp_set_num_threads(std::stoi(argv[1]));
  std::vector<std::pair<int, int>> matrix_sizes = {
      {10, 10}, {100, 100}, {1000, 1000}, {2000, 2000}};
  for (const auto& size : matrix_sizes) {
    int rows1 = size.first;
    int columns1 = size.second;
    int rows2 = columns1;
    int columns2 = size.first;
    std::vector<std::vector<int>> matrix1 =
        generateRandomMatrix(rows1, columns1);
    std::vector<std::vector<int>> matrix2 =
        generateRandomMatrix(rows2, columns2);
    double start_time = omp_get_wtime();
    std::vector<std::vector<int>> result = multiplyMatrices(matrix1, matrix2);
    double end_time = omp_get_wtime();
    std::cout << "Matrix size: " << rows1 << "x" << columns1 << " и " << rows2
              << "x" << columns2 << " Time elapsed: " << (end_time - start_time)
              << " seconds" << '\n';
  }
  return 0;
}
