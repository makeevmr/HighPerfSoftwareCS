#include <omp.h>

#include <cmath>
#include <iostream>
#include <vector>

double func(double x, double y) {
  return x * (sin(x) + cos(y));
}

void computeDerivativeX(const std::vector<std::vector<double>>& a,
                        std::vector<std::vector<double>>& b, const double dx) {
  int rows = a.size();
  int columns = a[0].size();

#pragma omp parallel for collapse(2)
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < columns - 1; ++j) {
      if (j == 0) {
        b[i][j] = (a[i][j + 1] - a[i][j]) / dx;
      } else {
        b[i][j] = (a[i][j + 1] - a[i][j - 1]) / (2 * dx);
      }
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "USAGE: ./src/Task3/OpenMP/openmp_derivative <THREADS>\n";
    return EXIT_FAILURE;
  }
  omp_set_num_threads(std::stoi(argv[1]));

  std::vector<int> matrix_sizes = {10, 100, 1000, 10000};
  for (auto size : matrix_sizes) {
    int rows = size;
    int cols = size;
    double dx = 0.01;
    std::vector<std::vector<double>> a(rows, std::vector<double>(cols));
    std::vector<std::vector<double>> b(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        a[i][j] = func(i * dx, j * dx);
      }
    }
    double start_time = omp_get_wtime();
    computeDerivativeX(a, b, dx);
    double end_time = omp_get_wtime();
    std::cout << "Matrix size: " << rows << "x" << cols
              << " Time elapsed: " << (end_time - start_time) << " seconds"
              << '\n';
  }

  return 0;
}
