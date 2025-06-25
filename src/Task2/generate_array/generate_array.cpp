#include "generate_array.hpp"

#include <ctime>
#include <cstdlib>

#include <cassert>

// Generate in place array of numbers from 0 to max - 1
void generateArray(int* arr, const std::size_t size, const int max) {
  assert(size > 0);
  assert(arr != nullptr);
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  for (std::size_t i = 0; i < size; ++i) {
    arr[i] = std::rand() % max;
  }
}
