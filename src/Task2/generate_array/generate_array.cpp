#include "generate_array.hpp"

#include <ctime>
#include <cstdlib>

#include <cassert>

int* generateArray(const std::size_t size, const int max) {
    assert(size > 0);
    int* arr = new int[size];
    if (arr == nullptr) {
        return nullptr;
    }
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for (std::size_t i = 0; i < size; ++i) {
        arr[i] = std::rand() % max;
    }
    return arr;
}
