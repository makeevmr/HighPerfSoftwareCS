#include "array_sum.hpp"

[[nodiscard]] int arraySum(const int* arr, const std::size_t size) {
    int result = 0;
    for (std::size_t i = 0; i < size; ++i) {
        result += arr[i];
    }
    return result;
}
