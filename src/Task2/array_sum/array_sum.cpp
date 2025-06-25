#include "array_sum.hpp"

[[nodiscard]] int64_t arraySum(const int* arr, const std::size_t size) {
    int64_t result = 0;
    for (std::size_t i = 0; i < size; ++i) {
        result += arr[i];
    }
    return result;
}
