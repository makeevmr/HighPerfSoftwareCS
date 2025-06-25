add_subdirectory(CUDA)
add_subdirectory(MPI)
add_subdirectory(OpenMP)

add_library(array_sum_objs OBJECT
  array_sum/array_sum.cpp
)

add_library(generate_array_objs OBJECT
  generate_array/generate_array.cpp
)
