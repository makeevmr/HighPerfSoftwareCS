add_subdirectory(MPI)

add_library(array_sum_objs OBJECT
    array_sum/array_sum.cpp
    generate_array/generate_array.cpp
)
