add_subdirectory(Task1)
add_subdirectory(Task2)
add_subdirectory(Task3)
add_subdirectory(Task4)

list(APPEND MPI_TARGETS
  mpi_hello_world
  mpi_array_sum
  mpi_derivative
)

list(APPEND OPENMP_TARGETS
  openmp_hello_world
  openmp_array_sum
  openmp_derivative
  openmp_matrix_composition
)

list(APPEND CUDA_TARGETS
  cuda_hello_world
  cuda_array_sum
  cuda_derivative
)

list(APPEND OBJ_LIBRARIES
  array_sum_objs
  generate_array_objs
)

# TODO toggle flags
list(APPEND GCC_RELEASE_BUILD_TYPE_COMPILE_FLAGS
  -DNDEBUG
  -O3

  # -Werror
  # -Wall
  # -Wextra
  # -Wpedantic
  # -Wcast-align
  # -Wcast-qual
  # -Wconversion
  # -Wctor-dtor-privacy
  # -Wenum-compare
  # -Wfloat-equal
  # -Wnon-virtual-dtor
  # -Wold-style-cast
  # -Woverloaded-virtual
  # -Wredundant-decls
  # -Wsign-conversion
  # -Wsign-promo
)

list(APPEND NVCC_RELEASE_BUILD_TYPE_COMPILE_FLAGS
  -arch=sm_61
  -DNDEBUG
  -O3
)

list(APPEND GCC_DEBUG_BUILD_TYPE_COMPILE_FLAGS
  -Werror
  -Wall
  -Wextra
  -Wpedantic
  -Wcast-align
  -Wcast-qual
  -Wconversion
  -Wctor-dtor-privacy
  -Wenum-compare
  -Wfloat-equal
  -Wnon-virtual-dtor
  -Wold-style-cast
  -Woverloaded-virtual
  -Wredundant-decls
  -Wsign-conversion
  -Wsign-promo
  -g
)

if(CMAKE_BUILD_TYPE STREQUAL Release)
  message(STATUS "RELEASE BUILD TYPE")
  foreach(target IN LISTS MPI_TARGETS)
    target_compile_options(
      ${target} PUBLIC ${GCC_RELEASE_BUILD_TYPE_COMPILE_FLAGS}
    )
  endforeach()
  foreach(target IN LISTS OPENMP_TARGETS)
    target_compile_options(
      ${target} PUBLIC ${GCC_RELEASE_BUILD_TYPE_COMPILE_FLAGS}
    )
  endforeach()
  foreach(target IN LISTS OBJ_LIBRARIES)
    target_compile_options(
      ${target} PUBLIC ${GCC_RELEASE_BUILD_TYPE_COMPILE_FLAGS}
    )
  endforeach()
  foreach(target IN LISTS CUDA_TARGETS)
    target_compile_options(
      ${target} PUBLIC ${NVCC_RELEASE_BUILD_TYPE_COMPILE_FLAGS}
    )
  endforeach()
endif()

if(CMAKE_BUILD_TYPE STREQUAL Debug)
  message(STATUS "DEBUG BUILD TYPE")
  foreach(target IN LISTS MPI_TARGETS)
    target_compile_options(
      ${target} PUBLIC ${GCC_DEBUG_BUILD_TYPE_COMPILE_FLAGS}
    )
  endforeach()
endif()

target_link_libraries(mpi_array_sum PUBLIC array_sum_objs)
target_link_libraries(mpi_array_sum PUBLIC generate_array_objs)

target_link_libraries(openmp_array_sum PUBLIC generate_array_objs)

target_link_libraries(cuda_array_sum PUBLIC generate_array_objs)

foreach(mpi_target IN LISTS MPI_TARGETS)
  target_link_libraries(${mpi_target} PUBLIC openmpi::openmpi)
endforeach()

foreach(openmp_target IN LISTS OPENMP_TARGETS)
  target_link_libraries(${openmp_target} PUBLIC llvm-openmp::llvm-openmp)
endforeach()
