add_subdirectory(Task1)
add_subdirectory(Task2)

list(APPEND MPI_TARGETS
    mpi_hello_world
    mpi_array_sum
)

list(APPEND OPENMP_TARGETS
    openmp_hello_world
)

list(APPEND CUDA_TARGETS
    cuda_hello_world
)

list(APPEND OBJ_LIBRARIES
    array_sum_objs
)

# TODO change later
list(APPEND GCC_RELEASE_BUILD_TYPE_COMPILE_FLAGS
    # -DNDEBUG
    # -O3

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
            ${target}
            PUBLIC
            ${GCC_RELEASE_BUILD_TYPE_COMPILE_FLAGS}
        )
    endforeach()
    foreach(target IN LISTS OPENMP_TARGETS)
        target_compile_options(
            ${target}
            PUBLIC
            ${GCC_RELEASE_BUILD_TYPE_COMPILE_FLAGS}
        )
    endforeach()
    foreach(target IN LISTS OBJ_LIBRARIES)
        target_compile_options(
            ${target}
            PUBLIC
            ${GCC_RELEASE_BUILD_TYPE_COMPILE_FLAGS}
        )
    endforeach()
    foreach(target IN LISTS CUDA_TARGETS)
        target_compile_options(
            ${target}
            PUBLIC
            ${NVCC_RELEASE_BUILD_TYPE_COMPILE_FLAGS}
        )
    endforeach()
endif()

if(CMAKE_BUILD_TYPE STREQUAL Debug)
    message(STATUS "DEBUG BUILD TYPE")
    foreach(target IN LISTS MPI_TARGETS)
        target_compile_options(
            ${target}
            PUBLIC
            ${GCC_DEBUG_BUILD_TYPE_COMPILE_FLAGS}
        )
    endforeach()
endif()

target_link_libraries(mpi_array_sum PUBLIC array_sum_objs)

foreach(mpi_target IN LISTS MPI_TARGETS)
    target_link_libraries(${mpi_target} PUBLIC openmpi::openmpi)
endforeach()

foreach(openmp_target IN LISTS OPENMP_TARGETS)
    target_link_libraries(${openmp_target} PUBLIC llvm-openmp::llvm-openmp)
endforeach()
