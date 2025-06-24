cmake_minimum_required(VERSION 3.23)

set(CMAKE_TOOLCHAIN_FILE
    "${CMAKE_CURRENT_SOURCE_DIR}/conan/Release/generators/conan_toolchain.cmake"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS_RELEASE "")
set(CMAKE_CXX_FLAGS_DEBUG "")

project(HighPerfSoftwareCS LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_subdirectory(src)

add_custom_target(Task1 DEPENDS
    mpi_hello_world
    openmp_hello_world
    cuda_hello_world
)

add_custom_target(Task2 DEPENDS
    mpi_array_sum
)
