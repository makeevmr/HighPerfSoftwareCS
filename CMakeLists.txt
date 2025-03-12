cmake_minimum_required(VERSION 3.23)

set(CMAKE_TOOLCHAIN_FILE
    "${CMAKE_CURRENT_SOURCE_DIR}/conan/Release/generators/conan_toolchain.cmake"
)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS_RELEASE "")
set(CMAKE_CXX_FLAGS_DEBUG "")

project(HighPerfSoftwareCS LANGUAGES CXX)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_subdirectory(src)

add_custom_target(Task1 DEPENDS
    mpi_hello_world
    openmp_hello_world
)
