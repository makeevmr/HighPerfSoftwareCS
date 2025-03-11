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

add_executable(mpi_hello_world Task1/src/mpi_hello_world.cpp)
add_executable(openmp_hello_world Task1/src/openmp_hello_world.cpp)

list(APPEND TARGETS
    mpi_hello_world
    openmp_hello_world
)

list(APPEND GCC_RELEASE_BUILD_TYPE_COMPILE_FLAGS
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
    foreach(target IN ZIP_LISTS TARGETS)
        target_compile_options(
            ${target_0}
            PUBLIC
            ${GCC_RELEASE_BUILD_TYPE_COMPILE_FLAGS}
        )
    endforeach()
endif()

if(CMAKE_BUILD_TYPE STREQUAL Debug)
    message(STATUS "DEBUG BUILD TYPE")
    foreach(target IN ZIP_LISTS TARGETS)
        target_compile_options(
            ${target_0}
            PUBLIC
            ${GCC_DEBUG_BUILD_TYPE_COMPILE_FLAGS}
        )
    endforeach()
endif()


target_link_libraries(mpi_hello_world openmpi::openmpi)
target_link_libraries(openmp_hello_world llvm-openmp::llvm-openmp)
