cmake_minimum_required(VERSION 3.23)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/conan/Release/generators/conan_toolchain.cmake")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(HighPerfSoftwareCS LANGUAGES CXX)

find_package(MPI REQUIRED)

add_executable(test Task1/src/main.cpp)

target_link_libraries(test openmpi::openmpi)