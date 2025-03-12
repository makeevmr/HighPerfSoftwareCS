add_subdirectory(Task1)

list(APPEND CPP_TARGETS
    mpi_hello_world
    openmp_hello_world
)

list(APPEND CUDA_TARGETS
    cuda_hello_world
)

list(APPEND GCC_RELEASE_BUILD_TYPE_COMPILE_FLAGS
    -DNDEBUG
    -O3
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
    foreach(target IN ZIP_LISTS CPP_TARGETS)
        target_compile_options(
            ${target_0}
            PUBLIC
            ${GCC_RELEASE_BUILD_TYPE_COMPILE_FLAGS}
        )
    endforeach()
    foreach(target IN ZIP_LISTS CUDA_TARGETS)
        target_compile_options(
            ${target_0}
            PUBLIC
            ${NVCC_RELEASE_BUILD_TYPE_COMPILE_FLAGS}
        )
    endforeach()
endif()

if(CMAKE_BUILD_TYPE STREQUAL Debug)
    message(STATUS "DEBUG BUILD TYPE")
    foreach(target IN ZIP_LISTS CPP_TARGETS)
        target_compile_options(
            ${target_0}
            PUBLIC
            ${GCC_DEBUG_BUILD_TYPE_COMPILE_FLAGS}
        )
    endforeach()
endif()


target_link_libraries(mpi_hello_world openmpi::openmpi)
target_link_libraries(openmp_hello_world llvm-openmp::llvm-openmp)
