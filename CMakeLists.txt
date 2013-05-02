cmake_minimum_required(VERSION 2.8.0)
project(gpuac)

set(VERSION 1)

if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE Debug)
endif()

find_package(CUDA REQUIRED /usr/local/cuda/lib64 )
find_package(FLEX REQUIRED)
find_package(BISON REQUIRED)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_definitions("-Wextra -pedantic -Wno-long-long")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
endif()

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --pre-include ${CMAKE_SOURCE_DIR}/src/gcc_cuda_compat.h")

add_subdirectory(src)

# vim:set ts=4 sw=4 et:
