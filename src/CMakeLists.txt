cmake_minimum_required(VERSION 2.8.0)
project(gpuac)

set(VERSION 1)

if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE Debug)
endif()

find_package(CUDA REQUIRED)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_definitions("-Wextra -pedantic -Wno-long-long")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
endif()

set(SOURCES
    util.c
    main.cu
)

cuda_add_executable(gpuac
    ${SOURCES}
)

# vim:set ts=4 sw=4 et:
