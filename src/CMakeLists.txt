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

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --pre-include gcc_cuda_compat.h")

# Parser generation

bison_target(Parser parse.y ${CMAKE_CURRENT_BINARY_DIR}/parse.cpp)
flex_target(Scanner scan.l ${CMAKE_CURRENT_BINARY_DIR}/scan.cpp)
add_flex_bison_dependency(Scanner Parser)

message(${BISON_Parser_OUTPUT_HEADER})

add_executable(tempParser
    ${BISON_Parser_OUTPUTS}
    ${FLEX_Scanner_OUTPUTS}
)

target_link_libraries(tempParser
    ${FLEX_LIBRARIES}
)

# The main CUDA binary

set(SOURCES
    sigpt.c
    util.c
    main.cu
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
)

cuda_add_executable(gpuac
    ${SOURCES}
)

# vim:set ts=4 sw=4 et:
