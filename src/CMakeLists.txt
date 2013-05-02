# Parser generation

bison_target(Parser parse.y ${CMAKE_CURRENT_BINARY_DIR}/parse.cpp)
flex_target(Scanner scan.l ${CMAKE_CURRENT_BINARY_DIR}/scan.cpp)
add_flex_bison_dependency(Scanner Parser)

add_executable(tempParser
    ${BISON_Parser_OUTPUTS}
    ${FLEX_Scanner_OUTPUTS}
)

target_link_libraries(tempParser
    ${FLEX_LIBRARIES}
)

# The main CUDA binary

cuda_add_library(stl_and
    and.cu
)

cuda_add_library(stl_evtl
    evtl.cu
)

set(SOURCES
    sigpt.c
    util.c
    main.cu
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
)

cuda_add_library(gpuac
    ${SOURCES}
)
target_link_libraries(gpuac
    stl_and
    stl_evtl
)

# vim:set ts=4 sw=4 et:
