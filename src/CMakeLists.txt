set(SOURCES
    sigpt.c
    util.c
    main.cu
    sigcmp.cu
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
)

cuda_add_library(consolidate consolidate.cu)

cuda_add_executable(stl_bench
    ${SOURCES}
)
target_link_libraries(stl_bench
    stl_and
    stl_evtl
    stl_not
    stl_or
    stl_alw
    stl_until
)

set(BUILD_SHARED_LIBS OFF)
cuda_add_library(stlcuda
    stlcuda.cu
    sigpt.c
    SHARED
)

add_subdirectory(operators)
add_subdirectory(parser)

# vim:set ts=4 sw=4 et:
