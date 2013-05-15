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
target_link_libraries(gpuac
    stl_and
    stl_evtl
    stl_not
    stl_or
)

add_subdirectory(operators)
add_subdirectory(parser)

# vim:set ts=4 sw=4 et:
