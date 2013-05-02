include_directories(
    ${CHECK_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src
)

cuda_add_executable(and_test and.cu)
target_link_libraries(and_test gpuac ${CHECK_LIBRARIES})
add_test(and_test ${CMAKE_CURRENT_BINARY_DIR}/and_test)
