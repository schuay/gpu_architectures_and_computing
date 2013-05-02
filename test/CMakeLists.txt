include_directories(
    ${CHECK_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src
)

cuda_add_executable(and_test and.cu ${CMAKE_SOURCE_DIR}/src/sigpt.c)
target_link_libraries(and_test stl_and ${CHECK_LIBRARIES})
add_test(and_test ${CMAKE_CURRENT_BINARY_DIR}/and_test)
