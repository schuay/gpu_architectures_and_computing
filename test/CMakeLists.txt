include_directories(
    ${CHECK_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src
)

add_executable(and_test and.c)
target_link_libraries(and_test ${CHECK_LIBRARIES})
add_test(and_test ${CMAKE_CURRENT_BINARY_DIR}/and_test)
