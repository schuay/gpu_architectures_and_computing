if (NOT SIG_PATH)
    set(SIG_PATH ${CMAKE_SOURCE_DIR}/matlab/traces)
endif()

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/config.h"
)

include_directories(
    ${CHECK_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}
)

cuda_add_executable(and_test
    and.cu
    ${CMAKE_SOURCE_DIR}/src/sigpt.c
    ${CMAKE_SOURCE_DIR}/src/util.c
)
target_link_libraries(and_test stl_and ${CHECK_LIBRARIES})
add_test(and_test ${CMAKE_CURRENT_BINARY_DIR}/and_test)

cuda_add_executable(evtl_test
    evtl.cu
    ${CMAKE_SOURCE_DIR}/src/sigpt.c
    ${CMAKE_SOURCE_DIR}/src/util.c
)
target_link_libraries(evtl_test stl_evtl ${CHECK_LIBRARIES})
add_test(evtl_test ${CMAKE_CURRENT_BINARY_DIR}/evtl_test)
