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

cuda_add_executable(alw_test
    alw.cu
    ${CMAKE_SOURCE_DIR}/src/sigpt.c
    ${CMAKE_SOURCE_DIR}/src/util.c
    ${CMAKE_SOURCE_DIR}/src/sigcmp.cu
)
target_link_libraries(alw_test stl_alw ${CHECK_LIBRARIES})
add_test(alw_test ${CMAKE_CURRENT_BINARY_DIR}/alw_test)

cuda_add_executable(and_test
    and.cu
    ${CMAKE_SOURCE_DIR}/src/sigpt.c
    ${CMAKE_SOURCE_DIR}/src/util.c
    ${CMAKE_SOURCE_DIR}/src/sigcmp.cu
)
target_link_libraries(and_test stl_and ${CHECK_LIBRARIES})
add_test(and_test ${CMAKE_CURRENT_BINARY_DIR}/and_test)

cuda_add_executable(balw_test
    balw.cu
    ${CMAKE_SOURCE_DIR}/src/sigpt.c
    ${CMAKE_SOURCE_DIR}/src/util.c
    ${CMAKE_SOURCE_DIR}/src/sigcmp.cu
)
target_link_libraries(balw_test stl_balw ${CHECK_LIBRARIES})
add_test(balw_test ${CMAKE_CURRENT_BINARY_DIR}/balw_test)

cuda_add_executable(bevtl_test
    bevtl.cu
    ${CMAKE_SOURCE_DIR}/src/sigpt.c
    ${CMAKE_SOURCE_DIR}/src/util.c
    ${CMAKE_SOURCE_DIR}/src/sigcmp.cu
)
target_link_libraries(bevtl_test stl_bevtl ${CHECK_LIBRARIES})
add_test(bevtl_test ${CMAKE_CURRENT_BINARY_DIR}/bevtl_test)

cuda_add_executable(buntil_test
    buntil.cu
    ${CMAKE_SOURCE_DIR}/src/sigpt.c
    ${CMAKE_SOURCE_DIR}/src/util.c
    ${CMAKE_SOURCE_DIR}/src/sigcmp.cu
)
target_link_libraries(buntil_test stl_buntil ${CHECK_LIBRARIES})
add_test(buntil_test ${CMAKE_CURRENT_BINARY_DIR}/buntil_test)

cuda_add_executable(evtl_test
    evtl.cu
    ${CMAKE_SOURCE_DIR}/src/sigpt.c
    ${CMAKE_SOURCE_DIR}/src/util.c
    ${CMAKE_SOURCE_DIR}/src/sigcmp.cu
)
target_link_libraries(evtl_test stl_evtl ${CHECK_LIBRARIES})
add_test(evtl_test ${CMAKE_CURRENT_BINARY_DIR}/evtl_test)

cuda_add_executable(not_test
    not.cu
    ${CMAKE_SOURCE_DIR}/src/sigpt.c
    ${CMAKE_SOURCE_DIR}/src/util.c
    ${CMAKE_SOURCE_DIR}/src/sigcmp.cu
)
target_link_libraries(not_test stl_not ${CHECK_LIBRARIES})
add_test(not_test ${CMAKE_CURRENT_BINARY_DIR}/not_test)

cuda_add_executable(or_test
    or.cu
    ${CMAKE_SOURCE_DIR}/src/sigpt.c
    ${CMAKE_SOURCE_DIR}/src/util.c
    ${CMAKE_SOURCE_DIR}/src/sigcmp.cu
)
target_link_libraries(or_test stl_or ${CHECK_LIBRARIES})
add_test(or_test ${CMAKE_CURRENT_BINARY_DIR}/or_test)

cuda_add_executable(until_test
    until.cu
    ${CMAKE_SOURCE_DIR}/src/sigpt.c
    ${CMAKE_SOURCE_DIR}/src/util.c
    ${CMAKE_SOURCE_DIR}/src/sigcmp.cu
)
target_link_libraries(until_test stl_until ${CHECK_LIBRARIES})
add_test(until_test ${CMAKE_CURRENT_BINARY_DIR}/until_test)
