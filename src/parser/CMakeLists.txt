include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
)

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
