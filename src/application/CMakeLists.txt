cmake_minimum_required(VERSION 3.12.4)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

add_executable(allreduce allreduce.cpp)
target_include_directories(allreduce PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
target_link_libraries(allreduce dccl)
add_dependencies(allreduce dccl)

add_custom_command(TARGET allreduce POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/cfg
        ${CMAKE_CURRENT_BINARY_DIR}/cfg
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n0/layout.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n1/layout.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/layout.json
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cfg/layout.json
    COMMENT "prepare demo configuration"
)
