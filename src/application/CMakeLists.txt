cmake_minimum_required(VERSION 3.12.4)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

add_executable(dccl_cli cli.cpp)
target_include_directories(dccl_cli PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
target_link_libraries(dccl_cli dccl)
if (CUDA_FOUND)
    target_link_libraries(dccl_cli CUDA::cudart)
endif()
add_dependencies(dccl_cli dccl)

add_custom_command(TARGET dccl_cli POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/cfg
        ${CMAKE_CURRENT_BINARY_DIR}/cfg
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n0/layout.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n1/layout.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/layout.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n3/layout.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n4/layout.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n5/layout.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n6/layout.json
    COMMAND ln -sf ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n7/layout.json
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cfg/layout.json
    COMMENT "prepare demo configuration"
)

if(MPI_CXX_FOUND AND MPI_CXX_VERSION VERSION_GREATER_EQUAL 3.1)
    add_executable(ompi_cli cli.cpp)
    target_include_directories(ompi_cli PRIVATE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${MPI_CXX_INCLUDE_DIRS}>)
    if (OMPI_WINDOW)
        target_compile_definitions(ompi_cli PRIVATE __BUILD_FOR_OMPI__ __USE_OMPI_WIN__)
    else()
        target_compile_definitions(ompi_cli PRIVATE __BUILD_FOR_OMPI__)
    endif()
    target_link_libraries(ompi_cli PUBLIC MPI::MPI_CXX dccl)
endif()

add_executable(p2p_perf p2p_perf.cpp)
target_include_directories(p2p_perf PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
if (CMAKE_CUDA_COMPILER)
    target_link_libraries(p2p_perf dccl derecho cuda CUDA::cudart)
else()
    target_link_libraries(p2p_perf dccl derecho)
endif()
add_dependencies(p2p_perf dccl)
