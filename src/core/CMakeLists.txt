cmake_minimum_required(VERSION 3.12.4)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

if ( ${CUDA_FOUND} )
    add_library(cureduce OBJECT reduce.cu)
    target_include_directories(cureduce PRIVATE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    )
endif()

add_library(core OBJECT 
    dccl.cpp blob.cpp internal_common.cpp 
    reduce_scatter_recursive_halving.cpp
    reduce_scatter_ring.cpp
    all_gather_recursive_doubling.cpp
    all_gather_ring.cpp
    all_reduce_recursive_halving_and_doubling.cpp
    all_reduce_ring.cpp)
target_include_directories(core PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
if ( ${CUDA_FOUND} )
    target_include_directories(core PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
    )
endif()
