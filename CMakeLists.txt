cmake_minimum_required(VERSION 3.12.4)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
project(dccl CXX)

# Version
set(dccl_VERSION 1.0rc0)
set(dccl_build_VERSION 1.0rc0)

# C++ STANDARD
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# cacheline size
execute_process(COMMAND getconf LEVEL1_DCACHE_LINESIZE OUTPUT_VARIABLE CLSZ)
string(STRIP ${CLSZ} CLSZ)
set(CACHELINE_SIZE ${CLSZ})

if (${USE_VERBS_API})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_VERBS_API")
endif()
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall -ggdb -gdwarf-3 -ftemplate-backtrace-limit=0 -DEVALUATION")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mprefer-vector-width=512 -Wall -DEVALUATION -fcompare-debug-second")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -Wall -ggdb -gdwarf-3")

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

include(GNUInstallDirs)

if ( NOT DEFINED CMAKE_INSTALL_LIBDIR )
    set( CMAKE_INSTALL_LIBDIR lib )
endif ()

if ( NOT DEFINED CMAKE_INSTALL_INCLUDEDIR )
    set( CMAKE_INSTALL_INCLUDEDIR include )
endif ( )

# The mutils package exports its location information in the "old" way,
# with INCLUDE_DIRS and LIBRARIES variables

# mutils_FOUND
# mutils_INCLUDE_DIRS
# mutils_LIBRARIES
find_package(mutils REQUIRED)
if(mutils_FOUND)
    message(STATUS "Found mutils in ${mutils_INCLUDE_DIRS}")
    include_directories(${mutils_INCLUDE_DIRS})
endif()

# spdlog_FOUND
find_package(spdlog 1.3.1 REQUIRED)

# openssl_FOUND
find_package(OpenSSL REQUIRED)

# derecho
find_package(derecho CONFIG REQUIRED)

# openmpi
find_package(MPI)

# doxygen
find_package(Doxygen)

# cuda found
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit)
    set( CUDA_FOUND 1 )
endif()

# Derecho exports an IMPORT target, following the "new" way of
# doing things, but for some reason we still need to do this
# in order for some of the sub-components of Cascade to compile.
# Ideally, if every target that needs the Derecho headers declares
# its dependency on the IMPORT target derecho::derecho (i.e. with
# target_link_libraries), they will each automatically include the
# Derecho headers from the location specified by that target
if(derecho_FOUND)
    include_directories(${derecho_INCLUDE_DIRS})
endif()

set(ENABLE_EVALUATION 1)
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/dccl/config.h)

add_subdirectory(src/core)
add_subdirectory(src/utils)

add_library(dccl SHARED
    $<TARGET_OBJECTS:core>
)
target_link_libraries(dccl 
    derecho
    spdlog::spdlog
    OpenSSL::Crypto
    ${mutils_LIBRARIES}
)
add_dependencies(dccl core utils)

add_subdirectory(src/application)

# make install
install(TARGETS dccl EXPORT dcclTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/include/dccl
        ${CMAKE_CURRENT_BINARY_DIR}/include/dccl
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp")

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/dccl/dcclConfigVersion.cmake"
    VERSION ${dccl_VERSION}
    COMPATIBILITY AnyNewerVersion
)

export(EXPORT dcclTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/dccl/dcclTargets.cmake"
)

set(ConfigPackageLocation ${CMAKE_INSTALL_LIBDIR}/cmake/dccl)

configure_package_config_file(dcclConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/dccl/dcclConfig.cmake"
    INSTALL_DESTINATION ${ConfigPackageLocation}
    PATH_VARS CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR ConfigPackageLocation
)

install(EXPORT dcclTargets
        FILE dcclTargets.cmake
        NAMESPACE derecho::
        DESTINATION ${ConfigPackageLocation}
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/dccl/dcclConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/dccl/dcclConfigVersion.cmake"
    DESTINATION ${ConfigPackageLocation}
)

if (DOXYGEN_FOUND)
    # create docs
    doxygen_add_docs(docs
        ${CMAKE_SOURCE_DIR}/README.md
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ALL
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endif()
