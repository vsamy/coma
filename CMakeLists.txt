#
# Copyright 2020-2021 CNRS-UM LIRMM,CNRS-AIST JRL
#

# Version minimum
cmake_minimum_required(VERSION 3.8.2) # For c++17

set(CMAKE_CXX_STANDARD 17)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(PROJECT_NAME coma)
set(PROJECT_DESCRIPTION "JRL Generic coma (Comprehensive Motion Algebra) library")
set(PROJECT_URL "...")
set(PROJECT_VERSION 1.0.0)
set(PROJECT_USE_CMAKE_EXPORT TRUE)
set(DOXYGEN_USE_MATHJAX "YES")

include(cmake/base.cmake)
include(cmake/msvc-specific.cmake)

# SET(CXX_DISABLE_WERROR True)
project(${PROJECT_NAME} CXX)

option(BUILD_TESTING "Build unit tests." ON)
option(BUILD_EXAMPLE "Build example" ON)
option(BUILD_BENCHMARK "Build benchmarks" ON)

if(MSVC)
    set(CMAKE_MSVCIDE_RUN_PATH "\$(SolutionDir)/src/\$(Configuration)")
endif()

# External dependencies
add_project_dependency(Eigen3 REQUIRED NO_MODULE)

# Minimize warnings
if (MSVC)
    add_compile_options(/W4 /WX)
endif()

add_subdirectory(include)

if (${BUILD_TESTING})
    add_subdirectory(test)
endif()

if (${BUILD_EXAMPLE})
    add_subdirectory(example)
endif()

if (${BUILD_BENCHMARK})
    add_subdirectory(benchmark)
endif()