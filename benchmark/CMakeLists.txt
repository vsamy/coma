#
# Copyright 2020-2021 CNRS-UM LIRMM,CNRS-AIST JRL
#

find_package(benchmark REQUIRED)

macro(addBench benchName)
    add_executable(${benchName} ${benchName}.cpp)
    target_compile_definitions(${benchName} PRIVATE _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
    target_link_libraries(${benchName} PUBLIC ${PROJECT_NAME} Eigen3::Eigen benchmark::benchmark)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${benchName} PUBLIC /W4 /WX)
    else()
        target_compile_options(${benchName} PUBLIC -Wall)
    endif()
    # Adding a project configuration file (for MSVC only)
    generate_msvc_dot_user_file(${benchName})
endmacro()

addBench(BM_CMTM)