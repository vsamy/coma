#
# Copyright 2020-2021 CNRS-UM LIRMM,CNRS-AIST JRL
#

add_executable(simple_example simple_example.cpp)
target_link_libraries(simple_example PUBLIC Eigen3::Eigen)
target_compile_definitions(simple_example PRIVATE _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)