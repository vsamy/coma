#
# Copyright 2020-2021 CNRS-UM LIRMM,CNRS-AIST JRL
#

enable_testing()

macro(addTest testName)
    add_executable(${testName} ${testName}.cpp)
    target_compile_definitions(${testName} PUBLIC DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS COMA_THROW_ON_ASSERT_FAIL)
    target_link_libraries(${testName} PUBLIC ${PROJECT_NAME} Eigen3::Eigen)
    # Adding a project configuration file (for MSVC only)
    generate_msvc_dot_user_file(${testName})

    add_test(${testName}Unit ${testName})
endmacro(addTest)

addTest(utils_test)
addTest(spatial_vector_test)
addTest(spatial_algebra_test)
addTest(lie_algebra_test)
addTest(cmtm_test)
addTest(lie_map_test)


