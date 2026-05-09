# Build testing helper library.
set(TEST_LIBRARY_NAME ${PROJECT_NAME}_test_utils)
add_library(${TEST_LIBRARY_NAME} STATIC test_utils.cc test_utils.hpp)
target_compile_options(${TEST_LIBRARY_NAME} PRIVATE ${COMPILATION_FLAGS})
target_link_libraries(${TEST_LIBRARY_NAME} PUBLIC ${LIBRARY_NAME} Eigen3::Eigen
                                                  gtest fmt::fmt-header-only)

# Function to add a unit test.
function(add_cpp_test NAME)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs SOURCE_FILES)
  cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})
  # add executable for the test:
  add_executable(${NAME} ${ARGS_SOURCE_FILES})
  target_link_libraries(${NAME} ${TEST_LIBRARY_NAME} gtest_main)
  target_compile_options(${NAME} PRIVATE ${COMPILATION_FLAGS})
  add_test(${NAME} ${NAME})
endfunction()

add_cpp_test(nonlinear_test SOURCE_FILES nonlinear_test.cc)
add_cpp_test(qp_test SOURCE_FILES qp_test.cc)
add_cpp_test(residual_test SOURCE_FILES residual_test.cc)
add_cpp_test(angle_utils_test SOURCE_FILES angle_utils_test.cc)
