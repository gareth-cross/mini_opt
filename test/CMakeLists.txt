include(FetchContent)

# Download google test.
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.10.0
)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()

# Build testing helper library.
add_library(${PROJECT_NAME}_test_utils test_utils.cc)

# Depend on gtest.
add_dependencies(${PROJECT_NAME}_test_utils gtest)
target_link_libraries(${PROJECT_NAME}_test_utils PUBLIC ${PROJECT_NAME} gtest)

# Find all the test files.
FILE(GLOB ${PROJECT_NAME}_TEST_SOURCES "*_test.cc")

foreach(test_file ${${PROJECT_NAME}_TEST_SOURCES})
  # get just the filename w/o extension
  get_filename_component(test_name ${test_file} NAME_WE)

  # add test executable for every file
  add_executable(${test_name} ${test_file})

  # make sure it depends on the test utils
  add_dependencies(${test_name} ${PROJECT_NAME} ${PROJECT_NAME}_test_utils)
  target_link_libraries(${test_name} ${PROJECT_NAME}_test_utils gtest_main)

  # turn on maximum warnings
  if(MSVC)
    # USE_MATH_DEFINES is required for M_PI
    target_compile_options(${test_name} PRIVATE /W4 /WX /D_USE_MATH_DEFINES)
  else()
    target_compile_options(${test_name} PRIVATE -Wall -Wextra -pedantic -Werror)
  endif()

  if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Building ${test_name} with -DNDEBUG (build type = ${CMAKE_BUILD_TYPE})")
    target_compile_definitions(${test_name} PUBLIC NDEBUG)
  endif()

  add_test(${test_name} ${test_name})
endforeach(test_file)
