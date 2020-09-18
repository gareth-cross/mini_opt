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

# Download geometry utils.
if (NOT DEFINED GEOMETRY_UTILS_DIRECTORY)
  FetchContent_Declare(
    geometry_utils
    GIT_REPOSITORY https://github.com/gareth-cross/geometry_utils
  )
  FetchContent_GetProperties(geometry_utils)
  if(NOT geometry_utils_POPULATED)
    FetchContent_Populate(geometry_utils)
    add_subdirectory(${geometry_utils_SOURCE_DIR} ${geometry_utils_BINARY_DIR})
  endif()
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
  message(STATUS "Discovered test: ${test_name}")
  add_executable(${test_name} ${test_file})

  # make sure it depends on the test utils
  add_dependencies(${test_name} ${PROJECT_NAME} ${PROJECT_NAME}_test_utils)
  add_dependencies(${test_name} geometry_utils)
  target_link_libraries(${test_name} ${PROJECT_NAME}_test_utils gtest_main)
  target_link_libraries(${test_name} geometry_utils)

  # turn on maximum warnings
  if(MSVC)
    # USE_MATH_DEFINES is required for M_PI
    target_compile_options(${test_name} PRIVATE /W4 /WX /D_USE_MATH_DEFINES)
  else()
    target_compile_options(${test_name} PRIVATE -Wall -Wextra -pedantic -Werror)
  endif()

  if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(${test_name} PUBLIC EIGEN_NO_DEBUG)
  endif()

  add_test(${test_name} ${test_name})
endforeach(test_file)
