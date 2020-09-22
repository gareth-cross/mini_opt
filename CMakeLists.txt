cmake_minimum_required(VERSION 3.17)
project(mini_opt VERSION 0.1
                 DESCRIPTION "Optimization"
                 LANGUAGES CXX)
enable_testing()
include(FetchContent)

# TODO(gareth): All this cmake is shameless copy-pasta from geometry_utils.
# Should probably just make a math_utils or something and put it all together.

set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  ${PROJECT_SOURCE_DIR}/cmake
)

# Find the eigen package.
find_package(Eigen REQUIRED)
if (${Eigen_FOUND})
  MESSAGE("Eigen Includes: ${EIGEN_INCLUDE_DIRS}")
endif()

# Download geometry utils.
FetchContent_Declare(
  geometry_utils
  GIT_REPOSITORY https://github.com/gareth-cross/geometry_utils
)
FetchContent_GetProperties(geometry_utils)
if(NOT geometry_utils_POPULATED)
  FetchContent_Populate(geometry_utils)
  add_subdirectory(${geometry_utils_SOURCE_DIR} ${geometry_utils_BINARY_DIR})
endif()

# Find all the source files.
FILE(GLOB ${PROJECT_NAME}_SOURCES "source/*.cc")

# Main project library
add_library(${PROJECT_NAME} STATIC ${${PROJECT_NAME}_SOURCES})

# Link to geometry utils
add_dependencies(${PROJECT_NAME} geometry_utils)
target_link_libraries(${PROJECT_NAME} geometry_utils)

# Turn on C++14
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

# turn on maximum warnings
if(MSVC)
	# USE_MATH_DEFINES is required for M_PI
	target_compile_options(${PROJECT_NAME} PRIVATE /W4 /WX /D_USE_MATH_DEFINES)
else()
        target_compile_options(${PROJECT_NAME} PRIVATE
            -Wall -Wextra -pedantic -Werror -Wno-sign-compare)
endif()

if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(${PROJECT_NAME} PUBLIC EIGEN_NO_DEBUG)
endif()

# Specify both build and install include location
# TODO(gareth): Not sure this is correct
target_include_directories(${PROJECT_NAME} 
  PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN_INCLUDE_DIRS}
)

# Unit tests
add_subdirectory(test)
