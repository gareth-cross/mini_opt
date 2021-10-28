cmake_minimum_required(VERSION 3.17)
project(mini_opt VERSION 0.1
    DESCRIPTION "Optimization"
    LANGUAGES CXX)
enable_testing()
include(FetchContent)

set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${PROJECT_SOURCE_DIR}/cmake
    )

# Find the eigen package.
find_package(Eigen REQUIRED)
if (${Eigen_FOUND})
  MESSAGE("Eigen Includes: ${EIGEN_INCLUDE_DIRS}")
endif ()

option(INSTALL_LIBFMT "Include libfmt in installation" OFF)
option(INSTALL_EIGEN "Add a install target for Eigen public headers" OFF)

# Get libfmt
FetchContent_Declare(
    libfmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 8.0.1
)
FetchContent_GetProperties(libfmt)
if (NOT libfmt_POPULATED)
  FetchContent_Populate(libfmt)
  if (INSTALL_LIBFMT)
    # Don't exclude libfmt, we want it to be installed.
    set(FMT_INSTALL ON CACHE BOOL "" FORCE)
    add_subdirectory(${libfmt_SOURCE_DIR} ${libfmt_BINARY_DIR})
  else ()
    add_subdirectory(${libfmt_SOURCE_DIR} ${libfmt_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif ()
endif ()

# Check if user specified the path. If not, clone it from github.
if (NOT DEFINED ${GEOMETRY_UTILS_DIRECTORY})
  FetchContent_Declare(
      geometry_utils
      GIT_REPOSITORY https://github.com/gareth-cross/geometry_utils
  )
  FetchContent_GetProperties(geometry_utils)
  if (NOT geometry_utils_POPULATED)
    FetchContent_Populate(geometry_utils)
    add_subdirectory(${geometry_utils_SOURCE_DIR} ${geometry_utils_BINARY_DIR})
  endif ()
else ()
  # If they did specify the path, just include that directory directly.
  # It will get build as a dependency of this project.
  add_subdirectory(${GEOMETRY_UTILS_DIRECTORY})
endif ()

# Find all the source files.
FILE(GLOB ${PROJECT_NAME}_SOURCES "source/*.cc")

# Main project library
add_library(${PROJECT_NAME} STATIC ${${PROJECT_NAME}_SOURCES})

# Link to geometry utils
add_dependencies(${PROJECT_NAME} geometry_utils fmt Eigen::Eigen)
target_link_libraries(${PROJECT_NAME} geometry_utils fmt)
target_link_libraries(${PROJECT_NAME} Eigen::Eigen)

# Turn on C++14
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

# turn on maximum warnings
if (MSVC)
  # USE_MATH_DEFINES is required for M_PI
  target_compile_options(${PROJECT_NAME} PRIVATE /W4 /WX /D_USE_MATH_DEFINES /wd4244)
else ()
  target_compile_options(${PROJECT_NAME} PRIVATE
      -Wall -Wextra -pedantic -Werror -Wno-sign-compare)
endif ()

if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(${PROJECT_NAME} PUBLIC EIGEN_NO_DEBUG)
endif ()

# Specify both build and install include location
target_include_directories(${PROJECT_NAME}
    PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
    )

# make cache variables for install destinations
include(GNUInstallDirs)

# Specify public headers
FILE(GLOB_RECURSE ${PROJECT_NAME}_HEADERS "include/*.hpp")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${${PROJECT_NAME}_HEADERS}")

# Setup install target
install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}"
)

if (WIN32)
  configure_file("scripts/make.ps1" "make.ps1" COPYONLY)
endif ()

# Unit tests
add_subdirectory(test)

# Check if we want to install Eigen as well:
if (INSTALL_EIGEN)
  # Install eigen as well. We use DIRECTORY since INTERFACE target doesn't work w/ INSTALL
  install(DIRECTORY "${EIGEN_INCLUDE_DIRS}/Eigen" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif ()
