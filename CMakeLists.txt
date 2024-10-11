cmake_minimum_required(VERSION 3.20)
project(mini_opt CXX)
include(GNUInstallDirs)

# Options controlling compilation:
option(MINI_OPT_BUILD_TESTS "Build with tests enabled." ON)

if(MINI_OPT_BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory(dependencies)

if(MSVC)
  set(COMPILATION_FLAGS /W4 /WX /bigobj /D_USE_MATH_DEFINES)
else()
  set(COMPILATION_FLAGS -Wall -Wextra -pedantic -Werror)
endif()

# Main project library
set(LIBRARY_NAME ${PROJECT_NAME})
add_library(
  ${LIBRARY_NAME} STATIC
  include/mini_opt/assertions.hpp
  include/mini_opt/nonlinear.hpp
  include/mini_opt/qp.hpp
  include/mini_opt/residual.hpp
  include/mini_opt/structs.hpp
  source/nonlinear.cc
  source/qp.cc
  source/residual.cc
  source/structs.cc)
target_link_libraries(${LIBRARY_NAME} PUBLIC fmt::fmt-header-only eigen)
target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_17)
target_compile_options(${LIBRARY_NAME} PRIVATE ${COMPILATION_FLAGS})
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(${LIBRARY_NAME} PUBLIC EIGEN_NO_DEBUG)
endif()

target_include_directories(
  ${LIBRARY_NAME}
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
         "$<INSTALL_INTERFACE:include>")

install(
  TARGETS ${LIBRARY_NAME}
  EXPORT ${LIBRARY_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${LIBRARY_NAME}")

# Unit tests
if(MINI_OPT_BUILD_TESTS)
  add_subdirectory(test)
endif()
