cmake_policy(SET CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

function(add_libfmt)
  set(FMT_TEST OFF)
  if(NOT TARGET fmt::fmt-header-only)
    add_subdirectory(fmt EXCLUDE_FROM_ALL)
  endif()
endfunction()
add_libfmt()

function(add_gtest)
  set(BUILD_GMOCK OFF)
  set(INSTALL_GTEST OFF)
  if(NOT TARGET gtest)
    add_subdirectory(googletest)
  endif()
endfunction()

if(MINI_OPT_BUILD_TESTS)
  add_gtest()
endif()

# Add nlohmann json
function(add_json)
  if(NOT TARGET nlohmann_json::nlohmann_json)
    add_subdirectory(json)
  endif()
endfunction()

if(MINI_OPT_SERIALIZATION)
  add_json()
endif()

# Create a target for eigen.
function(add_eigen)
  if(NOT TARGET eigen)
    set(EIGEN_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/eigen")
    add_library(eigen INTERFACE IMPORTED GLOBAL)
    set_target_properties(eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                           "${EIGEN_INCLUDE_DIRS}")
    add_library(Eigen3::Eigen ALIAS eigen)
  endif()
endfunction()
add_eigen()
