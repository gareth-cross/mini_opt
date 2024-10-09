function(add_libfmt)
  if(NOT TARGET fmt::fmt)
    set(FMT_TEST
        OFF
        CACHE BOOL "" FORCE)
    add_subdirectory(fmt EXCLUDE_FROM_ALL)
  endif()
endfunction()
add_libfmt()

function(add_gtest)
  if(NOT TARGET gtest)
    set(BUILD_GMOCK
        OFF
        CACHE BOOL "" FORCE)
    set(INSTALL_GTEST
        OFF
        CACHE BOOL "" FORCE)
    add_subdirectory(googletest)
  endif()
endfunction()

if(MINI_OPT_BUILD_TESTS)
  add_gtest()
endif()

# Create a target for eigen.
function(add_eigen)
  if(NOT TARGET eigen)
    set(EIGEN_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/eigen")
    add_library(eigen INTERFACE IMPORTED GLOBAL)
    set_target_properties(eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                           "${EIGEN_INCLUDE_DIRS}")
  endif()
endfunction()
add_eigen()

function(add_geometry_utils)
  if(NOT TARGET geometry_utils)
    set(GEO_UTILS_BUILD_TESTING
        OFF
        CACHE BOOL "" FORCE)
    add_subdirectory(geometry_utils)
  endif()
endfunction()

if(MINI_OPT_BUILD_TESTS)
  add_geometry_utils()
endif()
