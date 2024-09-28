function(add_libfmt)
  set(FMT_TEST
      OFF
      CACHE BOOL "" FORCE)
  add_subdirectory(fmt EXCLUDE_FROM_ALL)
endfunction()
add_libfmt()

function(add_gtest)
  set(BUILD_GMOCK
      OFF
      CACHE BOOL "" FORCE)
  set(INSTALL_GTEST
      OFF
      CACHE BOOL "" FORCE)
  add_subdirectory(googletest)
endfunction()
add_gtest()

# Create a target for eigen.
function(add_eigen)
  set(EIGEN_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/eigen")
  if(NOT TARGET eigen)
    add_library(eigen INTERFACE IMPORTED GLOBAL)
    set_target_properties(eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                           "${EIGEN_INCLUDE_DIRS}")
  endif()
endfunction()
add_eigen()

function(add_geometry_utils)
  set(GEO_UTILS_BUILD_TESTING
      OFF
      CACHE BOOL "" FORCE)
  add_subdirectory(geometry_utils)
endfunction()
add_geometry_utils()
