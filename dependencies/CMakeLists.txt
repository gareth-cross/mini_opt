add_subdirectory(geometry_utils)

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

if(WF_BUILD_EXTRAS)
  add_gtest()
endif()

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
