# FindEigen.cmake
list(APPEND EIGEN_CHECK_INCLUDE_DIRS
    /usr/local/include
    /usr/local/homebrew/include
    /opt/local/var/macports/software
    /opt/local/include
    /usr/include)

# Check if the user specified a location w/ -D, and look there as well.
if (DEFINED EIGEN_DIRECTORY)
  list(APPEND EIGEN_CHECK_INCLUDE_DIRS ${EIGEN_DIRECTORY})
endif(DEFINED EIGEN_DIRECTORY)

list(APPEND EIGEN_CHECK_PATH_SUFFIXES
     eigen3)

set(EIGEN_SIGNATURE signature_of_eigen3_matrix_library)

find_path(EIGEN_INCLUDE_DIR
        NAMES ${EIGEN_SIGNATURE}
        PATHS ${EIGEN_CHECK_INCLUDE_DIRS}
        PATH_SUFFIXES ${EIGEN_CHECK_PATH_SUFFIXES}
        NO_DEFAULT_PATH)

if(NOT EIGEN_INCLUDE_DIR OR NOT EXISTS ${EIGEN_INCLUDE_DIR})
  set(EIGEN_FOUND FALSE)
  message(WARNING "Failed to locate Eigen.")
else()
  set(EIGEN_FOUND TRUE)
  message(STATUS "Eigen found: ${EIGEN_INCLUDE_DIR}")
endif()

set(EIGEN_INCLUDE_DIRS ${EIGEN_INCLUDE_DIR})
