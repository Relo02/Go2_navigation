# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_d1_sim_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED d1_sim_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(d1_sim_FOUND FALSE)
  elseif(NOT d1_sim_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(d1_sim_FOUND FALSE)
  endif()
  return()
endif()
set(_d1_sim_CONFIG_INCLUDED TRUE)

# output package information
if(NOT d1_sim_FIND_QUIETLY)
  message(STATUS "Found d1_sim: 0.1.0 (${d1_sim_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'd1_sim' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${d1_sim_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(d1_sim_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${d1_sim_DIR}/${_extra}")
endforeach()
