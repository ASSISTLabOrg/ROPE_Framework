#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "physics_prediction::physics_prediction" for configuration ""
set_property(TARGET physics_prediction::physics_prediction APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(physics_prediction::physics_prediction PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libphysics_prediction.so"
  IMPORTED_SONAME_NOCONFIG "libphysics_prediction.so"
  )

list(APPEND _cmake_import_check_targets physics_prediction::physics_prediction )
list(APPEND _cmake_import_check_files_for_physics_prediction::physics_prediction "${_IMPORT_PREFIX}/lib/libphysics_prediction.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
