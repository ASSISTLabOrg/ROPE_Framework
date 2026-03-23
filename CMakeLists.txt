cmake_minimum_required(VERSION 3.15)
project(physics_prediction VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(USE_CUDA "Enable CUDA support" OFF)
option(BUILD_TESTS "Build tests" OFF)

# Find ONNX Runtime
# Priority: bundled external/ → ONNXRUNTIME_ROOT env var → system paths
set(ONNXRUNTIME_EXTERNAL_DIR ${CMAKE_SOURCE_DIR}/external/onnxruntime)

if(EXISTS ${ONNXRUNTIME_EXTERNAL_DIR})
    message(STATUS "Using bundled ONNX Runtime from external/")
    set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_EXTERNAL_DIR}/include)
    find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime
        PATHS ${ONNXRUNTIME_EXTERNAL_DIR}/lib
        NO_DEFAULT_PATH
    )
elseif(DEFINED ENV{ONNXRUNTIME_ROOT})
    message(STATUS "Using ONNX Runtime from ONNXRUNTIME_ROOT env var")
    set(ONNXRUNTIME_INCLUDE_DIR $ENV{ONNXRUNTIME_ROOT}/include)
    if(WIN32)
        set(ONNXRUNTIME_LIB $ENV{ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
    else()
        find_library(ONNXRUNTIME_LIB
            NAMES onnxruntime
            PATHS $ENV{ONNXRUNTIME_ROOT}/lib
            NO_DEFAULT_PATH
        )
    endif()
else()
    message(STATUS "Searching for ONNX Runtime in system paths...")
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATHS
            /usr/include/onnxruntime
            /usr/local/include/onnxruntime
    )
    find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime
        PATHS
            /usr/lib
            /usr/local/lib
    )
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime not found! Please set ONNXRUNTIME_ROOT or install ONNX Runtime.")
else()
    message(STATUS "Found ONNX Runtime:")
    message(STATUS "  Include: ${ONNXRUNTIME_INCLUDE_DIR}")
    message(STATUS "  Library: ${ONNXRUNTIME_LIB}")
endif()

# Main library - split into multiple source files
add_library(physics_prediction
    src/prediction_model.cpp
    src/onnx_model.cpp
    src/physics_pipeline.cpp
    src/utilities.cpp
)

target_include_directories(physics_prediction
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(physics_prediction
    PRIVATE
        ${ONNXRUNTIME_LIB}
)

if(USE_CUDA)
    target_compile_definitions(physics_prediction PRIVATE USE_CUDA)
endif()

# Platform-specific settings
if(WIN32)
    target_compile_definitions(physics_prediction PRIVATE _WIN32)
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(physics_prediction PRIVATE pthread)
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Installation
include(GNUInstallDirs)

install(TARGETS physics_prediction
    EXPORT physics_prediction_targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT physics_prediction_targets
    FILE physics_prediction_targets.cmake
    NAMESPACE physics_prediction::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/physics_prediction
)

# Generate config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/physics_prediction_config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/physics_prediction_config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/physics_prediction
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/physics_prediction_config_version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/physics_prediction_config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/physics_prediction_config_version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/physics_prediction
)
