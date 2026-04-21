cmake_minimum_required(VERSION 4.2)
project(ROPE VERSION 0.2.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Version header — generated into build tree
# ---------------------------------------------------------------------------
configure_file(cmake/version.h.in
    "${CMAKE_CURRENT_BINARY_DIR}/include/rope/core/version.h" @ONLY)

# ---------------------------------------------------------------------------
# Build type
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(ROPE_USE_LIBTORCH  "Build the TorchScript / LibTorch backend" ON)
option(ROPE_BUILD_LEGACY  "Also build the legacy rope_demo target"   OFF)

# ---------------------------------------------------------------------------
# External dependencies (nlohmann/json, CLI11) via FetchContent
# ---------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.4.2
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nlohmann_json CLI11)

# ---------------------------------------------------------------------------
# ONNX Runtime + LibTorch — acquired via cmake/Dependencies.cmake.
# Set ROPE_DOWNLOAD_DEPS=ON to auto-fetch; or supply ONNXRUNTIME_ROOT /
# ORT_INC / ORT_LIB / Torch_DIR manually.
# ---------------------------------------------------------------------------
include(cmake/Dependencies.cmake)

if(ORT_INC AND ORT_LIB)
    message(STATUS "ONNX Runtime include: ${ORT_INC}")
    message(STATUS "ONNX Runtime library: ${ORT_LIB}")
else()
    message(STATUS "ONNX Runtime not found — forecast/ will not build")
endif()

# LibTorch — optional, needed only by forecast/
if(ROPE_USE_LIBTORCH)
    set(LIBTORCH_ROOT "" CACHE PATH
        "Root of LibTorch distribution (sets Torch_DIR if Torch_DIR not already set)")
    if(LIBTORCH_ROOT AND NOT Torch_DIR)
        set(Torch_DIR "${LIBTORCH_ROOT}/share/cmake/Torch" CACHE PATH "" FORCE)
    endif()
    find_package(Torch QUIET)
    if(Torch_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
        message(STATUS "LibTorch found: ${TORCH_INCLUDE_DIRS}")
    else()
        message(STATUS "LibTorch not found — COAE TorchScript backend disabled")
    endif()
endif()

# ---------------------------------------------------------------------------
# OpenMP — optional, used by forecast/ for parallel base-model rollouts
# ---------------------------------------------------------------------------
find_package(OpenMP)

# ---------------------------------------------------------------------------
# Compiler warnings and release optimisations
# ---------------------------------------------------------------------------
add_library(rope_compile_opts INTERFACE)
target_compile_options(rope_compile_opts INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang>>:-O3 -march=native>
    $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:/O2>
)

# ---------------------------------------------------------------------------
# Symbol visibility — hidden by default on Unix shared libraries
# ---------------------------------------------------------------------------
set(CMAKE_C_VISIBILITY_PRESET   hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

# All objects that end up in a shared library must be PIC.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# rope_core — platform layer (types, datetime, IpcSocket, spawn)
# ---------------------------------------------------------------------------
add_library(rope_core STATIC)

target_sources(rope_core PRIVATE
    $<IF:$<PLATFORM_ID:Windows>,
        src/core/platform/windows.cpp,
        src/core/platform/posix.cpp>
    src/core/version.cpp
)

target_include_directories(rope_core
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
)
target_link_libraries(rope_core
    PUBLIC  rope_compile_opts
)
if(WIN32)
    target_link_libraries(rope_core PUBLIC ws2_32)
endif()

# ---------------------------------------------------------------------------
# rope_io — CSV parsing, config, stats, driver DB, IC table
# ---------------------------------------------------------------------------
add_library(rope_io STATIC
    src/io/csv_reader.cpp
    src/io/config_reader.cpp
    src/io/driver_db.cpp
    src/io/ic_table.cpp
)
target_link_libraries(rope_io PUBLIC rope_core)

# ---------------------------------------------------------------------------
# rope_interpolate — in-memory grid interpolation
# ---------------------------------------------------------------------------
add_library(rope_interpolate STATIC
    src/interpolate/grid_interpolator.cpp
)
target_link_libraries(rope_interpolate PUBLIC rope_core)

# ---------------------------------------------------------------------------
# rope_client — IPC socket transport + typed request/response API
# ---------------------------------------------------------------------------
add_library(rope_client STATIC
    src/client/client.cpp
)
target_include_directories(rope_client
    PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/client>
)
target_link_libraries(rope_client
    PUBLIC  rope_core
    PRIVATE nlohmann_json::nlohmann_json
)

# ---------------------------------------------------------------------------
# rope_forecast — full inference pipeline
# ---------------------------------------------------------------------------
if(ORT_INC AND ORT_LIB)
    add_library(rope_forecast STATIC
        src/forecast/pipeline.cpp
    )
    target_include_directories(rope_forecast
        PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/forecast>
                $<BUILD_INTERFACE:${ORT_INC}>
    )
    target_link_libraries(rope_forecast
        PUBLIC  rope_core
                rope_io
        PRIVATE ${ORT_LIB}
    )
    if(Torch_FOUND)
        target_link_libraries(rope_forecast PRIVATE ${TORCH_LIBRARIES})
        target_include_directories(rope_forecast PRIVATE ${TORCH_INCLUDE_DIRS})
        target_compile_definitions(rope_forecast PRIVATE ROPE_USE_LIBTORCH)
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(rope_forecast PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(rope_forecast PRIVATE ROPE_USE_OPENMP)
    endif()
    message(STATUS "rope_forecast: enabled")
else()
    message(STATUS "rope_forecast: disabled (ONNX Runtime not found)")
endif()

# ---------------------------------------------------------------------------
# rope_server — request router, cache, and pipeline integration
# ---------------------------------------------------------------------------
add_library(rope_server STATIC
    src/server/server.cpp
)
target_include_directories(rope_server
    PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/server>
)
target_link_libraries(rope_server
    PUBLIC  rope_core
            rope_interpolate
            rope_io
    PRIVATE nlohmann_json::nlohmann_json
)
if(TARGET rope_forecast)
    target_link_libraries(rope_server PRIVATE rope_forecast)
    target_compile_definitions(rope_server PRIVATE ROPE_HAS_FORECAST)
endif()

# ---------------------------------------------------------------------------
# rope (shared library) — C API
# ---------------------------------------------------------------------------
add_library(rope SHARED
    src/capi/rope.cpp
)
target_compile_definitions(rope PRIVATE ROPE_BUILDING_SHARED)
target_link_libraries(rope
    PRIVATE rope_client
            rope_interpolate
            nlohmann_json::nlohmann_json
)
# Runtime library path so the shared lib finds siblings in lib/
if(UNIX AND NOT APPLE)
    set_target_properties(rope PROPERTIES INSTALL_RPATH "$ORIGIN")
elseif(APPLE)
    set_target_properties(rope PROPERTIES INSTALL_RPATH "@loader_path")
endif()

# ---------------------------------------------------------------------------
# rope (executable) — CLI
# ---------------------------------------------------------------------------
add_executable(rope_exe
    src/cli/main.cpp
)
set_target_properties(rope_exe PROPERTIES OUTPUT_NAME rope)
target_link_libraries(rope_exe
    PRIVATE rope_client
            rope_server
            rope_io
            CLI11::CLI11
            nlohmann_json::nlohmann_json
)
if(TARGET rope_forecast)
    target_link_libraries(rope_exe PRIVATE rope_forecast)
endif()
get_filename_component(ORT_LIB_DIR "${ORT_LIB}" DIRECTORY)
if(ORT_LIB_DIR)
    set_target_properties(rope_exe PROPERTIES
        BUILD_RPATH  "${ORT_LIB_DIR}"
        INSTALL_RPATH "$ORIGIN/../lib"
    )
else()
    set_target_properties(rope_exe PROPERTIES
        INSTALL_RPATH "$ORIGIN/../lib"
    )
endif()

# ---------------------------------------------------------------------------
# Install rules
# ---------------------------------------------------------------------------
include(GNUInstallDirs)

install(TARGETS rope_exe
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS rope
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})   # .dll on Windows

install(FILES include/rope/capi/rope.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rope/capi)
install(FILES config/rope.conf DESTINATION config)

# Bundle ORT runtime libraries
if(ORT_LIB)
    get_filename_component(_ort_lib_dir "${ORT_LIB}" DIRECTORY)
    install(DIRECTORY "${_ort_lib_dir}/"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILES_MATCHING
            PATTERN "libonnxruntime*.so*"
            PATTERN "libonnxruntime*.dylib"
            PATTERN "libonnxruntime_providers*.so*")
    if(WIN32)
        install(DIRECTORY "${_ort_lib_dir}/"
            DESTINATION ${CMAKE_INSTALL_BINDIR}
            FILES_MATCHING PATTERN "onnxruntime*.dll")
    endif()
endif()

# Bundle LibTorch runtime libraries
if(Torch_FOUND)
    foreach(_lib IN LISTS TORCH_LIBRARIES)
        if(_lib MATCHES "\\.so" OR _lib MATCHES "\\.dylib")
            install(FILES "${_lib}" DESTINATION ${CMAKE_INSTALL_LIBDIR})
        endif()
    endforeach()
    if(WIN32)
        file(GLOB _torch_dlls "${LIBTORCH_ROOT}/lib/*.dll")
        install(FILES ${_torch_dlls} DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
endif()

# Model artifacts (optional — point -DROPE_MODELS_DIR= at the exported dir)
set(ROPE_MODELS_DIR "" CACHE PATH
    "Directory of exported model artifacts to bundle in the package")
if(ROPE_MODELS_DIR AND EXISTS "${ROPE_MODELS_DIR}")
    install(DIRECTORY "${ROPE_MODELS_DIR}/" DESTINATION models/exported)
endif()

# ---------------------------------------------------------------------------
# CPack
# ---------------------------------------------------------------------------
set(CPACK_PACKAGE_NAME             "rope_framework")
set(CPACK_PACKAGE_VENDOR           "ROPE")
set(CPACK_PACKAGE_VERSION          "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "ROPE atmospheric density forecasting runtime")
set(CPACK_STRIP_FILES TRUE)

if(NOT ROPE_PACKAGE_SUFFIX)
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(_cpack_os "macos")
    else()
        string(TOLOWER "${CMAKE_SYSTEM_NAME}" _cpack_os)
    endif()
    set(ROPE_PACKAGE_SUFFIX "${_cpack_os}-${CMAKE_SYSTEM_PROCESSOR}-${ROPE_HARDWARE}")
endif()
set(CPACK_PACKAGE_FILE_NAME
    "rope_framework-${PROJECT_VERSION}-${ROPE_PACKAGE_SUFFIX}")

if(WIN32)
    set(CPACK_GENERATOR "ZIP")
else()
    set(CPACK_GENERATOR "TGZ")
endif()

include(CPack)

# ---------------------------------------------------------------------------
# Legacy build (opt-in with -DROPE_BUILD_LEGACY=ON)
# ---------------------------------------------------------------------------
if(ROPE_BUILD_LEGACY)
    if(NOT ORT_INC OR NOT ORT_LIB)
        message(FATAL_ERROR "ROPE_BUILD_LEGACY requires ONNX Runtime")
    endif()
    add_library(rope_legacy STATIC
        legacy/src/rope.cpp
        legacy/src/interpolator.cpp
    )
    target_include_directories(rope_legacy
        PUBLIC legacy/include
               ${ORT_INC}
    )
    target_link_libraries(rope_legacy PUBLIC ${ORT_LIB})
    if(Torch_FOUND)
        target_link_libraries(rope_legacy PUBLIC ${TORCH_LIBRARIES})
        target_include_directories(rope_legacy PUBLIC ${TORCH_INCLUDE_DIRS})
        target_compile_definitions(rope_legacy PUBLIC ROPE_USE_LIBTORCH)
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(rope_legacy PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(rope_legacy PUBLIC ROPE_USE_OPENMP)
    endif()

    add_executable(rope_demo legacy/src/driver.cpp)
    target_link_libraries(rope_demo PRIVATE rope_legacy)
    if(ORT_LIB_DIR)
        set_target_properties(rope_demo PROPERTIES
            BUILD_RPATH "${ORT_LIB_DIR};${ORT_LIB_DIR}/dnnl/install/lib"
        )
    endif()
endif()
