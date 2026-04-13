cmake_minimum_required(VERSION 3.18)
project(ROPE VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# ONNX Runtime
# Defaults to the bundled copy in external/onnxruntime.
# Override with -DONNXRUNTIME_ROOT=/path/to/other/onnxruntime if needed.
# ---------------------------------------------------------------------------
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/onnxruntime"
    CACHE PATH "Root of the ONNX Runtime source/installation")

# Support both a prebuilt layout (include/ + lib/) and a source-build layout
# where the shared library lives in build/Linux/RelWithDebInfo/.
find_path(ORT_INC onnxruntime_cxx_api.h
    PATHS "${ONNXRUNTIME_ROOT}/include"
          "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session"
          /usr/include /usr/local/include
    NO_DEFAULT_PATH)
find_library(ORT_LIB NAMES onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}/lib"
          "${ONNXRUNTIME_ROOT}/build/Linux/RelWithDebInfo"
          "${ONNXRUNTIME_ROOT}/build/Linux/Release"
    PATHS /usr/lib /usr/local/lib
    NO_DEFAULT_PATH)

if(NOT ORT_INC OR NOT ORT_LIB)
    message(FATAL_ERROR
        "ONNX Runtime not found at ${ONNXRUNTIME_ROOT}.\n"
        "  Ensure external/onnxruntime/include and external/onnxruntime/lib exist,\n"
        "  or set -DONNXRUNTIME_ROOT=/path/to/onnxruntime on the cmake command line.")
endif()
message(STATUS "ONNX Runtime include: ${ORT_INC}")
message(STATUS "ONNX Runtime library: ${ORT_LIB}")

# ---------------------------------------------------------------------------
# LibTorch (TorchScript backend for the COAE decoder) — optional
# Set -DLIBTORCH_ROOT=/path/to/libtorch on the cmake command line, or install
# PyTorch into the active conda environment (find_package will locate it via
# Python's site-packages).
# ---------------------------------------------------------------------------
option(ROPE_USE_LIBTORCH "Build the TorchScript / LibTorch backend" ON)
set(LIBTORCH_ROOT "" CACHE PATH "Root of the LibTorch distribution")
if(ROPE_USE_LIBTORCH)
    if(LIBTORCH_ROOT)
        # Set Torch_DIR explicitly so find_package ignores conda/system paths
        # and goes straight to the requested libtorch installation.
        set(Torch_DIR "${LIBTORCH_ROOT}/share/cmake/Torch"
            CACHE PATH "Path to TorchConfig.cmake" FORCE)
    endif()
    find_package(Torch REQUIRED)
    # LibTorch may set a CXX ABI flag that must propagate to every TU.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    message(STATUS "LibTorch include dirs : ${TORCH_INCLUDE_DIRS}")
    message(STATUS "LibTorch libraries    : ${TORCH_LIBRARIES}")
else()
    message(STATUS "LibTorch backend disabled (ROPE_USE_LIBTORCH=OFF)")
endif()

# ---------------------------------------------------------------------------
# oneDNN / DNNL execution provider — optional
# Requires an ONNX Runtime source build compiled with --use_dnnl.
# ---------------------------------------------------------------------------
option(ROPE_USE_DNNL "Enable oneDNN/DNNL execution provider for ONNX Runtime" OFF)

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
add_library(rope_core STATIC
    src/rope.cpp
    src/interpolator.cpp
)

target_include_directories(rope_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${ORT_INC}>
)

target_link_libraries(rope_core PUBLIC ${ORT_LIB})

if(ROPE_USE_LIBTORCH)
    target_link_libraries(rope_core PUBLIC ${TORCH_LIBRARIES})
    target_include_directories(rope_core PUBLIC ${TORCH_INCLUDE_DIRS})
    target_compile_definitions(rope_core PUBLIC ROPE_USE_LIBTORCH)
endif()

if(ROPE_USE_DNNL)
    target_compile_definitions(rope_core PUBLIC ROPE_USE_DNNL)
    target_include_directories(rope_core PUBLIC
        $<BUILD_INTERFACE:${ORT_INC}/../providers/dnnl>
    )
endif()

# Optional: enable OpenMP for parallel base-model rollouts
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(rope_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(rope_core PUBLIC ROPE_USE_OPENMP)
    message(STATUS "OpenMP enabled — base-model rollouts run in parallel")
else()
    message(STATUS "OpenMP not found — base-model rollouts run sequentially")
endif()

# Demo executable
add_executable(rope_demo src/driver.cpp)
target_link_libraries(rope_demo PRIVATE rope_core)

# Ensure the runtime linker can find libdnnl.so next to libonnxruntime.so
# (only needed when using a source build rather than an installed package).
get_filename_component(ORT_LIB_DIR "${ORT_LIB}" DIRECTORY)
set_target_properties(rope_demo PROPERTIES
    BUILD_RPATH "${ORT_LIB_DIR};${ORT_LIB_DIR}/dnnl/install/lib"
    INSTALL_RPATH "${ORT_LIB_DIR};${ORT_LIB_DIR}/dnnl/install/lib"
)

# Compiler warnings
target_compile_options(rope_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# ---------------------------------------------------------------------------
# Build type defaults + profiling
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# -DENABLE_PROFILING=ON injects gprof instrumentation (-pg) into all targets.
# Build in RelWithDebInfo or Release for meaningful results (not Debug).
# Usage:
#   cmake -B build -DENABLE_PROFILING=ON [-DCMAKE_BUILD_TYPE=RelWithDebInfo]
#   cmake --build build
#   ./build/rope_demo config/rope.conf          # generates gmon.out
#   gprof build/rope_demo gmon.out > profile.txt
option(ENABLE_PROFILING "Build with gprof instrumentation (-pg)" OFF)
if(ENABLE_PROFILING)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # Must use target_* variants — add_compile/link_options only affect
        # targets defined *after* the call, but rope_core/rope_demo are already
        # defined above.
        foreach(_t rope_core rope_demo)
            target_compile_options(${_t} PUBLIC -pg)
            target_link_options(${_t} PUBLIC -pg)
        endforeach()
        message(STATUS "gprof profiling enabled (-pg)")
    else()
        message(WARNING "ENABLE_PROFILING is only supported with GCC/Clang; ignoring")
    endif()
endif()

# Release optimizations
target_compile_options(rope_core PRIVATE
    $<$<CONFIG:Release>:$<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native>>
    $<$<CONFIG:Release>:$<$<CXX_COMPILER_ID:MSVC>:/O2>>
)
