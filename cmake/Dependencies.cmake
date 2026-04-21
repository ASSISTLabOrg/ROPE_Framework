# cmake/Dependencies.cmake — external runtime dependency acquisition.
#
# Set ROPE_DOWNLOAD_DEPS=ON to have CMake fetch ORT (and optionally LibTorch)
# automatically.  Default is OFF so developers with manually installed deps
# are not forced to re-download.
#
# Outputs (consumed by CMakeLists.txt):
#   ORT_INC   — include directory
#   ORT_LIB   — path to the import lib / shared lib
#   Torch_DIR — set before find_package(Torch) when ROPE_USE_LIBTORCH=ON

include(FetchContent)

option(ROPE_DOWNLOAD_DEPS
    "Automatically download ONNX Runtime (and LibTorch when ROPE_USE_LIBTORCH=ON)"
    OFF)

set(ROPE_HARDWARE "cpu" CACHE STRING
    "Hardware variant: cpu | cuda12 | cuda11 | rocm6")
set_property(CACHE ROPE_HARDWARE PROPERTY STRINGS cpu cuda12 cuda11 rocm6)

# ---------------------------------------------------------------------------
# Platform / architecture detection
# ---------------------------------------------------------------------------
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(_ROPE_PLAT "linux-aarch64")
    else()
        set(_ROPE_PLAT "linux-x86_64")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(_ROPE_PLAT "macos-arm64")
    else()
        set(_ROPE_PLAT "macos-x86_64")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(_ROPE_PLAT "windows-x64")
else()
    message(WARNING "Dependencies.cmake: unknown platform ${CMAKE_SYSTEM_NAME}, "
                    "set ORT_INC / ORT_LIB manually")
    set(_ROPE_PLAT "unknown")
endif()

message(STATUS "ROPE platform variant: ${_ROPE_PLAT}  hardware: ${ROPE_HARDWARE}")

# ---------------------------------------------------------------------------
# ONNX Runtime
# ---------------------------------------------------------------------------
set(_ORT_VERSION "1.25.0")
set(_ORT_BASE "https://github.com/microsoft/onnxruntime/releases/download/v${_ORT_VERSION}")

# GPU suffix for ORT archive name
if(ROPE_HARDWARE STREQUAL "cuda12")
    set(_ORT_GPU_SUFFIX "-gpu")
elseif(ROPE_HARDWARE STREQUAL "cuda11")
    set(_ORT_GPU_SUFFIX "-gpu")  # ORT 1.25 ships one CUDA build per platform
else()
    set(_ORT_GPU_SUFFIX "")
endif()

if(_ROPE_PLAT STREQUAL "linux-x86_64")
    if(ROPE_HARDWARE STREQUAL "rocm6")
        message(FATAL_ERROR
            "ROCm ORT builds are not on GitHub Releases. "
            "Run scripts/get-ort-libs.sh --hardware rocm6 to obtain them, "
            "then set ONNXRUNTIME_ROOT manually.")
    endif()
    set(_ORT_ARCHIVE "onnxruntime-linux-x64${_ORT_GPU_SUFFIX}-${_ORT_VERSION}.tgz")
    set(_ORT_ROOT_NAME "onnxruntime-linux-x64${_ORT_GPU_SUFFIX}-${_ORT_VERSION}")
elseif(_ROPE_PLAT STREQUAL "linux-aarch64")
    set(_ORT_ARCHIVE "onnxruntime-linux-aarch64-${_ORT_VERSION}.tgz")
    set(_ORT_ROOT_NAME "onnxruntime-linux-aarch64-${_ORT_VERSION}")
elseif(_ROPE_PLAT STREQUAL "macos-arm64")
    set(_ORT_ARCHIVE "onnxruntime-osx-arm64-${_ORT_VERSION}.tgz")
    set(_ORT_ROOT_NAME "onnxruntime-osx-arm64-${_ORT_VERSION}")
elseif(_ROPE_PLAT STREQUAL "macos-x86_64")
    set(_ORT_ARCHIVE "onnxruntime-osx-x86_64-${_ORT_VERSION}.tgz")
    set(_ORT_ROOT_NAME "onnxruntime-osx-x86_64-${_ORT_VERSION}")
elseif(_ROPE_PLAT STREQUAL "windows-x64")
    set(_ORT_ARCHIVE "onnxruntime-win-x64${_ORT_GPU_SUFFIX}-${_ORT_VERSION}.zip")
    set(_ORT_ROOT_NAME "onnxruntime-win-x64${_ORT_GPU_SUFFIX}-${_ORT_VERSION}")
endif()

if(ROPE_DOWNLOAD_DEPS AND DEFINED _ORT_ARCHIVE)
    set(_ORT_URL "${_ORT_BASE}/${_ORT_ARCHIVE}")
    FetchContent_Declare(onnxruntime
        URL "${_ORT_URL}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(onnxruntime)

    set(ORT_INC "${onnxruntime_SOURCE_DIR}/include" CACHE PATH
        "ONNX Runtime include directory" FORCE)

    if(WIN32)
        set(ORT_LIB "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib" CACHE FILEPATH
            "ONNX Runtime import library" FORCE)
        set(_ORT_DLL_DIR "${onnxruntime_SOURCE_DIR}/lib")
    elseif(APPLE)
        file(GLOB _ort_dylibs "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime*.dylib")
        list(GET _ort_dylibs 0 _ort_main)
        set(ORT_LIB "${_ort_main}" CACHE FILEPATH
            "ONNX Runtime shared library" FORCE)
    else()
        file(GLOB _ort_sos "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so*")
        # prefer the unversioned symlink (libonnxruntime.so) for linking
        foreach(_so IN LISTS _ort_sos)
            if(_so MATCHES "libonnxruntime\\.so$")
                set(_ort_link "${_so}")
            endif()
        endforeach()
        if(NOT DEFINED _ort_link)
            list(GET _ort_sos 0 _ort_link)
        endif()
        set(ORT_LIB "${_ort_link}" CACHE FILEPATH
            "ONNX Runtime shared library" FORCE)
    endif()

    message(STATUS "ORT_INC = ${ORT_INC}")
    message(STATUS "ORT_LIB = ${ORT_LIB}")
else()
    # Developer path: expect ONNXRUNTIME_ROOT to be set externally
    if(NOT DEFINED ORT_INC OR NOT DEFINED ORT_LIB)
        if(DEFINED ONNXRUNTIME_ROOT)
            set(ORT_INC "${ONNXRUNTIME_ROOT}/include")
            if(WIN32)
                set(ORT_LIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
            elseif(APPLE)
                file(GLOB _ort_dylibs "${ONNXRUNTIME_ROOT}/lib/libonnxruntime*.dylib")
                list(GET _ort_dylibs 0 ORT_LIB)
            else()
                set(ORT_LIB "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
            endif()
        else()
            message(FATAL_ERROR
                "ONNX Runtime not found. Either:\n"
                "  cmake -DROPE_DOWNLOAD_DEPS=ON ...          (auto-download)\n"
                "  cmake -DONNXRUNTIME_ROOT=/path/to/ort ...  (manual install)\n"
                "  cmake -DORT_INC=... -DORT_LIB=... ...      (explicit paths)")
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# LibTorch (optional, only when ROPE_USE_LIBTORCH=ON)
# ---------------------------------------------------------------------------
if(ROPE_USE_LIBTORCH AND ROPE_DOWNLOAD_DEPS)
    set(_TORCH_VERSION "2.7.0")

    if(_ROPE_PLAT STREQUAL "linux-x86_64")
        if(ROPE_HARDWARE STREQUAL "cuda12")
            set(_TORCH_URL
                "https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-${_TORCH_VERSION}%2Bcu124.zip")
        elseif(ROPE_HARDWARE STREQUAL "cuda11")
            set(_TORCH_URL
                "https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-${_TORCH_VERSION}%2Bcu118.zip")
        elseif(ROPE_HARDWARE STREQUAL "rocm6")
            set(_TORCH_URL
                "https://download.pytorch.org/libtorch/rocm6.2/libtorch-cxx11-abi-shared-with-deps-${_TORCH_VERSION}%2Brocm6.2.zip")
        else()
            set(_TORCH_URL
                "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${_TORCH_VERSION}%2Bcpu.zip")
        endif()
    elseif(_ROPE_PLAT STREQUAL "linux-aarch64")
        # PyTorch does not publish pre-built aarch64 CPU tarballs on download.pytorch.org.
        # Build from source or provide Torch_DIR manually.
        message(FATAL_ERROR
            "ROPE_USE_LIBTORCH=ON with auto-download is not supported on linux-aarch64. "
            "Build LibTorch from source and set Torch_DIR manually.")
    elseif(_ROPE_PLAT STREQUAL "macos-arm64")
        set(_TORCH_URL
            "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${_TORCH_VERSION}.zip")
    elseif(_ROPE_PLAT STREQUAL "macos-x86_64")
        set(_TORCH_URL
            "https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-${_TORCH_VERSION}.zip")
    elseif(_ROPE_PLAT STREQUAL "windows-x64")
        if(ROPE_HARDWARE STREQUAL "cuda12")
            set(_TORCH_URL
                "https://download.pytorch.org/libtorch/cu124/libtorch-win-shared-with-deps-${_TORCH_VERSION}%2Bcu124.zip")
        elseif(ROPE_HARDWARE STREQUAL "cuda11")
            set(_TORCH_URL
                "https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-${_TORCH_VERSION}%2Bcu118.zip")
        else()
            set(_TORCH_URL
                "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-${_TORCH_VERSION}%2Bcpu.zip")
        endif()
    endif()

    if(DEFINED _TORCH_URL)
        FetchContent_Declare(libtorch
            URL "${_TORCH_URL}"
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(libtorch)

        set(Torch_DIR "${libtorch_SOURCE_DIR}/share/cmake/Torch" CACHE PATH
            "LibTorch CMake config directory" FORCE)
        message(STATUS "Torch_DIR = ${Torch_DIR}")
    endif()
endif()
