include(FetchContent)

FetchContent_Declare(
    Catch2
    URL      https://github.com/catchorg/Catch2/archive/refs/tags/v3.7.1.tar.gz
    URL_HASH SHA256=c991b247a1a0d7bb9c39aa35faf0fe9e19764213f28ffba3109388e62ee0269c
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

set(ROPE_FIXTURE_DIR      "${CMAKE_CURRENT_SOURCE_DIR}/../fixtures")
set(ROPE_FIXTURE_MODELS   "${ROPE_FIXTURE_DIR}/test_models")
set(ROPE_CPP_FIXTURE_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/fixtures")

# ---------------------------------------------------------------------------
# Category A — pure-logic tests (no ONNX Runtime required)
# ---------------------------------------------------------------------------
add_executable(rope_tests
    test_interpolator.cpp
    test_csv_reader.cpp
    test_config_reader.cpp
    test_datetime.cpp
    test_version.cpp
)
target_link_libraries(rope_tests
    PRIVATE Catch2::Catch2WithMain
            rope_interpolate
            rope_io
            rope_core
)
target_compile_definitions(rope_tests PRIVATE
    ROPE_FIXTURE_DIR="${ROPE_FIXTURE_DIR}"
    ROPE_FIXTURE_MODELS="${ROPE_FIXTURE_MODELS}"
    ROPE_CPP_FIXTURE_DIR="${ROPE_CPP_FIXTURE_DIR}"
)
catch_discover_tests(rope_tests)

# ---------------------------------------------------------------------------
# Category B — pipeline integration test (requires ONNX Runtime)
# ---------------------------------------------------------------------------
if(TARGET rope_forecast)
    add_executable(rope_pipeline_tests
        test_pipeline.cpp
    )
    target_link_libraries(rope_pipeline_tests
        PRIVATE Catch2::Catch2WithMain
                rope_forecast
                rope_io
                rope_core
    )
    target_compile_definitions(rope_pipeline_tests PRIVATE
        ROPE_FIXTURE_DIR="${ROPE_FIXTURE_DIR}"
        ROPE_FIXTURE_MODELS="${ROPE_FIXTURE_MODELS}"
    )
    # Point the test binary at the ORT (and LibTorch) runtime libraries so it
    # can be run in-place by CTest without setting LD_LIBRARY_PATH.
    if(ORT_LIB)
        get_filename_component(_ort_rpath "${ORT_LIB}" DIRECTORY)
        set(_test_rpath "${_ort_rpath}")
        if(Torch_FOUND)
            if(TORCH_INSTALL_PREFIX)
                list(APPEND _test_rpath "${TORCH_INSTALL_PREFIX}/lib")
            else()
                list(GET TORCH_LIBRARIES 0 _first_torch)
                get_filename_component(_torch_dir "${_first_torch}" DIRECTORY)
                list(APPEND _test_rpath "${_torch_dir}")
            endif()
        endif()
        set_target_properties(rope_pipeline_tests PROPERTIES
            BUILD_RPATH "${_test_rpath}"
        )
    endif()

    # When LibTorch is linked, generate the .pt fixture at test time using the
    # exact LibTorch version in the build (guaranteeing compatibility).
    # Pass FIXTURES_REQUIRED via catch_discover_tests PROPERTIES so it is
    # applied to each individual test case (catch_discover_tests does not
    # register the target name itself as a CTest test).
    if(Torch_FOUND)
        add_executable(gen_decoder_pt ../gen_decoder_pt.cpp)
        target_link_libraries(gen_decoder_pt PRIVATE ${TORCH_LIBRARIES})
        target_include_directories(gen_decoder_pt PRIVATE ${TORCH_INCLUDE_DIRS})

        add_test(
            NAME    gen_coae_decoder_pt
            COMMAND gen_decoder_pt "${ROPE_FIXTURE_MODELS}/coae_decoder.pt"
        )
        set_tests_properties(gen_coae_decoder_pt PROPERTIES
            FIXTURES_SETUP rope_pipeline_fixtures
        )
    endif()

    # On Windows there is no RPATH equivalent, so ORT (and LibTorch when used)
    # DLLs are not found at test run time via BUILD_RPATH.  Use
    # ENVIRONMENT_MODIFICATION to prepend the required DLL directories to PATH
    # for every discovered test case.  This is encoded into the CTest metadata
    # at configure time and is reliable regardless of how ctest is invoked.
    set(_pipeline_test_props "")
    if(Torch_FOUND)
        list(APPEND _pipeline_test_props FIXTURES_REQUIRED rope_pipeline_fixtures)
    endif()
    if(WIN32 AND ORT_LIB)
        get_filename_component(_win_ort_dll_dir "${ORT_LIB}" DIRECTORY)
        list(APPEND _pipeline_test_props
             ENVIRONMENT_MODIFICATION "PATH=list_prepend:${_win_ort_dll_dir}")
        if(Torch_FOUND AND TORCH_INSTALL_PREFIX)
            list(APPEND _pipeline_test_props
                 ENVIRONMENT_MODIFICATION "PATH=list_prepend:${TORCH_INSTALL_PREFIX}/lib")
        elseif(Torch_FOUND AND TORCH_LIBRARIES)
            list(GET TORCH_LIBRARIES 0 _first_torch_dll)
            get_filename_component(_win_torch_dll_dir "${_first_torch_dll}" DIRECTORY)
            list(APPEND _pipeline_test_props
                 ENVIRONMENT_MODIFICATION "PATH=list_prepend:${_win_torch_dll_dir}")
        endif()
    endif()

    if(_pipeline_test_props)
        catch_discover_tests(rope_pipeline_tests PROPERTIES ${_pipeline_test_props})
    else()
        catch_discover_tests(rope_pipeline_tests)
    endif()
endif()

# ---------------------------------------------------------------------------
# Category C — CLI integration tests (requires ONNX Runtime + Python + pytest)
# ---------------------------------------------------------------------------
if(TARGET rope_forecast)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_FOUND)
        add_test(
            NAME    rope_cli_tests
            COMMAND ${Python3_EXECUTABLE} -m pytest
                    ${CMAKE_CURRENT_SOURCE_DIR}/../python/test_cli.py
                    -v --tb=short
        )
        set_tests_properties(rope_cli_tests PROPERTIES
            ENVIRONMENT
                "ROPE_EXE=$<TARGET_FILE:rope_exe>;ROPE_FIXTURE_DIR=${ROPE_FIXTURE_DIR}"
        )
        if(Torch_FOUND)
            set_tests_properties(rope_cli_tests PROPERTIES
                FIXTURES_REQUIRED rope_pipeline_fixtures
            )
        endif()
    endif()
endif()
