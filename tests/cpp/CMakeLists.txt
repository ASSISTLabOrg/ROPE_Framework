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
set(ROPE_FIXTURE_MODELS   "${ROPE_FIXTURE_DIR}/models")
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

        catch_discover_tests(rope_pipeline_tests
            PROPERTIES FIXTURES_REQUIRED rope_pipeline_fixtures
        )
    else()
        catch_discover_tests(rope_pipeline_tests)
    endif()
endif()
