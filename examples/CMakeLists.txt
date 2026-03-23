add_executable(drivers_predictor_decoder drivers_predictor_decoder.cpp)
target_link_libraries(drivers_predictor_decoder PRIVATE physics_prediction ${ONNXRUNTIME_LIB})
target_include_directories(drivers_predictor_decoder PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})

add_executable(custom_models custom_models.cpp)
target_link_libraries(custom_models PRIVATE physics_prediction ${ONNXRUNTIME_LIB})
target_include_directories(custom_models PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})

add_executable(forecast forecast.cpp)
target_link_libraries(forecast PRIVATE physics_prediction ${ONNXRUNTIME_LIB})
target_include_directories(forecast PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})

# Install examples
install(TARGETS drivers_predictor_decoder custom_models forecast
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/examples
)
