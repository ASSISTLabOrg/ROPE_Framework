#ifndef PHYSICS_PREDICTION_H
#define PHYSICS_PREDICTION_H

/**
 * Physics Prediction Library
 * 
 * A flexible, high-performance C++ library for physics predictions using
 * machine learning models and custom algorithms.
 * 
 * Main components:
 * - PredictionModel: Abstract interface for all models
 * - ONNXModel: Neural network models via ONNX Runtime
 * - PhysicsPipeline: Drivers → Predictor → Decoder workflow
 * - StateSpaceTransformer: Encoder/Decoder utilities
 * - SequenceProcessor: Batch and sequence handling
 * 
 * Example usage:
 * 
 *   // Standard ONNX model
 *   ONNXModel model("lstm.onnx");
 *   Tensor output = model.predict(input);
 * 
 *   // Physics pipeline (drivers → latent → full state)
 *   PhysicsPipeline pipeline("predictor.onnx", "decoder.onnx", "encoder.onnx");
 *   Tensor prediction = pipeline.predict(drivers);
 * 
 *   // State space transformation
 *   StateSpaceTransformer transformer("encoder.onnx", "decoder.onnx");
 *   Tensor latent = transformer.encode(full_state);
 *   Tensor reconstructed = transformer.decode(latent);
 */

// Core interfaces
#include "prediction_model.h"

// Model implementations
#include "onnx_model.h"

// Pipelines
#include "physics_pipeline.h"

// Utilities
#include "utilities.h"

namespace physics_prediction {

// Version information
constexpr const char* VERSION = "2.0.0";
constexpr int VERSION_MAJOR = 2;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

} // namespace physics_prediction

#endif // PHYSICS_PREDICTION_H
