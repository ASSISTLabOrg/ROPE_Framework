#ifndef ONNX_MODEL_H
#define ONNX_MODEL_H

#include "prediction_model.h"
#include <memory>

namespace physics_prediction {

/**
 * @brief ONNX Runtime-based prediction model
 * 
 * This class wraps ONNX Runtime for neural network inference.
 * Supports LSTM, GRU, CNN, Transformer, and other ONNX models.
 */
class ONNXModel : public PredictionModel {
public:
    /**
     * @brief Construct an ONNX model
     * @param model_path Path to the ONNX model file
     * @param config Configuration for inference
     */
    ONNXModel(const std::string& model_path, 
              const InferenceConfig& config = InferenceConfig());
    
    /**
     * @brief Destructor
     */
    ~ONNXModel() override;
    
    // Prevent copying
    ONNXModel(const ONNXModel&) = delete;
    ONNXModel& operator=(const ONNXModel&) = delete;
    
    // Allow moving
    ONNXModel(ONNXModel&&) noexcept;
    ONNXModel& operator=(ONNXModel&&) noexcept;
    
    /**
     * @brief Run inference on input data
     */
    std::map<std::string, Tensor> predict(
        const std::map<std::string, Tensor>& inputs) override;
    
    /**
     * @brief Run inference with single input/output
     */
    Tensor predict(const Tensor& input) override;
    
    /**
     * @brief Get model metadata
     */
    ModelMetadata getMetadata() const override;
    
    /**
     * @brief Print model information
     */
    void printModelInfo() const override;
    
    /**
     * @brief Get model type identifier
     */
    std::string getModelType() const override { return "ONNX"; }
    
    /**
     * @brief Check if model is ready
     */
    bool isReady() const override;
    
    /**
     * @brief Get profiling information (if enabled)
     */
    std::string getProfilingInfo() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace physics_prediction

#endif // ONNX_MODEL_H
