#ifndef PREDICTION_MODEL_H
#define PREDICTION_MODEL_H

#include <functional>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>

namespace physics_prediction {

/**
 * @brief Tensor data structure for input/output
 */
struct Tensor {
    std::vector<float> data;
    std::vector<int64_t> shape;
    
    Tensor() = default;
    Tensor(std::vector<float> d, std::vector<int64_t> s) 
        : data(std::move(d)), shape(std::move(s)) {}
    
    size_t size() const {
        size_t total = 1;
        for (auto dim : shape) total *= dim;
        return total;
    }
    
    /**
     * @brief Get total number of elements
     */
    size_t numElements() const { return data.size(); }
    
    /**
     * @brief Check if tensor is empty
     */
    bool empty() const { return data.empty(); }
};

/**
 * @brief Configuration for model inference
 */
struct InferenceConfig {
    // Number of threads for inference (0 = auto)
    int num_threads = 0;
    
    // Use GPU if available
    bool use_gpu = false;
    
    // GPU device ID
    int gpu_device_id = 0;
    
    // Graph optimization level (0-3, higher = more optimization)
    int optimization_level = 3;
    
    // Enable profiling
    bool enable_profiling = false;
};

/**
 * @brief Model metadata information
 */
struct ModelMetadata {
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::map<std::string, std::vector<int64_t>> input_shapes;
    std::map<std::string, std::vector<int64_t>> output_shapes;
    std::string model_type;  // "onnx", "custom", etc.
    std::string description;
};

/**
 * @brief Exception class for prediction errors
 */
class PredictionException : public std::runtime_error {
public:
    explicit PredictionException(const std::string& message)
        : std::runtime_error(message) {}
};

/**
 * @brief Abstract base class for all prediction models
 * 
 * This interface allows for different types of models:
 * - Neural networks (LSTM, GRU, Transformers, CNNs)
 * - Traditional ML models
 * - Physics-based models
 * - Hybrid models
 * - Custom algorithms
 */
class PredictionModel {
public:
    virtual ~PredictionModel() = default;
    
    /**
     * @brief Run inference on input data
     * @param inputs Map of input name to tensor data
     * @return Map of output name to tensor data
     */
    virtual std::map<std::string, Tensor> predict(
        const std::map<std::string, Tensor>& inputs) = 0;
    
    /**
     * @brief Run inference with single input/output (convenience method)
     * @param input Input tensor
     * @return Output tensor
     */
    virtual Tensor predict(const Tensor& input) = 0;
    
    /**
     * @brief Get model metadata
     * @return Model metadata structure
     */
    virtual ModelMetadata getMetadata() const = 0;
    
    /**
     * @brief Print model information to stdout
     */
    virtual void printModelInfo() const = 0;
    
    /**
     * @brief Get model type identifier
     * @return String identifying the model type
     */
    virtual std::string getModelType() const = 0;
    
    /**
     * @brief Check if model is ready for inference
     * @return True if model is loaded and ready
     */
    virtual bool isReady() const = 0;
};

/**
 * @brief Factory for creating prediction models
 */
class ModelFactory {
public:
    /**
     * @brief Create a model from file path
     * @param model_path Path to model file
     * @param config Configuration for inference
     * @return Unique pointer to created model
     */
    static std::unique_ptr<PredictionModel> createModel(
        const std::string& model_path,
        const InferenceConfig& config = InferenceConfig());
    
    /**
     * @brief Register a custom model creator
     * @param extension File extension (e.g., ".onnx", ".custom")
     * @param creator Function to create model from path and config
     */
    using ModelCreator = std::function<std::unique_ptr<PredictionModel>(
        const std::string&, const InferenceConfig&)>;
    
    static void registerModelType(const std::string& extension, ModelCreator creator);

private:
    static std::map<std::string, ModelCreator>& getRegistry();
};

} // namespace physics_prediction

#endif // PREDICTION_MODEL_H
