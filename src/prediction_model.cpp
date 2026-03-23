#include "prediction_model.h"
#include "onnx_model.h"
#include <algorithm>
#include <filesystem>

namespace physics_prediction {

std::map<std::string, ModelFactory::ModelCreator>& ModelFactory::getRegistry() {
    static std::map<std::string, ModelCreator> registry;
    return registry;
}

void ModelFactory::registerModelType(const std::string& extension, ModelCreator creator) {
    getRegistry()[extension] = creator;
}

std::unique_ptr<PredictionModel> ModelFactory::createModel(
    const std::string& model_path,
    const InferenceConfig& config) {
    
    // Extract file extension
    std::filesystem::path path(model_path);
    std::string extension = path.extension().string();
    
    // Convert to lowercase for case-insensitive matching
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    // Check if file exists
    if (!std::filesystem::exists(path)) {
        throw PredictionException("Model file not found: " + model_path);
    }
    
    // Default: .onnx files use ONNXModel
    if (extension == ".onnx") {
        return std::make_unique<ONNXModel>(model_path, config);
    }
    
    // Check registered custom types
    auto& registry = getRegistry();
    auto it = registry.find(extension);
    if (it != registry.end()) {
        return it->second(model_path, config);
    }
    
    throw PredictionException(
        "Unknown model type: " + extension + 
        ". Supported: .onnx or registered custom types");
}

// Register default model types at library initialization
namespace {
    struct DefaultRegistration {
        DefaultRegistration() {
            // ONNX is already handled in createModel, but we could register it here too
            ModelFactory::registerModelType(".onnx", 
                [](const std::string& path, const InferenceConfig& config) {
                    return std::make_unique<ONNXModel>(path, config);
                });
        }
    };
    static DefaultRegistration default_registration;
}

} // namespace physics_prediction
