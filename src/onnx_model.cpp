#include "onnx_model.h"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <algorithm>
#include <numeric>

namespace physics_prediction {

// PIMPL implementation to hide ONNX Runtime details
class ONNXModel::Impl {
public:
    Ort::Env env;
    Ort::Session session;
    Ort::SessionOptions session_options;
    Ort::AllocatorWithDefaultOptions allocator;
    
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<const char*> input_names_ptrs;
    std::vector<const char*> output_names_ptrs;
    
    std::map<std::string, std::vector<int64_t>> input_shapes;
    std::map<std::string, std::vector<int64_t>> output_shapes;
    
    InferenceConfig config;
    bool ready = false;
    
    Impl(const std::string& model_path, const InferenceConfig& cfg)
        : env(ORT_LOGGING_LEVEL_WARNING, "PhysicsPrediction"),
          session(nullptr),
          config(cfg) {
        
        try {
            // Configure session options
            session_options.SetIntraOpNumThreads(config.num_threads > 0 ? config.num_threads : 0);
            session_options.SetGraphOptimizationLevel(
                static_cast<GraphOptimizationLevel>(config.optimization_level));
            
            if (config.enable_profiling) {
                session_options.EnableProfiling("onnx_profile");
            }
            
            // Enable GPU if requested
            if (config.use_gpu) {
                #ifdef USE_CUDA
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = config.gpu_device_id;
                session_options.AppendExecutionProvider_CUDA(cuda_options);
                #else
                std::cerr << "Warning: GPU requested but CUDA not available. Using CPU.\n";
                #endif
            }
            
            // Create session
            #ifdef _WIN32
            std::wstring model_path_w(model_path.begin(), model_path.end());
            session = Ort::Session(env, model_path_w.c_str(), session_options);
            #else
            session = Ort::Session(env, model_path.c_str(), session_options);
            #endif
            
            // Get input info
            size_t num_inputs = session.GetInputCount();
            for (size_t i = 0; i < num_inputs; i++) {
                auto name_ptr = session.GetInputNameAllocated(i, allocator);
                std::string name(name_ptr.get());
                input_names.push_back(name);
                
                auto type_info = session.GetInputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                input_shapes[name] = tensor_info.GetShape();
            }
            
            // Get output info
            size_t num_outputs = session.GetOutputCount();
            for (size_t i = 0; i < num_outputs; i++) {
                auto name_ptr = session.GetOutputNameAllocated(i, allocator);
                std::string name(name_ptr.get());
                output_names.push_back(name);
                
                auto type_info = session.GetOutputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                output_shapes[name] = tensor_info.GetShape();
            }
            
            // Setup name pointers for inference
            for (const auto& name : input_names) {
                input_names_ptrs.push_back(name.c_str());
            }
            for (const auto& name : output_names) {
                output_names_ptrs.push_back(name.c_str());
            }
            
            ready = true;
            
        } catch (const Ort::Exception& e) {
            throw PredictionException("ONNX Runtime error: " + std::string(e.what()));
        }
    }
};

ONNXModel::ONNXModel(const std::string& model_path, const InferenceConfig& config)
    : pImpl(std::make_unique<Impl>(model_path, config)) {
}

ONNXModel::~ONNXModel() = default;

ONNXModel::ONNXModel(ONNXModel&&) noexcept = default;
ONNXModel& ONNXModel::operator=(ONNXModel&&) noexcept = default;

std::map<std::string, Tensor> ONNXModel::predict(
    const std::map<std::string, Tensor>& inputs) {
    
    if (!pImpl->ready) {
        throw PredictionException("Model not ready for inference");
    }
    
    try {
        // Create input tensors
        std::vector<Ort::Value> input_tensors;
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        
        for (const auto& name : pImpl->input_names) {
            auto it = inputs.find(name);
            if (it == inputs.end()) {
                throw PredictionException("Missing input: " + name);
            }
            
            const Tensor& tensor = it->second;
            
            // Validate size
            if (tensor.data.size() != tensor.size()) {
                throw PredictionException("Input tensor size mismatch for: " + name);
            }
            
            // Create ONNX tensor
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info,
                const_cast<float*>(tensor.data.data()),
                tensor.data.size(),
                tensor.shape.data(),
                tensor.shape.size()
            ));
        }
        
        // Run inference
        auto output_tensors = pImpl->session.Run(
            Ort::RunOptions{nullptr},
            pImpl->input_names_ptrs.data(),
            input_tensors.data(),
            input_tensors.size(),
            pImpl->output_names_ptrs.data(),
            pImpl->output_names_ptrs.size()
        );
        
        // Extract outputs
        std::map<std::string, Tensor> outputs;
        for (size_t i = 0; i < output_tensors.size(); i++) {
            const std::string& name = pImpl->output_names[i];
            
            auto type_info = output_tensors[i].GetTensorTypeAndShapeInfo();
            auto shape = type_info.GetShape();
            
            float* float_data = output_tensors[i].GetTensorMutableData<float>();
            size_t total_size = type_info.GetElementCount();
            
            std::vector<float> data(float_data, float_data + total_size);
            outputs[name] = Tensor(std::move(data), std::move(shape));
        }
        
        return outputs;
        
    } catch (const Ort::Exception& e) {
        throw PredictionException("ONNX inference error: " + std::string(e.what()));
    }
}

Tensor ONNXModel::predict(const Tensor& input) {
    if (pImpl->input_names.size() != 1 || pImpl->output_names.size() != 1) {
        throw PredictionException(
            "Single input/output inference requires model with exactly 1 input and 1 output");
    }
    
    std::map<std::string, Tensor> inputs;
    inputs[pImpl->input_names[0]] = input;
    
    auto outputs = predict(inputs);
    return outputs[pImpl->output_names[0]];
}

ModelMetadata ONNXModel::getMetadata() const {
    ModelMetadata metadata;
    metadata.input_names = pImpl->input_names;
    metadata.output_names = pImpl->output_names;
    metadata.input_shapes = pImpl->input_shapes;
    metadata.output_shapes = pImpl->output_shapes;
    metadata.model_type = "ONNX";
    return metadata;
}

void ONNXModel::printModelInfo() const {
    std::cout << "=== ONNX Model Information ===" << std::endl;
    std::cout << "\nInputs (" << pImpl->input_names.size() << "):" << std::endl;
    for (size_t i = 0; i < pImpl->input_names.size(); i++) {
        const auto& name = pImpl->input_names[i];
        std::cout << "  [" << i << "] " << name << " - Shape: [";
        const auto& shape = pImpl->input_shapes.at(name);
        for (size_t j = 0; j < shape.size(); j++) {
            if (j > 0) std::cout << ", ";
            if (shape[j] == -1) std::cout << "dynamic";
            else std::cout << shape[j];
        }
        std::cout << "]" << std::endl;
    }
    
    std::cout << "\nOutputs (" << pImpl->output_names.size() << "):" << std::endl;
    for (size_t i = 0; i < pImpl->output_names.size(); i++) {
        const auto& name = pImpl->output_names[i];
        std::cout << "  [" << i << "] " << name << " - Shape: [";
        const auto& shape = pImpl->output_shapes.at(name);
        for (size_t j = 0; j < shape.size(); j++) {
            if (j > 0) std::cout << ", ";
            if (shape[j] == -1) std::cout << "dynamic";
            else std::cout << shape[j];
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl;
}

bool ONNXModel::isReady() const {
    return pImpl && pImpl->ready;
}

std::string ONNXModel::getProfilingInfo() const {
    if (!pImpl->config.enable_profiling) {
        return "Profiling not enabled";
    }
    return "Check onnx_profile_*.json for profiling data";
}

} // namespace physics_prediction
