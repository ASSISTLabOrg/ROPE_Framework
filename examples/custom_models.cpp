#include "physics_prediction.h"
#include <iostream>
#include <cmath>

using namespace physics_prediction;

/**
 * Example: Custom Model Implementation
 * 
 * This demonstrates how to create your own prediction models
 * (non-ML, physics-based, hybrid, etc.) and use them alongside
 * ONNX models in the pipeline.
 */

/**
 * Example: Simple physics-based predictor
 * 
 * This could be a physics simulation, empirical model, or
 * any custom algorithm.
 */
class PhysicsBasedPredictor : public PredictionModel {
public:
    PhysicsBasedPredictor() = default;
    
    std::map<std::string, Tensor> predict(
        const std::map<std::string, Tensor>& inputs) override {
        
        // For this example, we'll just do a simple transformation
        auto it = inputs.find("input");
        if (it == inputs.end()) {
            throw PredictionException("Input 'input' not found");
        }
        
        const Tensor& input = it->second;
        
        // Apply some physics-based transformation
        // Example: exponential decay model
        std::vector<float> output_data(input.data.size());
        for (size_t i = 0; i < input.data.size(); i++) {
            output_data[i] = input.data[i] * std::exp(-0.1f * i);
        }
        
        std::map<std::string, Tensor> outputs;
        outputs["output"] = Tensor(output_data, input.shape);
        
        return outputs;
    }
    
    Tensor predict(const Tensor& input) override {
        std::map<std::string, Tensor> inputs;
        inputs["input"] = input;
        auto outputs = predict(inputs);
        return outputs["output"];
    }
    
    ModelMetadata getMetadata() const override {
        ModelMetadata meta;
        meta.input_names = {"input"};
        meta.output_names = {"output"};
        meta.model_type = "PhysicsBased";
        meta.description = "Custom physics-based predictor";
        // Shapes are dynamic
        meta.input_shapes["input"] = {-1, -1, -1};
        meta.output_shapes["output"] = {-1, -1, -1};
        return meta;
    }
    
    void printModelInfo() const override {
        std::cout << "=== Physics-Based Model ===" << std::endl;
        std::cout << "Type: Custom physics model" << std::endl;
        std::cout << "Input: Dynamic shape" << std::endl;
        std::cout << "Output: Same as input" << std::endl;
        std::cout << "Algorithm: Exponential decay" << std::endl;
    }
    
    std::string getModelType() const override {
        return "PhysicsBased";
    }
    
    bool isReady() const override {
        return true;
    }
};

/**
 * Example: Linear interpolation model
 */
class LinearInterpolator : public PredictionModel {
public:
    LinearInterpolator(float weight = 0.5f) : weight_(weight) {}
    
    std::map<std::string, Tensor> predict(
        const std::map<std::string, Tensor>& inputs) override {
        
        auto it1 = inputs.find("input1");
        auto it2 = inputs.find("input2");
        
        if (it1 == inputs.end() || it2 == inputs.end()) {
            throw PredictionException("Both input1 and input2 required");
        }
        
        const Tensor& in1 = it1->second;
        const Tensor& in2 = it2->second;
        
        if (in1.data.size() != in2.data.size()) {
            throw PredictionException("Inputs must have same size");
        }
        
        std::vector<float> output_data(in1.data.size());
        for (size_t i = 0; i < in1.data.size(); i++) {
            output_data[i] = weight_ * in1.data[i] + (1.0f - weight_) * in2.data[i];
        }
        
        std::map<std::string, Tensor> outputs;
        outputs["output"] = Tensor(output_data, in1.shape);
        
        return outputs;
    }
    
    Tensor predict(const Tensor& input) override {
        throw PredictionException("Linear interpolator requires two inputs");
    }
    
    ModelMetadata getMetadata() const override {
        ModelMetadata meta;
        meta.input_names = {"input1", "input2"};
        meta.output_names = {"output"};
        meta.model_type = "LinearInterpolator";
        meta.description = "Weighted linear interpolation";
        meta.input_shapes["input1"] = {-1};
        meta.input_shapes["input2"] = {-1};
        meta.output_shapes["output"] = {-1};
        return meta;
    }
    
    void printModelInfo() const override {
        std::cout << "=== Linear Interpolator ===" << std::endl;
        std::cout << "Type: Custom interpolation model" << std::endl;
        std::cout << "Weight: " << weight_ << std::endl;
        std::cout << "Inputs: input1, input2" << std::endl;
        std::cout << "Output: Weighted average" << std::endl;
    }
    
    std::string getModelType() const override {
        return "LinearInterpolator";
    }
    
    bool isReady() const override {
        return true;
    }

private:
    float weight_;
};

int main() {
    std::cout << "=== Custom Model Implementation Example ===" << std::endl;
    std::cout << "\nThis demonstrates creating and using custom models" << std::endl;
    std::cout << "alongside ONNX models in the physics prediction library.\n" << std::endl;
    
    try {
        // Test 1: Physics-based predictor
        std::cout << "=== Test 1: Physics-Based Predictor ===" << std::endl;
        
        PhysicsBasedPredictor physics_model;
        physics_model.printModelInfo();
        
        // Create test input
        std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        Tensor input(input_data, {1, 5});
        
        std::cout << "\nInput: ";
        for (auto val : input_data) std::cout << val << " ";
        std::cout << std::endl;
        
        Tensor output = physics_model.predict(input);
        
        std::cout << "Output: ";
        for (auto val : output.data) std::cout << val << " ";
        std::cout << std::endl;
        
        // Test 2: Linear interpolator
        std::cout << "\n=== Test 2: Linear Interpolator ===" << std::endl;
        
        LinearInterpolator interpolator(0.7f);
        interpolator.printModelInfo();
        
        std::vector<float> data1 = {10.0f, 20.0f, 30.0f};
        std::vector<float> data2 = {5.0f, 15.0f, 25.0f};
        
        Tensor tensor1(data1, {3});
        Tensor tensor2(data2, {3});
        
        std::map<std::string, Tensor> inputs;
        inputs["input1"] = tensor1;
        inputs["input2"] = tensor2;
        
        std::cout << "\nInput1: ";
        for (auto val : data1) std::cout << val << " ";
        std::cout << "\nInput2: ";
        for (auto val : data2) std::cout << val << " ";
        std::cout << std::endl;
        
        auto interp_outputs = interpolator.predict(inputs);
        auto interp_result = interp_outputs["output"];
        
        std::cout << "Output (0.7*input1 + 0.3*input2): ";
        for (auto val : interp_result.data) std::cout << val << " ";
        std::cout << std::endl;
        
        // Test 3: Using custom model in a pipeline
        std::cout << "\n=== Test 3: Custom Model in Pipeline ===" << std::endl;
        std::cout << "You can combine custom models with ONNX models:" << std::endl;
        std::cout << "\nExample pipeline:" << std::endl;
        std::cout << "  1. PhysicsBasedPredictor (custom) → latent" << std::endl;
        std::cout << "  2. ONNXModel (LSTM) → refined latent" << std::endl;
        std::cout << "  3. ONNXModel (decoder) → full state" << std::endl;
        
        // This would work if you have ONNX models:
        // auto predictor = std::make_unique<PhysicsBasedPredictor>();
        // auto decoder = std::make_unique<ONNXModel>("decoder.onnx");
        // PhysicsPipeline pipeline(std::move(predictor), std::move(decoder));
        
        // Test 4: Registering custom model types
        std::cout << "\n=== Test 4: Model Factory Registration ===" << std::endl;
        std::cout << "You can register custom model file types:\n" << std::endl;
        
        std::cout << "// Register .physics file extension" << std::endl;
        std::cout << "ModelFactory::registerModelType(\".physics\"," << std::endl;
        std::cout << "    [](const std::string& path, const InferenceConfig& config) {" << std::endl;
        std::cout << "        return std::make_unique<PhysicsBasedPredictor>();" << std::endl;
        std::cout << "    });" << std::endl;
        std::cout << "\n// Then use it like:" << std::endl;
        std::cout << "auto model = ModelFactory::createModel(\"model.physics\");" << std::endl;
        
        std::cout << "\n✓ All custom model tests completed!" << std::endl;
        
        std::cout << "\n=== Use Cases for Custom Models ===" << std::endl;
        std::cout << "1. Physics-based simulations (FEM, CFD solvers)" << std::endl;
        std::cout << "2. Empirical models (lookup tables, interpolation)" << std::endl;
        std::cout << "3. Hybrid models (ML + physics)" << std::endl;
        std::cout << "4. Ensemble models (combine multiple predictions)" << std::endl;
        std::cout << "5. Surrogate models (fast approximations)" << std::endl;
        std::cout << "6. Domain-specific algorithms" << std::endl;
        
    } catch (const PredictionException& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
