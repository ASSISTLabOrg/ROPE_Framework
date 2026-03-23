#include "physics_pipeline.h"
#include "onnx_model.h"
#include <iostream>
#include <algorithm>

namespace physics_prediction {

// PhysicsPipeline implementation

PhysicsPipeline::PhysicsPipeline(
    const std::string& predictor_path,
    const std::string& decoder_path,
    const std::string& encoder_path,
    const InferenceConfig& config) {
    
    // Load predictor (drivers → latent)
    predictor_ = ModelFactory::createModel(predictor_path, config);
    
    // Load decoder (latent → full state)
    decoder_ = ModelFactory::createModel(decoder_path, config);
    
    // Load encoder if provided (full state → latent, for validation/initialization)
    if (!encoder_path.empty()) {
        encoder_ = ModelFactory::createModel(encoder_path, config);
    }
    
    validatePipeline();
}

PhysicsPipeline::PhysicsPipeline(
    std::unique_ptr<PredictionModel> predictor,
    std::unique_ptr<PredictionModel> decoder,
    std::unique_ptr<PredictionModel> encoder)
    : predictor_(std::move(predictor)),
      decoder_(std::move(decoder)),
      encoder_(std::move(encoder)) {
    
    validatePipeline();
}

PhysicsPipeline::~PhysicsPipeline() = default;

PhysicsPipeline::PhysicsPipeline(PhysicsPipeline&&) noexcept = default;
PhysicsPipeline& PhysicsPipeline::operator=(PhysicsPipeline&&) noexcept = default;

void PhysicsPipeline::validatePipeline() {
    if (!predictor_ || !predictor_->isReady()) {
        throw PredictionException("Predictor model not ready");
    }
    
    if (!decoder_ || !decoder_->isReady()) {
        throw PredictionException("Decoder model not ready");
    }
    
    // Validate predictor output matches decoder input (latent dimension)
    auto predictor_meta = predictor_->getMetadata();
    auto decoder_meta = decoder_->getMetadata();
    
    if (!predictor_meta.output_shapes.empty() && !decoder_meta.input_shapes.empty()) {
        auto pred_out_shape = predictor_meta.output_shapes.begin()->second;
        auto dec_in_shape = decoder_meta.input_shapes.begin()->second;
        
        // Check latent dimension compatibility
        if (!pred_out_shape.empty() && !dec_in_shape.empty()) {
            int64_t pred_latent_dim = pred_out_shape.back();
            int64_t dec_latent_dim = dec_in_shape.back();
            
            if (pred_latent_dim != -1 && dec_latent_dim != -1 && 
                pred_latent_dim != dec_latent_dim) {
                throw PredictionException(
                    "Predictor output dimension (" + std::to_string(pred_latent_dim) + 
                    ") doesn't match decoder input dimension (" + std::to_string(dec_latent_dim) + ")");
            }
        }
    }
    
    // Validate encoder if present
    if (encoder_) {
        if (!encoder_->isReady()) {
            throw PredictionException("Encoder model not ready");
        }
        
        auto encoder_meta = encoder_->getMetadata();
        
        // Check encoder output matches decoder input
        if (!encoder_meta.output_shapes.empty() && !decoder_meta.input_shapes.empty()) {
            auto enc_out_shape = encoder_meta.output_shapes.begin()->second;
            auto dec_in_shape = decoder_meta.input_shapes.begin()->second;
            
            if (!enc_out_shape.empty() && !dec_in_shape.empty()) {
                int64_t enc_latent_dim = enc_out_shape.back();
                int64_t dec_latent_dim = dec_in_shape.back();
                
                if (enc_latent_dim != -1 && dec_latent_dim != -1 && 
                    enc_latent_dim != dec_latent_dim) {
                    throw PredictionException(
                        "Encoder output dimension (" + std::to_string(enc_latent_dim) + 
                        ") doesn't match decoder input dimension (" + std::to_string(dec_latent_dim) + ")");
                }
            }
        }
    }
}

Tensor PhysicsPipeline::predict(const Tensor& drivers) {
    // 1. Drivers → Latent (Predictor)
    Tensor latent = predictLatent(drivers);
    
    // 2. Latent → Full State (Decoder)
    return decode(latent);
}

std::map<std::string, Tensor> PhysicsPipeline::predict(
    const std::map<std::string, Tensor>& drivers) {
    
    // 1. Drivers → Latent
    auto latent_outputs = predictor_->predict(drivers);
    
    // 2. Latent → Full State
    return decoder_->predict(latent_outputs);
}

Tensor PhysicsPipeline::predictLatent(const Tensor& drivers) {
    return predictor_->predict(drivers);
}

Tensor PhysicsPipeline::decode(const Tensor& latent_state) {
    return decoder_->predict(latent_state);
}

Tensor PhysicsPipeline::encode(const Tensor& full_state) {
    if (!encoder_) {
        throw PredictionException("Encoder not available in this pipeline");
    }
    return encoder_->predict(full_state);
}

bool PhysicsPipeline::hasEncoder() const {
    return encoder_ != nullptr && encoder_->isReady();
}

std::vector<Tensor> PhysicsPipeline::predictBatch(const std::vector<Tensor>& batch_drivers) {
    std::vector<Tensor> results;
    results.reserve(batch_drivers.size());
    
    for (const auto& drivers : batch_drivers) {
        results.push_back(predict(drivers));
    }
    
    return results;
}

PhysicsPipeline::PipelineDimensions PhysicsPipeline::getDimensions() const {
    PipelineDimensions dims;
    
    auto predictor_meta = predictor_->getMetadata();
    auto decoder_meta = decoder_->getMetadata();
    
    if (!predictor_meta.input_shapes.empty()) {
        dims.driver_shape = predictor_meta.input_shapes.begin()->second;
    }
    
    if (!predictor_meta.output_shapes.empty()) {
        dims.latent_shape = predictor_meta.output_shapes.begin()->second;
    }
    
    if (!decoder_meta.output_shapes.empty()) {
        dims.full_state_shape = decoder_meta.output_shapes.begin()->second;
    }
    
    // Try to extract sequence length from predictor input
    if (dims.driver_shape.size() >= 2) {
        dims.sequence_length = dims.driver_shape[1];
    }
    
    return dims;
}

void PhysicsPipeline::printPipelineInfo() const {
    std::cout << "=== Physics Prediction Pipeline ===" << std::endl;
    
    std::cout << "\n[1] Predictor (" << predictor_->getModelType() << "):" << std::endl;
    std::cout << "    Drivers → Latent State" << std::endl;
    predictor_->printModelInfo();
    
    std::cout << "\n[2] Decoder (" << decoder_->getModelType() << "):" << std::endl;
    std::cout << "    Latent State → Full Physics State" << std::endl;
    decoder_->printModelInfo();
    
    if (encoder_) {
        std::cout << "\n[Optional] Encoder (" << encoder_->getModelType() << "):" << std::endl;
        std::cout << "    Full Physics State → Latent State" << std::endl;
        encoder_->printModelInfo();
    }
    
    auto dims = getDimensions();
    std::cout << "\nPipeline Flow:" << std::endl;
    std::cout << "  Drivers [";
    for (size_t i = 0; i < dims.driver_shape.size(); i++) {
        if (i > 0) std::cout << "x";
        if (dims.driver_shape[i] == -1) std::cout << "?";
        else std::cout << dims.driver_shape[i];
    }
    std::cout << "] → Latent [";
    for (size_t i = 0; i < dims.latent_shape.size(); i++) {
        if (i > 0) std::cout << "x";
        if (dims.latent_shape[i] == -1) std::cout << "?";
        else std::cout << dims.latent_shape[i];
    }
    std::cout << "] → Full State [";
    for (size_t i = 0; i < dims.full_state_shape.size(); i++) {
        if (i > 0) std::cout << "x";
        if (dims.full_state_shape[i] == -1) std::cout << "?";
        else std::cout << dims.full_state_shape[i];
    }
    std::cout << "]" << std::endl;
    
    // Calculate compression ratio if encoder available
    if (encoder_) {
        auto encoder_meta = encoder_->getMetadata();
        if (!encoder_meta.input_shapes.empty() && !dims.latent_shape.empty()) {
            auto full_shape = encoder_meta.input_shapes.begin()->second;
            
            size_t full_size = 1;
            size_t latent_size = 1;
            
            for (auto d : full_shape) if (d > 0) full_size *= d;
            for (auto d : dims.latent_shape) if (d > 0) latent_size *= d;
            
            if (latent_size > 0) {
                float compression = static_cast<float>(full_size) / latent_size;
                std::cout << "  Compression ratio: " << compression << "x" << std::endl;
            }
        }
    }
    
    std::cout << std::endl;
}

float PhysicsPipeline::getReconstructionError(const Tensor& full_state) {
    if (!encoder_) {
        throw PredictionException("Encoder required for reconstruction error calculation");
    }
    
    // Encode then decode
    Tensor latent = encode(full_state);
    Tensor reconstructed = decode(latent);
    
    // Calculate MSE
    if (reconstructed.data.size() != full_state.data.size()) {
        throw PredictionException("Reconstructed state size doesn't match input");
    }
    
    float mse = 0.0f;
    for (size_t i = 0; i < full_state.data.size(); i++) {
        float diff = full_state.data[i] - reconstructed.data[i];
        mse += diff * diff;
    }
    mse /= full_state.data.size();
    
    return mse;
}

// StateSpaceTransformer implementation

StateSpaceTransformer::StateSpaceTransformer(
    const std::string& encoder_path,
    const std::string& decoder_path,
    const InferenceConfig& config) {
    
    encoder_ = ModelFactory::createModel(encoder_path, config);
    decoder_ = ModelFactory::createModel(decoder_path, config);
}

StateSpaceTransformer::StateSpaceTransformer(
    std::unique_ptr<PredictionModel> encoder,
    std::unique_ptr<PredictionModel> decoder)
    : encoder_(std::move(encoder)),
      decoder_(std::move(decoder)) {
}

StateSpaceTransformer::~StateSpaceTransformer() = default;

StateSpaceTransformer::StateSpaceTransformer(StateSpaceTransformer&&) noexcept = default;
StateSpaceTransformer& StateSpaceTransformer::operator=(StateSpaceTransformer&&) noexcept = default;

Tensor StateSpaceTransformer::encode(const Tensor& full_state) {
    return encoder_->predict(full_state);
}

Tensor StateSpaceTransformer::decode(const Tensor& latent_state) {
    return decoder_->predict(latent_state);
}

std::vector<Tensor> StateSpaceTransformer::encodeBatch(const std::vector<Tensor>& full_states) {
    std::vector<Tensor> encoded;
    encoded.reserve(full_states.size());
    
    for (const auto& state : full_states) {
        encoded.push_back(encode(state));
    }
    
    return encoded;
}

std::vector<Tensor> StateSpaceTransformer::decodeBatch(const std::vector<Tensor>& latent_states) {
    std::vector<Tensor> decoded;
    decoded.reserve(latent_states.size());
    
    for (const auto& state : latent_states) {
        decoded.push_back(decode(state));
    }
    
    return decoded;
}

Tensor StateSpaceTransformer::reconstruct(const Tensor& full_state) {
    Tensor latent = encode(full_state);
    return decode(latent);
}

float StateSpaceTransformer::getReconstructionError(const Tensor& full_state) {
    Tensor reconstructed = reconstruct(full_state);
    
    if (reconstructed.data.size() != full_state.data.size()) {
        throw PredictionException("Reconstructed state size doesn't match input");
    }
    
    float mse = 0.0f;
    for (size_t i = 0; i < full_state.data.size(); i++) {
        float diff = full_state.data[i] - reconstructed.data[i];
        mse += diff * diff;
    }
    mse /= full_state.data.size();
    
    return mse;
}

float StateSpaceTransformer::getCompressionRatio() const {
    auto enc_meta = encoder_->getMetadata();
    auto dec_meta = decoder_->getMetadata();
    
    if (enc_meta.input_shapes.empty() || enc_meta.output_shapes.empty()) {
        return 1.0f;
    }
    
    auto full_shape = enc_meta.input_shapes.begin()->second;
    auto latent_shape = enc_meta.output_shapes.begin()->second;
    
    size_t full_size = 1;
    size_t latent_size = 1;
    
    for (auto d : full_shape) if (d > 0) full_size *= d;
    for (auto d : latent_shape) if (d > 0) latent_size *= d;
    
    if (latent_size == 0) return 1.0f;
    
    return static_cast<float>(full_size) / latent_size;
}

} // namespace physics_prediction
