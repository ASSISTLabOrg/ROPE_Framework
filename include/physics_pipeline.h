#ifndef PHYSICS_PIPELINE_H
#define PHYSICS_PIPELINE_H

#include "prediction_model.h"
#include <memory>

namespace physics_prediction {

/**
 * @brief Pipeline for drivers → predictor → decoder inference
 * 
 * This class manages the complete pipeline for physics simulations:
 * 1. External drivers (e.g., forcing functions, boundary conditions)
 * 2. Predictor model (predicts latent state from drivers)
 * 3. Decoder (reconstructs full physics state from latent)
 * 
 * The encoder is provided separately for encoding existing states
 * to latent space (e.g., for initialization or validation).
 */
class PhysicsPipeline {
public:
    /**
     * @brief Construct a pipeline with predictor and decoder
     * @param predictor_path Path to predictor model (e.g., LSTM, Transformer)
     * @param decoder_path Path to decoder model
     * @param encoder_path Optional path to encoder model (for state encoding)
     * @param config Configuration for inference
     */
    PhysicsPipeline(
        const std::string& predictor_path,
        const std::string& decoder_path,
        const std::string& encoder_path = "",
        const InferenceConfig& config = InferenceConfig());
    
    /**
     * @brief Construct with existing model instances
     * @param predictor Predictor model instance
     * @param decoder Decoder model instance
     * @param encoder Optional encoder model instance
     */
    PhysicsPipeline(
        std::unique_ptr<PredictionModel> predictor,
        std::unique_ptr<PredictionModel> decoder,
        std::unique_ptr<PredictionModel> encoder = nullptr);
    
    /**
     * @brief Destructor
     */
    ~PhysicsPipeline();
    
    // Prevent copying
    PhysicsPipeline(const PhysicsPipeline&) = delete;
    PhysicsPipeline& operator=(const PhysicsPipeline&) = delete;
    
    // Allow moving
    PhysicsPipeline(PhysicsPipeline&&) noexcept;
    PhysicsPipeline& operator=(PhysicsPipeline&&) noexcept;
    
    /**
     * @brief Run full pipeline: drivers → predict latent → decode to full state
     * @param drivers External driver data (forcing, boundary conditions, etc.)
     * @return Predicted full physics state
     */
    Tensor predict(const Tensor& drivers);
    
    /**
     * @brief Run full pipeline with named inputs/outputs
     * @param drivers Map of driver inputs
     * @return Map of predicted outputs
     */
    std::map<std::string, Tensor> predict(
        const std::map<std::string, Tensor>& drivers);
    
    /**
     * @brief Predict latent state from drivers (predictor only)
     * @param drivers External driver data
     * @return Predicted latent state
     */
    Tensor predictLatent(const Tensor& drivers);
    
    /**
     * @brief Decode latent state to full physics state (decoder only)
     * @param latent_state Latent representation
     * @return Full physics state
     */
    Tensor decode(const Tensor& latent_state);
    
    /**
     * @brief Encode full physics state to latent (encoder only, if available)
     * @param full_state Full physics state
     * @return Latent representation
     * @throws PredictionException if encoder not available
     */
    Tensor encode(const Tensor& full_state);
    
    /**
     * @brief Check if encoder is available
     */
    bool hasEncoder() const;
    
    /**
     * @brief Batch prediction with full pipeline
     * @param batch_drivers Batch of driver sequences
     * @return Batch of predictions in full state space
     */
    std::vector<Tensor> predictBatch(const std::vector<Tensor>& batch_drivers);
    
    /**
     * @brief Get dimensions of the pipeline
     */
    struct PipelineDimensions {
        std::vector<int64_t> driver_shape;
        std::vector<int64_t> latent_shape;
        std::vector<int64_t> full_state_shape;
        int64_t sequence_length;
    };
    
    PipelineDimensions getDimensions() const;
    
    /**
     * @brief Print pipeline information
     */
    void printPipelineInfo() const;
    
    /**
     * @brief Get reconstruction error (requires encoder)
     * Tests: full_state → encode → decode → compare with original
     * @param full_state Test state
     * @return Mean squared error
     * @throws PredictionException if encoder not available
     */
    float getReconstructionError(const Tensor& full_state);
    
    /**
     * @brief Access individual models
     */
    PredictionModel* getPredictor() { return predictor_.get(); }
    PredictionModel* getDecoder() { return decoder_.get(); }
    PredictionModel* getEncoder() { return encoder_.get(); }
    
    const PredictionModel* getPredictor() const { return predictor_.get(); }
    const PredictionModel* getDecoder() const { return decoder_.get(); }
    const PredictionModel* getEncoder() const { return encoder_.get(); }

private:
    std::unique_ptr<PredictionModel> predictor_;
    std::unique_ptr<PredictionModel> decoder_;
    std::unique_ptr<PredictionModel> encoder_;  // Optional
    
    void validatePipeline();
};

/**
 * @brief Helper for managing state space transformations
 * 
 * This class handles encoding/decoding operations separately from prediction.
 * Useful for pre-processing data or validating autoencoder quality.
 */
class StateSpaceTransformer {
public:
    StateSpaceTransformer(
        const std::string& encoder_path,
        const std::string& decoder_path,
        const InferenceConfig& config = InferenceConfig());
    
    StateSpaceTransformer(
        std::unique_ptr<PredictionModel> encoder,
        std::unique_ptr<PredictionModel> decoder);
    
    ~StateSpaceTransformer();
    
    StateSpaceTransformer(StateSpaceTransformer&&) noexcept;
    StateSpaceTransformer& operator=(StateSpaceTransformer&&) noexcept;
    
    /**
     * @brief Encode full state to latent
     */
    Tensor encode(const Tensor& full_state);
    
    /**
     * @brief Decode latent to full state
     */
    Tensor decode(const Tensor& latent_state);
    
    /**
     * @brief Encode batch of full states
     */
    std::vector<Tensor> encodeBatch(const std::vector<Tensor>& full_states);
    
    /**
     * @brief Decode batch of latent states
     */
    std::vector<Tensor> decodeBatch(const std::vector<Tensor>& latent_states);
    
    /**
     * @brief Test reconstruction: encode then decode
     */
    Tensor reconstruct(const Tensor& full_state);
    
    /**
     * @brief Get reconstruction error
     */
    float getReconstructionError(const Tensor& full_state);
    
    /**
     * @brief Get compression ratio
     */
    float getCompressionRatio() const;

private:
    std::unique_ptr<PredictionModel> encoder_;
    std::unique_ptr<PredictionModel> decoder_;
};

} // namespace physics_prediction

#endif // PHYSICS_PIPELINE_H
