#include "physics_prediction.h"
#include <iostream>
#include <random>
#include <chrono>

using namespace physics_prediction;

/**
 * Example: Drivers → Predictor → Decoder Pipeline
 * 
 * This demonstrates the physics prediction workflow where:
 * 1. External drivers (forcing functions, boundary conditions) are input
 * 2. Predictor (e.g., LSTM) predicts latent state from drivers
 * 3. Decoder reconstructs full physics state from latent
 * 
 * The encoder is available separately for encoding existing states.
 */

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <predictor.onnx> <decoder.onnx> [encoder.onnx]" << std::endl;
        std::cerr << "\nThis demonstrates the drivers → predictor → decoder pipeline." << std::endl;
        std::cerr << "\nRequired models:" << std::endl;
        std::cerr << "  predictor: Takes driver inputs, predicts latent state" << std::endl;
        std::cerr << "  decoder: Reconstructs full physics state from latent" << std::endl;
        std::cerr << "  encoder (optional): Encodes full state to latent (for validation)" << std::endl;
        return 1;
    }
    
    std::string predictor_path = argv[1];
    std::string decoder_path = argv[2];
    std::string encoder_path = (argc > 3) ? argv[3] : "";
    
    try {
        std::cout << "=== Physics Prediction Pipeline Demo ===" << std::endl;
        std::cout << "Version: " << VERSION << std::endl;
        std::cout << "\nModels:" << std::endl;
        std::cout << "  Predictor: " << predictor_path << std::endl;
        std::cout << "  Decoder: " << decoder_path << std::endl;
        if (!encoder_path.empty()) {
            std::cout << "  Encoder: " << encoder_path << std::endl;
        }
        std::cout << std::endl;
        
        // Configure for performance
        InferenceConfig config;
        config.num_threads = 4;
        config.optimization_level = 3;
        config.use_gpu = false;  // Set to true if GPU available
        
        // Create pipeline
        std::cout << "Loading pipeline..." << std::endl;
        PhysicsPipeline pipeline(predictor_path, decoder_path, encoder_path, config);
        
        // Print pipeline information
        pipeline.printPipelineInfo();
        
        // Get dimensions
        auto dims = pipeline.getDimensions();
        
        std::cout << "\n=== Test 1: Basic Prediction ===" << std::endl;
        
        // Create sample driver data
        // Example: drivers might be forcing functions, boundary conditions, etc.
        // Shape: [batch, seq_len, driver_features]
        int batch_size = 1;
        int seq_len = 10;
        int driver_features = 8;  // Adjust based on your model
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        std::vector<float> driver_data(batch_size * seq_len * driver_features);
        for (auto& val : driver_data) {
            val = dist(gen);
        }
        
        Tensor drivers(driver_data, {batch_size, seq_len, driver_features});
        
        std::cout << "Driver input shape: [";
        for (size_t i = 0; i < drivers.shape.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << drivers.shape[i];
        }
        std::cout << "]" << std::endl;
        
        // Run full pipeline
        std::cout << "\nRunning prediction..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        Tensor prediction = pipeline.predict(drivers);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Prediction completed in " << duration.count() << " ms" << std::endl;
        std::cout << "Output shape: [";
        for (size_t i = 0; i < prediction.shape.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << prediction.shape[i];
        }
        std::cout << "]" << std::endl;
        
        // Test individual pipeline components
        std::cout << "\n=== Test 2: Pipeline Components ===" << std::endl;
        
        start = std::chrono::high_resolution_clock::now();
        Tensor latent = pipeline.predictLatent(drivers);
        end = std::chrono::high_resolution_clock::now();
        auto predictor_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Predictor (drivers → latent): " << predictor_time.count() / 1000.0 << " ms" << std::endl;
        std::cout << "  Latent shape: [";
        for (size_t i = 0; i < latent.shape.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << latent.shape[i];
        }
        std::cout << "]" << std::endl;
        
        start = std::chrono::high_resolution_clock::now();
        Tensor decoded = pipeline.decode(latent);
        end = std::chrono::high_resolution_clock::now();
        auto decoder_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Decoder (latent → full state): " << decoder_time.count() / 1000.0 << " ms" << std::endl;
        std::cout << "  Full state shape: [";
        for (size_t i = 0; i < decoded.shape.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << decoded.shape[i];
        }
        std::cout << "]" << std::endl;
        
        // Test encoder if available
        if (pipeline.hasEncoder()) {
            std::cout << "\n=== Test 3: Encoder (for validation) ===" << std::endl;
            
            // Create a sample full state
            std::vector<float> full_state_data(decoded.data.size());
            for (auto& val : full_state_data) {
                val = dist(gen);
            }
            Tensor full_state(full_state_data, decoded.shape);
            
            // Test reconstruction quality
            float reconstruction_error = pipeline.getReconstructionError(full_state);
            
            std::cout << "Reconstruction test:" << std::endl;
            std::cout << "  Full state → Encoder → Latent → Decoder → Reconstructed" << std::endl;
            std::cout << "  MSE: " << reconstruction_error << std::endl;
            
            // Calculate relative error
            float state_norm = TensorUtils::computeL2Norm(full_state);
            float relative_error = std::sqrt(reconstruction_error) / state_norm * 100;
            std::cout << "  Relative error: " << relative_error << "%" << std::endl;
        }
        
        // Performance benchmark
        std::cout << "\n=== Test 4: Performance Benchmark ===" << std::endl;
        int num_iterations = 50;
        
        auto bench_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; i++) {
            pipeline.predict(drivers);
        }
        auto bench_end = std::chrono::high_resolution_clock::now();
        
        auto bench_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            bench_end - bench_start);
        
        double avg_time = bench_duration.count() / static_cast<double>(num_iterations);
        
        std::cout << "Average time (" << num_iterations << " iterations): " << avg_time << " ms" << std::endl;
        std::cout << "Throughput: " << 1000.0 / avg_time << " predictions/second" << std::endl;
        
        // Timing breakdown
        std::cout << "\nTiming breakdown (per prediction):" << std::endl;
        std::cout << "  Predictor: " << predictor_time.count() / 1000.0 << " ms" << std::endl;
        std::cout << "  Decoder: " << decoder_time.count() / 1000.0 << " ms" << std::endl;
        std::cout << "  Total: " << avg_time << " ms" << std::endl;
        
        // Test batch processing
        std::cout << "\n=== Test 5: Batch Processing ===" << std::endl;
        
        std::vector<Tensor> batch_drivers;
        for (int i = 0; i < 5; i++) {
            std::vector<float> batch_data(batch_size * seq_len * driver_features);
            for (auto& val : batch_data) {
                val = dist(gen);
            }
            batch_drivers.push_back(Tensor(batch_data, {batch_size, seq_len, driver_features}));
        }
        
        std::cout << "Processing batch of " << batch_drivers.size() << " driver sequences..." << std::endl;
        
        start = std::chrono::high_resolution_clock::now();
        auto batch_results = pipeline.predictBatch(batch_drivers);
        end = std::chrono::high_resolution_clock::now();
        
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Batch processing time: " << duration.count() << " ms" << std::endl;
        std::cout << "Average per sequence: " << duration.count() / static_cast<double>(batch_drivers.size()) 
                  << " ms" << std::endl;
        
        std::cout << "\n✓ All tests completed successfully!" << std::endl;
        
        // Summary
        std::cout << "\n=== Summary ===" << std::endl;
        std::cout << "Pipeline: Drivers → Predictor → Decoder" << std::endl;
        std::cout << "  Driver features: " << driver_features << std::endl;
        std::cout << "  Latent dimension: " << latent.data.size() / (batch_size * seq_len) << std::endl;
        std::cout << "  Full state size: " << decoded.data.size() / (batch_size * seq_len) << std::endl;
        std::cout << "  Average inference: " << avg_time << " ms" << std::endl;
        
    } catch (const PredictionException& e) {
        std::cerr << "Prediction error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
