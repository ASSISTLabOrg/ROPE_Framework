#ifndef UTILITIES_H
#define UTILITIES_H

#include "prediction_model.h"
#include <vector>
#include <utility>

namespace physics_prediction {

/**
 * @brief Helper class for sequence processing
 * 
 * Provides utilities for batching sequences and handling variable-length inputs
 */
class SequenceProcessor {
public:
    /**
     * @brief Pad sequences to same length
     * @param sequences Vector of sequences (each sequence is a vector of timesteps)
     * @param padding_value Value to use for padding
     * @return Padded tensor and original lengths
     */
    static std::pair<Tensor, std::vector<int64_t>> padSequences(
        const std::vector<std::vector<std::vector<float>>>& sequences,
        float padding_value = 0.0f);
    
    /**
     * @brief Create batched tensor from sequences
     * @param sequences Vector of sequences [batch_size][seq_len][features]
     * @return Tensor with shape [batch_size, seq_len, features]
     */
    static Tensor batchSequences(
        const std::vector<std::vector<std::vector<float>>>& sequences);
    
    /**
     * @brief Unbatch output tensor into individual sequences
     * @param tensor Batched tensor
     * @param batch_size Number of sequences in batch
     * @return Vector of individual sequences
     */
    static std::vector<std::vector<std::vector<float>>> unbatchSequences(
        const Tensor& tensor, int batch_size);
};

/**
 * @brief Tensor manipulation utilities
 */
class TensorUtils {
public:
    /**
     * @brief Compute mean squared error between tensors
     */
    static float computeMSE(const Tensor& a, const Tensor& b);
    
    /**
     * @brief Compute L2 norm of tensor
     */
    static float computeL2Norm(const Tensor& tensor);
    
    /**
     * @brief Compute mean of tensor values
     */
    static float computeMean(const Tensor& tensor);
    
    /**
     * @brief Compute standard deviation of tensor values
     */
    static float computeStdDev(const Tensor& tensor);
    
    /**
     * @brief Reshape tensor (does not copy data)
     */
    static Tensor reshape(const Tensor& tensor, const std::vector<int64_t>& new_shape);
    
    /**
     * @brief Concatenate tensors along a dimension
     */
    static Tensor concatenate(const std::vector<Tensor>& tensors, int64_t axis);
    
    /**
     * @brief Split tensor along a dimension
     */
    static std::vector<Tensor> split(const Tensor& tensor, int64_t axis, int64_t num_splits);
};

} // namespace physics_prediction

#endif // UTILITIES_H
