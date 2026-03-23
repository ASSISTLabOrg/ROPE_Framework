#include "utilities.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace physics_prediction {

// SequenceProcessor implementation

std::pair<Tensor, std::vector<int64_t>> SequenceProcessor::padSequences(
    const std::vector<std::vector<std::vector<float>>>& sequences,
    float padding_value) {
    
    if (sequences.empty()) {
        throw PredictionException("Empty sequence list");
    }
    
    size_t batch_size = sequences.size();
    size_t num_features = sequences[0][0].size();
    
    // Find max sequence length
    size_t max_len = 0;
    std::vector<int64_t> lengths;
    for (const auto& seq : sequences) {
        max_len = std::max(max_len, seq.size());
        lengths.push_back(seq.size());
    }
    
    // Create padded tensor [batch, max_len, features]
    std::vector<float> data(batch_size * max_len * num_features, padding_value);
    
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t t = 0; t < sequences[b].size(); t++) {
            for (size_t f = 0; f < num_features; f++) {
                size_t idx = b * max_len * num_features + t * num_features + f;
                data[idx] = sequences[b][t][f];
            }
        }
    }
    
    std::vector<int64_t> shape = {
        static_cast<int64_t>(batch_size),
        static_cast<int64_t>(max_len),
        static_cast<int64_t>(num_features)
    };
    
    return {Tensor(std::move(data), std::move(shape)), lengths};
}

Tensor SequenceProcessor::batchSequences(
    const std::vector<std::vector<std::vector<float>>>& sequences) {
    
    if (sequences.empty()) {
        throw PredictionException("Empty sequence list");
    }
    
    size_t batch_size = sequences.size();
    size_t seq_len = sequences[0].size();
    size_t num_features = sequences[0][0].size();
    
    // Validate all sequences have same length
    for (const auto& seq : sequences) {
        if (seq.size() != seq_len) {
            throw PredictionException(
                "All sequences must have same length for batching (use padSequences for variable length)");
        }
    }
    
    std::vector<float> data(batch_size * seq_len * num_features);
    
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t t = 0; t < seq_len; t++) {
            for (size_t f = 0; f < num_features; f++) {
                size_t idx = b * seq_len * num_features + t * num_features + f;
                data[idx] = sequences[b][t][f];
            }
        }
    }
    
    std::vector<int64_t> shape = {
        static_cast<int64_t>(batch_size),
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(num_features)
    };
    
    return Tensor(std::move(data), std::move(shape));
}

std::vector<std::vector<std::vector<float>>> SequenceProcessor::unbatchSequences(
    const Tensor& tensor, int batch_size) {
    
    if (tensor.shape.size() != 3) {
        throw PredictionException("Expected 3D tensor for unbatching");
    }
    
    int64_t seq_len = tensor.shape[1];
    int64_t num_features = tensor.shape[2];
    
    std::vector<std::vector<std::vector<float>>> sequences(batch_size);
    
    for (int b = 0; b < batch_size; b++) {
        sequences[b].resize(seq_len);
        for (int64_t t = 0; t < seq_len; t++) {
            sequences[b][t].resize(num_features);
            for (int64_t f = 0; f < num_features; f++) {
                size_t idx = b * seq_len * num_features + t * num_features + f;
                sequences[b][t][f] = tensor.data[idx];
            }
        }
    }
    
    return sequences;
}

// TensorUtils implementation

float TensorUtils::computeMSE(const Tensor& a, const Tensor& b) {
    if (a.data.size() != b.data.size()) {
        throw PredictionException("Tensors must have same size for MSE computation");
    }
    
    float mse = 0.0f;
    for (size_t i = 0; i < a.data.size(); i++) {
        float diff = a.data[i] - b.data[i];
        mse += diff * diff;
    }
    mse /= a.data.size();
    
    return mse;
}

float TensorUtils::computeL2Norm(const Tensor& tensor) {
    float sum = 0.0f;
    for (float val : tensor.data) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

float TensorUtils::computeMean(const Tensor& tensor) {
    if (tensor.data.empty()) {
        return 0.0f;
    }
    
    float sum = std::accumulate(tensor.data.begin(), tensor.data.end(), 0.0f);
    return sum / tensor.data.size();
}

float TensorUtils::computeStdDev(const Tensor& tensor) {
    if (tensor.data.empty()) {
        return 0.0f;
    }
    
    float mean = computeMean(tensor);
    
    float variance = 0.0f;
    for (float val : tensor.data) {
        float diff = val - mean;
        variance += diff * diff;
    }
    variance /= tensor.data.size();
    
    return std::sqrt(variance);
}

Tensor TensorUtils::reshape(const Tensor& tensor, const std::vector<int64_t>& new_shape) {
    // Validate new shape
    size_t new_size = 1;
    for (auto dim : new_shape) {
        if (dim <= 0) {
            throw PredictionException("All dimensions must be positive for reshape");
        }
        new_size *= dim;
    }
    
    if (new_size != tensor.data.size()) {
        throw PredictionException(
            "New shape must have same total size as original tensor");
    }
    
    // Create new tensor with same data, different shape
    return Tensor(tensor.data, new_shape);
}

Tensor TensorUtils::concatenate(const std::vector<Tensor>& tensors, int64_t axis) {
    if (tensors.empty()) {
        throw PredictionException("Cannot concatenate empty tensor list");
    }
    
    // Validate all tensors have same number of dimensions
    size_t num_dims = tensors[0].shape.size();
    for (const auto& t : tensors) {
        if (t.shape.size() != num_dims) {
            throw PredictionException("All tensors must have same number of dimensions");
        }
    }
    
    if (axis < 0 || axis >= static_cast<int64_t>(num_dims)) {
        throw PredictionException("Invalid axis for concatenation");
    }
    
    // Validate all other dimensions match
    for (size_t dim = 0; dim < num_dims; dim++) {
        if (dim == static_cast<size_t>(axis)) continue;
        
        int64_t expected_size = tensors[0].shape[dim];
        for (const auto& t : tensors) {
            if (t.shape[dim] != expected_size) {
                throw PredictionException(
                    "All tensors must have same size in non-concatenation dimensions");
            }
        }
    }
    
    // Calculate output shape
    std::vector<int64_t> out_shape = tensors[0].shape;
    for (size_t i = 1; i < tensors.size(); i++) {
        out_shape[axis] += tensors[i].shape[axis];
    }
    
    // Allocate output data
    size_t total_size = 1;
    for (auto dim : out_shape) total_size *= dim;
    std::vector<float> out_data;
    out_data.reserve(total_size);
    
    // Copy data (simplified for axis=0 case, general case more complex)
    if (axis == 0) {
        for (const auto& t : tensors) {
            out_data.insert(out_data.end(), t.data.begin(), t.data.end());
        }
    } else {
        throw PredictionException("Concatenation only implemented for axis=0");
    }
    
    return Tensor(std::move(out_data), std::move(out_shape));
}

std::vector<Tensor> TensorUtils::split(const Tensor& tensor, int64_t axis, int64_t num_splits) {
    if (axis < 0 || axis >= static_cast<int64_t>(tensor.shape.size())) {
        throw PredictionException("Invalid axis for split");
    }
    
    if (tensor.shape[axis] % num_splits != 0) {
        throw PredictionException("Tensor dimension must be divisible by num_splits");
    }
    
    int64_t split_size = tensor.shape[axis] / num_splits;
    std::vector<Tensor> results;
    
    // Simplified for axis=0 case
    if (axis == 0) {
        std::vector<int64_t> split_shape = tensor.shape;
        split_shape[0] = split_size;
        
        size_t elements_per_split = tensor.data.size() / num_splits;
        
        for (int64_t i = 0; i < num_splits; i++) {
            size_t start_idx = i * elements_per_split;
            std::vector<float> split_data(
                tensor.data.begin() + start_idx,
                tensor.data.begin() + start_idx + elements_per_split
            );
            results.push_back(Tensor(std::move(split_data), split_shape));
        }
    } else {
        throw PredictionException("Split only implemented for axis=0");
    }
    
    return results;
}

} // namespace physics_prediction
