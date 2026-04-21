#pragma once
// meta_model.hpp — ensemble fusion meta-model, implementing IModel.
//
// Wraps the underlying meta ONNX model and performs the weighted ensemble
// fusion internally.  Callers pass a packed concatenation of the driver
// chunk and all base-model predictions; MetaModel returns the fused latent
// sequence with uncertainty.
//
// Input convention for infer():
//   input       : [x_chunk_slice (T*S*D) | all_preds (M*T*K)]  flat, concatenated
//   input_shape : {T, S*D, M, K}
//
// Output: [mean (T*K) | std (T*K)] flat, concatenated
//   output_shape : {T, K, 2}   — slice [*,*,0] = mean, [*,*,1] = std
//
// Two weight shapes produced by the underlying model are supported:
//   coeff_level (W shape T×M×K):  w[t,m,k] per latent dimension
//     mean[t,k] = sum_m  W[t,m,k] * p[m,t,k]
//     var[t,k]  = sum_m  W[t,m,k] * (p[m,t,k] - mean[t,k])^2
//   time_level  (W shape T×M):    shared w[t,m] across dimensions
//     mean[t,k] = sum_m  W[t,m] * p[m,t,k]
//     var[t,k]  = sum_m  W[t,m] * (p[m,t,k] - mean[t,k])^2
// The mode is inferred automatically from the model's actual output shape.
//
// Mirrors rope.py's MetaFusion.fuse().

#include <cassert>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "model_interface.hpp"

namespace rope {

struct FusionResult {
    std::vector<float> mean;  // (T*K) weighted mean of base-model predictions
    std::vector<float> std;   // (T*K) weighted standard deviation
};

class MetaModel : public IModel {
public:
    // inner       : wrapped ONNX / LibTorch meta model
    // K           : latent_dim
    // S           : seq_len
    // D           : total_dim  (latent + driver features)
    // M           : number of base models
    // coeff_level : hint for the fusion mode — overridden by the actual
    //               output shape at runtime if unambiguous
    MetaModel(std::unique_ptr<IModel> inner,
              int K, int S, int D, int M,
              bool coeff_level = true)
        : inner_(std::move(inner))
        , K_(K), S_(S), D_(D), M_(M)
        , coeff_level_(coeff_level)
    {}

    // input       : [x_chunk_slice (T*S*D) | all_preds (M*T*K)]
    // input_shape : {T, S*D, M, K}
    // Returns [mean (T*K) | std (T*K)]; output_shape = {T, K, 2}.
    std::vector<float> infer(
        const std::vector<float>& input,
        const std::vector<int64_t>& input_shape,
        std::vector<int64_t>& output_shape
    ) override {
        const auto [W, all_preds, T, coeff_mode] = run_inner(input, input_shape);
        FusionResult r = fuse(W, all_preds, T, coeff_mode);

        std::vector<float> out;
        out.reserve(r.mean.size() + r.std.size());
        out.insert(out.end(), r.mean.begin(), r.mean.end());
        out.insert(out.end(), r.std.begin(),  r.std.end());

        output_shape = {T, K_, 2};
        return out;
    }

    const std::string& name() const override { return inner_->name(); }

private:
    std::unique_ptr<IModel> inner_;
    int  K_, S_, D_, M_;
    bool coeff_level_;

    // Run the inner model and validate / unpack the inputs.
    // Returns {W, all_preds_ptr, T, coeff_mode}.
    struct InnerResult {
        std::vector<float> W;
        const float*       all_preds;
        int                T;
        bool               coeff_mode;
    };

    InnerResult run_inner(const std::vector<float>& input,
                          const std::vector<int64_t>& input_shape) const {
        if (input_shape.size() != 4)
            throw std::runtime_error(
                "MetaModel::infer: input_shape must be {T, S*D, M, K}");

        const int T = static_cast<int>(input_shape[0]);
        const size_t chunk_size = static_cast<size_t>(T) * S_ * D_;
        const size_t preds_size = static_cast<size_t>(M_) * T * K_;

        if (input.size() != chunk_size + preds_size)
            throw std::runtime_error(
                "MetaModel::infer: input size " + std::to_string(input.size()) +
                " != expected " + std::to_string(chunk_size + preds_size));

        // Run inner model on x_chunk slice (T, S, D).
        std::vector<float>   inner_in(input.data(), input.data() + chunk_size);
        std::vector<int64_t> inner_in_shape = {T, S_, D_};
        std::vector<int64_t> inner_out_shape;
        std::vector<float>   W = inner_->infer(inner_in, inner_in_shape,
                                                inner_out_shape);

        // Resolve fusion mode from actual output shape.
        bool coeff_mode = coeff_level_;
        if (inner_out_shape.size() == 3) {
            coeff_mode = true;
            if (inner_out_shape[1] != M_ || inner_out_shape[2] != K_)
                throw std::runtime_error(
                    "MetaModel: inner output (3-D) shape mismatch: "
                    "M=" + std::to_string(M_) + " K=" + std::to_string(K_));
        } else if (inner_out_shape.size() == 2) {
            coeff_mode = false;
            if (inner_out_shape[1] != M_)
                throw std::runtime_error(
                    "MetaModel: inner output (2-D) shape mismatch: "
                    "M=" + std::to_string(M_));
        } else if (inner_out_shape.size() == 1) {
            const int64_t n = static_cast<int64_t>(W.size());
            if      (n == (int64_t)T * M_ * K_) coeff_mode = true;
            else if (n == (int64_t)T * M_)       coeff_mode = false;
            else throw std::runtime_error(
                "MetaModel: cannot infer fusion mode from flat output size "
                + std::to_string(n));
        }

        return {std::move(W), input.data() + chunk_size, T, coeff_mode};
    }

    // Compute weighted mean and weighted standard deviation.
    // Weights are assumed to be non-negative and sum to 1 per (t, k) or per t.
    FusionResult fuse(const std::vector<float>& W,
                      const float* all_preds,
                      int T,
                      bool coeff_mode) const {
        FusionResult r;
        r.mean.assign(T * K_, 0.0f);
        r.std.assign (T * K_, 0.0f);

        // --- weighted mean ---
        if (coeff_mode) {
            for (int t = 0; t < T; ++t)
                for (int m = 0; m < M_; ++m)
                    for (int k = 0; k < K_; ++k)
                        r.mean[t * K_ + k] +=
                            W[(t * M_ + m) * K_ + k] *
                            all_preds[(m * T + t) * K_ + k];
        } else {
            for (int t = 0; t < T; ++t)
                for (int m = 0; m < M_; ++m) {
                    const float w = W[t * M_ + m];
                    for (int k = 0; k < K_; ++k)
                        r.mean[t * K_ + k] +=
                            w * all_preds[(m * T + t) * K_ + k];
                }
        }

        // --- weighted variance, then std ---
        if (coeff_mode) {
            for (int t = 0; t < T; ++t)
                for (int m = 0; m < M_; ++m)
                    for (int k = 0; k < K_; ++k) {
                        const float w    = W[(t * M_ + m) * K_ + k];
                        const float diff = all_preds[(m * T + t) * K_ + k]
                                           - r.mean[t * K_ + k];
                        r.std[t * K_ + k] += w * diff * diff;
                    }
        } else {
            for (int t = 0; t < T; ++t)
                for (int m = 0; m < M_; ++m) {
                    const float w = W[t * M_ + m];
                    for (int k = 0; k < K_; ++k) {
                        const float diff = all_preds[(m * T + t) * K_ + k]
                                           - r.mean[t * K_ + k];
                        r.std[t * K_ + k] += w * diff * diff;
                    }
                }
        }
        for (float& v : r.std) v = std::sqrt(v);

        return r;
    }
};

} // namespace rope
