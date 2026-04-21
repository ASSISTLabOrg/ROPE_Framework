#pragma once
// meta_model.h — ensemble fusion meta-model, implementing IModel.
//
// Input convention for infer():
//   input       : [x_chunk_slice (T*S*D) | all_preds (M*T*K)]  flat
//   input_shape : {T, S*D, M, K}
//
// Output: [mean (T*K) | std (T*K)] flat; output_shape = {T, K, 2}.
//
// Two weight shapes are supported (inferred from the model's actual output):
//   coeff_level  (W shape T×M×K): per-latent-dimension weights
//   time_level   (W shape T×M):   shared weights across dimensions

#include <cassert>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "model_interface.h"

namespace rope::forecast {

struct FusionResult {
    std::vector<float> mean;  // (T*K)
    std::vector<float> std;   // (T*K)
};

class MetaModel : public IModel {
public:
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
        const std::vector<float>&   input,
        const std::vector<int64_t>& input_shape,
        std::vector<int64_t>&       output_shape
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

    struct InnerResult {
        std::vector<float> W;
        const float*       all_preds;
        int                T;
        bool               coeff_mode;
    };

    InnerResult run_inner(const std::vector<float>&   input,
                          const std::vector<int64_t>& input_shape) const {
        if (input_shape.size() != 4)
            throw std::runtime_error(
                "MetaModel::infer: input_shape must be {T, S*D, M, K}");

        const int    T          = static_cast<int>(input_shape[0]);
        const size_t chunk_size = static_cast<size_t>(T) * S_ * D_;
        const size_t preds_size = static_cast<size_t>(M_) * T * K_;

        if (input.size() != chunk_size + preds_size)
            throw std::runtime_error(
                "MetaModel::infer: input size " + std::to_string(input.size()) +
                " != expected " + std::to_string(chunk_size + preds_size));

        std::vector<float>   inner_in(input.data(), input.data() + chunk_size);
        std::vector<int64_t> inner_in_shape = {T, S_, D_};
        std::vector<int64_t> inner_out_shape;
        std::vector<float>   W = inner_->infer(inner_in, inner_in_shape,
                                                inner_out_shape);

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

    FusionResult fuse(const std::vector<float>& W,
                      const float* all_preds,
                      int T,
                      bool coeff_mode) const {
        FusionResult r;
        r.mean.assign(T * K_, 0.0f);
        r.std.assign (T * K_, 0.0f);

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

} // namespace rope::forecast
