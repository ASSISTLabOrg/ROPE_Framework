#pragma once
// dynamic_rollout.hpp — auto-regressive rollout for one base model.
//
// Each base model predicts normalized latents for the next time step.
// The prediction is fed back into the sliding-window input for the
// following step (replacing the zero-latent placeholder in x_chunk).
//
// Mirrors rope.py's DynamicRollout.run().
//
// Thread safety: run() is stateless; multiple threads may call it with
// different models and non-overlapping output buffers concurrently.

#include <algorithm>
#include <cassert>
#include <vector>

#include "model_interface.hpp"
#include "onnx_model.hpp"

namespace rope {

class DynamicRollout {
public:
    // K = latent_dim, S = seq_len, D = total_dim
    DynamicRollout(int K, int S, int D) : K_(K), S_(S), D_(D) {}

    // Run the rollout for a single model.
    //
    //   model        – base temporal model (IModel)
    //   x_chunk      – (H, S, D) flat float array (NOT modified)
    //   H            – forecast horizon
    //   preds_out    – (H-1, K) flat float array (caller allocates)
    //
    // Input shape for each model call: (1, S, D)
    // Output shape expected:           (1, K)
    void run(
        IModel&      model,
        const float* x_chunk,   // (H, S, D)
        int          H,
        float*       preds_out  // (H-1, K)
    ) const {
        const int window_size = S_ * D_;
        std::vector<float> inp(window_size);
        std::copy(x_chunk, x_chunk + window_size, inp.begin());

        // Single output buffer reused each step.
        std::vector<float> out_buf(K_);

        const std::vector<int64_t> in_shape  = {1, S_, D_};
        const std::vector<int64_t> out_shape = {1, K_};

        // Use IoBinding path when the model is an OnnxModel (avoids per-call
        // allocator overhead over the 119-step rollout loop).
        OnnxModel* onnx = dynamic_cast<OnnxModel*>(&model);

        for (int t = 1; t < H; ++t) {
            if (onnx) {
                onnx->infer_bound(inp.data(), in_shape,
                                  out_buf.data(), out_shape);
            } else {
                std::vector<int64_t> dummy_shape;
                std::vector<float> p = model.infer(inp, in_shape, dummy_shape);
                assert(static_cast<int>(p.size()) >= K_);
                std::copy(p.begin(), p.begin() + K_, out_buf.begin());
            }

            std::copy(out_buf.begin(), out_buf.end(),
                      preds_out + (t - 1) * K_);

            if (t + 1 < H) {
                // Slide the window: inp[0:S-1] = inp[1:S]
                std::copy(inp.begin() + D_, inp.begin() + S_ * D_, inp.begin());

                // Fill last row: latents ← prediction, drivers ← x_chunk future
                float* last_row = inp.data() + (S_ - 1) * D_;
                std::copy(out_buf.begin(), out_buf.end(), last_row);

                const float* next_drv = x_chunk
                    + static_cast<size_t>(t + 1) * S_ * D_
                    + static_cast<size_t>(S_ - 1) * D_
                    + K_;
                std::copy(next_drv, next_drv + (D_ - K_), last_row + K_);
            }
        }
    }

private:
    int K_, S_, D_;
};

} // namespace rope
