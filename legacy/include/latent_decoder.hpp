#pragma once
// latent_decoder.hpp — decode latent vectors to physical density via the COAE.
//
// The COAE decoder ONNX model maps:
//   input  "latent"  : (batch, K=10)
//   output "density" : (batch, 1, 72, 36, 45)
//
// After decoding we apply the CAE denormalization:
//   density_phys = 10 ^ (decoded_norm * sigma_cae + mu_cae)
//
// Mirrors rope.py's LatentDecoder.decode().

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "model_interface.hpp"
#include "normalizer.hpp"
#include "rope_types.hpp"

namespace rope {

class LatentDecoder {
public:
    // decoder_model : loaded COAE decoder ONNX model
    // cae_denorm    : CAEDenormalizer built from stats_cae
    // batch_size    : number of time steps decoded per ONNX call
    LatentDecoder(
        IModel&              decoder_model,
        const CAEDenormalizer& cae_denorm,
        int                  batch_size = 32
    )
        : model_(decoder_model)
        , denorm_(cae_denorm)
        , batch_size_(batch_size)
    {}

    // Decode (H, K) latent series → (H, 72, 36, 45) physical density.
    //
    //   latents : (H, K) flat float (physical space, already denormalized
    //             from z-score by FeatureNormalizer::denorm_latents_block)
    //
    // Returns (H * GRID_VOXELS) float in row-major order [t, lst, lat, alt].
    std::vector<float> decode(
        const std::vector<float>& latents,   // (H, K)
        int H, int K
    ) {
        assert(static_cast<int>(latents.size()) == H * K);

        // Output buffer: (H, 1, 72, 36, 45) — we drop the channel dim later.
        const int CH = 1;
        const int voxels_with_ch = CH * GRID_VOXELS;
        std::vector<float> density(H * GRID_VOXELS);

        int offset_in  = 0;
        int offset_out = 0;

        while (offset_in < H) {
            int batch = std::min(batch_size_, H - offset_in);

            // Build input slice (batch, K).
            std::vector<float> inp(latents.begin() + offset_in * K,
                                   latents.begin() + (offset_in + batch) * K);
            std::vector<int64_t> in_shape  = {batch, K};
            std::vector<int64_t> out_shape;

            std::vector<float> raw = model_.infer(inp, in_shape, out_shape);
            // raw shape: (batch, 1, 72, 36, 45)
            // expected numel: batch * GRID_VOXELS
            if (static_cast<int>(raw.size()) != batch * voxels_with_ch)
                throw std::runtime_error(
                    "LatentDecoder: unexpected decoder output size");

            // Denormalize in-place (log10 space → physical density).
            denorm_.apply_inplace(raw.data(), batch, voxels_with_ch);

            // Copy into output buffer, squeezing out the channel=1 dimension.
            // raw:     (batch, 1, 72, 36, 45) — already voxels_with_ch == GRID_VOXELS
            // density: (H, 72, 36, 45)
            std::copy(raw.begin(), raw.end(),
                      density.begin() + static_cast<size_t>(offset_out) * GRID_VOXELS);

            offset_in  += batch;
            offset_out += batch;
        }
        return density;
    }

private:
    IModel&              model_;
    const CAEDenormalizer& denorm_;
    int                  batch_size_;
};

} // namespace rope
