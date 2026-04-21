#pragma once
// latent_decoder.h — decode latent vectors to physical density via the COAE.
//
// COAE decoder maps:
//   input  "latent"  : (batch, K=10)
//   output "density" : (batch, 1, 72, 36, 45)
//
// After decoding:
//   density_phys = 10 ^ (decoded_norm * sigma_cae + mu_cae)
//
// decode() accepts (H, K) latents and returns (H * GRID_VOXELS) floats.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "rope/core/types.h"
#include "rope/io/stats.h"
#include "model_interface.h"

namespace rope::forecast {

class LatentDecoder {
public:
    // decoder_model : loaded COAE ONNX/LibTorch model
    // cae_denorm    : CAEDenormalizer from stats_cae
    // batch_size    : time steps per model call (120 = effectively unbatched)
    LatentDecoder(IModel&                  decoder_model,
                  const io::CAEDenormalizer& cae_denorm,
                  int                      batch_size = 120)
        : model_(decoder_model)
        , denorm_(cae_denorm)
        , batch_size_(batch_size)
    {}

    // Decode (H, K) latents → (H * GRID_VOXELS) physical density.
    // latents must already be in physical space (denormalized from z-score).
    std::vector<float> decode(const std::vector<float>& latents, int H, int K) {
        assert(static_cast<int>(latents.size()) == H * K);

        constexpr int CH = 1;
        const int voxels_with_ch = CH * GRID_VOXELS;
        std::vector<float> density(static_cast<size_t>(H) * GRID_VOXELS);

        int offset_in  = 0;
        int offset_out = 0;

        while (offset_in < H) {
            int batch = std::min(batch_size_, H - offset_in);

            std::vector<float>   inp(latents.begin() + offset_in * K,
                                     latents.begin() + (offset_in + batch) * K);
            std::vector<int64_t> in_shape  = {batch, K};
            std::vector<int64_t> out_shape;
            std::vector<float>   raw = model_.infer(inp, in_shape, out_shape);

            if (static_cast<int>(raw.size()) != batch * voxels_with_ch)
                throw std::runtime_error(
                    "LatentDecoder: unexpected decoder output size: got " +
                    std::to_string(raw.size()) + " expected " +
                    std::to_string(batch * voxels_with_ch));

            denorm_.apply_inplace(raw.data(), batch, voxels_with_ch);

            std::copy(raw.begin(), raw.end(),
                      density.begin() +
                      static_cast<size_t>(offset_out) * GRID_VOXELS);

            offset_in  += batch;
            offset_out += batch;
        }
        return density;
    }

private:
    IModel&                    model_;
    const io::CAEDenormalizer& denorm_;
    int                        batch_size_;
};

} // namespace rope::forecast
