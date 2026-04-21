#pragma once
// normalizer.hpp — load exported stats and apply z-score normalization.
//
// Binary format written by export_models.py:
//   uint32  ndim
//   uint32  shape[ndim]
//   float32 mu[product(shape)]
//   float32 sigma[product(shape)]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace rope {

// ---------------------------------------------------------------------------
// Raw statistics blob (mu + sigma, same flat shape).
// ---------------------------------------------------------------------------
struct Stats {
    std::vector<uint32_t> shape;
    std::vector<float>    mu;
    std::vector<float>    sigma;

    size_t numel() const { return mu.size(); }

    static Stats load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Stats::load: cannot open " + path);

        Stats s;
        uint32_t ndim = 0;
        f.read(reinterpret_cast<char*>(&ndim), 4);
        s.shape.resize(ndim);
        f.read(reinterpret_cast<char*>(s.shape.data()),
               static_cast<std::streamsize>(ndim * 4));

        size_t n = 1;
        for (auto d : s.shape) n *= d;
        s.mu.resize(n); s.sigma.resize(n);
        f.read(reinterpret_cast<char*>(s.mu.data()),
               static_cast<std::streamsize>(n * 4));
        f.read(reinterpret_cast<char*>(s.sigma.data()),
               static_cast<std::streamsize>(n * 4));
        if (!f)
            throw std::runtime_error("Stats::load: read failed for " + path);
        return s;
    }
};

// ---------------------------------------------------------------------------
// Feature normalizer for the time-series feature vector.
//
// The feature vector has layout:
//   [latent_0, …, latent_{K-1}, driver_0, …, driver_{D-K-1}]
//
// Matches rope.py's FeatureNormalizer.
// ---------------------------------------------------------------------------
class FeatureNormalizer {
public:
    // latent_dim: K  (= ROPEConfig::latent_dim = 10)
    FeatureNormalizer(const Stats& stats, int latent_dim)
        : K_(latent_dim)
        , D_(static_cast<int>(stats.numel()))
        , mu_(stats.mu)
        , sigma_(stats.sigma)
    {
        if (D_ <= K_)
            throw std::runtime_error(
                "FeatureNormalizer: stats dim must exceed latent_dim");
    }

    int latent_dim() const { return K_; }
    int total_dim()  const { return D_; }
    int driver_dim() const { return D_ - K_; }

    // Normalize a full feature vector of length D (in-place).
    void norm_full_inplace(float* x) const {
        for (int i = 0; i < D_; ++i)
            x[i] = (x[i] - mu_[i]) / sigma_[i];
    }

    // Normalize only the driver portion (length D-K).
    void norm_driver_inplace(float* drv) const {
        for (int i = 0; i < D_ - K_; ++i)
            drv[i] = (drv[i] - mu_[K_ + i]) / sigma_[K_ + i];
    }

    // De-normalize a latent vector of length K.
    void denorm_latents_inplace(float* lat) const {
        for (int i = 0; i < K_; ++i)
            lat[i] = lat[i] * sigma_[i] + mu_[i];
    }

    // Convenience: denormalize a contiguous (N, K) block.
    void denorm_latents_block(float* block, int N) const {
        for (int n = 0; n < N; ++n)
            denorm_latents_inplace(block + n * K_);
    }

private:
    int               K_, D_;
    std::vector<float> mu_, sigma_;
};

// ---------------------------------------------------------------------------
// CAE denormalizer — converts model output (log10 space) to physical density.
//
// Python equivalent:
//   density = 10 ** (decoder_output * stats_cae["sigma"] + stats_cae["mu"])
//
// Stats may be scalar (shape [1]) or spatial (shape [1,72,36,45]).
// In the spatial case the stats are broadcast per-voxel.
// ---------------------------------------------------------------------------
class CAEDenormalizer {
public:
    explicit CAEDenormalizer(const Stats& stats_cae) {
        mu_    = stats_cae.mu;
        sigma_ = stats_cae.sigma;
        n_     = mu_.size();   // 1 (scalar) or 72*36*45 (spatial)
    }

    // Apply denormalization in-place to a flat (batch, 1, 72, 36, 45) block.
    // voxels_per_t = 1*72*36*45 = 116640
    void apply_inplace(float* block, int T, int voxels_per_t) const {
        for (int t = 0; t < T; ++t) {
            float* ptr = block + static_cast<size_t>(t) * voxels_per_t;
            for (int v = 0; v < voxels_per_t; ++v) {
                float mu_v    = (n_ == 1) ? mu_[0]    : mu_[v];
                float sigma_v = (n_ == 1) ? sigma_[0] : sigma_[v];
                ptr[v] = std::pow(10.0f, ptr[v] * sigma_v + mu_v);
            }
        }
    }

private:
    std::vector<float> mu_, sigma_;
    size_t n_;
};

} // namespace rope
