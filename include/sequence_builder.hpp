#pragma once
// sequence_builder.hpp — builds the normalized feature sequences for ROPE.
//
// Mirrors rope.py's SequenceBuilder.build_X_init_norm() and build_x_chunk().
//
// Terminology:
//   K = latent_dim (10)
//   D = total_dim  (K + driver_dim, e.g. 16 or 17)
//   S = seq_len    (3)
//   H = horizon
//
// x_chunk layout:  (H, S, D)  row-major
//   x_chunk[t][:] = the S-step input window whose last row corresponds
//                   to forecast hour t.

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <vector>

#include "driver_window.hpp"
#include "ic_table.hpp"
#include "normalizer.hpp"

namespace rope {

// ---------------------------------------------------------------------------
// Selector for which driver columns to extract from DriverRow.
// Must match the training feature set used to build stats_ts.
// ---------------------------------------------------------------------------
enum class DriverCols {
    Six,   // [f10, kp, t1, t2, t3, t4]
    Seven, // [f10, kp, t1, t2, t3, t4, doy]
};

inline int driver_dim(DriverCols dc) {
    return (dc == DriverCols::Six) ? 6 : 7;
}

inline void fill_driver(const DriverRow& row, DriverCols dc, float* out) {
    out[0] = row.f10;
    out[1] = row.kp;
    out[2] = row.t1;
    out[3] = row.t2;
    out[4] = row.t3;
    out[5] = row.t4;
    if (dc == DriverCols::Seven)
        out[6] = row.doy;
}

// ---------------------------------------------------------------------------
// SequenceBuilder
// ---------------------------------------------------------------------------
class SequenceBuilder {
public:
    SequenceBuilder(
        const FeatureNormalizer& norm,
        const ICTable&           ic,
        DriverCols               driver_cols,
        int                      seq_len
    )
        : norm_(norm), ic_(ic), dc_(driver_cols), S_(seq_len)
    {
        K_  = norm_.latent_dim();
        D_  = norm_.total_dim();
        Dd_ = norm_.driver_dim();
        if (Dd_ != driver_dim(dc_))
            throw std::runtime_error(
                "SequenceBuilder: stats driver_dim does not match DriverCols selection");
    }

    // -----------------------------------------------------------------------
    // Build X_init_norm: (S, D) normalized, using the first S rows of
    // the internal driver window (history rows).
    // Returns flat (S*D) float array.
    // -----------------------------------------------------------------------
    std::vector<float> build_X_init_norm(
        const std::vector<DriverRow>& internal_rows
    ) const {
        assert(static_cast<int>(internal_rows.size()) >= S_);

        std::vector<float> X(S_ * D_, 0.0f);

        for (int s = 0; s < S_; ++s) {
            const DriverRow& row = internal_rows[s];
            float* dest = X.data() + s * D_;

            // Latent coefficients from IC interpolation.
            std::vector<float> coeffs = ic_.get_latent_coeffs(row.f10, row.kp);
            for (int k = 0; k < K_; ++k)
                dest[k] = coeffs[k];

            // Raw driver features.
            fill_driver(row, dc_, dest + K_);

            // Normalize the full row.
            norm_.norm_full_inplace(dest);
        }
        return X;  // (S, D) flat
    }

    // -----------------------------------------------------------------------
    // Build x_chunk: (H, S, D) normalized sliding windows for the
    // forecast horizon.
    //
    // forecast_rows: exactly H rows (starting at the forecast start time).
    // X_init_norm:   (S, D) flat, output of build_X_init_norm().
    //
    // Returns flat (H * S * D) float array.
    // -----------------------------------------------------------------------
    std::vector<float> build_x_chunk(
        const std::vector<float>&    X_init_norm,
        const std::vector<DriverRow>& forecast_rows,
        int                          H
    ) const {
        assert(static_cast<int>(forecast_rows.size()) == H);
        assert(static_cast<int>(X_init_norm.size())   == S_ * D_);

        std::vector<float> chunk(H * S_ * D_, 0.0f);

        // x_chunk[0] = X_init_norm
        std::copy(X_init_norm.begin(), X_init_norm.end(), chunk.begin());

        // Scratch buffer for a raw driver vector.
        std::vector<float> drv(Dd_);

        for (int t = 1; t < H; ++t) {
            const float* prev = chunk.data() + (t - 1) * S_ * D_;
            float*       curr = chunk.data() +  t      * S_ * D_;

            // Shift: curr[0:S-1] = prev[1:S]
            std::copy(prev + D_, prev + S_ * D_, curr);

            // Fill last row: latents zeroed, drivers normalized for time t.
            float* last_row = curr + (S_ - 1) * D_;
            std::fill(last_row, last_row + K_, 0.0f);
            fill_driver(forecast_rows[t], dc_, drv.data());
            norm_.norm_driver_inplace(drv.data());
            std::copy(drv.begin(), drv.end(), last_row + K_);
        }
        return chunk;   // (H, S, D) flat
    }

    int latent_dim() const { return K_; }
    int total_dim()  const { return D_; }
    int seq_len()    const { return S_; }

private:
    const FeatureNormalizer& norm_;
    const ICTable&           ic_;
    DriverCols               dc_;
    int S_, K_, D_, Dd_;
};

} // namespace rope
