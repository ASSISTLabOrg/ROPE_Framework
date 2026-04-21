#pragma once
// sequence_builder.h — builds normalized feature sequences for the pipeline.
//
// Terminology:
//   K = latent_dim (10)
//   D = total_dim  (K + driver_dim, e.g. 16 or 17)
//   S = seq_len    (3)
//   H = horizon
//
// x_chunk layout: (H, S, D) row-major
//   x_chunk[t][:] = the S-step input window whose last row is forecast hour t.

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <vector>

#include "rope/io/driver_db.h"
#include "rope/io/ic_table.h"
#include "rope/io/stats.h"

namespace rope::forecast {

// ---------------------------------------------------------------------------
// Selector for which driver columns to extract from DriverRow.
// Must match the training feature set in stats_ts.
// ---------------------------------------------------------------------------
enum class DriverCols { Six, Seven };

inline int driver_dim(DriverCols dc) { return (dc == DriverCols::Six) ? 6 : 7; }

inline void fill_driver(const io::DriverRow& row, DriverCols dc, float* out) {
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
    SequenceBuilder(const io::FeatureNormalizer& norm,
                    const io::ICTable&           ic,
                    DriverCols                   driver_cols,
                    int                          seq_len)
        : norm_(norm), ic_(ic), dc_(driver_cols), S_(seq_len)
    {
        K_  = norm_.latent_dim();
        D_  = norm_.total_dim();
        Dd_ = norm_.driver_dim();
        if (Dd_ != driver_dim(dc_))
            throw std::runtime_error(
                "SequenceBuilder: stats driver_dim does not match DriverCols");
    }

    // Build X_init_norm: (S, D) normalized from the first S history rows.
    // Returns flat (S*D) float array.
    std::vector<float> build_X_init_norm(
        const std::vector<io::DriverRow>& hist_rows
    ) const {
        assert(static_cast<int>(hist_rows.size()) >= S_);

        std::vector<float> X(static_cast<size_t>(S_) * D_, 0.0f);

        for (int s = 0; s < S_; ++s) {
            const io::DriverRow& row = hist_rows[s];
            float* dest = X.data() + s * D_;

            std::vector<float> coeffs = ic_.get_latent_coeffs(row.f10, row.kp);
            for (int k = 0; k < K_; ++k)
                dest[k] = coeffs[k];

            fill_driver(row, dc_, dest + K_);
            norm_.norm_full_inplace(dest);
        }
        return X;
    }

    // Build x_chunk: (H, S, D) normalized sliding windows for horizon H.
    // X_init_norm : (S*D) flat from build_X_init_norm().
    // forecast_rows: exactly H rows starting at the forecast start time.
    std::vector<float> build_x_chunk(
        const std::vector<float>&        X_init_norm,
        const std::vector<io::DriverRow>& forecast_rows,
        int                              H
    ) const {
        assert(static_cast<int>(forecast_rows.size()) == H);
        assert(static_cast<int>(X_init_norm.size())   == S_ * D_);

        std::vector<float> chunk(static_cast<size_t>(H) * S_ * D_, 0.0f);
        std::copy(X_init_norm.begin(), X_init_norm.end(), chunk.begin());

        std::vector<float> drv(Dd_);

        for (int t = 1; t < H; ++t) {
            const float* prev = chunk.data() + (t - 1) * S_ * D_;
            float*       curr = chunk.data() +  t      * S_ * D_;

            std::copy(prev + D_, prev + S_ * D_, curr);

            float* last_row = curr + (S_ - 1) * D_;
            std::fill(last_row, last_row + K_, 0.0f);
            fill_driver(forecast_rows[t], dc_, drv.data());
            norm_.norm_driver_inplace(drv.data());
            std::copy(drv.begin(), drv.end(), last_row + K_);
        }
        return chunk;
    }

    int latent_dim() const { return K_; }
    int total_dim()  const { return D_; }
    int seq_len()    const { return S_; }

private:
    const io::FeatureNormalizer& norm_;
    const io::ICTable&           ic_;
    DriverCols                   dc_;
    int                          S_, K_, D_, Dd_;
};

} // namespace rope::forecast
