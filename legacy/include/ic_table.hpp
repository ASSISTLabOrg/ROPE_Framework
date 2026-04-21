#pragma once
// ic_table.hpp — IC table interpolation: (F10, Kp) → 10-D latent coefficients.
//
// The IC_Table_modified.csv has columns: F10, Kp, y1, …, y10.
//
// Interpolation strategy (mirrors scipy.griddata behaviour):
//   1. Bilinear on the regular (F10, Kp) grid — exact when the query point
//      lies inside the convex hull of grid points.
//   2. Nearest-neighbour fallback when the point is outside the hull
//      (matches scipy's 'nearest' fallback in rope.py).
//
// The CSV is parsed once and the unique F10/Kp axes are sorted so that
// look-ups are O(log N).

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "io.hpp"

namespace rope {

class ICTable {
public:
    // Number of latent coefficients in the table.
    static constexpr int K = 10;

    explicit ICTable(const std::string& csv_path) {
        CSVReader csv(csv_path);

        size_t N = csv.nrows();
        pts_f10_.resize(N);
        pts_kp_.resize(N);
        vals_.resize(N * K);

        for (size_t i = 0; i < N; ++i) {
            pts_f10_[i] = csv.get_float("F10", i);
            pts_kp_[i]  = csv.get_float("Kp",  i);
            for (int k = 0; k < K; ++k) {
                std::string col = "y" + std::to_string(k + 1);
                vals_[i * K + k] = csv.get_float(col, i);
            }
        }

        // Build sorted unique axes for bilinear lookup.
        f10_axis_ = unique_sorted(pts_f10_);
        kp_axis_  = unique_sorted(pts_kp_);

        // Build a 2-D index: grid_idx_[fi * n_kp + ki] = row index in vals_.
        // Rows that are missing are set to -1.
        const float EPS = 1e-4f;
        size_t n_f10 = f10_axis_.size(), n_kp = kp_axis_.size();
        grid_idx_.assign(n_f10 * n_kp, -1);
        for (size_t i = 0; i < N; ++i) {
            int fi = static_cast<int>(
                std::lower_bound(f10_axis_.begin(), f10_axis_.end(),
                                 pts_f10_[i] - EPS) - f10_axis_.begin());
            int ki = static_cast<int>(
                std::lower_bound(kp_axis_.begin(),  kp_axis_.end(),
                                 pts_kp_[i]  - EPS) - kp_axis_.begin());
            if (fi < static_cast<int>(n_f10) && ki < static_cast<int>(n_kp))
                grid_idx_[fi * n_kp + ki] = static_cast<int>(i);
        }
        n_kp_ = n_kp;
    }

    // Interpolate: returns a vector of K latent coefficients.
    // Mirrors rope.py: linear (bilinear), fallback to nearest.
    std::vector<float> get_latent_coeffs(float f10, float kp) const {
        std::vector<float> out(K, 0.0f);

        // --- bilinear attempt ---
        if (bilinear(f10, kp, out.data()))
            return out;

        // --- nearest-neighbour fallback ---
        nearest(f10, kp, out.data());
        return out;
    }

private:
    std::vector<float> pts_f10_, pts_kp_;
    std::vector<float> vals_;      // (N, K) row-major
    std::vector<float> f10_axis_, kp_axis_;
    std::vector<int>   grid_idx_;  // -1 = missing
    size_t             n_kp_ = 0;

    // Build sorted unique values from a float vector (within EPS tolerance).
    static std::vector<float> unique_sorted(const std::vector<float>& v) {
        std::vector<float> u = v;
        std::sort(u.begin(), u.end());
        u.erase(std::unique(u.begin(), u.end(),
                            [](float a, float b){ return std::abs(a-b) < 1e-4f; }),
                u.end());
        return u;
    }

    // Look up a row in the pre-built index; returns nullptr if missing.
    const float* row_ptr(int fi, int ki) const {
        if (fi < 0 || fi >= static_cast<int>(f10_axis_.size()) ||
            ki < 0 || ki >= static_cast<int>(n_kp_))
            return nullptr;
        int idx = grid_idx_[fi * n_kp_ + ki];
        if (idx < 0) return nullptr;
        return vals_.data() + idx * K;
    }

    // Bilinear interpolation.  Returns true on success.
    bool bilinear(float f10, float kp, float* out) const {
        auto f10_it = std::lower_bound(f10_axis_.begin(), f10_axis_.end(), f10);
        auto kp_it  = std::lower_bound(kp_axis_.begin(),  kp_axis_.end(),  kp);

        // We need both lower and upper bracket indices.
        int fi1 = static_cast<int>(f10_it - f10_axis_.begin());
        int ki1 = static_cast<int>(kp_it  - kp_axis_.begin());
        int fi0 = fi1 - 1, ki0 = ki1 - 1;

        // If the query is outside the grid in either dimension, fail.
        if (fi0 < 0 || fi1 >= static_cast<int>(f10_axis_.size()) ||
            ki0 < 0 || ki1 >= static_cast<int>(n_kp_))
            return false;

        const float* v00 = row_ptr(fi0, ki0);
        const float* v01 = row_ptr(fi0, ki1);
        const float* v10 = row_ptr(fi1, ki0);
        const float* v11 = row_ptr(fi1, ki1);
        if (!v00 || !v01 || !v10 || !v11) return false;

        float tf = (f10 - f10_axis_[fi0]) / (f10_axis_[fi1] - f10_axis_[fi0]);
        float tk = (kp  - kp_axis_[ki0])  / (kp_axis_[ki1]  - kp_axis_[ki0]);

        // Standard bilinear weights.
        float w00 = (1-tf)*(1-tk), w01 = (1-tf)*tk;
        float w10 = tf*(1-tk),     w11 = tf*tk;

        for (int k = 0; k < K; ++k)
            out[k] = w00*v00[k] + w01*v01[k] + w10*v10[k] + w11*v11[k];
        return true;
    }

    // Nearest-neighbour fallback (squared Euclidean distance in F10-Kp space).
    void nearest(float f10, float kp, float* out) const {
        float best_dist = std::numeric_limits<float>::max();
        int   best_i    = 0;
        for (size_t i = 0; i < pts_f10_.size(); ++i) {
            float df = pts_f10_[i] - f10;
            float dk = pts_kp_[i]  - kp;
            float d  = df*df + dk*dk;
            if (d < best_dist) { best_dist = d; best_i = static_cast<int>(i); }
        }
        const float* src = vals_.data() + best_i * K;
        for (int k = 0; k < K; ++k) out[k] = src[k];
    }
};

} // namespace rope
