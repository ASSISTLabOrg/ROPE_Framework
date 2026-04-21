#pragma once
// IC table: (F10, Kp) → 10-dimensional latent initial condition coefficients.
//
// Columns: F10, Kp, y1 … y10.
//
// Interpolation: bilinear on the (F10, Kp) grid; nearest-neighbour fallback
// when the query is outside the convex hull.

#include "rope/io/csv_reader.h"

#include <filesystem>
#include <limits>
#include <vector>

namespace rope::io {

class ICTable {
public:
    static constexpr int K = 10;  // latent dimension

    explicit ICTable(const std::filesystem::path& csv_path);

    // Returns K latent coefficients for (f10, kp).
    std::vector<float> get_latent_coeffs(float f10, float kp) const;

private:
    std::vector<float> pts_f10_, pts_kp_;
    std::vector<float> vals_;       // (N, K) row-major
    std::vector<float> f10_axis_, kp_axis_;
    std::vector<int>   grid_idx_;   // -1 = missing cell
    std::size_t        n_kp_ = 0;

    static std::vector<float> unique_sorted(const std::vector<float>& v);
    const float* row_ptr(int fi, int ki) const noexcept;
    bool bilinear(float f10, float kp, float* out) const;
    void nearest(float f10, float kp, float* out) const;
};

} // namespace rope::io
