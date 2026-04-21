#include "rope/io/ic_table.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rope::io {

ICTable::ICTable(const std::filesystem::path& csv_path) {
    CsvReader csv(csv_path);
    const std::size_t N = csv.nrows();

    pts_f10_.resize(N);
    pts_kp_.resize(N);
    vals_.resize(N * K);

    for (std::size_t i = 0; i < N; ++i) {
        pts_f10_[i] = csv.get_float("F10", i);
        pts_kp_[i]  = csv.get_float("Kp",  i);
        for (int k = 0; k < K; ++k)
            vals_[i * K + k] = csv.get_float("y" + std::to_string(k + 1), i);
    }

    f10_axis_ = unique_sorted(pts_f10_);
    kp_axis_  = unique_sorted(pts_kp_);

    const float EPS     = 1e-4f;
    const std::size_t nf = f10_axis_.size();
    const std::size_t nk = kp_axis_.size();
    n_kp_ = nk;
    grid_idx_.assign(nf * nk, -1);

    for (std::size_t i = 0; i < N; ++i) {
        int fi = static_cast<int>(
            std::lower_bound(f10_axis_.begin(), f10_axis_.end(),
                             pts_f10_[i] - EPS) - f10_axis_.begin());
        int ki = static_cast<int>(
            std::lower_bound(kp_axis_.begin(), kp_axis_.end(),
                             pts_kp_[i] - EPS) - kp_axis_.begin());
        if (fi < static_cast<int>(nf) && ki < static_cast<int>(nk))
            grid_idx_[fi * nk + ki] = static_cast<int>(i);
    }
}

std::vector<float> ICTable::get_latent_coeffs(float f10, float kp) const {
    std::vector<float> out(K, 0.0f);
    if (bilinear(f10, kp, out.data())) return out;
    nearest(f10, kp, out.data());
    return out;
}

std::vector<float> ICTable::unique_sorted(const std::vector<float>& v) {
    std::vector<float> u = v;
    std::sort(u.begin(), u.end());
    u.erase(std::unique(u.begin(), u.end(),
                        [](float a, float b){ return std::abs(a - b) < 1e-4f; }),
            u.end());
    return u;
}

const float* ICTable::row_ptr(int fi, int ki) const noexcept {
    if (fi < 0 || fi >= static_cast<int>(f10_axis_.size()) ||
        ki < 0 || ki >= static_cast<int>(n_kp_))
        return nullptr;
    int idx = grid_idx_[fi * n_kp_ + ki];
    if (idx < 0) return nullptr;
    return vals_.data() + idx * K;
}

bool ICTable::bilinear(float f10, float kp, float* out) const {
    auto f10_it = std::lower_bound(f10_axis_.begin(), f10_axis_.end(), f10);
    auto kp_it  = std::lower_bound(kp_axis_.begin(),  kp_axis_.end(),  kp);
    int fi1 = static_cast<int>(f10_it - f10_axis_.begin());
    int ki1 = static_cast<int>(kp_it  - kp_axis_.begin());
    int fi0 = fi1 - 1, ki0 = ki1 - 1;

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
    float w00 = (1-tf)*(1-tk), w01 = (1-tf)*tk;
    float w10 = tf*(1-tk),     w11 = tf*tk;

    for (int k = 0; k < K; ++k)
        out[k] = w00*v00[k] + w01*v01[k] + w10*v10[k] + w11*v11[k];
    return true;
}

void ICTable::nearest(float f10, float kp, float* out) const {
    float best = std::numeric_limits<float>::max();
    int   best_i = 0;
    for (std::size_t i = 0; i < pts_f10_.size(); ++i) {
        float df = pts_f10_[i] - f10;
        float dk = pts_kp_[i]  - kp;
        float d  = df*df + dk*dk;
        if (d < best) { best = d; best_i = static_cast<int>(i); }
    }
    const float* src = vals_.data() + best_i * K;
    for (int k = 0; k < K; ++k) out[k] = src[k];
}

} // namespace rope::io
