#include "rope/io/ic_bin.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace rope::io {

static constexpr std::uint32_t IC_MAGIC   = 0x52504943u;  // "RPIC"
static constexpr std::uint32_t IC_VERSION = 1u;

ICTable IcBin::load(const std::filesystem::path& bin_path) {
    std::ifstream f(bin_path, std::ios::binary);
    if (!f)
        throw std::runtime_error(
            "IcBin::load: cannot open " + bin_path.string());

    std::uint32_t magic, version, nrows, latent_dim, reserved;
    f.read(reinterpret_cast<char*>(&magic),      4);
    f.read(reinterpret_cast<char*>(&version),    4);
    f.read(reinterpret_cast<char*>(&nrows),      4);
    f.read(reinterpret_cast<char*>(&latent_dim), 4);
    f.read(reinterpret_cast<char*>(&reserved),   4);

    if (!f)
        throw std::runtime_error(
            "IcBin::load: failed to read header from " + bin_path.string());
    if (magic != IC_MAGIC)
        throw std::runtime_error(
            "IcBin::load: bad magic in " + bin_path.string() +
            " (not a .icbin file)");
    if (version != IC_VERSION)
        throw std::runtime_error(
            "IcBin::load: unsupported version " +
            std::to_string(version) + " in " + bin_path.string());
    if (latent_dim == 0)
        throw std::runtime_error(
            "IcBin::load: latent_dim=0 in " + bin_path.string());

    const int K = static_cast<int>(latent_dim);

    std::vector<float> pts_f10(nrows), pts_kp(nrows);
    std::vector<float> vals(static_cast<std::size_t>(nrows) * K);

    for (std::uint32_t i = 0; i < nrows; ++i) {
        f.read(reinterpret_cast<char*>(&pts_f10[i]), 4);
        f.read(reinterpret_cast<char*>(&pts_kp[i]),  4);
        f.read(reinterpret_cast<char*>(&vals[i * K]), static_cast<std::streamsize>(K * 4));

        if (!f)
            throw std::runtime_error(
                "IcBin::load: unexpected EOF at record " +
                std::to_string(i) + " in " + bin_path.string());
    }

    // Rebuild axes and grid index (same logic as CSV constructor).
    auto unique_sorted = [](std::vector<float> v) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end(),
                            [](float a, float b){ return std::abs(a - b) < 1e-4f; }),
                v.end());
        return v;
    };

    std::vector<float> f10_axis = unique_sorted(pts_f10);
    std::vector<float> kp_axis  = unique_sorted(pts_kp);
    const std::size_t  nf = f10_axis.size();
    const std::size_t  nk = kp_axis.size();

    std::vector<int> grid_idx(nf * nk, -1);
    const float EPS = 1e-4f;
    for (std::uint32_t i = 0; i < nrows; ++i) {
        int fi = static_cast<int>(
            std::lower_bound(f10_axis.begin(), f10_axis.end(),
                             pts_f10[i] - EPS) - f10_axis.begin());
        int ki = static_cast<int>(
            std::lower_bound(kp_axis.begin(), kp_axis.end(),
                             pts_kp[i] - EPS) - kp_axis.begin());
        if (fi < static_cast<int>(nf) && ki < static_cast<int>(nk))
            grid_idx[fi * nk + ki] = static_cast<int>(i);
    }

    return ICTable{K,
                   std::move(pts_f10), std::move(pts_kp),
                   std::move(vals),
                   std::move(f10_axis), std::move(kp_axis),
                   std::move(grid_idx), nk};
}

void IcBin::save(const ICTable& table,
                 const std::filesystem::path& bin_path) {
    std::ofstream f(bin_path, std::ios::binary | std::ios::trunc);
    if (!f)
        throw std::runtime_error(
            "IcBin::save: cannot open " + bin_path.string());

    const auto nrows      = static_cast<std::uint32_t>(table.pts_f10_.size());
    const auto latent_dim = static_cast<std::uint32_t>(table.k_);
    const std::uint32_t reserved = 0u;

    f.write(reinterpret_cast<const char*>(&IC_MAGIC),   4);
    f.write(reinterpret_cast<const char*>(&IC_VERSION), 4);
    f.write(reinterpret_cast<const char*>(&nrows),      4);
    f.write(reinterpret_cast<const char*>(&latent_dim), 4);
    f.write(reinterpret_cast<const char*>(&reserved),   4);

    for (std::uint32_t i = 0; i < nrows; ++i) {
        f.write(reinterpret_cast<const char*>(&table.pts_f10_[i]), 4);
        f.write(reinterpret_cast<const char*>(&table.pts_kp_[i]),  4);
        f.write(reinterpret_cast<const char*>(&table.vals_[i * table.k_]),
                static_cast<std::streamsize>(table.k_ * 4));
    }

    if (!f)
        throw std::runtime_error(
            "IcBin::save: write failed for " + bin_path.string());
}

} // namespace rope::io
