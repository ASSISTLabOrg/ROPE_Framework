#pragma once
// Shared plain-data types used across all modules.

#include <cstdint>
#include <vector>

namespace rope {

// ---------------------------------------------------------------------------
// Grid constants — match COAE decoder output (72 LST × 36 lat × 45 alt).
// ---------------------------------------------------------------------------
constexpr int GRID_LST    = 72;
constexpr int GRID_LAT    = 36;
constexpr int GRID_ALT    = 45;
constexpr int GRID_VOXELS = GRID_LST * GRID_LAT * GRID_ALT;  // 116,640

// ---------------------------------------------------------------------------
// ForecastGrid — in-memory forecast produced by forecast/ and consumed by
// interpolate/ and client/.
//
// Both density and uncertainty are stored row-major [t, lst, lat, alt];
// alt is the fastest axis.  Units: kg/m³.
// Times are UTC seconds since the Unix epoch, one per forecast hour.
// ---------------------------------------------------------------------------
struct ForecastGrid {
    std::vector<float>        density;      // H * GRID_VOXELS floats
    std::vector<float>        uncertainty;  // H * GRID_VOXELS floats
    std::vector<std::int64_t> times;        // H timestamps (seconds since epoch)
    int H = 0;

    const float* density_at(int t) const noexcept {
        return density.data() + static_cast<std::size_t>(t) * GRID_VOXELS;
    }
    const float* uncertainty_at(int t) const noexcept {
        return uncertainty.data() + static_cast<std::size_t>(t) * GRID_VOXELS;
    }
};

} // namespace rope
