#pragma once
// rope_types.hpp — shared plain-data types used throughout the C++ ROPE library.

#include <string>
#include <vector>

namespace rope {

// ---------------------------------------------------------------------------
// Grid constants that match the COAE output  (72 LST × 36 lat × 45 alt).
// ---------------------------------------------------------------------------
constexpr int GRID_LST = 72;
constexpr int GRID_LAT = 36;
constexpr int GRID_ALT = 45;
constexpr int GRID_VOXELS = GRID_LST * GRID_LAT * GRID_ALT;   // 116 640

// ---------------------------------------------------------------------------
// Result returned by ROPE::run().
// meta_density is stored in row-major order: [t, lst, lat, alt].
// ---------------------------------------------------------------------------
struct ForecastResult {
    std::vector<std::string> datetimes;   // H datetime strings "YYYY-MM-DD HH:MM:SS"
    std::vector<float>       f10;         // H solar-flux values
    std::vector<float>       kp;          // H Kp indices
    std::vector<float>       meta_density;// H * GRID_VOXELS floats
    int                      H = 0;       // forecast horizon (hours)

    // Access density at time-step t
    const float* density_at(int t) const {
        return meta_density.data() + static_cast<size_t>(t) * GRID_VOXELS;
    }
};

} // namespace rope
