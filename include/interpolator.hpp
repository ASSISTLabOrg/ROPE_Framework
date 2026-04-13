#pragma once
// interpolator.hpp — single-point spatiotemporal density query on ROPE output.
//
// Mirrors rope.py's interpolator.py (DensityInterpolator).
//
// Grid axes (fixed, matching COAE output):
//   LST : linspace(0, 23.6̄, 72)     [hours]
//   lat : linspace(-87.5, 87.5, 36)  [degrees]
//   alt : linspace(100, 980, 45)     [km]
//
// Two time modes:
//   "hold_next_hour" – spatial interpolation only at ceil(requested_time)
//   "interp_time"    – trilinear spatial at both bracket hours, then linear
//                      time blend (identical to Python version)

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils.hpp"
#include "rope_types.hpp"

namespace rope {

// ---------------------------------------------------------------------------
// Grid axes (compile-time constants match the COAE output grid).
// ---------------------------------------------------------------------------
namespace grid {

inline std::vector<double> lst_axis() {
    std::vector<double> v(GRID_LST);
    for (int i = 0; i < GRID_LST; ++i)
        v[i] = i * (24.0 / GRID_LST);    // 0, 1/3, 2/3, …, 23.6̄
    return v;
}
inline std::vector<double> lat_axis() {
    std::vector<double> v(GRID_LAT);
    for (int i = 0; i < GRID_LAT; ++i)
        v[i] = -87.5 + i * (175.0 / (GRID_LAT - 1));
    return v;
}
inline std::vector<double> alt_axis() {
    std::vector<double> v(GRID_ALT);
    for (int i = 0; i < GRID_ALT; ++i)
        v[i] = 100.0 + i * (880.0 / (GRID_ALT - 1));
    return v;
}

} // namespace grid

// ---------------------------------------------------------------------------
// Custom exceptions — mirrors Python exceptions in interpolator.py.
// ---------------------------------------------------------------------------
struct TimeOutOfRangeError  : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct SpatialOutOfRangeError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// ---------------------------------------------------------------------------
// Query result structs — one per time mode.
// ---------------------------------------------------------------------------
struct HoldNextResult {
    std::string datetime_requested;
    std::string datetime_used;
    double      density;
    int         t_index;
};

struct InterpTimeResult {
    std::string datetime;
    double      density;
    int         t_index_left, t_index_right;
    std::string datetime_left, datetime_right;
    double      time_weight_right;
};

// ---------------------------------------------------------------------------
// DensityInterpolator
// ---------------------------------------------------------------------------
class DensityInterpolator {
public:
    explicit DensityInterpolator(const ForecastResult& res)
        : density_(res.meta_density)
        , times_(res.H)
        , H_(res.H)
    {
        if (res.H == 0)
            throw std::runtime_error("DensityInterpolator: empty ForecastResult");

        for (int i = 0; i < H_; ++i)
            times_[i] = dt::parse(res.datetimes[i]);

        lst_ax_ = grid::lst_axis();
        lat_ax_ = grid::lat_axis();
        alt_ax_ = grid::alt_axis();
    }

    // --- spatial bounds ---
    // LST has no meaningful min/max — it is periodic on [0, 24).
    double lst_min() const { return 0.0; }
    double lst_max() const { return lst_ax_.back(); }
    double lat_min() const { return lat_ax_.front(); }
    double lat_max() const { return lat_ax_.back();  }
    double alt_min() const { return alt_ax_.front(); }
    double alt_max() const { return alt_ax_.back();  }

    // --- time bounds ---
    std::string time_min() const { return dt::to_string(times_.front()); }
    std::string time_max() const { return dt::to_string(times_.back());  }

    HoldNextResult query_hold_next(
        const std::string& when, double lst, double lat, double alt_km
    ) const;

    InterpTimeResult query_interp_time(
        const std::string& when, double lst, double lat, double alt_km
    ) const;

private:
    const std::vector<float>&   density_;
    std::vector<dt::TimePoint>  times_;
    int                         H_;
    std::vector<double>         lst_ax_, lat_ax_, alt_ax_;

    // Validate spatial coordinates; throw SpatialOutOfRangeError if out of bounds.
    void check_spatial(double lst, double lat, double alt_km) const;

    // Validate time; throw TimeOutOfRangeError if out of bounds.
    // Returns bracket indices (i0, i1) with times[i0] <= tp <= times[i1].
    void check_time(dt::TimePoint tp) const;
    std::pair<int, int> bracket(dt::TimePoint tp) const;

    // Trilinear interpolation on the density grid at time index t.
    double spatial_interp(int t, double lst, double lat, double alt_km) const;

    // Linear interpolation between two scalars.
    static double lerp(double a, double b, double w) { return a + w * (b - a); }

    // Find lower-bound index in a sorted vector.
    static int lower_idx(const std::vector<double>& axis, double v) {
        auto it = std::lower_bound(axis.begin(), axis.end(), v);
        int  i  = static_cast<int>(it - axis.begin());
        return std::max(0, std::min(i - 1, static_cast<int>(axis.size()) - 2));
    }
};

} // namespace rope
