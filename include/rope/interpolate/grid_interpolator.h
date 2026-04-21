#pragma once
// GridInterpolator — spatiotemporal density and uncertainty query on a ForecastGrid.
//
// Grid axes (fixed, matching COAE output):
//   LST : linspace(0, 23.667, 72)    [hours, periodic]
//   lat : linspace(-87.5, 87.5, 36)  [degrees]
//   alt : linspace(100, 980, 45)     [km]
//
// Spatial interpolation is performed in log10 space, then exponentiated.
// Both density and uncertainty use the same spatial weights.
//
// Time modes:
//   HOLD   — snap to the next model hour; no temporal blending.
//   INTERP — trilinear spatial at both bracket hours, then linear time blend.

#include "rope/core/datetime.h"
#include "rope/core/types.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace rope::interpolate {

// ---------------------------------------------------------------------------
// Exceptions
// ---------------------------------------------------------------------------
struct TimeOutOfRangeError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct SpatialOutOfRangeError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------
struct InterpolationResult {
    double density;      // kg/m³
    double uncertainty;  // kg/m³
};

// ---------------------------------------------------------------------------
// GridInterpolator
// ---------------------------------------------------------------------------
class GridInterpolator {
public:
    explicit GridInterpolator(const ForecastGrid& grid);

    TimePoint time_min() const noexcept { return times_.front(); }
    TimePoint time_max() const noexcept { return times_.back();  }

    // Hold: snap to the next model hour, then spatial interpolate.
    InterpolationResult query_hold(
        TimePoint time_unix, double lst, double lat, double alt_km) const;

    // Interp: spatial interp at both bracket hours, then linear time blend.
    InterpolationResult query_interp(
        TimePoint time_unix, double lst, double lat, double alt_km) const;

private:
    const ForecastGrid&    grid_;
    std::vector<TimePoint> times_;
    int                    H_;
    std::vector<double>    lst_ax_, lat_ax_, alt_ax_;

    void check_spatial(double lst, double lat, double alt_km) const;
    void check_time(TimePoint tp) const;
    std::pair<int, int> bracket(TimePoint tp) const;

    // Trilinear interpolation in log10 space at time step t, for a given field.
    double spatial_interp(const float* time_slice,
                          double lst, double lat, double alt_km) const;

    // Interpolate both density and uncertainty at a fixed time step.
    InterpolationResult spatial_both(int t,
                                     double lst, double lat, double alt_km) const;

    static double lerp(double a, double b, double w) noexcept { return a + w*(b-a); }
    static int    lower_idx(const std::vector<double>& ax, double v) noexcept;
};

} // namespace rope::interpolate
