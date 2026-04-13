// interpolator.cpp — DensityInterpolator implementation.

#include "interpolator.hpp"

#include <algorithm>
#include <sstream>

namespace rope {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

void DensityInterpolator::check_spatial(
    double lst, double lat, double alt_km
) const {
    // LST is periodic — any value is valid; normalize_lst() handles the wrap.
    (void)lst;

    if (lat < lat_min() || lat > lat_max()) {
        std::ostringstream oss;
        oss << "Requested latitude " << lat << " is outside the ROPE grid bounds ["
            << lat_min() << ", " << lat_max() << "]";
        throw SpatialOutOfRangeError(oss.str());
    }
    if (alt_km < alt_min() || alt_km > alt_max()) {
        std::ostringstream oss;
        oss << "Requested altitude " << alt_km << " km is outside the ROPE grid bounds ["
            << alt_min() << ", " << alt_max() << "] km";
        throw SpatialOutOfRangeError(oss.str());
    }
}

void DensityInterpolator::check_time(dt::TimePoint tp) const {
    if (tp < times_.front() || tp > times_.back()) {
        std::ostringstream oss;
        oss << "Requested time " << dt::to_string(tp)
            << " is outside the propagated ROPE window ["
            << dt::to_string(times_.front()) << ", "
            << dt::to_string(times_.back()) << "]";
        throw TimeOutOfRangeError(oss.str());
    }
}

std::pair<int, int> DensityInterpolator::bracket(dt::TimePoint tp) const {
    // Binary search for the first time >= tp.
    auto it = std::lower_bound(times_.begin(), times_.end(), tp);
    int i1 = static_cast<int>(it - times_.begin());
    // Exact match: i0 == i1, so use the same index for both sides.
    if (it != times_.end() && *it == tp)
        return {i1, i1};
    // In range: i1 >= 1.
    int i0 = i1 - 1;
    return {i0, i1};
}

// ---------------------------------------------------------------------------
// Trilinear interpolation on the density grid at time step t.
//
// Grid layout: [t][lst][lat][alt]  (GRID_LST × GRID_LAT × GRID_ALT)
//
// ---------------------------------------------------------------------------
double DensityInterpolator::spatial_interp(
    int t, double lst, double lat, double alt_km
) const {
    const float* base = density_.data()
        + static_cast<size_t>(t) * GRID_VOXELS;

    // --- LST: uniform step = 24/72 h, periodic ---
    constexpr double LST_STEP = 24.0 / GRID_LST;
    // Normalize to [0, 24).
    lst = lst - std::floor(lst / 24.0) * 24.0;
    int li0 = static_cast<int>(lst / LST_STEP);
    if (li0 >= GRID_LST) li0 = GRID_LST - 1;   // clamp fp rounding at lst=24
    int    li1  = (li0 + 1) % GRID_LST;          // wraps 71 → 0
    double lst1 = (li1 == 0) ? 24.0 : lst_ax_[li1];
    double wl   = (lst - lst_ax_[li0]) / (lst1 - lst_ax_[li0]);

    // --- Latitude: uniform, -87.5° to +87.5° ---
    constexpr double LAT_STEP = 175.0 / (GRID_LAT - 1);
    int ai0 = static_cast<int>((lat - lat_ax_.front()) / LAT_STEP);
    ai0 = std::max(0, std::min(ai0, GRID_LAT - 2));
    int    ai1 = ai0 + 1;
    double wa  = (lat - lat_ax_[ai0]) / (lat_ax_[ai1] - lat_ax_[ai0]);

    // --- Altitude: uniform, 100 to 980 km ---
    constexpr double ALT_STEP = 880.0 / (GRID_ALT - 1);
    int zi0 = static_cast<int>((alt_km - alt_ax_.front()) / ALT_STEP);
    zi0 = std::max(0, std::min(zi0, GRID_ALT - 2));
    int    zi1 = zi0 + 1;
    double wz  = (alt_km - alt_ax_[zi0]) / (alt_ax_[zi1] - alt_ax_[zi0]);

    // Flat index into [GRID_LST][GRID_LAT][GRID_ALT].
    auto idx = [](int l, int a, int z) -> size_t {
        return static_cast<size_t>(l) * (GRID_LAT * GRID_ALT)
             + static_cast<size_t>(a) * GRID_ALT
             + z;
    };

    // Fetch 8 corners in log10 space.
    // Physical density is always positive; the floor guards against any
    // underflow zeros that might have crept through at extreme altitudes.
    auto logv = [&](int l, int a, int z) -> double {
        float v = base[idx(l, a, z)];
        return (v > 0.0f) ? std::log10(static_cast<double>(v)) : -300.0;
    };

    double v000 = logv(li0, ai0, zi0);
    double v001 = logv(li0, ai0, zi1);
    double v010 = logv(li0, ai1, zi0);
    double v011 = logv(li0, ai1, zi1);
    double v100 = logv(li1, ai0, zi0);
    double v101 = logv(li1, ai0, zi1);
    double v110 = logv(li1, ai1, zi0);
    double v111 = logv(li1, ai1, zi1);

    double c00 = lerp(v000, v100, wl);
    double c01 = lerp(v001, v101, wl);
    double c10 = lerp(v010, v110, wl);
    double c11 = lerp(v011, v111, wl);
    double c0  = lerp(c00,  c10,  wa);
    double c1  = lerp(c01,  c11,  wa);
    return std::pow(10.0, lerp(c0, c1, wz));
}

// ---------------------------------------------------------------------------
// Public query methods
// ---------------------------------------------------------------------------

HoldNextResult DensityInterpolator::query_hold_next(
    const std::string& when, double lst, double lat, double alt_km
) const {
    dt::TimePoint tp = dt::parse(when);
    check_time(tp);
    check_spatial(lst, lat, alt_km);

    auto [i0, i1] = bracket(tp);
    // Use i0 when exactly on a model time, otherwise i1 (ceil).
    int use_i = (times_[i0] == tp) ? i0 : i1;

    HoldNextResult r;
    r.datetime_requested = dt::to_string(tp);
    r.datetime_used      = dt::to_string(times_[use_i]);
    r.density            = spatial_interp(use_i, lst, lat, alt_km);
    r.t_index            = use_i;
    return r;
}

InterpTimeResult DensityInterpolator::query_interp_time(
    const std::string& when, double lst, double lat, double alt_km
) const {
    dt::TimePoint tp = dt::parse(when);
    check_time(tp);
    check_spatial(lst, lat, alt_km);

    auto [i0, i1] = bracket(tp);

    InterpTimeResult r;
    r.datetime = dt::to_string(tp);

    if (i0 == i1) {
        // Exactly on a model time — no temporal interpolation needed.
        r.density            = spatial_interp(i0, lst, lat, alt_km);
        r.t_index_left       = i0;
        r.t_index_right      = i0;
        r.datetime_left      = dt::to_string(times_[i0]);
        r.datetime_right     = dt::to_string(times_[i0]);
        r.time_weight_right  = 0.0;
    } else {
        double dt_span = static_cast<double>(times_[i1] - times_[i0]);
        double dt_off  = static_cast<double>(tp         - times_[i0]);
        double w       = dt_off / dt_span;

        double v0 = spatial_interp(i0, lst, lat, alt_km);
        double v1 = spatial_interp(i1, lst, lat, alt_km);

        r.density            = lerp(v0, v1, w);
        r.t_index_left       = i0;
        r.t_index_right      = i1;
        r.datetime_left      = dt::to_string(times_[i0]);
        r.datetime_right     = dt::to_string(times_[i1]);
        r.time_weight_right  = w;
    }
    return r;
}

} // namespace rope
