#include "rope/interpolate/grid_interpolator.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace rope::interpolate {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

static std::vector<double> make_lst_axis() {
    std::vector<double> v(GRID_LST);
    for (int i = 0; i < GRID_LST; ++i)
        v[i] = i * (24.0 / GRID_LST);
    return v;
}
static std::vector<double> make_lat_axis() {
    std::vector<double> v(GRID_LAT);
    for (int i = 0; i < GRID_LAT; ++i)
        v[i] = -87.5 + i * (175.0 / (GRID_LAT - 1));
    return v;
}
static std::vector<double> make_alt_axis() {
    std::vector<double> v(GRID_ALT);
    for (int i = 0; i < GRID_ALT; ++i)
        v[i] = 100.0 + i * (880.0 / (GRID_ALT - 1));
    return v;
}

GridInterpolator::GridInterpolator(const ForecastGrid& grid)
    : grid_(grid)
    , times_(grid.times)
    , H_(grid.H)
    , lst_ax_(make_lst_axis())
    , lat_ax_(make_lat_axis())
    , alt_ax_(make_alt_axis())
{
    if (H_ == 0)
        throw std::runtime_error("GridInterpolator: empty ForecastGrid");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

void GridInterpolator::check_spatial(double lst, double lat, double alt_km) const {
    (void)lst;  // LST is periodic — any value is valid
    if (lat < lat_ax_.front() || lat > lat_ax_.back()) {
        std::ostringstream oss;
        oss << "Requested latitude " << lat
            << " is outside the ROPE grid bounds ["
            << lat_ax_.front() << ", " << lat_ax_.back() << "]";
        throw SpatialOutOfRangeError(oss.str());
    }
    if (alt_km < alt_ax_.front() || alt_km > alt_ax_.back()) {
        std::ostringstream oss;
        oss << "Requested altitude " << alt_km
            << " km is outside the ROPE grid bounds ["
            << alt_ax_.front() << ", " << alt_ax_.back() << "] km";
        throw SpatialOutOfRangeError(oss.str());
    }
}

void GridInterpolator::check_time(TimePoint tp) const {
    if (tp < times_.front() || tp > times_.back()) {
        std::ostringstream oss;
        oss << "Requested time " << format_iso(tp)
            << " is outside the forecast window ["
            << format_iso(times_.front()) << ", "
            << format_iso(times_.back()) << "]";
        throw TimeOutOfRangeError(oss.str());
    }
}

std::pair<int, int> GridInterpolator::bracket(TimePoint tp) const {
    auto it = std::lower_bound(times_.begin(), times_.end(), tp);
    int i1 = static_cast<int>(it - times_.begin());
    if (it != times_.end() && *it == tp) return {i1, i1};
    return {i1 - 1, i1};
}

int GridInterpolator::lower_idx(const std::vector<double>& ax, double v) noexcept {
    auto it = std::lower_bound(ax.begin(), ax.end(), v);
    int i   = static_cast<int>(it - ax.begin());
    return std::max(0, std::min(i - 1, static_cast<int>(ax.size()) - 2));
}

// ---------------------------------------------------------------------------
// Trilinear interpolation in log10 space
// ---------------------------------------------------------------------------

double GridInterpolator::spatial_interp(const float* base,
                                         double lst, double lat, double alt_km) const {
    // --- LST: uniform step, periodic ---
    constexpr double LST_STEP = 24.0 / GRID_LST;
    lst = lst - std::floor(lst / 24.0) * 24.0;  // normalize to [0, 24)
    int li0 = static_cast<int>(lst / LST_STEP);
    if (li0 >= GRID_LST) li0 = GRID_LST - 1;
    int    li1  = (li0 + 1) % GRID_LST;          // wraps 71 → 0
    double lst1 = (li1 == 0) ? 24.0 : lst_ax_[li1];
    double wl   = (lst - lst_ax_[li0]) / (lst1 - lst_ax_[li0]);

    // --- Latitude: uniform ---
    constexpr double LAT_STEP = 175.0 / (GRID_LAT - 1);
    int ai0 = static_cast<int>((lat - lat_ax_.front()) / LAT_STEP);
    ai0 = std::max(0, std::min(ai0, GRID_LAT - 2));
    int    ai1 = ai0 + 1;
    double wa  = (lat - lat_ax_[ai0]) / (lat_ax_[ai1] - lat_ax_[ai0]);

    // --- Altitude: uniform ---
    constexpr double ALT_STEP = 880.0 / (GRID_ALT - 1);
    int zi0 = static_cast<int>((alt_km - alt_ax_.front()) / ALT_STEP);
    zi0 = std::max(0, std::min(zi0, GRID_ALT - 2));
    int    zi1 = zi0 + 1;
    double wz  = (alt_km - alt_ax_[zi0]) / (alt_ax_[zi1] - alt_ax_[zi0]);

    auto idx = [](int l, int a, int z) -> std::size_t {
        return static_cast<std::size_t>(l) * (GRID_LAT * GRID_ALT)
             + static_cast<std::size_t>(a) * GRID_ALT
             + z;
    };

    // Interpolate in log10 space; floor at -300 to guard against underflow zeros.
    auto logv = [&](int l, int a, int z) -> double {
        float v = base[idx(l, a, z)];
        return (v > 0.0f) ? std::log10(static_cast<double>(v)) : -300.0;
    };

    double c00 = lerp(logv(li0,ai0,zi0), logv(li1,ai0,zi0), wl);
    double c01 = lerp(logv(li0,ai0,zi1), logv(li1,ai0,zi1), wl);
    double c10 = lerp(logv(li0,ai1,zi0), logv(li1,ai1,zi0), wl);
    double c11 = lerp(logv(li0,ai1,zi1), logv(li1,ai1,zi1), wl);
    double c0  = lerp(c00, c10, wa);
    double c1  = lerp(c01, c11, wa);
    return std::pow(10.0, lerp(c0, c1, wz));
}

InterpolationResult GridInterpolator::spatial_both(
    int t, double lst, double lat, double alt_km) const
{
    const float* d = grid_.density_at(t);
    const float* u = grid_.uncertainty_at(t);
    return {
        spatial_interp(d, lst, lat, alt_km),
        spatial_interp(u, lst, lat, alt_km),
    };
}

// ---------------------------------------------------------------------------
// Public query methods
// ---------------------------------------------------------------------------

InterpolationResult GridInterpolator::query_hold(
    TimePoint time_unix, double lst, double lat, double alt_km) const
{
    check_time(time_unix);
    check_spatial(lst, lat, alt_km);

    auto [i0, i1] = bracket(time_unix);
    int use_i     = (times_[i0] == time_unix) ? i0 : i1;  // ceil
    return spatial_both(use_i, lst, lat, alt_km);
}

InterpolationResult GridInterpolator::query_interp(
    TimePoint time_unix, double lst, double lat, double alt_km) const
{
    check_time(time_unix);
    check_spatial(lst, lat, alt_km);

    auto [i0, i1] = bracket(time_unix);

    if (i0 == i1)
        return spatial_both(i0, lst, lat, alt_km);

    double span = static_cast<double>(times_[i1] - times_[i0]);
    double off  = static_cast<double>(time_unix   - times_[i0]);
    double w    = off / span;

    auto r0 = spatial_both(i0, lst, lat, alt_km);
    auto r1 = spatial_both(i1, lst, lat, alt_km);
    return {
        lerp(r0.density,     r1.density,     w),
        lerp(r0.uncertainty, r1.uncertainty, w),
    };
}

} // namespace rope::interpolate
