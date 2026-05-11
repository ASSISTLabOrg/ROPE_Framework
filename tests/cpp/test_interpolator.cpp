#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "rope/interpolate/grid_interpolator.h"
#include "rope/core/types.h"
#include "rope/core/datetime.h"
#include <vector>
#include <cmath>

using namespace Catch::Matchers;
using rope::ForecastGrid;
using rope::GRID_VOXELS;
using rope::interpolate::GridInterpolator;
using rope::interpolate::SpatialOutOfRangeError;
using rope::interpolate::TimeOutOfRangeError;

// Epoch-hour helper
static rope::TimePoint epoch_h(int h) {
    return rope::parse_datetime("2024-01-01T00:00:00") + static_cast<rope::TimePoint>(h) * 3600;
}

// Minimal 2-step grid, all density=1.0, uncertainty=0.5 at every voxel.
// log10(1.0)=0 → interpolation returns 10^0=1.0 everywhere.
static ForecastGrid make_uniform_grid(float den = 1.0f, float unc = 0.5f) {
    ForecastGrid g;
    g.H = 2;
    g.density.assign(static_cast<std::size_t>(2) * GRID_VOXELS, den);
    g.uncertainty.assign(static_cast<std::size_t>(2) * GRID_VOXELS, unc);
    g.times = {epoch_h(0), epoch_h(1)};
    return g;
}

TEST_CASE("GridInterpolator: construction and metadata") {
    auto grid = make_uniform_grid();
    GridInterpolator gi(grid);
    CHECK(gi.time_min() == epoch_h(0));
    CHECK(gi.time_max() == epoch_h(1));
}

TEST_CASE("GridInterpolator: empty grid throws on construction") {
    ForecastGrid empty;
    REQUIRE_THROWS_AS(GridInterpolator{empty}, std::runtime_error);
}

TEST_CASE("GridInterpolator: query_interp returns positive density and uncertainty") {
    auto grid = make_uniform_grid(1.0f, 0.5f);
    GridInterpolator gi(grid);

    auto r = gi.query_interp(epoch_h(0), 12.0, 0.0, 400.0);
    CHECK(r.density     > 0.0);
    CHECK(r.uncertainty > 0.0);
}

TEST_CASE("GridInterpolator: query_hold returns positive density and uncertainty") {
    auto grid = make_uniform_grid(1.0f, 0.5f);
    GridInterpolator gi(grid);

    auto r = gi.query_hold(epoch_h(0), 12.0, 0.0, 400.0);
    CHECK(r.density     > 0.0);
    CHECK(r.uncertainty > 0.0);
}

TEST_CASE("GridInterpolator: uniform grid interpolates to constant") {
    auto grid = make_uniform_grid(1.0f, 0.5f);
    GridInterpolator gi(grid);

    // With all voxels = 1.0, spatial_interp in log10 space returns 0, then 10^0 = 1.
    auto r = gi.query_interp(epoch_h(0), 6.0, 30.0, 300.0);
    CHECK_THAT(r.density, WithinRel(1.0, 1e-5));
}

TEST_CASE("GridInterpolator: time interpolation between two steps") {
    // density=1.0 at t0 and density=100.0 at t1.
    // Spatial interpolation is in log10 space → r0.density=1.0, r1.density=100.0.
    // Temporal interpolation is LINEAR: lerp(1.0, 100.0, 0.5) = 50.5.
    ForecastGrid g;
    g.H = 2;
    g.density.assign(static_cast<std::size_t>(2) * GRID_VOXELS, 1.0f);
    g.uncertainty.assign(static_cast<std::size_t>(2) * GRID_VOXELS, 1.0f);
    std::fill(g.density.begin() + GRID_VOXELS, g.density.end(), 100.0f);
    g.times = {epoch_h(0), epoch_h(2)};  // 2-hour span

    GridInterpolator gi(g);
    auto r = gi.query_interp(epoch_h(1), 12.0, 0.0, 400.0);  // midpoint
    CHECK_THAT(r.density, WithinRel(50.5, 1e-4));
}

TEST_CASE("GridInterpolator: query at exact time step equals hold") {
    auto grid = make_uniform_grid();
    GridInterpolator gi(grid);

    auto r_interp = gi.query_interp(epoch_h(0), 12.0, 0.0, 400.0);
    auto r_hold   = gi.query_hold(epoch_h(0), 12.0, 0.0, 400.0);
    CHECK_THAT(r_interp.density,     WithinRel(r_hold.density,     1e-9));
    CHECK_THAT(r_interp.uncertainty, WithinRel(r_hold.uncertainty, 1e-9));
}

TEST_CASE("GridInterpolator: LST is periodic - query at 25h equals query at 1h") {
    auto grid = make_uniform_grid();
    GridInterpolator gi(grid);

    auto r1  = gi.query_interp(epoch_h(0), 1.0,  0.0, 400.0);
    auto r25 = gi.query_interp(epoch_h(0), 25.0, 0.0, 400.0);
    CHECK_THAT(r1.density,  WithinRel(r25.density,  1e-9));
    CHECK_THAT(r1.uncertainty, WithinRel(r25.uncertainty, 1e-9));
}

TEST_CASE("GridInterpolator: out-of-range latitude throws") {
    auto grid = make_uniform_grid();
    GridInterpolator gi(grid);
    REQUIRE_THROWS_AS(gi.query_interp(epoch_h(0), 12.0,  90.0, 400.0), SpatialOutOfRangeError);
    REQUIRE_THROWS_AS(gi.query_interp(epoch_h(0), 12.0, -90.0, 400.0), SpatialOutOfRangeError);
}

TEST_CASE("GridInterpolator: out-of-range altitude throws") {
    auto grid = make_uniform_grid();
    GridInterpolator gi(grid);
    REQUIRE_THROWS_AS(gi.query_interp(epoch_h(0), 12.0, 0.0,  50.0), SpatialOutOfRangeError);
    REQUIRE_THROWS_AS(gi.query_interp(epoch_h(0), 12.0, 0.0, 990.0), SpatialOutOfRangeError);
}

TEST_CASE("GridInterpolator: out-of-range time throws") {
    auto grid = make_uniform_grid();
    GridInterpolator gi(grid);
    REQUIRE_THROWS_AS(gi.query_interp(epoch_h(0) - 1, 12.0, 0.0, 400.0), TimeOutOfRangeError);
    REQUIRE_THROWS_AS(gi.query_interp(epoch_h(1) + 1, 12.0, 0.0, 400.0), TimeOutOfRangeError);
}

TEST_CASE("GridInterpolator: density is non-negative for positive grid values") {
    auto grid = make_uniform_grid(0.001f, 0.001f);
    GridInterpolator gi(grid);
    for (double lst : {0.0, 6.0, 12.0, 18.0}) {
        for (double lat : {-80.0, 0.0, 80.0}) {
            auto r = gi.query_interp(epoch_h(0), lst, lat, 400.0);
            CHECK(r.density     >= 0.0);
            CHECK(r.uncertainty >= 0.0);
        }
    }
}
