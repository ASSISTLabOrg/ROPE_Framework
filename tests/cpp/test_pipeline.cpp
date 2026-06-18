#include <catch2/catch_test_macros.hpp>
#include "rope/forecast/pipeline.h"
#include "rope/core/types.h"
#include <filesystem>

namespace fs = std::filesystem;

static rope::forecast::Config make_test_config() {
    rope::forecast::Config cfg;
    cfg.exported_dir          = fs::path(ROPE_FIXTURE_MODELS);
    cfg.driver_path           = fs::path(ROPE_FIXTURE_DIR) / "sw_test.csv";
    cfg.intra_threads_base    = 1;
    cfg.intra_threads_meta    = 1;
    cfg.intra_threads_decoder = 1;
    cfg.compute_uncertainty   = true;
    return cfg;
}

// Start time that falls within the space weather fixture (sw_test.csv covers
// 2023-12-31T22:00:00 through 2024-01-01T03:00:00, 6 hourly rows).
// Horizon=3 with seq_len=3 requires (S-1)+(H+1) = 2+4 = 6 rows.
static const char* TEST_START = "2024-01-01 00:00:00";
static constexpr int TEST_HORIZON = 3;

TEST_CASE("Pipeline: loads successfully from synthetic fixtures") {
    auto pipe = rope::forecast::load(make_test_config());
    REQUIRE(pipe != nullptr);
}

TEST_CASE("Pipeline: run() returns correctly shaped ForecastGrid") {
    auto pipe = rope::forecast::load(make_test_config());
    auto grid = pipe->run(TEST_START, TEST_HORIZON);

    CHECK(grid.H == TEST_HORIZON);
    CHECK(static_cast<int>(grid.density.size())     == TEST_HORIZON * rope::GRID_VOXELS);
    CHECK(static_cast<int>(grid.uncertainty.size()) == TEST_HORIZON * rope::GRID_VOXELS);
    CHECK(static_cast<int>(grid.times.size())       == TEST_HORIZON);
}

TEST_CASE("Pipeline: density is positive everywhere") {
    auto pipe = rope::forecast::load(make_test_config());
    auto grid = pipe->run(TEST_START, TEST_HORIZON);
    for (float d : grid.density)
        CHECK(d > 0.0f);
}

TEST_CASE("Pipeline: uncertainty is non-negative everywhere") {
    auto pipe = rope::forecast::load(make_test_config());
    auto grid = pipe->run(TEST_START, TEST_HORIZON);
    for (float u : grid.uncertainty)
        CHECK(u >= 0.0f);
}

TEST_CASE("Pipeline: time steps are monotonically increasing") {
    auto pipe = rope::forecast::load(make_test_config());
    auto grid = pipe->run(TEST_START, TEST_HORIZON);
    for (int i = 1; i < grid.H; ++i)
        CHECK(grid.times[i] > grid.times[i - 1]);
}

TEST_CASE("Pipeline: uncertainty disabled sets it to zero") {
    auto cfg = make_test_config();
    cfg.compute_uncertainty = false;
    auto pipe = rope::forecast::load(cfg);
    auto grid = pipe->run(TEST_START, TEST_HORIZON);
    for (float u : grid.uncertainty)
        CHECK(u == 0.0f);
    // Density should still be positive (fast path without UT)
    for (float d : grid.density)
        CHECK(d > 0.0f);
}

TEST_CASE("Pipeline: run() is deterministic - repeated calls produce identical output") {
    auto pipe  = rope::forecast::load(make_test_config());
    auto grid1 = pipe->run(TEST_START, TEST_HORIZON);
    auto grid2 = pipe->run(TEST_START, TEST_HORIZON);

    CHECK(grid1.H == grid2.H);
    CHECK(grid1.times == grid2.times);
    CHECK(grid1.density == grid2.density);
    CHECK(grid1.uncertainty == grid2.uncertainty);
}
