#include <catch2/catch_test_macros.hpp>
#include "rope/io/driver_cache.h"
#include <chrono>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

// NOTE: DriverCacheManager::download() is a hard "not yet implemented" stub
// (see driver_cache.cpp), so refresh() always fails today. That makes the
// network path untestable, but it also gives us a deterministic way to
// exercise the stale-cache-fallback and hard-failure behaviors below without
// touching the network.

static fs::path make_cache_dir(const std::string& name) {
    auto dir = fs::temp_directory_path() / name;
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    return dir;
}

static void write_dummy_file(const fs::path& path) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f << "dummy";
}

TEST_CASE("known_sources: registry contains the documented CelesTrak sources") {
    const auto& sources = rope::io::known_sources();
    CHECK(sources.count("celestrak_sw")     == 1);
    CHECK(sources.count("celestrak_sw_all") == 1);
}

TEST_CASE("DriverCacheManager: unknown source throws") {
    auto dir = make_cache_dir("rope_test_cache_unknown");
    rope::io::DriverCacheManager mgr(dir, /*max_age_hours=*/24);
    REQUIRE_THROWS_AS(mgr.get_path("not_a_real_source"), std::runtime_error);
}

TEST_CASE("DriverCacheManager: no cached file and a failed refresh throws") {
    auto dir = make_cache_dir("rope_test_cache_missing");
    rope::io::DriverCacheManager mgr(dir, /*max_age_hours=*/24);
    // Nothing cached, and download() can't succeed — nothing to fall back
    // to, so this must fail loudly rather than return a bad path.
    REQUIRE_THROWS_AS(mgr.get_path("celestrak_sw"), std::runtime_error);
}

TEST_CASE("DriverCacheManager: stale cache falls back to the stale file when refresh fails") {
    auto dir  = make_cache_dir("rope_test_cache_stale");
    auto dest = dir / "celestrak_sw.swbin";
    write_dummy_file(dest);

    // Backdate the file well past max_age_hours so a refresh is attempted.
    fs::last_write_time(dest,
        fs::file_time_type::clock::now() - std::chrono::hours(48));

    rope::io::DriverCacheManager mgr(dir, /*max_age_hours=*/1);
    auto path = mgr.get_path("celestrak_sw");
    CHECK(path == dest);
    CHECK(fs::exists(path));
}

TEST_CASE("DriverCacheManager: fresh cache file is returned without refreshing") {
    auto dir  = make_cache_dir("rope_test_cache_fresh");
    auto dest = dir / "celestrak_sw.swbin";
    write_dummy_file(dest);  // mtime defaults to "now" — within max_age_hours

    rope::io::DriverCacheManager mgr(dir, /*max_age_hours=*/24);
    auto path = mgr.get_path("celestrak_sw");
    CHECK(path == dest);
}
