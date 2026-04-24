#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "rope/io/config_reader.h"
#include <filesystem>
#include <fstream>

using namespace Catch::Matchers;

static std::filesystem::path write_temp_conf(const std::string& content) {
    auto path = std::filesystem::temp_directory_path() / "rope_test_config.conf";
    std::ofstream f(path);
    f << content;
    return path;
}

TEST_CASE("ConfigReader: section.key lookup") {
    auto p = write_temp_conf(
        "[models]\n"
        "dir = /tmp/models\n"
        "count = 15\n"
        "[decoder]\n"
        "device = cpu\n"
    );
    rope::io::ConfigReader cfg(p);
    CHECK(cfg.get("models.dir")    == "/tmp/models");
    CHECK(cfg.get("models.count")  == "15");
    CHECK(cfg.get("decoder.device") == "cpu");
}

TEST_CASE("ConfigReader: has() and default fallback") {
    auto p = write_temp_conf("[section]\nkey = value\n");
    rope::io::ConfigReader cfg(p);
    CHECK(cfg.has("section.key"));
    CHECK_FALSE(cfg.has("section.missing"));
    CHECK(cfg.get("section.missing", "default") == "default");
}

TEST_CASE("ConfigReader: numeric getters") {
    auto p = write_temp_conf("[tuning]\nthreads = 4\nscale = 1.5\n");
    rope::io::ConfigReader cfg(p);
    CHECK(cfg.get_int("tuning.threads") == 4);
    CHECK_THAT(cfg.get_double("tuning.scale"), WithinAbs(1.5, 1e-9));
    CHECK(cfg.get_int("tuning.absent", 99) == 99);
}

TEST_CASE("ConfigReader: comments stripped") {
    auto p = write_temp_conf(
        "[s]\n"
        "k = hello  # this is a comment\n"
    );
    rope::io::ConfigReader cfg(p);
    CHECK(cfg.get("s.k") == "hello");
}

TEST_CASE("ConfigReader: required key throws when absent") {
    auto p = write_temp_conf("[s]\nk = v\n");
    rope::io::ConfigReader cfg(p);
    REQUIRE_THROWS_AS(cfg.get("s.missing"), std::runtime_error);
}

TEST_CASE("ConfigReader: nonexistent file throws") {
    REQUIRE_THROWS_AS(
        rope::io::ConfigReader(std::filesystem::path("no_such_file.conf")),
        std::runtime_error
    );
}
