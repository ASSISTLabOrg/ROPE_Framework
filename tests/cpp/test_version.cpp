#include <catch2/catch_test_macros.hpp>
#include "rope/core/version.h"
#include <string>
#include <regex>

TEST_CASE("version string is non-empty") {
    const char* v = rope::version::string();
    REQUIRE(v != nullptr);
    CHECK(std::string(v).size() > 0);
}

TEST_CASE("version string has semver shape X.Y.Z") {
    const std::string v = rope::version::string();
    const std::regex semver(R"(\d+\.\d+\.\d+)");
    CHECK(std::regex_match(v, semver));
}

TEST_CASE("version integer components are non-negative") {
    CHECK(rope::version::major_v >= 0);
    CHECK(rope::version::minor_v >= 0);
    CHECK(rope::version::patch_v >= 0);
}

TEST_CASE("version string is consistent with integer components") {
    const std::string v = rope::version::string();
    const std::string expected =
        std::to_string(rope::version::major_v) + "." +
        std::to_string(rope::version::minor_v) + "." +
        std::to_string(rope::version::patch_v);
    CHECK(v == expected);
}
