#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "rope/io/csv_reader.h"
#include <filesystem>
#include <fstream>

static const std::filesystem::path SAMPLE_CSV =
    std::filesystem::path(ROPE_CPP_FIXTURE_DIR) / "sample.csv";

using namespace Catch::Matchers;

TEST_CASE("CsvReader: basic structure") {
    rope::io::CsvReader csv(SAMPLE_CSV);
    REQUIRE(csv.nrows() == 4);
    CHECK(csv.has_column("name"));
    CHECK(csv.has_column("age"));
    CHECK(csv.has_column("score"));
    CHECK_FALSE(csv.has_column("nonexistent"));
}

TEST_CASE("CsvReader: string values") {
    rope::io::CsvReader csv(SAMPLE_CSV);
    CHECK(csv.get("name", 0) == "Alice");
    CHECK(csv.get("name", 1) == "Bob");
}

TEST_CASE("CsvReader: quoted field with comma") {
    rope::io::CsvReader csv(SAMPLE_CSV);
    // Row 2: "Charlie, Jr." — quote-enclosed field containing a comma
    CHECK(csv.get("name", 2) == "Charlie, Jr.");
}

TEST_CASE("CsvReader: leading and trailing whitespace stripped") {
    rope::io::CsvReader csv(SAMPLE_CSV);
    // Row 3: "  Diana  " in name, "  35  " in age
    CHECK(csv.get("name", 3) == "Diana");
    CHECK(csv.get_int("age", 3) == 35);
}

TEST_CASE("CsvReader: numeric conversions") {
    rope::io::CsvReader csv(SAMPLE_CSV);
    CHECK(csv.get_int("age", 0) == 30);
    CHECK_THAT(csv.get_float("score", 0), WithinAbs(98.5f, 1e-4f));
    CHECK_THAT(csv.get_float("score", 2), WithinAbs(85.3f, 1e-3f));
}

TEST_CASE("CsvReader: missing column throws") {
    rope::io::CsvReader csv(SAMPLE_CSV);
    REQUIRE_THROWS_AS(csv.get("missing", 0), std::runtime_error);
}

TEST_CASE("CsvReader: nonexistent file throws") {
    REQUIRE_THROWS_AS(
        rope::io::CsvReader(std::filesystem::path("does_not_exist.csv")),
        std::runtime_error
    );
}
