#include <catch2/catch_test_macros.hpp>
#include "rope/core/datetime.h"

TEST_CASE("parse_datetime: ISO 8601 T-separator") {
    rope::TimePoint tp = rope::parse_datetime("2024-01-15T06:30:00");
    CHECK(rope::format_iso(tp) == "2024-01-15T06:30:00");
}

TEST_CASE("parse_datetime: space separator") {
    rope::TimePoint tp = rope::parse_datetime("2024-01-15 06:30:00");
    CHECK(rope::format_iso(tp) == "2024-01-15T06:30:00");
}

TEST_CASE("parse_datetime and format_iso round-trip") {
    const char* cases[] = {
        "2000-01-01T00:00:00",
        "2024-06-21T12:00:00",
        "2024-12-31T23:59:59",
        "1970-01-01T00:00:00",
    };
    for (const char* s : cases) {
        INFO("testing: " << s);
        CHECK(rope::format_iso(rope::parse_datetime(s)) == s);
    }
}

TEST_CASE("parse_datetime: Unix epoch") {
    rope::TimePoint tp = rope::parse_datetime("1970-01-01T00:00:00");
    CHECK(tp == 0);
}

TEST_CASE("parse_datetime: known timestamp") {
    // 2024-01-01 00:00:00 UTC = 1704067200
    rope::TimePoint tp = rope::parse_datetime("2024-01-01T00:00:00");
    CHECK(tp == 1704067200LL);
}

TEST_CASE("parse_datetime: bad format throws") {
    REQUIRE_THROWS_AS(rope::parse_datetime("not-a-date"), std::runtime_error);
    REQUIRE_THROWS_AS(rope::parse_datetime("2024/01/01"), std::runtime_error);
}

TEST_CASE("floor_hour rounds down to whole hour") {
    // 2024-01-01T01:45:30 → 2024-01-01T01:00:00
    rope::TimePoint tp = rope::parse_datetime("2024-01-01T01:45:30");
    rope::TimePoint floored = rope::floor_hour(tp);
    CHECK(rope::format_iso(floored) == "2024-01-01T01:00:00");
}

TEST_CASE("floor_hour: already on the hour is unchanged") {
    rope::TimePoint tp = rope::parse_datetime("2024-01-01T06:00:00");
    CHECK(rope::floor_hour(tp) == tp);
}

TEST_CASE("unpack: extracts hour, doy, year") {
    // 2024-01-01T06:00:00 → hour=6, doy=1, year=2024
    rope::TimePoint tp = rope::parse_datetime("2024-01-01T06:00:00");
    int h, doy, yr;
    rope::unpack(tp, h, doy, yr);
    CHECK(h   == 6);
    CHECK(doy == 1);
    CHECK(yr  == 2024);
}

TEST_CASE("harmonics: unit-circle properties at midnight") {
    // At hour=0: sin(0)=0, cos(0)=1 for the diurnal harmonics
    rope::TimePoint tp = rope::parse_datetime("2024-01-01T00:00:00");
    float t1, t2, t3, t4, doy;
    rope::harmonics(tp, t1, t2, t3, t4, doy);
    CHECK(std::abs(t1) < 1e-5f);   // sin(0) ≈ 0
    CHECK(std::abs(t2 - 1.0f) < 1e-5f);  // cos(0) ≈ 1
    CHECK(doy >= 1.0f);
}
