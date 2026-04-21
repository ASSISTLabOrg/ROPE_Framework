#pragma once
// utils.hpp — UTC datetime parsing, arithmetic, and harmonic feature extraction.
//
// All times are handled as seconds since the Unix epoch (UTC).
// The string representation is always "YYYY-MM-DD HH:MM:SS".

#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <stdexcept>
#include <string>

namespace rope {
namespace dt {

using TimePoint = std::int64_t;   // UTC seconds since epoch

// ---------------------------------------------------------------------------
// Portable UTC mktime (uses timegm on POSIX, _mkgmtime on Windows).
// ---------------------------------------------------------------------------
inline TimePoint tm_to_utc(std::tm& t) {
#ifdef _WIN32
    return static_cast<TimePoint>(_mkgmtime(&t));
#else
    return static_cast<TimePoint>(timegm(&t));
#endif
}

// ---------------------------------------------------------------------------
// Parse "YYYY-MM-DD HH:MM:SS" → TimePoint (UTC).
// ---------------------------------------------------------------------------
inline TimePoint parse(const std::string& s) {
    std::tm t{};
#ifdef _WIN32
    int yr, mo, dy, hr, mn, sc;
    if (std::sscanf(s.c_str(), "%d-%d-%d %d:%d:%d", &yr, &mo, &dy, &hr, &mn, &sc) != 6)
        throw std::runtime_error("datetime: bad format: " + s);
    t.tm_year = yr - 1900; t.tm_mon = mo - 1; t.tm_mday = dy;
    t.tm_hour = hr; t.tm_min = mn; t.tm_sec = sc;
#else
    if (!strptime(s.c_str(), "%Y-%m-%d %H:%M:%S", &t))
        throw std::runtime_error("datetime: bad format: " + s);
#endif
    TimePoint tp = tm_to_utc(t);
    if (tp == TimePoint(-1))
        throw std::runtime_error("datetime: could not convert: " + s);
    return tp;
}

// ---------------------------------------------------------------------------
// Floor to the nearest whole hour.
// ---------------------------------------------------------------------------
inline TimePoint floor_hour(TimePoint tp) {
    return (tp / 3600) * 3600;
}

// ---------------------------------------------------------------------------
// Format a TimePoint as "YYYY-MM-DD HH:MM:SS".
// ---------------------------------------------------------------------------
inline std::string to_string(TimePoint tp) {
    std::time_t t = static_cast<std::time_t>(tp);
    std::tm tm_buf{};
#ifdef _WIN32
    gmtime_s(&tm_buf, &t);
#else
    gmtime_r(&t, &tm_buf);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm_buf);
    return buf;
}

// ---------------------------------------------------------------------------
// Extract UTC hour (0-23), day-of-year (1-366), and year from a TimePoint.
// ---------------------------------------------------------------------------
inline void unpack(TimePoint tp, int& hour, int& doy, int& year) {
    std::time_t t = static_cast<std::time_t>(tp);
    std::tm tm_buf{};
#ifdef _WIN32
    gmtime_s(&tm_buf, &t);
#else
    gmtime_r(&t, &tm_buf);
#endif
    hour = tm_buf.tm_hour;
    doy  = tm_buf.tm_yday + 1;   // 1-indexed
    year = tm_buf.tm_year + 1900;
}

// ---------------------------------------------------------------------------
// Compute the four harmonic driver features used by ROPE:
//   t1 = sin(2π·hour/24),    t2 = cos(2π·hour/24)
//   t3 = sin(2π·doy/365.25), t4 = cos(2π·doy/365.25)
// Also returns the continuous doy = int_doy + hour/24.
// ---------------------------------------------------------------------------
inline void harmonics(TimePoint tp,
                       float& t1, float& t2, float& t3, float& t4,
                       float& cont_doy) {
    int hour, doy, year;
    unpack(tp, hour, doy, year);
    constexpr double TWO_PI = 2.0 * 3.14159265358979323846;
    t1 = static_cast<float>(std::sin(TWO_PI * hour / 24.0));
    t2 = static_cast<float>(std::cos(TWO_PI * hour / 24.0));
    t3 = static_cast<float>(std::sin(TWO_PI * doy  / 365.25));
    t4 = static_cast<float>(std::cos(TWO_PI * doy  / 365.25));
    cont_doy = static_cast<float>(doy + hour / 24.0);
}

} // namespace dt
} // namespace rope
