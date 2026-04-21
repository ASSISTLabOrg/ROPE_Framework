#pragma once
// Space-weather database and hourly driver-window builder.
//
// SpaceWeatherDB loads the Celestrak CSV and provides O(log N) look-up.
// DriverWindowBuilder assembles the full (seq_len-1 + H) row window needed
// by the base models.

#include "rope/core/datetime.h"
#include "rope/io/csv_reader.h"

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace rope::io {

// ---------------------------------------------------------------------------
// DriverRow — one hour of derived driver features.
// ---------------------------------------------------------------------------
struct DriverRow {
    TimePoint tp;
    float f10, kp;
    float t1, t2, t3, t4;  // sin/cos harmonic features
    float doy;              // continuous = int_doy + hour/24
    int   hour_int;
};

// ---------------------------------------------------------------------------
// SpaceWeatherDB
// ---------------------------------------------------------------------------
class SpaceWeatherDB {
public:
    explicit SpaceWeatherDB(const std::filesystem::path& csv_path);

    // Look up a single TimePoint; throws if not found.
    DriverRow lookup(TimePoint tp) const;

    TimePoint time_min() const { return times_.front(); }
    TimePoint time_max() const { return times_.back(); }

private:
    std::vector<TimePoint> times_;
    std::vector<float>     f10_, kp_, doy_;
    std::vector<int>       hour_;

    DriverRow make_row(std::size_t idx) const;
};

// ---------------------------------------------------------------------------
// DriverWindowBuilder
// ---------------------------------------------------------------------------
class DriverWindowBuilder {
public:
    // Returns (seq_len-1 + horizon) DriverRows in chronological order.
    // Throws if any hourly slot is missing from the database.
    static std::vector<DriverRow> build(const SpaceWeatherDB& db,
                                        std::string_view      start_iso,
                                        int horizon,
                                        int seq_len);
};

} // namespace rope::io
