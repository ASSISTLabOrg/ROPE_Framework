#pragma once
// driver_window.hpp — space-weather database and hourly driver-window builder.

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "io.hpp"
#include "utils.hpp"

namespace rope {

// ---------------------------------------------------------------------------
// One row of derived driver features for a single hour.
// ---------------------------------------------------------------------------
struct DriverRow {
    dt::TimePoint tp;
    float f10, kp;
    float t1, t2, t3, t4;
    float doy;   // continuous  = int_doy + hour/24
    int   hour_int;
};

// ---------------------------------------------------------------------------
// SpaceWeatherDB — loads the celestrack CSV and provides O(log N) look-up.
// ---------------------------------------------------------------------------
class SpaceWeatherDB {
public:
    explicit SpaceWeatherDB(const std::string& csv_path) {
        CSVReader csv(csv_path);
        size_t N = csv.nrows();
        times_.reserve(N);
        f10_.reserve(N);
        kp_.reserve(N);
        doy_.reserve(N);
        hour_.reserve(N);

        for (size_t i = 0; i < N; ++i) {
            dt::TimePoint tp = dt::parse(csv.get("datetime", i));
            times_.push_back(tp);
            f10_.push_back(csv.get_float("f10", i));
            kp_.push_back(csv.get_float("kp",  i));
            // Prefer pre-computed doy/hour from CSV; re-compute if absent.
            if (csv.has_column("doy"))
                doy_.push_back(csv.get_float("doy", i));
            else
                doy_.push_back(static_cast<float>(
                    /* will be overwritten */ 0.0f));
            if (csv.has_column("hour"))
                hour_.push_back(csv.get_int("hour", i));
            else
                hour_.push_back(0);
        }

        // Ensure sorted by time (should already be, but be safe).
        if (!std::is_sorted(times_.begin(), times_.end())) {
            // Build index and sort.
            std::vector<size_t> idx(N);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(),
                      [&](size_t a, size_t b){ return times_[a] < times_[b]; });
            auto apply = [&](auto& v){
                auto tmp = v;
                for (size_t i = 0; i < N; ++i) v[i] = tmp[idx[i]];
            };
            apply(times_); apply(f10_); apply(kp_);
            apply(doy_);   apply(hour_);
        }

        // Fill any missing doy/hour from the TimePoint.
        for (size_t i = 0; i < N; ++i) {
            if (hour_[i] == 0 && doy_[i] == 0.0f) {
                int h, d, y;
                dt::unpack(times_[i], h, d, y);
                hour_[i] = h;
                doy_[i]  = static_cast<float>(d) + h / 24.0f;
            }
        }
    }

    // Look up a single TimePoint; throws if not found.
    // (The CSV must cover the entire requested range.)
    DriverRow lookup(dt::TimePoint tp) const {
        auto it = std::lower_bound(times_.begin(), times_.end(), tp);
        if (it == times_.end() || *it != tp)
            throw std::runtime_error(
                "SpaceWeatherDB: no entry for " + dt::to_string(tp));
        size_t idx = static_cast<size_t>(it - times_.begin());
        return make_row(idx);
    }

    dt::TimePoint time_min() const { return times_.front(); }
    dt::TimePoint time_max() const { return times_.back(); }

private:
    std::vector<dt::TimePoint> times_;
    std::vector<float>         f10_, kp_, doy_;
    std::vector<int>           hour_;

    DriverRow make_row(size_t idx) const {
        DriverRow r;
        r.tp       = times_[idx];
        r.f10      = f10_[idx];
        r.kp       = kp_[idx];
        r.hour_int = hour_[idx];
        r.doy      = doy_[idx];
        // Recompute harmonics from stored hour/doy for consistency.
        constexpr double TWO_PI = 2.0 * 3.14159265358979323846;
        double h = r.hour_int, d = static_cast<float>(static_cast<int>(r.doy));
        r.t1 = static_cast<float>(std::sin(TWO_PI * h / 24.0));
        r.t2 = static_cast<float>(std::cos(TWO_PI * h / 24.0));
        r.t3 = static_cast<float>(std::sin(TWO_PI * d / 365.25));
        r.t4 = static_cast<float>(std::cos(TWO_PI * d / 365.25));
        return r;
    }
};

// ---------------------------------------------------------------------------
// DriverWindowBuilder
//
// Builds the "internal" driver window that the ROPE pipeline needs:
//   - (seq_len-1) history rows  before start_datetime
//   - H            forecast rows starting at start_datetime
// Total rows: (seq_len-1) + H
// ---------------------------------------------------------------------------
class DriverWindowBuilder {
public:
    // Returns a flat vector of DriverRow objects in chronological order.
    // Throws if any hourly slot is missing in the database.
    static std::vector<DriverRow> build(
        const SpaceWeatherDB& db,
        const std::string&    start_datetime,
        int horizon,
        int seq_len
    ) {
        dt::TimePoint start = dt::parse(start_datetime);
        start = dt::floor_hour(start);

        int total = (seq_len - 1) + horizon;
        dt::TimePoint hist_start = start - static_cast<dt::TimePoint>(seq_len - 1) * 3600;

        std::vector<DriverRow> rows;
        rows.reserve(total);
        for (int i = 0; i < total; ++i) {
            dt::TimePoint tp = hist_start + static_cast<dt::TimePoint>(i) * 3600;
            rows.push_back(db.lookup(tp));
        }
        return rows;
    }
};

} // namespace rope
