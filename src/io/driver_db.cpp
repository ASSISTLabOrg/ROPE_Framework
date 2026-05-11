#include "rope/io/driver_db.h"
#include "rope/io/driver_bin.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rope::io {

// ---------------------------------------------------------------------------
// from_file() — dispatch on extension
// ---------------------------------------------------------------------------
SpaceWeatherDB SpaceWeatherDB::from_file(const std::filesystem::path& path) {
    if (path.extension() == ".swbin")
        return SpaceWeatherBin::load(path);
    return SpaceWeatherDB{path};
}

// ---------------------------------------------------------------------------
// Private constructor (used by SpaceWeatherBin::load)
// ---------------------------------------------------------------------------
SpaceWeatherDB::SpaceWeatherDB(std::vector<TimePoint> times,
                               std::vector<float>     f10,
                               std::vector<float>     kp,
                               std::vector<float>     doy,
                               std::vector<int>       hour)
    : times_(std::move(times))
    , f10_(std::move(f10))
    , kp_(std::move(kp))
    , doy_(std::move(doy))
    , hour_(std::move(hour))
{}

// ---------------------------------------------------------------------------
// CSV constructor
// ---------------------------------------------------------------------------
SpaceWeatherDB::SpaceWeatherDB(const std::filesystem::path& csv_path) {
    CsvReader csv(csv_path);
    const std::size_t N = csv.nrows();

    times_.reserve(N);
    f10_.reserve(N);
    kp_.reserve(N);
    doy_.reserve(N);
    hour_.reserve(N);

    for (std::size_t i = 0; i < N; ++i) {
        times_.push_back(parse_datetime(csv.get("datetime", i)));
        f10_.push_back(csv.get_float("f10", i));
        kp_.push_back(csv.get_float("kp",  i));
        doy_.push_back(csv.has_column("doy")  ? csv.get_float("doy",  i) : 0.0f);
        hour_.push_back(csv.has_column("hour") ? csv.get_int("hour", i)  : 0);
    }

    // Ensure sorted by time.
    if (!std::is_sorted(times_.begin(), times_.end())) {
        std::vector<std::size_t> idx(N);
        std::iota(idx.begin(), idx.end(), std::size_t{0});
        std::sort(idx.begin(), idx.end(),
                  [&](std::size_t a, std::size_t b){ return times_[a] < times_[b]; });
        auto reorder = [&](auto& v){
            auto tmp = v;
            for (std::size_t i = 0; i < N; ++i) v[i] = tmp[idx[i]];
        };
        reorder(times_); reorder(f10_); reorder(kp_);
        reorder(doy_);   reorder(hour_);
    }

    // Fill missing doy/hour from the TimePoint.
    for (std::size_t i = 0; i < N; ++i) {
        if (hour_[i] == 0 && doy_[i] == 0.0f) {
            int h, doy, yr;
            unpack(times_[i], h, doy, yr);
            hour_[i] = h;
            doy_[i]  = static_cast<float>(doy) + h / 24.0f;
        }
    }
}

DriverRow SpaceWeatherDB::lookup(TimePoint tp) const {
    auto it = std::lower_bound(times_.begin(), times_.end(), tp);
    if (it == times_.end() || *it != tp)
        throw std::runtime_error(
            "SpaceWeatherDB: no entry for " + format_iso(tp));
    return make_row(static_cast<std::size_t>(it - times_.begin()));
}

DriverRow SpaceWeatherDB::make_row(std::size_t idx) const {
    DriverRow r;
    r.tp       = times_[idx];
    r.f10      = f10_[idx];
    r.kp       = kp_[idx];
    r.hour_int = hour_[idx];
    r.doy      = doy_[idx];
    float cont_doy;
    harmonics(r.tp, r.t1, r.t2, r.t3, r.t4, cont_doy);
    return r;
}

std::vector<DriverRow> DriverWindowBuilder::build(const SpaceWeatherDB& db,
                                                   std::string_view      start_iso,
                                                   int horizon,
                                                   int seq_len) {
    TimePoint start      = floor_hour(parse_datetime(start_iso));
    int total            = (seq_len - 1) + horizon;
    TimePoint hist_start = start - static_cast<TimePoint>(seq_len - 1) * 3600;

    std::vector<DriverRow> rows;
    rows.reserve(static_cast<std::size_t>(total));
    for (int i = 0; i < total; ++i)
        rows.push_back(db.lookup(hist_start + static_cast<TimePoint>(i) * 3600));
    return rows;
}

} // namespace rope::io
