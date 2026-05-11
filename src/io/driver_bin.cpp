#include "rope/io/driver_bin.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace rope::io {

static constexpr std::uint32_t SW_MAGIC   = 0x52505357u;  // "RPSW"
static constexpr std::uint32_t SW_VERSION = 1u;

SpaceWeatherDB SpaceWeatherBin::load(const std::filesystem::path& bin_path) {
    std::ifstream f(bin_path, std::ios::binary);
    if (!f)
        throw std::runtime_error(
            "SpaceWeatherBin::load: cannot open " + bin_path.string());

    std::uint32_t magic, version, nrows, reserved;
    f.read(reinterpret_cast<char*>(&magic),    4);
    f.read(reinterpret_cast<char*>(&version),  4);
    f.read(reinterpret_cast<char*>(&nrows),    4);
    f.read(reinterpret_cast<char*>(&reserved), 4);

    if (!f)
        throw std::runtime_error(
            "SpaceWeatherBin::load: failed to read header from " +
            bin_path.string());
    if (magic != SW_MAGIC)
        throw std::runtime_error(
            "SpaceWeatherBin::load: bad magic in " + bin_path.string() +
            " (not a .swbin file)");
    if (version != SW_VERSION)
        throw std::runtime_error(
            "SpaceWeatherBin::load: unsupported version " +
            std::to_string(version) + " in " + bin_path.string());

    std::vector<TimePoint>  times(nrows);
    std::vector<float>      f10(nrows), kp(nrows), doy(nrows);
    std::vector<int>        hour(nrows);

    for (std::uint32_t i = 0; i < nrows; ++i) {
        std::int64_t  tp_raw;
        float         f10v, kpv, doyv;
        std::int32_t  hourv;

        f.read(reinterpret_cast<char*>(&tp_raw), 8);
        f.read(reinterpret_cast<char*>(&f10v),   4);
        f.read(reinterpret_cast<char*>(&kpv),    4);
        f.read(reinterpret_cast<char*>(&doyv),   4);
        f.read(reinterpret_cast<char*>(&hourv),  4);

        if (!f)
            throw std::runtime_error(
                "SpaceWeatherBin::load: unexpected EOF at record " +
                std::to_string(i) + " in " + bin_path.string());

        times[i] = static_cast<TimePoint>(tp_raw);
        f10[i]   = f10v;
        kp[i]    = kpv;
        doy[i]   = doyv;
        hour[i]  = static_cast<int>(hourv);
    }

    return SpaceWeatherDB{
        std::move(times),
        std::move(f10),
        std::move(kp),
        std::move(doy),
        std::move(hour)
    };
}

void SpaceWeatherBin::save(const SpaceWeatherDB& db,
                           const std::filesystem::path& bin_path) {
    std::ofstream f(bin_path, std::ios::binary | std::ios::trunc);
    if (!f)
        throw std::runtime_error(
            "SpaceWeatherBin::save: cannot open " + bin_path.string());

    const auto nrows = static_cast<std::uint32_t>(db.times_.size());
    const std::uint32_t reserved = 0u;

    f.write(reinterpret_cast<const char*>(&SW_MAGIC),   4);
    f.write(reinterpret_cast<const char*>(&SW_VERSION), 4);
    f.write(reinterpret_cast<const char*>(&nrows),      4);
    f.write(reinterpret_cast<const char*>(&reserved),   4);

    for (std::uint32_t i = 0; i < nrows; ++i) {
        auto         tp_raw = static_cast<std::int64_t>(db.times_[i]);
        float        f10v   = db.f10_[i];
        float        kpv    = db.kp_[i];
        float        doyv   = db.doy_[i];
        std::int32_t hourv  = static_cast<std::int32_t>(db.hour_[i]);

        f.write(reinterpret_cast<const char*>(&tp_raw), 8);
        f.write(reinterpret_cast<const char*>(&f10v),   4);
        f.write(reinterpret_cast<const char*>(&kpv),    4);
        f.write(reinterpret_cast<const char*>(&doyv),   4);
        f.write(reinterpret_cast<const char*>(&hourv),  4);
    }

    if (!f)
        throw std::runtime_error(
            "SpaceWeatherBin::save: write failed for " + bin_path.string());
}

} // namespace rope::io
