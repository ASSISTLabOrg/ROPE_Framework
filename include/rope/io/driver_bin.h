#pragma once
// Binary space-weather format (.swbin).
//
// Header (16 bytes, little-endian):
//   uint32  magic    = 0x52505357  ("RPSW")
//   uint32  version  = 1
//   uint32  nrows
//   uint32  reserved = 0
//
// Records (nrows × 24 bytes, little-endian):
//   int64   tp        -- Unix timestamp (seconds since epoch)
//   float32 f10
//   float32 kp
//   float32 doy
//   int32   hour_int

#include "rope/io/driver_db.h"

#include <filesystem>

namespace rope::io {

class SpaceWeatherBin {
public:
    // Load a .swbin file and return a SpaceWeatherDB ready for lookup.
    // Throws on bad magic, unsupported version, or read failure.
    static SpaceWeatherDB load(const std::filesystem::path& bin_path);

    // Serialise an existing SpaceWeatherDB to .swbin.
    static void save(const SpaceWeatherDB& db,
                     const std::filesystem::path& bin_path);
};

} // namespace rope::io
