#pragma once
// Binary IC-table format (.icbin).
//
// Header (20 bytes, little-endian):
//   uint32  magic      = 0x52504943  ("RPIC")
//   uint32  version    = 1
//   uint32  nrows
//   uint32  latent_dim  (K)
//   uint32  reserved   = 0
//
// Records (nrows × (2 + K) × 4 bytes, little-endian):
//   float32 f10
//   float32 kp
//   float32 y[K]

#include "rope/io/ic_table.h"

#include <filesystem>

namespace rope::io {

class IcBin {
public:
    // Load a .icbin file and return an ICTable ready for interpolation.
    // Throws on bad magic, unsupported version, or read failure.
    static ICTable load(const std::filesystem::path& bin_path);

    // Serialise an existing ICTable to .icbin.
    static void save(const ICTable& table,
                     const std::filesystem::path& bin_path);
};

} // namespace rope::io
