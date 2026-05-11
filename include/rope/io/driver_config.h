#pragma once
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace rope::io {

struct DriverConfig {
    int version = 1;
    std::vector<std::string> columns;  // e.g. ["f10","kp","t1","t2","t3","t4"]
    std::string source;                // e.g. "celestrak_sw"

    // Returns {} if driver_config.json is absent from exported_dir.
    // Throws on malformed JSON or missing required fields.
    static std::optional<DriverConfig>
    try_load(const std::filesystem::path& exported_dir);
};

} // namespace rope::io
