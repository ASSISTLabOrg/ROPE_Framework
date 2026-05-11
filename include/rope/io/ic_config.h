#pragma once
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace rope::io {

struct IcConfig {
    int version = 1;
    std::vector<std::string> grid_axes;  // e.g. ["f10","kp"]
    int latent_dim = 10;                 // K

    // Returns {} if ic_config.json is absent from exported_dir.
    // Throws on malformed JSON or missing required fields.
    static std::optional<IcConfig>
    try_load(const std::filesystem::path& exported_dir);
};

} // namespace rope::io
