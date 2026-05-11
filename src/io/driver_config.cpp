#include "rope/io/driver_config.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>

namespace rope::io {

std::optional<DriverConfig>
DriverConfig::try_load(const std::filesystem::path& exported_dir) {
    auto path = exported_dir / "driver_config.json";
    if (!std::filesystem::exists(path)) return std::nullopt;

    std::ifstream f(path);
    if (!f)
        throw std::runtime_error(
            "DriverConfig::try_load: cannot open " + path.string());

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(f);
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error(
            "DriverConfig::try_load: malformed JSON in " +
            path.string() + ": " + e.what());
    }

    if (!j.contains("columns") || !j["columns"].is_array())
        throw std::runtime_error(
            "DriverConfig::try_load: missing or invalid 'columns' array in " +
            path.string());

    DriverConfig dc;
    dc.version = j.value("version", 1);
    dc.columns = j.at("columns").get<std::vector<std::string>>();
    dc.source  = j.value("source", "");

    if (dc.columns.empty())
        throw std::runtime_error(
            "DriverConfig::try_load: 'columns' must not be empty in " +
            path.string());

    return dc;
}

} // namespace rope::io
