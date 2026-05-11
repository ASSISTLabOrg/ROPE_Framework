#include "rope/io/ic_config.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>

namespace rope::io {

std::optional<IcConfig>
IcConfig::try_load(const std::filesystem::path& exported_dir) {
    auto path = exported_dir / "ic_config.json";
    if (!std::filesystem::exists(path)) return std::nullopt;

    std::ifstream f(path);
    if (!f)
        throw std::runtime_error(
            "IcConfig::try_load: cannot open " + path.string());

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(f);
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error(
            "IcConfig::try_load: malformed JSON in " +
            path.string() + ": " + e.what());
    }

    if (!j.contains("grid_axes") || !j["grid_axes"].is_array())
        throw std::runtime_error(
            "IcConfig::try_load: missing or invalid 'grid_axes' array in " +
            path.string());

    if (!j.contains("latent_dim") || !j["latent_dim"].is_number_integer())
        throw std::runtime_error(
            "IcConfig::try_load: missing or invalid 'latent_dim' in " +
            path.string());

    IcConfig ic;
    ic.version    = j.value("version", 1);
    ic.grid_axes  = j.at("grid_axes").get<std::vector<std::string>>();
    ic.latent_dim = j.at("latent_dim").get<int>();

    if (ic.latent_dim <= 0)
        throw std::runtime_error(
            "IcConfig::try_load: 'latent_dim' must be positive in " +
            path.string());

    return ic;
}

} // namespace rope::io
