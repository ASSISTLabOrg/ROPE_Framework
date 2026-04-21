#include "rope/io/config_reader.h"

#include <fstream>
#include <stdexcept>

namespace rope::io {

ConfigReader::ConfigReader(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("ConfigReader: cannot open '" + path.string() + "'");

    std::string section, line;
    while (std::getline(f, line)) {
        // Strip comments
        auto comment = line.find('#');
        if (comment != std::string::npos) line = line.substr(0, comment);
        line = trim(line);
        if (line.empty()) continue;

        if (line.front() == '[' && line.back() == ']') {
            section = trim(line.substr(1, line.size() - 2));
            continue;
        }

        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key   = trim(line.substr(0, eq));
        std::string value = trim(line.substr(eq + 1));
        data_[section.empty() ? key : section + "." + key] = value;
    }
}

bool ConfigReader::has(std::string_view key) const noexcept {
    return data_.count(std::string{key}) > 0;
}

std::string ConfigReader::get(std::string_view key) const {
    auto it = data_.find(std::string{key});
    if (it == data_.end())
        throw std::runtime_error("ConfigReader: missing key '" + std::string{key} + "'");
    return it->second;
}

std::string ConfigReader::get(std::string_view key, std::string_view def) const {
    auto it = data_.find(std::string{key});
    return (it != data_.end()) ? it->second : std::string{def};
}

int ConfigReader::get_int(std::string_view key, int def) const {
    auto it = data_.find(std::string{key});
    return (it != data_.end()) ? std::stoi(it->second) : def;
}

double ConfigReader::get_double(std::string_view key, double def) const {
    auto it = data_.find(std::string{key});
    return (it != data_.end()) ? std::stod(it->second) : def;
}

std::string ConfigReader::trim(const std::string& s) {
    const std::string ws = " \t\r\n";
    auto a = s.find_first_not_of(ws);
    if (a == std::string::npos) return {};
    auto b = s.find_last_not_of(ws);
    return s.substr(a, b - a + 1);
}

} // namespace rope::io
