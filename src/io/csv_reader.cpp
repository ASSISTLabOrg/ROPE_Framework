#include "rope/io/csv_reader.h"

#include <fstream>
#include <stdexcept>
#include <string>

namespace rope::io {

CsvReader::CsvReader(const std::filesystem::path& path, char delim) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("CsvReader: cannot open " + path.string());

    std::string line;

    // Header row
    if (!std::getline(f, line))
        throw std::runtime_error("CsvReader: empty file " + path.string());
    headers_ = split(strip(line), delim);
    for (std::size_t i = 0; i < headers_.size(); ++i)
        index_[strip(headers_[i])] = i;
    columns_.resize(headers_.size());

    // Data rows
    while (std::getline(f, line)) {
        line = strip(line);
        if (line.empty()) continue;
        auto fields = split(line, delim);
        for (std::size_t i = 0; i < columns_.size(); ++i)
            columns_[i].push_back(i < fields.size() ? strip(fields[i]) : std::string{});
    }
    n_rows_ = columns_.empty() ? 0 : columns_[0].size();
}

bool CsvReader::has_column(std::string_view name) const noexcept {
    return index_.count(std::string{name}) > 0;
}

const std::string& CsvReader::get(std::string_view col, std::size_t row) const {
    return columns_.at(col_index(col)).at(row);
}

float CsvReader::get_float(std::string_view col, std::size_t row) const {
    return std::stof(get(col, row));
}

int CsvReader::get_int(std::string_view col, std::size_t row) const {
    return std::stoi(get(col, row));
}

std::size_t CsvReader::col_index(std::string_view name) const {
    auto it = index_.find(std::string{name});
    if (it == index_.end())
        throw std::runtime_error("CsvReader: unknown column '" + std::string{name} + "'");
    return it->second;
}

std::string CsvReader::strip(const std::string& s) {
    std::size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    std::size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

std::vector<std::string> CsvReader::split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::string field;
    bool in_quotes = false;
    for (char c : s) {
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == delim && !in_quotes) {
            out.push_back(field);
            field.clear();
        } else {
            field += c;
        }
    }
    out.push_back(field);
    return out;
}

} // namespace rope::io
