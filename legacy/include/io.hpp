#pragma once
// io.hpp — file I/O utilities: CSV reader and INI-style config reader.

// ---------------------------------------------------------------------------
// CSVReader — lightweight column-oriented CSV parser.
//
// Reads the entire file on construction; columns are accessed by name.
// Handles quoted fields and optional trailing whitespace.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace rope {

class CSVReader {
public:
    explicit CSVReader(const std::string& path, char delim = ',') {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("CSVReader: cannot open " + path);

        std::string line;

        // Header
        if (!std::getline(f, line))
            throw std::runtime_error("CSVReader: empty file " + path);
        headers_ = split(strip(line), delim);
        for (size_t i = 0; i < headers_.size(); ++i)
            index_[strip(headers_[i])] = i;
        columns_.resize(headers_.size());

        // Data rows
        while (std::getline(f, line)) {
            line = strip(line);
            if (line.empty()) continue;
            auto fields = split(line, delim);
            for (size_t i = 0; i < columns_.size(); ++i) {
                if (i < fields.size())
                    columns_[i].push_back(strip(fields[i]));
                else
                    columns_[i].push_back("");
            }
        }
        n_rows_ = columns_.empty() ? 0 : columns_[0].size();
    }

    size_t nrows() const { return n_rows_; }

    bool has_column(const std::string& name) const {
        return index_.count(name) > 0;
    }

    const std::string& get(const std::string& col, size_t row) const {
        return columns_.at(col_index(col)).at(row);
    }

    float get_float(const std::string& col, size_t row) const {
        return std::stof(get(col, row));
    }
    int get_int(const std::string& col, size_t row) const {
        return std::stoi(get(col, row));
    }

private:
    std::vector<std::string>                 headers_;
    std::vector<std::vector<std::string>>    columns_;
    std::unordered_map<std::string, size_t>  index_;
    size_t n_rows_ = 0;

    size_t col_index(const std::string& name) const {
        auto it = index_.find(name);
        if (it == index_.end())
            throw std::runtime_error("CSVReader: unknown column '" + name + "'");
        return it->second;
    }

    static std::string strip(const std::string& s) {
        size_t b = s.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) return "";
        size_t e = s.find_last_not_of(" \t\r\n");
        return s.substr(b, e - b + 1);
    }

    static std::vector<std::string> split(const std::string& s, char delim) {
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
};

} // namespace rope

// ---------------------------------------------------------------------------
// ConfigReader — lightweight INI-style configuration file parser.
//
// Format:
//   [section]       # section header
//   key = value     # leading/trailing whitespace stripped; # starts a comment
//
// Keys are accessed as "section.key".
// ---------------------------------------------------------------------------

class ConfigReader {
public:
    explicit ConfigReader(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("ConfigReader: cannot open '" + path + "'");

        std::string section, line;
        while (std::getline(f, line)) {
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

    std::string get(const std::string& key) const {
        auto it = data_.find(key);
        if (it == data_.end())
            throw std::runtime_error("ConfigReader: missing key '" + key + "'");
        return it->second;
    }
    std::string get(const std::string& key, const std::string& def) const {
        auto it = data_.find(key);
        return (it != data_.end()) ? it->second : def;
    }

    int get_int(const std::string& key, int def = 0) const {
        auto it = data_.find(key);
        return (it != data_.end()) ? std::stoi(it->second) : def;
    }
    double get_double(const std::string& key, double def = 0.0) const {
        auto it = data_.find(key);
        return (it != data_.end()) ? std::stod(it->second) : def;
    }

    bool has(const std::string& key) const {
        return data_.count(key) > 0;
    }

private:
    std::map<std::string, std::string> data_;

    static std::string trim(const std::string& s) {
        const std::string ws = " \t\r\n";
        size_t a = s.find_first_not_of(ws);
        if (a == std::string::npos) return {};
        size_t b = s.find_last_not_of(ws);
        return s.substr(a, b - a + 1);
    }
};
