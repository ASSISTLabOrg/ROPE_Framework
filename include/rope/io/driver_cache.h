#pragma once
#include <chrono>
#include <filesystem>
#include <string>
#include <unordered_map>

namespace rope::io {

struct DriverSource {
    std::string url;
    std::string description;
};

// Registry of known online sources, keyed by the "source" field in
// driver_config.json.  Edit driver_cache.cpp to add new entries.
const std::unordered_map<std::string, DriverSource>& known_sources();

// ---------------------------------------------------------------------------
// DriverCacheManager
// ---------------------------------------------------------------------------
// Maintains a local cache of driver .swbin files refreshed from online
// sources.  Integrated into Pipeline::load() so auto-refresh fires
// regardless of how the pipeline is invoked (CLI, C API, Python bindings).
//
// Download requires HTTP(S) support (cpp-httplib + OpenSSL — to be wired in
// once the convert_and_write() conversion stub is implemented).
// ---------------------------------------------------------------------------
class DriverCacheManager {
public:
    DriverCacheManager(std::filesystem::path cache_dir,
                       int max_age_hours = 24);

    // Returns path to a fresh .swbin file for the given source name.
    // Downloads and converts if stale; falls back to a stale file on failure
    // when one exists.  Throws if the source is unknown or the cache is
    // absent and download/conversion fails.
    std::filesystem::path get_path(const std::string& source);

private:
    bool is_stale(const std::filesystem::path& path) const;
    void refresh(const std::string& source, const std::filesystem::path& dest);

    // Download the raw CelesTrak CSV.  Returns the raw text content.
    // TODO: implement using cpp-httplib + OpenSSL.
    std::string download(const std::string& url);

    // Convert raw CelesTrak CSV → ROPE internal format → write as .swbin.
    //
    // Input columns:  DATE, F10.7_OBS, F10.7_OBS_CENTER81, AP_AVG, AP1..AP8
    // Output columns: datetime, f10, kp (hourly rows)
    //
    // TODO: implement Ap→Kp conversion and 3-hourly→hourly interpolation.
    void convert_and_write(const std::string& raw_csv,
                           const std::filesystem::path& dest);

    std::filesystem::path cache_dir_;
    std::chrono::seconds  max_age_;
};

} // namespace rope::io
