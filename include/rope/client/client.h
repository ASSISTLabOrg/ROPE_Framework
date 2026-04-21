#pragma once
// IpcClient — typed request/response API over the ROPE socket protocol.
//
// Owns one socket connection for its lifetime.  Not thread-safe; callers
// that share a client must synchronise externally.
//
// Wire format: 32-bit little-endian length prefix followed by UTF-8 JSON.
// JSON serialisation details are implementation-private.

#include "rope/core/platform.h"
#include "rope/core/types.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace rope::client {

// ---------------------------------------------------------------------------
// Typed response types
// ---------------------------------------------------------------------------

struct ForecastResult {
    std::string window_start;  // ISO 8601
    std::string window_end;    // ISO 8601
};

struct QueryResult {
    double density;      // kg/m³
    double uncertainty;  // kg/m³
};

struct BatchPoint {
    std::string time_iso;  // ISO 8601
    double lst, lat, alt_km;
};

// ---------------------------------------------------------------------------
// IpcClient
// ---------------------------------------------------------------------------
class IpcClient {
public:
    // Connect to the server at socket_path.  Throws on failure.
    explicit IpcClient(const std::filesystem::path& socket_path);
    ~IpcClient();

    ForecastResult           forecast(const std::string& start_iso, int horizon);

    QueryResult              get(const std::string& mode,
                                 const std::string& time_iso,
                                 double lst, double lat, double alt_km);

    std::vector<QueryResult> batch_get(const std::string& mode,
                                       const std::vector<BatchPoint>& points);

    ForecastGrid             fetch_grid();

    void                     exit_server();

private:
    // Pimpl hides nlohmann/json from the public header.
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rope::client
