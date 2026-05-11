// rope — ROPE atmospheric density forecasting CLI.
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif
//
// Commands:
//   rope forecast --start <ISO8601> --horizon <h> [--config <path>]
//   rope get      --mode hold|interp --time <ISO8601> --lst <f> --lat <f> --alt <f>
//   rope get      --mode hold|interp --file <csv> [--output <path>]
//   rope exit
//
// Hidden internal command (spawned by 'rope forecast'):
//   rope --serve --socket-path <path> --config-path <path>

#include "rope/client/client.h"
#include "rope/core/datetime.h"
#include "rope/core/platform.h"
#include "rope/io/csv_reader.h"
#include "rope/io/driver_db.h"
#include "rope/io/driver_bin.h"
#include "rope/io/ic_table.h"
#include "rope/io/ic_bin.h"
#include "rope/server/server.h"

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using json   = nlohmann::json;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static fs::path exe_path() {
#ifdef __linux__
    return fs::canonical("/proc/self/exe");
#elif defined(_WIN32)
    char buf[MAX_PATH];
    GetModuleFileNameA(nullptr, buf, MAX_PATH);
    return fs::path{buf};
#else
    // macOS: use _NSGetExecutablePath or fall back to argv[0]
    return fs::current_path() / "rope";
#endif
}

static fs::path default_config(const fs::path& exe) {
    // <exe_dir>/../config/rope.conf
    return exe.parent_path().parent_path() / "config" / "rope.conf";
}

// Wait up to timeout_ms for the socket file to appear and accept a connection.
static bool wait_for_server(const fs::path& sock, int timeout_ms = 6000) {
    const int step_ms = 200;
    for (int elapsed = 0; elapsed < timeout_ms; elapsed += step_ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(step_ms));
        try {
            auto s = rope::platform::IpcSocket::connect(sock);
            return true;
        } catch (...) {}
    }
    return false;
}

// Connect to the server, spawning it first if needed.
// Returns a connected IpcClient.
static rope::client::IpcClient connect_or_spawn(
    const fs::path& socket_path, const fs::path& config_path,
    bool spawn_if_absent)
{
    try {
        return rope::client::IpcClient{socket_path};
    } catch (...) {}

    if (!spawn_if_absent) {
        std::cerr << "rope: server is not running. "
                     "Run 'rope forecast' first.\n";
        std::exit(1);
    }

    // Remove stale socket file if it exists but is unresponsive.
    if (fs::exists(socket_path))
        fs::remove(socket_path);

    rope::platform::spawn_server(exe_path(), socket_path, config_path);

    if (!wait_for_server(socket_path)) {
        std::cerr << "rope: timed out waiting for server to start\n";
        std::exit(1);
    }

    return rope::client::IpcClient{socket_path};
}

// ---------------------------------------------------------------------------
// Batch-file processing for 'rope get --file'
// ---------------------------------------------------------------------------

static int run_batch_get(rope::client::IpcClient& client,
                          const std::string& mode,
                          const fs::path& file_path,
                          const fs::path& output_path) {
    rope::io::CsvReader csv{file_path};
    std::size_t N = csv.nrows();

    std::vector<rope::client::BatchPoint> pts;
    pts.reserve(N);

    for (std::size_t i = 0; i < N; ++i) {
        int yr  = csv.get_int("YYYY", i);
        int mo  = csv.get_int("MM",   i);
        int dy  = csv.get_int("DD",   i);
        int hr  = csv.get_int("HH",   i);
        int mn  = csv.get_int("MIN",  i);
        int sc  = csv.get_int("SS",   i);

        char buf[24];
        std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d",
                      yr, mo, dy, hr, mn, sc);

        pts.push_back({
            std::string{buf},
            csv.get_float("lst",    i),
            csv.get_float("lat",    i),
            csv.get_float("alt_km", i)
        });
    }

    auto results = client.batch_get(mode, pts);

    json out = json::array();
    for (std::size_t i = 0; i < results.size(); ++i) {
        out.push_back({
            {"time",        pts[i].time_iso},
            {"lst",         pts[i].lst},
            {"lat",         pts[i].lat},
            {"alt_km",      pts[i].alt_km},
            {"density",     results[i].density},
            {"uncertainty", results[i].uncertainty}
        });
    }

    std::string text = out.dump(2);
    if (output_path.empty()) {
        std::cout << text << "\n";
    } else {
        std::ofstream f{output_path};
        if (!f) {
            std::cerr << "rope: cannot open output file: " << output_path << "\n";
            return 1;
        }
        f << text << "\n";
    }
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    CLI::App app{"ROPE atmospheric density forecasting service", "rope"};
    app.require_subcommand(0, 1);

    // ---- hidden --serve mode ----
    bool        do_serve    = false;
    std::string serve_sock, serve_conf;
    app.add_flag("--serve", do_serve)->group("");
    app.add_option("--socket-path", serve_sock)->group("");
    app.add_option("--config-path", serve_conf)->group("");

    // ---- hidden --socket override (testing / multi-instance use) ----
    std::string cli_socket;
    app.add_option("--socket", cli_socket, "Override the server socket path")->group("");

    // ---- forecast subcommand ----
    auto* fc = app.add_subcommand("forecast",
                                   "Run a forecast and cache the grid in the server");
    std::string fc_start, fc_config;
    int         fc_horizon = 0;
    fc->add_option("--start",   fc_start,   "Forecast start time (ISO 8601, UTC)")
      ->required();
    fc->add_option("--horizon", fc_horizon, "Forecast duration in hours")
      ->required();
    fc->add_option("--config",  fc_config,  "Config file path");

    // ---- get subcommand ----
    auto* gc = app.add_subcommand("get",
                                   "Query the cached forecast grid");
    std::string gc_mode, gc_time, gc_file, gc_output;
    double      gc_lst = 0, gc_lat = 0, gc_alt = 0;
    gc->add_option("--mode", gc_mode, "hold or interp")->required();
    gc->add_option("--time", gc_time, "Query time (ISO 8601, UTC)");
    gc->add_option("--lst",  gc_lst,  "Local Solar Time [hours]");
    gc->add_option("--lat",  gc_lat,  "Geodetic latitude [degrees]");
    gc->add_option("--alt",  gc_alt,  "Altitude [km]");
    gc->add_option("--file", gc_file, "Batch CSV input (replaces point flags)");
    gc->add_option("--output", gc_output, "Output file for --file results");

    // ---- convert-sw subcommand ----
    auto* sw = app.add_subcommand("convert-sw",
                                   "Convert a space-weather CSV to .swbin binary format");
    std::string sw_input, sw_output;
    sw->add_option("--input",  sw_input,  "Input CSV file (datetime,f10,kp,...)")
      ->required();
    sw->add_option("--output", sw_output, "Output .swbin file")
      ->required();

    // ---- convert-ic subcommand ----
    auto* ic = app.add_subcommand("convert-ic",
                                   "Convert an IC-table CSV to .icbin binary format");
    std::string ic_input, ic_output;
    ic->add_option("--input",  ic_input,  "Input CSV file (F10,Kp,y1,...,yK)")
      ->required();
    ic->add_option("--output", ic_output, "Output .icbin file")
      ->required();

    // ---- exit subcommand ----
    auto* ec = app.add_subcommand("exit", "Terminate the server cleanly");

    CLI11_PARSE(app, argc, argv);

    // ---- --serve path ----
    if (do_serve) {
        if (serve_sock.empty() || serve_conf.empty()) {
            std::cerr << "rope --serve: --socket-path and --config-path are required\n";
            return 1;
        }
        try {
            rope::server::run(fs::path{serve_sock}, fs::path{serve_conf});
        } catch (const std::exception& e) {
            std::cerr << "rope --serve: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }

    // ---- Determine socket and config paths ----
    fs::path socket_path = cli_socket.empty()
        ? rope::platform::default_socket_path()
        : fs::path{cli_socket};
    fs::path config_path;

    if (fc->parsed()) {
        config_path = fc_config.empty()
            ? default_config(exe_path())
            : fs::path{fc_config};
    } else {
        config_path = default_config(exe_path());
    }

    // ---- forecast ----
    if (fc->parsed()) {
        try {
            auto client = connect_or_spawn(socket_path, config_path, /*spawn=*/true);
            auto res    = client.forecast(fc_start, fc_horizon);
            std::cout << json{
                {"status",       "ok"},
                {"window_start", res.window_start},
                {"window_end",   res.window_end}
            }.dump() << "\n";
        } catch (const std::exception& e) {
            std::cerr << "rope forecast: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }

    // ---- get ----
    if (gc->parsed()) {
        try {
            auto client = connect_or_spawn(socket_path, config_path, /*spawn=*/false);

            if (!gc_file.empty()) {
                return run_batch_get(client, gc_mode,
                                     fs::path{gc_file},
                                     fs::path{gc_output});
            }

            // Single-point query
            if (gc_time.empty()) {
                std::cerr << "rope get: --time is required when --file is not given\n";
                return 1;
            }
            auto res = client.get(gc_mode, gc_time, gc_lst, gc_lat, gc_alt);
            std::cout << json{
                {"status",      "ok"},
                {"density",     res.density},
                {"uncertainty", res.uncertainty}
            }.dump() << "\n";
        } catch (const std::exception& e) {
            std::cerr << "rope get: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }

    // ---- exit ----
    if (ec->parsed()) {
        try {
            rope::client::IpcClient client{socket_path};
            client.exit_server();
        } catch (...) {
            // If nothing is running, exit is a no-op success.
        }
        return 0;
    }

    // ---- convert-sw ----
    if (sw->parsed()) {
        try {
            std::cout << "Loading " << sw_input << "…\n";
            auto db = rope::io::SpaceWeatherDB::from_file(fs::path{sw_input});
            std::cout << "  " << db.size() << " rows  ["
                      << rope::format_iso(db.time_min()) << " → "
                      << rope::format_iso(db.time_max()) << "]\n";
            std::cout << "Writing " << sw_output << "…\n";
            rope::io::SpaceWeatherBin::save(db, fs::path{sw_output});
            std::cout << "Done.\n";
        } catch (const std::exception& e) {
            std::cerr << "rope convert-sw: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }

    // ---- convert-ic ----
    if (ic->parsed()) {
        try {
            std::cout << "Loading " << ic_input << "…\n";
            auto table = rope::io::ICTable::from_file(fs::path{ic_input});
            std::cout << "  latent_dim=" << table.latent_dim() << "\n";
            std::cout << "Writing " << ic_output << "…\n";
            rope::io::IcBin::save(table, fs::path{ic_output});
            std::cout << "Done.\n";
        } catch (const std::exception& e) {
            std::cerr << "rope convert-ic: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }

    std::cout << app.help();
    return 0;
}
