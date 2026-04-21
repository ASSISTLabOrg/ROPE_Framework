// driver.cpp — ROPE entry point.
//
// Usage:
//   rope_demo [config_file]
//
// config_file defaults to "<exe_dir>/../config/rope.conf".
// Relative paths inside the config file are resolved relative to the config
// file's own directory.
//
// Session mode is set via [session] mode in the config file:
//   mode = repl     interactive REPL (commands listed below)
//   mode = oneshot  run a fixed demo forecast and print stats, then exit
//   mode = socket   Unix-domain socket server; run the forecast once and serve
//                   repeated density queries from a connected program
//
// REPL commands:
//   run   <YYYY-MM-DD> <HH:MM:SS> <horizon_h>
//   stats
//   query hold   <YYYY-MM-DD> <HH:MM:SS> <lst> <lat> <alt_km>
//   query interp <YYYY-MM-DD> <HH:MM:SS> <lst> <lat> <alt_km>
//   help
//   quit / exit
//
// Socket protocol (one request / response per line):
//   hold   <YYYY-MM-DD> <HH:MM:SS> <lst> <lat> <alt_km>
//   interp <YYYY-MM-DD> <HH:MM:SS> <lst> <lat> <alt_km>
//   ping
//   quit
//
//   ok <density_kg_m3>
//   pong
//   bye
//   err <type>: <message>
//     types: time_out_of_range | spatial_out_of_range | bad_request | internal

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <csignal>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "io.hpp"
#include "interpolator.hpp"
#include "rope.hpp"

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Graceful shutdown support — set by signal handler, checked by accept loop.
// ---------------------------------------------------------------------------
static std::atomic<bool> g_shutdown{false};
static std::string       g_sock_path;   // set before installing handler

static void on_shutdown_signal(int) {
    g_shutdown = true;
    if (!g_sock_path.empty())
        ::unlink(g_sock_path.c_str());
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

static fs::path exe_dir() {
#ifdef __linux__
    return fs::canonical("/proc/self/exe").parent_path();
#else
    return fs::current_path();
#endif
}

static std::string find_config(const char* arg) {
    if (arg) {
        if (fs::exists(arg)) return arg;
        throw std::runtime_error(std::string("Config file not found: ") + arg);
    }
    for (auto& p : {
        fs::current_path() / "config" / "rope.conf",
        exe_dir()          / "config" / "rope.conf",
        exe_dir()          / ".."     / "config" / "rope.conf",
    })
        if (fs::exists(p)) return fs::canonical(p).string();
    throw std::runtime_error(
        "rope.conf not found.  Pass the path as the first argument.");
}

static std::string resolve_path(const std::string& raw,
                                 const fs::path&    config_dir) {
    fs::path p(raw);
    return fs::weakly_canonical(p.is_absolute() ? p : config_dir / p).string();
}

// ---------------------------------------------------------------------------
// ForecastState — owns both ForecastResult and DensityInterpolator together.
//
// DensityInterpolator holds a const& into ForecastResult::meta_density, so
// the two must share a lifetime.  Wrapping them in one heap-allocated struct
// and distributing it via shared_ptr lets multiple client threads hold onto
// the state safely.
// ---------------------------------------------------------------------------
struct ForecastState {
    rope::ForecastResult     result;
    rope::DensityInterpolator interp;

    explicit ForecastState(rope::ForecastResult r)
        : result(std::move(r)), interp(result) {}
};

// ---------------------------------------------------------------------------
// Tokeniser
// ---------------------------------------------------------------------------

static std::vector<std::string> tokenize(const std::string& line) {
    std::istringstream iss(line);
    std::vector<std::string> tok;
    std::string t;
    while (iss >> t) tok.push_back(t);
    return tok;
}

// Datetimes are stored as "YYYY-MM-DD HH:MM:SS", so the date and time parts
// arrive as two separate tokens.  Combine them here.
static std::string parse_datetime(const std::vector<std::string>& tok,
                                   size_t idx) {
    if (idx + 1 >= tok.size())
        throw std::invalid_argument(
            "Expected <YYYY-MM-DD> <HH:MM:SS> at token " + std::to_string(idx));
    return tok[idx] + " " + tok[idx + 1];
}

// ---------------------------------------------------------------------------
// Session state
// ---------------------------------------------------------------------------

struct Session {
    const rope::ROPE&                          rope;
    std::optional<rope::ForecastResult>        result;
    std::unique_ptr<rope::DensityInterpolator> interp;
};

// ---------------------------------------------------------------------------
// Command: help
// ---------------------------------------------------------------------------

static void cmd_help() {
    std::cout <<
        "Commands:\n"
        "  run   <YYYY-MM-DD> <HH:MM:SS> <horizon_h>\n"
        "      Run a forecast and store the result.\n"
        "  stats\n"
        "      Print density stats and the first 5 rows of the last result.\n"
        "  query hold   <YYYY-MM-DD> <HH:MM:SS> <lst> <lat> <alt_km>\n"
        "      Snap to the next model hour, then interpolate in space.\n"
        "  query interp <YYYY-MM-DD> <HH:MM:SS> <lst> <lat> <alt_km>\n"
        "      Trilinear spatial interpolation + linear time blend.\n"
        "  query file hold|interp <in_path> [out_path]\n"
        "      Batch query from a CSV file; results written to out_path (or stdout).\n"
        "      Required columns: YYYY, MM, DD, HH, MIN, SS, lst, lat, alt_km\n"
        "  help         Show this message.\n"
        "  quit / exit  Exit the session.\n"
        "\n"
        "  <lst>    Local Solar Time  [0, 24) hours\n"
        "  <lat>    Geodetic latitude [-87.5, 87.5] degrees\n"
        "  <alt_km> Altitude          [100, 980] km\n";
}

// ---------------------------------------------------------------------------
// Command: run
// ---------------------------------------------------------------------------

static void cmd_run(Session& s, const std::vector<std::string>& tok) {
    // run <YYYY-MM-DD> <HH:MM:SS> <horizon>
    if (tok.size() < 4) {
        std::cerr << "Usage: run <YYYY-MM-DD> <HH:MM:SS> <horizon_h>\n";
        return;
    }
    std::string start;
    int H;
    try {
        start = parse_datetime(tok, 1);
        H     = std::stoi(tok[3]);
    } catch (const std::exception& e) {
        std::cerr << "Bad arguments: " << e.what() << "\n";
        return;
    }

    auto t0 = std::chrono::steady_clock::now();
    s.result = s.rope.run(start, H);
    auto t1 = std::chrono::steady_clock::now();

    s.interp = std::make_unique<rope::DensityInterpolator>(*s.result);

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const auto& d = s.result->meta_density;
    float  mn   = *std::min_element(d.begin(), d.end());
    float  mx   = *std::max_element(d.begin(), d.end());
    double sum  = std::accumulate(d.begin(), d.end(), 0.0);
    float  mean = static_cast<float>(sum / d.size());

    std::cout << "Run time  : " << ms << " ms\n"
              << std::scientific << std::setprecision(3)
              << "Density   : min=" << mn << "  max=" << mx
              << "  mean=" << mean << "\n"
              << std::defaultfloat;
}

// ---------------------------------------------------------------------------
// Command: stats
// ---------------------------------------------------------------------------

static void cmd_stats(const Session& s) {
    if (!s.result) {
        std::cerr << "No result yet — run a forecast first.\n";
        return;
    }
    const auto& r = *s.result;
    const auto& d = r.meta_density;
    float  mn   = *std::min_element(d.begin(), d.end());
    float  mx   = *std::max_element(d.begin(), d.end());
    double sum  = std::accumulate(d.begin(), d.end(), 0.0);
    float  mean = static_cast<float>(sum / d.size());

    std::cout << std::scientific << std::setprecision(3)
              << "Density stats : min=" << mn << "  max=" << mx
              << "  mean=" << mean << "  nelem=" << d.size() << "\n"
              << std::defaultfloat
              << "Forecast window (first 5 rows):\n"
              << "  datetime              f10      kp\n";
    for (int t = 0; t < std::min(5, r.H); ++t) {
        std::cout << "  " << r.datetimes[t]
                  << "  " << std::setw(8) << r.f10[t]
                  << "  " << r.kp[t] << "\n";
    }
}

// ---------------------------------------------------------------------------
// Command: query file  (forward declaration — defined after cmd_query)
// ---------------------------------------------------------------------------

static void cmd_query_file(const Session&, const std::vector<std::string>&);

// ---------------------------------------------------------------------------
// Command: query
// ---------------------------------------------------------------------------

static void cmd_query(const Session& s, const std::vector<std::string>& tok) {
    // query hold|interp <YYYY-MM-DD> <HH:MM:SS> <lst> <lat> <alt_km>
    if (!s.interp) {
        std::cerr << "No result yet — run a forecast first.\n";
        return;
    }
    if (tok.size() < 3) {
        std::cerr << "Usage: query hold|interp <YYYY-MM-DD> <HH:MM:SS> <lst> <lat> <alt_km>\n"
                     "       query file hold|interp <path>\n";
        return;
    }

    const std::string& mode = tok[1];

    if (mode == "file") {
        cmd_query_file(s, tok);   // forward-declared below
        return;
    }

    if (tok.size() < 7) {
        std::cerr << "Usage: query hold|interp <YYYY-MM-DD> <HH:MM:SS> <lst> <lat> <alt_km>\n";
        return;
    }
    std::string when;
    double lst, lat, alt_km;
    try {
        when   = parse_datetime(tok, 2);
        lst    = std::stod(tok[4]);
        lat    = std::stod(tok[5]);
        alt_km = std::stod(tok[6]);
    } catch (const std::exception& e) {
        std::cerr << "Bad arguments: " << e.what() << "\n";
        return;
    }

    try {
        if (mode == "hold") {
            auto r = s.interp->query_hold_next(when, lst, lat, alt_km);
            std::cout << "hold_next_hour:\n"
                      << "  requested : " << r.datetime_requested << "\n"
                      << "  used      : " << r.datetime_used      << "\n"
                      << std::scientific << std::setprecision(4)
                      << "  density   : " << r.density            << " kg/m³\n"
                      << std::defaultfloat
                      << "  t_index   : " << r.t_index            << "\n";

        } else if (mode == "interp") {
            auto r = s.interp->query_interp_time(when, lst, lat, alt_km);
            std::cout << "interp_time:\n"
                      << "  datetime  : " << r.datetime           << "\n"
                      << std::scientific << std::setprecision(4)
                      << "  density   : " << r.density            << " kg/m³\n"
                      << std::defaultfloat
                      << "  left  t=" << r.t_index_left
                      << " (" << r.datetime_left  << ")\n"
                      << "  right t=" << r.t_index_right
                      << " (" << r.datetime_right << ")\n"
                      << "  w_right   : " << r.time_weight_right  << "\n";

        } else {
            std::cerr << "Unknown mode '" << mode << "'. Use 'hold' or 'interp'.\n";
        }
    } catch (const rope::TimeOutOfRangeError& e) {
        std::cerr << "TimeOutOfRangeError: "    << e.what() << "\n";
    } catch (const rope::SpatialOutOfRangeError& e) {
        std::cerr << "SpatialOutOfRangeError: " << e.what() << "\n";
    }
}

// ---------------------------------------------------------------------------
// Command: query file
//
// Reads a CSV with columns:
//   YYYY, MM, DD, HH, MIN, SS, lst, lat, alt_km
// Runs a spatial query for each row. Results are written to <out_path> when
// provided, otherwise to stdout. Status/errors always go to stderr.
//
// Usage:
//   query file hold|interp <in_path> [out_path]
//
// Hold output columns:
//   datetime_requested, datetime_used, lst, lat, alt_km, density_kg_m3, t_index
// Interp output columns:
//   datetime, lst, lat, alt_km, density_kg_m3, t_left, t_right, w_right
// ---------------------------------------------------------------------------

static void cmd_query_file(const Session& s, const std::vector<std::string>& tok) {
    // tok: ["query", "file", "hold|interp", "<in_path>", (optional)"<out_path>"]
    if (!s.interp) {
        std::cerr << "No result yet — run a forecast first.\n";
        return;
    }
    if (tok.size() < 4) {
        std::cerr << "Usage: query file hold|interp <in_path> [out_path]\n";
        return;
    }

    const std::string& mode    = tok[2];
    const std::string& in_path = tok[3];

    if (mode != "hold" && mode != "interp") {
        std::cerr << "Unknown mode '" << mode << "'. Use 'hold' or 'interp'.\n";
        return;
    }

    // Optional output file; default to stdout.
    std::ofstream out_file;
    if (tok.size() >= 5) {
        out_file.open(tok[4]);
        if (!out_file.is_open()) {
            std::cerr << "query file: cannot open output file '" << tok[4] << "'\n";
            return;
        }
        std::cerr << "Writing results to " << tok[4] << "\n";
    }
    std::ostream& out = out_file.is_open() ? out_file : std::cout;

    rope::CSVReader csv(in_path);

    for (const auto& col : {"YYYY","MM","DD","HH","MIN","SS","lst","lat","alt_km"}) {
        if (!csv.has_column(col)) {
            std::cerr << "query file: missing column '" << col << "' in " << in_path << "\n"
                      << "  Required: YYYY, MM, DD, HH, MIN, SS, lst, lat, alt_km\n";
            return;
        }
    }

    if (mode == "hold") {
        out << "datetime_requested,datetime_used,lst,lat,alt_km,density_kg_m3,t_index\n";
    } else {
        out << "datetime,lst,lat,alt_km,density_kg_m3,t_left,t_right,w_right\n";
    }

    int n_ok = 0, n_err = 0;
    for (size_t i = 0; i < csv.nrows(); ++i) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d",
            csv.get_int("YYYY", i), csv.get_int("MM",  i), csv.get_int("DD",  i),
            csv.get_int("HH",   i), csv.get_int("MIN", i), csv.get_int("SS",  i));
        const std::string when(buf);

        const double lst    = csv.get_float("lst",    i);
        const double lat    = csv.get_float("lat",    i);
        const double alt_km = csv.get_float("alt_km", i);

        try {
            if (mode == "hold") {
                auto r = s.interp->query_hold_next(when, lst, lat, alt_km);
                out << r.datetime_requested << "," << r.datetime_used << ","
                    << lst << "," << lat << "," << alt_km << ","
                    << std::scientific << std::setprecision(6) << r.density
                    << std::defaultfloat
                    << "," << r.t_index << "\n";
            } else {
                auto r = s.interp->query_interp_time(when, lst, lat, alt_km);
                out << r.datetime << ","
                    << lst << "," << lat << "," << alt_km << ","
                    << std::scientific << std::setprecision(6) << r.density
                    << std::defaultfloat
                    << "," << r.t_index_left
                    << "," << r.t_index_right
                    << "," << r.time_weight_right << "\n";
            }
            ++n_ok;
        } catch (const rope::TimeOutOfRangeError& e) {
            std::cerr << "row " << i << " skipped (TimeOutOfRange): " << e.what() << "\n";
            out << when << ",,,,,,\n";
            ++n_err;
        } catch (const rope::SpatialOutOfRangeError& e) {
            std::cerr << "row " << i << " skipped (SpatialOutOfRange): " << e.what() << "\n";
            out << when << ",,,,,,\n";
            ++n_err;
        }
    }
    std::cerr << "query file: " << n_ok << " ok, " << n_err << " skipped.\n";
}

// ---------------------------------------------------------------------------
// One-shot run — fixed demo forecast + two example point queries.
// ---------------------------------------------------------------------------

static void run_oneshot(const rope::ROPE& rope) {
    const std::string start     = "2024-02-09 00:00:00";
    const int         H         = 120;
    const std::string dt_hold   = "2024-02-10 00:00:00";
    const std::string dt_interp = "2024-02-10 00:30:00";
    const double q_lst    = 10.5;
    const double q_lat    = 25.0;
    const double q_alt_km = 400.0;

    std::cout << "=== ROPE C++ Oneshot ===\n"
              << "  start   : " << start << "\n"
              << "  horizon : " << H << " hours\n\n";

    // --- Forecast ---
    auto t0 = std::chrono::steady_clock::now();
    rope::ForecastResult res = rope.run(start, H);
    auto t1 = std::chrono::steady_clock::now();

    double run_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const auto& d = res.meta_density;
    float  mn   = *std::min_element(d.begin(), d.end());
    float  mx   = *std::max_element(d.begin(), d.end());
    double sum  = std::accumulate(d.begin(), d.end(), 0.0);
    float  mean = static_cast<float>(sum / d.size());

    std::cout << "Run time  : " << run_ms << " ms\n"
              << std::scientific << std::setprecision(3)
              << "Density   : min=" << mn << "  max=" << mx
              << "  mean=" << mean << "\n\n"
              << std::defaultfloat;

    std::cout << "Forecast window (first 5 rows):\n"
              << "  datetime              f10      kp\n";
    for (int t = 0; t < std::min(5, res.H); ++t) {
        std::cout << "  " << res.datetimes[t]
                  << "  " << std::setw(8) << res.f10[t]
                  << "  " << res.kp[t] << "\n";
    }
    std::cout << "\n";

    // --- Queries ---
    rope::DensityInterpolator qi(res);
    auto t2 = std::chrono::steady_clock::now();

    try {
        auto r1 = qi.query_hold_next(dt_hold, q_lst, q_lat, q_alt_km);
        std::cout << "hold_next_hour:\n"
                  << "  requested : " << r1.datetime_requested << "\n"
                  << "  used      : " << r1.datetime_used      << "\n"
                  << std::scientific << std::setprecision(4)
                  << "  density   : " << r1.density            << " kg/m³\n"
                  << std::defaultfloat
                  << "  t_index   : " << r1.t_index            << "\n\n";

        auto r2 = qi.query_interp_time(dt_interp, q_lst, q_lat, q_alt_km);
        std::cout << "interp_time:\n"
                  << "  datetime  : " << r2.datetime           << "\n"
                  << std::scientific << std::setprecision(4)
                  << "  density   : " << r2.density            << " kg/m³\n"
                  << std::defaultfloat
                  << "  left  t=" << r2.t_index_left
                  << " (" << r2.datetime_left  << ")\n"
                  << "  right t=" << r2.t_index_right
                  << " (" << r2.datetime_right << ")\n"
                  << "  w_right   : " << r2.time_weight_right  << "\n\n";

    } catch (const rope::TimeOutOfRangeError& e) {
        std::cerr << "TimeOutOfRangeError: "    << e.what() << "\n";
    } catch (const rope::SpatialOutOfRangeError& e) {
        std::cerr << "SpatialOutOfRangeError: " << e.what() << "\n";
    }

    auto t3 = std::chrono::steady_clock::now();
    double itp_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::cout << "Interp time : " << itp_ms << " ms\n";
}

// ---------------------------------------------------------------------------
// REPL loop
// ---------------------------------------------------------------------------

static void run_repl(const rope::ROPE& rope) {
    Session s{rope, std::nullopt, nullptr};
    std::cout << "ROPE ready.  Type 'help' for available commands.\n\n";

    std::string line;
    while (true) {
        std::cout << "> " << std::flush;
        if (!std::getline(std::cin, line)) break;   // EOF / Ctrl-D

        const auto tok = tokenize(line);
        if (tok.empty()) continue;

        const std::string& cmd = tok[0];

        if      (cmd == "quit" || cmd == "exit") { break; }
        else if (cmd == "help")                  { cmd_help(); }
        else if (cmd == "run")                   { cmd_run(s, tok); }
        else if (cmd == "stats")                 { cmd_stats(s); }
        else if (cmd == "query")                 { cmd_query(s, tok); }
        else {
            std::cerr << "Unknown command '" << cmd << "'.  Type 'help'.\n";
        }
    }

    std::cout << "\nGoodbye.\n";
}

// ---------------------------------------------------------------------------
// Socket helpers
// ---------------------------------------------------------------------------

// Read one newline-terminated line from fd into `line` (strips \r\n).
// Returns false on EOF or error.
static bool sock_read_line(int fd, std::string& line) {
    line.clear();
    char c;
    while (true) {
        ssize_t n = ::read(fd, &c, 1);
        if (n <= 0) return false;
        if (c == '\n') return true;
        if (c != '\r') line += c;
    }
}

// Write a line (appends \n) to fd.  Ignores SIGPIPE-style errors.
static void sock_write_line(int fd, const std::string& msg) {
    std::string s = msg + "\n";
    ::write(fd, s.c_str(), s.size());
}

// ---------------------------------------------------------------------------
// Socket client handler — one per accepted connection, runs in its own thread.
// ---------------------------------------------------------------------------
static void handle_socket_client(
    std::shared_ptr<const ForecastState> state,
    int fd
) {
    std::string line;
    while (sock_read_line(fd, line)) {
        if (line.empty()) continue;

        // Tokenise
        std::istringstream iss(line);
        std::vector<std::string> tok;
        std::string t;
        while (iss >> t) tok.push_back(t);
        if (tok.empty()) continue;

        const std::string& cmd = tok[0];

        // ping — liveness check
        if (cmd == "ping") {
            sock_write_line(fd, "pong");
            continue;
        }

        // quit — clean disconnect
        if (cmd == "quit") {
            sock_write_line(fd, "bye");
            break;
        }

        // hold / interp — density query
        if (cmd == "hold" || cmd == "interp") {
            // Expected tokens: cmd date time lst lat alt_km  (6 total)
            if (tok.size() < 6) {
                sock_write_line(fd,
                    "err bad_request: usage: " + cmd +
                    " <YYYY-MM-DD> <HH:MM:SS> <lst> <lat> <alt_km>");
                continue;
            }
            std::string when = tok[1] + " " + tok[2];
            double lst, lat, alt_km;
            try {
                lst    = std::stod(tok[3]);
                lat    = std::stod(tok[4]);
                alt_km = std::stod(tok[5]);
            } catch (...) {
                sock_write_line(fd, "err bad_request: could not parse numeric arguments");
                continue;
            }

            try {
                double density;
                if (cmd == "hold") {
                    density = state->interp.query_hold_next(when, lst, lat, alt_km).density;
                } else {
                    density = state->interp.query_interp_time(when, lst, lat, alt_km).density;
                }
                std::ostringstream oss;
                oss << "ok " << std::scientific << std::setprecision(6) << density;
                sock_write_line(fd, oss.str());

            } catch (const rope::TimeOutOfRangeError& e) {
                sock_write_line(fd, std::string("err time_out_of_range: ") + e.what());
            } catch (const rope::SpatialOutOfRangeError& e) {
                sock_write_line(fd, std::string("err spatial_out_of_range: ") + e.what());
            } catch (const std::exception& e) {
                sock_write_line(fd, std::string("err internal: ") + e.what());
            }
            continue;
        }

        sock_write_line(fd,
            "err bad_request: unknown command '" + cmd +
            "' (expected: hold | interp | ping | quit)");
    }
    ::close(fd);
}

// ---------------------------------------------------------------------------
// Socket server — run forecast once, then serve clients indefinitely.
// ---------------------------------------------------------------------------
static void run_socket(const rope::ROPE& rope, const ConfigReader& cfg,
                       const fs::path& cfg_dir) {
    // Read socket config
    const std::string sock_path =
        resolve_path(cfg.get("socket.path", "/tmp/rope.sock"), cfg_dir);
    const std::string start = cfg.get("socket.start",   "2024-02-09 00:00:00");
    const int         H     = cfg.get_int("socket.horizon", 120);

    std::cout << "Socket    : " << sock_path << "\n"
              << "Forecast  : start=" << start << "  horizon=" << H << " h\n\n";

    // --- Forecast (run once) ---
    auto t0 = std::chrono::steady_clock::now();
    auto state = std::make_shared<ForecastState>(rope.run(start, H));
    auto t1 = std::chrono::steady_clock::now();

    const auto& d = state->result.meta_density;
    float mn   = *std::min_element(d.begin(), d.end());
    float mx   = *std::max_element(d.begin(), d.end());
    double sum = std::accumulate(d.begin(), d.end(), 0.0);
    float mean = static_cast<float>(sum / d.size());

    std::cout << "Forecast  : "
              << std::chrono::duration<double, std::milli>(t1 - t0).count()
              << " ms\n"
              << std::scientific << std::setprecision(3)
              << "Density   : min=" << mn << "  max=" << mx
              << "  mean=" << mean << "\n"
              << "Window    : " << state->interp.time_min()
              << " → " << state->interp.time_max() << "\n\n"
              << std::defaultfloat;

    // --- Create Unix domain socket ---
    ::unlink(sock_path.c_str());   // remove stale socket if present

    int server_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd < 0) {
        throw std::runtime_error("socket(): " + std::string(std::strerror(errno)));
    }

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (sock_path.size() >= sizeof(addr.sun_path)) {
        throw std::runtime_error("Socket path too long (max " +
            std::to_string(sizeof(addr.sun_path) - 1) + " chars): " + sock_path);
    }
    sock_path.copy(addr.sun_path, sock_path.size());

    if (::bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(server_fd);
        throw std::runtime_error("bind(" + sock_path + "): " +
                                 std::string(std::strerror(errno)));
    }
    if (::listen(server_fd, /*backlog=*/8) < 0) {
        ::close(server_fd);
        throw std::runtime_error("listen(): " + std::string(std::strerror(errno)));
    }

    // Accept times out every second so we can check g_shutdown.
    timeval tv{1, 0};
    ::setsockopt(server_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    // Install signal handlers now that the socket path is known.
    g_sock_path = sock_path;
    ::signal(SIGINT,  on_shutdown_signal);
    ::signal(SIGTERM, on_shutdown_signal);

    std::cout << "Listening on " << sock_path << "  (Ctrl-C to stop)\n\n";

    // --- Accept loop ---
    while (!g_shutdown) {
        int client_fd = ::accept(server_fd, nullptr, nullptr);
        if (client_fd < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
                continue;  // timeout — check g_shutdown and loop
            if (g_shutdown) break;
            std::cerr << "accept(): " << std::strerror(errno) << "\n";
            continue;
        }

        // Spawn a detached thread per client.  The shared_ptr keeps `state`
        // alive for the thread's lifetime even if another thread or main
        // replaces it.
        std::thread([state, client_fd]() {
            handle_socket_client(state, client_fd);
        }).detach();
    }

    ::close(server_fd);
    ::unlink(sock_path.c_str());
    std::cout << "\nSocket closed.\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    std::string cfg_path;
    try {
        cfg_path = find_config(argc >= 2 ? argv[1] : nullptr);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Config    : " << cfg_path << "\n";
    ConfigReader cfg(cfg_path);
    fs::path cfg_dir = fs::path(cfg_path).parent_path();

    const std::string exported_dir =
        resolve_path(cfg.get("paths.exported_dir",  "exported"),              cfg_dir);
    const std::string driver_csv =
        resolve_path(cfg.get("paths.driver_csv",    "data/sw_celestrack_1957.csv"), cfg_dir);
    const std::string ic_csv =
        resolve_path(cfg.get("paths.ic_csv",        "data/IC_Table_modified.csv"),  cfg_dir);

    const int         intra_base      = cfg.get_int("threads.intra_threads_base",    1);
    const int         intra_decoder   = cfg.get_int("threads.intra_threads_decoder", 0);
    const std::string decoder_device  = cfg.get("decoder.device", "cpu");

    // --- Load ---
    auto t0 = std::chrono::steady_clock::now();
    rope::ROPE rope(exported_dir, driver_csv, ic_csv,
                    intra_base, intra_decoder, decoder_device);
    auto t1 = std::chrono::steady_clock::now();

    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Load time : " << std::fixed << std::setprecision(1)
              << load_ms << " ms\n"
              << std::defaultfloat;

    const std::string mode = cfg.get("session.mode", "repl");

    if (mode == "oneshot") {
        run_oneshot(rope);
    } else if (mode == "socket") {
        run_socket(rope, cfg, cfg_dir);
    } else {
        if (mode != "repl")
            std::cerr << "Warning: unknown session.mode '" << mode
                      << "', defaulting to repl.\n";
        // Warm up ORT kernels before the first user-issued 'run'.
        rope.run("2024-02-09 00:00:00", 1);
        std::cout << "Warm-up   : done\n\n";
        run_repl(rope);
    }
    return 0;
}
