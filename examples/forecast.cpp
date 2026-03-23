#include "physics_prediction.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>
#include <chrono>
#include <ctime>
#include <cstring>
#include <stdexcept>

using namespace physics_prediction;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct ForecastConfig {
    // Model paths
    std::string predictor_path;
    std::string decoder_path;
    std::string encoder_path;   // optional

    // Forecast parameters
    std::string start_time;     // ISO 8601 UTC  e.g. "2025-01-15T12:00:00Z"
    int         forecast_length = 0;  // minutes

    // Inference tuning
    int  num_threads  = 4;
    bool use_gpu      = false;
    int  gpu_device_id = 0;

    // Output
    std::string output_path;    // empty = auto-generate
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string trim(const std::string& s) {
    const char* ws = " \t\r\n";
    size_t a = s.find_first_not_of(ws);
    if (a == std::string::npos) return {};
    size_t b = s.find_last_not_of(ws);
    return s.substr(a, b - a + 1);
}

/**
 * Parse a simple INI-style config file.
 * Lines: key = value   (# comments, [sections] ignored)
 */
static std::map<std::string, std::string> parseConfigFile(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open config file: " + path);

    std::map<std::string, std::string> cfg;
    std::string line;
    while (std::getline(f, line)) {
        auto comment = line.find('#');
        if (comment != std::string::npos) line = line.substr(0, comment);
        line = trim(line);
        if (line.empty() || line[0] == '[') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));
        if (!key.empty()) cfg[key] = val;
    }
    return cfg;
}

/**
 * Parse ISO 8601 UTC string → time_t.
 * Accepted formats:
 *   YYYY-MM-DDTHH:MM:SSZ
 *   YYYY-MM-DD HH:MM:SS
 */
static time_t parseUTCTime(const std::string& s) {
    struct tm tm = {};
    const char* p = nullptr;
    p = strptime(s.c_str(), "%Y-%m-%dT%H:%M:%SZ", &tm);
    if (!p || *p != '\0')
        p = strptime(s.c_str(), "%Y-%m-%d %H:%M:%S", &tm);
    if (!p)
        throw std::invalid_argument(
            "Cannot parse time \"" + s + "\". "
            "Expected YYYY-MM-DDTHH:MM:SSZ or YYYY-MM-DD HH:MM:SS");
    return timegm(&tm);
}

static void printUsage(const char* prog) {
    std::cout
        << "Usage:\n"
        << "  " << prog << " --config FILE\n"
        << "  " << prog << " --predictor P --decoder D "
                           "--start-time T --forecast-length N [OPTIONS]\n"
        << "\nRequired:\n"
        << "  --predictor  PATH    Predictor ONNX model\n"
        << "  --decoder    PATH    Decoder ONNX model\n"
        << "  --start-time TIME    Forecast start time in UTC "
                                  "(YYYY-MM-DDTHH:MM:SSZ)\n"
        << "  --forecast-length N  Forecast length in minutes\n"
        << "\nOptional:\n"
        << "  --config     FILE    Load parameters from config file\n"
        << "                       (CLI args override file values)\n"
        << "  --encoder    PATH    Encoder ONNX model\n"
        << "  --threads    N       Inference threads (default: 4)\n"
        << "  --gpu               Use GPU if available\n"
        << "  --gpu-device N       GPU device ID (default: 0)\n"
        << "  --output     PATH    Output file (default: auto-named .bin)\n"
        << "  --help               Show this message\n";
}

/**
 * Build ForecastConfig from argv, merging any --config file first
 * then applying per-argument overrides.
 */
static ForecastConfig loadConfig(int argc, char* argv[]) {
    // Collect raw key→value from file (if any) then CLI
    std::map<std::string, std::string> raw;

    // First pass: find --config
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--config" && i + 1 < argc) {
            auto file = parseConfigFile(argv[i + 1]);
            raw.insert(file.begin(), file.end());
        }
    }

    // Second pass: CLI args override file
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + a);
            return argv[++i];
        };
        if      (a == "--predictor")       raw["predictor"]       = next();
        else if (a == "--decoder")         raw["decoder"]         = next();
        else if (a == "--encoder")         raw["encoder"]         = next();
        else if (a == "--start-time")      raw["start_time"]      = next();
        else if (a == "--forecast-length") raw["forecast_length"] = next();
        else if (a == "--threads")         raw["num_threads"]     = next();
        else if (a == "--gpu")             raw["use_gpu"]         = "true";
        else if (a == "--gpu-device")      raw["gpu_device_id"]   = next();
        else if (a == "--output")          raw["output"]          = next();
        else if (a == "--config")          { ++i; /* already handled */ }
        else if (a == "--help")            { printUsage(argv[0]); exit(0); }
        else {
            std::cerr << "Unknown option: " << a << "\n";
            printUsage(argv[0]);
            exit(1);
        }
    }

    ForecastConfig cfg;
    if (raw.count("predictor"))       cfg.predictor_path    = raw["predictor"];
    if (raw.count("decoder"))         cfg.decoder_path      = raw["decoder"];
    if (raw.count("encoder"))         cfg.encoder_path      = raw["encoder"];
    if (raw.count("start_time"))      cfg.start_time        = raw["start_time"];
    if (raw.count("forecast_length")) cfg.forecast_length   = std::stoi(raw["forecast_length"]);
    if (raw.count("num_threads"))     cfg.num_threads       = std::stoi(raw["num_threads"]);
    if (raw.count("use_gpu"))         cfg.use_gpu           = (raw["use_gpu"] == "true" || raw["use_gpu"] == "1");
    if (raw.count("gpu_device_id"))   cfg.gpu_device_id     = std::stoi(raw["gpu_device_id"]);
    if (raw.count("output"))          cfg.output_path       = raw["output"];
    return cfg;
}

// ---------------------------------------------------------------------------
// Driver construction
// ---------------------------------------------------------------------------

/**
 * Build the driver tensor for the forecast window.
 *
 * This reference implementation encodes two features per timestep:
 *   [0] normalised Unix epoch  (seconds / 1e9)
 *   [1] normalised step index  (step / num_steps)
 *
 * Output shape: [1, num_steps, 2]
 *
 * Replace this function with your actual geophysical driver construction
 * once the model's input schema is known (e.g. F10.7, Kp, Ap indices,
 * altitude levels, etc.).
 */
static Tensor buildDrivers(time_t start_epoch, int forecast_length_min) {
    int num_steps = forecast_length_min;
    std::vector<float> data;
    data.reserve(static_cast<size_t>(num_steps) * 2);
    for (int step = 0; step < num_steps; ++step) {
        float t_norm    = static_cast<float>(start_epoch + step * 60) / 1e9f;
        float step_norm = static_cast<float>(step) / static_cast<float>(num_steps);
        data.push_back(t_norm);
        data.push_back(step_norm);
    }
    return Tensor(data, {1, static_cast<int64_t>(num_steps), 2});
}

// ---------------------------------------------------------------------------
// Output writer
// ---------------------------------------------------------------------------

/**
 * Write forecast output to a binary file.
 *
 * File layout:
 *   4 bytes  magic  "UPCA"
 *   8 bytes  start_epoch (time_t, signed 64-bit)
 *   4 bytes  forecast_length_min (int32)
 *   4 bytes  ndims (uint32)
 *   ndims*8  shape dims (int64 each)
 *   N*4      float32 prediction data (row-major)
 */
static void writeOutput(const std::string& path, time_t start_epoch,
                        int forecast_length_min, const Tensor& pred) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Cannot write output file: " + path);

    const uint32_t MAGIC = 0x55504341u; // "UPCA"
    f.write(reinterpret_cast<const char*>(&MAGIC), 4);

    int64_t se = static_cast<int64_t>(start_epoch);
    f.write(reinterpret_cast<const char*>(&se), 8);

    int32_t fl = forecast_length_min;
    f.write(reinterpret_cast<const char*>(&fl), 4);

    uint32_t ndims = static_cast<uint32_t>(pred.shape.size());
    f.write(reinterpret_cast<const char*>(&ndims), 4);

    for (int64_t d : pred.shape)
        f.write(reinterpret_cast<const char*>(&d), 8);

    f.write(reinterpret_cast<const char*>(pred.data.data()),
            static_cast<std::streamsize>(pred.data.size() * sizeof(float)));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    try {
        ForecastConfig cfg = loadConfig(argc, argv);

        // Validate
        if (cfg.predictor_path.empty())
            throw std::runtime_error("--predictor (or config key 'predictor') is required");
        if (cfg.decoder_path.empty())
            throw std::runtime_error("--decoder (or config key 'decoder') is required");
        if (cfg.start_time.empty())
            throw std::runtime_error("--start-time (or config key 'start_time') is required");
        if (cfg.forecast_length <= 0)
            throw std::runtime_error("--forecast-length must be a positive integer (minutes)");

        time_t start_epoch = parseUTCTime(cfg.start_time);

        // Compute and format end time
        time_t end_epoch = start_epoch + static_cast<time_t>(cfg.forecast_length) * 60;
        char end_buf[32];
        strftime(end_buf, sizeof(end_buf), "%Y-%m-%dT%H:%M:%SZ", gmtime(&end_epoch));

        std::cout << "=== Upper Atmosphere Density Forecast ===\n"
                  << "Start time (UTC):  " << cfg.start_time << "\n"
                  << "End time   (UTC):  " << end_buf << "\n"
                  << "Forecast length:   " << cfg.forecast_length << " min\n"
                  << "\nModels:\n"
                  << "  Predictor: " << cfg.predictor_path << "\n"
                  << "  Decoder:   " << cfg.decoder_path << "\n";
        if (!cfg.encoder_path.empty())
            std::cout << "  Encoder:   " << cfg.encoder_path << "\n";
        std::cout << "\nInference: threads=" << cfg.num_threads
                  << " gpu=" << (cfg.use_gpu ? "true" : "false") << "\n\n";

        // Build pipeline
        InferenceConfig inf;
        inf.num_threads     = cfg.num_threads;
        inf.use_gpu         = cfg.use_gpu;
        inf.gpu_device_id   = cfg.gpu_device_id;
        inf.optimization_level = 3;

        std::cout << "Loading pipeline...\n";
        PhysicsPipeline pipeline(cfg.predictor_path, cfg.decoder_path,
                                  cfg.encoder_path, inf);
        pipeline.printPipelineInfo();

        // Build driver tensor
        Tensor drivers = buildDrivers(start_epoch, cfg.forecast_length);

        // Run forecast
        std::cout << "\nRunning forecast...\n";
        auto t0 = std::chrono::high_resolution_clock::now();
        Tensor prediction = pipeline.predict(drivers);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        std::cout << "Completed in " << ms << " ms\n";
        std::cout << "Output shape: [";
        for (size_t i = 0; i < prediction.shape.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << prediction.shape[i];
        }
        std::cout << "]\n";

        // Determine output path
        std::string outpath = cfg.output_path;
        if (outpath.empty()) {
            char buf[48];
            strftime(buf, sizeof(buf), "forecast_%Y%m%d_%H%M", gmtime(&start_epoch));
            outpath = std::string(buf) + "_" + std::to_string(cfg.forecast_length) + "min.bin";
        }

        writeOutput(outpath, start_epoch, cfg.forecast_length, prediction);
        std::cout << "Output written to: " << outpath << "\n"
                  << "\n✓ Forecast complete.\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
