#pragma once
// forecast/pipeline.h — public interface for the ROPE forecast pipeline.
//
// Usage:
//   auto cfg = rope::forecast::Config{...};
//   auto pipe = rope::forecast::load(cfg);
//   ForecastGrid grid = pipe->run("2024-02-09 00:00:00", 120);

#include "rope/core/types.h"

#include <filesystem>
#include <memory>
#include <string>

namespace rope::forecast {

// ---------------------------------------------------------------------------
// Config — all paths and tuning parameters needed to construct a Pipeline.
// ---------------------------------------------------------------------------
struct Config {
    std::filesystem::path exported_dir;   // directory produced by export_models.py
    std::filesystem::path driver_csv;     // sw_celestrack_1957.csv or equivalent
    std::filesystem::path ic_csv;         // IC_Table_modified.csv

    // ORT intra-op threads for the 15 base models and meta model.
    // 1 is fine; OpenMP parallelises across the 15 models already.
    int intra_threads_base    = 1;

    // ORT intra-op threads for the COAE decoder (conv-heavy).
    // 0 = std::thread::hardware_concurrency().
    // Ignored when decoder_device != "cpu".
    int intra_threads_decoder = 0;

    // PyTorch device string for the COAE decoder when LibTorch is linked.
    // "cpu", "cuda", "cuda:0", etc.  Ignored when using the ONNX backend.
    std::string decoder_device = "cpu";

    // When false, skip the Unscented Transform entirely.  The decoder runs
    // once on the meta-model mean latents; uncertainty is set to 0.
    bool compute_uncertainty = true;
};

// ---------------------------------------------------------------------------
// Pipeline — abstract interface; constructed by load().
// ---------------------------------------------------------------------------
class Pipeline {
public:
    virtual ~Pipeline() = default;

    // Run the full ROPE forecast pipeline.
    //   start_iso : "YYYY-MM-DD HH:MM:SS" UTC (rounded to the hour internally)
    //   horizon   : forecast length in hours (H)
    // Returns a ForecastGrid with H time steps, density and uncertainty fields.
    virtual ForecastGrid run(const std::string& start_iso, int horizon) = 0;
};

// ---------------------------------------------------------------------------
// load() — construct and initialize the pipeline from cfg.
// Loads all models and data tables from disk.  Throws on any error.
// ---------------------------------------------------------------------------
std::unique_ptr<Pipeline> load(const Config& cfg);

} // namespace rope::forecast
