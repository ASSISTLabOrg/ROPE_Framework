#pragma once
// rope.hpp — ROPE main class.
//
// Usage (mirrors rope.py):
//
//   rope::ROPE rope("exported/");
//   rope::ForecastResult res = rope.run("2024-02-09 00:00:00", 120);
//   // res.meta_density has shape (H, 72, 36, 45)
//
// All models are loaded once at construction time.
// run() is thread-safe (reads only).

#include <memory>
#include <string>
#include <vector>

#include "driver_window.hpp"
#include "dynamic_rollout.hpp"
#include "ic_table.hpp"
#include "latent_decoder.hpp"
#include "meta_model.hpp"
#include "model_interface.hpp"
#include "normalizer.hpp"
#include "rope_types.hpp"
#include "sequence_builder.hpp"

namespace rope {

// Configuration constants — kept separate from the heavy headers so
// callers can read them without including everything.
struct ROPEConfig {
    int latent_dim  = 10;
    int seq_len     = 3;
    int n_base_models = 15;   // 5 LSTM + 5 GRU + 5 Transformer
    int decode_batch  = 120; // time steps per COAE decoder call (effectively unbatched)
};

class ROPE {
public:
    // exported_dir : path to the directory produced by export_models.py
    // driver_csv   : path to sw_celestrack_1957.csv  (or equivalent)
    // ic_csv       : path to IC_Table_modified.csv
    // intra_threads: ORT intra-op threads per model session
    // intra_threads_base   : ORT intra-op threads for the 15 base models and
    //                        meta model (small matmul-heavy models; 1 is fine
    //                        since OpenMP already parallelises across models).
    // intra_threads_decoder: ORT intra-op threads for the COAE decoder
    //                        (conv-heavy; benefits from all available cores).
    //                        0 = use hardware_concurrency().
    //                        Ignored when decoder_device != "cpu".
    // decoder_device       : PyTorch device string for the COAE decoder when
    //                        ROPE_USE_LIBTORCH is enabled.  "cpu" runs on CPU;
    //                        "cuda" / "cuda:0" runs on NVIDIA or AMD GPU (the
    //                        distinction depends on which LibTorch build is
    //                        linked — CUDA or ROCm).  Ignored when using the
    //                        ONNX Runtime backend.
    explicit ROPE(
        const std::string& exported_dir,
        const std::string& driver_csv,
        const std::string& ic_csv,
        int intra_threads_base    = 1,
        int intra_threads_decoder = 0,
        const std::string& decoder_device = "cpu"
    );

    // Run the full ROPE forecast pipeline.
    //   start_datetime : "YYYY-MM-DD HH:MM:SS" (UTC, rounded to hour)
    //   horizon        : forecast length in hours (H)
    ForecastResult run(const std::string& start_datetime, int horizon = 120) const;

    const ROPEConfig& config() const { return cfg_; }

private:
    ROPEConfig cfg_;

    // Data
    std::unique_ptr<SpaceWeatherDB> sw_db_;
    std::unique_ptr<ICTable>        ic_table_;

    // Normalizers
    std::unique_ptr<FeatureNormalizer> ts_norm_;
    std::unique_ptr<CAEDenormalizer>   cae_denorm_;

    // Sequence builder
    DriverCols driver_cols_;
    std::unique_ptr<SequenceBuilder> seq_builder_;

    // Models
    std::vector<std::unique_ptr<IModel>> base_models_;  // 15 base models
    std::unique_ptr<IModel>              meta_model_;
    std::unique_ptr<IModel>              decoder_model_;

    // Pipeline helpers (stateless → const methods are fine)
    std::unique_ptr<DynamicRollout> rollout_;

    std::unique_ptr<LatentDecoder>  decoder_;
};

} // namespace rope
