// rope.cpp — ROPE main class implementation.

#include "rope.hpp"
#include "model_factory.hpp"   // make_model (OnnxModel + LibTorchModel)
#include "onnx_model.hpp"      // OnnxModel — needed for direct instantiation in base-model loop

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef ROPE_USE_OPENMP
#  include <omp.h>
#endif

namespace rope {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Helper: join a directory path with a file name.
// ---------------------------------------------------------------------------
static std::string jpath(const std::string& dir, const std::string& name) {
    return (fs::path(dir) / name).string();
}

// ---------------------------------------------------------------------------
// Constructor — loads everything from disk once.
// ---------------------------------------------------------------------------
ROPE::ROPE(
    const std::string& exported_dir,
    const std::string& driver_csv,
    const std::string& ic_csv,
    int intra_threads_base,
    int intra_threads_decoder
) {
    if (intra_threads_decoder <= 0)
        intra_threads_decoder = static_cast<int>(
            std::thread::hardware_concurrency());
    std::cout << "Decoder intra-op threads: " << intra_threads_decoder << "\n";
    // --- Normalization statistics ---
    Stats stats_ts  = Stats::load(jpath(exported_dir, "stats_ts.bin"));
    Stats stats_cae = Stats::load(jpath(exported_dir, "stats_cae.bin"));

    ts_norm_   = std::make_unique<FeatureNormalizer>(stats_ts,  cfg_.latent_dim);
    cae_denorm_= std::make_unique<CAEDenormalizer>(stats_cae);

    // Infer driver_cols from the total_dim in the statistics.
    int dd = ts_norm_->driver_dim();
    if (dd == 6)       driver_cols_ = DriverCols::Six;
    else if (dd == 7)  driver_cols_ = DriverCols::Seven;
    else throw std::runtime_error(
        "ROPE: unsupported driver_dim=" + std::to_string(dd) +
        "; expected 6 or 7.  Re-export models or extend DriverCols.");

    // --- Data ---
    std::cout << "Loading space-weather database…\n";
    sw_db_ = std::make_unique<SpaceWeatherDB>(driver_csv);

    std::cout << "Loading IC table…\n";
    ic_table_ = std::make_unique<ICTable>(ic_csv);

    // --- Sequence builder ---
    seq_builder_ = std::make_unique<SequenceBuilder>(
        *ts_norm_, *ic_table_, driver_cols_, cfg_.seq_len
    );

    // Dimension constants used throughout the rest of the constructor.
    const int K = cfg_.latent_dim;
    const int S = cfg_.seq_len;
    const int D = ts_norm_->total_dim();
    const int M = cfg_.n_base_models;

    // --- Base models (15): 0-4 LSTM, 5-9 GRU, 10-14 Transformer ---
    // Transformer models have parallel attention heads; inter-op threads help.
    // LSTM/GRU are sequential by nature; inter-op threads are wasted there.
    std::cout << "Loading 15 base models…\n";
    base_models_.reserve(cfg_.n_base_models);
    for (int i = 0; i < cfg_.n_base_models; ++i) {
        std::string fname = std::string("base_model_") +
            (i < 10 ? "0" : "") + std::to_string(i) + ".onnx";
        const bool is_transformer = (i >= 10);
        base_models_.push_back(
            std::make_unique<OnnxModel>(
                jpath(exported_dir, fname),
                intra_threads_base,
                is_transformer ? 2 : 1   // inter-op threads
            )
        );
    }

    // --- Meta model (wraps inner ONNX model + ensemble fusion) ---
    std::cout << "Loading meta model…\n";
    meta_model_ = std::make_unique<MetaModel>(
        make_model(jpath(exported_dir, "meta_model.onnx"),
                   ModelBackend::ONNX, intra_threads_base),
        K, S, D, M
    );

    // --- COAE decoder ---
    std::cout << "Loading COAE decoder…\n";
#ifdef ROPE_USE_LIBTORCH
    decoder_model_ = make_model(
        jpath(exported_dir, "coae_decoder.pt"),
        ModelBackend::LibTorch, intra_threads_decoder
    );
#else
    decoder_model_ = make_model(
        jpath(exported_dir, "coae_decoder.onnx"),
        ModelBackend::ONNX, intra_threads_decoder
    );
#endif

    // --- Pipeline helpers ---
    rollout_ = std::make_unique<DynamicRollout>(K, S, D);
    decoder_ = std::make_unique<LatentDecoder>(
        *decoder_model_, *cae_denorm_, cfg_.decode_batch
    );

    std::cout << "ROPE loaded.  total_dim=" << D
              << "  driver_dim=" << dd << "\n";
}

// ---------------------------------------------------------------------------
// run() — full forecast pipeline.
// ---------------------------------------------------------------------------
ForecastResult ROPE::run(const std::string& start_datetime, int horizon) const {
    const int H = horizon;
    const int K = cfg_.latent_dim;
    const int S = cfg_.seq_len;
    const int D = ts_norm_->total_dim();
    const int M = cfg_.n_base_models;

    // 1. Build driver window: (seq_len-1 history) + H forecast rows.
    std::vector<DriverRow> all_rows = DriverWindowBuilder::build(
        *sw_db_, start_datetime, H, S
    );
    // Forecast rows start at index (S-1).
    std::vector<DriverRow> hist_rows (all_rows.begin(), all_rows.begin() + S);
    std::vector<DriverRow> fcast_rows(all_rows.begin() + S - 1,
                                      all_rows.begin() + S - 1 + H);

    // 2. Build initial normalized sequence: (S, D).
    std::vector<float> X_init = seq_builder_->build_X_init_norm(hist_rows);

    // 3. Build full x_chunk: (H, S, D).
    std::vector<float> x_chunk = seq_builder_->build_x_chunk(
        X_init, fcast_rows, H
    );

    // 4. Run all M base models (in parallel when OpenMP is available).
    //    base_latents_norm: (M, H-1, K) flat
    std::vector<float> base_latents_norm(M * (H - 1) * K, 0.0f);

#ifdef ROPE_USE_OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int m = 0; m < M; ++m) {
        float* out = base_latents_norm.data() + m * (H - 1) * K;
        rollout_->run(*base_models_[m], x_chunk.data(), H, out);
    }

    // 5. Meta fusion: (H-1, K) normalized.
    //    Pack [x_chunk_slice (T*S*D) | base_latents_norm (M*T*K)] for MetaModel.
    const int T = H - 1;
    std::vector<float> meta_in;
    meta_in.reserve(static_cast<size_t>(T) * S * D +
                    static_cast<size_t>(M) * T * K);
    meta_in.insert(meta_in.end(), x_chunk.begin(),
                   x_chunk.begin() + static_cast<size_t>(T) * S * D);
    meta_in.insert(meta_in.end(),
                   base_latents_norm.begin(), base_latents_norm.end());

    std::vector<int64_t> meta_in_shape = {T, (int64_t)(S * D), M, K};
    std::vector<int64_t> meta_out_shape;
    std::vector<float> meta_out =
        meta_model_->infer(meta_in, meta_in_shape, meta_out_shape);
    // meta_out = [mean (T*K) | std (T*K)]; take the mean half.
    std::vector<float> meta_latents_norm(meta_out.begin(),
                                         meta_out.begin() + T * K);

    // 6. De-normalize predicted latents (H-1, K).
    ts_norm_->denorm_latents_block(meta_latents_norm.data(), H - 1);
    // meta_latents_norm is now physical-space (H-1, K).

    // 7. Prepend t=0 latent (from last row of X_init, index S-1).
    //    init_lat_norm = X_init[(S-1)*D : (S-1)*D + K]
    std::vector<float> init_lat_norm(K);
    std::copy(X_init.begin() + (S - 1) * D,
              X_init.begin() + (S - 1) * D + K,
              init_lat_norm.begin());
    ts_norm_->denorm_latents_inplace(init_lat_norm.data());

    // Full latent series: (H, K)
    std::vector<float> latents_full(H * K);
    std::copy(init_lat_norm.begin(), init_lat_norm.end(), latents_full.begin());
    std::copy(meta_latents_norm.begin(), meta_latents_norm.end(),
              latents_full.begin() + K);

    // 8. Decode latents → physical density (H, 72, 36, 45).
    std::vector<float> density = decoder_->decode(latents_full, H, K);

    // 9. Package result.
    ForecastResult res;
    res.H = H;
    res.meta_density = std::move(density);
    res.datetimes.reserve(H);
    res.f10.reserve(H);
    res.kp.reserve(H);
    for (int t = 0; t < H; ++t) {
        res.datetimes.push_back(dt::to_string(fcast_rows[t].tp));
        res.f10.push_back(fcast_rows[t].f10);
        res.kp.push_back(fcast_rows[t].kp);
    }
    return res;
}

} // namespace rope
