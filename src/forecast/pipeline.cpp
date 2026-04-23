// pipeline.cpp — ROPE forecast pipeline implementation.
//
// Uncertainty propagation strategy: run the COAE decoder twice.
//   1. Decode with the mean latents   → density_mean
//   2. Decode with mean+std latents   → density_plus
//   uncertainty = |density_plus - density_mean|  per voxel

#include "rope/forecast/pipeline.h"

// Internal headers — heavy backends included here only, not in public header.
#include "model_factory.h"
#include "meta_model.h"
#include "latent_decoder.h"
#include "sequence_builder.h"
#include "dynamic_rollout.h"

// io/ and core/ public headers
#include "rope/io/driver_db.h"
#include "rope/io/ic_table.h"
#include "rope/io/stats.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef ROPE_USE_OPENMP
#  include <omp.h>
#endif

namespace rope::forecast {

namespace fs = std::filesystem;

// Cholesky-Banachiewicz: factorise a k×k SPD matrix A (row-major) in-place,
// leaving the lower triangle of L in A.  Returns false if A is not SPD.
static bool cholesky_inplace(float* a, int k) {
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j <= i; ++j) {
            float s = a[i * k + j];
            for (int r = 0; r < j; ++r)
                s -= a[i * k + r] * a[j * k + r];
            if (i == j) {
                if (s <= 0.0f) return false;
                a[i * k + j] = std::sqrt(s);
            } else {
                a[i * k + j] = s / a[j * k + j];
            }
        }
        for (int j = i + 1; j < k; ++j)
            a[i * k + j] = 0.0f;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Helper: join a directory with a filename.
// ---------------------------------------------------------------------------
static fs::path jpath(const fs::path& dir, const char* name) {
    return dir / name;
}

// ---------------------------------------------------------------------------
// PipelineImpl
// ---------------------------------------------------------------------------
class PipelineImpl : public Pipeline {
public:
    // Pipeline constants.
    static constexpr int K = 10;   // latent_dim
    static constexpr int S = 3;    // seq_len
    static constexpr int M = 15;   // n_base_models
    static constexpr int DECODE_BATCH = 120;

    explicit PipelineImpl(const Config& cfg) {
        const fs::path& dir = cfg.exported_dir;

        // --- Normalisation statistics ---
        io::Stats stats_ts  = io::Stats::load(jpath(dir, "stats_ts.bin"));
        io::Stats stats_cae = io::Stats::load(jpath(dir, "stats_cae.bin"));

        ts_norm_    = std::make_unique<io::FeatureNormalizer>(stats_ts,  K);
        cae_denorm_ = std::make_unique<io::CAEDenormalizer>(stats_cae);

        // Infer driver columns from total_dim.
        const int D  = ts_norm_->total_dim();
        const int dd = ts_norm_->driver_dim();
        if      (dd == 6) driver_cols_ = DriverCols::Six;
        else if (dd == 7) driver_cols_ = DriverCols::Seven;
        else throw std::runtime_error(
            "Pipeline: unsupported driver_dim=" + std::to_string(dd) +
            "; expected 6 or 7.");

        // --- Data tables ---
        std::cout << "Loading space-weather database…\n";
        sw_db_    = std::make_unique<io::SpaceWeatherDB>(cfg.driver_csv);

        std::cout << "Loading IC table…\n";
        ic_table_ = std::make_unique<io::ICTable>(cfg.ic_csv);

        // --- Sequence builder ---
        seq_builder_ = std::make_unique<SequenceBuilder>(
            *ts_norm_, *ic_table_, driver_cols_, S);

        // --- Base models (15): 0-4 LSTM, 5-9 GRU, 10-14 Transformer ---
        std::cout << "Loading 15 base models…\n";
        base_models_.reserve(M);
        for (int i = 0; i < M; ++i) {
            std::string fname = std::string("base_model_") +
                (i < 10 ? "0" : "") + std::to_string(i) + ".onnx";
            const bool is_transformer = (i >= 10);
            base_models_.push_back(
                std::make_unique<OnnxModel>(
                    (dir / fname).string(),
                    cfg.intra_threads_base,
                    is_transformer ? 2 : 1
                )
            );
        }

        // --- Meta model ---
        std::cout << "Loading meta model…\n";
        meta_model_ = std::make_unique<MetaModel>(
            make_model((dir / "meta_model.onnx").string(),
                       ModelBackend::ONNX, cfg.intra_threads_base,
                       false, "cpu"),
            K, S, D, M
        );

        // --- COAE decoder (resolve thread count) ---
        int dec_threads = cfg.intra_threads_decoder;
        if (dec_threads <= 0)
            dec_threads = static_cast<int>(std::thread::hardware_concurrency());

        std::cout << "Loading COAE decoder…\n";
#ifdef ROPE_USE_LIBTORCH
        std::cout << "  backend: LibTorch  device=" << cfg.decoder_device << "\n";
        decoder_model_ = make_model(
            (dir / "coae_decoder.pt").string(),
            ModelBackend::LibTorch,
            dec_threads,
            false,
            cfg.decoder_device
        );
#else
        std::cout << "  backend: ONNX Runtime  threads=" << dec_threads << "\n";
        decoder_model_ = make_model(
            (dir / "coae_decoder.onnx").string(),
            ModelBackend::ONNX,
            dec_threads
        );
#endif

        // --- Pipeline helpers ---
        rollout_ = std::make_unique<DynamicRollout>(K, S, D);
        decoder_ = std::make_unique<LatentDecoder>(
            *decoder_model_, *cae_denorm_, DECODE_BATCH);

        compute_uncertainty_ = cfg.compute_uncertainty;

        std::cout << "Pipeline loaded.  total_dim=" << D
                  << "  driver_dim=" << dd
                  << "  uncertainty=" << (compute_uncertainty_ ? "on" : "off") << "\n";
    }

    ForecastGrid run(const std::string& start_iso, int horizon) override {
        const int H = horizon;
        const int D = ts_norm_->total_dim();

        // 1. Build driver window: (S-1 history) + H forecast rows.
        std::vector<io::DriverRow> all_rows =
            io::DriverWindowBuilder::build(*sw_db_, start_iso, H, S);

        std::vector<io::DriverRow> hist_rows(
            all_rows.begin(), all_rows.begin() + S);
        std::vector<io::DriverRow> fcast_rows(
            all_rows.begin() + S - 1,
            all_rows.begin() + S - 1 + H);

        // 2. Build initial normalized sequence: (S, D).
        std::vector<float> X_init = seq_builder_->build_X_init_norm(hist_rows);

        // 3. Build x_chunk: (H, S, D).
        std::vector<float> x_chunk =
            seq_builder_->build_x_chunk(X_init, fcast_rows, H);

        // 4. Run 15 base models (in parallel when OpenMP is available).
        //    base_latents_norm: (M, H-1, K) flat
        const int T = H - 1;
        std::vector<float> base_latents_norm(
            static_cast<size_t>(M) * T * K, 0.0f);

#ifdef ROPE_USE_OPENMP
        #pragma omp parallel for schedule(dynamic)
#endif
        for (int m = 0; m < M; ++m) {
            float* out = base_latents_norm.data() + m * T * K;
            rollout_->run(*base_models_[m], x_chunk.data(), H, out);
        }

        // 5. Meta fusion.
        //    Pack [x_chunk_slice (T*S*D) | base_latents_norm (M*T*K)].
        std::vector<float> meta_in;
        meta_in.reserve(
            static_cast<size_t>(T) * S * D +
            static_cast<size_t>(M) * T * K);
        meta_in.insert(meta_in.end(), x_chunk.begin(),
                       x_chunk.begin() + static_cast<size_t>(T) * S * D);
        meta_in.insert(meta_in.end(),
                       base_latents_norm.begin(), base_latents_norm.end());

        std::vector<int64_t> meta_in_shape = {T, (int64_t)(S * D), M, K};
        std::vector<int64_t> meta_out_shape;
        std::vector<float>   meta_out =
            meta_model_->infer(meta_in, meta_in_shape, meta_out_shape);

        // meta_out = [mean(T*K) | std(T*K)] — std unused; UT derives spread from base-model cloud.
        std::vector<float> meta_mean_norm(meta_out.begin(),
                                          meta_out.begin() + T * K);

        // 6. Denorm meta mean (H-1, K) and build mu_lat (H, K).
        ts_norm_->denorm_latents_block(meta_mean_norm.data(), T);

        std::vector<float> init_lat(K);
        std::copy(X_init.begin() + (S - 1) * D,
                  X_init.begin() + (S - 1) * D + K,
                  init_lat.begin());
        ts_norm_->denorm_latents_inplace(init_lat.data());

        std::vector<float> mu_lat(static_cast<size_t>(H) * K);
        std::copy(init_lat.begin(), init_lat.end(), mu_lat.begin());
        std::copy(meta_mean_norm.begin(), meta_mean_norm.end(), mu_lat.begin() + K);

        const size_t total_voxels = static_cast<size_t>(H) * GRID_VOXELS;
        std::vector<float> density_mean;
        std::vector<float> uncertainty;

        if (!compute_uncertainty_) {
            // 7. Fast path: decode mu_lat once, zero uncertainty.
            density_mean = decoder_->decode(mu_lat, H, K);
            uncertainty.assign(total_voxels, 0.0f);
        } else {
            // 7. Full path: Unscented Transform.

            // Denorm base-model latents — needed for per-timestep covariance.
            ts_norm_->denorm_latents_block(base_latents_norm.data(), M * T);

            // UT hyperparameters (α=1, β=2, κ=0 → λ=0, c=K).
            const float c_ut  = static_cast<float>(K);
            const int   N_SIG = 2 * K + 1;

            std::vector<float> Wm(N_SIG, 1.0f / (2.0f * c_ut));
            std::vector<float> Wc(N_SIG, 1.0f / (2.0f * c_ut));
            Wm[0] = 0.0f;
            Wc[0] = 2.0f;  // 1 - α² + β = 1 - 1 + 2

            // Build sigma points sigma_lat (H, N_SIG, K).
            const size_t sig_stride = static_cast<size_t>(N_SIG) * K;
            std::vector<float> sigma_lat(static_cast<size_t>(H) * sig_stride, 0.0f);

            std::vector<float> Pt(K * K);
            std::vector<float> cPt(K * K);
            constexpr float EPS_JIT  = 1e-6f;
            constexpr float EPS_JIT2 = 1e-3f;

            for (int t = 0; t < H; ++t) {
                const float* mu_t  = mu_lat.data() + t * K;
                float*       sig_t = sigma_lat.data() + t * static_cast<ptrdiff_t>(sig_stride);

                std::copy(mu_t, mu_t + K, sig_t);

                std::fill(Pt.begin(), Pt.end(), 0.0f);
                for (int m = 0; m < M; ++m) {
                    const float* x_mt = (t == 0)
                        ? init_lat.data()
                        : base_latents_norm.data() + m * T * K + (t - 1) * K;
                    for (int i = 0; i < K; ++i) {
                        float di = x_mt[i] - mu_t[i];
                        for (int j = 0; j <= i; ++j)
                            Pt[i * K + j] += di * (x_mt[j] - mu_t[j]);
                    }
                }
                const float inv_m = (M > 1) ? 1.0f / static_cast<float>(M - 1) : 0.0f;
                for (int i = 0; i < K; ++i)
                    for (int j = 0; j <= i; ++j) {
                        float v = Pt[i * K + j] * inv_m;
                        Pt[i * K + j] = v;
                        Pt[j * K + i] = v;
                    }

                for (int idx = 0; idx < K * K; ++idx) cPt[idx] = c_ut * Pt[idx];
                for (int i   = 0; i   < K;     ++i)   cPt[i * K + i] += c_ut * EPS_JIT;

                if (!cholesky_inplace(cPt.data(), K)) {
                    for (int i = 0; i < K; ++i) cPt[i * K + i] += c_ut * EPS_JIT2;
                    if (!cholesky_inplace(cPt.data(), K))
                        throw std::runtime_error(
                            "UT: Cholesky failed at t=" + std::to_string(t));
                }

                for (int i = 0; i < K; ++i) {
                    float* sp = sig_t + (1 + i)     * K;
                    float* sm = sig_t + (1 + K + i) * K;
                    for (int j = 0; j < K; ++j) {
                        float sji = cPt[j * K + i];
                        sp[j] = mu_t[j] + sji;
                        sm[j] = mu_t[j] - sji;
                    }
                }
            }

            // Batch-decode all H*N_SIG sigma points.
            std::vector<float> dens_sigmas =
                decoder_->decode(sigma_lat, H * N_SIG, K);

            // UT mean then variance → density and uncertainty.
            density_mean.assign(total_voxels, 0.0f);

            for (int t = 0; t < H; ++t)
                for (int s = 0; s < N_SIG; ++s) {
                    const float* src = dens_sigmas.data() +
                                       (static_cast<size_t>(t) * N_SIG + s) * GRID_VOXELS;
                    float* dst = density_mean.data() + static_cast<size_t>(t) * GRID_VOXELS;
                    const float w = Wm[s];
                    for (int v = 0; v < GRID_VOXELS; ++v)
                        dst[v] += w * src[v];
                }

            uncertainty.assign(total_voxels, 0.0f);

            for (int t = 0; t < H; ++t)
                for (int s = 0; s < N_SIG; ++s) {
                    const float* src = dens_sigmas.data() +
                                       (static_cast<size_t>(t) * N_SIG + s) * GRID_VOXELS;
                    const float* mu  = density_mean.data() + static_cast<size_t>(t) * GRID_VOXELS;
                    float* dst = uncertainty.data() + static_cast<size_t>(t) * GRID_VOXELS;
                    const float w = Wc[s];
                    for (int v = 0; v < GRID_VOXELS; ++v) {
                        float d = src[v] - mu[v];
                        dst[v] += w * d * d;
                    }
                }

            for (float& u : uncertainty)
                u = std::sqrt(std::max(u, 0.0f));
        }

        // 11. Build ForecastGrid.
        ForecastGrid grid;
        grid.H          = H;
        grid.density    = std::move(density_mean);
        grid.uncertainty= std::move(uncertainty);
        grid.times.reserve(H);
        for (int t = 0; t < H; ++t)
            grid.times.push_back(fcast_rows[t].tp);

        return grid;
    }

private:
    bool       compute_uncertainty_{true};
    DriverCols driver_cols_{DriverCols::Six};

    // Data
    std::unique_ptr<io::SpaceWeatherDB>     sw_db_;
    std::unique_ptr<io::ICTable>            ic_table_;

    // Normalizers
    std::unique_ptr<io::FeatureNormalizer>  ts_norm_;
    std::unique_ptr<io::CAEDenormalizer>    cae_denorm_;

    // Sequence builder
    std::unique_ptr<SequenceBuilder>        seq_builder_;

    // Models
    std::vector<std::unique_ptr<IModel>>    base_models_;
    std::unique_ptr<IModel>                 meta_model_;
    std::unique_ptr<IModel>                 decoder_model_;

    // Pipeline helpers
    std::unique_ptr<DynamicRollout>         rollout_;
    std::unique_ptr<LatentDecoder>          decoder_;
};

// ---------------------------------------------------------------------------
// load()
// ---------------------------------------------------------------------------
std::unique_ptr<Pipeline> load(const Config& cfg) {
    return std::make_unique<PipelineImpl>(cfg);
}

} // namespace rope::forecast
