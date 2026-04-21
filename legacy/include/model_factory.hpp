#pragma once
// model_factory.hpp — implementation of the make_model factory.
//
// Include this only in translation units that construct models (rope.cpp).
// Everything else should include only model_interface.hpp so that heavy
// backend headers (ORT, LibTorch) do not leak into the rest of the pipeline.

#include <stdexcept>
#include <memory>
#include <string>

#include "model_interface.hpp"
#include "onnx_model.hpp"
#ifdef ROPE_USE_LIBTORCH
#  include "libtorch_model.hpp"
#endif

namespace rope {

inline std::unique_ptr<IModel> make_model(
    const std::string& path,
    ModelBackend        backend,
    int                 intra_op_threads,
    bool                use_dnnl,
    const std::string&  device
) {
    switch (backend) {
    case ModelBackend::ONNX:
        return std::make_unique<OnnxModel>(
            path, intra_op_threads, /*inter_op_threads=*/1, use_dnnl);
#ifdef ROPE_USE_LIBTORCH
    case ModelBackend::LibTorch:
        return std::make_unique<LibTorchModel>(path, device, intra_op_threads);
#endif
    }
    throw std::runtime_error("make_model: unknown backend");
}

} // namespace rope
