#pragma once
// model_factory.h — implementation of make_model().
//
// Include only in pipeline.cpp (the single translation unit that constructs
// models).  Everything else uses model_interface.h only.

#include <memory>
#include <stdexcept>
#include <string>

#include "model_interface.h"
#include "onnx_model.h"
#ifdef ROPE_USE_LIBTORCH
#  include "libtorch_model.h"
#endif

namespace rope::forecast {

inline std::unique_ptr<IModel> make_model(
    const std::string& path,
    ModelBackend       backend,
    int                intra_op_threads,
    bool               use_dnnl,
    const std::string& device
) {
    switch (backend) {
    case ModelBackend::ONNX:
        (void)device;
        return std::make_unique<OnnxModel>(
            path, intra_op_threads, /*inter_op_threads=*/1, use_dnnl);
#ifdef ROPE_USE_LIBTORCH
    case ModelBackend::LibTorch:
        return std::make_unique<LibTorchModel>(path, device, intra_op_threads);
#endif
    }
    throw std::runtime_error("make_model: unknown backend");
}

} // namespace rope::forecast
