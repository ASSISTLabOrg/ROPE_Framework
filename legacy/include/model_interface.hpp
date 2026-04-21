#pragma once
// model_interface.hpp — abstract model interface for the ROPE C++ runtime.
//
// Design goals:
//   1. Backend-agnostic: callers only interact with IModel.
//   2. Future-proof: swapping to ONNX Runtime, TorchScript, or any other
//      backend requires only a new IModel subclass, not changes to the
//      ROPE pipeline.
//   3. Minimal: no framework headers leak through this file.
//
// Concrete implementation: OnnxModel in onnx_model.hpp.
// Other backends (LibTorch TorchScript, TensorFlow C API, …) can be added
// by implementing IModel without touching the pipeline code.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rope_types.hpp"

namespace rope {

// ---------------------------------------------------------------------------
// Pure virtual model interface.
// ---------------------------------------------------------------------------
class IModel {
public:
    virtual ~IModel() = default;

    // Run a single forward pass.
    //   input       – flat float32 data in row-major order
    //   input_shape – shape of the input tensor (e.g. {1, 3, 16})
    //   output_shape – filled by the implementation with the output shape
    // Returns flat float32 output data in row-major order.
    virtual std::vector<float> infer(
        const std::vector<float>& input,
        const std::vector<int64_t>& input_shape,
        std::vector<int64_t>& output_shape
    ) = 0;

    // Human-readable name / path for error messages.
    virtual const std::string& name() const = 0;
};

// ---------------------------------------------------------------------------
// Backend tag used by the factory.
// ---------------------------------------------------------------------------
enum class ModelBackend {
    ONNX,       // ONNX Runtime           — onnx_model.hpp
#ifdef ROPE_USE_LIBTORCH
    LibTorch,   // TorchScript (libtorch) — libtorch_model.hpp
#endif
};

// ---------------------------------------------------------------------------
// Factory — implemented in model_factory.hpp (include only where models are
// constructed; this declaration is sufficient everywhere else).
// ---------------------------------------------------------------------------
std::unique_ptr<IModel> make_model(
    const std::string& path,
    ModelBackend backend       = ModelBackend::ONNX,
    int intra_op_threads       = 1,
    bool use_dnnl              = false,
    const std::string& device  = "cpu"   // LibTorch only; ignored for ONNX
);

} // namespace rope
