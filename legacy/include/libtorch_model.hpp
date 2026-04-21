#pragma once
// libtorch_model.hpp — TorchScript implementation of IModel.
//
// Loads a model saved with torch.jit.save() and runs it on the requested
// device.  Requires LibTorch (find_package(Torch) in CMake).
//
// Device strings follow PyTorch conventions:
//   "cpu"      — CPU (default)
//   "cuda"     — first available NVIDIA GPU  (CUDA build of LibTorch)
//   "cuda:0"   — specific NVIDIA GPU index
//   "cuda"     — first available AMD GPU     (ROCm build of LibTorch)
//   "cuda:1"   — specific AMD GPU index
//
// ROCm and CUDA builds of LibTorch share the same C++ API.  The distinction
// is purely which LibTorch distribution was linked at build time.  At runtime,
// torch::cuda::is_available() returns true for both CUDA and ROCm builds when
// a compatible GPU is present.
//
// Do not include this header in files that should remain torch-free
// (e.g. dynamic_rollout.hpp, interpolator.hpp).  Use model_factory.hpp
// as the single inclusion point when constructing models.

#include <stdexcept>
#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/utils.h>   // at::set_num_threads

#include "model_interface.hpp"

namespace rope {

class LibTorchModel : public IModel {
public:
    // path        : path to a TorchScript (.pt) file
    // device_str  : PyTorch device string — "cpu", "cuda", "cuda:0", etc.
    //               Both CUDA (NVIDIA) and ROCm (AMD) builds use "cuda".
    // num_threads : intra-op thread count for CPU execution; ignored on GPU.
    //               Since this is a process-wide setting, pass the value only
    //               for the one model that should own threading (the decoder).
    explicit LibTorchModel(const std::string& path,
                           const std::string& device_str = "cpu",
                           int                num_threads = 0)
        : name_(path)
        , device_(device_str)
    {
        // Validate GPU availability before trying to load onto the device.
        if (device_.is_cuda() && !torch::cuda::is_available()) {
            throw std::runtime_error(
                "LibTorchModel: device=\"" + device_str + "\" requested but "
                "torch::cuda::is_available() returned false.\n"
                "  For NVIDIA GPUs: ensure a CUDA-enabled LibTorch build and "
                "a compatible CUDA driver.\n"
                "  For AMD GPUs: ensure a ROCm-enabled LibTorch build and "
                "ROCm drivers are installed.");
        }

        if (num_threads > 0 && device_.is_cpu())
            at::set_num_threads(num_threads);

        module_ = torch::jit::load(path, device_);
        module_.eval();
    }

    std::vector<float> infer(
        const std::vector<float>& input,
        const std::vector<int64_t>& input_shape,
        std::vector<int64_t>& output_shape
    ) override {
        torch::NoGradGuard no_grad;

        // Wrap caller's buffer (CPU), then move to the target device.
        auto in_cpu = torch::from_blob(
            const_cast<float*>(input.data()),
            input_shape,
            torch::kFloat32
        );
        auto in_tensor = in_cpu.to(device_);

        // Forward pass on device; bring result back to CPU.
        auto out = module_.forward({in_tensor})
                          .toTensor()
                          .to(torch::kCPU)
                          .contiguous();

        const auto& sizes = out.sizes();
        output_shape.assign(sizes.begin(), sizes.end());

        const float* ptr = out.data_ptr<float>();
        return std::vector<float>(ptr, ptr + out.numel());
    }

    const std::string& name() const override { return name_; }

    // Exposed for diagnostics (e.g. logging which device is active).
    const torch::Device& device() const { return device_; }

private:
    std::string        name_;
    torch::Device      device_;
    torch::jit::Module module_;
};

} // namespace rope
