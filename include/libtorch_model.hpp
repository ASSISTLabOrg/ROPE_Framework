#pragma once
// libtorch_model.hpp — TorchScript implementation of IModel.
//
// Loads a model saved with torch.jit.save() and runs it on the CPU.
// Requires LibTorch (find_package(Torch) in CMake).
//
// Do not include this header in files that should remain torch-free
// (e.g. dynamic_rollout.hpp, interpolator.hpp).  Use model_factory.hpp
// as the single inclusion point when constructing models.

#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/utils.h>   // at::set_num_threads

#include "model_interface.hpp"

namespace rope {

class LibTorchModel : public IModel {
public:
    // path        : path to a TorchScript (.pt) file
    // num_threads : intra-op thread count passed to torch::set_num_threads().
    //               Since this is a process-wide setting, pass the value only
    //               for the one model that should own threading (the decoder).
    explicit LibTorchModel(const std::string& path, int num_threads = 0)
        : name_(path)
    {
        if (num_threads > 0)
            at::set_num_threads(num_threads);

        module_ = torch::jit::load(path, torch::kCPU);
        module_.eval();
    }

    std::vector<float> infer(
        const std::vector<float>& input,
        const std::vector<int64_t>& input_shape,
        std::vector<int64_t>& output_shape
    ) override {
        torch::NoGradGuard no_grad;

        // Wrap the caller's buffer without copying.
        // The tensor is only used within this scope, so the raw pointer is safe.
        auto in_tensor = torch::from_blob(
            const_cast<float*>(input.data()),
            input_shape,
            torch::kFloat32
        );

        auto out = module_.forward({in_tensor}).toTensor().contiguous();

        const auto& sizes = out.sizes();
        output_shape.assign(sizes.begin(), sizes.end());

        const float* ptr = out.data_ptr<float>();
        return std::vector<float>(ptr, ptr + out.numel());
    }

    const std::string& name() const override { return name_; }

private:
    std::string        name_;
    torch::jit::Module module_;
};

} // namespace rope
