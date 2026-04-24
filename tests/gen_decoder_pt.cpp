// gen_decoder_pt.cpp — generates tests/fixtures/test_models/coae_decoder.pt
// using the LibTorch version that is actually linked, ensuring compatibility.
// Built and run by CMake as a CTest FIXTURES_SETUP step.

#include <torch/script.h>
#include <filesystem>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: gen_decoder_pt <output.pt>\n";
        return 1;
    }
    try {
        std::filesystem::create_directories(
            std::filesystem::path(argv[1]).parent_path());
        torch::jit::Module m("FakeDecoder");
        // Defines forward: (batch, 10) -> (batch, 116640) filled with zeros.
        // 116640 = GRID_LST * GRID_LAT * GRID_ALT = 72 * 36 * 45
        m.define("def forward(self, x: Tensor) -> Tensor:\n"
                 "    return torch.zeros(x.shape[0], 116640)\n");
        m.save(argv[1]);
        std::cout << "Generated " << argv[1] << "\n";
    } catch (const c10::Error& e) {
        std::cerr << "LibTorch error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
