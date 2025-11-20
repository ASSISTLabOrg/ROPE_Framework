#include <torch/torch.h>
#include <iostream>

int main() {
  torch::Tensor tensor = torch::eye(3);
  std::cout << tensor << std::endl;
}

// template <typename T> std::vector<T> Encoder::encode(std::vector<T>){}