#include <torch/extension.h>

torch::Tensor fa2forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fa2forward, "FlashAttention-2 forward (CUDA)");
}
