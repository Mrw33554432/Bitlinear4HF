#include <torch/extension.h>
#include <vector>

torch::Tensor matMulCUDA(torch::Tensor A, torch::Tensor B, c10::optional<torch::Tensor> bias = c10::nullopt);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mat_mul", &matMulCUDA, "Matrix multiplication with optional bias (CUDA)",
          py::arg("A"), py::arg("B"), py::arg("bias") = py::none());
}
