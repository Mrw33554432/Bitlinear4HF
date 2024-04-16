#include <torch/extension.h>
#include <ATen/ATen.h>

template <typename x_scalar_t, typename w_scalar_t>
__global__ void matMulOptimized(const x_scalar_t *x, const w_scalar_t *w, x_scalar_t *C, int batchSize, int N, int K, int M, const x_scalar_t *bias) {
    int batchIndex = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batchIndex < batchSize && row < N && col < M) {
        x_scalar_t sum = 0;
        for (int k = 0; k < K; ++k) {
            w_scalar_t weight = w[col * K + k];
            if (weight != 0) {
                sum += (weight == 1 ? x[batchIndex * N * K + row * K + k] : -x[batchIndex * N * K + row * K + k]);
            }
        }
        C[batchIndex * N * M + row * M + col] = sum + (bias ? bias[col] : 0);
    }
}

torch::Tensor matMulCUDA(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> bias = {}) {
    AT_ASSERTM(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16, "Input tensor must be float32 or float16");
    AT_ASSERTM(w.scalar_type() == torch::kInt8 || w.scalar_type() == torch::kFloat32 || w.scalar_type() == torch::kFloat16, "Weights tensor must be int8, float32, or float16");

    int batchSize = x.dim() > 2 ? x.size(0) : 1;
    auto N = x.size(-2);
    auto K = x.size(-1);
    auto M = w.size(0);

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    torch::Tensor C = torch::empty({batchSize, N, M}, options);

    const int threads = 16;
    const dim3 blocks((M + threads - 1) / threads, (N + threads - 1) / threads, batchSize);

    // Dispatch based on 'x' data type
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "matMulOptimized", ([&] {
        // Now dispatch based on 'w' data type
        switch (w.scalar_type()) {
            case torch::kInt8:
                matMulOptimized<scalar_t, int8_t><<<blocks, dim3(threads, threads, 1)>>>(
                    x.data_ptr<scalar_t>(),
                    w.data_ptr<int8_t>(),
                    C.data_ptr<scalar_t>(),
                    batchSize, N, K, M,
                    bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr);
                break;
            case torch::kFloat32:
            case torch::kFloat16:
                matMulOptimized<scalar_t, scalar_t><<<blocks, dim3(threads, threads, 1)>>>(
                    x.data_ptr<scalar_t>(),
                    w.data_ptr<scalar_t>(),
                    C.data_ptr<scalar_t>(),
                    batchSize, N, K, M,
                    bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr);
                break;
            default:
                AT_ERROR("Unsupported weight type");
        }
    }));

    return C;
}
