#include <torch/extension.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <ATen/ATen.h>

template <typename scalar_t>
__global__ void matMulOptimized(const scalar_t *x, const scalar_t *w, scalar_t *C, int batchSize, int N, int K, int M, const scalar_t *bias) {
    int batchIndex = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batchIndex < batchSize && row < N && col < M) {
        scalar_t sum = 0;
        for (int k = 0; k < K; ++k) {
            scalar_t weight = w[col * K + k];
            if (weight != 0) {
                sum += (weight == 1 ? x[batchIndex * N * K + row * K + k] : -x[batchIndex * N * K + row * K + k]);
            }
        }
        C[batchIndex * N * M + row * M + col] = sum + (bias ? bias[col] : 0);
    }
}

torch::Tensor matMulCUDA(torch::Tensor x, torch::Tensor w, torch::optional<torch::Tensor> bias = {}) {
    int batchSize = x.dim() > 2 ? x.size(0) : 1;
    auto N = x.size(-2);
    auto K = x.size(-1);
    auto M = w.size(0); // Adjusted for expected transposed w

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    torch::Tensor C = torch::empty({batchSize, N, M}, options);

    const int threads = 16;
    const dim3 blocks((M + threads - 1) / threads, (N + threads - 1) / threads, batchSize);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "matMulOptimized", ([&] {
        matMulOptimized<scalar_t><<<blocks, dim3(threads, threads, 1)>>>(
            x.data_ptr<scalar_t>(),
            w.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            batchSize, N, K, M,
            bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr);
    }));

    return C;
}
