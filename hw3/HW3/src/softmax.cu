#include "softmax.cuh"

namespace {
__global__ void softmax_forward_kernel(const float* logits, float* probs, int N, int C) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const float* row_in = logits + static_cast<std::size_t>(n) * C;
    float* row_out = probs + static_cast<std::size_t>(n) * C;
    float max_value = row_in[0];
    for (int c = 1; c < C; ++c) max_value = fmaxf(max_value, row_in[c]);

    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        row_out[c] = expf(row_in[c] - max_value);
        sum += row_out[c];
    }
    for (int c = 0; c < C; ++c) row_out[c] /= sum;
}
}

void softmax_forward(const float* d_logits, float* d_probs, int N, int C) {
    if (N <= 0 || C <= 0) throw std::invalid_argument("softmax_forward dimensions must be positive");
    constexpr int threads = 128;
    softmax_forward_kernel<<<(N + threads - 1) / threads, threads>>>(d_logits, d_probs, N, C);
    CUDA_KERNEL_CHECK();
}
