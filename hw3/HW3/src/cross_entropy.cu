#include "cross_entropy.cuh"

namespace {
__global__ void cross_entropy_forward_kernel(const float* probs, const int* labels,
                                             int N, int C, float* loss) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float sum = 0.0f;
    constexpr float eps = 1.0e-12f;
    for (int n = 0; n < N; ++n) {
        const int label = labels[n];
        const float p = fmaxf(probs[n * C + label], eps);
        sum -= logf(p);
    }
    *loss = sum / static_cast<float>(N);
}

__global__ void cross_entropy_backward_kernel(const float* probs, const int* labels,
                                              float* dlogits, int N, int C) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C;
    if (idx >= total) return;
    const int n = idx / C;
    const int c = idx % C;
    const float target = (c == labels[n]) ? 1.0f : 0.0f;
    dlogits[idx] = (probs[idx] - target) / static_cast<float>(N);
}
}

void cross_entropy_forward(const float* d_probs, const int* d_labels, int N, int C, float* d_loss) {
    if (N <= 0 || C <= 0) throw std::invalid_argument("cross_entropy_forward dimensions must be positive");
    cross_entropy_forward_kernel<<<1, 1>>>(d_probs, d_labels, N, C, d_loss);
    CUDA_KERNEL_CHECK();
}

void cross_entropy_backward(const float* d_probs, const int* d_labels, float* d_dlogits, int N, int C) {
    if (N <= 0 || C <= 0) throw std::invalid_argument("cross_entropy_backward dimensions must be positive");
    constexpr int threads = 256;
    const int total = N * C;
    cross_entropy_backward_kernel<<<(total + threads - 1) / threads, threads>>>(d_probs, d_labels, d_dlogits, N, C);
    CUDA_KERNEL_CHECK();
}
