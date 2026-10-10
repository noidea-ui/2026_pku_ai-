#include "max_pool.cuh"

namespace {
__global__ void max_pool_forward_kernel(const float* x, int N, int C, int H, int W,
                                        float* y, float* mask) {
    const int out_h = H / 2;
    const int out_w = W / 2;
    const int total = N * C * out_h * out_w;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int ow = idx % out_w;
    const int oh = (idx / out_w) % out_h;
    const int c = (idx / (out_w * out_h)) % C;
    const int n = idx / (out_w * out_h * C);
    const int base = ((n * C + c) * H + 2 * oh) * W + 2 * ow;

    // Stable tie-breaking: first maximum in top-left, top-right, bottom-left, bottom-right order.
    int best_idx = base;
    float best = x[base];
    const int candidates[3] = {base + 1, base + W, base + W + 1};
    for (int i = 0; i < 3; ++i) {
        const float value = x[candidates[i]];
        if (value > best) {
            best = value;
            best_idx = candidates[i];
        }
    }
    y[idx] = best;
    mask[idx] = static_cast<float>(best_idx);
}

__global__ void max_pool_backward_kernel(const float* dy, const float* mask, int total, float* dx) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) dx[static_cast<int>(mask[idx])] = dy[idx];
}
}

void max_pool_forward(const float* d_x, int N, int C, int H, int W, float* d_y, float* d_mask) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0 || H % 2 != 0 || W % 2 != 0)
        throw std::invalid_argument("max_pool_forward requires positive dimensions and even H,W");
    const int total = N * C * (H / 2) * (W / 2);
    constexpr int threads = 256;
    max_pool_forward_kernel<<<(total + threads - 1) / threads, threads>>>(d_x, N, C, H, W, d_y, d_mask);
    CUDA_KERNEL_CHECK();
}

void max_pool_backward(const float* d_dy, const float* d_mask, int N, int C, int H, int W, float* d_dx) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0 || H % 2 != 0 || W % 2 != 0)
        throw std::invalid_argument("max_pool_backward requires positive dimensions and even H,W");
    CUDA_CHECK(cudaMemset(d_dx, 0, static_cast<std::size_t>(N) * C * H * W * sizeof(float)));
    const int total = N * C * (H / 2) * (W / 2);
    constexpr int threads = 256;
    max_pool_backward_kernel<<<(total + threads - 1) / threads, threads>>>(d_dy, d_mask, total, d_dx);
    CUDA_KERNEL_CHECK();
}
