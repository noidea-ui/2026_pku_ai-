#include "conv2d.cuh"

namespace {
__global__ void im2col_kernel(const float* x, float* col, int Cin, int H, int W) {
    const int K = Cin * 9;
    const int P = H * W;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K * P) return;

    // col is column-major [K,P], so k changes fastest.
    const int k = idx % K;
    const int p = idx / K;
    const int c = k / 9;
    const int kh = (k % 9) / 3;
    const int kw = k % 3;
    const int oh = p / W;
    const int ow = p % W;
    const int ih = oh + kh - 1;
    const int iw = ow + kw - 1;

    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
        col[idx] = x[(c * H + ih) * W + iw];
    } else {
        col[idx] = 0.0f;
    }
}

__global__ void col2im_kernel(const float* col, float* x, int Cin, int H, int W) {
    const int total = Cin * H * W;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int w = idx % W;
    const int h = (idx / W) % H;
    const int c = idx / (H * W);
    const int K = Cin * 9;
    float sum = 0.0f;

    // For an input (h,w), find every output window and kernel offset that covers it.
    for (int kh = 0; kh < 3; ++kh) {
        const int oh = h - kh + 1;
        if (oh < 0 || oh >= H) continue;
        for (int kw = 0; kw < 3; ++kw) {
            const int ow = w - kw + 1;
            if (ow < 0 || ow >= W) continue;
            const int k = c * 9 + kh * 3 + kw;
            const int p = oh * W + ow;
            sum += col[k + p * K];
        }
    }
    x[idx] = sum;
}

__global__ void conv_add_bias_kernel(float* y, const float* b, int N, int Cout, int P) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * Cout * P;
    if (idx < total) {
        const int c = (idx / P) % Cout;
        y[idx] += b[c];
    }
}

__global__ void conv_bias_grad_kernel(const float* dy, float* db, int N, int Cout, int P) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= Cout) return;
    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        const float* channel = dy + (n * Cout + c) * P;
        for (int p = 0; p < P; ++p) sum += channel[p];
    }
    db[c] = sum;
}
}

void im2col_3x3(const float* d_x, float* d_col, int Cin, int H, int W) {
    if (Cin <= 0 || H <= 0 || W <= 0) throw std::invalid_argument("im2col dimensions must be positive");
    constexpr int threads = 256;
    const int total = Cin * 9 * H * W;
    im2col_kernel<<<(total + threads - 1) / threads, threads>>>(d_x, d_col, Cin, H, W);
    CUDA_KERNEL_CHECK();
}

void col2im_3x3(const float* d_col, float* d_x, int Cin, int H, int W) {
    if (Cin <= 0 || H <= 0 || W <= 0) throw std::invalid_argument("col2im dimensions must be positive");
    constexpr int threads = 256;
    const int total = Cin * H * W;
    col2im_kernel<<<(total + threads - 1) / threads, threads>>>(d_col, d_x, Cin, H, W);
    CUDA_KERNEL_CHECK();
}

void conv2d_forward(cublasHandle_t handle, const float* d_x, const float* d_w, const float* d_b,
                    float* d_y, int N, int Cin, int Cout, int H, int W) {
    if (N <= 0 || Cin <= 0 || Cout <= 0 || H <= 0 || W <= 0)
        throw std::invalid_argument("conv2d_forward dimensions must be positive");

    const int K = Cin * 9;
    const int P = H * W;
    DeviceBuffer<float> d_col(static_cast<std::size_t>(K) * P);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int n = 0; n < N; ++n) {
        im2col_3x3(d_x + static_cast<std::size_t>(n) * Cin * P, d_col.get(), Cin, H, W);
        // col: column-major [K,P], weight buffer is interpreted as column-major [K,Cout],
        // and NCHW output is interpreted as column-major [P,Cout]. Thus Y^T = col^T * W^T.
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 P, Cout, K, &alpha,
                                 d_col.get(), K, d_w, K, &beta,
                                 d_y + static_cast<std::size_t>(n) * Cout * P, P));
    }

    constexpr int threads = 256;
    const int total = N * Cout * P;
    conv_add_bias_kernel<<<(total + threads - 1) / threads, threads>>>(d_y, d_b, N, Cout, P);
    CUDA_KERNEL_CHECK();
}

void conv2d_backward(cublasHandle_t handle, const float* d_x, const float* d_w, const float* d_dy,
                     float* d_dx, float* d_dw, float* d_db,
                     int N, int Cin, int Cout, int H, int W) {
    if (N <= 0 || Cin <= 0 || Cout <= 0 || H <= 0 || W <= 0)
        throw std::invalid_argument("conv2d_backward dimensions must be positive");

    const int K = Cin * 9;
    const int P = H * W;
    DeviceBuffer<float> d_col(static_cast<std::size_t>(K) * P);
    DeviceBuffer<float> d_dcol(static_cast<std::size_t>(K) * P);

    constexpr int threads = 256;
    conv_bias_grad_kernel<<<(Cout + threads - 1) / threads, threads>>>(d_dy, d_db, N, Cout, P);
    CUDA_KERNEL_CHECK();

    const float alpha = 1.0f;
    for (int n = 0; n < N; ++n) {
        const float* dy_n = d_dy + static_cast<std::size_t>(n) * Cout * P;
        im2col_3x3(d_x + static_cast<std::size_t>(n) * Cin * P, d_col.get(), Cin, H, W);

        // Stored dW is column-major [K,Cout] = sum_n col[K,P] * dY[P,Cout].
        const float beta_dw = (n == 0) ? 0.0f : 1.0f;
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 K, Cout, P, &alpha,
                                 d_col.get(), K, dy_n, P, &beta_dw, d_dw, K));

        // dcol[K,P] = W^T[K,Cout] * dY^T[Cout,P].
        const float beta_dxcol = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                 K, P, Cout, &alpha,
                                 d_w, K, dy_n, P, &beta_dxcol, d_dcol.get(), K));
        col2im_3x3(d_dcol.get(), d_dx + static_cast<std::size_t>(n) * Cin * P, Cin, H, W);
    }
}
