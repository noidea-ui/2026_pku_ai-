#include "fc.cuh"

namespace {
__global__ void fc_add_bias_kernel(float* y, const float* b, int N, int Cout) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * Cout;
    if (idx < total) y[idx] += b[idx % Cout];
}

__global__ void fc_bias_grad_kernel(const float* dy, float* db, int N, int Cout) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < Cout) {
        float sum = 0.0f;
        for (int n = 0; n < N; ++n) sum += dy[n * Cout + c];
        db[c] = sum;
    }
}
}

void fc_forward(cublasHandle_t handle, const float* d_x, const float* d_w, const float* d_b,
                float* d_y, int N, int Cin, int Cout) {
    if (N <= 0 || Cin <= 0 || Cout <= 0) throw std::invalid_argument("fc_forward dimensions must be positive");

    // Row-major buffers are interpreted by cuBLAS as their column-major transposes:
    // Y^T = W * X^T, so stored W is transposed and stored X is not.
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             Cout, N, Cin, &alpha,
                             d_w, Cin, d_x, Cin, &beta, d_y, Cout));

    constexpr int threads = 256;
    const int total = N * Cout;
    fc_add_bias_kernel<<<(total + threads - 1) / threads, threads>>>(d_y, d_b, N, Cout);
    CUDA_KERNEL_CHECK();
}

void fc_backward(cublasHandle_t handle, const float* d_x, const float* d_w, const float* d_dy,
                 float* d_dx, float* d_dw, float* d_db, int N, int Cin, int Cout) {
    if (N <= 0 || Cin <= 0 || Cout <= 0) throw std::invalid_argument("fc_backward dimensions must be positive");

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Stored dW is column-major [Cin,Cout] = X^T * dY.
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                             Cin, Cout, N, &alpha,
                             d_x, Cin, d_dy, Cout, &beta, d_dw, Cin));

    // Stored dX is column-major [Cin,N] = W^T * dY^T.
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             Cin, N, Cout, &alpha,
                             d_w, Cin, d_dy, Cout, &beta, d_dx, Cin));

    constexpr int threads = 256;
    fc_bias_grad_kernel<<<(Cout + threads - 1) / threads, threads>>>(d_dy, d_db, N, Cout);
    CUDA_KERNEL_CHECK();
}
