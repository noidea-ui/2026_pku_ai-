#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

inline void hw3_check_cuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s failed: %s\n", file, line, expression,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

inline void hw3_check_cublas(cublasStatus_t status, const char* expression, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error at %s:%d: %s failed (status=%d)\n", file, line,
                     expression, static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

inline void hw3_check_curand(curandStatus_t status, const char* expression, const char* file, int line) {
    if (status != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuRAND error at %s:%d: %s failed (status=%d)\n", file, line,
                     expression, static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expr) hw3_check_cuda((expr), #expr, __FILE__, __LINE__)
#define CUBLAS_CHECK(expr) hw3_check_cublas((expr), #expr, __FILE__, __LINE__)
#define CURAND_CHECK(expr) hw3_check_curand((expr), #expr, __FILE__, __LINE__)
#define CUDA_KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())

// Small RAII wrapper used by the test harness. Module APIs still use plain device pointers.
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : ptr_(nullptr), count_(count) {
        if (count_ > 0) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (ptr_ != nullptr) cudaFree(ptr_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_ != nullptr) cudaFree(ptr_);
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    std::size_t size() const { return count_; }

private:
    T* ptr_;
    std::size_t count_;
};

inline void copy_to_device(float* dst, const std::vector<float>& src) {
    if (!src.empty()) CUDA_CHECK(cudaMemcpy(dst, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice));
}

inline void copy_to_device(int* dst, const std::vector<int>& src) {
    if (!src.empty()) CUDA_CHECK(cudaMemcpy(dst, src.data(), src.size() * sizeof(int), cudaMemcpyHostToDevice));
}

inline void copy_to_host(std::vector<float>& dst, const float* src) {
    if (!dst.empty()) CUDA_CHECK(cudaMemcpy(dst.data(), src, dst.size() * sizeof(float), cudaMemcpyDeviceToHost));
}

inline void copy_to_host(std::vector<int>& dst, const int* src) {
    if (!dst.empty()) CUDA_CHECK(cudaMemcpy(dst.data(), src, dst.size() * sizeof(int), cudaMemcpyDeviceToHost));
}

static __global__ void hw3_scale_uniform_kernel(float* data, std::size_t count, float scale) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) data[idx] = (2.0f * data[idx] - 1.0f) * scale;
}

// Initialize values in [-scale, scale] using cuRAND. A fixed seed makes tests reproducible.
inline void curand_initialize(float* device_data, std::size_t count, float scale, unsigned long long seed) {
    if (count == 0) return;
    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, seed));
    CURAND_CHECK(curandGenerateUniform(generator, device_data, count));
    CURAND_CHECK(curandDestroyGenerator(generator));

    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    hw3_scale_uniform_kernel<<<blocks, threads>>>(device_data, count, scale);
    CUDA_KERNEL_CHECK();
}
