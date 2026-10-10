#include"tensor.h"
#include <cstring>

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), numel_(other.numel_), device_(other.device_), data_(nullptr) {
    if (numel_ == 0) return;

    allocate_memory();

    if (device_ == Device::CPU) {
        std::memcpy(data_.get(), other.data_.get(), numel_ * sizeof(float));
    }
    else {
        cudaError_t err = cudaMemcpy(data_.get(), other.data_.get(),
                                     numel_ * sizeof(float),
                                     cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy DeviceToDevice failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;

    shape_ = other.shape_;
    numel_ = other.numel_;
    device_ = other.device_;
    data_.reset();

    if (numel_ == 0) return *this;

    allocate_memory();

    if (device_ == Device::CPU) {
        std::memcpy(data_.get(), other.data_.get(), numel_ * sizeof(float));
    }
    else {
        cudaError_t err = cudaMemcpy(data_.get(), other.data_.get(),
                                     numel_ * sizeof(float),
                                     cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy DeviceToDevice failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)),
      numel_(other.numel_),
      device_(other.device_),
      data_(std::move(other.data_)) {
    other.numel_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;

    shape_ = std::move(other.shape_);
    numel_ = other.numel_;
    device_ = other.device_;
    data_ = std::move(other.data_);

    other.numel_ = 0;
    return *this;
}

void Tensor::allocate_memory(){
    if(numel_ == 0) return;

    if(device_ == Device::CPU){
        float* ptr = new float[numel_]();

        data_ = std::shared_ptr<float>(ptr,[](float* p){ delete[] p;});

    }
    else{
        float* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr,numel_ *sizeof(float));
        if(err != cudaSuccess){
            throw std::runtime_error("CUDA MALLOC FAILED: "+ std::string(cudaGetErrorString(err)));
        }
        cudaMemset(ptr,0,numel_*sizeof(float));
        data_ = std::shared_ptr<float>(ptr,[](float* p){ cudaFree(p);});
    }
}

Tensor Tensor::cpu() const{
    if(device_ == Device::CPU){
        return *this;
    }

    Tensor res(shape_,Device::CPU);
    cudaError_t err = cudaMemcpy(res.data(),this->data(),numel_* sizeof(float),cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){
        throw std::runtime_error("cudaMemcpy GPU->CPU failed!");
    }
    return res;
}

Tensor Tensor::gpu() const{
    if(device_ == Device::GPU){
        return *this;
    }
    Tensor res(shape_,Device::GPU);
    cudaError_t err = cudaMemcpy(res.data(),this->data(),numel_*sizeof(float),cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
        throw std::runtime_error("cudaMemcpy CPU->GPU failed!");
    }
    return res;
}