#include"tensor.h"

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