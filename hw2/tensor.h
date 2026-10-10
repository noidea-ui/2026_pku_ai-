#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <memory>
#include <numeric>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

enum class Device{
    CPU,
    GPU
};

class Tensor {

public:
    Tensor() : shape_({}),numel_(0),device_(Device::CPU),data_(nullptr){};

    // 深拷贝：复制独立的内存副本，而不是共享 data_
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);

    // 移动语义：转移所有权，保持返回临时对象时的高效性
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    Tensor(const std::vector<int>& shape,Device device = Device::CPU)
        :shape_(shape),device_(device){
            numel_ = 1;
            for (int dim:shape_){
                numel_ *= dim;
            }
            allocate_memory();
        }

    const std::vector<int>&shape() const{return shape_;}
    Device device() const{return device_;}
    size_t numel() const{return numel_;}
    float* data(){return data_.get();}
    const float* data() const {return data_.get();}

    Tensor cpu() const;
    Tensor gpu() const;

    
private:
    void allocate_memory();

    std::vector<int> shape_;
    size_t numel_;
    Device device_;
    std::shared_ptr<float> data_;

};


#endif // TENSOR_H