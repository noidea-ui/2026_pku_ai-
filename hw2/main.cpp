#include<iostream>
#include"tensor.h"

Tensor relu_forward(const Tensor& input);
Tensor relu_backward(const Tensor& grad_output,const Tensor& input);
Tensor sigmoid_forward(const Tensor& input);
Tensor sigmoid_backward(const Tensor& grad_output,const Tensor& output);

int main(){
    std::cout<<"=== Test 1: Tensor Device Management ==="<<std::endl;
    Tensor t_cpu({2,3},Device::CPU);

    for(int i = 0;i<t_cpu.numel();++i){
        t_cpu.data()[i] = static_cast<float>(i-2);
    }

    std::cout<<"Original CPU Data: ";
    for(int i = 0;i<t_cpu.numel();++i) std::cout<<t_cpu.data()[i] <<" ";
    std::cout<<std::endl;

    Tensor t_gpu = t_cpu.gpu();
    Tensor t_back = t_gpu.cpu();

    std::cout<<"Transferred Back to CPU :";
    for(int i = 0;i<t_back.numel();++i) std::cout<<t_back.data()[i]<< " ";
    std::cout<<"\n\n";

    std::cout<<"=== Test 2: ReLU CPU & GPU Execution ==="<<std::endl;

    Tensor relu_out_gpu = relu_forward(t_gpu);
    Tensor relu_out_cpu = relu_out_gpu.cpu();

    std::cout<<"ReLU Forward Output (GPU Result): ";
    for(int i = 0;i<relu_out_cpu.numel();++i) std::cout<<relu_out_cpu.data()[i]<<' ';
    std::cout<<std::endl;

    Tensor grad_out({2,3},Device::CPU);
    for(int i = 0;i<grad_out.numel();++i) grad_out.data()[i] = 1.0f;

    Tensor relu_grad_gpu = relu_backward(grad_out.gpu(),t_gpu);
    Tensor relu_grad_cpu = relu_grad_gpu.cpu();

    std::cout<<"ReLU Backward Grad (GPU Result): ";
    for(int i = 0;i<relu_grad_cpu.numel();++i) std::cout<< relu_grad_cpu.data()[i]<<" ";
    std::cout<<"\n\n";

    std::cout<<"=== Test 3: Sigmoid CPU & GPU Execution ==="<<std::endl;

    Tensor sig_out_gpu = sigmoid_forward(t_gpu);
    Tensor sig_out_cpu = sig_out_gpu.cpu();

    std::cout<<"Sigmoid Forward Output (GPU Result): ";
    for(int i = 0;i<sig_out_cpu.numel();++i) std::cout<<sig_out_cpu.data()[i]<<" ";
    std::cout<<std::endl;

    Tensor sig_grad_gpu = sigmoid_backward(grad_out.gpu(),sig_out_gpu);
    Tensor sig_grad_cpu = sig_grad_gpu.cpu();

    std::cout<<"Sigmoid Backward Grad (GPU Result): ";
    for(int i = 0;i<sig_grad_cpu.numel();++i) std::cout<<sig_grad_cpu.data()[i]<<" ";
    std::cout<<std::endl;

    return 0;
}