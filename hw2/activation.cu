#include"tensor.h"

__global__ void relu_forward_kernel(const float* x,float* y,int n) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<n){
        y[idx] = x[idx]>0.0f?x[idx]:0.0f;
    }
}

__global__ void relu_backward_kernel(const float* grad_output,const float* x,float* grad_input,int n){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<n){
        grad_input[idx] = (x[idx] > 0.0f)?grad_output[idx] :0.0f;
    }
}

__global__ void sigmoid_forward_kernel(const float* x,float* y,int n){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<n){
        y[idx] = 1.0f / (1.0f + expf(-x[idx]));
    }
}

__global__ void sigmoid_backward_kernel(const float* grad_output,const float* y, float* grad_input,int n){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<n){
        float y_val = y[idx];
        grad_input[idx] = grad_output[idx]*y_val*(1.0f-y_val);
    }
}


//=======================激活函数API===========================

Tensor relu_forward(const Tensor& input){
    Tensor output(input.shape(),input.device());
    int n = input.numel();

    if(input.device() == Device::CPU){
        const float* x = input.data();
        float* y = output.data();
        for(int i = 0;i<n;++i){
            y[i] = std::max(0.0f,x[i]);
        }
    }
    else{
        int block_size = 256;
        int grid_size = (n+block_size-1)/block_size;
        relu_forward_kernel<<<grid_size,block_size>>> (input.data(),output.data(),n);
        //这个东西可能会报错但是不用管，在nvcc编译之后就可以解决了
        cudaDeviceSynchronize();
    }
    return output;
}

Tensor relu_backward(const Tensor& grad_output,const Tensor& input){
    Tensor grad_input(input.shape(),input.device());
    int n = input.numel();

    if(input.device() == Device::CPU){
        const float* go = grad_output.data();
        const float* x = input.data();
        float* gi = grad_input.data();
        for(int i = 0;i<n;i++){
            gi[i] = (x[i]>0.0f) ? go[i] : 0.0f;
        }
    }
    else{
        int block_size = 256;
        int grid_size = (n+block_size-1)/block_size;
        relu_backward_kernel<<<grid_size,block_size>>>(grad_output.data(),input.data(),grad_input.data(),n);
        cudaDeviceSynchronize();
    }
    return grad_input;
}

Tensor sigmoid_forward(const Tensor& input){
    Tensor output(input.shape(),input.device());
    int n = input.numel();

    if(input.device() == Device::CPU){
        const float* x = input.data();
        float* y = output.data();
        for(int i = 0;i<n;++i){
            y[i] = 1.0f/(1.0f+std::exp(-x[i]));
        }
    }
    else{
        int block_size = 256;
        int grid_size = (n+block_size -1) / block_size;
        sigmoid_forward_kernel<<<grid_size,block_size>>>(input.data(),output.data(),n);
        cudaDeviceSynchronize();
    }
    return output;
}

Tensor sigmoid_backward(const Tensor& grad_output,const Tensor& output){
    Tensor grad_input(output.shape(),output.device());
    int n = output.numel();

    if(output.device() == Device::CPU){
        const float* go = grad_output.data();
        const float* y = output.data();
        float* gi = grad_input.data();
        
        for(int i = 0;i<n;++i){
            gi[i] = go[i]*y[i]*(1.0f-y[i]);
        }

    }
    else{
        int block_size = 256;
        int grid_size = (n+block_size-1)/block_size;
        sigmoid_backward_kernel<<<grid_size,block_size>>>(grad_output.data(),output.data(),grad_input.data(),n);
        cudaDeviceSynchronize();

    }
    return grad_input;
}