#pragma once
#include "common.cuh"

// Single-sample helpers. Input is [Cin,H,W]. d_col is logically [Cin*3*3,H*W]
// in column-major layout: d_col[k + p*(Cin*9)]. Padding=1, stride=1.
void im2col_3x3(const float* d_x, float* d_col, int Cin, int H, int W);
// Each input element sums all col entries that map back to it; no prior memset is needed.
void col2im_3x3(const float* d_col, float* d_x, int Cin, int H, int W);

// X/Y are NCHW; W is [Cout,Cin,3,3]; b is [Cout]. Output spatial size is H x W.
void conv2d_forward(cublasHandle_t handle, const float* d_x, const float* d_w, const float* d_b,
                    float* d_y, int N, int Cin, int Cout, int H, int W);
void conv2d_backward(cublasHandle_t handle, const float* d_x, const float* d_w, const float* d_dy,
                     float* d_dx, float* d_dw, float* d_db,
                     int N, int Cin, int Cout, int H, int W);
