#pragma once
#include "common.cuh"

// Input NCHW with even H,W; output is [N,C,H/2,W/2]. d_mask stores absolute input offsets.
void max_pool_forward(const float* d_x, int N, int C, int H, int W, float* d_y, float* d_mask);
void max_pool_backward(const float* d_dy, const float* d_mask, int N, int C, int H, int W, float* d_dx);
