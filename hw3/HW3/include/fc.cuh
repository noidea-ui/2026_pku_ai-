#pragma once
#include "common.cuh"

// X: [N, Cin], W: [Cout, Cin], b: [Cout], Y: [N, Cout]. All tensors are row-major.
void fc_forward(cublasHandle_t handle, const float* d_x, const float* d_w, const float* d_b,
                float* d_y, int N, int Cin, int Cout);

// d_dy: [N, Cout]; outputs d_dx: [N, Cin], d_dw: [Cout, Cin], d_db: [Cout].
void fc_backward(cublasHandle_t handle, const float* d_x, const float* d_w, const float* d_dy,
                 float* d_dx, float* d_dw, float* d_db, int N, int Cin, int Cout);
