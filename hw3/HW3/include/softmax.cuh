#pragma once
#include "common.cuh"

// Stable row-wise SoftMax. Input/output shape: [N,C].
void softmax_forward(const float* d_logits, float* d_probs, int N, int C);
