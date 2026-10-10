#pragma once
#include "common.cuh"

// Computes mean(-log(prob[n,label[n]])); labels are integer values in [0,C-1].
void cross_entropy_forward(const float* d_probs, const int* d_labels, int N, int C, float* d_loss);
// Combined SoftMax + mean Cross Entropy gradient w.r.t. logits: (p - one_hot(label))/N.
void cross_entropy_backward(const float* d_probs, const int* d_labels, float* d_dlogits, int N, int C);
