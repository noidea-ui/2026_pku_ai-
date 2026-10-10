#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace reference {

inline std::vector<float> fc_forward(const std::vector<float>& x, const std::vector<float>& w,
                                     const std::vector<float>& b, int N, int Cin, int Cout) {
    std::vector<float> y(static_cast<std::size_t>(N) * Cout, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int o = 0; o < Cout; ++o) {
            float sum = b[o];
            for (int i = 0; i < Cin; ++i) sum += x[n * Cin + i] * w[o * Cin + i];
            y[n * Cout + o] = sum;
        }
    }
    return y;
}

inline void fc_backward(const std::vector<float>& x, const std::vector<float>& w,
                        const std::vector<float>& dy, int N, int Cin, int Cout,
                        std::vector<float>& dx, std::vector<float>& dw, std::vector<float>& db) {
    dx.assign(static_cast<std::size_t>(N) * Cin, 0.0f);
    dw.assign(static_cast<std::size_t>(Cout) * Cin, 0.0f);
    db.assign(Cout, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int o = 0; o < Cout; ++o) {
            const float g = dy[n * Cout + o];
            db[o] += g;
            for (int i = 0; i < Cin; ++i) {
                dw[o * Cin + i] += g * x[n * Cin + i];
                dx[n * Cin + i] += g * w[o * Cin + i];
            }
        }
    }
}

inline std::vector<float> conv2d_forward(const std::vector<float>& x, const std::vector<float>& w,
                                         const std::vector<float>& b, int N, int Cin, int Cout, int H, int W) {
    const int P = H * W;
    std::vector<float> y(static_cast<std::size_t>(N) * Cout * P, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < Cout; ++oc) {
            for (int oh = 0; oh < H; ++oh) {
                for (int ow = 0; ow < W; ++ow) {
                    float sum = b[oc];
                    for (int ic = 0; ic < Cin; ++ic) {
                        for (int kh = 0; kh < 3; ++kh) {
                            for (int kw = 0; kw < 3; ++kw) {
                                const int ih = oh + kh - 1;
                                const int iw = ow + kw - 1;
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                const int xi = ((n * Cin + ic) * H + ih) * W + iw;
                                const int wi = ((oc * Cin + ic) * 3 + kh) * 3 + kw;
                                sum += x[xi] * w[wi];
                            }
                        }
                    }
                    y[((n * Cout + oc) * H + oh) * W + ow] = sum;
                }
            }
        }
    }
    return y;
}

inline void conv2d_backward(const std::vector<float>& x, const std::vector<float>& w,
                            const std::vector<float>& dy, int N, int Cin, int Cout, int H, int W,
                            std::vector<float>& dx, std::vector<float>& dw, std::vector<float>& db) {
    const int P = H * W;
    dx.assign(static_cast<std::size_t>(N) * Cin * P, 0.0f);
    dw.assign(static_cast<std::size_t>(Cout) * Cin * 9, 0.0f);
    db.assign(Cout, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < Cout; ++oc) {
            for (int oh = 0; oh < H; ++oh) {
                for (int ow = 0; ow < W; ++ow) {
                    const int yi = ((n * Cout + oc) * H + oh) * W + ow;
                    const float g = dy[yi];
                    db[oc] += g;
                    for (int ic = 0; ic < Cin; ++ic) {
                        for (int kh = 0; kh < 3; ++kh) {
                            for (int kw = 0; kw < 3; ++kw) {
                                const int ih = oh + kh - 1;
                                const int iw = ow + kw - 1;
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                const int xi = ((n * Cin + ic) * H + ih) * W + iw;
                                const int wi = ((oc * Cin + ic) * 3 + kh) * 3 + kw;
                                dw[wi] += g * x[xi];
                                dx[xi] += g * w[wi];
                            }
                        }
                    }
                }
            }
        }
    }
}

inline void max_pool_forward(const std::vector<float>& x, int N, int C, int H, int W,
                             std::vector<float>& y, std::vector<float>& mask) {
    const int OH = H / 2;
    const int OW = W / 2;
    y.assign(static_cast<std::size_t>(N) * C * OH * OW, 0.0f);
    mask.assign(y.size(), 0);
    for (int n = 0; n < N; ++n) for (int c = 0; c < C; ++c)
    for (int oh = 0; oh < OH; ++oh) for (int ow = 0; ow < OW; ++ow) {
        const int base = ((n * C + c) * H + 2 * oh) * W + 2 * ow;
        int best_idx = base;
        float best = x[base];
        const int cand[3] = {base + 1, base + W, base + W + 1};
        for (int j = 0; j < 3; ++j) if (x[cand[j]] > best) { best = x[cand[j]]; best_idx = cand[j]; }
        const int oi = ((n * C + c) * OH + oh) * OW + ow;
        y[oi] = best;
        mask[oi] = static_cast<float>(best_idx);
    }
}

inline std::vector<float> max_pool_backward(const std::vector<float>& dy, const std::vector<float>& mask,
                                            int N, int C, int H, int W) {
    std::vector<float> dx(static_cast<std::size_t>(N) * C * H * W, 0.0f);
    for (std::size_t i = 0; i < dy.size(); ++i) dx[static_cast<int>(mask[i])] = dy[i];
    return dx;
}

inline std::vector<float> softmax(const std::vector<float>& logits, int N, int C) {
    std::vector<float> probs(logits.size(), 0.0f);
    for (int n = 0; n < N; ++n) {
        const float* row = logits.data() + static_cast<std::size_t>(n) * C;
        float m = row[0];
        for (int c = 1; c < C; ++c) m = std::max(m, row[c]);
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            probs[n * C + c] = std::exp(row[c] - m);
            sum += probs[n * C + c];
        }
        for (int c = 0; c < C; ++c) probs[n * C + c] /= sum;
    }
    return probs;
}

inline float cross_entropy(const std::vector<float>& probs, const std::vector<int>& labels, int N, int C) {
    float loss = 0.0f;
    for (int n = 0; n < N; ++n) loss -= std::log(std::max(probs[n * C + labels[n]], 1.0e-12f));
    return loss / static_cast<float>(N);
}

inline std::vector<float> cross_entropy_backward(const std::vector<float>& probs,
                                                 const std::vector<int>& labels, int N, int C) {
    std::vector<float> dz(probs.size(), 0.0f);
    for (int n = 0; n < N; ++n) for (int c = 0; c < C; ++c)
        dz[n * C + c] = (probs[n * C + c] - (c == labels[n] ? 1.0f : 0.0f)) / static_cast<float>(N);
    return dz;
}

inline float dot(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) throw std::runtime_error("dot: vector sizes differ");
    float sum = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
    return sum;
}

}  // namespace reference

inline void assert_close(const std::string& name, const std::vector<float>& actual,
                         const std::vector<float>& expected, float atol = 1.0e-4f, float rtol = 1.0e-4f) {
    if (actual.size() != expected.size()) {
        std::ostringstream oss;
        oss << name << ": size mismatch actual=" << actual.size() << " expected=" << expected.size();
        throw std::runtime_error(oss.str());
    }
    float max_abs = 0.0f;
    std::size_t max_idx = 0;
    bool ok = true;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float err = std::fabs(actual[i] - expected[i]);
        if (err > max_abs) { max_abs = err; max_idx = i; }
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) || err > atol + rtol * std::fabs(expected[i])) ok = false;
    }
    if (!ok) {
        std::ostringstream oss;
        oss << std::setprecision(8) << name << " failed: max_abs=" << max_abs << " at index " << max_idx
            << ", actual=" << actual[max_idx] << ", expected=" << expected[max_idx]
            << ", atol=" << atol << ", rtol=" << rtol;
        throw std::runtime_error(oss.str());
    }
    std::cout << "  [PASS] " << name << " (max_abs=" << std::scientific << max_abs << std::defaultfloat << ")\n";
}

inline void assert_close(const std::string& name, const std::vector<int>& actual,
                         const std::vector<int>& expected) {
    if (actual != expected) {
        std::size_t idx = 0;
        while (idx < actual.size() && idx < expected.size() && actual[idx] == expected[idx]) ++idx;
        std::ostringstream oss;
        oss << name << " failed at index " << idx;
        if (idx < actual.size()) oss << ": actual=" << actual[idx];
        if (idx < expected.size()) oss << ", expected=" << expected[idx];
        throw std::runtime_error(oss.str());
    }
    std::cout << "  [PASS] " << name << "\n";
}

inline void assert_scalar_close(const std::string& name, float actual, float expected,
                                float atol = 1.0e-4f, float rtol = 1.0e-4f) {
    const float err = std::fabs(actual - expected);
    if (!std::isfinite(actual) || !std::isfinite(expected) || err > atol + rtol * std::fabs(expected)) {
        std::ostringstream oss;
        oss << std::setprecision(9) << name << " failed: actual=" << actual << " expected=" << expected
            << " abs_error=" << err << " atol=" << atol << " rtol=" << rtol;
        throw std::runtime_error(oss.str());
    }
    std::cout << "  [PASS] " << name << " (abs_error=" << std::scientific << err << std::defaultfloat << ")\n";
}

template <typename LossFunction>
inline float finite_difference(std::vector<float>& values, std::size_t index, LossFunction loss, float eps = 1.0e-3f) {
    const float old = values.at(index);
    values[index] = old + eps;
    const float plus = loss();
    values[index] = old - eps;
    const float minus = loss();
    values[index] = old;
    return (plus - minus) / (2.0f * eps);
}
