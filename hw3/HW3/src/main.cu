#include "fc.cuh"
#include "conv2d.cuh"
#include "max_pool.cuh"
#include "softmax.cuh"
#include "cross_entropy.cuh"
#include "test_utils.h"

#include <functional>
#include <iostream>
#include <string>

namespace {

void run_test(const std::string& name, const std::function<void()>& fn, int& passed, int& failed) {
    std::cout << "\n========== " << name << " ==========" << std::endl;
    try {
        fn();
        ++passed;
        std::cout << "[PASS] " << name << std::endl;
    } catch (const std::exception& e) {
        ++failed;
        std::cerr << "[FAIL] " << name << ": " << e.what() << std::endl;
    }
}

std::vector<float> make_sequence(std::size_t count, float scale, float offset = 0.0f) {
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i) {
        const int centered = static_cast<int>(i % 17) - 8;
        values[i] = offset + scale * static_cast<float>(centered);
    }
    return values;
}

void test_fc(cublasHandle_t handle) {
    const int N = 3, Cin = 4, Cout = 2;
    std::vector<float> x = make_sequence(N * Cin, 0.07f, 0.02f);
    std::vector<float> dy = make_sequence(N * Cout, 0.11f, -0.03f);
    std::vector<float> w(Cout * Cin), b(Cout);

    DeviceBuffer<float> dx_dev(x.size()), w_dev(w.size()), b_dev(b.size()), y_dev(N * Cout), dy_dev(dy.size());
    DeviceBuffer<float> ddx_dev(x.size()), ddw_dev(w.size()), ddb_dev(b.size());
    copy_to_device(dx_dev.get(), x);
    copy_to_device(dy_dev.get(), dy);
    curand_initialize(w_dev.get(), w.size(), 0.2f, 20261010ULL);
    curand_initialize(b_dev.get(), b.size(), 0.1f, 20261011ULL);
    copy_to_host(w, w_dev.get());
    copy_to_host(b, b_dev.get());

    fc_forward(handle, dx_dev.get(), w_dev.get(), b_dev.get(), y_dev.get(), N, Cin, Cout);
    std::vector<float> y(N * Cout);
    copy_to_host(y, y_dev.get());
    const auto y_ref = reference::fc_forward(x, w, b, N, Cin, Cout);
    assert_close("FC forward vs CPU", y, y_ref, 2.0e-5f, 2.0e-5f);

    fc_backward(handle, dx_dev.get(), w_dev.get(), dy_dev.get(), ddx_dev.get(), ddw_dev.get(), ddb_dev.get(), N, Cin, Cout);
    std::vector<float> dx(x.size()), dw(w.size()), db(b.size());
    copy_to_host(dx, ddx_dev.get());
    copy_to_host(dw, ddw_dev.get());
    copy_to_host(db, ddb_dev.get());
    std::vector<float> dx_ref, dw_ref, db_ref;
    reference::fc_backward(x, w, dy, N, Cin, Cout, dx_ref, dw_ref, db_ref);
    assert_close("FC dX vs CPU", dx, dx_ref, 2.0e-5f, 2.0e-5f);
    assert_close("FC dW vs CPU", dw, dw_ref, 2.0e-5f, 2.0e-5f);
    assert_close("FC db vs CPU", db, db_ref, 2.0e-5f, 2.0e-5f);

    auto loss = [&]() { return reference::dot(reference::fc_forward(x, w, b, N, Cin, Cout), dy); };
    const float num_dx = finite_difference(x, 2, loss);
    const float num_dw = finite_difference(w, 3, loss);
    const float num_db = finite_difference(b, 1, loss);
    assert_scalar_close("FC finite-difference dX[2]", dx[2], num_dx, 2.0e-4f, 2.0e-3f);
    assert_scalar_close("FC finite-difference dW[3]", dw[3], num_dw, 2.0e-4f, 2.0e-3f);
    assert_scalar_close("FC finite-difference db[1]", db[1], num_db, 2.0e-4f, 2.0e-3f);
}

void test_conv2d(cublasHandle_t handle) {
    const int N = 2, Cin = 2, Cout = 2, H = 4, W = 5;
    const int P = H * W;
    const int K = Cin * 9;
    std::vector<float> x = make_sequence(N * Cin * P, 0.025f, 0.01f);
    std::vector<float> dy = make_sequence(N * Cout * P, 0.035f, -0.02f);
    std::vector<float> w(Cout * K), b(Cout);

    DeviceBuffer<float> x_dev(x.size()), w_dev(w.size()), b_dev(b.size()), y_dev(N * Cout * P), dy_dev(dy.size());
    DeviceBuffer<float> dx_dev(x.size()), dw_dev(w.size()), db_dev(b.size());
    copy_to_device(x_dev.get(), x);
    copy_to_device(dy_dev.get(), dy);
    curand_initialize(w_dev.get(), w.size(), 0.08f, 20261012ULL);
    curand_initialize(b_dev.get(), b.size(), 0.05f, 20261013ULL);
    copy_to_host(w, w_dev.get());
    copy_to_host(b, b_dev.get());

    // Verify im2col's column-major [K,P] layout independently.
    DeviceBuffer<float> col_dev(static_cast<std::size_t>(K) * P);
    im2col_3x3(x_dev.get(), col_dev.get(), Cin, H, W);
    std::vector<float> col(static_cast<std::size_t>(K) * P), col_ref(col.size(), 0.0f);
    copy_to_host(col, col_dev.get());
    const std::vector<float> x_first(x.begin(), x.begin() + Cin * P);
    for (int p = 0; p < P; ++p) {
        const int oh = p / W, ow = p % W;
        for (int c = 0; c < Cin; ++c) for (int kh = 0; kh < 3; ++kh) for (int kw = 0; kw < 3; ++kw) {
            const int ih = oh + kh - 1, iw = ow + kw - 1;
            const int k = c * 9 + kh * 3 + kw;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) col_ref[k + p * K] = x_first[(c * H + ih) * W + iw];
        }
    }
    assert_close("im2col layout/zero-padding", col, col_ref, 0.0f, 0.0f);

    // col2im must add every overlapping patch contribution, not overwrite it.
    std::vector<float> ones_col(static_cast<std::size_t>(K) * P, 1.0f), reconstructed(Cin * P);
    DeviceBuffer<float> ones_col_dev(ones_col.size()), reconstructed_dev(reconstructed.size());
    copy_to_device(ones_col_dev.get(), ones_col);
    col2im_3x3(ones_col_dev.get(), reconstructed_dev.get(), Cin, H, W);
    copy_to_host(reconstructed, reconstructed_dev.get());
    std::vector<float> reconstructed_ref(Cin * P, 0.0f);
    for (int c = 0; c < Cin; ++c) for (int h = 0; h < H; ++h) for (int wpos = 0; wpos < W; ++wpos) {
        int count = 0;
        for (int kh = 0; kh < 3; ++kh) for (int kw = 0; kw < 3; ++kw) {
            const int oh = h - kh + 1, ow = wpos - kw + 1;
            if (oh >= 0 && oh < H && ow >= 0 && ow < W) ++count;
        }
        reconstructed_ref[(c * H + h) * W + wpos] = static_cast<float>(count);
    }
    assert_close("col2im overlap accumulation", reconstructed, reconstructed_ref, 0.0f, 0.0f);

    conv2d_forward(handle, x_dev.get(), w_dev.get(), b_dev.get(), y_dev.get(), N, Cin, Cout, H, W);
    std::vector<float> y(y_dev.size());
    copy_to_host(y, y_dev.get());
    const auto y_ref = reference::conv2d_forward(x, w, b, N, Cin, Cout, H, W);
    assert_close("Conv2D forward vs CPU", y, y_ref, 1.0e-4f, 1.0e-4f);

    conv2d_backward(handle, x_dev.get(), w_dev.get(), dy_dev.get(), dx_dev.get(), dw_dev.get(), db_dev.get(), N, Cin, Cout, H, W);
    std::vector<float> dx(x.size()), dw(w.size()), db(b.size());
    copy_to_host(dx, dx_dev.get()); copy_to_host(dw, dw_dev.get()); copy_to_host(db, db_dev.get());
    std::vector<float> dx_ref, dw_ref, db_ref;
    reference::conv2d_backward(x, w, dy, N, Cin, Cout, H, W, dx_ref, dw_ref, db_ref);
    assert_close("Conv2D dX vs CPU", dx, dx_ref, 1.5e-4f, 1.5e-4f);
    assert_close("Conv2D dW vs CPU", dw, dw_ref, 1.5e-4f, 1.5e-4f);
    assert_close("Conv2D db vs CPU", db, db_ref, 1.5e-4f, 1.5e-4f);

    auto loss = [&]() { return reference::dot(reference::conv2d_forward(x, w, b, N, Cin, Cout, H, W), dy); };
    const std::size_t x_idx = 7, w_idx = 20, b_idx = 1;
    const float num_dx = finite_difference(x, x_idx, loss);
    const float num_dw = finite_difference(w, w_idx, loss);
    const float num_db = finite_difference(b, b_idx, loss);
    assert_scalar_close("Conv2D finite-difference dX", dx[x_idx], num_dx, 1.0e-3f, 4.0e-3f);
    assert_scalar_close("Conv2D finite-difference dW", dw[w_idx], num_dw, 1.0e-3f, 4.0e-3f);
    assert_scalar_close("Conv2D finite-difference db", db[b_idx], num_db, 1.0e-3f, 4.0e-3f);
}

void test_max_pool() {
    const int N = 2, C = 2, H = 4, W = 6;
    std::vector<float> x(static_cast<std::size_t>(N) * C * H * W);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>((i * 13) % 37) - 15.0f + 0.01f * static_cast<float>(i);
    std::vector<float> y_ref;
    std::vector<float> mask_ref;
    reference::max_pool_forward(x, N, C, H, W, y_ref, mask_ref);
    std::vector<float> dy(y_ref.size());
    for (std::size_t i = 0; i < dy.size(); ++i) dy[i] = 0.1f * static_cast<float>(static_cast<int>(i) - 4);
    const auto dx_ref = reference::max_pool_backward(dy, mask_ref, N, C, H, W);

    DeviceBuffer<float> x_dev(x.size()), y_dev(y_ref.size()), dy_dev(dy.size()), dx_dev(x.size());
    DeviceBuffer<float> mask_dev(mask_ref.size());
    copy_to_device(x_dev.get(), x); copy_to_device(dy_dev.get(), dy);
    max_pool_forward(x_dev.get(), N, C, H, W, y_dev.get(), mask_dev.get());
    std::vector<float> y(y_ref.size()), mask(mask_ref.size());
    copy_to_host(y, y_dev.get()); copy_to_host(mask, mask_dev.get());
    assert_close("MaxPool forward values", y, y_ref, 0.0f, 0.0f);
    assert_close("MaxPool saved absolute indices (float mask per assignment interface)", mask, mask_ref, 0.0f, 0.0f);
    max_pool_backward(dy_dev.get(), mask_dev.get(), N, C, H, W, dx_dev.get());
    std::vector<float> dx(x.size()); copy_to_host(dx, dx_dev.get());
    assert_close("MaxPool backward", dx, dx_ref, 0.0f, 0.0f);
}

void test_softmax() {
    const int N = 3, C = 5;
    const std::vector<float> logits = {
        0.2f, -1.0f, 3.0f, 1.5f, 0.1f,
        1000.0f, 1001.0f, 999.0f, 998.0f, 997.0f,
        -1000.0f, -1001.0f, -999.0f, -998.0f, -997.0f
    };
    const auto probs_ref = reference::softmax(logits, N, C);
    DeviceBuffer<float> logits_dev(logits.size()), probs_dev(logits.size());
    copy_to_device(logits_dev.get(), logits);
    softmax_forward(logits_dev.get(), probs_dev.get(), N, C);
    std::vector<float> probs(logits.size()); copy_to_host(probs, probs_dev.get());
    assert_close("SoftMax vs CPU including extreme logits", probs, probs_ref, 2.0e-6f, 2.0e-5f);
    for (int n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) sum += probs[n * C + c];
        assert_scalar_close("SoftMax row sum " + std::to_string(n), sum, 1.0f, 2.0e-6f, 2.0e-6f);
    }
}

void test_cross_entropy() {
    const int N = 3, C = 5;
    std::vector<float> logits = {
        0.2f, -1.0f, 3.0f, 1.5f, 0.1f,
        1000.0f, 1001.0f, 999.0f, 998.0f, 997.0f,
        -1000.0f, -1001.0f, -999.0f, -998.0f, -997.0f
    };
    const std::vector<int> labels = {4, 1, 0};
    const auto probs_ref = reference::softmax(logits, N, C);
    const float loss_ref = reference::cross_entropy(probs_ref, labels, N, C);
    const auto dz_ref = reference::cross_entropy_backward(probs_ref, labels, N, C);

    DeviceBuffer<float> logits_dev(logits.size()), probs_dev(logits.size()), loss_dev(1), dz_dev(logits.size());
    DeviceBuffer<int> labels_dev(labels.size());
    copy_to_device(logits_dev.get(), logits); copy_to_device(labels_dev.get(), labels);
    softmax_forward(logits_dev.get(), probs_dev.get(), N, C);
    cross_entropy_forward(probs_dev.get(), labels_dev.get(), N, C, loss_dev.get());
    float loss = 0.0f; CUDA_CHECK(cudaMemcpy(&loss, loss_dev.get(), sizeof(float), cudaMemcpyDeviceToHost));
    assert_scalar_close("CrossEntropy mean loss", loss, loss_ref, 2.0e-5f, 2.0e-5f);

    cross_entropy_backward(probs_dev.get(), labels_dev.get(), dz_dev.get(), N, C);
    std::vector<float> dz(logits.size()); copy_to_host(dz, dz_dev.get());
    assert_close("CrossEntropy combined logits gradient", dz, dz_ref, 2.0e-6f, 2.0e-5f);

    auto loss_from_logits = [&]() { return reference::cross_entropy(reference::softmax(logits, N, C), labels, N, C); };
    const std::size_t idx = 6;
    const float num_dz = finite_difference(logits, idx, loss_from_logits, 1.0e-2f);
    assert_scalar_close("CrossEntropy finite-difference logits gradient", dz[idx], num_dz, 2.0e-4f, 2.0e-3f);
}

}  // namespace

int main() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA-capable GPU was found.\n";
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "HW3 CUDA tests | device: " << prop.name << " | compute capability "
              << prop.major << "." << prop.minor << "\n";

    cublasHandle_t handle{};
    CUBLAS_CHECK(cublasCreate(&handle));
    int passed = 0, failed = 0;
    run_test("Fully Connected", [&]() { test_fc(handle); }, passed, failed);
    run_test("Max Pooling", [&]() { test_max_pool(); }, passed, failed);
    run_test("SoftMax", [&]() { test_softmax(); }, passed, failed);
    run_test("Cross Entropy", [&]() { test_cross_entropy(); }, passed, failed);
    run_test("Conv2D + im2col/col2im", [&]() { test_conv2d(handle); }, passed, failed);
    CUBLAS_CHECK(cublasDestroy(handle));

    std::cout << "\n========================================\n"
              << "Test summary: " << passed << " passed, " << failed << " failed.\n";
    return failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
