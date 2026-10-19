// float16 storage tests. Run via `make test`.
#include "../src/half.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <random>

using namespace inferno;

int main() {
    // Exactly representable values survive the round trip unchanged.
    Tensor exact({1, 8});
    const float vals[8] = {0.0f, 1.0f, -2.0f, 0.5f, 0.25f, 1024.0f, -0.125f, 65504.0f};
    for (int i = 0; i < 8; ++i) exact.data()[i] = vals[i];
    Tensor back = to_float(to_half(exact));
    for (int i = 0; i < 8; ++i) assert(back.data()[i] == vals[i]);

    // Typical GPT-2 weights (normal, std ~0.02 to 0.2): relative error of
    // one rounding is at most 2^-11 ~ 4.9e-4 for normal halves. Check it
    // over 100k values, including the scalar tail path (odd length).
    std::mt19937 g(3);
    std::normal_distribution<float> d(0.0f, 0.1f);
    Tensor w({317, 333});
    for (size_t i = 0; i < w.size(); ++i) w.data()[i] = d(g);
    HalfMatrix h = to_half(w);
    assert(h.rows == 317 && h.cols == 333 && h.bytes() == w.size() * 2);
    Tensor r = to_float(h);
    double worst = 0.0;
    for (size_t i = 0; i < w.size(); ++i) {
        const float x = w.data()[i];
        if (std::fabs(x) < 6.1e-5f) continue;  // subnormal halves have less precision
        worst = std::max(worst, static_cast<double>(std::fabs((r.data()[i] - x) / x)));
    }
    assert(worst <= 4.9e-4);

    // Kernels: matmul_h(A, half(B)) must equal the fp32 matmul of A with
    // the SAME rounded weights, to(float(half(B))). That isolates kernel
    // correctness from rounding: any difference beyond summation-order
    // noise is a kernel bug. Shapes cover the row split, the column
    // split (decode, M=1), the small serial path, and ragged tails.
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    const size_t shapes[][3] = {{1, 768, 2304}, {1, 3072, 768}, {64, 768, 3072},
                                {3, 5, 7}, {1, 17, 33}, {40, 31, 50}};
    for (auto& s : shapes) {
        Tensor a({s[0], s[1]}), b({s[1], s[2]}), bt({s[2], s[1]});
        for (size_t i = 0; i < a.size(); ++i) a.data()[i] = u(g);
        for (size_t i = 0; i < b.size(); ++i) b.data()[i] = u(g) * 0.05f;
        for (size_t i = 0; i < bt.size(); ++i) bt.data()[i] = u(g) * 0.05f;
        HalfMatrix bh = to_half(b), bth = to_half(bt);
        Tensor ref = matmul_reordered(a, to_float(bh));
        Tensor got = matmul_h(a, bh);
        Tensor ref_bt = matmul_bt(a, to_float(bth));
        Tensor got_bt = matmul_bt_h(a, bth);
        for (size_t i = 0; i < ref.size(); ++i) assert(std::fabs(ref.data()[i] - got.data()[i]) < 1e-4f);
        for (size_t i = 0; i < ref_bt.size(); ++i) assert(std::fabs(ref_bt.data()[i] - got_bt.data()[i]) < 1e-4f);
    }
    // And the LM-head shape the model will actually use.
    {
        Tensor x({1, 768}), wte({50257, 768});
        for (size_t i = 0; i < x.size(); ++i) x.data()[i] = u(g);
        for (size_t i = 0; i < wte.size(); ++i) wte.data()[i] = u(g) * 0.05f;
        HalfMatrix wh = to_half(wte);
        Tensor ref = matmul_bt(x, to_float(wh)), got = matmul_bt_h(x, wh);
        for (size_t i = 0; i < ref.size(); ++i) assert(std::fabs(ref.data()[i] - got.data()[i]) < 1e-4f);
    }

    std::printf("half tests passed (exact round trip; worst relative error %.2e over %zu weights; "
                "matmul_h and matmul_bt_h agree with fp32 on the same rounded weights)\n",
                worst, w.size());
    return 0;
}
