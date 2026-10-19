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

    std::printf("half tests passed (exact round trip; worst relative error %.2e over %zu weights)\n",
                worst, w.size());
    return 0;
}
